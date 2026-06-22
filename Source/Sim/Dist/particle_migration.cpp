// particle_migration.cpp

// external
#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// internal
#include "particle_migration.h"
#include "mpi_context.h"
#include "particle_migration_diagnostics.h"



using namespace Dist;

namespace
{
	constexpr int PARTICLE_MIGRATION_TAG_FLUID_PAYLOAD = 5102;
	constexpr uint64_t PARTICLE_MIGRATION_PACK_CHUNK_SIZE = 4096;
	constexpr int PARTICLE_MIGRATION_ROUTE_KEEP_LOCAL = 0;
	constexpr uint8_t PARTICLE_MIGRATION_LOCAL_STATUS_NONE = 0;
	constexpr uint8_t PARTICLE_MIGRATION_LOCAL_STATUS_KEPT = 1;
	constexpr uint8_t PARTICLE_MIGRATION_LOCAL_STATUS_OUT = 2;

	static_assert(
		sizeof(MigratingFluidParticlePacket) == sizeof(int) + 16 * sizeof(float),
		"MigratingFluidParticlePacket must remain a tightly packed MPI byte payload.");

	enum class ParticleMigrationPackFailure : int
	{
		None = 0,
		LocalIdxOutOfRange,
		MissingMetadata,
		InvalidParticleClass,
		MissingOwnerRank,
		OwnerRankOutOfWorld,
		OwnerRankWithoutMigrationBuffer
	};

	void StorePackFailure(
		std::atomic<int>& failureCode,
		std::atomic<int>& failureLocalIdx,
		std::atomic<int>& failureOwnerRank,
		const ParticleMigrationPackFailure code,
		const int localIdx,
		const int ownerRank = -1)
	{
		int expected = static_cast<int>(ParticleMigrationPackFailure::None);
		if (failureCode.compare_exchange_strong(
			expected, static_cast<int>(code), std::memory_order_relaxed))
		{
			failureLocalIdx.store(localIdx, std::memory_order_relaxed);
			failureOwnerRank.store(ownerRank, std::memory_order_relaxed);
		}
	}

	const char* PackFailureMessage(const ParticleMigrationPackFailure failure)
	{
		switch (failure)
		{
		case ParticleMigrationPackFailure::LocalIdxOutOfRange:
			return "owned fluid local index is outside the local fluid particle range.";
		case ParticleMigrationPackFailure::MissingMetadata:
			return "owned fluid local index has no particle metadata.";
		case ParticleMigrationPackFailure::InvalidParticleClass:
			return "owned fluid local index class is not Inner or Outer.";
		case ParticleMigrationPackFailure::MissingOwnerRank:
			return "owned fluid particle has no valid owner rank.";
		case ParticleMigrationPackFailure::OwnerRankOutOfWorld:
			return "owned fluid particle owner rank is outside the MPI world.";
		case ParticleMigrationPackFailure::OwnerRankWithoutMigrationBuffer:
			return "owned fluid particle owner rank does not map to a migration buffer.";
		case ParticleMigrationPackFailure::None:
		default:
			return "";
		}
	}

	template <typename T>
	int CheckedPacketCount(const std::vector<T>& packets)
	{
		if (packets.size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
		{
			throw std::runtime_error("particle migration packet count exceeds MPI int count range.");
		}

		return static_cast<int>(packets.size());
	}

	template <typename T>
	uint64_t CheckedByteSize(const std::vector<T>& packets)
	{
		const uint64_t packetCount = packets.size();
		const uint64_t maxCount =
			std::numeric_limits<uint64_t>::max() / sizeof(T);

		if (packetCount > maxCount)
		{
			throw std::runtime_error("particle migration payload byte size exceeds uint64_t range.");
		}

		return packetCount * sizeof(T);
	}

	template <typename T>
	const void* DataOrNull(const std::vector<T>& packets)
	{
		return packets.empty() ? nullptr : packets.data();
	}

	template <typename T>
	void* DataOrNull(std::vector<T>& packets)
	{
		return packets.empty() ? nullptr : packets.data();
	}

	glm::vec3 FluidPositionAtLocalIdx(
		const Sync::FluidParticles& fluidParticles,
		const int localIdx)
	{
		return
		{
			fluidParticles.xPos[localIdx],
			fluidParticles.yPos[localIdx],
			fluidParticles.zPos[localIdx]
		};
	}

	glm::vec3 FluidPositionFromPacket(const MigratingFluidParticlePacket& packet)
	{
		return
		{
			packet.xPos,
			packet.yPos,
			packet.zPos
		};
	}

	bool IsLocalIdxInRange(
		const int localIdx,
		const uint32_t particleCount)
	{
		return localIdx >= 0 &&
			static_cast<uint32_t>(localIdx) < particleCount;
	}

	bool IsGlobalIdxInRange(
		const int globalIdx,
		const uint32_t particleCount)
	{
		return globalIdx >= 0 &&
			static_cast<uint32_t>(globalIdx) < particleCount;
	}

	bool HasParticleMetadataAtLocalIdx(
		const DistParticleState& particleState,
		const int localIdx)
	{
		return localIdx >= 0 &&
			static_cast<uint64_t>(localIdx) < particleState.particleClasses.size() &&
			static_cast<uint64_t>(localIdx) < particleState.particleGlobalIdx_FromLocalIdx.size();
	}

	MigratingFluidParticlePacket PackFluidParticle(
		const Sync::FluidParticles& fluidParticles,
		const DistParticleState& particleState,
		const int localIdx)
	{
		MigratingFluidParticlePacket packet{};
		packet.globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];

		packet.xPos = fluidParticles.xPos[localIdx];
		packet.yPos = fluidParticles.yPos[localIdx];
		packet.zPos = fluidParticles.zPos[localIdx];

		packet.xVel = fluidParticles.xVel[localIdx];
		packet.yVel = fluidParticles.yVel[localIdx];
		packet.zVel = fluidParticles.zVel[localIdx];

		packet.xVelPred = fluidParticles.xVelPred[localIdx];
		packet.yVelPred = fluidParticles.yVelPred[localIdx];
		packet.zVelPred = fluidParticles.zVelPred[localIdx];

		packet.xAccPress = fluidParticles.xAccPress[localIdx];
		packet.yAccPress = fluidParticles.yAccPress[localIdx];
		packet.zAccPress = fluidParticles.zAccPress[localIdx];

		packet.sourceTerm = fluidParticles.sourceTerm[localIdx];
		packet.diagElem = fluidParticles.diagElem[localIdx];
		packet.pressure = fluidParticles.pressure[localIdx];
		packet.density = fluidParticles.density[localIdx];

		return packet;
	}

	void CopyFluidParticle_LocalToLocal(
		const Sync::FluidParticles& src,
		Sync::FluidParticles& dst,
		const int srcLocalIdx,
		const int dstLocalIdx)
	{
		dst.xPos[dstLocalIdx] = src.xPos[srcLocalIdx];
		dst.yPos[dstLocalIdx] = src.yPos[srcLocalIdx];
		dst.zPos[dstLocalIdx] = src.zPos[srcLocalIdx];

		dst.xVel[dstLocalIdx] = src.xVel[srcLocalIdx];
		dst.yVel[dstLocalIdx] = src.yVel[srcLocalIdx];
		dst.zVel[dstLocalIdx] = src.zVel[srcLocalIdx];

		dst.xVelPred[dstLocalIdx] = src.xVelPred[srcLocalIdx];
		dst.yVelPred[dstLocalIdx] = src.yVelPred[srcLocalIdx];
		dst.zVelPred[dstLocalIdx] = src.zVelPred[srcLocalIdx];

		dst.xAccPress[dstLocalIdx] = src.xAccPress[srcLocalIdx];
		dst.yAccPress[dstLocalIdx] = src.yAccPress[srcLocalIdx];
		dst.zAccPress[dstLocalIdx] = src.zAccPress[srcLocalIdx];

		dst.sourceTerm[dstLocalIdx] = src.sourceTerm[srcLocalIdx];
		dst.diagElem[dstLocalIdx] = src.diagElem[srcLocalIdx];
		dst.pressure[dstLocalIdx] = src.pressure[srcLocalIdx];
		dst.density[dstLocalIdx] = src.density[srcLocalIdx];
	}

	void CopyFluidPacketToLocal(
		const MigratingFluidParticlePacket& packet,
		Sync::FluidParticles& dst,
		const int localIdx)
	{
		dst.xPos[localIdx] = packet.xPos;
		dst.yPos[localIdx] = packet.yPos;
		dst.zPos[localIdx] = packet.zPos;

		dst.xVel[localIdx] = packet.xVel;
		dst.yVel[localIdx] = packet.yVel;
		dst.zVel[localIdx] = packet.zVel;

		dst.xVelPred[localIdx] = packet.xVelPred;
		dst.yVelPred[localIdx] = packet.yVelPred;
		dst.zVelPred[localIdx] = packet.zVelPred;

		dst.xAccPress[localIdx] = packet.xAccPress;
		dst.yAccPress[localIdx] = packet.yAccPress;
		dst.zAccPress[localIdx] = packet.zAccPress;

		dst.sourceTerm[localIdx] = packet.sourceTerm;
		dst.diagElem[localIdx] = packet.diagElem;
		dst.pressure[localIdx] = packet.pressure;
		dst.density[localIdx] = packet.density;
	}

	bool AddParticleMapEntry(
		DistParticleState& particleState,
		const int globalIdx,
		const int localIdx,
		const uint32_t globalParticleCount)
	{
		if (!IsGlobalIdxInRange(globalIdx, globalParticleCount)) { return false; }
		if (particleState.particleLocalIdx_FromGlobalIdx[globalIdx] != -1) { return false; }

		particleState.particleGlobalIdx_FromLocalIdx[localIdx] = globalIdx;
		particleState.particleLocalIdx_FromGlobalIdx[globalIdx] = localIdx;
		return true;
	}

	uint32_t PrepareMigrationLocalStatusStamp(
		ParticleMigrationWorkBuffers& workBuffers,
		const uint32_t localParticleCount)
	{
		workBuffers.migrationLocalStatusStamp_FromLocalIdx.resize(localParticleCount, 0);
		workBuffers.migrationLocalStatus_FromLocalIdx.resize(
			localParticleCount, PARTICLE_MIGRATION_LOCAL_STATUS_NONE);

		workBuffers.migrationLocalStatusStamp++;
		if (workBuffers.migrationLocalStatusStamp == 0)
		{
			std::fill(
				workBuffers.migrationLocalStatusStamp_FromLocalIdx.begin(),
				workBuffers.migrationLocalStatusStamp_FromLocalIdx.end(),
				0);
			workBuffers.migrationLocalStatusStamp = 1;
		}

		return workBuffers.migrationLocalStatusStamp;
	}

	uint8_t MigrationLocalStatusAt(
		const ParticleMigrationWorkBuffers& workBuffers,
		const uint32_t stamp,
		const uint32_t localIdx)
	{
		if (workBuffers.migrationLocalStatusStamp_FromLocalIdx[localIdx] != stamp)
		{
			return PARTICLE_MIGRATION_LOCAL_STATUS_NONE;
		}

		return workBuffers.migrationLocalStatus_FromLocalIdx[localIdx];
	}

	void SetMigrationLocalStatus(
		ParticleMigrationWorkBuffers& workBuffers,
		const uint32_t stamp,
		const uint32_t localIdx,
		const uint8_t status)
	{
		workBuffers.migrationLocalStatusStamp_FromLocalIdx[localIdx] = stamp;
		workBuffers.migrationLocalStatus_FromLocalIdx[localIdx] = status;
	}

	bool ClearLocalMapEntryIfStillMapped(
		DistParticleState& particleState,
		const int localIdx,
		const uint32_t globalParticleCount)
	{
		if (localIdx < 0 ||
			static_cast<uint64_t>(localIdx) >= particleState.particleGlobalIdx_FromLocalIdx.size())
		{
			return false;
		}

		const int globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
		if (!IsGlobalIdxInRange(globalIdx, globalParticleCount)) { return false; }
		if (static_cast<uint64_t>(globalIdx) >= particleState.particleLocalIdx_FromGlobalIdx.size())
		{
			return false;
		}

		if (particleState.particleLocalIdx_FromGlobalIdx[globalIdx] == localIdx)
		{
			particleState.particleLocalIdx_FromGlobalIdx[globalIdx] = -1;
		}

		return true;
	}

	void PrepareParticleMigrationRankBuffers(
		ParticleMigrationWorkBuffers& workBuffers,
		const int rank,
		const int rankCount)
	{
		std::vector<ParticleMigrationWorkBuffers::RankBuffers>& migrations =
			workBuffers.rankBuffers;
		migrations.resize(static_cast<uint64_t>(std::max(0, rankCount - 1)));

		workBuffers.rankBufferIdx_FromRank.assign(
			static_cast<uint64_t>(std::max(0, rankCount)), -1);

		uint64_t migrationIdx = 0;
		for (int dstRank = 0; dstRank < rankCount; dstRank++)
		{
			if (dstRank == rank) { continue; }

			ParticleMigrationWorkBuffers::RankBuffers& migration =
				migrations[migrationIdx];
			migration.rank = dstRank;
			migration.sendFluidCount = 0;
			migration.recvFluidCount = 0;
			migration.sendBuffer.Clear();

			workBuffers.rankBufferIdx_FromRank[static_cast<uint64_t>(dstRank)] =
				static_cast<int>(migrationIdx);
			migrationIdx++;
		}
	}

	// pack owned fluid particles using a parallel count -> prefix sum -> parallel fill pipeline
	bool PackMigratingFluidParticlesByOwnerRank(
		ParticleMigrationWorkBuffers& workBuffers,
		ParticleMigrationStats& stats,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const IDomainDecomp& domainDecomp,
		const int rank,
		std::string* failMsg)
	{
		if (failMsg) { failMsg->clear(); }

		const auto fail = [failMsg](std::string message) -> bool
		{
			if (failMsg) { *failMsg = std::move(message); }
			return false;
		};

		stats.sentFluidParticleCount = 0;

		const std::vector<int>& ownedLocalIdxs =
			fluidParticleState.particleLocalIdx_Owned;
		const uint64_t ownedParticleCount = ownedLocalIdxs.size();
		const uint64_t migrationCount = workBuffers.rankBuffers.size();
		const uint64_t routeCount = migrationCount + 1;
		const uint64_t chunkCount = ownedParticleCount > 0
			? (ownedParticleCount + PARTICLE_MIGRATION_PACK_CHUNK_SIZE - 1) /
				PARTICLE_MIGRATION_PACK_CHUNK_SIZE
			: 0;

		workBuffers.migrationRouteIdx_FromOwnedIdx.assign(
			ownedParticleCount, PARTICLE_MIGRATION_ROUTE_KEEP_LOCAL);
		workBuffers.migrationRouteCounts_ByChunkRoute.assign(
			chunkCount * routeCount, 0ull);
		workBuffers.migrationRouteWriteOffsets_ByChunkRoute.assign(
			chunkCount * routeCount, 0ull);
		workBuffers.migrationRouteTotals.assign(routeCount, 0ull);
		workBuffers.migrationOutCounts_ByChunk.assign(chunkCount, 0ull);
		workBuffers.migrationOutWriteOffsets_ByChunk.assign(chunkCount, 0ull);

		std::atomic<int> failureCode{
			static_cast<int>(ParticleMigrationPackFailure::None)
		};
		std::atomic<int> failureLocalIdx{ -1 };
		std::atomic<int> failureOwnerRank{ -1 };

		// classify owned fluid particles as kept local or migrated to their new owner rank
		tbb::parallel_for(
			tbb::blocked_range<uint64_t>(0, chunkCount),
			[&](const tbb::blocked_range<uint64_t>& range)
		{
			for (uint64_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint64_t beginOwnedIdx =
					chunkIdx * PARTICLE_MIGRATION_PACK_CHUNK_SIZE;
				const uint64_t endOwnedIdx = std::min(
					beginOwnedIdx + PARTICLE_MIGRATION_PACK_CHUNK_SIZE,
					ownedParticleCount);
				const uint64_t chunkRouteBase = chunkIdx * routeCount;

				for (uint64_t ownedIdx = beginOwnedIdx; ownedIdx != endOwnedIdx; ownedIdx++)
				{
					if (failureCode.load(std::memory_order_relaxed) !=
						static_cast<int>(ParticleMigrationPackFailure::None))
					{
						continue;
					}

					const int localIdx = ownedLocalIdxs[ownedIdx];

					// invariant checks for correct metadata
					if (!IsLocalIdxInRange(localIdx, localState.fluidParticles.N_))
					{
						StorePackFailure(
							failureCode, failureLocalIdx, failureOwnerRank,
							ParticleMigrationPackFailure::LocalIdxOutOfRange,
							localIdx);
						continue;
					}
					if (!HasParticleMetadataAtLocalIdx(fluidParticleState, localIdx))
					{
						StorePackFailure(
							failureCode, failureLocalIdx, failureOwnerRank,
							ParticleMigrationPackFailure::MissingMetadata,
							localIdx);
						continue;
					}

					const DistParticleClass particleClass =
						fluidParticleState.particleClasses[localIdx];
					if (particleClass != DistParticleClass::Inner &&
						particleClass != DistParticleClass::Outer)
					{
						StorePackFailure(
							failureCode, failureLocalIdx, failureOwnerRank,
							ParticleMigrationPackFailure::InvalidParticleClass,
							localIdx);
						continue;
					}

					// compute owner once and use it for both keeping and migration routing
					const int ownerRank =
						domainDecomp.OwnerRank(
							FluidPositionAtLocalIdx(localState.fluidParticles, localIdx));
					if (ownerRank < 0)
					{
						StorePackFailure(
							failureCode, failureLocalIdx, failureOwnerRank,
							ParticleMigrationPackFailure::MissingOwnerRank,
							localIdx, ownerRank);
						continue;
					}

					uint64_t routeIdx = PARTICLE_MIGRATION_ROUTE_KEEP_LOCAL;
					if (ownerRank != rank)
					{
						// route particles that moved into another rank's domain
						if (static_cast<uint64_t>(ownerRank) >=
							workBuffers.rankBufferIdx_FromRank.size())
						{
							StorePackFailure(
								failureCode, failureLocalIdx, failureOwnerRank,
								ParticleMigrationPackFailure::OwnerRankOutOfWorld,
								localIdx, ownerRank);
							continue;
						}

						const int migrationIdx =
							workBuffers.rankBufferIdx_FromRank[static_cast<uint64_t>(ownerRank)];
						if (migrationIdx < 0 ||
							static_cast<uint64_t>(migrationIdx) >= migrationCount)
						{
							StorePackFailure(
								failureCode, failureLocalIdx, failureOwnerRank,
								ParticleMigrationPackFailure::OwnerRankWithoutMigrationBuffer,
								localIdx, ownerRank);
							continue;
						}

						routeIdx = static_cast<uint64_t>(migrationIdx) + 1;
					}

					workBuffers.migrationRouteIdx_FromOwnedIdx[ownedIdx] =
						static_cast<int>(routeIdx);
					workBuffers.migrationRouteCounts_ByChunkRoute[
						chunkRouteBase + routeIdx]++;
					if (routeIdx != PARTICLE_MIGRATION_ROUTE_KEEP_LOCAL)
					{
						workBuffers.migrationOutCounts_ByChunk[chunkIdx]++;
					}
				}
			}
		});

		const ParticleMigrationPackFailure failure =
			static_cast<ParticleMigrationPackFailure>(
				failureCode.load(std::memory_order_relaxed));
		if (failure != ParticleMigrationPackFailure::None)
		{
			return fail(PackFailureMessage(failure));
		}

		// prefix chunk-local route counts so each worker can fill a disjoint output range
		for (uint64_t routeIdx = 0; routeIdx != routeCount; routeIdx++)
		{
			uint64_t routeTotal = 0;
			for (uint64_t chunkIdx = 0; chunkIdx != chunkCount; chunkIdx++)
			{
				const uint64_t chunkRouteIdx = chunkIdx * routeCount + routeIdx;
				workBuffers.migrationRouteWriteOffsets_ByChunkRoute[chunkRouteIdx] =
					routeTotal;
				routeTotal += workBuffers.migrationRouteCounts_ByChunkRoute[chunkRouteIdx];
			}
			workBuffers.migrationRouteTotals[routeIdx] = routeTotal;
		}

		uint64_t migratingOutParticleCount = 0;
		for (uint64_t chunkIdx = 0; chunkIdx != chunkCount; chunkIdx++)
		{
			workBuffers.migrationOutWriteOffsets_ByChunk[chunkIdx] =
				migratingOutParticleCount;
			migratingOutParticleCount += workBuffers.migrationOutCounts_ByChunk[chunkIdx];
		}

		workBuffers.keptOwnedFluidLocalIdx.resize(
			workBuffers.migrationRouteTotals[PARTICLE_MIGRATION_ROUTE_KEEP_LOCAL]);
		workBuffers.migratingOutFluidLocalIdx.resize(migratingOutParticleCount);
		for (uint64_t migrationIdx = 0; migrationIdx != migrationCount; migrationIdx++)
		{
			ParticleMigrationWorkBuffers::RankBuffers& migration =
				workBuffers.rankBuffers[migrationIdx];
			migration.sendBuffer.fluidParticles.resize(
				workBuffers.migrationRouteTotals[migrationIdx + 1]);
		}

		// pack kept indices and migrating payloads into pre-sized output buffers
		tbb::parallel_for(
			tbb::blocked_range<uint64_t>(0, chunkCount),
			[&](const tbb::blocked_range<uint64_t>& range)
		{
			for (uint64_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint64_t beginOwnedIdx =
					chunkIdx * PARTICLE_MIGRATION_PACK_CHUNK_SIZE;
				const uint64_t endOwnedIdx = std::min(
					beginOwnedIdx + PARTICLE_MIGRATION_PACK_CHUNK_SIZE,
					ownedParticleCount);
				const uint64_t chunkRouteBase = chunkIdx * routeCount;

				for (uint64_t ownedIdx = beginOwnedIdx; ownedIdx != endOwnedIdx; ownedIdx++)
				{
					const int localIdx = ownedLocalIdxs[ownedIdx];
					const uint64_t routeIdx =
						static_cast<uint64_t>(workBuffers.migrationRouteIdx_FromOwnedIdx[ownedIdx]);
					const uint64_t dstIdx =
						workBuffers.migrationRouteWriteOffsets_ByChunkRoute[
							chunkRouteBase + routeIdx]++;

					if (routeIdx == PARTICLE_MIGRATION_ROUTE_KEEP_LOCAL)
					{
						workBuffers.keptOwnedFluidLocalIdx[dstIdx] = localIdx;
					}
					else
					{
						const uint64_t outIdx =
							workBuffers.migrationOutWriteOffsets_ByChunk[chunkIdx]++;
						workBuffers.migratingOutFluidLocalIdx[outIdx] = localIdx;

						ParticleMigrationWorkBuffers::RankBuffers& migration =
							workBuffers.rankBuffers[routeIdx - 1];
						migration.sendBuffer.fluidParticles[dstIdx] =
							PackFluidParticle(
								localState.fluidParticles, fluidParticleState, localIdx);
					}
				}
			}
		});

		uint64_t sentFluidParticleCount = 0;
		for (ParticleMigrationWorkBuffers::RankBuffers& migration : workBuffers.rankBuffers)
		{
			migration.sendFluidCount = CheckedPacketCount(migration.sendBuffer.fluidParticles);
			sentFluidParticleCount += migration.sendBuffer.fluidParticles.size();
		}
		if (sentFluidParticleCount > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
		{
			return fail("sent migrating fluid particle count exceeds uint32_t range.");
		}
		stats.sentFluidParticleCount = static_cast<uint32_t>(sentFluidParticleCount);

		return true;
	}
}

void ParticleMigrationBuffer::Clear()
{
	fluidParticles.clear();
}

// migrate owned fluid particles with parallel packing, MPI exchange, aggregation, and rebuild
bool Dist::MigrateFluidParticlesToOwnerRanks(
	Sync::UniversalSimState& localState,
	DistParticleState& fluidParticleState,
	const IDomainDecomp& domainDecomp,
	const MpiContext& mpiContext,
	const uint32_t globalFluidParticleCount,
	ParticleMigrationWorkBuffers& workBuffers,
	ParticleMigrationStats* stats,
	std::string* failMsg,
	ParticleMigrationTimingReport* timingReport)
{
	if (failMsg) { failMsg->clear(); }
	if (timingReport)
	{
		BeginParticleMigrationTimingReport(*timingReport, mpiContext.Rank());
	}

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};

	// prepare migration packages for every peer rank
	ParticleMigrationStats localStats{};
	std::vector<ParticleMigrationWorkBuffers::RankBuffers>& migrations = workBuffers.rankBuffers;
	PrepareParticleMigrationRankBuffers(workBuffers, mpiContext.Rank(), mpiContext.Size());

	{
		[[maybe_unused]] const ParticleMigrationTimingScope timingScope(
			timingReport, ParticleMigrationTimingOp::Pack);

		// route and pack migration packages in one owned-particle scan
		std::string localPackFailMsg{};
		const bool localPackOk = PackMigratingFluidParticlesByOwnerRank(
			workBuffers, localStats, localState, fluidParticleState,
			domainDecomp, mpiContext.Rank(), &localPackFailMsg);

		if (!mpiContext.AllTrue(localPackOk))
		{
			if (!localPackOk) { return fail(localPackFailMsg); }
			return fail("at least one rank failed to pack particle migration buffers.");
		}
	}

	{
		[[maybe_unused]] const ParticleMigrationTimingScope timingScope(
			timingReport, ParticleMigrationTimingOp::CountExchange);

		// exchange rank-indexed particle counts before exchanging raw payloads
		workBuffers.sendFluidCount_FromRank.assign(
			static_cast<uint64_t>(mpiContext.Size()), 0);
		for (const ParticleMigrationWorkBuffers::RankBuffers& migration : migrations)
		{
			workBuffers.sendFluidCount_FromRank[static_cast<uint64_t>(migration.rank)] =
				migration.sendFluidCount;
		}

		mpiContext.AllToAllInts(
			workBuffers.sendFluidCount_FromRank,
			workBuffers.recvFluidCount_FromRank);

		// resize receive buffers according to exchanged particle counts
		bool localCountOk = true;
		std::string localCountFailMsg{};
		for (ParticleMigrationWorkBuffers::RankBuffers& migration : migrations)
		{
			migration.recvFluidCount =
				workBuffers.recvFluidCount_FromRank[static_cast<uint64_t>(migration.rank)];
			if (migration.recvFluidCount < 0)
			{
				localCountOk = false;
				localCountFailMsg = "received negative particle migration packet count.";
				break;
			}

			migration.recvBuffer.fluidParticles.resize(
				static_cast<uint64_t>(migration.recvFluidCount));
		}
		if (!mpiContext.AllTrue(localCountOk))
		{
			if (!localCountOk) { return fail(localCountFailMsg); }
			return fail("at least one rank received an invalid particle migration packet count.");
		}
	}

	{
		[[maybe_unused]] const ParticleMigrationTimingScope timingScope(
			timingReport, ParticleMigrationTimingOp::PayloadExchange);

		std::vector<MPI_Request> requests{};
		requests.reserve(migrations.size() * 2);

		// exchange only non-empty payload chunks after all receive buffers are sized correctly
		for (ParticleMigrationWorkBuffers::RankBuffers& migration : migrations) // post non-blocking MPI receives
		{
			if (migration.recvFluidCount <= 0) { continue; }

			mpiContext.IRecvBytesChunked(
				DataOrNull(migration.recvBuffer.fluidParticles),
				CheckedByteSize(migration.recvBuffer.fluidParticles),
				migration.rank,
				PARTICLE_MIGRATION_TAG_FLUID_PAYLOAD,
				requests);
		}
		for (ParticleMigrationWorkBuffers::RankBuffers& migration : migrations) // post non-blocking MPI sends
		{
			if (migration.sendFluidCount <= 0) { continue; }

			mpiContext.ISendBytesChunked(
				DataOrNull(migration.sendBuffer.fluidParticles),
				CheckedByteSize(migration.sendBuffer.fluidParticles),
				migration.rank,
				PARTICLE_MIGRATION_TAG_FLUID_PAYLOAD,
				requests);
		}
		mpiContext.WaitAll(requests);
		requests.clear();
	}

	ParticleMigrationBuffer& recvBuffer = workBuffers.aggregatedRecvBuffer;

	{
		[[maybe_unused]] const ParticleMigrationTimingScope timingScope(
			timingReport, ParticleMigrationTimingOp::Aggregate);

		// combine received migrating particles from all ranks into a single receive buffer
		recvBuffer.Clear();

		uint64_t recvFluidParticleCount = 0;
		for (const ParticleMigrationWorkBuffers::RankBuffers& migration : migrations)
		{
			recvFluidParticleCount += migration.recvBuffer.fluidParticles.size();
		}
		recvBuffer.fluidParticles.reserve(recvFluidParticleCount);

		for (const ParticleMigrationWorkBuffers::RankBuffers& migration : migrations)
		{
			recvBuffer.fluidParticles.insert(
				recvBuffer.fluidParticles.end(),
				migration.recvBuffer.fluidParticles.begin(),
				migration.recvBuffer.fluidParticles.end());
		}

		localStats.recvFluidParticleCount = static_cast<uint32_t>(recvBuffer.fluidParticles.size());
	}

	{
		[[maybe_unused]] const ParticleMigrationTimingScope timingScope(
			timingReport, ParticleMigrationTimingOp::Rebuild);

		std::string rebuildFailMsg{};
		const bool localRebuildOk = RebuildLocalStateWithMigratedFluidParticles(
			recvBuffer, workBuffers.keptOwnedFluidLocalIdx,
			workBuffers.migratingOutFluidLocalIdx,
			localState, fluidParticleState, domainDecomp, mpiContext.Rank(),
			globalFluidParticleCount, workBuffers, &rebuildFailMsg);

		const bool globalRebuildOk = mpiContext.AllTrue(localRebuildOk);
		if (!globalRebuildOk)
		{
			if (!localRebuildOk)
			{
				return fail("failed to rebuild local state with migrated fluid particles: " +
					rebuildFailMsg);
			}
			return fail("at least one rank failed to rebuild local state with migrated fluid particles.");
		}
	}

	if (stats) { *stats = localStats; }

	return true;
}

bool Dist::RebuildLocalStateWithMigratedFluidParticles(
	const ParticleMigrationBuffer& recvBuffer,
	const std::vector<int>& keptOwnedFluidLocalIdx,
	const std::vector<int>& migratingOutFluidLocalIdx,
	Sync::UniversalSimState& localState,
	DistParticleState& fluidParticleState,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const uint32_t globalFluidParticleCount,
	ParticleMigrationWorkBuffers& workBuffers,
	std::string* failMsg)
{
	if (failMsg) { failMsg->clear(); }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};

	const uint32_t oldLocalParticleCount = localState.fluidParticles.N_;
	const uint64_t oldOwnedParticleCount = fluidParticleState.particleLocalIdx_Owned.size();

	if (oldOwnedParticleCount != fluidParticleState.ownedParticleCount)
	{
		return fail("owned fluid class list size does not match ownedParticleCount.");
	}
	if (static_cast<uint64_t>(fluidParticleState.particleGlobalIdx_FromLocalIdx.size()) !=
		oldLocalParticleCount)
	{
		return fail("fluid local-to-global map size does not match local fluid particle count.");
	}
	if (static_cast<uint64_t>(fluidParticleState.particleLocalIdx_FromGlobalIdx.size()) !=
		globalFluidParticleCount)
	{
		return fail("fluid global-to-local map size does not match global fluid particle count.");
	}
	if (keptOwnedFluidLocalIdx.size() > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
	{
		return fail("kept owned fluid particle count exceeds uint32_t local particle count range.");
	}
	if (migratingOutFluidLocalIdx.size() >
		static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
	{
		return fail("migrating-out fluid particle count exceeds uint32_t local particle count range.");
	}
	if (keptOwnedFluidLocalIdx.size() + migratingOutFluidLocalIdx.size() !=
		oldOwnedParticleCount)
	{
		return fail("kept plus migrating-out fluid particle counts do not match ownedParticleCount.");
	}

	const uint32_t keptParticleCount =
		static_cast<uint32_t>(keptOwnedFluidLocalIdx.size());

	if (recvBuffer.fluidParticles.size() >
		static_cast<uint64_t>(std::numeric_limits<uint32_t>::max() - keptParticleCount))
	{
		return fail("migrated fluid particle count exceeds uint32_t local particle count range.");
	}

	const uint32_t receivedParticleCount = static_cast<uint32_t>(recvBuffer.fluidParticles.size());
	const uint32_t localParticleCount = keptParticleCount + receivedParticleCount;

	const uint32_t migrationStatusStamp =
		PrepareMigrationLocalStatusStamp(workBuffers, oldLocalParticleCount);

	// validate and mark owned fluid particles that stay on this rank
	for (const int localIdx : keptOwnedFluidLocalIdx)
	{
		if (!IsLocalIdxInRange(localIdx, oldLocalParticleCount))
		{
			return fail("kept owned fluid local index is outside the local fluid particle range.");
		}
		if (!HasParticleMetadataAtLocalIdx(fluidParticleState, localIdx))
		{
			return fail("kept owned fluid local index has no particle metadata.");
		}
		if (MigrationLocalStatusAt(
			workBuffers, migrationStatusStamp, static_cast<uint32_t>(localIdx)) !=
			PARTICLE_MIGRATION_LOCAL_STATUS_NONE)
		{
			return fail("kept owned fluid local index appears more than once.");
		}

		const int globalIdx = fluidParticleState.particleGlobalIdx_FromLocalIdx[localIdx];
		if (!IsGlobalIdxInRange(globalIdx, globalFluidParticleCount))
		{
			return fail("kept owned fluid local index maps to an invalid global index.");
		}
		if (fluidParticleState.particleLocalIdx_FromGlobalIdx[globalIdx] != localIdx)
		{
			return fail("kept owned fluid particle map entry does not round-trip.");
		}

		SetMigrationLocalStatus(
			workBuffers, migrationStatusStamp, static_cast<uint32_t>(localIdx),
			PARTICLE_MIGRATION_LOCAL_STATUS_KEPT);
	}

	// validate and mark owned fluid particles that migrated out of this rank
	for (const int localIdx : migratingOutFluidLocalIdx)
	{
		if (!IsLocalIdxInRange(localIdx, oldLocalParticleCount))
		{
			return fail("migrating-out fluid local index is outside the local fluid particle range.");
		}
		if (!HasParticleMetadataAtLocalIdx(fluidParticleState, localIdx))
		{
			return fail("migrating-out fluid local index has no particle metadata.");
		}
		if (MigrationLocalStatusAt(
			workBuffers, migrationStatusStamp, static_cast<uint32_t>(localIdx)) !=
			PARTICLE_MIGRATION_LOCAL_STATUS_NONE)
		{
			return fail("migrating-out fluid local index appears more than once.");
		}

		const int globalIdx = fluidParticleState.particleGlobalIdx_FromLocalIdx[localIdx];
		if (!IsGlobalIdxInRange(globalIdx, globalFluidParticleCount))
		{
			return fail("migrating-out fluid local index maps to an invalid global index.");
		}
		if (fluidParticleState.particleLocalIdx_FromGlobalIdx[globalIdx] != localIdx)
		{
			return fail("migrating-out fluid particle map entry does not round-trip.");
		}

		SetMigrationLocalStatus(
			workBuffers, migrationStatusStamp, static_cast<uint32_t>(localIdx),
			PARTICLE_MIGRATION_LOCAL_STATUS_OUT);
	}

	// remove global map entries for particles that left this rank
	for (const int localIdx : migratingOutFluidLocalIdx)
	{
		if (!ClearLocalMapEntryIfStillMapped(
			fluidParticleState, localIdx, globalFluidParticleCount))
		{
			return fail("failed to clear migrating-out fluid particle map entry.");
		}
	}

	// fill compact-prefix holes with kept particles that currently live outside the prefix
	uint64_t keptSourceIdx = 0;
	for (uint32_t dstLocalIdx = 0; dstLocalIdx < keptParticleCount; dstLocalIdx++)
	{
		if (MigrationLocalStatusAt(
			workBuffers, migrationStatusStamp, dstLocalIdx) ==
			PARTICLE_MIGRATION_LOCAL_STATUS_KEPT)
		{
			continue;
		}

		int srcLocalIdx = -1;
		while (keptSourceIdx < keptOwnedFluidLocalIdx.size())
		{
			const int candidateLocalIdx = keptOwnedFluidLocalIdx[keptSourceIdx++];
			if (static_cast<uint32_t>(candidateLocalIdx) >= keptParticleCount)
			{
				srcLocalIdx = candidateLocalIdx;
				break;
			}
		}
		if (srcLocalIdx < 0)
		{
			return fail("not enough kept owned fluid particles outside the compact prefix.");
		}

		// clear stale halo/outgoing metadata at the destination before overwriting it
		if (!ClearLocalMapEntryIfStillMapped(
			fluidParticleState, static_cast<int>(dstLocalIdx), globalFluidParticleCount))
		{
			return fail("failed to clear overwritten fluid particle map entry.");
		}

		const int movedGlobalIdx =
			fluidParticleState.particleGlobalIdx_FromLocalIdx[srcLocalIdx];
		if (!IsGlobalIdxInRange(movedGlobalIdx, globalFluidParticleCount))
		{
			return fail("kept fluid compaction source maps to an invalid global index.");
		}
		if (fluidParticleState.particleLocalIdx_FromGlobalIdx[movedGlobalIdx] !=
			srcLocalIdx)
		{
			return fail("kept fluid compaction source map entry does not round-trip.");
		}

		CopyFluidParticle_LocalToLocal(
			localState.fluidParticles, localState.fluidParticles,
			srcLocalIdx, static_cast<int>(dstLocalIdx));
		fluidParticleState.particleGlobalIdx_FromLocalIdx[dstLocalIdx] = movedGlobalIdx;
		fluidParticleState.particleLocalIdx_FromGlobalIdx[movedGlobalIdx] =
			static_cast<int>(dstLocalIdx);
	}

	// clear stale map entries from the dropped tail (old halos, outgoing particles, moved sources)
	for (uint32_t localIdx = keptParticleCount; localIdx < oldLocalParticleCount; localIdx++)
	{
		if (!ClearLocalMapEntryIfStillMapped(
			fluidParticleState, static_cast<int>(localIdx), globalFluidParticleCount))
		{
			return fail("failed to clear dropped fluid particle map entry.");
		}
	}

	localState.fluidParticles.resizeAll(localParticleCount);
	fluidParticleState.particleGlobalIdx_FromLocalIdx.resize(localParticleCount);
	fluidParticleState.particleClasses.resize(localParticleCount, DistParticleClass::None);
	fluidParticleState.particleLocalIdx_Owned.clear();
	fluidParticleState.particleLocalIdx_Inner.clear();
	fluidParticleState.particleLocalIdx_Outer.clear();
	fluidParticleState.particleLocalIdx_Halo.clear();
	fluidParticleState.ownedParticleCount = localParticleCount;
	fluidParticleState.haloParticleCount = 0;

	// map received fluid particles whose new owner is this rank
	for (uint64_t packetIdx = 0; packetIdx != recvBuffer.fluidParticles.size(); packetIdx++)
	{
		const MigratingFluidParticlePacket& packet =
			recvBuffer.fluidParticles[packetIdx];
		const int dstLocalIdx =
			static_cast<int>(keptParticleCount + packetIdx);

		if (domainDecomp.OwnerRank(FluidPositionFromPacket(packet)) != rank)
		{
			return fail("received migrating fluid particle does not belong to this rank.");
		}
		if (!AddParticleMapEntry(
			fluidParticleState, packet.globalIdx, dstLocalIdx, globalFluidParticleCount))
		{
			return fail("failed to add received migrating fluid particle map entry.");
		}

		CopyFluidPacketToLocal(packet, localState.fluidParticles, dstLocalIdx);
	}

	if (fluidParticleState.particleGlobalIdx_FromLocalIdx.size() != localParticleCount)
	{
		return fail("rebuilt fluid local-to-global map size does not match expected local particle count.");
	}

	if (localState.fluidParticles.N_ != localParticleCount)
	{
		return fail("rebuilt fluid particle count does not match expected local particle count.");
	}

	// full class-list/layout validation runs after the single Dist_ParticleClass pass
	return true;
}
