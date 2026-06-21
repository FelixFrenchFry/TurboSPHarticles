// particle_migration.cpp

// external
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

// internal
#include "particle_migration.h"
#include "mpi_context.h"



using namespace Dist;

namespace
{
	constexpr int PARTICLE_MIGRATION_TAG_FLUID_COUNT = 5101;
	constexpr int PARTICLE_MIGRATION_TAG_FLUID_PAYLOAD = 5102;

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
	int CheckedByteCount(const std::vector<T>& packets)
	{
		const uint64_t packetCount = packets.size();
		const uint64_t maxCount =
			static_cast<uint64_t>(std::numeric_limits<int>::max()) / sizeof(T);

		if (packetCount > maxCount)
		{
			throw std::runtime_error("particle migration payload exceeds MPI byte count range.");
		}

		return static_cast<int>(packetCount * sizeof(T));
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

	bool CountKeptOwnedFluidParticles(
		uint32_t& keptParticleCount,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const IDomainDecomp& domainDecomp,
		const int rank,
		std::string* failMsg)
	{
		keptParticleCount = 0;

		const auto fail = [failMsg](std::string message) -> bool
		{
			if (failMsg) { *failMsg = std::move(message); }
			return false;
		};

		for (const int localIdx : fluidParticleState.particleLocalIdx_Owned)
		{
			if (!IsLocalIdxInRange(localIdx, localState.fluidParticles.N_))
			{
				return fail("owned fluid local index is outside the local fluid particle range.");
			}
			if (!HasParticleMetadataAtLocalIdx(fluidParticleState, localIdx))
			{
				return fail("owned fluid local index has no particle metadata.");
			}

			const int ownerRank =
				domainDecomp.OwnerRank(FluidPositionAtLocalIdx(localState.fluidParticles, localIdx));
			if (ownerRank < 0)
			{
				return fail("owned fluid particle has no valid owner rank.");
			}
			if (ownerRank == rank) { keptParticleCount++; }
		}

		return true;
	}
}

void ParticleMigrationBuffer::Clear()
{
	fluidParticles.clear();
}

// TODO: re-implement with TBB (parallel count -> prefix sum -> parallel fill)?
bool Dist::PackMigratingFluidParticlesForRank(
	ParticleMigrationBuffer& sendBuffer,
	const Sync::UniversalSimState& localState,
	const DistParticleState& fluidParticleState,
	const IDomainDecomp& domainDecomp,
	const int dstRank,
	std::string* failMsg)
{
	if (failMsg) { failMsg->clear(); }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};

	sendBuffer.Clear();
	if (dstRank < 0) { return fail("destination rank is negative."); }

	// pack owned fluid particles that have moved into the domain of destination rank
	for (const int localIdx : fluidParticleState.particleLocalIdx_Owned)
	{
		// invariant checks for correct metadata
		if (!IsLocalIdxInRange(localIdx, localState.fluidParticles.N_))
		{
			return fail("owned fluid local index is outside the local fluid particle range.");
		}
		if (!HasParticleMetadataAtLocalIdx(fluidParticleState, localIdx))
		{
			return fail("owned fluid local index has no particle metadata.");
		}

		const DistParticleClass particleClass = fluidParticleState.particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer)
		{
			return fail("owned fluid local index class is not Inner or Outer.");
		}

		const int ownerRank =
			domainDecomp.OwnerRank(FluidPositionAtLocalIdx(localState.fluidParticles, localIdx));
		if (ownerRank < 0)
		{
			return fail("owned fluid particle has no valid owner rank.");
		}
		if (ownerRank != dstRank) { continue; }

		sendBuffer.fluidParticles.push_back(
			PackFluidParticle(localState.fluidParticles, fluidParticleState, localIdx));
	}

	return true;
}

// TODO: re-implement with TBB (parallel count -> prefix sum -> parallel fill)?
bool Dist::MigrateFluidParticlesToOwnerRanks(
	Sync::UniversalSimState& localState,
	DistParticleState& fluidParticleState,
	const IDomainDecomp& domainDecomp,
	const MpiContext& mpiContext,
	const float kernelRadius,
	const uint32_t globalFluidParticleCount,
	ParticleMigrationWorkBuffers& workBuffers,
	ParticleMigrationStats* stats,
	std::string* failMsg)
{
	if (failMsg) { failMsg->clear(); }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};


	// prepare migration package for each neighbor
	ParticleMigrationStats localStats{};
	std::vector<ParticleMigrationWorkBuffers::RankBuffers>& migrations = workBuffers.rankBuffers;
	migrations.resize(static_cast<uint64_t>(std::max(0, mpiContext.Size() - 1)));

	// pack migration packages
	bool localPackOk = true;
	std::string localPackFailMsg{};
	uint64_t migrationIdx = 0;
	for (int rank = 0; rank < mpiContext.Size(); rank++)
	{
		if (!localPackOk) { break; }
		if (rank == mpiContext.Rank()) { continue; }

		ParticleMigrationWorkBuffers::RankBuffers& migration = migrations[migrationIdx++];
		migration.rank = rank;
		migration.sendFluidCount = 0;
		migration.recvFluidCount = 0;

		std::string packFailMsg{};
		if (!PackMigratingFluidParticlesForRank(
			migration.sendBuffer, localState, fluidParticleState,
			domainDecomp, rank, &packFailMsg))
		{
			localPackOk = false;
			localPackFailMsg =
				"failed to pack particle migration buffer for rank " +
				std::to_string(rank) + ": " + packFailMsg;
			break;
		}

		migration.sendFluidCount = CheckedPacketCount(migration.sendBuffer.fluidParticles);
		localStats.sentFluidParticleCount += static_cast<uint32_t>(migration.sendBuffer.fluidParticles.size());
	}

	if (!mpiContext.AllTrue(localPackOk))
	{
		if (!localPackOk) { return fail(localPackFailMsg); }
		return fail("at least one rank failed to pack particle migration buffers.");
	}

	std::vector<MPI_Request> requests{};
	requests.reserve(migrations.size() * 2);

	// exchange particle counts before exchanging raw payloads
	for (ParticleMigrationWorkBuffers::RankBuffers& migration : migrations) // post non-blocking MPI receives
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.IRecvInt(
			migration.recvFluidCount,
			migration.rank,
			PARTICLE_MIGRATION_TAG_FLUID_COUNT,
			requests.back());
	}
	for (ParticleMigrationWorkBuffers::RankBuffers& migration : migrations) // post non-blocking MPI sends
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			migration.sendFluidCount,
			migration.rank,
			PARTICLE_MIGRATION_TAG_FLUID_COUNT,
			requests.back());
	}
	mpiContext.WaitAll(requests);
	requests.clear();

	// resize receive buffers according to exchanged particle counts
	bool localCountOk = true;
	std::string localCountFailMsg{};
	for (ParticleMigrationWorkBuffers::RankBuffers& migration : migrations)
	{
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

	// exchange payloads after all receive buffers are sized correctly
	for (ParticleMigrationWorkBuffers::RankBuffers& migration : migrations) // post non-blocking MPI receives
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.IRecvBytes(
			DataOrNull(migration.recvBuffer.fluidParticles),
			CheckedByteCount(migration.recvBuffer.fluidParticles),
			migration.rank,
			PARTICLE_MIGRATION_TAG_FLUID_PAYLOAD,
			requests.back());
	}
	for (ParticleMigrationWorkBuffers::RankBuffers& migration : migrations) // post non-blocking MPI sends
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendBytes(
			DataOrNull(migration.sendBuffer.fluidParticles),
			CheckedByteCount(migration.sendBuffer.fluidParticles),
			migration.rank,
			PARTICLE_MIGRATION_TAG_FLUID_PAYLOAD,
			requests.back());
	}
	mpiContext.WaitAll(requests);

	// combine received migrating particles from all ranks into a single receive buffer
	ParticleMigrationBuffer& recvBuffer = workBuffers.aggregatedRecvBuffer;
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

	localStats.receivedFluidParticleCount = static_cast<uint32_t>(recvBuffer.fluidParticles.size());

	std::string rebuildFailMsg{};
	const bool localRebuildOk = RebuildLocalStateWithMigratedFluidParticles(
		recvBuffer, localState, fluidParticleState, domainDecomp, mpiContext.Rank(),
		kernelRadius, globalFluidParticleCount, workBuffers, &rebuildFailMsg);

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

	if (stats) { *stats = localStats; }

	return true;
}

bool Dist::RebuildLocalStateWithMigratedFluidParticles(
	const ParticleMigrationBuffer& recvBuffer,
	Sync::UniversalSimState& localState,
	DistParticleState& fluidParticleState,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float kernelRadius,
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

	uint32_t keptParticleCount = 0;
	std::string countFailMsg{};
	if (!CountKeptOwnedFluidParticles(
		keptParticleCount, localState, fluidParticleState, 
		domainDecomp, rank, &countFailMsg))
	{
		return fail("failed to count kept owned fluid particles: " + countFailMsg);
	}

	if (recvBuffer.fluidParticles.size() >
		static_cast<uint64_t>(std::numeric_limits<uint32_t>::max() - keptParticleCount))
	{
		return fail("migrated fluid particle count exceeds uint32_t local particle count range.");
	}

	const uint32_t receivedParticleCount = static_cast<uint32_t>(recvBuffer.fluidParticles.size());
	const uint32_t localParticleCount = keptParticleCount + receivedParticleCount;

	Sync::FluidParticles& rebuiltFluidParticles = workBuffers.rebuiltFluidParticles;
	DistParticleState& rebuiltParticleState = workBuffers.rebuiltParticleState;

	rebuiltFluidParticles.resizeAll(localParticleCount);
	rebuiltParticleState.Clear();
	rebuiltParticleState.particleGlobalIdx_FromLocalIdx.assign(localParticleCount, -1);
	rebuiltParticleState.particleLocalIdx_FromGlobalIdx.assign(globalFluidParticleCount, -1);

	// keep owned fluid particles that still belong to this rank
	int dstLocalIdx = 0;
	for (const int srcLocalIdx : fluidParticleState.particleLocalIdx_Owned)
	{
		if (!IsLocalIdxInRange(srcLocalIdx, localState.fluidParticles.N_))
		{
			return fail("owned fluid local index is outside the local fluid particle range.");
		}
		if (!HasParticleMetadataAtLocalIdx(fluidParticleState, srcLocalIdx))
		{
			return fail("owned fluid local index has no particle metadata.");
		}

		const int ownerRank =
			domainDecomp.OwnerRank(FluidPositionAtLocalIdx(localState.fluidParticles, srcLocalIdx));
		if (ownerRank < 0)
		{
			return fail("owned fluid particle has no valid owner rank.");
		}
		if (ownerRank != rank) { continue; }

		const int globalIdx = fluidParticleState.particleGlobalIdx_FromLocalIdx[srcLocalIdx];
		if (!AddParticleMapEntry(
			rebuiltParticleState, globalIdx, dstLocalIdx, globalFluidParticleCount))
		{
			return fail("failed to add kept owned fluid particle map entry.");
		}

		CopyFluidParticle_LocalToLocal(
			localState.fluidParticles, rebuiltFluidParticles, srcLocalIdx, dstLocalIdx);

		dstLocalIdx++;
	}

	// append received fluid particles whose new owner is this rank
	for (const MigratingFluidParticlePacket& packet : recvBuffer.fluidParticles)
	{
		if (domainDecomp.OwnerRank(FluidPositionFromPacket(packet)) != rank)
		{
			return fail("received migrating fluid particle does not belong to this rank.");
		}
		if (!AddParticleMapEntry(
			rebuiltParticleState, packet.globalIdx, dstLocalIdx, globalFluidParticleCount))
		{
			return fail("failed to add received migrating fluid particle map entry.");
		}

		CopyFluidPacketToLocal(packet, rebuiltFluidParticles, dstLocalIdx);
		dstLocalIdx++;
	}

	if (dstLocalIdx != static_cast<int>(localParticleCount))
	{
		return fail("rebuilt fluid particle count does not match expected local particle count.");
	}

	// publish rebuilt state and keep old allocations available for the next migration
	std::swap(localState.fluidParticles, rebuiltFluidParticles);
	std::swap(fluidParticleState, rebuiltParticleState);

	// refresh inner/outer class lists for the compact owned-only rebuilt state
	ClassifyOwnedFluidParticlesIntoInnerOuter(
		fluidParticleState, localState.fluidParticles,
		domainDecomp, rank, kernelRadius);

#if defined(_DEBUG)
	return fluidParticleState.IsValidLocalLayout(localParticleCount, failMsg);
#else
	return true;
#endif
}
