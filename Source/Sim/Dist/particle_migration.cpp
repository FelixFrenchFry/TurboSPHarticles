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



using namespace Dist;

namespace
{
	constexpr int PARTICLE_MIGRATION_TAG_FLUID_PAYLOAD = 5102;
	constexpr uint64_t PARTICLE_MIGRATION_PACK_CHUNK_SIZE = 4096;
	constexpr int PARTICLE_MIGRATION_ROUTE_KEEP_LOCAL = 0;

	struct MigrationLocalStatusStamps
	{
		uint32_t kept = 0;
		uint32_t out = 0;
	};

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
	uint64_t CheckedByteSize(const std::vector<T>& values)
	{
		const uint64_t valueCount = values.size();
		const uint64_t maxCount =
			std::numeric_limits<uint64_t>::max() / sizeof(T);

		if (valueCount > maxCount)
		{
			throw std::runtime_error("Particle migration payload byte size exceeds uint64_t range.");
		}

		return valueCount * sizeof(T);
	}

	template <typename T>
	uint64_t UsedByteSize(const std::vector<T>& values)
	{
		return static_cast<uint64_t>(values.size()) * sizeof(T);
	}

	template <typename T>
	uint64_t AllocByteSize(const std::vector<T>& values)
	{
		return static_cast<uint64_t>(values.capacity()) * sizeof(T);
	}

	template <typename T>
	const void* DataOrNull(const std::vector<T>& values)
	{
		return values.empty() ? nullptr : values.data();
	}

	template <typename T>
	void* DataOrNull(std::vector<T>& values)
	{
		return values.empty() ? nullptr : values.data();
	}

	template <typename T>
	void QueueIRecvVector(
		const MpiContext& mpiContext,
		std::vector<T>& values,
		const int peerRank,
		const int tag,
		std::vector<MPI_Request>& requests)
	{
		mpiContext.IRecvBytesChunked(
			DataOrNull(values),
			CheckedByteSize(values),
			peerRank,
			tag,
			requests);
	}

	template <typename T>
	void QueueISendVector(
		const MpiContext& mpiContext,
		const std::vector<T>& values,
		const int peerRank,
		const int tag,
		std::vector<MPI_Request>& requests)
	{
		mpiContext.ISendBytesChunked(
			DataOrNull(values),
			CheckedByteSize(values),
			peerRank,
			tag,
			requests);
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

	void CopyFluidParticle_LocalToLocal(
		const Sync::FluidParticles& src,
		Sync::FluidParticles& dst,
		const int srcLocalIdx,
		const int dstLocalIdx)
	{
		dst.id[dstLocalIdx] = src.id[srcLocalIdx];

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
}

void MigratingParticleBufferF::Clear()
{
	globalIdx_.clear();
	id_.clear();
	xPos_.clear();
	yPos_.clear();
	zPos_.clear();
	xVel_.clear();
	yVel_.clear();
	zVel_.clear();
	xVelPred_.clear();
	yVelPred_.clear();
	zVelPred_.clear();
	xAccPress_.clear();
	yAccPress_.clear();
	zAccPress_.clear();
	sourceTerm_.clear();
	diagElem_.clear();
	pressure_.clear();
	density_.clear();
}

void MigratingParticleBufferF::Resize(const uint64_t particleCount)
{
	globalIdx_.resize(particleCount);
	id_.resize(particleCount);
	xPos_.resize(particleCount);
	yPos_.resize(particleCount);
	zPos_.resize(particleCount);
	xVel_.resize(particleCount);
	yVel_.resize(particleCount);
	zVel_.resize(particleCount);
	xVelPred_.resize(particleCount);
	yVelPred_.resize(particleCount);
	zVelPred_.resize(particleCount);
	xAccPress_.resize(particleCount);
	yAccPress_.resize(particleCount);
	zAccPress_.resize(particleCount);
	sourceTerm_.resize(particleCount);
	diagElem_.resize(particleCount);
	pressure_.resize(particleCount);
	density_.resize(particleCount);
}

int MigratingParticleBufferF::GlobalIdxAt(
	const uint64_t bufferIdx) const
{
	return globalIdx_[bufferIdx];
}

glm::vec3 MigratingParticleBufferF::PositionAt(
	const uint64_t bufferIdx) const
{
	return
	{
		xPos_[bufferIdx],
		yPos_[bufferIdx],
		zPos_[bufferIdx]
	};
}

void MigratingParticleBufferF::SetFromLocalParticle(
	const Sync::FluidParticles& fluidParticles,
	const DistParticleState& particleState,
	const uint64_t bufferIdx,
	const int localIdx)
{
	globalIdx_[bufferIdx] = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
	id_[bufferIdx] = fluidParticles.id[localIdx];
	xPos_[bufferIdx] = fluidParticles.xPos[localIdx];
	yPos_[bufferIdx] = fluidParticles.yPos[localIdx];
	zPos_[bufferIdx] = fluidParticles.zPos[localIdx];
	xVel_[bufferIdx] = fluidParticles.xVel[localIdx];
	yVel_[bufferIdx] = fluidParticles.yVel[localIdx];
	zVel_[bufferIdx] = fluidParticles.zVel[localIdx];
	xVelPred_[bufferIdx] = fluidParticles.xVelPred[localIdx];
	yVelPred_[bufferIdx] = fluidParticles.yVelPred[localIdx];
	zVelPred_[bufferIdx] = fluidParticles.zVelPred[localIdx];
	xAccPress_[bufferIdx] = fluidParticles.xAccPress[localIdx];
	yAccPress_[bufferIdx] = fluidParticles.yAccPress[localIdx];
	zAccPress_[bufferIdx] = fluidParticles.zAccPress[localIdx];
	sourceTerm_[bufferIdx] = fluidParticles.sourceTerm[localIdx];
	diagElem_[bufferIdx] = fluidParticles.diagElem[localIdx];
	pressure_[bufferIdx] = fluidParticles.pressure[localIdx];
	density_[bufferIdx] = fluidParticles.density[localIdx];
}

void MigratingParticleBufferF::CopyToLocalParticle(
	Sync::FluidParticles& dst,
	const uint64_t bufferIdx,
	const int localIdx) const
{
	dst.id[localIdx] = id_[bufferIdx];
	dst.xPos[localIdx] = xPos_[bufferIdx];
	dst.yPos[localIdx] = yPos_[bufferIdx];
	dst.zPos[localIdx] = zPos_[bufferIdx];
	dst.xVel[localIdx] = xVel_[bufferIdx];
	dst.yVel[localIdx] = yVel_[bufferIdx];
	dst.zVel[localIdx] = zVel_[bufferIdx];
	dst.xVelPred[localIdx] = xVelPred_[bufferIdx];
	dst.yVelPred[localIdx] = yVelPred_[bufferIdx];
	dst.zVelPred[localIdx] = zVelPred_[bufferIdx];
	dst.xAccPress[localIdx] = xAccPress_[bufferIdx];
	dst.yAccPress[localIdx] = yAccPress_[bufferIdx];
	dst.zAccPress[localIdx] = zAccPress_[bufferIdx];
	dst.sourceTerm[localIdx] = sourceTerm_[bufferIdx];
	dst.diagElem[localIdx] = diagElem_[bufferIdx];
	dst.pressure[localIdx] = pressure_[bufferIdx];
	dst.density[localIdx] = density_[bufferIdx];
}

void MigratingParticleBufferF::QueueIRecvPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests)
{
	if (Empty()) { return; }

	QueueIRecvVector(mpiContext, globalIdx_, peerRank, tagBase + 1, requests);
	QueueIRecvVector(mpiContext, xPos_, peerRank, tagBase + 2, requests);
	QueueIRecvVector(mpiContext, yPos_, peerRank, tagBase + 3, requests);
	QueueIRecvVector(mpiContext, zPos_, peerRank, tagBase + 4, requests);
	QueueIRecvVector(mpiContext, xVel_, peerRank, tagBase + 5, requests);
	QueueIRecvVector(mpiContext, yVel_, peerRank, tagBase + 6, requests);
	QueueIRecvVector(mpiContext, zVel_, peerRank, tagBase + 7, requests);
	QueueIRecvVector(mpiContext, xVelPred_, peerRank, tagBase + 8, requests);
	QueueIRecvVector(mpiContext, yVelPred_, peerRank, tagBase + 9, requests);
	QueueIRecvVector(mpiContext, zVelPred_, peerRank, tagBase + 10, requests);
	QueueIRecvVector(mpiContext, xAccPress_, peerRank, tagBase + 11, requests);
	QueueIRecvVector(mpiContext, yAccPress_, peerRank, tagBase + 12, requests);
	QueueIRecvVector(mpiContext, zAccPress_, peerRank, tagBase + 13, requests);
	QueueIRecvVector(mpiContext, sourceTerm_, peerRank, tagBase + 14, requests);
	QueueIRecvVector(mpiContext, diagElem_, peerRank, tagBase + 15, requests);
	QueueIRecvVector(mpiContext, pressure_, peerRank, tagBase + 16, requests);
	QueueIRecvVector(mpiContext, density_, peerRank, tagBase + 17, requests);
	QueueIRecvVector(mpiContext, id_, peerRank, tagBase + 18, requests);
}

void MigratingParticleBufferF::QueueISendPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests) const
{
	if (Empty()) { return; }

	QueueISendVector(mpiContext, globalIdx_, peerRank, tagBase + 1, requests);
	QueueISendVector(mpiContext, xPos_, peerRank, tagBase + 2, requests);
	QueueISendVector(mpiContext, yPos_, peerRank, tagBase + 3, requests);
	QueueISendVector(mpiContext, zPos_, peerRank, tagBase + 4, requests);
	QueueISendVector(mpiContext, xVel_, peerRank, tagBase + 5, requests);
	QueueISendVector(mpiContext, yVel_, peerRank, tagBase + 6, requests);
	QueueISendVector(mpiContext, zVel_, peerRank, tagBase + 7, requests);
	QueueISendVector(mpiContext, xVelPred_, peerRank, tagBase + 8, requests);
	QueueISendVector(mpiContext, yVelPred_, peerRank, tagBase + 9, requests);
	QueueISendVector(mpiContext, zVelPred_, peerRank, tagBase + 10, requests);
	QueueISendVector(mpiContext, xAccPress_, peerRank, tagBase + 11, requests);
	QueueISendVector(mpiContext, yAccPress_, peerRank, tagBase + 12, requests);
	QueueISendVector(mpiContext, zAccPress_, peerRank, tagBase + 13, requests);
	QueueISendVector(mpiContext, sourceTerm_, peerRank, tagBase + 14, requests);
	QueueISendVector(mpiContext, diagElem_, peerRank, tagBase + 15, requests);
	QueueISendVector(mpiContext, pressure_, peerRank, tagBase + 16, requests);
	QueueISendVector(mpiContext, density_, peerRank, tagBase + 17, requests);
	QueueISendVector(mpiContext, id_, peerRank, tagBase + 18, requests);
}

int MigratingParticleBufferF::CheckedBufferCount() const
{
	if (Size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
	{
		throw std::runtime_error("Particle migration buffer count exceeds MPI int count range.");
	}

	return static_cast<int>(Size());
}

uint64_t MigratingParticleBufferF::UsedBytes() const
{
	uint64_t usedBytes = 0;

	usedBytes += UsedByteSize(globalIdx_);
	usedBytes += UsedByteSize(id_);
	usedBytes += UsedByteSize(xPos_);
	usedBytes += UsedByteSize(yPos_);
	usedBytes += UsedByteSize(zPos_);
	usedBytes += UsedByteSize(xVel_);
	usedBytes += UsedByteSize(yVel_);
	usedBytes += UsedByteSize(zVel_);
	usedBytes += UsedByteSize(xVelPred_);
	usedBytes += UsedByteSize(yVelPred_);
	usedBytes += UsedByteSize(zVelPred_);
	usedBytes += UsedByteSize(xAccPress_);
	usedBytes += UsedByteSize(yAccPress_);
	usedBytes += UsedByteSize(zAccPress_);
	usedBytes += UsedByteSize(sourceTerm_);
	usedBytes += UsedByteSize(diagElem_);
	usedBytes += UsedByteSize(pressure_);
	usedBytes += UsedByteSize(density_);

	return usedBytes;
}

uint64_t MigratingParticleBufferF::AllocBytes() const
{
	uint64_t allocBytes = 0;

	allocBytes += AllocByteSize(globalIdx_);
	allocBytes += AllocByteSize(id_);
	allocBytes += AllocByteSize(xPos_);
	allocBytes += AllocByteSize(yPos_);
	allocBytes += AllocByteSize(zPos_);
	allocBytes += AllocByteSize(xVel_);
	allocBytes += AllocByteSize(yVel_);
	allocBytes += AllocByteSize(zVel_);
	allocBytes += AllocByteSize(xVelPred_);
	allocBytes += AllocByteSize(yVelPred_);
	allocBytes += AllocByteSize(zVelPred_);
	allocBytes += AllocByteSize(xAccPress_);
	allocBytes += AllocByteSize(yAccPress_);
	allocBytes += AllocByteSize(zAccPress_);
	allocBytes += AllocByteSize(sourceTerm_);
	allocBytes += AllocByteSize(diagElem_);
	allocBytes += AllocByteSize(pressure_);
	allocBytes += AllocByteSize(density_);

	return allocBytes;
}

namespace
{
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

	MigrationLocalStatusStamps PrepareMigrationLocalStatusStamps(
		MigrationWorkBuffers& workBuffers,
		const uint32_t localParticleCount)
	{
		workBuffers.migrationLocalStatusStamp_FromLocalIdx.resize(localParticleCount, 0);

		if (workBuffers.migrationLocalStatusStamp >
			std::numeric_limits<uint32_t>::max() - 2u)
		{
			std::ranges::fill(workBuffers.migrationLocalStatusStamp_FromLocalIdx, 0);
			workBuffers.migrationLocalStatusStamp = 0;
		}

		const MigrationLocalStatusStamps stamps
		{
			workBuffers.migrationLocalStatusStamp + 1u,
			workBuffers.migrationLocalStatusStamp + 2u
		};
		workBuffers.migrationLocalStatusStamp = stamps.out;

		return stamps;
	}

	bool IsMigrationLocalStatusMarked(
		const MigrationWorkBuffers& workBuffers,
		const MigrationLocalStatusStamps& stamps,
		const uint32_t localIdx)
	{
		const uint32_t stamp = workBuffers.migrationLocalStatusStamp_FromLocalIdx[localIdx];
		return stamp == stamps.kept || stamp == stamps.out;
	}

	bool IsMigrationLocalStatusKept(
		const MigrationWorkBuffers& workBuffers,
		const MigrationLocalStatusStamps& stamps,
		const uint32_t localIdx)
	{
		return workBuffers.migrationLocalStatusStamp_FromLocalIdx[localIdx] == stamps.kept;
	}

	void SetMigrationLocalStatusKept(
		MigrationWorkBuffers& workBuffers,
		const MigrationLocalStatusStamps& stamps,
		const uint32_t localIdx)
	{
		workBuffers.migrationLocalStatusStamp_FromLocalIdx[localIdx] = stamps.kept;
	}

	void SetMigrationLocalStatusOut(
		MigrationWorkBuffers& workBuffers,
		const MigrationLocalStatusStamps& stamps,
		const uint32_t localIdx)
	{
		workBuffers.migrationLocalStatusStamp_FromLocalIdx[localIdx] = stamps.out;
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
		MigrationWorkBuffers& workBuffers,
		const int rank,
		const int rankCount)
	{
		std::vector<MigrationWorkBuffers::RankBuffers>& migrations = workBuffers.rankBuffers;
		migrations.resize(static_cast<uint64_t>(std::max(0, rankCount - 1)));

		workBuffers.rankBufferIdx_FromRank.assign(
			static_cast<uint64_t>(std::max(0, rankCount)), -1);

		uint64_t migrationIdx = 0;
		for (int dstRank = 0; dstRank < rankCount; dstRank++)
		{
			if (dstRank == rank) { continue; }

			MigrationWorkBuffers::RankBuffers& migration = migrations[migrationIdx];
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
		MigrationWorkBuffers& workBuffers,
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

		const std::vector<int>& ownedLocalIdxs = fluidParticleState.particleLocalIdx_Owned;
		const uint64_t ownedParticleCount = ownedLocalIdxs.size();
		const uint64_t migrationCount = workBuffers.rankBuffers.size();
		const uint64_t routeCount = migrationCount + 1;
		const uint64_t chunkCount = ownedParticleCount > 0
			? (ownedParticleCount + PARTICLE_MIGRATION_PACK_CHUNK_SIZE - 1) /
				PARTICLE_MIGRATION_PACK_CHUNK_SIZE
			: 0;

		workBuffers.migrationRouteIdx_FromOwnedIdx.resize(ownedParticleCount);
		workBuffers.migrationRouteCounts_ByChunkRoute.assign(chunkCount * routeCount, 0ull);
		workBuffers.migrationRouteWriteOffsets_ByChunkRoute.resize(chunkCount * routeCount);
		workBuffers.migrationRouteTotals.resize(routeCount);
		workBuffers.migrationOutCounts_ByChunk.assign(chunkCount, 0ull);
		workBuffers.migrationOutWriteOffsets_ByChunk.resize(chunkCount);

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
				const uint64_t beginOwnedIdx = chunkIdx * PARTICLE_MIGRATION_PACK_CHUNK_SIZE;
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

		workBuffers.keptOwnedFluidLocalIdx.resize(workBuffers.migrationRouteTotals[PARTICLE_MIGRATION_ROUTE_KEEP_LOCAL]);
		workBuffers.migratingOutFluidLocalIdx.resize(migratingOutParticleCount);
		for (uint64_t migrationIdx = 0; migrationIdx != migrationCount; migrationIdx++)
		{
			MigrationWorkBuffers::RankBuffers& migration =
				workBuffers.rankBuffers[migrationIdx];
			const uint64_t sendParticleCount =
				workBuffers.migrationRouteTotals[migrationIdx + 1];
			migration.sendBuffer.fluidParticles.Resize(sendParticleCount);
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

						MigrationWorkBuffers::RankBuffers& migration =
							workBuffers.rankBuffers[routeIdx - 1];
						migration.sendBuffer.fluidParticles.SetFromLocalParticle(
							localState.fluidParticles, fluidParticleState, dstIdx, localIdx);
					}
				}
			}
		});

		uint64_t sentFluidParticleCount = 0;
		for (MigrationWorkBuffers::RankBuffers& migration : workBuffers.rankBuffers)
		{
			migration.sendFluidCount = migration.sendBuffer.fluidParticles.CheckedBufferCount();
			sentFluidParticleCount += migration.sendBuffer.fluidParticles.Size();
		}
		if (sentFluidParticleCount > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
		{
			return fail("sent migrating fluid particle count exceeds uint32_t range.");
		}
		stats.sentFluidParticleCount = static_cast<uint32_t>(sentFluidParticleCount);

		return true;
	}

} // namespace anonymous

void ParticleMigrationBuffer::Clear()
{
	fluidParticles.Clear();
}

// migrate owned fluid particles with parallel packing, MPI exchange, and rebuild
bool Dist::MigrateFluidParticlesToOwnerRanks(
	Sync::UniversalSimState& localState,
	DistParticleState& fluidParticleState,
	const IDomainDecomp& domainDecomp,
	const MpiContext& mpiContext,
	const uint32_t globalFluidParticleCount,
	MigrationWorkBuffers& workBuffers,
	ParticleMigrationStats* stats,
	std::string* failMsg)
{
	if (failMsg) { failMsg->clear(); }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};

	// prepare migration buffers for every peer rank
	ParticleMigrationStats localStats{};
	std::vector<MigrationWorkBuffers::RankBuffers>& migrations = workBuffers.rankBuffers;
	PrepareParticleMigrationRankBuffers(workBuffers, mpiContext.Rank(), mpiContext.Size());

	{
		// route and pack migration payloads in one owned-particle scan
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
		// exchange rank-indexed particle counts before exchanging raw payloads
		workBuffers.sendFluidCount_FromRank.assign(static_cast<uint64_t>(mpiContext.Size()), 0);
		for (const MigrationWorkBuffers::RankBuffers& migration : migrations)
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
		for (MigrationWorkBuffers::RankBuffers& migration : migrations)
		{
			migration.recvFluidCount = workBuffers.recvFluidCount_FromRank[static_cast<uint64_t>(migration.rank)];
			if (migration.recvFluidCount < 0)
			{
				localCountOk = false;
				localCountFailMsg = "received negative particle migration buffer count.";
				break;
			}

			const uint64_t recvParticleCount = static_cast<uint64_t>(migration.recvFluidCount);
			migration.recvBuffer.fluidParticles.Resize(recvParticleCount);
		}
		if (!mpiContext.AllTrue(localCountOk))
		{
			if (!localCountOk) { return fail(localCountFailMsg); }
			return fail("at least one rank received an invalid particle migration buffer count.");
		}
	}

	{
		std::vector<MPI_Request> requests{};
		requests.reserve(migrations.size() * 36);

		// exchange only non-empty payload chunks after all receive buffers are sized correctly
		for (MigrationWorkBuffers::RankBuffers& migration : migrations) // post non-blocking MPI receives
		{
			if (migration.recvFluidCount <= 0) { continue; }

			migration.recvBuffer.fluidParticles.QueueIRecvPayload(
				mpiContext, migration.rank, PARTICLE_MIGRATION_TAG_FLUID_PAYLOAD, requests);
		}
		for (MigrationWorkBuffers::RankBuffers& migration : migrations) // post non-blocking MPI sends
		{
			if (migration.sendFluidCount <= 0) { continue; }

			migration.sendBuffer.fluidParticles.QueueISendPayload(
				mpiContext, migration.rank, PARTICLE_MIGRATION_TAG_FLUID_PAYLOAD, requests);
		}
		mpiContext.WaitAll(requests);
		requests.clear();
	}

	uint64_t recvFluidParticleCount = 0;
	for (const MigrationWorkBuffers::RankBuffers& migration : migrations)
	{
		const uint64_t recvParticleCount = migration.recvBuffer.fluidParticles.Size();
		if (recvParticleCount > std::numeric_limits<uint64_t>::max() - recvFluidParticleCount)
		{
			return fail("received migrating fluid particle count exceeds uint64_t range.");
		}
		recvFluidParticleCount += recvParticleCount;
	}
	if (recvFluidParticleCount > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
	{
		return fail("received migrating fluid particle count exceeds uint32_t range.");
	}
	localStats.recvFluidParticleCount = static_cast<uint32_t>(recvFluidParticleCount);

	{
		std::string rebuildFailMsg{};
		const bool localRebuildOk = RebuildLocalStateWithMigratedFluidParticles(
			migrations, workBuffers.keptOwnedFluidLocalIdx,
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
	const std::vector<MigrationWorkBuffers::RankBuffers>& migrations,
	const std::vector<int>& keptOwnedFluidLocalIdx,
	const std::vector<int>& migratingOutFluidLocalIdx,
	Sync::UniversalSimState& localState,
	DistParticleState& fluidParticleState,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const uint32_t globalFluidParticleCount,
	MigrationWorkBuffers& workBuffers,
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

	const uint32_t keptParticleCount = static_cast<uint32_t>(keptOwnedFluidLocalIdx.size());

	uint64_t recvFluidParticleCount = 0;
	for (const MigrationWorkBuffers::RankBuffers& migration : migrations)
	{
		const uint64_t recvParticleCount = migration.recvBuffer.fluidParticles.Size();
		if (recvParticleCount >
			std::numeric_limits<uint64_t>::max() - recvFluidParticleCount)
		{
			return fail("received migrating fluid particle count exceeds uint64_t range.");
		}
		recvFluidParticleCount += recvParticleCount;
	}

	if (recvFluidParticleCount >
		static_cast<uint64_t>(std::numeric_limits<uint32_t>::max() - keptParticleCount))
	{
		return fail("migrated fluid particle count exceeds uint32_t local particle count range.");
	}

	const uint32_t receivedParticleCount = static_cast<uint32_t>(recvFluidParticleCount);
	const uint32_t localParticleCount = keptParticleCount + receivedParticleCount;

	const MigrationLocalStatusStamps migrationStatusStamps =
		PrepareMigrationLocalStatusStamps(workBuffers, oldLocalParticleCount);

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
		if (IsMigrationLocalStatusMarked(
			workBuffers, migrationStatusStamps, static_cast<uint32_t>(localIdx)))
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

		SetMigrationLocalStatusKept(
			workBuffers, migrationStatusStamps, static_cast<uint32_t>(localIdx));
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
		if (IsMigrationLocalStatusMarked(
			workBuffers, migrationStatusStamps, static_cast<uint32_t>(localIdx)))
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

		SetMigrationLocalStatusOut(
			workBuffers, migrationStatusStamps, static_cast<uint32_t>(localIdx));
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
		if (IsMigrationLocalStatusKept(
			workBuffers, migrationStatusStamps, dstLocalIdx))
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

		const int movedGlobalIdx = fluidParticleState.particleGlobalIdx_FromLocalIdx[srcLocalIdx];
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
		fluidParticleState.particleLocalIdx_FromGlobalIdx[movedGlobalIdx] = static_cast<int>(dstLocalIdx);
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

	localState.ResizeFluid(localParticleCount);
	fluidParticleState.particleGlobalIdx_FromLocalIdx.resize(localParticleCount);
	fluidParticleState.particleClasses.resize(localParticleCount, DistParticleClass::None);
	fluidParticleState.particleLocalIdx_Owned.clear();
	fluidParticleState.particleLocalIdx_Inner.clear();
	fluidParticleState.particleLocalIdx_Outer.clear();
	fluidParticleState.particleLocalIdx_Halo.clear();
	fluidParticleState.ownedParticleCount = localParticleCount;
	fluidParticleState.haloParticleCount = 0;

	// map received fluid particles from each peer directly into the appended local tail
	uint64_t recvParticleIdx = 0;
	for (const MigrationWorkBuffers::RankBuffers& migration : migrations)
	{
		const MigratingParticleBufferF& recvFluidParticles =
			migration.recvBuffer.fluidParticles;

		for (uint64_t bufferIdx = 0; bufferIdx != recvFluidParticles.Size(); bufferIdx++)
		{
			const int dstLocalIdx =
				static_cast<int>(keptParticleCount + recvParticleIdx);

			if (domainDecomp.OwnerRank(recvFluidParticles.PositionAt(bufferIdx)) != rank)
			{
				return fail("received migrating fluid particle does not belong to this rank.");
			}
			if (!AddParticleMapEntry(
				fluidParticleState, recvFluidParticles.GlobalIdxAt(bufferIdx),
				dstLocalIdx, globalFluidParticleCount))
			{
				return fail("failed to add received migrating fluid particle map entry.");
			}

			recvFluidParticles.CopyToLocalParticle(
				localState.fluidParticles, bufferIdx, dstLocalIdx);
			recvParticleIdx++;
		}
	}

	if (recvParticleIdx != recvFluidParticleCount)
	{
		return fail("rebuilt received fluid particle count does not match expected count.");
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
