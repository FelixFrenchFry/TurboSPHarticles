// particle_redistribution.cpp

// external
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

// internal
#include "source/base/helpers.h"
#include "source/base/parallel_checks.h"
#include "parallel_utils.h"
#include "particle_redistribution.h"
#include "mpi_context.h"



using namespace Dist;

namespace
{
	template <typename T>
	void EnsureVectorCapacityWithExtra(
		std::vector<T>& values,
		const uint64_t N_elements,
		const float extraAllocCapacityPct)
	{
		if (static_cast<uint64_t>(values.capacity()) >= N_elements) { return; }

		const uint64_t capacityTarget = Sim::CheckedCapacityWithSlack(
			N_elements,
			Base::ExtraPercent(N_elements, extraAllocCapacityPct),
			values.max_size(),
			"Redistribution scratch reserve");
		values.reserve(static_cast<size_t>(capacityTarget));
	}

	template <typename T>
	void ResizeVectorWithExtraCapacity(
		std::vector<T>& values,
		const uint64_t N_elements,
		const float extraAllocCapacityPct)
	{
		EnsureVectorCapacityWithExtra(values, N_elements, extraAllocCapacityPct);
		values.resize(N_elements);
	}

	uint32_t ParticleCapacityWithExtra(
		const uint32_t N_particles,
		const float extraAllocCapacityPct)
	{
		const uint64_t capacityTarget = Sim::CheckedCapacityWithSlack(
			N_particles,
			Base::ExtraPercent(N_particles, extraAllocCapacityPct),
			Sim::MaxLocalParticleCount,
			"Redistribution particle reserve");

		return static_cast<uint32_t>(capacityTarget);
	}

	bool CheckedLocalParticleCount(
		const uint64_t value,
		uint32_t& result,
		const char* const valueName,
		std::string* const failMsg)
	{
		if (value > Sim::MaxLocalParticleCount)
		{
			if (failMsg)
			{
				*failMsg = std::string(valueName) + " exceeds the local particle limit.";
			}
			return false;
		}

		result = static_cast<uint32_t>(value);
		return true;
	}

	// only the fields carried by the fluid payload are reserved and sized,
	// (the fields velPred, accPress, sourceTerm, diagElem, pressure, and density are omitted,
	// because the following solver step re-writes them before reading them)
	// TODO: keep it like this or reserve and size them too, if solver step behavior changes?
	void EnsureFluidPayloadCapacityWithExtra(
		Sync::FluidParticles& particles,
		const uint32_t N_particles,
		const float extraAllocCapacityPct)
	{
		if (particles.id.capacity() >= N_particles) { return; }

		const uint32_t N_capacity = ParticleCapacityWithExtra(
			N_particles,
			extraAllocCapacityPct);

		particles.id.reserve(N_capacity);

		particles.xPos.reserve(N_capacity);
		particles.yPos.reserve(N_capacity);
		particles.zPos.reserve(N_capacity);

		particles.xVel.reserve(N_capacity);
		particles.yVel.reserve(N_capacity);
		particles.zVel.reserve(N_capacity);
	}

	void ResizeFluidPayloadFields(
		Sync::FluidParticles& particles,
		const uint32_t N_particles)
	{
		particles.N_ = N_particles;

		particles.id.resize(N_particles);

		particles.xPos.resize(N_particles);
		particles.yPos.resize(N_particles);
		particles.zPos.resize(N_particles);

		particles.xVel.resize(N_particles);
		particles.yVel.resize(N_particles);
		particles.zVel.resize(N_particles);
	}

	void EnsureBoundaryParticleCapacityWithExtra(
		Sync::BoundaryParticles& particles,
		const uint32_t N_particles,
		const float extraAllocCapacityPct)
	{
		if (particles.id.capacity() >= N_particles) { return; }

		particles.reserveAll(ParticleCapacityWithExtra(
			N_particles,
			extraAllocCapacityPct));
	}

	std::span<float> PayloadFieldValues(
		std::vector<float>& fieldValues,
		const uint32_t fieldIdx,
		const uint32_t N_particles)
	{
		return
		{
			fieldValues.data() + static_cast<uint64_t>(fieldIdx) * N_particles,
			N_particles
		};
	}

	std::span<const float> PayloadFieldValues(
		const std::vector<float>& fieldValues,
		const uint32_t fieldIdx,
		const uint32_t N_particles)
	{
		return
		{
			fieldValues.data() + static_cast<uint64_t>(fieldIdx) * N_particles,
			N_particles
		};
	}

	// checks a rebuilt particle ID range for duplicate global IDs
	bool HasDuplicateParticleIdsInRange(
		const std::span<const uint64_t> particleIds,
		const uint32_t firstLocalIdx,
		const uint32_t lastLocalIdx_Excl,
		std::vector<uint64_t>& sortScratch,
		const float extraAllocCapacityPct)
	{
		#if defined(_DEBUG)
		ResizeVectorWithExtraCapacity(
			sortScratch,
			lastLocalIdx_Excl - firstLocalIdx,
			extraAllocCapacityPct);
		std::copy(
			particleIds.begin() + firstLocalIdx,
			particleIds.begin() + lastLocalIdx_Excl,
			sortScratch.begin());

		tbb::parallel_sort(sortScratch.begin(), sortScratch.end());
		return Check::CountAdjacentViolations(
			0,
			static_cast<uint32_t>(sortScratch.size()),
			[&](const uint32_t previousIdx, const uint32_t currentIdx)
			{
				return sortScratch[previousIdx] == sortScratch[currentIdx];
			}) != 0;
		#else
		(void)particleIds;
		(void)firstLocalIdx;
		(void)lastLocalIdx_Excl;
		(void)sortScratch;
		(void)extraAllocCapacityPct;
		return false;
		#endif
	}

	// checks that retained and appended particle IDs land inside the global particle range
	bool ValidatePreparedParticleIds(
		const std::vector<uint32_t>& retainedOldLocalIdxs,
		const std::span<const uint64_t> oldParticleIds,
		const std::span<const uint64_t> appendedParticleIds,
		const uint64_t N_global,
		const char* const particleTypeName,
		std::string* failMsg)
	{
		if (Check::CountViolations(0, static_cast<uint32_t>(retainedOldLocalIdxs.size()),
			[&](const uint32_t localIdx)
			{
				const uint32_t oldLocalIdx = retainedOldLocalIdxs[localIdx];
				return oldLocalIdx >= oldParticleIds.size() || oldParticleIds[oldLocalIdx] >= N_global;
			}) != 0)
		{
			if (failMsg) { *failMsg = std::string("Invalid owned ") + particleTypeName + " particle ID."; }
			return false;
		}

		if (Check::CountViolations(0, static_cast<uint32_t>(appendedParticleIds.size()),
			[&](const uint32_t appendIdx)
			{
				return appendedParticleIds[appendIdx] >= N_global;
			}) != 0)
		{
			if (failMsg)
			{
				*failMsg = std::string("Invalid migrated or halo ") + particleTypeName + " particle ID.";
			}
			return false;
		}

		return true;
	}

	// gather one field of a particle block, reading the source at scattered local
	// indices and writing the destination contiguously
	template <typename DstValues, typename SrcValues>
	void GatherParticleField(
		DstValues& dstValues,
		const SrcValues& srcValues,
		const std::vector<uint32_t>& localIdxs,
		const tbb::blocked_range<uint32_t>& blockRange)
	{
		for (uint32_t bufferIdx = blockRange.begin(); bufferIdx != blockRange.end(); bufferIdx++)
		{
			dstValues[bufferIdx] = srcValues[localIdxs[bufferIdx]];
		}
	}

	// copy one field of a particle block into a contiguous destination range
	template <typename DstValues, typename SrcValues>
	void CopyParticleField(
		DstValues& dstValues,
		const SrcValues& srcValues,
		const uint32_t dstBeginIdx,
		const tbb::blocked_range<uint32_t>& blockRange)
	{
		std::copy(
			srcValues.begin() + blockRange.begin(),
			srcValues.begin() + blockRange.end(),
			dstValues.begin() + dstBeginIdx + blockRange.begin());
	}

	// byte-sized payload helpers for chunked MPI transfers
	template <typename T>
	uint64_t CheckedByteSize(const uint64_t N_elements)
	{
		const uint64_t N_max = std::numeric_limits<uint64_t>::max() / sizeof(T);

		if (N_elements > N_max)
		{
			throw std::runtime_error("Particle redistribution payload byte size exceeds uint64_t range.");
		}

		return static_cast<uint64_t>(N_elements) * sizeof(T);
	}

	template <typename T>
	void QueueIRecvVector(
		const MpiContext& mpiContext,
		std::vector<T>& values,
		const uint64_t N_elements,
		const int peerRank,
		const int tag,
		std::vector<MPI_Request>& requests)
	{
		mpiContext.IRecvBytesChunked(
			(N_elements == 0) ? nullptr : values.data(),
			CheckedByteSize<T>(N_elements),
			peerRank,
			tag,
			requests);
	}

	template <typename T>
	void QueueISendVector(
		const MpiContext& mpiContext,
		const std::vector<T>& values,
		const uint64_t N_elements,
		const int peerRank,
		const int tag,
		std::vector<MPI_Request>& requests)
	{
		mpiContext.ISendBytesChunked(
			(N_elements == 0) ? nullptr : values.data(),
			CheckedByteSize<T>(N_elements),
			peerRank,
			tag,
			requests);
	}

	struct ParticleRouteOutputs
	{
		std::vector<std::vector<uint32_t>>& ownedLocalIdxs_ByDstRank;
		std::vector<std::vector<uint32_t>>& haloLocalIdxs_ByDstRank;
		std::vector<uint32_t>& keptOwnedLocalIdxs;
		std::vector<uint32_t>& localHaloLocalIdxs;
		std::vector<uint8_t>& isOuterOwned_ByLocalIdx;
	};

	template <bool CheckFinitePosition, typename PositionAtLocalIdx>
	bool RouteOwnedParticleIndices_Parallel(
		const uint32_t N_owned,
		const uint32_t N_local,
		const PositionAtLocalIdx& positionAtLocalIdx,
		const ParticleRouteOutputs& outputs,
		ParticleRoutingWorkBuffers& workBuffers,
		const IDomainDecomp& domainDecomp,
		const int rank, const int N_ranks,
		const float extraAllocCapacityPct,
		const char* const particleTypeName,
		std::string* failMsg)
	{
		const uint32_t N_ranks_U32 = static_cast<uint32_t>(N_ranks);
		const uint32_t N_buckets = 2 * N_ranks_U32;
		const uint32_t N_chunks = N_owned > 0
			? 1 + (N_owned - 1) / PARTICLE_ROUTING_CHUNK_SIZE
			: 0;
		const uint64_t N_chunkBuckets = static_cast<uint64_t>(N_chunks) * N_buckets;
		// self-rank buckets are stored in the retained and local-halo routes
		outputs.ownedLocalIdxs_ByDstRank[rank].clear();
		outputs.haloLocalIdxs_ByDstRank[rank].clear();

		Base::ResizeGrowScratchVector(
			workBuffers.routeCounts_ByChunkBucket,
			N_chunkBuckets,
			extraAllocCapacityPct);
		Base::ResizeGrowScratchVector(
			workBuffers.routeWriteOffsets_ByChunkBucket,
			N_chunkBuckets,
			extraAllocCapacityPct);
		Base::ResizeGrowScratchVector(
			workBuffers.routeTotals,
			N_buckets,
			extraAllocCapacityPct);
		Base::ResizeGrowScratchVector(
			workBuffers.ownerRank_ByOwnedIdx,
			N_owned,
			extraAllocCapacityPct);

		// used to skip halo checks for inner particles and retained for later classification
		// indexed by local particle index (only owned entries carry a value)
		auto& isOuterOwned_ByLocalIdx = outputs.isOuterOwned_ByLocalIdx;
		ResizeVectorWithExtraCapacity(
			isOuterOwned_ByLocalIdx,
			N_local,
			extraAllocCapacityPct);

		// filled by the caller
		const std::vector<AABB>& haloExtentAABB_ByRank =
			workBuffers.haloExtentAABB_ByRank;

		// first failure seen by the counting pass, reported once it completes
		std::atomic<const char*> routeFailMsg{ nullptr };

		// decide every destination and count each bucket per chunk
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginOwnedIdx = chunkIdx * PARTICLE_ROUTING_CHUNK_SIZE;
				const uint32_t endOwnedIdx =
					beginOwnedIdx + std::min(
						PARTICLE_ROUTING_CHUNK_SIZE,
						N_owned - beginOwnedIdx);
				uint32_t* const chunkCounts =
					workBuffers.routeCounts_ByChunkBucket.data() +
					static_cast<uint64_t>(chunkIdx) * N_buckets;
				std::fill(chunkCounts, chunkCounts + N_buckets, uint32_t{ 0 });

				for (uint32_t ownedIdx = beginOwnedIdx; ownedIdx != endOwnedIdx; ownedIdx++)
				{
					const uint32_t localIdx = ownedIdx;

					const glm::vec3 position = positionAtLocalIdx(localIdx);
					if constexpr (CheckFinitePosition)
					{
						if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z))
						{
							routeFailMsg.store(" particle position is not finite.", std::memory_order_relaxed);
							continue;
						}
					}
					const int ownerRank = domainDecomp.OwnerRank(position);
					if (ownerRank < 0 || ownerRank >= N_ranks)
					{
						routeFailMsg.store(
							" particle owner rank is out of range.", std::memory_order_relaxed);
						continue;
					}

					workBuffers.ownerRank_ByOwnedIdx[ownedIdx] = ownerRank;
					chunkCounts[static_cast<uint32_t>(ownerRank)]++;

					// testing every neighbor halo directly, so inclusive halo boundaries stay
					// symmetric with the distributed solver exchange plan
					uint8_t isOuterOwned = 0;
					for (const int haloRank : domainDecomp.NeighborRanks(ownerRank))
					{
						#if defined(_DEBUG)
						if (haloRank < 0 || haloRank >= N_ranks || haloRank == ownerRank)
						{
							routeFailMsg.store(
								" halo destination rank is invalid.", std::memory_order_relaxed);
							continue;
						}
						#endif

						if (!haloExtentAABB_ByRank[haloRank].Contains(position)) { continue; }

						isOuterOwned = 1;
						chunkCounts[N_ranks_U32 + static_cast<uint32_t>(haloRank)]++;
					}

					isOuterOwned_ByLocalIdx[localIdx] = isOuterOwned;
				}
			}
		});

		if (const char* const routeFail = routeFailMsg.load(std::memory_order_relaxed))
		{
			if (failMsg) { *failMsg = std::string(particleTypeName) + routeFail; }
			return false;
		}

		PrefixCountsByChunkAndBucket_ParallelScan(
			workBuffers.routeCounts_ByChunkBucket,
			workBuffers.routeWriteOffsets_ByChunkBucket,
			workBuffers.routeTotals,
			N_chunks,
			N_buckets);

		Base::ResizeScratchVector(
			outputs.keptOwnedLocalIdxs,
			workBuffers.routeTotals[static_cast<uint32_t>(rank)],
			extraAllocCapacityPct);
		Base::ResizeScratchVector(
			outputs.localHaloLocalIdxs,
			workBuffers.routeTotals[N_ranks_U32 + static_cast<uint32_t>(rank)],
			extraAllocCapacityPct);

		for (int dstRank = 0; dstRank != N_ranks; dstRank++)
		{
			if (dstRank == rank) { continue; }

			Base::ResizeScratchVector(
				outputs.ownedLocalIdxs_ByDstRank[dstRank],
				workBuffers.routeTotals[static_cast<uint32_t>(dstRank)],
				extraAllocCapacityPct);
			Base::ResizeScratchVector(
				outputs.haloLocalIdxs_ByDstRank[dstRank],
				workBuffers.routeTotals[N_ranks_U32 + static_cast<uint32_t>(dstRank)],
				extraAllocCapacityPct);
		}

		uint32_t* const keptOwnedDst = outputs.keptOwnedLocalIdxs.data();
		uint32_t* const localHaloDst = outputs.localHaloLocalIdxs.data();
		const uint32_t rank_U32 = static_cast<uint32_t>(rank);

		// scatter into the reserved ranges, preserving owned index order within every bucket
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginOwnedIdx = chunkIdx * PARTICLE_ROUTING_CHUNK_SIZE;
				const uint32_t endOwnedIdx =
					beginOwnedIdx + std::min(
						PARTICLE_ROUTING_CHUNK_SIZE,
						N_owned - beginOwnedIdx);

				// each chunk owns its offset row and advances it in place while writing
				uint32_t* const chunkWriteIdxs =
					workBuffers.routeWriteOffsets_ByChunkBucket.data() +
					static_cast<uint64_t>(chunkIdx) * N_buckets;

				for (uint32_t ownedIdx = beginOwnedIdx; ownedIdx != endOwnedIdx; ownedIdx++)
				{
					const int ownerRank = workBuffers.ownerRank_ByOwnedIdx[ownedIdx];
					const uint32_t localIdx = ownedIdx;

					if (ownerRank == rank) [[likely]]
					{
						keptOwnedDst[chunkWriteIdxs[rank_U32]++] = localIdx;
					}
					else
					{
						outputs.ownedLocalIdxs_ByDstRank[ownerRank][
							chunkWriteIdxs[static_cast<uint32_t>(ownerRank)]++] = localIdx;
					}

					if (isOuterOwned_ByLocalIdx[localIdx] == 0) { continue; }

					// halo destinations are re-tested on the outer subset
					const glm::vec3 position = positionAtLocalIdx(localIdx);
					for (const int haloRank : domainDecomp.NeighborRanks(ownerRank))
					{
						if (!haloExtentAABB_ByRank[haloRank].Contains(position)) { continue; }

						if (haloRank == rank)
						{
							localHaloDst[chunkWriteIdxs[N_ranks_U32 + rank_U32]++] = localIdx;
						}
						else
						{
							outputs.haloLocalIdxs_ByDstRank[haloRank][
								chunkWriteIdxs[N_ranks_U32 + static_cast<uint32_t>(haloRank)]++] = localIdx;
						}
					}
				}
			}
		});

		return true;
	}

} // namespace anonymous

uint64_t Dist::AggregateParticleTrafficCounts(
	const std::span<const int> packedCounts,
	const int lane,
	const int laneCount,
	const char* const countName)
{
	if (laneCount <= 0 || lane < 0 || lane >= laneCount ||
		packedCounts.size() % static_cast<uint64_t>(laneCount) != 0)
	{
		throw std::runtime_error("Invalid packed particle traffic count layout.");
	}

	uint64_t total = 0;
	for (uint64_t countIdx = static_cast<uint64_t>(lane);
		countIdx < packedCounts.size();
		countIdx += static_cast<uint64_t>(laneCount))
	{
		const int count = packedCounts[countIdx];
		if (count < 0)
		{
			throw std::runtime_error(std::string(countName) + " contains a negative count.");
		}
		total = Sim::CheckedParticleAdd(
			total, static_cast<uint64_t>(count), countName);
	}
	return total;
}

void ParticleRedistributionMgr::FluidExchangeBuffer::Clear()
{
	N_particles_ = 0;
}

void ParticleRedistributionMgr::FluidExchangeBuffer::Resize(
	const uint32_t N)
{
	// grow reusable storage to the largest requested particle count
	const uint64_t N_fieldValues =
		static_cast<uint64_t>(FLUID_PAYLOAD_FIELD_COUNT) * N;
	Base::ResizeGrowScratchVector(
		particleIds_, N, exchangeBufferExtraAllocCapacityPct_);
	Base::ResizeGrowScratchVector(
		fieldValues_, N_fieldValues, exchangeBufferExtraAllocCapacityPct_);

	N_particles_ = N;
}

void ParticleRedistributionMgr::FluidExchangeBuffer::SetBulkFromParticleView(
	const Sync::FluidConstView& src,
	const std::vector<uint32_t>& localIdxs)
{
	Resize(static_cast<uint32_t>(localIdxs.size()));
	if (localIdxs.empty()) { return; }

	const std::span<uint64_t> ids{ particleIds_.data(), N_particles_ };
	const std::span<float> xPos =
		PayloadFieldValues(fieldValues_, FLUID_PAYLOAD_FIELD_X_POS, N_particles_);
	const std::span<float> yPos =
		PayloadFieldValues(fieldValues_, FLUID_PAYLOAD_FIELD_Y_POS, N_particles_);
	const std::span<float> zPos =
		PayloadFieldValues(fieldValues_, FLUID_PAYLOAD_FIELD_Z_POS, N_particles_);
	const std::span<float> xVel =
		PayloadFieldValues(fieldValues_, FLUID_PAYLOAD_FIELD_X_VEL, N_particles_);
	const std::span<float> yVel =
		PayloadFieldValues(fieldValues_, FLUID_PAYLOAD_FIELD_Y_VEL, N_particles_);
	const std::span<float> zVel =
		PayloadFieldValues(fieldValues_, FLUID_PAYLOAD_FIELD_Z_VEL, N_particles_);

	// gather particle fields sequentially within each parallel block
	// TODO: use explicit SIMD similarly to sph_solver.cpp?
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_particles_, PARTICLE_GATHER_BLOCK_SIZE),
		[&](const tbb::blocked_range<uint32_t>& blockRange)
	{
		GatherParticleField(ids, src.id, localIdxs, blockRange);
		GatherParticleField(xPos, src.xPos, localIdxs, blockRange);
		GatherParticleField(yPos, src.yPos, localIdxs, blockRange);
		GatherParticleField(zPos, src.zPos, localIdxs, blockRange);
		GatherParticleField(xVel, src.xVel, localIdxs, blockRange);
		GatherParticleField(yVel, src.yVel, localIdxs, blockRange);
		GatherParticleField(zVel, src.zVel, localIdxs, blockRange);

		// the fields velPred, accPress, sourceTerm, diagElem, pressure, and density are omitted,
		// because the following solver step re-writes them before reading them
		// TODO: keep it like this or copy them too, if solver step behavior changes?
	});
}

void ParticleRedistributionMgr::FluidExchangeBuffer::CopyBulkToLocalParticles(
	Sync::FluidParticles& dst,
	const uint32_t dstBeginLocalIdx) const
{
	if (IsEmpty()) { return; }

	const std::span<const uint64_t> ids{ particleIds_.data(), N_particles_ };
	const std::span<const float> xPos =
		PayloadFieldValues(fieldValues_, FLUID_PAYLOAD_FIELD_X_POS, N_particles_);
	const std::span<const float> yPos =
		PayloadFieldValues(fieldValues_, FLUID_PAYLOAD_FIELD_Y_POS, N_particles_);
	const std::span<const float> zPos =
		PayloadFieldValues(fieldValues_, FLUID_PAYLOAD_FIELD_Z_POS, N_particles_);
	const std::span<const float> xVel =
		PayloadFieldValues(fieldValues_, FLUID_PAYLOAD_FIELD_X_VEL, N_particles_);
	const std::span<const float> yVel =
		PayloadFieldValues(fieldValues_, FLUID_PAYLOAD_FIELD_Y_VEL, N_particles_);
	const std::span<const float> zVel =
		PayloadFieldValues(fieldValues_, FLUID_PAYLOAD_FIELD_Z_VEL, N_particles_);

	// copy particle fields sequentially within each parallel block
	// TODO: use explicit SIMD similarly to sph_solver.cpp?
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_particles_, PARTICLE_GATHER_BLOCK_SIZE),
		[&](const tbb::blocked_range<uint32_t>& blockRange)
	{
		CopyParticleField(dst.id, ids, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.xPos, xPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.yPos, yPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.zPos, zPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.xVel, xVel, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.yVel, yVel, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.zVel, zVel, dstBeginLocalIdx, blockRange);

		// the fields velPred, accPress, sourceTerm, diagElem, pressure, and density are omitted,
		// because the following solver step re-writes them before reading them
		// TODO: keep it like this or copy them too, if solver step behavior changes?
	});
}

void ParticleRedistributionMgr::FluidExchangeBuffer::QueueIRecvPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests)
{
	if (IsEmpty()) { return; }

	QueueIRecvVector(mpiContext, particleIds_, N_particles_, peerRank, tagBase + 2, requests);
	QueueIRecvVector(
		mpiContext, fieldValues_,
		static_cast<uint64_t>(FLUID_PAYLOAD_FIELD_COUNT) * N_particles_,
		peerRank, tagBase + 3, requests);

	// the fields velPred, accPress, sourceTerm, diagElem, pressure, and density are omitted,
	// because the following solver step re-writes them before reading them
	// TODO: keep it like this or copy them too, if solver step behavior changes?
}

void ParticleRedistributionMgr::FluidExchangeBuffer::QueueISendPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests) const
{
	if (IsEmpty()) { return; }

	QueueISendVector(mpiContext, particleIds_, N_particles_, peerRank, tagBase + 2, requests);
	QueueISendVector(
		mpiContext, fieldValues_,
		static_cast<uint64_t>(FLUID_PAYLOAD_FIELD_COUNT) * N_particles_,
		peerRank, tagBase + 3, requests);

	// the fields velPred, accPress, sourceTerm, diagElem, pressure, and density are omitted,
	// because the following solver step re-writes them before reading them
	// TODO: keep it like this or copy them too, if solver step behavior changes?
}

int ParticleRedistributionMgr::FluidExchangeBuffer::CheckedCount() const
{
	if (Size() > static_cast<uint32_t>(std::numeric_limits<int>::max()))
	{
		throw std::runtime_error("Fluid particle redistribution count exceeds MPI int range.");
	}

	return static_cast<int>(Size());
}

Mem::MemUsage ParticleRedistributionMgr::FluidExchangeBuffer::GetMemoryUsage() const
{
	Mem::MemUsage usage{};

	Mem::Add(usage, Mem::MemUsageOf_Vector(particleIds_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(fieldValues_));

	return usage;
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::Clear()
{
	N_particles_ = 0;
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::Resize(
	const uint32_t N)
{
	// grow reusable storage to the largest requested particle count
	const uint64_t N_fieldValues =
		static_cast<uint64_t>(BOUNDARY_PAYLOAD_FIELD_COUNT) * N;
	Base::ResizeGrowScratchVector(
		particleIds_, N, exchangeBufferExtraAllocCapacityPct_);
	Base::ResizeGrowScratchVector(
		fieldValues_, N_fieldValues, exchangeBufferExtraAllocCapacityPct_);

	N_particles_ = N;
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::SetBulkFromParticleView(
	const Sync::BoundaryConstView& src,
	const std::vector<uint32_t>& localIdxs)
{
	Resize(static_cast<uint32_t>(localIdxs.size()));
	if (localIdxs.empty()) { return; }

	const std::span<uint64_t> ids{ particleIds_.data(), N_particles_ };
	const std::span<float> xPos =
		PayloadFieldValues(fieldValues_, BOUNDARY_PAYLOAD_FIELD_X_POS, N_particles_);
	const std::span<float> yPos =
		PayloadFieldValues(fieldValues_, BOUNDARY_PAYLOAD_FIELD_Y_POS, N_particles_);
	const std::span<float> zPos =
		PayloadFieldValues(fieldValues_, BOUNDARY_PAYLOAD_FIELD_Z_POS, N_particles_);
	const std::span<float> volB =
		PayloadFieldValues(fieldValues_, BOUNDARY_PAYLOAD_FIELD_VOL, N_particles_);
	const std::span<float> psiB =
		PayloadFieldValues(fieldValues_, BOUNDARY_PAYLOAD_FIELD_PSI, N_particles_);

	// gather particle fields sequentially within each parallel block
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_particles_, PARTICLE_GATHER_BLOCK_SIZE),
		[&](const tbb::blocked_range<uint32_t>& blockRange)
	{
		GatherParticleField(ids, src.id, localIdxs, blockRange);
		GatherParticleField(xPos, src.xPos, localIdxs, blockRange);
		GatherParticleField(yPos, src.yPos, localIdxs, blockRange);
		GatherParticleField(zPos, src.zPos, localIdxs, blockRange);
		GatherParticleField(volB, src.volB, localIdxs, blockRange);
		GatherParticleField(psiB, src.psiB, localIdxs, blockRange);
	});
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::CopyBulkToLocalParticles(
	Sync::BoundaryParticles& dst,
	const uint32_t dstBeginLocalIdx) const
{
	if (IsEmpty()) { return; }

	const std::span<const uint64_t> ids{ particleIds_.data(), N_particles_ };
	const std::span<const float> xPos =
		PayloadFieldValues(fieldValues_, BOUNDARY_PAYLOAD_FIELD_X_POS, N_particles_);
	const std::span<const float> yPos =
		PayloadFieldValues(fieldValues_, BOUNDARY_PAYLOAD_FIELD_Y_POS, N_particles_);
	const std::span<const float> zPos =
		PayloadFieldValues(fieldValues_, BOUNDARY_PAYLOAD_FIELD_Z_POS, N_particles_);
	const std::span<const float> volB =
		PayloadFieldValues(fieldValues_, BOUNDARY_PAYLOAD_FIELD_VOL, N_particles_);
	const std::span<const float> psiB =
		PayloadFieldValues(fieldValues_, BOUNDARY_PAYLOAD_FIELD_PSI, N_particles_);

	// copy particle fields sequentially within each parallel block
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_particles_, PARTICLE_GATHER_BLOCK_SIZE),
		[&](const tbb::blocked_range<uint32_t>& blockRange)
	{
		CopyParticleField(dst.id, ids, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.xPos, xPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.yPos, yPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.zPos, zPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.volB, volB, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.psiB, psiB, dstBeginLocalIdx, blockRange);
	});
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::QueueIRecvPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests)
{
	if (IsEmpty()) { return; }

	QueueIRecvVector(mpiContext, particleIds_, N_particles_, peerRank, tagBase + 2, requests);
	QueueIRecvVector(
		mpiContext, fieldValues_,
		static_cast<uint64_t>(BOUNDARY_PAYLOAD_FIELD_COUNT) * N_particles_,
		peerRank, tagBase + 3, requests);
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::QueueISendPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests) const
{
	if (IsEmpty()) { return; }

	QueueISendVector(mpiContext, particleIds_, N_particles_, peerRank, tagBase + 2, requests);
	QueueISendVector(
		mpiContext, fieldValues_,
		static_cast<uint64_t>(BOUNDARY_PAYLOAD_FIELD_COUNT) * N_particles_,
		peerRank, tagBase + 3, requests);
}

int ParticleRedistributionMgr::BoundaryExchangeBuffer::CheckedCount() const
{
	if (Size() > static_cast<uint32_t>(std::numeric_limits<int>::max()))
	{
		throw std::runtime_error("Boundary particle redistribution count exceeds MPI int range.");
	}

	return static_cast<int>(Size());
}

Mem::MemUsage ParticleRedistributionMgr::BoundaryExchangeBuffer::GetMemoryUsage() const
{
	Mem::MemUsage usage{};

	Mem::Add(usage, Mem::MemUsageOf_Vector(particleIds_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(fieldValues_));

	return usage;
}

void ParticleRedistributionMgr::RigidExchangeBuffer::Clear()
{
	N_particles_ = 0;
}

void ParticleRedistributionMgr::RigidExchangeBuffer::Resize(
	const uint32_t N)
{
	// grow reusable storage to the largest requested particle count
	const uint64_t N_fieldValues =
		static_cast<uint64_t>(RIGID_PAYLOAD_FIELD_COUNT) * N;
	Base::ResizeGrowScratchVector(
		particleIds_, N, exchangeBufferExtraAllocCapacityPct_);
	Base::ResizeGrowScratchVector(
		bodyIds_, N, exchangeBufferExtraAllocCapacityPct_);
	Base::ResizeGrowScratchVector(
		fieldValues_, N_fieldValues, exchangeBufferExtraAllocCapacityPct_);

	N_particles_ = N;
}

void ParticleRedistributionMgr::RigidExchangeBuffer::SetBulkFromParticleView(
	const Sync::RigidConstView& src,
	const std::vector<uint32_t>& localIdxs)
{
	Resize(static_cast<uint32_t>(localIdxs.size()));
	if (localIdxs.empty()) { return; }

	const std::span<uint64_t> ids{ particleIds_.data(), N_particles_ };
	const std::span<uint32_t> bodyIds{ bodyIds_.data(), N_particles_ };
	const std::span<float> xRelPos =
		PayloadFieldValues(fieldValues_, RIGID_PAYLOAD_FIELD_X_REL_POS, N_particles_);
	const std::span<float> yRelPos =
		PayloadFieldValues(fieldValues_, RIGID_PAYLOAD_FIELD_Y_REL_POS, N_particles_);
	const std::span<float> zRelPos =
		PayloadFieldValues(fieldValues_, RIGID_PAYLOAD_FIELD_Z_REL_POS, N_particles_);
	const std::span<float> volR =
		PayloadFieldValues(fieldValues_, RIGID_PAYLOAD_FIELD_VOL, N_particles_);
	const std::span<float> psiR =
		PayloadFieldValues(fieldValues_, RIGID_PAYLOAD_FIELD_PSI, N_particles_);

	// gather particle fields sequentially within each parallel block
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_particles_, PARTICLE_GATHER_BLOCK_SIZE),
		[&](const tbb::blocked_range<uint32_t>& blockRange)
	{
		GatherParticleField(ids, src.id, localIdxs, blockRange);
		GatherParticleField(bodyIds, src.bodyId, localIdxs, blockRange);
		GatherParticleField(xRelPos, src.xRelPos, localIdxs, blockRange);
		GatherParticleField(yRelPos, src.yRelPos, localIdxs, blockRange);
		GatherParticleField(zRelPos, src.zRelPos, localIdxs, blockRange);
		GatherParticleField(volR, src.volR, localIdxs, blockRange);
		GatherParticleField(psiR, src.psiR, localIdxs, blockRange);
	});
}

void ParticleRedistributionMgr::RigidExchangeBuffer::CopyBulkToLocalParticles(
	Sync::RigidParticles& dst,
	const uint32_t dstBeginLocalIdx) const
{
	if (IsEmpty()) { return; }

	const std::span<const uint64_t> ids{ particleIds_.data(), N_particles_ };
	const std::span<const uint32_t> bodyIds{ bodyIds_.data(), N_particles_ };
	const std::span<const float> xRelPos =
		PayloadFieldValues(fieldValues_, RIGID_PAYLOAD_FIELD_X_REL_POS, N_particles_);
	const std::span<const float> yRelPos =
		PayloadFieldValues(fieldValues_, RIGID_PAYLOAD_FIELD_Y_REL_POS, N_particles_);
	const std::span<const float> zRelPos =
		PayloadFieldValues(fieldValues_, RIGID_PAYLOAD_FIELD_Z_REL_POS, N_particles_);
	const std::span<const float> volR =
		PayloadFieldValues(fieldValues_, RIGID_PAYLOAD_FIELD_VOL, N_particles_);
	const std::span<const float> psiR =
		PayloadFieldValues(fieldValues_, RIGID_PAYLOAD_FIELD_PSI, N_particles_);

	// copy particle fields sequentially within each parallel block
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_particles_, PARTICLE_GATHER_BLOCK_SIZE),
		[&](const tbb::blocked_range<uint32_t>& blockRange)
	{
		CopyParticleField(dst.id, ids, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.bodyId, bodyIds, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.xRelPos, xRelPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.yRelPos, yRelPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.zRelPos, zRelPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.volR, volR, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.psiR, psiR, dstBeginLocalIdx, blockRange);
	});
}

void ParticleRedistributionMgr::RigidExchangeBuffer::QueueIRecvPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests)
{
	if (IsEmpty()) { return; }

	QueueIRecvVector(mpiContext, particleIds_, N_particles_, peerRank, tagBase + 2, requests);
	QueueIRecvVector(mpiContext, bodyIds_, N_particles_, peerRank, tagBase + 3, requests);
	QueueIRecvVector(
		mpiContext, fieldValues_,
		static_cast<uint64_t>(RIGID_PAYLOAD_FIELD_COUNT) * N_particles_,
		peerRank, tagBase + 4, requests);
}

void ParticleRedistributionMgr::RigidExchangeBuffer::QueueISendPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests) const
{
	if (IsEmpty()) { return; }

	QueueISendVector(mpiContext, particleIds_, N_particles_, peerRank, tagBase + 2, requests);
	QueueISendVector(mpiContext, bodyIds_, N_particles_, peerRank, tagBase + 3, requests);
	QueueISendVector(
		mpiContext, fieldValues_,
		static_cast<uint64_t>(RIGID_PAYLOAD_FIELD_COUNT) * N_particles_,
		peerRank, tagBase + 4, requests);
}

int ParticleRedistributionMgr::RigidExchangeBuffer::CheckedCount() const
{
	if (Size() > static_cast<uint32_t>(std::numeric_limits<int>::max()))
	{
		throw std::runtime_error("Rigid particle redistribution count exceeds MPI int range.");
	}

	return static_cast<int>(Size());
}

Mem::MemUsage ParticleRedistributionMgr::RigidExchangeBuffer::GetMemoryUsage() const
{
	Mem::MemUsage usage{};

	Mem::Add(usage, Mem::MemUsageOf_Vector(particleIds_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(bodyIds_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(fieldValues_));

	return usage;
}

Mem::MemUsage ParticleRedistributionMgr::GetExchangeBufferMemoryUsage() const
{
	Mem::MemUsage usage = Mem::MemUsageOf_Vector(rankBuffers_);

	for (const RankBuffers& rankBuffers : rankBuffers_)
	{
		Mem::Add(usage, rankBuffers.sendOwnedParticlesF.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.recvOwnedParticlesF.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.sendHaloParticlesF.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.recvHaloParticlesF.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.sendOwnedParticlesB.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.recvOwnedParticlesB.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.sendOwnedParticlesR.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.recvOwnedParticlesR.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.sendHaloParticlesB.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.recvHaloParticlesB.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.sendHaloParticlesR.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.recvHaloParticlesR.GetMemoryUsage());
	}

	return usage;
}

Mem::MemUsage ParticleRedistributionMgr::GetScratchMemoryUsage() const
{
	Mem::MemUsage usage{};

	Mem::Add(usage, Mem::MemUsageOf_VectorOfVector(routes_.ownedFluidLocalIdxs_ByDstRank));
	Mem::Add(usage, Mem::MemUsageOf_VectorOfVector(routes_.haloFluidLocalIdxs_ByDstRank));
	Mem::Add(usage, Mem::MemUsageOf_VectorOfVector(routes_.ownedBoundaryLocalIdxs_ByDstRank));
	Mem::Add(usage, Mem::MemUsageOf_VectorOfVector(routes_.haloBoundaryLocalIdxs_ByDstRank));
	Mem::Add(usage, Mem::MemUsageOf_VectorOfVector(routes_.ownedRigidLocalIdxs_ByDstRank));
	Mem::Add(usage, Mem::MemUsageOf_VectorOfVector(routes_.haloRigidLocalIdxs_ByDstRank));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.keptOwnedFluidLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.localHaloFluidLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.keptOwnedBoundaryLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.localHaloBoundaryLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.keptOwnedRigidLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.localHaloRigidLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routingWorkBuffers_.routeCounts_ByChunkBucket));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routingWorkBuffers_.routeWriteOffsets_ByChunkBucket));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routingWorkBuffers_.routeTotals));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routingWorkBuffers_.ownerRank_ByOwnedIdx));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routingWorkBuffers_.haloExtentAABB_ByRank));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.fluidIsOuterOwned_ByLocalIdx));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.boundaryIsOuterOwned_ByLocalIdx));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.rigidIsOuterOwned_ByLocalIdx));
	Mem::Add(usage, Mem::MemUsageOf_Vector(ownedClassificationWorkBuffers_.classCounts_ByChunkClass));
	Mem::Add(usage, Mem::MemUsageOf_Vector(ownedClassificationWorkBuffers_.classWriteOffsets_ByChunkClass));
	Mem::Add(usage, Mem::MemUsageOf_Vector(ownedClassificationWorkBuffers_.classTotals));
	Mem::Add(usage, Mem::MemUsageOf_Vector(sendCounts_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(recvCounts_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(retainedFluidOldLocalIdx_FromNewLocalIdx_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(retainedBoundaryOldLocalIdx_FromNewLocalIdx_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(retainedRigidOldLocalIdx_FromNewLocalIdx_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(migratedParticleIdScratch_));
	Mem::Add(usage, GetParticleMemoryUsage(appendedParticlesF_));
	Mem::Add(usage, GetParticleMemoryUsage(appendedParticlesB_));
	Mem::Add(usage, GetParticleMemoryUsage(appendedParticlesR_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(rebuiltOuterLocalIdxsF_));

	return usage;
}

glm::vec3 ParticleRedistributionMgr::FluidPositionAtLocalIdx(
	const Sync::FluidConstView& particles,
	const uint32_t localIdx)
{
	return
	{
		particles.xPos[localIdx],
		particles.yPos[localIdx],
		particles.zPos[localIdx]
	};
}

glm::vec3 ParticleRedistributionMgr::BoundaryPositionAtLocalIdx(
	const Sync::BoundaryConstView& particles,
	const uint32_t localIdx)
{
	return
	{
		particles.xPos[localIdx],
		particles.yPos[localIdx],
		particles.zPos[localIdx]
	};
}

glm::vec3 ParticleRedistributionMgr::RigidPositionAtLocalIdx(
	const Sync::RigidConstView& particles,
	const Sync::RigidBodies& bodies,
	const uint32_t localIdx)
{
	const uint32_t bodyIdx = particles.bodyId[localIdx];
	const float x = particles.xRelPos[localIdx];
	const float y = particles.yRelPos[localIdx];
	const float z = particles.zRelPos[localIdx];
	return
	{
		bodies.xCM[bodyIdx] + std::fmaf(bodies.a00[bodyIdx], x,
			std::fmaf(bodies.a01[bodyIdx], y, bodies.a02[bodyIdx] * z)),
		bodies.yCM[bodyIdx] + std::fmaf(bodies.a10[bodyIdx], x,
			std::fmaf(bodies.a11[bodyIdx], y, bodies.a12[bodyIdx] * z)),
		bodies.zCM[bodyIdx] + std::fmaf(bodies.a20[bodyIdx], x,
			std::fmaf(bodies.a21[bodyIdx], y, bodies.a22[bodyIdx] * z))
	};
}

void ParticleRedistributionMgr::CopyFluidParticle(
	const Sync::FluidConstView& src,
	Sync::FluidParticles& dst,
	const uint32_t srcLocalIdx,
	const uint32_t dstLocalIdx)
{
	dst.id[dstLocalIdx] = src.id[srcLocalIdx];
	dst.xPos[dstLocalIdx] = src.xPos[srcLocalIdx];
	dst.yPos[dstLocalIdx] = src.yPos[srcLocalIdx];
	dst.zPos[dstLocalIdx] = src.zPos[srcLocalIdx];
	dst.xVel[dstLocalIdx] = src.xVel[srcLocalIdx];
	dst.yVel[dstLocalIdx] = src.yVel[srcLocalIdx];
	dst.zVel[dstLocalIdx] = src.zVel[srcLocalIdx];

	// the fields velPred, accPress, sourceTerm, diagElem, pressure, and density are omitted,
	// because the following solver step re-writes them before reading them
	// TODO: keep it like this or copy them too, if solver step behavior changes?
}

void ParticleRedistributionMgr::CopyBoundaryParticle(
	const Sync::BoundaryConstView& src,
	Sync::BoundaryParticles& dst,
	const uint32_t srcLocalIdx,
	const uint32_t dstLocalIdx)
{
	dst.id[dstLocalIdx] = src.id[srcLocalIdx];
	dst.xPos[dstLocalIdx] = src.xPos[srcLocalIdx];
	dst.yPos[dstLocalIdx] = src.yPos[srcLocalIdx];
	dst.zPos[dstLocalIdx] = src.zPos[srcLocalIdx];
	dst.volB[dstLocalIdx] = src.volB[srcLocalIdx];
	dst.psiB[dstLocalIdx] = src.psiB[srcLocalIdx];
}

void ParticleRedistributionMgr::CopyRigidParticle(
	const Sync::RigidConstView& src,
	Sync::RigidParticles& dst,
	const uint32_t srcLocalIdx,
	const uint32_t dstLocalIdx)
{
	dst.id[dstLocalIdx] = src.id[srcLocalIdx];
	dst.bodyId[dstLocalIdx] = src.bodyId[srcLocalIdx];
	dst.xRelPos[dstLocalIdx] = src.xRelPos[srcLocalIdx];
	dst.yRelPos[dstLocalIdx] = src.yRelPos[srcLocalIdx];
	dst.zRelPos[dstLocalIdx] = src.zRelPos[srcLocalIdx];
	dst.volR[dstLocalIdx] = src.volR[srcLocalIdx];
	dst.psiR[dstLocalIdx] = src.psiR[srcLocalIdx];
}

bool ParticleRedistributionMgr::BuildDenseCompactionMap(
	const uint32_t N_oldLocal,
	const std::vector<uint32_t>& retainedOldLocalIdxs,
	std::vector<uint32_t>& oldLocalIdx_FromNewLocalIdx,
	std::string* failMsg)
{
	if (retainedOldLocalIdxs.size() > N_oldLocal)
	{
		if (failMsg) { *failMsg = "Retained particle count exceeds old local particle count."; }
		return false;
	}
	const uint32_t N_retained = static_cast<uint32_t>(retainedOldLocalIdxs.size());

	// routing preserves owned index order, so retained old local indices ascend
	if (N_retained > 0 && retainedOldLocalIdxs.back() >= N_oldLocal)
	{
		if (failMsg) { *failMsg = "Retained particle local index is invalid or duplicated."; }
		return false;
	}

	#if defined(_DEBUG)
	// exhaustively validate the ascending order this map is derived from
	if (Check::CountAdjacentViolations(
		0,
		N_retained,
		[&](const uint32_t previousIdx, const uint32_t currentIdx)
		{
			return retainedOldLocalIdxs[previousIdx] >= retainedOldLocalIdxs[currentIdx];
		}) != 0)
	{
		if (failMsg) { *failMsg = "Retained particle local index is invalid or duplicated."; }
		return false;
	}
	#endif

	// retained particles above the compacted prefix form the tail of the ascending list
	const uint32_t firstSourceIdx = static_cast<uint32_t>(
		std::lower_bound(
			retainedOldLocalIdxs.begin(),
			retainedOldLocalIdxs.end(),
			N_retained) - retainedOldLocalIdxs.begin());
	const uint32_t N_holes = N_retained - firstSourceIdx;

	ResizeVectorWithExtraCapacity(
		oldLocalIdx_FromNewLocalIdx,
		N_retained,
		scratchBufferExtraAllocCapacityPct_);

	// retained particles keep the slot they already occupy
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_retained),
		[&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t newLocalIdx = range.begin(); newLocalIdx != range.end(); newLocalIdx++)
		{
			oldLocalIdx_FromNewLocalIdx[newLocalIdx] = newLocalIdx;
		}
	});

	// every removed slot below the retained prefix takes a retained particle from above it,
	// where the first hole takes the highest one, so the old tail empties first
	// (the gaps between retained slots are processed in parallel, each holding its own holes)
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, firstSourceIdx + 1),
		[&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t gapIdx = range.begin(); gapIdx != range.end(); gapIdx++)
		{
			const uint32_t beginHoleOldLocalIdx = gapIdx > 0
				? retainedOldLocalIdxs[gapIdx - 1] + 1
				: 0;
			const uint32_t endHoleOldLocalIdx = gapIdx != firstSourceIdx
				? retainedOldLocalIdxs[gapIdx]
				: N_retained;

			// slots below a gap are either retained in place or already filled holes
			uint32_t holeIdx = beginHoleOldLocalIdx - gapIdx;

			for (uint32_t holeOldLocalIdx = beginHoleOldLocalIdx;
				 holeOldLocalIdx != endHoleOldLocalIdx;
				 holeOldLocalIdx++)
			{
				oldLocalIdx_FromNewLocalIdx[holeOldLocalIdx] =
					retainedOldLocalIdxs[firstSourceIdx + (N_holes - 1 - holeIdx)];
				holeIdx++;
			}
		}
	});

	return true;
}

bool ParticleRedistributionMgr::ValidateParticleViews(
	const Sync::FluidConstView& particlesF,
	const Sync::BoundaryConstView& particlesB,
	const DistParticleState& particleStateF,
	const DistParticleState& particleStateB,
	const uint64_t N_globalF,
	const uint64_t N_globalB,
	std::string* failMsg)
{
	const uint32_t N_localF = particlesF.N_;
	const uint32_t N_localB = particlesB.N_;

	// check that all fluid particle fields have the same size
	const auto fluidSizeMatches = [N_localF](const auto& values)
	{
		return values.size() == N_localF;
	};
	if (!fluidSizeMatches(particlesF.id) ||
		!fluidSizeMatches(particlesF.xPos) ||
		!fluidSizeMatches(particlesF.yPos) ||
		!fluidSizeMatches(particlesF.zPos) ||
		!fluidSizeMatches(particlesF.xVel) ||
		!fluidSizeMatches(particlesF.yVel) ||
		!fluidSizeMatches(particlesF.zVel) ||
		!fluidSizeMatches(particlesF.xVelPred) ||
		!fluidSizeMatches(particlesF.yVelPred) ||
		!fluidSizeMatches(particlesF.zVelPred) ||
		!fluidSizeMatches(particlesF.xAccPress) ||
		!fluidSizeMatches(particlesF.yAccPress) ||
		!fluidSizeMatches(particlesF.zAccPress) ||
		!fluidSizeMatches(particlesF.sourceTerm) ||
		!fluidSizeMatches(particlesF.diagElem) ||
		!fluidSizeMatches(particlesF.pressure) ||
		!fluidSizeMatches(particlesF.density))
	{
		if (failMsg) { *failMsg = "Backend fluid particle view has inconsistent field sizes."; }
		return false;
	}

	// check that all boundary particle fields have the same size
	const auto boundarySizeMatches = [N_localB](const auto& values)
	{
		return values.size() == N_localB;
	};
	if (!boundarySizeMatches(particlesB.id) ||
		!boundarySizeMatches(particlesB.xPos) ||
		!boundarySizeMatches(particlesB.yPos) ||
		!boundarySizeMatches(particlesB.zPos) ||
		!boundarySizeMatches(particlesB.volB) ||
		!boundarySizeMatches(particlesB.psiB))
	{
		if (failMsg) { *failMsg = "Backend boundary particle view has inconsistent field sizes."; }
		return false;
	}

	// check that the particle layouts match the views (exhaustive in debug builds)
	if (!particleStateF.IsValidLocalParticleState(
		particlesF.id, N_globalF, failMsg)) { return false; }
	if (!particleStateB.IsValidLocalParticleState(
		particlesB.id, N_globalB, failMsg)) { return false; }

	return true;
}

void ParticleRedistributionMgr::ClearFluidRouteOutputs()
{
	for (auto& route : routes_.ownedFluidLocalIdxs_ByDstRank) { route.clear(); }
	for (auto& route : routes_.haloFluidLocalIdxs_ByDstRank) { route.clear(); }
	routes_.keptOwnedFluidLocalIdxs.clear();
	routes_.localHaloFluidLocalIdxs.clear();
}

void ParticleRedistributionMgr::ClearBoundaryRouteOutputs()
{
	for (auto& route : routes_.ownedBoundaryLocalIdxs_ByDstRank) { route.clear(); }
	for (auto& route : routes_.haloBoundaryLocalIdxs_ByDstRank) { route.clear(); }
	routes_.keptOwnedBoundaryLocalIdxs.clear();
	routes_.localHaloBoundaryLocalIdxs.clear();
}

void ParticleRedistributionMgr::ClearRigidRouteOutputs()
{
	for (auto& route : routes_.ownedRigidLocalIdxs_ByDstRank) { route.clear(); }
	for (auto& route : routes_.haloRigidLocalIdxs_ByDstRank) { route.clear(); }
	routes_.keptOwnedRigidLocalIdxs.clear();
	routes_.localHaloRigidLocalIdxs.clear();
}

void ParticleRedistributionMgr::PrepareBuffers(const int N_ranks)
{
	rankBuffers_.resize(static_cast<uint64_t>(N_ranks));
	routes_.ownedFluidLocalIdxs_ByDstRank.resize(static_cast<uint64_t>(N_ranks));
	routes_.haloFluidLocalIdxs_ByDstRank.resize(static_cast<uint64_t>(N_ranks));
	routes_.ownedBoundaryLocalIdxs_ByDstRank.resize(static_cast<uint64_t>(N_ranks));
	routes_.haloBoundaryLocalIdxs_ByDstRank.resize(static_cast<uint64_t>(N_ranks));
	routes_.ownedRigidLocalIdxs_ByDstRank.resize(static_cast<uint64_t>(N_ranks));
	routes_.haloRigidLocalIdxs_ByDstRank.resize(static_cast<uint64_t>(N_ranks));

	// reset the send and receive buffers between this rank and every other rank
	for (int rank = 0; rank != N_ranks; rank++)
	{
		auto& rankBuffers = rankBuffers_[rank];
		rankBuffers.sendOwnedParticlesF.Clear();
		rankBuffers.recvOwnedParticlesF.Clear();
		rankBuffers.sendHaloParticlesF.Clear();
		rankBuffers.recvHaloParticlesF.Clear();
		rankBuffers.sendOwnedParticlesB.Clear();
		rankBuffers.recvOwnedParticlesB.Clear();
		rankBuffers.sendHaloParticlesB.Clear();
		rankBuffers.recvHaloParticlesB.Clear();
		rankBuffers.sendOwnedParticlesR.Clear();
		rankBuffers.recvOwnedParticlesR.Clear();
		rankBuffers.sendHaloParticlesR.Clear();
		rankBuffers.recvHaloParticlesR.Clear();
		rankBuffers.N_sendOwnedF = 0;
		rankBuffers.N_recvOwnedF = 0;
		rankBuffers.N_sendHaloF = 0;
		rankBuffers.N_recvHaloF = 0;
		rankBuffers.N_sendOwnedB = 0;
		rankBuffers.N_recvOwnedB = 0;
		rankBuffers.N_sendHaloB = 0;
		rankBuffers.N_recvHaloB = 0;
		rankBuffers.N_sendOwnedR = 0;
		rankBuffers.N_recvOwnedR = 0;
		rankBuffers.N_sendHaloR = 0;
		rankBuffers.N_recvHaloR = 0;
	}

	sendCounts_.assign(static_cast<uint64_t>(N_ranks) * PARTICLE_REDISTRIBUTION_LANE_COUNT, 0);
	recvCounts_.clear();
}

bool ParticleRedistributionMgr::RouteOwnedFluidParticles(
	const Sync::FluidConstView& particlesF,
	const DistParticleState& particleStateF,
	const IDomainDecomp& domainDecomp,
	const int rank, const int N_ranks,
	std::string* failMsg)
{
	return RouteOwnedParticleIndices_Parallel<false>(
		particleStateF.N_owned, particlesF.N_,
		[&](const uint32_t localIdx)
		{
			return FluidPositionAtLocalIdx(particlesF, localIdx);
		},
		{
			.ownedLocalIdxs_ByDstRank = routes_.ownedFluidLocalIdxs_ByDstRank,
			.haloLocalIdxs_ByDstRank = routes_.haloFluidLocalIdxs_ByDstRank,
			.keptOwnedLocalIdxs = routes_.keptOwnedFluidLocalIdxs,
			.localHaloLocalIdxs = routes_.localHaloFluidLocalIdxs,
			.isOuterOwned_ByLocalIdx = routes_.fluidIsOuterOwned_ByLocalIdx
		},
		routingWorkBuffers_, domainDecomp, rank, N_ranks,
		scratchBufferExtraAllocCapacityPct_, "Fluid", failMsg);
}

bool ParticleRedistributionMgr::RouteOwnedBoundaryParticles(
	const Sync::BoundaryConstView& particlesB,
	const DistParticleState& particleStateB,
	const IDomainDecomp& domainDecomp,
	const int rank, const int N_ranks,
	std::string* failMsg)
{
	return RouteOwnedParticleIndices_Parallel<false>(
		particleStateB.N_owned, particlesB.N_,
		[&](const uint32_t localIdx)
		{
			return BoundaryPositionAtLocalIdx(particlesB, localIdx);
		},
		{
			.ownedLocalIdxs_ByDstRank = routes_.ownedBoundaryLocalIdxs_ByDstRank,
			.haloLocalIdxs_ByDstRank = routes_.haloBoundaryLocalIdxs_ByDstRank,
			.keptOwnedLocalIdxs = routes_.keptOwnedBoundaryLocalIdxs,
			.localHaloLocalIdxs = routes_.localHaloBoundaryLocalIdxs,
			.isOuterOwned_ByLocalIdx = routes_.boundaryIsOuterOwned_ByLocalIdx
		},
		routingWorkBuffers_, domainDecomp, rank, N_ranks,
		scratchBufferExtraAllocCapacityPct_, "Boundary", failMsg);
}

bool ParticleRedistributionMgr::RouteOwnedRigidParticles(
	const Sync::RigidConstView& particlesR,
	const Sync::RigidBodies& rigidBodies,
	const DistParticleState& particleStateR,
	const IDomainDecomp& domainDecomp,
	const int rank, const int N_ranks,
	std::string* failMsg)
{
	return RouteOwnedParticleIndices_Parallel<true>(
		particleStateR.N_owned, particlesR.N_,
		[&](const uint32_t localIdx)
		{
			return RigidPositionAtLocalIdx(particlesR, rigidBodies, localIdx);
		},
		{
			.ownedLocalIdxs_ByDstRank = routes_.ownedRigidLocalIdxs_ByDstRank,
			.haloLocalIdxs_ByDstRank = routes_.haloRigidLocalIdxs_ByDstRank,
			.keptOwnedLocalIdxs = routes_.keptOwnedRigidLocalIdxs,
			.localHaloLocalIdxs = routes_.localHaloRigidLocalIdxs,
			.isOuterOwned_ByLocalIdx = routes_.rigidIsOuterOwned_ByLocalIdx
		},
		routingWorkBuffers_, domainDecomp, rank, N_ranks,
		scratchBufferExtraAllocCapacityPct_, "Rigid", failMsg);
}

bool ParticleRedistributionMgr::RouteOwnedParticles(
	const Sync::FluidConstView& particlesF,
	const Sync::BoundaryConstView& particlesB,
	const Sync::RigidConstView& particlesR,
	const Sync::RigidBodies& rigidBodies,
	const DistParticleState& particleStateF,
	const DistParticleState& particleStateB,
	const DistParticleState& particleStateR,
	const IDomainDecomp& domainDecomp,
	const bool redistributeParticlesB,
	const int rank, const int N_ranks,
	std::string* failMsg)
{
	// halo bounds stay fixed for both routing passes, which read them per particle and destination
	routingWorkBuffers_.haloExtentAABB_ByRank.resize(static_cast<uint64_t>(N_ranks));
	for (int haloRank = 0; haloRank != N_ranks; haloRank++)
	{
		routingWorkBuffers_.haloExtentAABB_ByRank[haloRank] =
			domainDecomp.HaloExtentAABB(haloRank);
	}
	if (!redistributeParticlesB)
	{
		ClearBoundaryRouteOutputs();
	}

	// route fluid migration and halos
	{
		if (!RouteOwnedFluidParticles(
			particlesF, particleStateF,
			domainDecomp, rank, N_ranks, failMsg))
		{
			return false;
		}
	}

	// route boundary migration and halos
	// (only if needed, since boundary particles are static)
	if (redistributeParticlesB)
	{
		if (!RouteOwnedBoundaryParticles(
			particlesB, particleStateB,
			domainDecomp, rank, N_ranks, failMsg))
		{
			return false;
		}
	}

	return RouteOwnedRigidParticles(
		particlesR, rigidBodies, particleStateR,
		domainDecomp, rank, N_ranks, failMsg);
}

void ParticleRedistributionMgr::PackSendBuffers(
	const Sync::FluidConstView& particlesF,
	const Sync::BoundaryConstView& particlesB,
	const Sync::RigidConstView& particlesR,
	const bool redistributeParticlesB,
	const int rank, const int N_ranks)
{
	// build one compact send buffer per destination rank
	for (int dstRank = 0; dstRank != N_ranks; dstRank++)
	{
		auto& rankBuffers = rankBuffers_[dstRank];
		if (dstRank == rank) { continue; }

		// fluid particles
		const auto& ownedFluidLocalIdxs =
			routes_.ownedFluidLocalIdxs_ByDstRank[dstRank];
		const auto& haloFluidLocalIdxs =
			routes_.haloFluidLocalIdxs_ByDstRank[dstRank];
		rankBuffers.sendOwnedParticlesF.SetBulkFromParticleView(
			particlesF, ownedFluidLocalIdxs);
		rankBuffers.sendHaloParticlesF.SetBulkFromParticleView(
			particlesF, haloFluidLocalIdxs);
		rankBuffers.N_sendOwnedF = rankBuffers.sendOwnedParticlesF.CheckedCount();
		rankBuffers.N_sendHaloF = rankBuffers.sendHaloParticlesF.CheckedCount();

		// boundary particles
		if (redistributeParticlesB) // (only if required)
		{
			const auto& ownedBoundaryLocalIdxs =
				routes_.ownedBoundaryLocalIdxs_ByDstRank[dstRank];
			const auto& haloBoundaryLocalIdxs =
				routes_.haloBoundaryLocalIdxs_ByDstRank[dstRank];
			rankBuffers.sendOwnedParticlesB.SetBulkFromParticleView(
				particlesB, ownedBoundaryLocalIdxs);
			rankBuffers.sendHaloParticlesB.SetBulkFromParticleView(
				particlesB, haloBoundaryLocalIdxs);
			rankBuffers.N_sendOwnedB = rankBuffers.sendOwnedParticlesB.CheckedCount();
			rankBuffers.N_sendHaloB = rankBuffers.sendHaloParticlesB.CheckedCount();
		}

		// rigid particles
		const auto& ownedRigidLocalIdxs =
			routes_.ownedRigidLocalIdxs_ByDstRank[dstRank];
		const auto& haloRigidLocalIdxs =
			routes_.haloRigidLocalIdxs_ByDstRank[dstRank];
		rankBuffers.sendOwnedParticlesR.SetBulkFromParticleView(
			particlesR, ownedRigidLocalIdxs);
		rankBuffers.sendHaloParticlesR.SetBulkFromParticleView(
			particlesR, haloRigidLocalIdxs);
		rankBuffers.N_sendOwnedR = rankBuffers.sendOwnedParticlesR.CheckedCount();
		rankBuffers.N_sendHaloR = rankBuffers.sendHaloParticlesR.CheckedCount();

		const uint64_t countBase = static_cast<uint64_t>(dstRank) * PARTICLE_REDISTRIBUTION_LANE_COUNT;
		sendCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_OWNED_FLUID] =
			rankBuffers.N_sendOwnedF;
		sendCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_HALO_FLUID] =
			rankBuffers.N_sendHaloF;
		sendCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_OWNED_BOUNDARY] =
			rankBuffers.N_sendOwnedB;
		sendCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_HALO_BOUNDARY] =
			rankBuffers.N_sendHaloB;
		sendCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_OWNED_RIGID] =
			rankBuffers.N_sendOwnedR;
		sendCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_HALO_RIGID] =
			rankBuffers.N_sendHaloR;

	}

	// store aggregate traffic counts for diagnostics
	redistStats_.N_sentOwnedF = Sim::CheckedLocalParticleCount(
		AggregateParticleTrafficCounts(
			sendCounts_, PARTICLE_REDISTRIBUTION_LANE_OWNED_FLUID,
			PARTICLE_REDISTRIBUTION_LANE_COUNT, "Sent owned fluid particle count"),
		"Sent owned fluid particle count");
	redistStats_.N_sentHaloF = AggregateParticleTrafficCounts(
		sendCounts_, PARTICLE_REDISTRIBUTION_LANE_HALO_FLUID,
		PARTICLE_REDISTRIBUTION_LANE_COUNT, "Sent halo fluid particle count");
	redistStats_.N_sentOwnedB = Sim::CheckedLocalParticleCount(
		AggregateParticleTrafficCounts(
			sendCounts_, PARTICLE_REDISTRIBUTION_LANE_OWNED_BOUNDARY,
			PARTICLE_REDISTRIBUTION_LANE_COUNT, "Sent owned boundary particle count"),
		"Sent owned boundary particle count");
	redistStats_.N_sentHaloB = AggregateParticleTrafficCounts(
		sendCounts_, PARTICLE_REDISTRIBUTION_LANE_HALO_BOUNDARY,
		PARTICLE_REDISTRIBUTION_LANE_COUNT, "Sent halo boundary particle count");
	redistStats_.N_sentOwnedR = Sim::CheckedLocalParticleCount(
		AggregateParticleTrafficCounts(
			sendCounts_, PARTICLE_REDISTRIBUTION_LANE_OWNED_RIGID,
			PARTICLE_REDISTRIBUTION_LANE_COUNT, "Sent owned rigid particle count"),
		"Sent owned rigid particle count");
	redistStats_.N_sentHaloR = AggregateParticleTrafficCounts(
		sendCounts_, PARTICLE_REDISTRIBUTION_LANE_HALO_RIGID,
		PARTICLE_REDISTRIBUTION_LANE_COUNT, "Sent halo rigid particle count");
}

bool ParticleRedistributionMgr::PrepareReceiveBuffers(
	const int rank, const int N_ranks,
	std::string* failMsg)
{
	// validate the final retained + received + halo layout before any receive payload allocation
	const uint64_t N_recvOwnedF = AggregateParticleTrafficCounts(
		recvCounts_, PARTICLE_REDISTRIBUTION_LANE_OWNED_FLUID,
		PARTICLE_REDISTRIBUTION_LANE_COUNT, "Received owned fluid particle count");
	const uint64_t N_recvHaloF = AggregateParticleTrafficCounts(
		recvCounts_, PARTICLE_REDISTRIBUTION_LANE_HALO_FLUID,
		PARTICLE_REDISTRIBUTION_LANE_COUNT, "Received halo fluid particle count");
	const uint64_t N_recvOwnedB = AggregateParticleTrafficCounts(
		recvCounts_, PARTICLE_REDISTRIBUTION_LANE_OWNED_BOUNDARY,
		PARTICLE_REDISTRIBUTION_LANE_COUNT, "Received owned boundary particle count");
	const uint64_t N_recvHaloB = AggregateParticleTrafficCounts(
		recvCounts_, PARTICLE_REDISTRIBUTION_LANE_HALO_BOUNDARY,
		PARTICLE_REDISTRIBUTION_LANE_COUNT, "Received halo boundary particle count");

	const uint64_t N_recvOwnedR = AggregateParticleTrafficCounts(
		recvCounts_, PARTICLE_REDISTRIBUTION_LANE_OWNED_RIGID,
		PARTICLE_REDISTRIBUTION_LANE_COUNT, "Received owned rigid particle count");
	const uint64_t N_recvHaloR = AggregateParticleTrafficCounts(
		recvCounts_, PARTICLE_REDISTRIBUTION_LANE_HALO_RIGID,
		PARTICLE_REDISTRIBUTION_LANE_COUNT, "Received halo rigid particle count");

	const uint64_t N_ownedF = Sim::CheckedParticleAdd(
		routes_.keptOwnedFluidLocalIdxs.size(),
		N_recvOwnedF,
		"Owned fluid particle count");
	const uint64_t N_haloF = Sim::CheckedParticleAdd(
		routes_.localHaloFluidLocalIdxs.size(),
		N_recvHaloF,
		"Halo fluid particle count");
	(void)Sim::CheckedCombinedLocalParticleCount(
		N_ownedF, N_haloF, "Local fluid particle count");

	const uint64_t N_ownedB = Sim::CheckedParticleAdd(
		routes_.keptOwnedBoundaryLocalIdxs.size(),
		N_recvOwnedB,
		"Owned boundary particle count");
	const uint64_t N_haloB = Sim::CheckedParticleAdd(
		routes_.localHaloBoundaryLocalIdxs.size(),
		N_recvHaloB,
		"Halo boundary particle count");
	(void)Sim::CheckedCombinedLocalParticleCount(
		N_ownedB, N_haloB, "Local boundary particle count");

	const uint64_t N_ownedR = Sim::CheckedParticleAdd(
		routes_.keptOwnedRigidLocalIdxs.size(),
		N_recvOwnedR,
		"Owned rigid particle count");
	const uint64_t N_haloR = Sim::CheckedParticleAdd(
		routes_.localHaloRigidLocalIdxs.size(),
		N_recvHaloR,
		"Halo rigid particle count");
	(void)Sim::CheckedCombinedLocalParticleCount(
		N_ownedR, N_haloR, "Local rigid particle count");

	// receive totals fit the validated local populations
	redistStats_.N_recvOwnedF = static_cast<uint32_t>(N_recvOwnedF);
	redistStats_.N_recvHaloF = N_recvHaloF;
	redistStats_.N_recvOwnedB = static_cast<uint32_t>(N_recvOwnedB);
	redistStats_.N_recvHaloB = N_recvHaloB;
	redistStats_.N_recvOwnedR = static_cast<uint32_t>(N_recvOwnedR);
	redistStats_.N_recvHaloR = N_recvHaloR;

	// configure one compact receive buffer per source rank
	for (int srcRank = 0; srcRank != N_ranks; srcRank++)
	{
		auto& rankBuffers = rankBuffers_[srcRank];
		if (srcRank == rank) { continue; }

		const uint64_t countBase = static_cast<uint64_t>(srcRank) * PARTICLE_REDISTRIBUTION_LANE_COUNT;
		rankBuffers.N_recvOwnedF = recvCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_OWNED_FLUID];
		rankBuffers.N_recvHaloF = recvCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_HALO_FLUID];
		rankBuffers.N_recvOwnedB = recvCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_OWNED_BOUNDARY];
		rankBuffers.N_recvHaloB = recvCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_HALO_BOUNDARY];
		rankBuffers.N_recvOwnedR = recvCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_OWNED_RIGID];
		rankBuffers.N_recvHaloR = recvCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_HALO_RIGID];

		if (rankBuffers.N_recvOwnedF < 0 ||
			rankBuffers.N_recvHaloF < 0 ||
			rankBuffers.N_recvOwnedB < 0 ||
			rankBuffers.N_recvHaloB < 0 ||
			rankBuffers.N_recvOwnedR < 0 ||
			rankBuffers.N_recvHaloR < 0)
		{
			if (failMsg) { *failMsg = "Received particle count is negative."; }
			return false;
		}

		rankBuffers.recvOwnedParticlesF.Resize(
			static_cast<uint32_t>(rankBuffers.N_recvOwnedF));
		rankBuffers.recvHaloParticlesF.Resize(
			static_cast<uint32_t>(rankBuffers.N_recvHaloF));
		rankBuffers.recvOwnedParticlesB.Resize(
			static_cast<uint32_t>(rankBuffers.N_recvOwnedB));
		rankBuffers.recvHaloParticlesB.Resize(
			static_cast<uint32_t>(rankBuffers.N_recvHaloB));
		rankBuffers.recvOwnedParticlesR.Resize(
			static_cast<uint32_t>(rankBuffers.N_recvOwnedR));
		rankBuffers.recvHaloParticlesR.Resize(
			static_cast<uint32_t>(rankBuffers.N_recvHaloR));

	}

	return true;
}

void ParticleRedistributionMgr::ExchangePayloads(
	const int rank, const int N_ranks,
	const MpiContext& mpiContext)
{
	std::vector<MPI_Request> requests;

	// post receives before matching sends
	for (int srcRank = 0; srcRank != N_ranks; srcRank++)
	{
		if (srcRank == rank) { continue; }

		auto& rankBuffers = rankBuffers_[srcRank];
		rankBuffers.recvOwnedParticlesF.QueueIRecvPayload(
			mpiContext, srcRank, PARTICLE_REDISTRIBUTION_TAG_OWNED_FLUID,
			requests);
		rankBuffers.recvHaloParticlesF.QueueIRecvPayload(
			mpiContext, srcRank, PARTICLE_REDISTRIBUTION_TAG_HALO_FLUID,
			requests);
		rankBuffers.recvOwnedParticlesB.QueueIRecvPayload(
			mpiContext, srcRank, PARTICLE_REDISTRIBUTION_TAG_OWNED_BOUNDARY,
			requests);
		rankBuffers.recvHaloParticlesB.QueueIRecvPayload(
			mpiContext, srcRank, PARTICLE_REDISTRIBUTION_TAG_HALO_BOUNDARY,
			requests);
		rankBuffers.recvOwnedParticlesR.QueueIRecvPayload(
			mpiContext, srcRank, PARTICLE_REDISTRIBUTION_TAG_OWNED_RIGID,
			requests);
		rankBuffers.recvHaloParticlesR.QueueIRecvPayload(
			mpiContext, srcRank, PARTICLE_REDISTRIBUTION_TAG_HALO_RIGID,
			requests);
	}

	// post sends for all six payload lanes
	for (int dstRank = 0; dstRank != N_ranks; dstRank++)
	{
		if (dstRank == rank) { continue; }

		const auto& rankBuffers = rankBuffers_[dstRank];
		rankBuffers.sendOwnedParticlesF.QueueISendPayload(
			mpiContext, dstRank, PARTICLE_REDISTRIBUTION_TAG_OWNED_FLUID,
			requests);
		rankBuffers.sendHaloParticlesF.QueueISendPayload(
			mpiContext, dstRank, PARTICLE_REDISTRIBUTION_TAG_HALO_FLUID,
			requests);
		rankBuffers.sendOwnedParticlesB.QueueISendPayload(
			mpiContext, dstRank, PARTICLE_REDISTRIBUTION_TAG_OWNED_BOUNDARY,
			requests);
		rankBuffers.sendHaloParticlesB.QueueISendPayload(
			mpiContext, dstRank, PARTICLE_REDISTRIBUTION_TAG_HALO_BOUNDARY,
			requests);
		rankBuffers.sendOwnedParticlesR.QueueISendPayload(
			mpiContext, dstRank, PARTICLE_REDISTRIBUTION_TAG_OWNED_RIGID,
			requests);
		rankBuffers.sendHaloParticlesR.QueueISendPayload(
			mpiContext, dstRank, PARTICLE_REDISTRIBUTION_TAG_HALO_RIGID,
			requests);
	}

	mpiContext.WaitAll(requests, "particle redistribution payloads");
}

bool ParticleRedistributionMgr::PrepareFluidParticleMutation(
	const Sync::FluidConstView& oldParticles,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float haloRadius,
	const uint64_t N_global,
	std::string* failMsg)
{
	auto& appendedParticles = appendedParticlesF_;
	auto& rebuiltParticleState = rebuiltParticleStateF_;
	uint32_t N_retained = 0;
	uint32_t N_appended = 0;
	uint32_t N_owned = 0;
	uint32_t N_halo = 0;
	uint32_t N_local = 0;

	// compute sizes and initialize append-only fluid storage and metadata
	{
		const uint64_t N_retained_U64 =
			routes_.keptOwnedFluidLocalIdxs.size();
		const uint64_t N_owned_U64 = Sim::CheckedParticleAdd(
			routes_.keptOwnedFluidLocalIdxs.size(),
			redistStats_.N_recvOwnedF,
			"Owned fluid particle count");
		const uint64_t N_halo_U64 = Sim::CheckedParticleAdd(
			routes_.localHaloFluidLocalIdxs.size(),
			redistStats_.N_recvHaloF,
			"Halo fluid particle count");
		const uint64_t N_appended_U64 = Sim::CheckedParticleAdd(
			redistStats_.N_recvOwnedF,
			N_halo_U64,
			"Appended fluid particle count");
		const uint64_t N_local_U64 = Sim::CheckedParticleAdd(
			N_owned_U64, N_halo_U64, "Local fluid particle count");

		if (!CheckedLocalParticleCount(
				N_retained_U64, N_retained,
				"Retained fluid particle count", failMsg) ||
			!CheckedLocalParticleCount(
				N_appended_U64, N_appended,
				"Appended fluid particle count", failMsg) ||
			!CheckedLocalParticleCount(
				N_owned_U64, N_owned,
				"Owned fluid particle count", failMsg) ||
			!CheckedLocalParticleCount(
				N_halo_U64, N_halo,
				"Halo fluid particle count", failMsg) ||
			!CheckedLocalParticleCount(
				N_local_U64,
				N_local,
				"Local fluid particle count", failMsg) ||
			!BuildDenseCompactionMap(
				oldParticles.N_,
				routes_.keptOwnedFluidLocalIdxs,
				retainedFluidOldLocalIdx_FromNewLocalIdx_,
				failMsg))
		{
			return false;
		}

		EnsureFluidPayloadCapacityWithExtra(
			appendedParticles,
			N_appended,
			scratchBufferExtraAllocCapacityPct_);
		ResizeFluidPayloadFields(appendedParticles, N_appended);
		rebuiltParticleState = { .N_owned = N_owned, .N_halo = N_halo };
	}

	uint32_t appendLocalIdx = 0;

	// append owned particles migrated from other ranks
	{
		for (const auto& rankBuffers : rankBuffers_)
		{
			const auto& recvParticles = rankBuffers.recvOwnedParticlesF;
			recvParticles.CopyBulkToLocalParticles(appendedParticles, appendLocalIdx);
			appendLocalIdx += recvParticles.Size();
		}

		// check that no two ranks migrated the same owned fluid particle into this rank
		if (HasDuplicateParticleIdsInRange(
			appendedParticles.id, 0, appendLocalIdx,
			migratedParticleIdScratch_,
			scratchBufferExtraAllocCapacityPct_))
		{
			if (failMsg) { *failMsg = "Duplicate migrated fluid particle ID."; }
			return false;
		}
	}

	// append local and received fluid halos
	{
		const uint32_t N_localHalo =
			static_cast<uint32_t>(routes_.localHaloFluidLocalIdxs.size());
		const uint32_t haloAppendLocalIdxBase = appendLocalIdx;

		// append halos created locally from particles migrating away
		// TODO: use explicit SIMD for particle field gathers similarly to sph_solver.cpp?
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_localHalo),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t haloIdx = range.begin(); haloIdx != range.end(); haloIdx++)
			{
				CopyFluidParticle(
					oldParticles, appendedParticles,
					routes_.localHaloFluidLocalIdxs[haloIdx],
					haloAppendLocalIdxBase + haloIdx);
			}
		});
		appendLocalIdx += N_localHalo;

		// append halo particles received from other ranks
		for (const auto& rankBuffers : rankBuffers_)
		{
			const auto& recvParticles = rankBuffers.recvHaloParticlesF;
			recvParticles.CopyBulkToLocalParticles(appendedParticles, appendLocalIdx);
			appendLocalIdx += recvParticles.Size();
		}
	}

	// validate rebuilt fluid counts and particle IDs
	if (appendLocalIdx != N_appended ||
		retainedFluidOldLocalIdx_FromNewLocalIdx_.size() != N_retained)
	{
		if (failMsg) { *failMsg = "Prepared fluid particle mutation count is inconsistent."; }
		return false;
	}
	if (!ValidatePreparedParticleIds(
		retainedFluidOldLocalIdx_FromNewLocalIdx_,
		oldParticles.id,
		appendedParticles.id,
		N_global,
		"fluid",
		failMsg))
	{
		return false;
	}

	// classify the rebuilt owned prefix, where routing already flagged retained outer particles
	rebuiltParticleState.N_outer = CollectPreparedOuterFluidLocalIdxs(
		N_owned,
		retainedFluidOldLocalIdx_FromNewLocalIdx_,
		routes_.fluidIsOuterOwned_ByLocalIdx,
		appendedParticles.constView(),
		domainDecomp,
		rank,
		haloRadius,
		ownedClassificationWorkBuffers_,
		rebuiltOuterLocalIdxsF_,
		scratchBufferExtraAllocCapacityPct_);

	return rebuiltParticleState.IsValidLocalParticleLayout(N_local, failMsg);
}

// prepare retained compaction, appended particles, and matching boundary metadata
bool ParticleRedistributionMgr::PrepareBoundaryParticleMutation(
	const Sync::BoundaryConstView& oldParticles,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float haloRadius,
	const uint64_t N_global,
	std::string* failMsg)
{
	auto& appendedParticles = appendedParticlesB_;
	auto& rebuiltParticleState = rebuiltParticleStateB_;
	uint32_t N_retained = 0;
	uint32_t N_appended = 0;
	uint32_t N_owned = 0;
	uint32_t N_halo = 0;
	uint32_t N_local = 0;

	// compute sizes and initialize append-only boundary storage and metadata
	{
		const uint64_t N_retained_U64 =
			routes_.keptOwnedBoundaryLocalIdxs.size();
		const uint64_t N_owned_U64 = Sim::CheckedParticleAdd(
			routes_.keptOwnedBoundaryLocalIdxs.size(),
			redistStats_.N_recvOwnedB,
			"Owned boundary particle count");
		const uint64_t N_halo_U64 = Sim::CheckedParticleAdd(
			routes_.localHaloBoundaryLocalIdxs.size(),
			redistStats_.N_recvHaloB,
			"Halo boundary particle count");
		const uint64_t N_appended_U64 = Sim::CheckedParticleAdd(
			redistStats_.N_recvOwnedB,
			N_halo_U64,
			"Appended boundary particle count");
		const uint64_t N_local_U64 = Sim::CheckedParticleAdd(
			N_owned_U64, N_halo_U64, "Local boundary particle count");

		if (!CheckedLocalParticleCount(
				N_retained_U64, N_retained,
				"Retained boundary particle count", failMsg) ||
			!CheckedLocalParticleCount(
				N_appended_U64, N_appended,
				"Appended boundary particle count", failMsg) ||
			!CheckedLocalParticleCount(
				N_owned_U64, N_owned,
				"Owned boundary particle count", failMsg) ||
			!CheckedLocalParticleCount(
				N_halo_U64, N_halo,
				"Halo boundary particle count", failMsg) ||
			!CheckedLocalParticleCount(
				N_local_U64,
				N_local,
				"Local boundary particle count", failMsg) ||
			!BuildDenseCompactionMap(
				oldParticles.N_,
				routes_.keptOwnedBoundaryLocalIdxs,
				retainedBoundaryOldLocalIdx_FromNewLocalIdx_,
				failMsg))
		{
			return false;
		}

		EnsureBoundaryParticleCapacityWithExtra(
			appendedParticles,
			N_appended,
			scratchBufferExtraAllocCapacityPct_);
		appendedParticles.resizeAll(N_appended);
		rebuiltParticleState = { .N_owned = N_owned, .N_halo = N_halo };
	}

	uint32_t appendLocalIdx = 0;

	// append owned particles migrated from other ranks
	{
		for (const auto& rankBuffers : rankBuffers_)
		{
			const auto& recvParticles = rankBuffers.recvOwnedParticlesB;
			recvParticles.CopyBulkToLocalParticles(appendedParticles, appendLocalIdx);
			appendLocalIdx += recvParticles.Size();
		}

		// check that no two ranks migrated the same owned boundary particle into this rank
		if (HasDuplicateParticleIdsInRange(
			appendedParticles.id, 0, appendLocalIdx,
			migratedParticleIdScratch_,
			scratchBufferExtraAllocCapacityPct_))
		{
			if (failMsg) { *failMsg = "Duplicate migrated boundary particle ID."; }
			return false;
		}
	}

	// append local and received boundary halos
	{
		const uint32_t N_localHalo =
			static_cast<uint32_t>(routes_.localHaloBoundaryLocalIdxs.size());
		const uint32_t haloAppendLocalIdxBase = appendLocalIdx;

		// append halos created locally from particles migrating away
		// TODO: use explicit SIMD for particle field gathers similarly to sph_solver.cpp?
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_localHalo),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t haloIdx = range.begin(); haloIdx != range.end(); haloIdx++)
			{
				CopyBoundaryParticle(
					oldParticles, appendedParticles,
					routes_.localHaloBoundaryLocalIdxs[haloIdx],
					haloAppendLocalIdxBase + haloIdx);
			}
		});
		appendLocalIdx += N_localHalo;

		// append halo particles received from other ranks
		for (const auto& rankBuffers : rankBuffers_)
		{
			const auto& recvParticles = rankBuffers.recvHaloParticlesB;
			recvParticles.CopyBulkToLocalParticles(appendedParticles, appendLocalIdx);
			appendLocalIdx += recvParticles.Size();
		}
	}

	// validate rebuilt boundary counts and particle IDs
	if (appendLocalIdx != N_appended ||
		retainedBoundaryOldLocalIdx_FromNewLocalIdx_.size() != N_retained)
	{
		if (failMsg) { *failMsg = "Prepared boundary particle mutation count is inconsistent."; }
		return false;
	}
	if (!ValidatePreparedParticleIds(
		retainedBoundaryOldLocalIdx_FromNewLocalIdx_,
		oldParticles.id,
		appendedParticles.id,
		N_global,
		"boundary",
		failMsg))
	{
		return false;
	}

	// classify the rebuilt owned prefix, where routing already flagged retained outer particles
	rebuiltParticleState.N_outer = CountPreparedOuterBoundaryParticles(
		N_owned,
		retainedBoundaryOldLocalIdx_FromNewLocalIdx_,
		routes_.boundaryIsOuterOwned_ByLocalIdx,
		appendedParticles.constView(),
		domainDecomp,
		rank,
		haloRadius);

	return rebuiltParticleState.IsValidLocalParticleLayout(N_local, failMsg);
}

// prepare retained compaction, appended particles, and matching rigid metadata
bool ParticleRedistributionMgr::PrepareRigidParticleMutation(
	const Sync::RigidConstView& oldParticles,
	const Sync::RigidBodies& rigidBodies,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float haloRadius,
	const uint64_t N_global,
	std::string* failMsg)
{
	auto& appendedParticles = appendedParticlesR_;
	auto& rebuiltParticleState = rebuiltParticleStateR_;
	const uint32_t N_retained = static_cast<uint32_t>(routes_.keptOwnedRigidLocalIdxs.size());
	uint64_t N_owned_U64 = N_retained;
	uint64_t N_halo_U64 = routes_.localHaloRigidLocalIdxs.size();
	for (const RankBuffers& rankBuffers : rankBuffers_)
	{
		N_owned_U64 = Sim::CheckedParticleAdd(
			N_owned_U64, rankBuffers.recvOwnedParticlesR.Size(), "Owned rigid particle count");
		N_halo_U64 = Sim::CheckedParticleAdd(
			N_halo_U64, rankBuffers.recvHaloParticlesR.Size(), "Halo rigid particle count");
	}
	const uint32_t N_local = Sim::CheckedCombinedLocalParticleCount(
		N_owned_U64, N_halo_U64, "Local rigid particle count");
	const uint32_t N_owned = static_cast<uint32_t>(N_owned_U64);
	const uint32_t N_halo = static_cast<uint32_t>(N_halo_U64);
	const uint32_t N_appended = N_local - N_retained;
	if (!BuildDenseCompactionMap(
		oldParticles.N_, routes_.keptOwnedRigidLocalIdxs,
		retainedRigidOldLocalIdx_FromNewLocalIdx_, failMsg))
	{
		return false;
	}
	if (appendedParticles.id.capacity() < N_appended)
	{
		appendedParticles.reserveAll(
			ParticleCapacityWithExtra(N_appended, scratchBufferExtraAllocCapacityPct_));
	}
	appendedParticles.resizeAll(N_appended);
	rebuiltParticleState = { .N_owned = N_owned, .N_halo = N_halo };

	// migrated owners precede all halos in the append-only block
	uint32_t appendLocalIdx = 0;
	for (const RankBuffers& rankBuffers : rankBuffers_)
	{
		rankBuffers.recvOwnedParticlesR.CopyBulkToLocalParticles(appendedParticles, appendLocalIdx);
		appendLocalIdx += rankBuffers.recvOwnedParticlesR.Size();
	}

	// save local halos before in-place compaction can overwrite their old source slots
	const auto& localHaloLocalIdxs = routes_.localHaloRigidLocalIdxs;
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, static_cast<uint32_t>(localHaloLocalIdxs.size())),
		[&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i != range.end(); i++)
		{
			CopyRigidParticle(oldParticles, appendedParticles, localHaloLocalIdxs[i], appendLocalIdx + i);
		}
	});
	appendLocalIdx += static_cast<uint32_t>(localHaloLocalIdxs.size());
	for (const RankBuffers& rankBuffers : rankBuffers_)
	{
		rankBuffers.recvHaloParticlesR.CopyBulkToLocalParticles(appendedParticles, appendLocalIdx);
		appendLocalIdx += rankBuffers.recvHaloParticlesR.Size();
	}
	if (appendLocalIdx != N_appended ||
		retainedRigidOldLocalIdx_FromNewLocalIdx_.size() != N_retained)
	{
		if (failMsg) { *failMsg = "Prepared rigid particle mutation count is inconsistent."; }
		return false;
	}

	// validate the final layout through the compaction map and appended block
	const Sync::RigidConstView appendedView = appendedParticles.constView();
	#if defined(_DEBUG)
	ResizeVectorWithExtraCapacity(migratedParticleIdScratch_, N_local, scratchBufferExtraAllocCapacityPct_);
	#endif
	for (uint32_t localIdx = 0; localIdx != N_local; localIdx++)
	{
		const bool retained = localIdx < N_retained;
		const auto& src = retained ? oldParticles : appendedView;
		const uint32_t srcLocalIdx = retained
			? retainedRigidOldLocalIdx_FromNewLocalIdx_[localIdx] : localIdx - N_retained;
		const uint32_t bodyIdx = src.bodyId[srcLocalIdx];
		if (src.id[srcLocalIdx] >= N_global)
		{
			if (failMsg) { *failMsg = "Invalid prepared rigid particle ID."; }
			return false;
		}
		if (bodyIdx >= rigidBodies.N_)
		{
			if (failMsg) { *failMsg = "Invalid local rigid body index."; }
			return false;
		}
		#if defined(_DEBUG)
		migratedParticleIdScratch_[localIdx] = src.id[srcLocalIdx];
		#endif
	}
	rebuiltParticleState.N_outer = CountPreparedOuterRigidParticles(
		N_owned,
		retainedRigidOldLocalIdx_FromNewLocalIdx_,
		oldParticles,
		appendedView,
		rigidBodies,
		domainDecomp,
		rank,
		haloRadius);
	#if defined(_DEBUG)
	return rebuiltParticleState.IsValidLocalParticleState(migratedParticleIdScratch_, N_global, failMsg);
	#else
	return rebuiltParticleState.IsValidLocalParticleLayout(N_local, failMsg);
	#endif
}

bool ParticleRedistributionMgr::BuildExchangeRoutes(
	const Sync::FluidConstView& particlesF,
	const Sync::BoundaryConstView& particlesB,
	const Sync::RigidConstView& particlesR,
	const Sync::RigidBodies& rigidBodies,
	const DistParticleState& particleStateF,
	const DistParticleState& particleStateB,
	const DistParticleState& particleStateR,
	const IDomainDecomp& domainDecomp,
	const bool redistributeParticlesB,
	const int rank, const int N_ranks,
	const uint64_t N_globalF, const uint64_t N_globalB, const uint64_t N_globalR,
	std::string* failMsg)
{
	// validate the rank and backend views before replacing reusable routes
	{
		if (failMsg) { failMsg->clear(); }
		if (rank < 0 || rank >= N_ranks)
		{
			if (failMsg) { *failMsg = "Particle redistribution rank is out of range."; }
			return false;
		}
		if (!ValidateParticleViews(
			particlesF, particlesB,
			particleStateF, particleStateB,
			N_globalF, N_globalB, failMsg))
		{
			return false;
		}

		if (!particleStateR.IsValidLocalParticleState(particlesR.id, N_globalR, failMsg))
		{
			return false;
		}
		if (!ValidateLocalRigidParticleAttachments(particlesR, rigidBodies, failMsg))
		{
			return false;
		}

		PrepareBuffers(N_ranks);
	}

	try
	{
		const bool routesValid = RouteOwnedParticles(
			particlesF, particlesB, particlesR, rigidBodies,
			particleStateF, particleStateB, particleStateR,
			domainDecomp, redistributeParticlesB,
			rank, N_ranks, failMsg);
		if (!routesValid)
		{
			ClearFluidRouteOutputs();
			ClearBoundaryRouteOutputs();
			ClearRigidRouteOutputs();
		}
		return routesValid;
	}
	catch (...)
	{
		ClearFluidRouteOutputs();
		ClearBoundaryRouteOutputs();
		ClearRigidRouteOutputs();
		throw;
	}
}

bool ParticleRedistributionMgr::RedistributeParticles(
	Sim::I_DistributedParticleBackend& distParticleBackend,
	DistParticleState& particleStateF,
	DistParticleState& particleStateB,
	DistParticleState& particleStateR,
	std::vector<uint32_t>& outerLocalIdxsF,
	const Sync::RigidBodies& rigidBodies,
	const IDomainDecomp& domainDecomp,
	const MpiContext& mpiContext,
	const Sim::SimParams& simParams,
	const bool redistributeParticlesB,
	const uint64_t N_globalF, const uint64_t N_globalB, const uint64_t N_globalR,
	std::string* failMsg)
{
	if (failMsg) { failMsg->clear(); }

	// reset per-step statistics before building rank-local exchange state
	ResetRedistStats();
	const int rank = mpiContext.Rank();
	const int N_ranks = mpiContext.Size();
	Sync::FluidConstView particlesF{};
	Sync::BoundaryConstView particlesB{};
	Sync::RigidConstView particlesR{};
	bool localOk = true;

	// ensure explicit agreement between ranks independently of local work
	const auto AllRanksTrue = [&](const bool value)
	{
		return mpiContext.AllTrue(value);
	};

	// build exchange routes directly from backend particle views
	try
	{
		{
			particlesF = distParticleBackend.GetFluidParticleView();
			particlesB = distParticleBackend.GetBoundaryParticleView();
			particlesR = distParticleBackend.GetRigidParticleView();
		}

		localOk = BuildExchangeRoutes(
			particlesF, particlesB, particlesR, rigidBodies,
			particleStateF, particleStateB, particleStateR,
			domainDecomp, redistributeParticlesB,
			rank, N_ranks, N_globalF, N_globalB, N_globalR, failMsg);

		// pack outgoing migration and halo payloads
		if (localOk)
		{
			PackSendBuffers(
				particlesF, particlesB, particlesR,
				redistributeParticlesB,
				rank, N_ranks);
		}
	}
	catch (const std::exception& ex)
	{
		if (failMsg) { *failMsg = ex.what(); }
		localOk = false;
	}

	// no rank may enter the count exchange while another rank failed validation
	if (!AllRanksTrue(localOk)) { return false; }

	// exchange migration and halo particle counts and resize all receive buffers accordingly
	try
	{
		mpiContext.AllToAllInts(
			sendCounts_, recvCounts_,
			PARTICLE_REDISTRIBUTION_LANE_COUNT);
		localOk = PrepareReceiveBuffers(rank, N_ranks, failMsg);
	}
	catch (const std::exception& ex)
	{
		if (failMsg) { *failMsg = ex.what(); }
		localOk = false;
	}
	if (!AllRanksTrue(localOk)) { return false; }

	// transfer migration and halo payloads together
	{
		ExchangePayloads(rank, N_ranks, mpiContext);
	}

	// prepare retained compaction, appended storage, and matching metadata
	// (backend mutation starts only after every rank completes this preparation)
	try
	{
		localOk = PrepareFluidParticleMutation(
			particlesF,
			domainDecomp, rank, simParams.distHaloRadius,
			N_globalF, failMsg);
		if (localOk && redistributeParticlesB)
		{
			localOk = PrepareBoundaryParticleMutation(
				particlesB,
				domainDecomp, rank, simParams.distHaloRadius,
				N_globalB, failMsg);
		}
		if (localOk)
		{
			localOk = PrepareRigidParticleMutation(
				particlesR, rigidBodies, domainDecomp,
				rank, simParams.distHaloRadius, N_globalR, failMsg);
		}
	}
	catch (const std::exception& ex)
	{
		if (failMsg) { *failMsg = ex.what(); }
		localOk = false;
	}
	if (!AllRanksTrue(localOk)) { return false; }

	// apply prepared fluid, boundary, and rigid mutations
	try
	{
		localOk = distParticleBackend.ApplyParticleRedistribution(
			retainedFluidOldLocalIdx_FromNewLocalIdx_,
			retainedBoundaryOldLocalIdx_FromNewLocalIdx_,
			retainedRigidOldLocalIdx_FromNewLocalIdx_,
			appendedParticlesF_,
			appendedParticlesB_,
			appendedParticlesR_,
			rebuiltParticleStateF_.N_owned,
			redistributeParticlesB
				? rebuiltParticleStateB_.N_owned
				: particleStateB.N_owned,
			rebuiltParticleStateR_.N_owned,
			redistributeParticlesB,
			simParams);
		if (!localOk && failMsg)
		{
			*failMsg = "Backend rejected particle redistribution mutation.";
		}
	}
	catch (const std::exception& ex)
	{
		if (failMsg) { *failMsg = ex.what(); }
		localOk = false;
	}
	if (!AllRanksTrue(localOk)) { return false; }

	// publish matching metadata only after every backend mutation succeeded
	std::swap(particleStateF, rebuiltParticleStateF_);
	std::swap(outerLocalIdxsF, rebuiltOuterLocalIdxsF_);
	if (redistributeParticlesB)
	{
		std::swap(particleStateB, rebuiltParticleStateB_);
	}

	std::swap(particleStateR, rebuiltParticleStateR_);

	return true;
}
