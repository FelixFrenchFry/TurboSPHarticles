// particle_redistribution.cpp

// external
#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// internal
#include "Source/Base/helpers.h"
#include "particle_redistribution.h"
#include "dist_redistribution_diagnostics.h"
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

		const uint64_t capacityTarget =
			N_elements + Base::ExtraPercent(N_elements, extraAllocCapacityPct);
		values.reserve(capacityTarget);
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

	template <typename T>
	void AssignVectorWithExtraCapacity(
		std::vector<T>& values,
		const uint64_t N_elements,
		const T& value,
		const float extraAllocCapacityPct)
	{
		EnsureVectorCapacityWithExtra(values, N_elements, extraAllocCapacityPct);
		values.assign(N_elements, value);
	}

	uint32_t ParticleCapacityWithExtra(
		const uint32_t N_particles,
		const float extraAllocCapacityPct)
	{
		const uint64_t capacityTarget =
			N_particles + Base::ExtraPercent(
				N_particles,
				extraAllocCapacityPct);

		return static_cast<uint32_t>(std::min(
			capacityTarget,
			static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
	}

	void EnsureFluidParticleCapacityWithExtra(
		Sync::FluidParticles& particles,
		const uint32_t N_particles,
		const float extraAllocCapacityPct)
	{
		if (particles.id.capacity() >= N_particles) { return; }

		particles.reserveAll(ParticleCapacityWithExtra(
			N_particles,
			extraAllocCapacityPct));
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

	// optimized parallel passes rely on validated particle indices being distinct
	// (debug passes claim each flag atomically so invalid duplicates fail without races)
	bool TryClaimParticleFlag(
		std::vector<uint8_t>& flags,
		const uint32_t idx)
	{
		#if defined(_DEBUG)
		std::atomic_ref<uint8_t> flag{ flags[idx] };
		return flag.exchange(1, std::memory_order_relaxed) == 0;
		#else
		if (flags[idx] != 0) { return false; }
		flags[idx] = 1;
		return true;
		#endif
	}

	// scratch states built by this pipeline contain distinct global indices, while the
	// atomic debug store also keeps clearing safe if that invariant is ever violated
	void ClearParticleFlag(
		std::vector<uint8_t>& flags,
		const uint32_t idx)
	{
		#if defined(_DEBUG)
		std::atomic_ref<uint8_t> flag{ flags[idx] };
		flag.store(0, std::memory_order_relaxed);
		#else
		flags[idx] = 0;
		#endif
	}

	// prefix chunk-local bucket counts into disjoint output ranges
	// (block sums are computed in parallel, short bucket rows are prefixed serially,
	// and every block expands from its bucket offsets in parallel)
	void PrefixChunkCountsPerBucket(
		const std::vector<uint32_t>& counts_ByChunkBucket,
		std::vector<uint32_t>& writeOffsets_ByChunkBucket,
		std::vector<uint32_t>& totals,
		const uint32_t N_chunks,
		const uint32_t N_buckets)
	{
		const uint32_t N_blocks = N_chunks > 0
			? 1 + (N_chunks - 1) / PARTICLE_PREFIX_BLOCK_CHUNK_COUNT
			: 0;
		const uint32_t blockStride =
			std::max(N_buckets, PARTICLE_PREFIX_BLOCK_STRIDE_MIN);

		// per block bucket sums, replaced in place by the offset each block starts at
		std::vector<uint32_t> blockSums_ByBlockBucket(
			static_cast<uint64_t>(N_blocks) * blockStride, 0);

		// sum every block independently, so each task owns its own row
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_blocks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t blockIdx = range.begin(); blockIdx != range.end(); blockIdx++)
			{
				const uint32_t beginChunkIdx = blockIdx * PARTICLE_PREFIX_BLOCK_CHUNK_COUNT;
				const uint32_t endChunkIdx =
					beginChunkIdx + std::min(
						PARTICLE_PREFIX_BLOCK_CHUNK_COUNT,
						N_chunks - beginChunkIdx);
				uint32_t* const blockSums =
					blockSums_ByBlockBucket.data() +
					static_cast<uint64_t>(blockIdx) * blockStride;

				for (uint32_t chunkIdx = beginChunkIdx; chunkIdx != endChunkIdx; chunkIdx++)
				{
					const uint64_t chunkBase = static_cast<uint64_t>(chunkIdx) * N_buckets;
					for (uint32_t bucketIdx = 0; bucketIdx != N_buckets; bucketIdx++)
					{
						blockSums[bucketIdx] += counts_ByChunkBucket[chunkBase + bucketIdx];
					}
				}
			}
		});

		// prefix one short row per block, which also yields the bucket totals
		for (uint32_t bucketIdx = 0; bucketIdx != N_buckets; bucketIdx++)
		{
			uint32_t N_bucket = 0;
			for (uint32_t blockIdx = 0; blockIdx != N_blocks; blockIdx++)
			{
				uint32_t& blockSum = blockSums_ByBlockBucket[
					static_cast<uint64_t>(blockIdx) * blockStride + bucketIdx];
				const uint32_t N_block = blockSum;

				blockSum = N_bucket;
				N_bucket += N_block;
			}
			totals[bucketIdx] = N_bucket;
		}

		// expand each block from its own starting offsets, advancing them in place
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_blocks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t blockIdx = range.begin(); blockIdx != range.end(); blockIdx++)
			{
				const uint32_t beginChunkIdx = blockIdx * PARTICLE_PREFIX_BLOCK_CHUNK_COUNT;
				const uint32_t endChunkIdx =
					beginChunkIdx + std::min(
						PARTICLE_PREFIX_BLOCK_CHUNK_COUNT,
						N_chunks - beginChunkIdx);
				uint32_t* const bucketTotals =
					blockSums_ByBlockBucket.data() +
					static_cast<uint64_t>(blockIdx) * blockStride;

				for (uint32_t chunkIdx = beginChunkIdx; chunkIdx != endChunkIdx; chunkIdx++)
				{
					const uint64_t chunkBase = static_cast<uint64_t>(chunkIdx) * N_buckets;
					for (uint32_t bucketIdx = 0; bucketIdx != N_buckets; bucketIdx++)
					{
						const uint64_t chunkBucketIdx = chunkBase + bucketIdx;

						writeOffsets_ByChunkBucket[chunkBucketIdx] = bucketTotals[bucketIdx];
						bucketTotals[bucketIdx] += counts_ByChunkBucket[chunkBucketIdx];
					}
				}
			}
		});
	}

	// rebuild metadata for retained owned particles in the compacted backend prefix,
	// where local index matches compaction map position and the owned list is identity
	bool ClassifyRetainedOwnedParticles_Parallel(
		DistParticleState& rebuiltParticleState,
		ParticleClassificationWorkBuffers& workBuffers,
		const std::vector<uint32_t>& retainedOldLocalIdxs,
		const DistParticleState& oldParticleState,
		const std::vector<uint32_t>& haloNeighborMask_ByLocalIdx,
		const uint32_t N_global,
		const float extraAllocCapacityPct,
		const char* const invalidGlobalIdxMsg,
		std::string* failMsg)
	{
		const uint32_t N_retained =
			static_cast<uint32_t>(retainedOldLocalIdxs.size());
		const uint32_t N_chunks = N_retained > 0
			? 1 + (N_retained - 1) / RETAINED_CLASSIFICATION_CHUNK_SIZE
			: 0;

		AssignVectorWithExtraCapacity(
			workBuffers.classCounts_ByChunkClass,
			static_cast<uint64_t>(N_chunks) * RETAINED_CLASSIFICATION_CLASS_COUNT,
			uint32_t{ 0 },
			extraAllocCapacityPct);
		AssignVectorWithExtraCapacity(
			workBuffers.classWriteOffsets_ByChunkClass,
			static_cast<uint64_t>(N_chunks) * RETAINED_CLASSIFICATION_CLASS_COUNT,
			uint32_t{ 0 },
			extraAllocCapacityPct);
		AssignVectorWithExtraCapacity(
			workBuffers.classTotals,
			RETAINED_CLASSIFICATION_CLASS_COUNT,
			uint32_t{ 0 },
			extraAllocCapacityPct);
		ResizeVectorWithExtraCapacity(
			rebuiltParticleState.particleLocalIdx_Owned,
			N_retained,
			extraAllocCapacityPct);

		// first failure seen by the classification pass, reported once it completes
		std::atomic<const char*> classifyFailMsg{ nullptr };

		// classify retained particles and count each class per chunk
		// (chunks processed in parallel, with each worker writing only its own count row)
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginLocalIdx =
					chunkIdx * RETAINED_CLASSIFICATION_CHUNK_SIZE;
				const uint32_t endLocalIdx =
					beginLocalIdx + std::min(
						RETAINED_CLASSIFICATION_CHUNK_SIZE,
						N_retained - beginLocalIdx);
				const uint64_t chunkClassBase =
					static_cast<uint64_t>(chunkIdx) * RETAINED_CLASSIFICATION_CLASS_COUNT;

				for (uint32_t localIdx = beginLocalIdx; localIdx != endLocalIdx; localIdx++)
				{
					const uint32_t oldLocalIdx = retainedOldLocalIdxs[localIdx];
					const uint32_t globalIdx = oldParticleState.particleGlobalIdx_FromLocalIdx[oldLocalIdx];

					if (globalIdx >= N_global)
					{
						classifyFailMsg.store(invalidGlobalIdxMsg, std::memory_order_relaxed);
						continue;
					}
					if (!TryClaimParticleFlag(
						rebuiltParticleState.activeGlobalIdxFlags,
						globalIdx))
					{
						classifyFailMsg.store(invalidGlobalIdxMsg, std::memory_order_relaxed);
						continue;
					}

					// routing tested retained particles against their owner-rank neighbor halo bounds,
					// so a particle that no neighbor requested is inner
					const bool isInner = haloNeighborMask_ByLocalIdx[oldLocalIdx] == 0;
					rebuiltParticleState.particleClasses[localIdx] = isInner
						? DistParticleClass::Inner
						: DistParticleClass::Outer;
					workBuffers.classCounts_ByChunkClass[
						chunkClassBase + (isInner
							? RETAINED_CLASSIFICATION_CLASS_INNER
							: RETAINED_CLASSIFICATION_CLASS_OUTER)]++;

					rebuiltParticleState.particleGlobalIdx_FromLocalIdx[localIdx] = globalIdx;
					rebuiltParticleState.particleLocalIdx_Owned[localIdx] = localIdx;
				}
			}
		});

		if (const char* const classifyFail =
			classifyFailMsg.load(std::memory_order_relaxed))
		{
			if (failMsg) { *failMsg = classifyFail; }
			return false;
		}

		// prefix chunk-local class counts to compute each chunk's output range
		PrefixChunkCountsPerBucket(
			workBuffers.classCounts_ByChunkClass,
			workBuffers.classWriteOffsets_ByChunkClass,
			workBuffers.classTotals,
			N_chunks,
			RETAINED_CLASSIFICATION_CLASS_COUNT);

		ResizeVectorWithExtraCapacity(
			rebuiltParticleState.particleLocalIdx_Inner,
			workBuffers.classTotals[RETAINED_CLASSIFICATION_CLASS_INNER],
			extraAllocCapacityPct);
		ResizeVectorWithExtraCapacity(
			rebuiltParticleState.particleLocalIdx_Outer,
			workBuffers.classTotals[RETAINED_CLASSIFICATION_CLASS_OUTER],
			extraAllocCapacityPct);

		// fill inner and outer index lists while preserving local-index order
		// (chunks processed in parallel, with each worker writing to disjoint ranges)
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginLocalIdx =
					chunkIdx * RETAINED_CLASSIFICATION_CHUNK_SIZE;
				const uint32_t endLocalIdx =
					beginLocalIdx + std::min(
						RETAINED_CLASSIFICATION_CHUNK_SIZE,
						N_retained - beginLocalIdx);
				const uint64_t chunkClassBase =
					static_cast<uint64_t>(chunkIdx) * RETAINED_CLASSIFICATION_CLASS_COUNT;
				uint32_t innerWriteIdx =
					workBuffers.classWriteOffsets_ByChunkClass[
						chunkClassBase + RETAINED_CLASSIFICATION_CLASS_INNER];
				uint32_t outerWriteIdx =
					workBuffers.classWriteOffsets_ByChunkClass[
						chunkClassBase + RETAINED_CLASSIFICATION_CLASS_OUTER];

				for (uint32_t localIdx = beginLocalIdx;
					 localIdx != endLocalIdx;
					 localIdx++)
				{
					if (rebuiltParticleState.particleClasses[localIdx] ==
						DistParticleClass::Inner)
					{
						rebuiltParticleState.particleLocalIdx_Inner[innerWriteIdx++] = localIdx;
					}
					else
					{
						rebuiltParticleState.particleLocalIdx_Outer[outerWriteIdx++] = localIdx;
					}
				}
			}
		});

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
	uint64_t CheckedByteSize(const uint32_t N_elements)
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
		const uint32_t N_elements,
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
		const uint32_t N_elements,
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

} // namespace anonymous

void ParticleRedistributionMgr::FluidExchangeBuffer::Clear()
{
	N_particles_ = 0;
}

void ParticleRedistributionMgr::FluidExchangeBuffer::Resize(
	const uint32_t N)
{
	// grow reusable storage to the largest requested particle count
	if (globalIdx_.size() < N)
	{
		EnsureVectorCapacityWithExtra(
			globalIdx_, N, exchangeBufferExtraAllocCapacityPct_);
		EnsureFluidParticleCapacityWithExtra(
			particles_, N, exchangeBufferExtraAllocCapacityPct_);
		globalIdx_.resize(N);
		particles_.resizeAll(N);
	}

	N_particles_ = N;
}

void ParticleRedistributionMgr::FluidExchangeBuffer::SetBulkFromParticleView(
	const Sync::FluidConstView& src,
	const DistParticleState& particleState,
	const std::vector<uint32_t>& localIdxs)
{
	Resize(static_cast<uint32_t>(localIdxs.size()));
	if (localIdxs.empty()) { return; }

	// gather particle fields sequentially within each parallel block
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_particles_, PARTICLE_GATHER_BLOCK_SIZE),
		[&](const tbb::blocked_range<uint32_t>& blockRange)
	{
		GatherParticleField(globalIdx_, particleState.particleGlobalIdx_FromLocalIdx, localIdxs, blockRange);
		GatherParticleField(particles_.id, src.id, localIdxs, blockRange);
		GatherParticleField(particles_.xPos, src.xPos, localIdxs, blockRange);
		GatherParticleField(particles_.yPos, src.yPos, localIdxs, blockRange);
		GatherParticleField(particles_.zPos, src.zPos, localIdxs, blockRange);
		GatherParticleField(particles_.xVel, src.xVel, localIdxs, blockRange);
		GatherParticleField(particles_.yVel, src.yVel, localIdxs, blockRange);
		GatherParticleField(particles_.zVel, src.zVel, localIdxs, blockRange);

		// the fields velPred, accPress, sourceTerm, diagElem, pressure, and density are omitted,
		// because the following solver step re-writes them before reading them
		// TODO: keep it like this or copy them too, if solver step behavior changes?
	});
}

void ParticleRedistributionMgr::FluidExchangeBuffer::CopyToLocalParticle(
	Sync::FluidParticles& dst,
	const uint32_t bufferIdx,
	const uint32_t localIdx) const
{
	dst.id[localIdx] = particles_.id[bufferIdx];
	dst.xPos[localIdx] = particles_.xPos[bufferIdx];
	dst.yPos[localIdx] = particles_.yPos[bufferIdx];
	dst.zPos[localIdx] = particles_.zPos[bufferIdx];
	dst.xVel[localIdx] = particles_.xVel[bufferIdx];
	dst.yVel[localIdx] = particles_.yVel[bufferIdx];
	dst.zVel[localIdx] = particles_.zVel[bufferIdx];

	// the fields velPred, accPress, sourceTerm, diagElem, pressure, and density are omitted,
	// because the following solver step re-writes them before reading them
	// TODO: keep it like this or copy them too, if solver step behavior changes?
}

void ParticleRedistributionMgr::FluidExchangeBuffer::CopyBulkToLocalParticles(
	Sync::FluidParticles& dst,
	const uint32_t dstBeginLocalIdx) const
{
	if (IsEmpty()) { return; }

	// copy particle fields sequentially within each parallel block
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_particles_, PARTICLE_GATHER_BLOCK_SIZE),
		[&](const tbb::blocked_range<uint32_t>& blockRange)
	{
		CopyParticleField(dst.id, particles_.id, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.xPos, particles_.xPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.yPos, particles_.yPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.zPos, particles_.zPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.xVel, particles_.xVel, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.yVel, particles_.yVel, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.zVel, particles_.zVel, dstBeginLocalIdx, blockRange);

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

	QueueIRecvVector(mpiContext, globalIdx_, N_particles_, peerRank, tagBase + 1, requests);
	QueueIRecvVector(mpiContext, particles_.id, N_particles_, peerRank, tagBase + 2, requests);
	QueueIRecvVector(mpiContext, particles_.xPos, N_particles_, peerRank, tagBase + 3, requests);
	QueueIRecvVector(mpiContext, particles_.yPos, N_particles_, peerRank, tagBase + 4, requests);
	QueueIRecvVector(mpiContext, particles_.zPos, N_particles_, peerRank, tagBase + 5, requests);
	QueueIRecvVector(mpiContext, particles_.xVel, N_particles_, peerRank, tagBase + 6, requests);
	QueueIRecvVector(mpiContext, particles_.yVel, N_particles_, peerRank, tagBase + 7, requests);
	QueueIRecvVector(mpiContext, particles_.zVel, N_particles_, peerRank, tagBase + 8, requests);

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

	QueueISendVector(mpiContext, globalIdx_, N_particles_, peerRank, tagBase + 1, requests);
	QueueISendVector(mpiContext, particles_.id, N_particles_, peerRank, tagBase + 2, requests);
	QueueISendVector(mpiContext, particles_.xPos, N_particles_, peerRank, tagBase + 3, requests);
	QueueISendVector(mpiContext, particles_.yPos, N_particles_, peerRank, tagBase + 4, requests);
	QueueISendVector(mpiContext, particles_.zPos, N_particles_, peerRank, tagBase + 5, requests);
	QueueISendVector(mpiContext, particles_.xVel, N_particles_, peerRank, tagBase + 6, requests);
	QueueISendVector(mpiContext, particles_.yVel, N_particles_, peerRank, tagBase + 7, requests);
	QueueISendVector(mpiContext, particles_.zVel, N_particles_, peerRank, tagBase + 8, requests);

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

uint32_t ParticleRedistributionMgr::FluidExchangeBuffer::GlobalIdxAt(
	const uint32_t bufferIdx) const
{
	return globalIdx_[bufferIdx];
}

Mem::MemUsage ParticleRedistributionMgr::FluidExchangeBuffer::GetMemoryUsage() const
{
	Mem::MemUsage usage = Mem::MemUsageOf_Vector(globalIdx_);

	Mem::Add(usage, GetParticleMemoryUsage(particles_));

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
	if (globalIdx_.size() < N)
	{
		EnsureVectorCapacityWithExtra(
			globalIdx_, N, exchangeBufferExtraAllocCapacityPct_);
		EnsureBoundaryParticleCapacityWithExtra(
			particles_, N, exchangeBufferExtraAllocCapacityPct_);
		globalIdx_.resize(N);
		particles_.resizeAll(N);
	}

	N_particles_ = N;
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::SetBulkFromParticleView(
	const Sync::BoundaryConstView& src,
	const DistParticleState& particleState,
	const std::vector<uint32_t>& localIdxs)
{
	Resize(static_cast<uint32_t>(localIdxs.size()));
	if (localIdxs.empty()) { return; }

	// gather particle fields sequentially within each parallel block
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_particles_, PARTICLE_GATHER_BLOCK_SIZE),
		[&](const tbb::blocked_range<uint32_t>& blockRange)
	{
		GatherParticleField(globalIdx_, particleState.particleGlobalIdx_FromLocalIdx, localIdxs, blockRange);
		GatherParticleField(particles_.id, src.id, localIdxs, blockRange);
		GatherParticleField(particles_.xPos, src.xPos, localIdxs, blockRange);
		GatherParticleField(particles_.yPos, src.yPos, localIdxs, blockRange);
		GatherParticleField(particles_.zPos, src.zPos, localIdxs, blockRange);
		GatherParticleField(particles_.volB, src.volB, localIdxs, blockRange);
		GatherParticleField(particles_.psiB, src.psiB, localIdxs, blockRange);
	});
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::CopyToLocalParticle(
	Sync::BoundaryParticles& dst,
	const uint32_t bufferIdx,
	const uint32_t localIdx) const
{
	dst.id[localIdx] = particles_.id[bufferIdx];
	dst.xPos[localIdx] = particles_.xPos[bufferIdx];
	dst.yPos[localIdx] = particles_.yPos[bufferIdx];
	dst.zPos[localIdx] = particles_.zPos[bufferIdx];
	dst.volB[localIdx] = particles_.volB[bufferIdx];
	dst.psiB[localIdx] = particles_.psiB[bufferIdx];
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::CopyBulkToLocalParticles(
	Sync::BoundaryParticles& dst,
	const uint32_t dstBeginLocalIdx) const
{
	if (IsEmpty()) { return; }

	// copy particle fields sequentially within each parallel block
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_particles_, PARTICLE_GATHER_BLOCK_SIZE),
		[&](const tbb::blocked_range<uint32_t>& blockRange)
	{
		CopyParticleField(dst.id, particles_.id, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.xPos, particles_.xPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.yPos, particles_.yPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.zPos, particles_.zPos, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.volB, particles_.volB, dstBeginLocalIdx, blockRange);
		CopyParticleField(dst.psiB, particles_.psiB, dstBeginLocalIdx, blockRange);
	});
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::QueueIRecvPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests)
{
	if (IsEmpty()) { return; }

	QueueIRecvVector(mpiContext, globalIdx_, N_particles_, peerRank, tagBase + 1, requests);
	QueueIRecvVector(mpiContext, particles_.id, N_particles_, peerRank, tagBase + 2, requests);
	QueueIRecvVector(mpiContext, particles_.xPos, N_particles_, peerRank, tagBase + 3, requests);
	QueueIRecvVector(mpiContext, particles_.yPos, N_particles_, peerRank, tagBase + 4, requests);
	QueueIRecvVector(mpiContext, particles_.zPos, N_particles_, peerRank, tagBase + 5, requests);
	QueueIRecvVector(mpiContext, particles_.volB, N_particles_, peerRank, tagBase + 6, requests);
	QueueIRecvVector(mpiContext, particles_.psiB, N_particles_, peerRank, tagBase + 7, requests);
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::QueueISendPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests) const
{
	if (IsEmpty()) { return; }

	QueueISendVector(mpiContext, globalIdx_, N_particles_, peerRank, tagBase + 1, requests);
	QueueISendVector(mpiContext, particles_.id, N_particles_, peerRank, tagBase + 2, requests);
	QueueISendVector(mpiContext, particles_.xPos, N_particles_, peerRank, tagBase + 3, requests);
	QueueISendVector(mpiContext, particles_.yPos, N_particles_, peerRank, tagBase + 4, requests);
	QueueISendVector(mpiContext, particles_.zPos, N_particles_, peerRank, tagBase + 5, requests);
	QueueISendVector(mpiContext, particles_.volB, N_particles_, peerRank, tagBase + 6, requests);
	QueueISendVector(mpiContext, particles_.psiB, N_particles_, peerRank, tagBase + 7, requests);
}

int ParticleRedistributionMgr::BoundaryExchangeBuffer::CheckedCount() const
{
	if (Size() > static_cast<uint32_t>(std::numeric_limits<int>::max()))
	{
		throw std::runtime_error("Boundary particle redistribution count exceeds MPI int range.");
	}

	return static_cast<int>(Size());
}

uint32_t ParticleRedistributionMgr::BoundaryExchangeBuffer::GlobalIdxAt(
	const uint32_t bufferIdx) const
{
	return globalIdx_[bufferIdx];
}

Mem::MemUsage ParticleRedistributionMgr::BoundaryExchangeBuffer::GetMemoryUsage() const
{
	Mem::MemUsage usage = Mem::MemUsageOf_Vector(globalIdx_);

	Mem::Add(usage, GetParticleMemoryUsage(particles_));

	return usage;
}

Mem::MemUsage ParticleRedistributionMgr::GetExchangeBufferMemoryUsage() const
{
	Mem::MemUsage usage = Mem::MemUsageOf_Vector(rankBuffers_);

	for (const RankBuffers& rankBuffers : rankBuffers_)
	{
		Mem::Add(usage, rankBuffers.sendOwnedFluidParticles.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.recvOwnedFluidParticles.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.sendHaloFluidParticles.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.recvHaloFluidParticles.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.sendOwnedBoundaryParticles.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.recvOwnedBoundaryParticles.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.sendHaloBoundaryParticles.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.recvHaloBoundaryParticles.GetMemoryUsage());
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
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.keptOwnedFluidLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.localHaloFluidLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.keptOwnedBoundaryLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.localHaloBoundaryLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routingWorkBuffers_.routeCounts_ByChunkBucket));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routingWorkBuffers_.routeWriteOffsets_ByChunkBucket));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routingWorkBuffers_.routeTotals));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routingWorkBuffers_.ownerRank_ByOwnedIdx));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.fluidHaloNeighborMask_ByLocalIdx));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.boundaryHaloNeighborMask_ByLocalIdx));
	Mem::Add(usage, Mem::MemUsageOf_Vector(compactionWorkBuffers_.compactionCounts_ByChunkKind));
	Mem::Add(usage, Mem::MemUsageOf_Vector(compactionWorkBuffers_.compactionWriteOffsets_ByChunkKind));
	Mem::Add(usage, Mem::MemUsageOf_Vector(compactionWorkBuffers_.compactionTotals));
	Mem::Add(usage, Mem::MemUsageOf_Vector(compactionWorkBuffers_.holeOldLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(compactionWorkBuffers_.sourceOldLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(retainedClassificationWorkBuffers_.classCounts_ByChunkClass));
	Mem::Add(usage, Mem::MemUsageOf_Vector(retainedClassificationWorkBuffers_.classWriteOffsets_ByChunkClass));
	Mem::Add(usage, Mem::MemUsageOf_Vector(retainedClassificationWorkBuffers_.classTotals));
	Mem::Add(usage, Mem::MemUsageOf_Vector(sendCounts_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(recvCounts_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(retainedFluidOldLocalIdx_FromNewLocalIdx_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(retainedBoundaryOldLocalIdx_FromNewLocalIdx_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(retainedOldLocalIdxFlags_));
	Mem::Add(usage, GetParticleMemoryUsage(appendedFluidParticles_));
	Mem::Add(usage, GetParticleMemoryUsage(appendedBoundaryParticles_));
	Mem::Add(usage, GetParticleMemoryUsage(rebuiltFluidParticleState_));
	Mem::Add(usage, GetParticleMemoryUsage(rebuiltBoundaryParticleState_));

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

bool ParticleRedistributionMgr::BuildDenseCompactionMap(
	const uint32_t N_oldLocal,
	const std::vector<uint32_t>& retainedOldLocalIdxs,
	std::vector<uint32_t>& oldLocalIdx_FromNewLocalIdx,
	std::vector<uint8_t>& retainedOldLocalIdxFlags,
	ParticleCompactionWorkBuffers& workBuffers,
	std::string* failMsg)
{
	if (retainedOldLocalIdxs.size() > N_oldLocal)
	{
		if (failMsg) { *failMsg = "Retained particle count exceeds old local particle count."; }
		return false;
	}
	const uint32_t N_retained = static_cast<uint32_t>(retainedOldLocalIdxs.size());

	ResizeVectorWithExtraCapacity(
		retainedOldLocalIdxFlags,
		N_oldLocal,
		scratchBufferExtraAllocCapacityPct_);
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_oldLocal),
		[&](const tbb::blocked_range<uint32_t>& range)
	{
		std::fill(
			retainedOldLocalIdxFlags.begin() + range.begin(),
			retainedOldLocalIdxFlags.begin() + range.end(),
			uint8_t{ 0 });
	});

	// first failure seen by the marking pass, reported once it completes
	std::atomic<const char*> markFailMsg{ nullptr };

	// mark retained particles in the old local range
	// (validated inputs write distinct flags, while debug claims reject duplicates atomically)
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_retained),
		[&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t retainedIdx = range.begin(); retainedIdx != range.end(); retainedIdx++)
		{
			const uint32_t oldLocalIdx = retainedOldLocalIdxs[retainedIdx];
			if (oldLocalIdx >= N_oldLocal)
			{
				markFailMsg.store(
					"Retained particle local index is invalid or duplicated.",
					std::memory_order_relaxed);
				continue;
			}

			if (!TryClaimParticleFlag(
				retainedOldLocalIdxFlags,
				oldLocalIdx))
			{
				markFailMsg.store(
					"Retained particle local index is invalid or duplicated.",
					std::memory_order_relaxed);
			}
		}
	});

	if (const char* const markFail = markFailMsg.load(std::memory_order_relaxed))
	{
		if (failMsg) { *failMsg = markFail; }
		return false;
	}

	// every removed slot below the retained prefix takes a retained particle from above
	// it, so both sides are collected as compacted lists and then paired by position
	const uint32_t N_chunks = N_oldLocal > 0
		? 1 + (N_oldLocal - 1) / PARTICLE_COMPACTION_CHUNK_SIZE
		: 0;
	const uint64_t N_chunkKinds =
		static_cast<uint64_t>(N_chunks) * PARTICLE_COMPACTION_KIND_COUNT;

	AssignVectorWithExtraCapacity(
		workBuffers.compactionCounts_ByChunkKind,
		N_chunkKinds,
		uint32_t{ 0 },
		scratchBufferExtraAllocCapacityPct_);
	AssignVectorWithExtraCapacity(
		workBuffers.compactionWriteOffsets_ByChunkKind,
		N_chunkKinds,
		uint32_t{ 0 },
		scratchBufferExtraAllocCapacityPct_);
	AssignVectorWithExtraCapacity(
		workBuffers.compactionTotals,
		PARTICLE_COMPACTION_KIND_COUNT,
		uint32_t{ 0 },
		scratchBufferExtraAllocCapacityPct_);

	// count compact-prefix holes and retained tail sources per chunk
	// (chunks processed in parallel, with each worker writing only its own count row)
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_chunks),
		[&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
		{
			const uint32_t beginOldLocalIdx = chunkIdx * PARTICLE_COMPACTION_CHUNK_SIZE;
			const uint32_t endOldLocalIdx =
				beginOldLocalIdx + std::min(
					PARTICLE_COMPACTION_CHUNK_SIZE,
					N_oldLocal - beginOldLocalIdx);
			uint32_t N_holes = 0;
			uint32_t N_sources = 0;

			for (uint32_t oldLocalIdx = beginOldLocalIdx; oldLocalIdx != endOldLocalIdx; oldLocalIdx++)
			{
				const bool retained = retainedOldLocalIdxFlags[oldLocalIdx] != 0;

				if (oldLocalIdx < N_retained)
				{
					if (!retained) { N_holes++; }
				}
				else if (retained)
				{
					N_sources++;
				}
			}

			const uint64_t chunkKindBase =
				static_cast<uint64_t>(chunkIdx) * PARTICLE_COMPACTION_KIND_COUNT;
			workBuffers.compactionCounts_ByChunkKind[
				chunkKindBase + PARTICLE_COMPACTION_KIND_HOLE] = N_holes;
			workBuffers.compactionCounts_ByChunkKind[
				chunkKindBase + PARTICLE_COMPACTION_KIND_SOURCE] = N_sources;
		}
	});

	PrefixChunkCountsPerBucket(
		workBuffers.compactionCounts_ByChunkKind,
		workBuffers.compactionWriteOffsets_ByChunkKind,
		workBuffers.compactionTotals,
		N_chunks,
		PARTICLE_COMPACTION_KIND_COUNT);

	const uint32_t N_holes = workBuffers.compactionTotals[PARTICLE_COMPACTION_KIND_HOLE];
	if (N_holes != workBuffers.compactionTotals[PARTICLE_COMPACTION_KIND_SOURCE])
	{
		if (failMsg) { *failMsg = "Retained particle compaction map is inconsistent."; }
		return false;
	}

	ResizeVectorWithExtraCapacity(
		workBuffers.holeOldLocalIdxs,
		N_holes,
		scratchBufferExtraAllocCapacityPct_);
	ResizeVectorWithExtraCapacity(
		workBuffers.sourceOldLocalIdxs,
		N_holes,
		scratchBufferExtraAllocCapacityPct_);

	// fill hole and source lists while preserving ascending old local index order
	// (chunks processed in parallel, with each worker writing to disjoint ranges)
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_chunks),
		[&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
		{
			const uint32_t beginOldLocalIdx = chunkIdx * PARTICLE_COMPACTION_CHUNK_SIZE;
			const uint32_t endOldLocalIdx =
				beginOldLocalIdx + std::min(
					PARTICLE_COMPACTION_CHUNK_SIZE,
					N_oldLocal - beginOldLocalIdx);
			const uint64_t chunkKindBase =
				static_cast<uint64_t>(chunkIdx) * PARTICLE_COMPACTION_KIND_COUNT;
			uint32_t holeWriteIdx = workBuffers.compactionWriteOffsets_ByChunkKind[
				chunkKindBase + PARTICLE_COMPACTION_KIND_HOLE];
			uint32_t sourceWriteIdx = workBuffers.compactionWriteOffsets_ByChunkKind[
				chunkKindBase + PARTICLE_COMPACTION_KIND_SOURCE];

			for (uint32_t oldLocalIdx = beginOldLocalIdx; oldLocalIdx != endOldLocalIdx; oldLocalIdx++)
			{
				const bool retained = retainedOldLocalIdxFlags[oldLocalIdx] != 0;

				if (oldLocalIdx < N_retained)
				{
					if (!retained)
					{
						workBuffers.holeOldLocalIdxs[holeWriteIdx++] = oldLocalIdx;
					}
				}
				else if (retained)
				{
					workBuffers.sourceOldLocalIdxs[sourceWriteIdx++] = oldLocalIdx;
				}
			}
		}
	});

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
			if (retainedOldLocalIdxFlags[newLocalIdx] != 0)
			{
				oldLocalIdx_FromNewLocalIdx[newLocalIdx] = newLocalIdx;
			}
		}
	});

	// the first hole takes the highest retained particle, so the old tail empties first
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_holes),
		[&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t holeIdx = range.begin(); holeIdx != range.end(); holeIdx++)
		{
			oldLocalIdx_FromNewLocalIdx[workBuffers.holeOldLocalIdxs[holeIdx]] =
				workBuffers.sourceOldLocalIdxs[N_holes - 1 - holeIdx];
		}
	});

	return true;
}

void ParticleRedistributionMgr::PrepareRebuiltParticleState(
	DistParticleState& particleState,
	const uint32_t N_owned,
	const uint32_t N_halo,
	const uint32_t N_local,
	const uint32_t N_global)
{
	// clear only global flags set by the previous scratch state
	auto& activeGlobalIdxFlags = particleState.activeGlobalIdxFlags;
	if (activeGlobalIdxFlags.size() == N_global)
	{
		const std::vector<uint32_t>& globalIdx_FromLocalIdx =
			particleState.particleGlobalIdx_FromLocalIdx;
		const uint32_t N_oldLocal =
			static_cast<uint32_t>(globalIdx_FromLocalIdx.size());

		// clear scattered global flags in parallel
		// (debug stores remain race-free if invalid duplicate mappings reach this point)
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_oldLocal),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t localIdx = range.begin(); localIdx != range.end(); localIdx++)
			{
				const uint32_t globalIdx = globalIdx_FromLocalIdx[localIdx];
				if (globalIdx >= activeGlobalIdxFlags.size())
				{
					continue;
				}
				ClearParticleFlag(activeGlobalIdxFlags, globalIdx);
			}
		});
	}
	else
	{
		AssignVectorWithExtraCapacity(
			activeGlobalIdxFlags,
			N_global,
			uint8_t{ 0 },
			scratchBufferExtraAllocCapacityPct_);
	}

	// initialize compact local metadata, while retaining per-particle storage
	particleState.N_owned = N_owned;
	particleState.N_halo = N_halo;
	particleState.particleLocalIdx_Owned.clear();
	particleState.particleLocalIdx_Inner.clear();
	particleState.particleLocalIdx_Outer.clear();
	particleState.particleLocalIdx_Halo.clear();
	ResizeVectorWithExtraCapacity(
		particleState.particleClasses,
		N_local,
		scratchBufferExtraAllocCapacityPct_);
	ResizeVectorWithExtraCapacity(
		particleState.particleGlobalIdx_FromLocalIdx,
		N_local,
		scratchBufferExtraAllocCapacityPct_);

	// reset particle classes and local-to-global mappings in parallel
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_local),
		[&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t localIdx = range.begin(); localIdx != range.end(); localIdx++)
		{
			particleState.particleClasses[localIdx] = DistParticleClass::None;
			particleState.particleGlobalIdx_FromLocalIdx[localIdx] = NO_GLOBAL_PARTICLE_IDX;
		}
	});

	EnsureVectorCapacityWithExtra(
		particleState.particleLocalIdx_Owned,
		N_owned,
		scratchBufferExtraAllocCapacityPct_);
	EnsureVectorCapacityWithExtra(
		particleState.particleLocalIdx_Inner,
		N_owned,
		scratchBufferExtraAllocCapacityPct_);
	EnsureVectorCapacityWithExtra(
		particleState.particleLocalIdx_Outer,
		N_owned,
		scratchBufferExtraAllocCapacityPct_);
	EnsureVectorCapacityWithExtra(
		particleState.particleLocalIdx_Halo,
		N_halo,
		scratchBufferExtraAllocCapacityPct_);
}

bool ParticleRedistributionMgr::AddParticleMetadata(
	DistParticleState& particleState,
	const DistParticleClass particleClass,
	const uint32_t N_global,
	const uint32_t globalIdx,
	const uint32_t localIdx)
{
	if (globalIdx >= N_global ||
		particleState.activeGlobalIdxFlags[globalIdx] != 0)
	{
		return false;
	}

	particleState.particleGlobalIdx_FromLocalIdx[localIdx] = globalIdx;
	particleState.activeGlobalIdxFlags[globalIdx] = 1;
	particleState.particleClasses[localIdx] = particleClass;

	if (particleClass == DistParticleClass::Halo)
	{
		particleState.particleLocalIdx_Halo.push_back(localIdx);
	}
	else
	{
		particleState.particleLocalIdx_Owned.push_back(localIdx);
		if (particleClass == DistParticleClass::Inner)
		{
			particleState.particleLocalIdx_Inner.push_back(localIdx);
		}
		else
		{
			particleState.particleLocalIdx_Outer.push_back(localIdx);
		}
	}

	return true;
}

bool ParticleRedistributionMgr::AddHaloParticleMetadata(
	DistParticleState& particleState,
	const uint32_t N_global,
	const uint32_t globalIdx,
	const uint32_t localIdx,
	const uint32_t haloIdx)
{
	if (globalIdx >= N_global)
	{
		return false;
	}
	if (!TryClaimParticleFlag(particleState.activeGlobalIdxFlags, globalIdx))
	{
		return false;
	}

	particleState.particleGlobalIdx_FromLocalIdx[localIdx] = globalIdx;
	particleState.particleClasses[localIdx] = DistParticleClass::Halo;
	particleState.particleLocalIdx_Halo[haloIdx] = localIdx;

	return true;
}

bool ParticleRedistributionMgr::CheckedUint32(
	const uint64_t value,
	uint32_t& result,
	const char* valueName,
	std::string* failMsg)
{
	if (value > std::numeric_limits<uint32_t>::max())
	{
		if (failMsg)
		{
			*failMsg = std::string(valueName) + " exceeds uint32_t range.";
		}
		return false;
	}

	result = static_cast<uint32_t>(value);
	return true;
}

bool ParticleRedistributionMgr::ValidateParticleViews(
	const Sync::FluidConstView& fluidParticles,
	const Sync::BoundaryConstView& boundaryParticles,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	std::string* failMsg)
{
	const uint32_t N_localF = fluidParticles.N_;
	const uint32_t N_localB = boundaryParticles.N_;

	// check that all fluid particle fields have the same size
	const auto fluidSizeMatches = [N_localF](const auto& values)
	{
		return values.size() == N_localF;
	};
	if (!fluidSizeMatches(fluidParticles.id) ||
		!fluidSizeMatches(fluidParticles.xPos) ||
		!fluidSizeMatches(fluidParticles.yPos) ||
		!fluidSizeMatches(fluidParticles.zPos) ||
		!fluidSizeMatches(fluidParticles.xVel) ||
		!fluidSizeMatches(fluidParticles.yVel) ||
		!fluidSizeMatches(fluidParticles.zVel) ||
		!fluidSizeMatches(fluidParticles.xVelPred) ||
		!fluidSizeMatches(fluidParticles.yVelPred) ||
		!fluidSizeMatches(fluidParticles.zVelPred) ||
		!fluidSizeMatches(fluidParticles.xAccPress) ||
		!fluidSizeMatches(fluidParticles.yAccPress) ||
		!fluidSizeMatches(fluidParticles.zAccPress) ||
		!fluidSizeMatches(fluidParticles.sourceTerm) ||
		!fluidSizeMatches(fluidParticles.diagElem) ||
		!fluidSizeMatches(fluidParticles.pressure) ||
		!fluidSizeMatches(fluidParticles.density))
	{
		if (failMsg) { *failMsg = "Backend fluid particle view has inconsistent field sizes."; }
		return false;
	}

	// check that all boundary particle fields have the same size
	const auto boundarySizeMatches = [N_localB](const auto& values)
	{
		return values.size() == N_localB;
	};
	if (!boundarySizeMatches(boundaryParticles.id) ||
		!boundarySizeMatches(boundaryParticles.xPos) ||
		!boundarySizeMatches(boundaryParticles.yPos) ||
		!boundarySizeMatches(boundaryParticles.zPos) ||
		!boundarySizeMatches(boundaryParticles.volB) ||
		!boundarySizeMatches(boundaryParticles.psiB))
	{
		if (failMsg) { *failMsg = "Backend boundary particle view has inconsistent field sizes."; }
		return false;
	}

	#if defined(_DEBUG)
	// exhaustively validate distributed particle metadata
	const uint32_t N_globalF = static_cast<uint32_t>(
		fluidParticleState.activeGlobalIdxFlags.size());
	if (!fluidParticleState.IsValidLocalParticleState(
		N_localF, N_globalF, failMsg))
	{
		return false;
	}
	const uint32_t N_globalB = static_cast<uint32_t>(
		boundaryParticleState.activeGlobalIdxFlags.size());
	if (!boundaryParticleState.IsValidLocalParticleState(
		N_localB, N_globalB, failMsg))
	{
		return false;
	}
	#else
	(void)fluidParticleState;
	(void)boundaryParticleState;
	#endif

	return true;
}

void ParticleRedistributionMgr::PrepareBuffers(const int N_ranks)
{
	rankBuffers_.resize(static_cast<uint64_t>(N_ranks));
	routes_.ownedFluidLocalIdxs_ByDstRank.resize(static_cast<uint64_t>(N_ranks));
	routes_.haloFluidLocalIdxs_ByDstRank.resize(static_cast<uint64_t>(N_ranks));
	routes_.ownedBoundaryLocalIdxs_ByDstRank.resize(static_cast<uint64_t>(N_ranks));
	routes_.haloBoundaryLocalIdxs_ByDstRank.resize(static_cast<uint64_t>(N_ranks));

	// reset the send and receive buffers between this rank and every other rank
	for (int rank = 0; rank != N_ranks; rank++)
	{
		auto& rankBuffers = rankBuffers_[rank];
		rankBuffers.sendOwnedFluidParticles.Clear();
		rankBuffers.recvOwnedFluidParticles.Clear();
		rankBuffers.sendHaloFluidParticles.Clear();
		rankBuffers.recvHaloFluidParticles.Clear();
		rankBuffers.sendOwnedBoundaryParticles.Clear();
		rankBuffers.recvOwnedBoundaryParticles.Clear();
		rankBuffers.sendHaloBoundaryParticles.Clear();
		rankBuffers.recvHaloBoundaryParticles.Clear();
		rankBuffers.N_sendOwnedF = 0;
		rankBuffers.N_recvOwnedF = 0;
		rankBuffers.N_sendHaloF = 0;
		rankBuffers.N_recvHaloF = 0;
		rankBuffers.N_sendOwnedB = 0;
		rankBuffers.N_recvOwnedB = 0;
		rankBuffers.N_sendHaloB = 0;
		rankBuffers.N_recvHaloB = 0;

		routes_.ownedFluidLocalIdxs_ByDstRank[rank].clear();
		routes_.haloFluidLocalIdxs_ByDstRank[rank].clear();
		routes_.ownedBoundaryLocalIdxs_ByDstRank[rank].clear();
		routes_.haloBoundaryLocalIdxs_ByDstRank[rank].clear();
	}

	routes_.keptOwnedFluidLocalIdxs.clear();
	routes_.localHaloFluidLocalIdxs.clear();
	routes_.keptOwnedBoundaryLocalIdxs.clear();
	routes_.localHaloBoundaryLocalIdxs.clear();
	sendCounts_.assign(static_cast<uint64_t>(N_ranks) * PARTICLE_REDISTRIBUTION_LANE_COUNT, 0);
	recvCounts_.clear();
}

bool ParticleRedistributionMgr::RouteOwnedFluidParticles(
	const Sync::FluidConstView& fluidParticles,
	const DistParticleState& fluidParticleState,
	const IDomainDecomp& domainDecomp,
	const int rank, const int N_ranks,
	std::string* failMsg)
{
	const std::vector<uint32_t>& ownedLocalIdxs =
		fluidParticleState.particleLocalIdx_Owned;
	const uint32_t N_owned = static_cast<uint32_t>(ownedLocalIdxs.size());
	const uint32_t N_ranks_U32 = static_cast<uint32_t>(N_ranks);
	const uint32_t N_buckets = 2 * N_ranks_U32;
	const uint32_t N_chunks = N_owned > 0
		? 1 + (N_owned - 1) / PARTICLE_ROUTING_CHUNK_SIZE
		: 0;
	const uint64_t N_chunkBuckets = static_cast<uint64_t>(N_chunks) * N_buckets;

	AssignVectorWithExtraCapacity(
		routingWorkBuffers_.routeCounts_ByChunkBucket,
		N_chunkBuckets,
		uint32_t{ 0 },
		scratchBufferExtraAllocCapacityPct_);
	AssignVectorWithExtraCapacity(
		routingWorkBuffers_.routeWriteOffsets_ByChunkBucket,
		N_chunkBuckets,
		uint32_t{ 0 },
		scratchBufferExtraAllocCapacityPct_);
	AssignVectorWithExtraCapacity(
		routingWorkBuffers_.routeTotals,
		N_buckets,
		uint32_t{ 0 },
		scratchBufferExtraAllocCapacityPct_);
	AssignVectorWithExtraCapacity(
		routingWorkBuffers_.ownerRank_ByOwnedIdx,
		N_owned,
		-1,
		scratchBufferExtraAllocCapacityPct_);

	// kept for the retained particle classification, so it outlives this routing pass
	// and is indexed by local particle index instead of owned list position
	auto& haloNeighborMask_ByLocalIdx = routes_.fluidHaloNeighborMask_ByLocalIdx;
	AssignVectorWithExtraCapacity(
		haloNeighborMask_ByLocalIdx,
		fluidParticles.N_,
		uint32_t{ 0 },
		scratchBufferExtraAllocCapacityPct_);

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
				routingWorkBuffers_.routeCounts_ByChunkBucket.data() +
				static_cast<uint64_t>(chunkIdx) * N_buckets;

			for (uint32_t ownedIdx = beginOwnedIdx; ownedIdx != endOwnedIdx; ownedIdx++)
			{
				const uint32_t localIdx = ownedLocalIdxs[ownedIdx];

				#if defined(_DEBUG)
				if (localIdx >= fluidParticles.N_ ||
					!fluidParticleState.HasMetadataAtLocalIdx(localIdx) ||
					(fluidParticleState.particleClasses[localIdx] != DistParticleClass::Inner &&
					 fluidParticleState.particleClasses[localIdx] != DistParticleClass::Outer))
				{
					routeFailMsg.store(
						"Invalid owned fluid particle metadata.", std::memory_order_relaxed);
					continue;
				}
				#endif

				const glm::vec3 position = FluidPositionAtLocalIdx(fluidParticles, localIdx);
				const int ownerRank = domainDecomp.OwnerRank(position);
				if (ownerRank < 0 || ownerRank >= N_ranks)
				{
					routeFailMsg.store(
						"Fluid particle owner rank is out of range.", std::memory_order_relaxed);
					continue;
				}

				routingWorkBuffers_.ownerRank_ByOwnedIdx[ownedIdx] = ownerRank;
				chunkCounts[static_cast<uint32_t>(ownerRank)]++;

				const std::vector<int>& neighborRanks = domainDecomp.NeighborRanks(ownerRank);
				const uint32_t N_neighborRanks =
					static_cast<uint32_t>(neighborRanks.size());
				uint32_t haloNeighborMask = 0;
				for (uint32_t neighborIdx = 0; neighborIdx != N_neighborRanks; neighborIdx++)
				{
					const int haloRank = neighborRanks[neighborIdx];

					#if defined(_DEBUG)
					if (haloRank < 0 || haloRank >= N_ranks || haloRank == ownerRank)
					{
						routeFailMsg.store(
							"Fluid halo destination rank is invalid.", std::memory_order_relaxed);
						continue;
					}
					#endif

					if (!domainDecomp.HaloExtentAABB(haloRank).Contains(position)) { continue; }

					haloNeighborMask |= (1u << neighborIdx);
					chunkCounts[N_ranks_U32 + static_cast<uint32_t>(haloRank)]++;
				}

				haloNeighborMask_ByLocalIdx[localIdx] = haloNeighborMask;
			}
		}
	});

	if (const char* const routeFail = routeFailMsg.load(std::memory_order_relaxed))
	{
		if (failMsg) { *failMsg = routeFail; }
		return false;
	}

	PrefixChunkCountsPerBucket(
		routingWorkBuffers_.routeCounts_ByChunkBucket,
		routingWorkBuffers_.routeWriteOffsets_ByChunkBucket,
		routingWorkBuffers_.routeTotals,
		N_chunks,
		N_buckets);

	ResizeVectorWithExtraCapacity(
		routes_.keptOwnedFluidLocalIdxs,
		routingWorkBuffers_.routeTotals[static_cast<uint32_t>(rank)],
		scratchBufferExtraAllocCapacityPct_);
	ResizeVectorWithExtraCapacity(
		routes_.localHaloFluidLocalIdxs,
		routingWorkBuffers_.routeTotals[
			N_ranks_U32 + static_cast<uint32_t>(rank)],
		scratchBufferExtraAllocCapacityPct_);

	for (int dstRank = 0; dstRank != N_ranks; dstRank++)
	{
		if (dstRank == rank) { continue; }

		ResizeVectorWithExtraCapacity(
			routes_.ownedFluidLocalIdxs_ByDstRank[dstRank],
			routingWorkBuffers_.routeTotals[static_cast<uint32_t>(dstRank)],
			scratchBufferExtraAllocCapacityPct_);
		ResizeVectorWithExtraCapacity(
			routes_.haloFluidLocalIdxs_ByDstRank[dstRank],
			routingWorkBuffers_.routeTotals[
				N_ranks_U32 + static_cast<uint32_t>(dstRank)],
			scratchBufferExtraAllocCapacityPct_);
	}

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
				routingWorkBuffers_.routeWriteOffsets_ByChunkBucket.data() +
				static_cast<uint64_t>(chunkIdx) * N_buckets;

			for (uint32_t ownedIdx = beginOwnedIdx; ownedIdx != endOwnedIdx; ownedIdx++)
			{
				const int ownerRank = routingWorkBuffers_.ownerRank_ByOwnedIdx[ownedIdx];
				if (ownerRank < 0) { continue; }

				const uint32_t localIdx = ownedLocalIdxs[ownedIdx];
				std::vector<uint32_t>& ownedDstLocalIdxs = (ownerRank == rank)
					? routes_.keptOwnedFluidLocalIdxs
					: routes_.ownedFluidLocalIdxs_ByDstRank[ownerRank];
				ownedDstLocalIdxs[chunkWriteIdxs[static_cast<uint32_t>(ownerRank)]++] = localIdx;

				const uint32_t haloNeighborMask = haloNeighborMask_ByLocalIdx[localIdx];
				if (haloNeighborMask == 0) { continue; }

				const std::vector<int>& neighborRanks = domainDecomp.NeighborRanks(ownerRank);
				const uint32_t N_neighborRanks = static_cast<uint32_t>(neighborRanks.size());
				for (uint32_t neighborIdx = 0; neighborIdx != N_neighborRanks; neighborIdx++)
				{
					if ((haloNeighborMask & (1u << neighborIdx)) == 0) { continue; }

					const int haloRank = neighborRanks[neighborIdx];
					std::vector<uint32_t>& haloDstLocalIdxs = (haloRank == rank)
						? routes_.localHaloFluidLocalIdxs
						: routes_.haloFluidLocalIdxs_ByDstRank[haloRank];
					haloDstLocalIdxs[
						chunkWriteIdxs[N_ranks_U32 + static_cast<uint32_t>(haloRank)]++] = localIdx;
				}
			}
		}
	});

	return true;
}

bool ParticleRedistributionMgr::RouteOwnedBoundaryParticles(
	const Sync::BoundaryConstView& boundaryParticles,
	const DistParticleState& boundaryParticleState,
	const IDomainDecomp& domainDecomp,
	const int rank, const int N_ranks,
	std::string* failMsg)
{
	const std::vector<uint32_t>& ownedLocalIdxs =
		boundaryParticleState.particleLocalIdx_Owned;
	const uint32_t N_owned = static_cast<uint32_t>(ownedLocalIdxs.size());
	const uint32_t N_ranks_U32 = static_cast<uint32_t>(N_ranks);
	const uint32_t N_buckets = 2 * N_ranks_U32;
	const uint32_t N_chunks = N_owned > 0
		? 1 + (N_owned - 1) / PARTICLE_ROUTING_CHUNK_SIZE
		: 0;
	const uint64_t N_chunkBuckets = static_cast<uint64_t>(N_chunks) * N_buckets;

	AssignVectorWithExtraCapacity(
		routingWorkBuffers_.routeCounts_ByChunkBucket,
		N_chunkBuckets,
		uint32_t{ 0 },
		scratchBufferExtraAllocCapacityPct_);
	AssignVectorWithExtraCapacity(
		routingWorkBuffers_.routeWriteOffsets_ByChunkBucket,
		N_chunkBuckets,
		uint32_t{ 0 },
		scratchBufferExtraAllocCapacityPct_);
	AssignVectorWithExtraCapacity(
		routingWorkBuffers_.routeTotals,
		N_buckets,
		uint32_t{ 0 },
		scratchBufferExtraAllocCapacityPct_);
	AssignVectorWithExtraCapacity(
		routingWorkBuffers_.ownerRank_ByOwnedIdx,
		N_owned,
		-1,
		scratchBufferExtraAllocCapacityPct_);

	// kept for the retained particle classification, so it outlives this routing pass
	// and is indexed by local particle index instead of owned list position
	auto& haloNeighborMask_ByLocalIdx = routes_.boundaryHaloNeighborMask_ByLocalIdx;
	AssignVectorWithExtraCapacity(
		haloNeighborMask_ByLocalIdx,
		boundaryParticles.N_,
		uint32_t{ 0 },
		scratchBufferExtraAllocCapacityPct_);

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
				routingWorkBuffers_.routeCounts_ByChunkBucket.data() +
				static_cast<uint64_t>(chunkIdx) * N_buckets;

			for (uint32_t ownedIdx = beginOwnedIdx; ownedIdx != endOwnedIdx; ownedIdx++)
			{
				const uint32_t localIdx = ownedLocalIdxs[ownedIdx];

				#if defined(_DEBUG)
				if (localIdx >= boundaryParticles.N_ ||
					!boundaryParticleState.HasMetadataAtLocalIdx(localIdx) ||
					(boundaryParticleState.particleClasses[localIdx] != DistParticleClass::Inner &&
					 boundaryParticleState.particleClasses[localIdx] != DistParticleClass::Outer))
				{
					routeFailMsg.store(
						"Invalid owned boundary particle metadata.", std::memory_order_relaxed);
					continue;
				}
				#endif

				const glm::vec3 position = BoundaryPositionAtLocalIdx(boundaryParticles, localIdx);
				const int ownerRank = domainDecomp.OwnerRank(position);
				if (ownerRank < 0 || ownerRank >= N_ranks)
				{
					routeFailMsg.store(
						"Boundary particle owner rank is out of range.", std::memory_order_relaxed);
					continue;
				}

				routingWorkBuffers_.ownerRank_ByOwnedIdx[ownedIdx] = ownerRank;
				chunkCounts[static_cast<uint32_t>(ownerRank)]++;

				const std::vector<int>& neighborRanks = domainDecomp.NeighborRanks(ownerRank);
				const uint32_t N_neighborRanks =
					static_cast<uint32_t>(neighborRanks.size());
				uint32_t haloNeighborMask = 0;
				for (uint32_t neighborIdx = 0; neighborIdx != N_neighborRanks; neighborIdx++)
				{
					const int haloRank = neighborRanks[neighborIdx];

					#if defined(_DEBUG)
					if (haloRank < 0 || haloRank >= N_ranks || haloRank == ownerRank)
					{
						routeFailMsg.store(
							"Boundary halo destination rank is invalid.", std::memory_order_relaxed);
						continue;
					}
					#endif

					if (!domainDecomp.HaloExtentAABB(haloRank).Contains(position)) { continue; }

					haloNeighborMask |= (1u << neighborIdx);
					chunkCounts[N_ranks_U32 + static_cast<uint32_t>(haloRank)]++;
				}

				haloNeighborMask_ByLocalIdx[localIdx] = haloNeighborMask;
			}
		}
	});

	if (const char* const routeFail = routeFailMsg.load(std::memory_order_relaxed))
	{
		if (failMsg) { *failMsg = routeFail; }
		return false;
	}

	PrefixChunkCountsPerBucket(
		routingWorkBuffers_.routeCounts_ByChunkBucket,
		routingWorkBuffers_.routeWriteOffsets_ByChunkBucket,
		routingWorkBuffers_.routeTotals,
		N_chunks,
		N_buckets);

	ResizeVectorWithExtraCapacity(
		routes_.keptOwnedBoundaryLocalIdxs,
		routingWorkBuffers_.routeTotals[static_cast<uint32_t>(rank)],
		scratchBufferExtraAllocCapacityPct_);
	ResizeVectorWithExtraCapacity(
		routes_.localHaloBoundaryLocalIdxs,
		routingWorkBuffers_.routeTotals[
			N_ranks_U32 + static_cast<uint32_t>(rank)],
		scratchBufferExtraAllocCapacityPct_);

	for (int dstRank = 0; dstRank != N_ranks; dstRank++)
	{
		if (dstRank == rank) { continue; }

		ResizeVectorWithExtraCapacity(
			routes_.ownedBoundaryLocalIdxs_ByDstRank[dstRank],
			routingWorkBuffers_.routeTotals[static_cast<uint32_t>(dstRank)],
			scratchBufferExtraAllocCapacityPct_);
		ResizeVectorWithExtraCapacity(
			routes_.haloBoundaryLocalIdxs_ByDstRank[dstRank],
			routingWorkBuffers_.routeTotals[
				N_ranks_U32 + static_cast<uint32_t>(dstRank)],
			scratchBufferExtraAllocCapacityPct_);
	}

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
				routingWorkBuffers_.routeWriteOffsets_ByChunkBucket.data() +
				static_cast<uint64_t>(chunkIdx) * N_buckets;

			for (uint32_t ownedIdx = beginOwnedIdx; ownedIdx != endOwnedIdx; ownedIdx++)
			{
				const int ownerRank = routingWorkBuffers_.ownerRank_ByOwnedIdx[ownedIdx];
				if (ownerRank < 0) { continue; }

				const uint32_t localIdx = ownedLocalIdxs[ownedIdx];
				std::vector<uint32_t>& ownedDstLocalIdxs = (ownerRank == rank)
					? routes_.keptOwnedBoundaryLocalIdxs
					: routes_.ownedBoundaryLocalIdxs_ByDstRank[ownerRank];
				ownedDstLocalIdxs[chunkWriteIdxs[static_cast<uint32_t>(ownerRank)]++] = localIdx;

				const uint32_t haloNeighborMask = haloNeighborMask_ByLocalIdx[localIdx];
				if (haloNeighborMask == 0) { continue; }

				const std::vector<int>& neighborRanks = domainDecomp.NeighborRanks(ownerRank);
				const uint32_t N_neighborRanks = static_cast<uint32_t>(neighborRanks.size());
				for (uint32_t neighborIdx = 0; neighborIdx != N_neighborRanks; neighborIdx++)
				{
					if ((haloNeighborMask & (1u << neighborIdx)) == 0) { continue; }

					const int haloRank = neighborRanks[neighborIdx];
					std::vector<uint32_t>& haloDstLocalIdxs = (haloRank == rank)
						? routes_.localHaloBoundaryLocalIdxs
						: routes_.haloBoundaryLocalIdxs_ByDstRank[haloRank];
					haloDstLocalIdxs[
						chunkWriteIdxs[N_ranks_U32 + static_cast<uint32_t>(haloRank)]++] = localIdx;
				}
			}
		}
	});

	return true;
}

bool ParticleRedistributionMgr::RouteOwnedParticles(
	const Sync::FluidConstView& fluidParticles,
	const Sync::BoundaryConstView& boundaryParticles,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	const IDomainDecomp& domainDecomp,
	const int rank, const int N_ranks,
	std::string* failMsg,
	ParticleRedistributionTimingReport* timingReport)
{
	// halo destinations are cached as one bit per neighbor list entry
	for (int otherRank = 0; otherRank != N_ranks; otherRank++)
	{
		if (domainDecomp.NeighborRanks(otherRank).size() > PARTICLE_ROUTING_MAX_NEIGHBOR_COUNT)
		{
			if (failMsg) { *failMsg = "Domain decomp exceeds the supported neighbor rank count."; }
			return false;
		}
	}

	// route fluid migration and halos
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::RouteFluid);

		if (!RouteOwnedFluidParticles(
			fluidParticles, fluidParticleState,
			domainDecomp, rank, N_ranks, failMsg))
		{
			return false;
		}
	}

	// route boundary migration and halos
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::RouteBoundary);

		if (!RouteOwnedBoundaryParticles(
			boundaryParticles, boundaryParticleState,
			domainDecomp, rank, N_ranks, failMsg))
		{
			return false;
		}
	}

	return true;
}

bool ParticleRedistributionMgr::PackSendBuffers(
	const Sync::FluidConstView& fluidParticles,
	const Sync::BoundaryConstView& boundaryParticles,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	const int rank, const int N_ranks,
	std::string* failMsg)
{
	// TODO: use uint32_t counters after proving aggregate traffic cannot overflow?
	uint64_t N_sentOwnedF = 0;
	uint64_t N_sentHaloF = 0;
	uint64_t N_sentOwnedB = 0;
	uint64_t N_sentHaloB = 0;

	// build one compact send buffer per destination rank
	for (int dstRank = 0; dstRank != N_ranks; dstRank++)
	{
		auto& rankBuffers = rankBuffers_[dstRank];
		if (dstRank == rank) { continue; }

		const auto& ownedFluidLocalIdxs =
			routes_.ownedFluidLocalIdxs_ByDstRank[dstRank];
		const auto& haloFluidLocalIdxs =
			routes_.haloFluidLocalIdxs_ByDstRank[dstRank];
		const auto& ownedBoundaryLocalIdxs =
			routes_.ownedBoundaryLocalIdxs_ByDstRank[dstRank];
		const auto& haloBoundaryLocalIdxs =
			routes_.haloBoundaryLocalIdxs_ByDstRank[dstRank];

		// TODO: use explicit SIMD for particle field gathers similarly to sph_solver.cpp?
		rankBuffers.sendOwnedFluidParticles.SetBulkFromParticleView(
			fluidParticles, fluidParticleState, ownedFluidLocalIdxs);
		rankBuffers.sendHaloFluidParticles.SetBulkFromParticleView(
			fluidParticles, fluidParticleState, haloFluidLocalIdxs);
		rankBuffers.sendOwnedBoundaryParticles.SetBulkFromParticleView(
			boundaryParticles, boundaryParticleState, ownedBoundaryLocalIdxs);
		rankBuffers.sendHaloBoundaryParticles.SetBulkFromParticleView(
			boundaryParticles, boundaryParticleState, haloBoundaryLocalIdxs);

		rankBuffers.N_sendOwnedF = rankBuffers.sendOwnedFluidParticles.CheckedCount();
		rankBuffers.N_sendHaloF = rankBuffers.sendHaloFluidParticles.CheckedCount();
		rankBuffers.N_sendOwnedB = rankBuffers.sendOwnedBoundaryParticles.CheckedCount();
		rankBuffers.N_sendHaloB = rankBuffers.sendHaloBoundaryParticles.CheckedCount();

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

		N_sentOwnedF += rankBuffers.N_sendOwnedF;
		N_sentHaloF += rankBuffers.N_sendHaloF;
		N_sentOwnedB += rankBuffers.N_sendOwnedB;
		N_sentHaloB += rankBuffers.N_sendHaloB;
	}

	// store aggregate traffic counts for diagnostics
	return
		CheckedUint32(
			N_sentOwnedF, redistStats_.N_sentOwnedF,
			"Sent owned fluid particle count", failMsg) &&
		CheckedUint32(
			N_sentHaloF, redistStats_.N_sentHaloF,
			"Sent halo fluid particle count", failMsg) &&
		CheckedUint32(
			N_sentOwnedB, redistStats_.N_sentOwnedB,
			"Sent owned boundary particle count", failMsg) &&
		CheckedUint32(
			N_sentHaloB, redistStats_.N_sentHaloB,
			"Sent halo boundary particle count", failMsg);
}

bool ParticleRedistributionMgr::PrepareReceiveBuffers(
	const int rank, const int N_ranks,
	std::string* failMsg)
{
	uint64_t N_recvOwnedF = 0;
	uint64_t N_recvHaloF = 0;
	uint64_t N_recvOwnedB = 0;
	uint64_t N_recvHaloB = 0;

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

		if (rankBuffers.N_recvOwnedF < 0 ||
			rankBuffers.N_recvHaloF < 0 ||
			rankBuffers.N_recvOwnedB < 0 ||
			rankBuffers.N_recvHaloB < 0)
		{
			if (failMsg) { *failMsg = "Received particle count is negative."; }
			return false;
		}

		rankBuffers.recvOwnedFluidParticles.Resize(
			static_cast<uint32_t>(rankBuffers.N_recvOwnedF));
		rankBuffers.recvHaloFluidParticles.Resize(
			static_cast<uint32_t>(rankBuffers.N_recvHaloF));
		rankBuffers.recvOwnedBoundaryParticles.Resize(
			static_cast<uint32_t>(rankBuffers.N_recvOwnedB));
		rankBuffers.recvHaloBoundaryParticles.Resize(
			static_cast<uint32_t>(rankBuffers.N_recvHaloB));

		N_recvOwnedF += rankBuffers.N_recvOwnedF;
		N_recvHaloF += rankBuffers.N_recvHaloF;
		N_recvOwnedB += rankBuffers.N_recvOwnedB;
		N_recvHaloB += rankBuffers.N_recvHaloB;
	}

	// store aggregate traffic counts for diagnostics
	return
		CheckedUint32(
			N_recvOwnedF, redistStats_.N_recvOwnedF,
			"Received owned fluid particle count", failMsg) &&
		CheckedUint32(
			N_recvHaloF, redistStats_.N_recvHaloF,
			"Received halo fluid particle count", failMsg) &&
		CheckedUint32(
			N_recvOwnedB, redistStats_.N_recvOwnedB,
			"Received owned boundary particle count", failMsg) &&
		CheckedUint32(
			N_recvHaloB, redistStats_.N_recvHaloB,
			"Received halo boundary particle count", failMsg);
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
		rankBuffers.recvOwnedFluidParticles.QueueIRecvPayload(
			mpiContext, srcRank, PARTICLE_REDISTRIBUTION_TAG_OWNED_FLUID,
			requests);
		rankBuffers.recvHaloFluidParticles.QueueIRecvPayload(
			mpiContext, srcRank, PARTICLE_REDISTRIBUTION_TAG_HALO_FLUID,
			requests);
		rankBuffers.recvOwnedBoundaryParticles.QueueIRecvPayload(
			mpiContext, srcRank, PARTICLE_REDISTRIBUTION_TAG_OWNED_BOUNDARY,
			requests);
		rankBuffers.recvHaloBoundaryParticles.QueueIRecvPayload(
			mpiContext, srcRank, PARTICLE_REDISTRIBUTION_TAG_HALO_BOUNDARY,
			requests);
	}

	// post sends for all four payload lanes
	for (int dstRank = 0; dstRank != N_ranks; dstRank++)
	{
		if (dstRank == rank) { continue; }

		const auto& rankBuffers = rankBuffers_[dstRank];
		rankBuffers.sendOwnedFluidParticles.QueueISendPayload(
			mpiContext, dstRank, PARTICLE_REDISTRIBUTION_TAG_OWNED_FLUID,
			requests);
		rankBuffers.sendHaloFluidParticles.QueueISendPayload(
			mpiContext, dstRank, PARTICLE_REDISTRIBUTION_TAG_HALO_FLUID,
			requests);
		rankBuffers.sendOwnedBoundaryParticles.QueueISendPayload(
			mpiContext, dstRank, PARTICLE_REDISTRIBUTION_TAG_OWNED_BOUNDARY,
			requests);
		rankBuffers.sendHaloBoundaryParticles.QueueISendPayload(
			mpiContext, dstRank, PARTICLE_REDISTRIBUTION_TAG_HALO_BOUNDARY,
			requests);
	}

	mpiContext.WaitAll(requests);
}

bool ParticleRedistributionMgr::PrepareFluidParticleMutation(
	const Sync::FluidConstView& oldParticles,
	const DistParticleState& oldParticleState,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float kernelRadius,
	const uint32_t N_global,
	std::string* failMsg,
	ParticleRedistributionTimingReport* timingReport)
{
	auto& appendedParticles = appendedFluidParticles_;
	auto& rebuiltParticleState = rebuiltFluidParticleState_;
	uint32_t N_retained = 0;
	uint32_t N_appended = 0;
	uint32_t N_owned = 0;
	uint32_t N_halo = 0;
	uint32_t N_local = 0;

	// compute sizes and initialize append-only fluid storage and metadata
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::FluidPrep);

		const uint64_t N_retained_U64 =
			routes_.keptOwnedFluidLocalIdxs.size();
		const uint64_t N_owned_U64 =
			routes_.keptOwnedFluidLocalIdxs.size() +
			redistStats_.N_recvOwnedF;
		const uint64_t N_halo_U64 =
			routes_.localHaloFluidLocalIdxs.size() +
			redistStats_.N_recvHaloF;
		const uint64_t N_appended_U64 =
			redistStats_.N_recvOwnedF +
			N_halo_U64;

		if (!CheckedUint32(
				N_retained_U64, N_retained,
				"Retained fluid particle count", failMsg) ||
			!CheckedUint32(
				N_appended_U64, N_appended,
				"Appended fluid particle count", failMsg) ||
			!CheckedUint32(
				N_owned_U64, N_owned,
				"Owned fluid particle count", failMsg) ||
			!CheckedUint32(
				N_halo_U64, N_halo,
				"Halo fluid particle count", failMsg) ||
			!CheckedUint32(
				N_owned_U64 + N_halo_U64,
				N_local,
				"Local fluid particle count", failMsg) ||
			!BuildDenseCompactionMap(
				oldParticles.N_,
				routes_.keptOwnedFluidLocalIdxs,
				retainedFluidOldLocalIdx_FromNewLocalIdx_,
				retainedOldLocalIdxFlags_,
				compactionWorkBuffers_,
				failMsg))
		{
			return false;
		}

		EnsureFluidParticleCapacityWithExtra(
			appendedParticles,
			N_appended,
			scratchBufferExtraAllocCapacityPct_);
		appendedParticles.resizeAll(N_appended);
		PrepareRebuiltParticleState(
			rebuiltParticleState,
			N_owned,
			N_halo,
			N_local,
			N_global);
	}

	uint32_t dstLocalIdx = 0;
	uint32_t appendLocalIdx = 0;

	// rebuild metadata for owned particles retained in backend storage
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::KeptFluid);

		if (!ClassifyRetainedOwnedParticles_Parallel(
			rebuiltParticleState,
			retainedClassificationWorkBuffers_,
			retainedFluidOldLocalIdx_FromNewLocalIdx_,
			oldParticleState,
			routes_.fluidHaloNeighborMask_ByLocalIdx,
			N_global,
			scratchBufferExtraAllocCapacityPct_,
			"Duplicate or invalid owned fluid global index.", failMsg))
		{
			return false;
		}

		dstLocalIdx =
			static_cast<uint32_t>(retainedFluidOldLocalIdx_FromNewLocalIdx_.size());
	}

	// append owned particles migrated from other ranks
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::MigratedFluid);

		for (const auto& rankBuffers : rankBuffers_)
		{
			// TODO: parallelize migrated particle copies across receive buffers with TBB?
			// TODO: use explicit SIMD for particle field gathers similarly to sph_solver.cpp?
			const auto& recvParticles = rankBuffers.recvOwnedFluidParticles;
			for (uint32_t bufferIdx = 0; bufferIdx != recvParticles.Size(); bufferIdx++)
			{
				recvParticles.CopyToLocalParticle( appendedParticles, bufferIdx, appendLocalIdx);
				const glm::vec3 position
				{
					appendedParticles.xPos[appendLocalIdx],
					appendedParticles.yPos[appendLocalIdx],
					appendedParticles.zPos[appendLocalIdx]
				};
				const DistParticleClass particleClass =
					domainDecomp.IsInnerOwnedParticle(rank, position, kernelRadius)
						? DistParticleClass::Inner
						: DistParticleClass::Outer;
				if (!AddParticleMetadata(
					rebuiltParticleState, particleClass, N_global,
					recvParticles.GlobalIdxAt(bufferIdx), dstLocalIdx))
				{
					if (failMsg) { *failMsg = "Duplicate or invalid migrated fluid global index."; }
					return false;
				}
				appendLocalIdx++;
				dstLocalIdx++;
			}
		}
	}

	// append local and received fluid halos
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::HaloFluid);

		// halos occupy one contiguous local index range, so every particle knows its
		// destination slot up front and the index list needs no shared append
		const uint32_t N_localHalo =
			static_cast<uint32_t>(routes_.localHaloFluidLocalIdxs.size());
		const uint32_t haloDstLocalIdxBase = dstLocalIdx;
		const uint32_t haloAppendLocalIdxBase = appendLocalIdx;
		rebuiltParticleState.particleLocalIdx_Halo.resize(N_halo);

		// first failure seen by the halo append passes, reported once they complete
		std::atomic<const char*> haloFailMsg{ nullptr };

		// append halos created locally from particles migrating away
		// TODO: use explicit SIMD for particle field gathers similarly to sph_solver.cpp?
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_localHalo),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t haloIdx = range.begin(); haloIdx != range.end(); haloIdx++)
			{
				const uint32_t oldLocalIdx = routes_.localHaloFluidLocalIdxs[haloIdx];

				CopyFluidParticle(
					oldParticles, appendedParticles, oldLocalIdx,
					haloAppendLocalIdxBase + haloIdx);

				if (!AddHaloParticleMetadata(
					rebuiltParticleState, N_global,
					oldParticleState.particleGlobalIdx_FromLocalIdx[oldLocalIdx],
					haloDstLocalIdxBase + haloIdx,
					haloIdx))
				{
					haloFailMsg.store(
						"Duplicate or invalid local fluid halo global index.",
						std::memory_order_relaxed);
				}
			}
		});

		// append halo particles received from other ranks
		uint32_t recvHaloIdxBase = N_localHalo;
		for (const auto& rankBuffers : rankBuffers_)
		{
			const auto& recvParticles = rankBuffers.recvHaloFluidParticles;
			const uint32_t N_recvHalo = recvParticles.Size();
			if (N_recvHalo == 0) { continue; }

			recvParticles.CopyBulkToLocalParticles(
				appendedParticles,
				haloAppendLocalIdxBase + recvHaloIdxBase);

			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0, N_recvHalo),
				[&](const tbb::blocked_range<uint32_t>& range)
			{
				for (uint32_t bufferIdx = range.begin(); bufferIdx != range.end(); bufferIdx++)
				{
					const uint32_t haloIdx = recvHaloIdxBase + bufferIdx;

					if (!AddHaloParticleMetadata(
						rebuiltParticleState, N_global,
						recvParticles.GlobalIdxAt(bufferIdx),
						haloDstLocalIdxBase + haloIdx,
						haloIdx))
					{
						haloFailMsg.store(
							"Duplicate or invalid received fluid halo global index.",
							std::memory_order_relaxed);
					}
				}
			});

			recvHaloIdxBase += N_recvHalo;
		}

		if (const char* const haloFail = haloFailMsg.load(std::memory_order_relaxed))
		{
			if (failMsg) { *failMsg = haloFail; }
			return false;
		}

		appendLocalIdx = haloAppendLocalIdxBase + N_halo;
		dstLocalIdx = haloDstLocalIdxBase + N_halo;
	}

	// validate rebuilt fluid metadata
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::FluidPrep);
		if (appendLocalIdx != N_appended ||
			dstLocalIdx != N_local ||
			retainedFluidOldLocalIdx_FromNewLocalIdx_.size() != N_retained)
		{
			if (failMsg) { *failMsg = "Prepared fluid particle mutation count is inconsistent."; }
			return false;
		}
		return rebuiltParticleState.IsValidLocalLayout(N_local, failMsg);
	}
}

// prepare retained compaction, appended particles, and matching boundary metadata
bool ParticleRedistributionMgr::PrepareBoundaryParticleMutation(
	const Sync::BoundaryConstView& oldParticles,
	const DistParticleState& oldParticleState,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float kernelRadius,
	const uint32_t N_global,
	std::string* failMsg,
	ParticleRedistributionTimingReport* timingReport)
{
	auto& appendedParticles = appendedBoundaryParticles_;
	auto& rebuiltParticleState = rebuiltBoundaryParticleState_;
	uint32_t N_retained = 0;
	uint32_t N_appended = 0;
	uint32_t N_owned = 0;
	uint32_t N_halo = 0;
	uint32_t N_local = 0;

	// compute sizes and initialize append-only boundary storage and metadata
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::BoundaryPrep);

		const uint64_t N_retained_U64 =
			routes_.keptOwnedBoundaryLocalIdxs.size();
		const uint64_t N_owned_U64 =
			routes_.keptOwnedBoundaryLocalIdxs.size() +
			redistStats_.N_recvOwnedB;
		const uint64_t N_halo_U64 =
			routes_.localHaloBoundaryLocalIdxs.size() +
			redistStats_.N_recvHaloB;
		const uint64_t N_appended_U64 =
			redistStats_.N_recvOwnedB +
			N_halo_U64;

		if (!CheckedUint32(
				N_retained_U64, N_retained,
				"Retained boundary particle count", failMsg) ||
			!CheckedUint32(
				N_appended_U64, N_appended,
				"Appended boundary particle count", failMsg) ||
			!CheckedUint32(
				N_owned_U64, N_owned,
				"Owned boundary particle count", failMsg) ||
			!CheckedUint32(
				N_halo_U64, N_halo,
				"Halo boundary particle count", failMsg) ||
			!CheckedUint32(
				N_owned_U64 + N_halo_U64,
				N_local,
				"Local boundary particle count", failMsg) ||
			!BuildDenseCompactionMap(
				oldParticles.N_,
				routes_.keptOwnedBoundaryLocalIdxs,
				retainedBoundaryOldLocalIdx_FromNewLocalIdx_,
				retainedOldLocalIdxFlags_,
				compactionWorkBuffers_,
				failMsg))
		{
			return false;
		}

		EnsureBoundaryParticleCapacityWithExtra(
			appendedParticles,
			N_appended,
			scratchBufferExtraAllocCapacityPct_);
		appendedParticles.resizeAll(N_appended);
		PrepareRebuiltParticleState(
			rebuiltParticleState,
			N_owned,
			N_halo,
			N_local,
			N_global);
	}

	uint32_t dstLocalIdx = 0;
	uint32_t appendLocalIdx = 0;

	// rebuild metadata for owned boundary particles retained in backend storage
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::KeptBoundary);

		if (!ClassifyRetainedOwnedParticles_Parallel(
			rebuiltParticleState,
			retainedClassificationWorkBuffers_,
			retainedBoundaryOldLocalIdx_FromNewLocalIdx_,
			oldParticleState,
			routes_.boundaryHaloNeighborMask_ByLocalIdx,
			N_global,
			scratchBufferExtraAllocCapacityPct_,
			"Duplicate or invalid owned boundary global index.", failMsg))
		{
			return false;
		}

		dstLocalIdx =
			static_cast<uint32_t>(retainedBoundaryOldLocalIdx_FromNewLocalIdx_.size());
	}

	// append owned particles migrated from other ranks
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::MigratedBoundary);

		for (const auto& rankBuffers : rankBuffers_)
		{
			// TODO: parallelize migrated particle copies across receive buffers with TBB?
			// TODO: use explicit SIMD for particle field gathers similarly to sph_solver.cpp?
			const auto& recvParticles = rankBuffers.recvOwnedBoundaryParticles;
			for (uint32_t bufferIdx = 0; bufferIdx != recvParticles.Size(); bufferIdx++)
			{
				recvParticles.CopyToLocalParticle(appendedParticles, bufferIdx, appendLocalIdx);
				const glm::vec3 position
				{
					appendedParticles.xPos[appendLocalIdx],
					appendedParticles.yPos[appendLocalIdx],
					appendedParticles.zPos[appendLocalIdx]
				};
				const DistParticleClass particleClass =
					domainDecomp.IsInnerOwnedParticle(rank, position, kernelRadius)
						? DistParticleClass::Inner
						: DistParticleClass::Outer;
				if (!AddParticleMetadata(
					rebuiltParticleState, particleClass, N_global,
					recvParticles.GlobalIdxAt(bufferIdx), dstLocalIdx))
				{
					if (failMsg) { *failMsg = "Duplicate or invalid migrated boundary global index."; }
					return false;
				}
				appendLocalIdx++;
				dstLocalIdx++;
			}
		}
	}

	// append local and received boundary halos
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::HaloBoundary);

		// halos occupy one contiguous local index range, so every particle knows its
		// destination slot up front and the index list needs no shared append
		const uint32_t N_localHalo =
			static_cast<uint32_t>(routes_.localHaloBoundaryLocalIdxs.size());
		const uint32_t haloDstLocalIdxBase = dstLocalIdx;
		const uint32_t haloAppendLocalIdxBase = appendLocalIdx;
		rebuiltParticleState.particleLocalIdx_Halo.resize(N_halo);

		// first failure seen by the halo append passes, reported once they complete
		std::atomic<const char*> haloFailMsg{ nullptr };

		// append halos created locally from particles migrating away
		// TODO: use explicit SIMD for particle field gathers similarly to sph_solver.cpp?
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_localHalo),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t haloIdx = range.begin(); haloIdx != range.end(); haloIdx++)
			{
				const uint32_t oldLocalIdx = routes_.localHaloBoundaryLocalIdxs[haloIdx];

				CopyBoundaryParticle(
					oldParticles, appendedParticles, oldLocalIdx,
					haloAppendLocalIdxBase + haloIdx);

				if (!AddHaloParticleMetadata(
					rebuiltParticleState, N_global,
					oldParticleState.particleGlobalIdx_FromLocalIdx[oldLocalIdx],
					haloDstLocalIdxBase + haloIdx,
					haloIdx))
				{
					haloFailMsg.store(
						"Duplicate or invalid local boundary halo global index.",
						std::memory_order_relaxed);
				}
			}
		});

		// append halo particles received from other ranks
		uint32_t recvHaloIdxBase = N_localHalo;
		for (const auto& rankBuffers : rankBuffers_)
		{
			const auto& recvParticles = rankBuffers.recvHaloBoundaryParticles;
			const uint32_t N_recvHalo = recvParticles.Size();
			if (N_recvHalo == 0) { continue; }

			recvParticles.CopyBulkToLocalParticles(
				appendedParticles,
				haloAppendLocalIdxBase + recvHaloIdxBase);

			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0, N_recvHalo),
				[&](const tbb::blocked_range<uint32_t>& range)
			{
				for (uint32_t bufferIdx = range.begin(); bufferIdx != range.end(); bufferIdx++)
				{
					const uint32_t haloIdx = recvHaloIdxBase + bufferIdx;

					if (!AddHaloParticleMetadata(
						rebuiltParticleState, N_global,
						recvParticles.GlobalIdxAt(bufferIdx),
						haloDstLocalIdxBase + haloIdx,
						haloIdx))
					{
						haloFailMsg.store(
							"Duplicate or invalid received boundary halo global index.",
							std::memory_order_relaxed);
					}
				}
			});

			recvHaloIdxBase += N_recvHalo;
		}

		if (const char* const haloFail = haloFailMsg.load(std::memory_order_relaxed))
		{
			if (failMsg) { *failMsg = haloFail; }
			return false;
		}

		appendLocalIdx = haloAppendLocalIdxBase + N_halo;
		dstLocalIdx = haloDstLocalIdxBase + N_halo;
	}

	// validate rebuilt boundary metadata
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::BoundaryPrep);
		if (appendLocalIdx != N_appended ||
			dstLocalIdx != N_local ||
			retainedBoundaryOldLocalIdx_FromNewLocalIdx_.size() != N_retained)
		{
			if (failMsg) { *failMsg = "Prepared boundary particle mutation count is inconsistent."; }
			return false;
		}
		return rebuiltParticleState.IsValidLocalLayout(N_local, failMsg);
	}
}

bool ParticleRedistributionMgr::BuildExchangeRoutes(
	const Sync::FluidConstView& fluidParticles,
	const Sync::BoundaryConstView& boundaryParticles,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const int N_ranks,
	std::string* failMsg,
	ParticleRedistributionTimingReport* timingReport)
{
	// validate the rank and backend views before replacing reusable routes
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::RoutePrep);

		if (failMsg) { failMsg->clear(); }
		if (rank < 0 || rank >= N_ranks)
		{
			if (failMsg) { *failMsg = "Particle redistribution rank is out of range."; }
			return false;
		}
		if (!ValidateParticleViews(
			fluidParticles, boundaryParticles,
			fluidParticleState, boundaryParticleState, failMsg))
		{
			return false;
		}

		PrepareBuffers(N_ranks);
	}

	return RouteOwnedParticles(
		fluidParticles, boundaryParticles,
		fluidParticleState, boundaryParticleState,
		domainDecomp, rank, N_ranks, failMsg, timingReport);
}

bool ParticleRedistributionMgr::RedistributeParticles(
	Sim::I_DistributedParticleBackend& distParticleBackend,
	DistParticleState& fluidParticleState,
	DistParticleState& boundaryParticleState,
	const IDomainDecomp& domainDecomp,
	const MpiContext& mpiContext,
	const Sim::SimParams& simParams,
	const uint32_t N_globalF,
	const uint32_t N_globalB,
	std::string* failMsg,
	ParticleRedistributionTimingReport* timingReport)
{
	if (failMsg) { failMsg->clear(); }

	// reset per-step statistics before building rank-local exchange state
	ResetRedistStats();
	const int rank = mpiContext.Rank();
	const int N_ranks = mpiContext.Size();
	Sync::FluidConstView fluidParticles{};
	Sync::BoundaryConstView boundaryParticles{};
	bool localOk = true;

	// time explicit agreement between ranks independently of local work
	const auto AllRanksTrue = [&](
		const bool value,
		const ParticleRedistributionTimingOp timingOp)
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, timingOp);
		return mpiContext.AllTrue(value);
	};

	// build exchange routes directly from backend particle views
	try
	{
		{
			[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
				timingReport, ParticleRedistributionTimingOp::RoutePrep);
			fluidParticles = distParticleBackend.GetFluidParticleView();
			boundaryParticles = distParticleBackend.GetBoundaryParticleView();
		}

		localOk = BuildExchangeRoutes(
			fluidParticles, boundaryParticles,
			fluidParticleState, boundaryParticleState,
			domainDecomp, rank, N_ranks, failMsg, timingReport);

		// pack outgoing migration and halo payloads
		if (localOk)
		{
			[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
				timingReport, ParticleRedistributionTimingOp::Pack);
			localOk = PackSendBuffers(
				fluidParticles, boundaryParticles,
				fluidParticleState, boundaryParticleState,
				rank, N_ranks, failMsg);
		}
	}
	catch (const std::exception& ex)
	{
		if (failMsg) { *failMsg = ex.what(); }
		localOk = false;
	}

	// no rank may enter the count exchange while another rank failed validation
	if (!AllRanksTrue(
		localOk,
		ParticleRedistributionTimingOp::SyncPack)) { return false; }

	// exchange migration and halo particle counts and resize all receive buffers accordingly
	try
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::Exchange);
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
	if (!AllRanksTrue(
		localOk,
		ParticleRedistributionTimingOp::SyncCounts)) { return false; }

	// transfer migration and halo payloads together
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::Exchange);
		ExchangePayloads(rank, N_ranks, mpiContext);
	}

	// prepare retained compaction maps, appended particle storage, and matching metadata
	// (backend mutation starts only after every rank completes this preparation)
	try
	{
		localOk = PrepareFluidParticleMutation(
			fluidParticles, fluidParticleState,
			domainDecomp, rank, simParams.kernelRadius,
			N_globalF, failMsg, timingReport);
		if (localOk)
		{
			localOk = PrepareBoundaryParticleMutation(
				boundaryParticles, boundaryParticleState,
				domainDecomp, rank, simParams.kernelRadius,
				N_globalB, failMsg, timingReport);
		}
	}
	catch (const std::exception& ex)
	{
		if (failMsg) { *failMsg = ex.what(); }
		localOk = false;
	}
	if (!AllRanksTrue(
		localOk,
		ParticleRedistributionTimingOp::SyncRebuild)) { return false; }

	// compact retained particles and append migrated and halo particles in the backend
	try
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::Backend);
		localOk = distParticleBackend.ApplyParticleRedistribution(
			retainedFluidOldLocalIdx_FromNewLocalIdx_,
			retainedBoundaryOldLocalIdx_FromNewLocalIdx_,
			appendedFluidParticles_,
			appendedBoundaryParticles_,
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
	if (!AllRanksTrue(
		localOk,
		ParticleRedistributionTimingOp::SyncBackend)) { return false; }

	// publish matching metadata only after every backend mutation succeeded
	std::swap(fluidParticleState, rebuiltFluidParticleState_);
	std::swap(boundaryParticleState, rebuiltBoundaryParticleState_);

	return true;
}
