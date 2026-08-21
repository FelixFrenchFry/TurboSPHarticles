// parallel_utils.cpp

// external
#include <algorithm>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// internal
#include "parallel_utils.h"



namespace Dist
{

void PrefixCountsByChunkAndBucket(
	const std::vector<uint32_t>& counts_ByChunkBucket,
	std::vector<uint32_t>& writeOffsets_ByChunkBucket,
	std::vector<uint32_t>& totals,
	std::vector<uint32_t>& blockSumsScratch,
	const uint32_t N_chunks,
	const uint32_t N_buckets)
{
	const uint32_t N_blocks = N_chunks > 0
		? 1 + (N_chunks - 1) / PARTICLE_PREFIX_BLOCK_CHUNK_COUNT
		: 0;
	const uint32_t blockStride =
		std::max(N_buckets, PARTICLE_PREFIX_BLOCK_STRIDE_MIN);

	// per block bucket sums, replaced in place by the offset each block starts at
	blockSumsScratch.resize(static_cast<uint64_t>(N_blocks) * blockStride);

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
				blockSumsScratch.data() +
				static_cast<uint64_t>(blockIdx) * blockStride;
			std::fill(blockSums, blockSums + N_buckets, uint32_t{ 0 });

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
	// (buckets processed in parallel, in groups keeping neighboring worker writes on separate cache lines)
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_buckets, PARTICLE_PREFIX_BUCKET_GRAIN_SIZE),
		[&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t bucketIdx = range.begin(); bucketIdx != range.end(); bucketIdx++)
		{
			uint32_t N_bucket = 0;
			for (uint32_t blockIdx = 0; blockIdx != N_blocks; blockIdx++)
			{
				uint32_t& blockSum = blockSumsScratch[
					static_cast<uint64_t>(blockIdx) * blockStride + bucketIdx];
				const uint32_t N_block = blockSum;

				blockSum = N_bucket;
				N_bucket += N_block;
			}
			totals[bucketIdx] = N_bucket;
		}
	});

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
				blockSumsScratch.data() +
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

} // namespace Dist
