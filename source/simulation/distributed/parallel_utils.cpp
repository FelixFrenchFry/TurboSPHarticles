// parallel_utils.cpp

// external
#include <algorithm>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/partitioner.h>

// internal
#include "parallel_utils.h"



namespace Dist
{

void PrefixCountsByChunkAndBucket_ParallelScan(
	const std::vector<uint32_t>& counts_ByChunkBucket,
	std::vector<uint32_t>& writeOffsets_ByChunkBucket,
	std::vector<uint32_t>& totals,
	const uint32_t N_chunks,
	const uint32_t N_buckets)
{
	std::fill(totals.begin(), totals.begin() + N_buckets, uint32_t{ 0 });
	if (N_chunks == 0u || N_buckets == 0u) { return; }

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_buckets),
		[&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t bucketIdx = range.begin(); bucketIdx != range.end(); bucketIdx++)
		{
			totals[bucketIdx] = tbb::parallel_scan(
				tbb::blocked_range<uint32_t>(0, N_chunks),
				uint32_t{ 0 },
				[&](const tbb::blocked_range<uint32_t>& scanRange,
					const uint32_t prefix,
					const bool isFinalScan)
				{
					uint32_t N_bucket = prefix;
					for (uint32_t chunkIdx = scanRange.begin();
						chunkIdx != scanRange.end();
						chunkIdx++)
					{
						const uint64_t chunkBucketIdx =
							static_cast<uint64_t>(chunkIdx) * N_buckets + bucketIdx;

						if (isFinalScan)
						{
							writeOffsets_ByChunkBucket[chunkBucketIdx] = N_bucket;
						}
						N_bucket += counts_ByChunkBucket[chunkBucketIdx];
					}
					return N_bucket;
				},
				[](const uint32_t lhs, const uint32_t rhs)
				{
					return lhs + rhs;
				},
				tbb::auto_partitioner{}
			);
		}
	},
	tbb::auto_partitioner{}
	);
}

} // namespace Dist
