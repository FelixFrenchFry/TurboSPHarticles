#pragma once

// external
#include <cstdint>
#include <vector>



namespace Dist
{
	// prefixes each bucket independently with parallel scans
	void PrefixCountsByChunkAndBucket_ParallelScan(
		const std::vector<uint32_t>& counts_ByChunkBucket,
		std::vector<uint32_t>& writeOffsets_ByChunkBucket,
		std::vector<uint32_t>& totals,
		uint32_t N_chunks,
		uint32_t N_buckets);

} // namespace Dist
