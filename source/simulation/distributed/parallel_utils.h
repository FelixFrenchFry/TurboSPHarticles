#pragma once

// external
#include <cstdint>
#include <vector>



namespace Dist
{
	// chunks prefixed per block to keep parallel work tied to the chunk count
	// TODO: fine-tune?
	inline constexpr uint32_t PARTICLE_PREFIX_BLOCK_CHUNK_COUNT = 32;
	// minimum block stride, keeping neighboring worker totals on separate cache lines
	// TODO: fine-tune?
	inline constexpr uint32_t PARTICLE_PREFIX_BLOCK_STRIDE_MIN = 16;
	// buckets prefixed per worker, keeping neighboring worker writes on separate cache lines
	// TODO: fine-tune?
	inline constexpr uint32_t PARTICLE_PREFIX_BUCKET_GRAIN_SIZE = 16;

	// prefixes chunk-local bucket counts into disjoint output ranges
	void PrefixCountsByChunkAndBucket(
		const std::vector<uint32_t>& counts_ByChunkBucket,
		std::vector<uint32_t>& writeOffsets_ByChunkBucket,
		std::vector<uint32_t>& totals,
		std::vector<uint32_t>& blockSumsScratch,
		uint32_t N_chunks,
		uint32_t N_buckets);

} // namespace Dist
