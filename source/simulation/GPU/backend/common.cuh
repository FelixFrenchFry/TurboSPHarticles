// common.cuh
//
// TODO

#pragma once

// external
#include <cstdint>
#include <thrust/tuple.h>



namespace GPU
{
	static constexpr uint32_t N_SELL_WIDTH = 32u;

	enum class NbrChunkFormat : uint8_t
	{
		Delta8 = 0u,
		Delta16 = 1u,
		Abs32 = 2u
	};

	// bit-mask written by chunked SELL32 neighbor-list safety checks
	static constexpr uint32_t NbrListError_FillOverflow = 1u << 0u;
	static constexpr uint32_t NbrListError_InvalidChunkFormat = 1u << 1u;
	static constexpr uint32_t NbrListError_DeltaRangeOverflow = 1u << 2u;

	// cell ID of the particles that left the cell frame
	static constexpr uint64_t EMPTY_CELL_ID = 0xffffffffffffffffull;

	// particle index ranges of one merged non-empty cell
	// (a particle class absent from the cell is encoded as begin == end)
	struct alignas(16) CellRangeByAny
	{
		uint32_t beginF, endF;
		uint32_t beginB, endB;
	};
	static_assert(sizeof(CellRangeByAny) == 16u,
		"Merged cell ranges must stay one vectorized load wide.");

	// host + device struct for AABB accumulation
	struct MinMax3HD
	{
		float xMin, yMin, zMin;
		float xMax, yMax, zMax;
	};

	// map (x, y, z) to MinMax3HD
	struct ToMinMax3
	{
		__host__ __device__
		MinMax3HD operator()(const thrust::tuple<float, float, float>& t) const
		{
			const float x = thrust::get<0>(t);
			const float y = thrust::get<1>(t);
			const float z = thrust::get<2>(t);

			return MinMax3HD{ x, y, z, x, y, z };
		}
	};

	// reduce two MinMax3HDs to their combined min/max
	struct ReduceMinMax3
	{
		__host__ __device__
		MinMax3HD operator()(const MinMax3HD& a, const MinMax3HD& b) const
		{
			MinMax3HD r;

			r.xMin = (a.xMin < b.xMin) ? a.xMin : b.xMin;
			r.yMin = (a.yMin < b.yMin) ? a.yMin : b.yMin;
			r.zMin = (a.zMin < b.zMin) ? a.zMin : b.zMin;
			r.xMax = (a.xMax > b.xMax) ? a.xMax : b.xMax;
			r.yMax = (a.yMax > b.yMax) ? a.yMax : b.yMax;
			r.zMax = (a.zMax > b.zMax) ? a.zMax : b.zMax;

			return r;
		}
	};

} // namespace GPU
