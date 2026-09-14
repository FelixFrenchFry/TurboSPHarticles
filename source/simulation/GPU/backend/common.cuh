// common.cuh
#pragma once

// external
#include <cstdint>



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

	// particle index ranges of one merged non-empty cell
	// (a particle class absent from the cell is encoded as begin == end)
	struct alignas(16) CellRangeByAny
	{
		uint32_t beginF, endF;
		uint32_t beginB, endB;
	};
	static_assert(sizeof(CellRangeByAny) == 16u,
		"Merged cell ranges must stay one vectorized load wide.");

} // namespace GPU
