// statistics_kernels.cuh

#pragma once

// external
#include <cuda_runtime.h>
#include <cstdint>



namespace GPU01
{
	static constexpr uint32_t NbrCountHistogramBinCount = 65536u;
	static constexpr uint32_t NbrCountHistogramChunkSize = 256u;
	static constexpr uint32_t NbrCountHistogramChunkCount =
		NbrCountHistogramBinCount / NbrCountHistogramChunkSize;

	struct NeighborCountStats
	{
		uint32_t p95FF = 0;
		uint32_t p99FF = 0;
		uint32_t minFF = 0;
		uint32_t maxFF = 0;

		uint32_t p95FB = 0;
		uint32_t p99FB = 0;
		uint32_t minFB = 0;
		uint32_t maxFB = 0;
	};

	void Launch_ComputeNeighborCountStats(
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		uint32_t N_F,
		cudaStream_t stream,
		// scratch
		uint32_t* nbrCountHistFF,
		uint32_t* nbrCountHistFB,
		uint32_t* nbrCountHistChunkFF,
		uint32_t* nbrCountHistChunkFB,
		// destination
		NeighborCountStats* stats);

} // namespace GPU01
