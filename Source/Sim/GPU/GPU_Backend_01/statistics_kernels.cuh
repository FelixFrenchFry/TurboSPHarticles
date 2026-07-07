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

	struct DensityStats
	{
		float sumRelErrRaw = 0.0f;
		float sumRelErrPos = 0.0f;
		float sumRelErrAbs = 0.0f;
	};

	struct VelocityStats
	{
		float velMagSum = 0.0f;
		float velMagMin = 0.0f;
		float velMagMax = 0.0f;
	};

	static constexpr uint32_t StatisticsReductionBlockSize = 256u;

	inline uint32_t StatisticsReductionBlockCount(const uint32_t N_F)
	{
		return (N_F + StatisticsReductionBlockSize - 1u) / StatisticsReductionBlockSize;
	}

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

	void Launch_ComputeDensityStats(
		const float* densityF,
		float restDensity,
		uint32_t N_F,
		cudaStream_t stream,
		// scratch
		DensityStats* partialStats,
		// destination
		DensityStats* stats);

	void Launch_ComputeVelocityStats(
		const float* xVelF,
		const float* yVelF,
		const float* zVelF,
		uint32_t N_F,
		cudaStream_t stream,
		// scratch
		VelocityStats* partialStats,
		// destination
		VelocityStats* stats);

} // namespace GPU01
