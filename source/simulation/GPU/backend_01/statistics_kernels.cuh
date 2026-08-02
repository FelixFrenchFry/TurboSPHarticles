// statistics_kernels.cuh

#pragma once

// external
#include <cuda_runtime.h>
#include <cstdint>



namespace GPU01
{
	static constexpr uint32_t N_NBR_COUNT_HISTOGRAM_BINS = 65536u;
	static constexpr uint32_t N_NBR_COUNT_HISTOGRAM_CHUNK = 256u;
	static constexpr uint32_t N_NBR_COUNT_HISTOGRAM_CHUNKS =
		N_NBR_COUNT_HISTOGRAM_BINS / N_NBR_COUNT_HISTOGRAM_CHUNK;

	struct NeighborCountStats
	{
		uint32_t min = 0;
		uint32_t max = 0;
		uint32_t p99 = 0;
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

	static constexpr uint32_t N_STATISTICS_REDUCTION_BLOCK = 256u;

	inline uint32_t StatisticsReductionBlockCount(const uint32_t N_F)
	{
		return (N_F + N_STATISTICS_REDUCTION_BLOCK - 1u) /
			N_STATISTICS_REDUCTION_BLOCK;
	}

	void Launch_ComputeNeighborCountStats(
		const uint16_t* nbrCount,
		uint32_t N,
		cudaStream_t stream,
		// scratch
		uint32_t* nbrCountHist,
		uint32_t* nbrCountHistChunk,
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
