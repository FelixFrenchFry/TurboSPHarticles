// statistics_kernels.cu

// external
#include <cuda_runtime.h>

// internal
#include "statistics_kernels.cuh"
#include "Source/Sim/GPU/gpu_utils.h"



namespace GPU01
{
	__global__
	void K_BuildNeighborCountHistograms(
		const uint16_t* __restrict__ nbrCountFF,
		const uint16_t* __restrict__ nbrCountFB,
		const uint32_t N_F,
		uint32_t* __restrict__ nbrCountHistFF,
		uint32_t* __restrict__ nbrCountHistFB)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_F) { return; }

		atomicAdd(&nbrCountHistFF[nbrCountFF[i]], 1u);
		atomicAdd(&nbrCountHistFB[nbrCountFB[i]], 1u);
	}

	__device__
	uint32_t FindPercentileInChunk(
		const uint32_t* __restrict__ hist,
		const uint32_t chunkIdx,
		const uint64_t targetIdx,
		uint64_t countBefore)
	{
		const uint32_t begin = chunkIdx * NbrCountHistogramChunkSize;
		const uint32_t end = begin + NbrCountHistogramChunkSize;

		for (uint32_t value = begin; value != end; value++)
		{
			const uint32_t count = hist[value];
			if (count == 0u) { continue; }

			countBefore += static_cast<uint64_t>(count);
			if (targetIdx < countBefore) { return value; }
		}

		return end - 1u;
	}

	__device__
	uint32_t FindMinInChunk(
		const uint32_t* __restrict__ hist,
		const uint32_t chunkIdx)
	{
		const uint32_t begin = chunkIdx * NbrCountHistogramChunkSize;
		const uint32_t end = begin + NbrCountHistogramChunkSize;

		for (uint32_t value = begin; value != end; value++)
		{
			if (hist[value] > 0u) { return value; }
		}

		return 0u;
	}

	__device__
	uint32_t FindMaxInChunk(
		const uint32_t* __restrict__ hist,
		const uint32_t chunkIdx)
	{
		const uint32_t begin = chunkIdx * NbrCountHistogramChunkSize;
		const uint32_t end = begin + NbrCountHistogramChunkSize;

		for (uint32_t value = end; value != begin; value--)
		{
			const uint32_t idx = value - 1u;
			if (hist[idx] > 0u) { return idx; }
		}

		return 0u;
	}

	__device__
	void ComputeStatsFromHistogram(
		const uint32_t* __restrict__ hist,
		const uint32_t* __restrict__ chunkCounts,
		const uint32_t N_F,
		uint32_t& p95,
		uint32_t& p99,
		uint32_t& minValue,
		uint32_t& maxValue)
	{
		const uint64_t i95 = (95ull * static_cast<uint64_t>(N_F - 1u)) / 100ull;
		const uint64_t i99 = (99ull * static_cast<uint64_t>(N_F - 1u)) / 100ull;

		uint64_t countBefore = 0ull;
		bool minWritten = false;
		bool p95Written = false;
		bool p99Written = false;
		uint32_t lastNonEmptyChunk = 0u;

		minValue = 0u;
		maxValue = 0u;
		p95 = 0u;
		p99 = 0u;

		for (uint32_t chunkIdx = 0u; chunkIdx != NbrCountHistogramChunkCount; chunkIdx++)
		{
			const uint32_t count = chunkCounts[chunkIdx];
			if (count == 0u) { continue; }

			if (!minWritten)
			{
				minValue = FindMinInChunk(hist, chunkIdx);
				minWritten = true;
			}
			lastNonEmptyChunk = chunkIdx;

			const uint64_t countAfter =
				countBefore + static_cast<uint64_t>(count);
			if (!p95Written && i95 < countAfter)
			{
				p95 = FindPercentileInChunk(hist, chunkIdx, i95, countBefore);
				p95Written = true;
			}
			if (!p99Written && i99 < countAfter)
			{
				p99 = FindPercentileInChunk(hist, chunkIdx, i99, countBefore);
				p99Written = true;
			}
			countBefore = countAfter;
		}

		if (minWritten)
		{
			maxValue = FindMaxInChunk(hist, lastNonEmptyChunk);
		}
	}

	__global__
	void K_BuildNeighborCountHistogramChunks(
		const uint32_t* __restrict__ nbrCountHistFF,
		const uint32_t* __restrict__ nbrCountHistFB,
		uint32_t* __restrict__ nbrCountHistChunkFF,
		uint32_t* __restrict__ nbrCountHistChunkFB)
	{
		const uint32_t chunkIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (chunkIdx >= NbrCountHistogramChunkCount) { return; }

		const uint32_t begin = chunkIdx * NbrCountHistogramChunkSize;
		const uint32_t end = begin + NbrCountHistogramChunkSize;

		uint32_t countFF = 0u;
		uint32_t countFB = 0u;
		for (uint32_t value = begin; value != end; value++)
		{
			countFF += nbrCountHistFF[value];
			countFB += nbrCountHistFB[value];
		}

		nbrCountHistChunkFF[chunkIdx] = countFF;
		nbrCountHistChunkFB[chunkIdx] = countFB;
	}

	__global__
	void K_ComputeNeighborCountStats(
		const uint32_t* __restrict__ nbrCountHistFF,
		const uint32_t* __restrict__ nbrCountHistFB,
		const uint32_t* __restrict__ nbrCountHistChunkFF,
		const uint32_t* __restrict__ nbrCountHistChunkFB,
		const uint32_t N_F,
		NeighborCountStats* __restrict__ stats)
	{
		if (threadIdx.x != 0u || blockIdx.x != 0u) { return; }

		NeighborCountStats out{};
		if (N_F > 0u)
		{
			// scan exact histograms to obtain percentile indices
			ComputeStatsFromHistogram(
				nbrCountHistFF, nbrCountHistChunkFF, N_F,
				out.p95FF, out.p99FF, out.minFF, out.maxFF);
			ComputeStatsFromHistogram(
				nbrCountHistFB, nbrCountHistChunkFB, N_F,
				out.p95FB, out.p99FB, out.minFB, out.maxFB);
		}

		*stats = out;
	}

} // namespace GPU01

// -------------------------------------------------------------------------------------------------
// --- [ kernel launchers ] ------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

namespace GPU01
{
	void Launch_ComputeNeighborCountStats(
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint32_t N_F,
		cudaStream_t stream,
		// scratch
		uint32_t* nbrCountHistFF,
		uint32_t* nbrCountHistFB,
		uint32_t* nbrCountHistChunkFF,
		uint32_t* nbrCountHistChunkFB,
		// destination
		NeighborCountStats* stats)
	{
		if (N_F == 0u)
		{
			GPU::MemsetZero(stats, 1, stream);
			return;
		}

		GPU::MemsetZero(nbrCountHistFF, NbrCountHistogramBinCount, stream);
		GPU::MemsetZero(nbrCountHistFB, NbrCountHistogramBinCount, stream);

		const uint32_t N_BLOCKSIZE = 256u;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1u) / N_BLOCKSIZE;

		K_BuildNeighborCountHistograms<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			nbrCountFF,
			nbrCountFB,
			N_F,
			nbrCountHistFF,
			nbrCountHistFB);
		GPU::CheckLastKernel("K_BuildNeighborCountHistograms()");

		const uint32_t N_CHUNK_GRIDSIZE =
			(NbrCountHistogramChunkCount + N_BLOCKSIZE - 1u) / N_BLOCKSIZE;

		K_BuildNeighborCountHistogramChunks<<<N_CHUNK_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			nbrCountHistFF,
			nbrCountHistFB,
			nbrCountHistChunkFF,
			nbrCountHistChunkFB);
		GPU::CheckLastKernel("K_BuildNeighborCountHistogramChunks()");

		K_ComputeNeighborCountStats<<<1, 1, 0, stream>>>(
			nbrCountHistFF,
			nbrCountHistFB,
			nbrCountHistChunkFF,
			nbrCountHistChunkFB,
			N_F,
			stats);
		GPU::CheckLastKernel("K_ComputeNeighborCountStats()");
	}

} // namespace GPU01
