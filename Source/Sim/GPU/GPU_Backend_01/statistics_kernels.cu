// statistics_kernels.cu

// external
#include <cuda_runtime.h>
#include <cfloat>

// internal
#include "statistics_kernels.cuh"
#include "Source/Sim/GPU/gpu_utils.h"



namespace GPU01
{
	__global__
	void K_BuildNeighborCountHistogram(
		const uint16_t* __restrict__ nbrCount,
		const uint32_t N,
		uint32_t* __restrict__ nbrCountHist)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N) { return; }

		atomicAdd(&nbrCountHist[nbrCount[i]], 1u);
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
		uint32_t& minValue,
		uint32_t& maxValue,
		uint32_t& p99)
	{
		const uint64_t i99 = (99ull * static_cast<uint64_t>(N_F - 1u)) / 100ull;

		uint64_t countBefore = 0ull;
		bool minWritten = false;
		bool p99Written = false;
		uint32_t lastNonEmptyChunk = 0u;

		minValue = 0u;
		maxValue = 0u;
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
		const uint32_t* __restrict__ nbrCountHist,
		uint32_t* __restrict__ nbrCountHistChunk)
	{
		const uint32_t chunkIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (chunkIdx >= NbrCountHistogramChunkCount) { return; }

		const uint32_t begin = chunkIdx * NbrCountHistogramChunkSize;
		const uint32_t end = begin + NbrCountHistogramChunkSize;

		uint32_t count = 0u;
		for (uint32_t value = begin; value != end; value++)
		{
			count += nbrCountHist[value];
		}

		nbrCountHistChunk[chunkIdx] = count;
	}

	__global__
	void K_ComputeNeighborCountStats(
		const uint32_t* __restrict__ nbrCountHist,
		const uint32_t* __restrict__ nbrCountHistChunk,
		const uint32_t N,
		NeighborCountStats* __restrict__ stats)
	{
		if (threadIdx.x != 0u || blockIdx.x != 0u) { return; }

		NeighborCountStats out{};
		if (N > 0u)
		{
			// scan exact histograms to obtain the p99 index
			ComputeStatsFromHistogram(
				nbrCountHist, nbrCountHistChunk, N,
				out.min, out.max, out.p99);
		}

		*stats = out;
	}

	__global__
	void K_ComputeDensityStatsPartials(
		const float* __restrict__ densityF,
		const float restDensity,
		const uint32_t N_F,
		DensityStats* __restrict__ partialStats)
	{
		const uint32_t tid = threadIdx.x;
		const uint32_t i = blockIdx.x * blockDim.x + tid;

		float sumRelErrRaw = 0.0f;
		float sumRelErrPos = 0.0f;
		float sumRelErrAbs = 0.0f;

		if (i < N_F)
		{
			const float relErr = (densityF[i] / restDensity) - 1.0f;
			sumRelErrRaw = relErr;
			sumRelErrPos = fmaxf(0.0f, relErr);
			sumRelErrAbs = fabsf(relErr);
		}

		extern __shared__ float shared[];
		float* sRaw = shared;
		float* sPos = sRaw + blockDim.x;
		float* sAbs = sPos + blockDim.x;

		sRaw[tid] = sumRelErrRaw;
		sPos[tid] = sumRelErrPos;
		sAbs[tid] = sumRelErrAbs;
		__syncthreads();

		for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
		{
			if (tid < stride)
			{
				sRaw[tid] += sRaw[tid + stride];
				sPos[tid] += sPos[tid + stride];
				sAbs[tid] += sAbs[tid + stride];
			}
			__syncthreads();
		}

		if (tid == 0u)
		{
			DensityStats out{};
			out.sumRelErrRaw = sRaw[0];
			out.sumRelErrPos = sPos[0];
			out.sumRelErrAbs = sAbs[0];

			partialStats[blockIdx.x] = out;
		}
	}

	__global__
	void K_FinalizeDensityStats(
		const DensityStats* __restrict__ partialStats,
		const uint32_t N_partials,
		DensityStats* __restrict__ stats)
	{
		const uint32_t tid = threadIdx.x;

		float sumRelErrRaw = 0.0f;
		float sumRelErrPos = 0.0f;
		float sumRelErrAbs = 0.0f;

		for (uint32_t i = tid; i < N_partials; i += blockDim.x)
		{
			sumRelErrRaw += partialStats[i].sumRelErrRaw;
			sumRelErrPos += partialStats[i].sumRelErrPos;
			sumRelErrAbs += partialStats[i].sumRelErrAbs;
		}

		extern __shared__ float shared[];
		float* sRaw = shared;
		float* sPos = sRaw + blockDim.x;
		float* sAbs = sPos + blockDim.x;

		sRaw[tid] = sumRelErrRaw;
		sPos[tid] = sumRelErrPos;
		sAbs[tid] = sumRelErrAbs;
		__syncthreads();

		for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
		{
			if (tid < stride)
			{
				sRaw[tid] += sRaw[tid + stride];
				sPos[tid] += sPos[tid + stride];
				sAbs[tid] += sAbs[tid + stride];
			}
			__syncthreads();
		}

		if (tid == 0u)
		{
			DensityStats out{};
			out.sumRelErrRaw = sRaw[0];
			out.sumRelErrPos = sPos[0];
			out.sumRelErrAbs = sAbs[0];

			*stats = out;
		}
	}

	__global__
	void K_ComputeVelocityStatsPartials(
		const float* __restrict__ xVelF,
		const float* __restrict__ yVelF,
		const float* __restrict__ zVelF,
		const uint32_t N_F,
		VelocityStats* __restrict__ partialStats)
	{
		const uint32_t tid = threadIdx.x;
		const uint32_t i = blockIdx.x * blockDim.x + tid;

		float velMagSum = 0.0f;
		float velMagMin = FLT_MAX;
		float velMagMax = 0.0f;

		if (i < N_F)
		{
			const float vx = xVelF[i];
			const float vy = yVelF[i];
			const float vz = zVelF[i];
			const float velMag = sqrtf(vx * vx + vy * vy + vz * vz);

			velMagSum = velMag;
			velMagMin = velMag;
			velMagMax = velMag;
		}

		extern __shared__ float shared[];
		float* sSum = shared;
		float* sMin = sSum + blockDim.x;
		float* sMax = sMin + blockDim.x;

		sSum[tid] = velMagSum;
		sMin[tid] = velMagMin;
		sMax[tid] = velMagMax;
		__syncthreads();

		for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
		{
			if (tid < stride)
			{
				sSum[tid] += sSum[tid + stride];
				sMin[tid] = fminf(sMin[tid], sMin[tid + stride]);
				sMax[tid] = fmaxf(sMax[tid], sMax[tid + stride]);
			}
			__syncthreads();
		}

		if (tid == 0u)
		{
			VelocityStats out{};
			out.velMagSum = sSum[0];
			out.velMagMin = sMin[0];
			out.velMagMax = sMax[0];

			partialStats[blockIdx.x] = out;
		}
	}

	__global__
	void K_FinalizeVelocityStats(
		const VelocityStats* __restrict__ partialStats,
		const uint32_t N_partials,
		VelocityStats* __restrict__ stats)
	{
		const uint32_t tid = threadIdx.x;

		float velMagSum = 0.0f;
		float velMagMin = FLT_MAX;
		float velMagMax = 0.0f;

		for (uint32_t i = tid; i < N_partials; i += blockDim.x)
		{
			velMagSum += partialStats[i].velMagSum;
			velMagMin = fminf(velMagMin, partialStats[i].velMagMin);
			velMagMax = fmaxf(velMagMax, partialStats[i].velMagMax);
		}

		extern __shared__ float shared[];
		float* sSum = shared;
		float* sMin = sSum + blockDim.x;
		float* sMax = sMin + blockDim.x;

		sSum[tid] = velMagSum;
		sMin[tid] = velMagMin;
		sMax[tid] = velMagMax;
		__syncthreads();

		for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
		{
			if (tid < stride)
			{
				sSum[tid] += sSum[tid + stride];
				sMin[tid] = fminf(sMin[tid], sMin[tid + stride]);
				sMax[tid] = fmaxf(sMax[tid], sMax[tid + stride]);
			}
			__syncthreads();
		}

		if (tid == 0u)
		{
			VelocityStats out{};
			out.velMagSum = sSum[0];
			out.velMagMin = sMin[0];
			out.velMagMax = sMax[0];

			*stats = out;
		}
	}

} // namespace GPU01

// -------------------------------------------------------------------------------------------------
// --- [ kernel launchers ] ------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

namespace GPU01
{
	void Launch_ComputeNeighborCountStats(
		const uint16_t* nbrCount,
		const uint32_t N,
		cudaStream_t stream,
		// scratch
		uint32_t* nbrCountHist,
		uint32_t* nbrCountHistChunk,
		// destination
		NeighborCountStats* stats)
	{
		if (N == 0u)
		{
			GPU::MemsetZero(stats, 1, stream);
			return;
		}

		GPU::MemsetZero(nbrCountHist, NbrCountHistogramBinCount, stream);

		const uint32_t N_BLOCKSIZE = 256u;
		const uint32_t N_GRIDSIZE = (N + N_BLOCKSIZE - 1u) / N_BLOCKSIZE;

		K_BuildNeighborCountHistogram<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			nbrCount, N, nbrCountHist);
		GPU::CheckLastKernel("K_BuildNeighborCountHistogram()");

		const uint32_t N_CHUNK_GRIDSIZE = (NbrCountHistogramChunkCount + N_BLOCKSIZE - 1u) / N_BLOCKSIZE;

		K_BuildNeighborCountHistogramChunks<<<N_CHUNK_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			nbrCountHist, nbrCountHistChunk);
		GPU::CheckLastKernel("K_BuildNeighborCountHistogramChunks()");

		K_ComputeNeighborCountStats<<<1, 1, 0, stream>>>(
			nbrCountHist, nbrCountHistChunk, N, stats);
		GPU::CheckLastKernel("K_ComputeNeighborCountStats()");
	}

	void Launch_ComputeDensityStats(
		const float* densityF,
		const float restDensity,
		const uint32_t N_F,
		cudaStream_t stream,
		// scratch
		DensityStats* partialStats,
		// destination
		DensityStats* stats)
	{
		if (N_F == 0u)
		{
			GPU::MemsetZero(stats, 1, stream);
			return;
		}

		const uint32_t N_BLOCKSIZE = StatisticsReductionBlockSize;
		const uint32_t N_GRIDSIZE = StatisticsReductionBlockCount(N_F);
		const uint32_t sharedBytes = 3u * N_BLOCKSIZE * static_cast<uint32_t>(sizeof(float));

		// reduce on device and copy only the final stats struct to host
		K_ComputeDensityStatsPartials<<<N_GRIDSIZE, N_BLOCKSIZE, sharedBytes, stream>>>(
			densityF, restDensity, N_F, partialStats);
		GPU::CheckLastKernel("K_ComputeDensityStatsPartials()");

		K_FinalizeDensityStats<<<1, N_BLOCKSIZE, sharedBytes, stream>>>(
			partialStats, N_GRIDSIZE, stats);
		GPU::CheckLastKernel("K_FinalizeDensityStats()");
	}

	void Launch_ComputeVelocityStats(
		const float* xVelF,
		const float* yVelF,
		const float* zVelF,
		const uint32_t N_F,
		cudaStream_t stream,
		// scratch
		VelocityStats* partialStats,
		// destination
		VelocityStats* stats)
	{
		if (N_F == 0u)
		{
			GPU::MemsetZero(stats, 1, stream);
			return;
		}

		const uint32_t N_BLOCKSIZE = StatisticsReductionBlockSize;
		const uint32_t N_GRIDSIZE = StatisticsReductionBlockCount(N_F);
		const uint32_t sharedBytes = 3u * N_BLOCKSIZE * static_cast<uint32_t>(sizeof(float));

		// reduce on device and copy only the final stats struct to host
		K_ComputeVelocityStatsPartials<<<N_GRIDSIZE, N_BLOCKSIZE, sharedBytes, stream>>>(
			xVelF, yVelF, zVelF, N_F, partialStats);
		GPU::CheckLastKernel("K_ComputeVelocityStatsPartials()");

		K_FinalizeVelocityStats<<<1, N_BLOCKSIZE, sharedBytes, stream>>>(
			partialStats, N_GRIDSIZE, stats);
		GPU::CheckLastKernel("K_FinalizeVelocityStats()");
	}

} // namespace GPU01
