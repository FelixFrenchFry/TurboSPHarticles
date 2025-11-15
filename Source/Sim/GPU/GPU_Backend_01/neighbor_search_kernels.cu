// neighbor_search_kernels.cu

// external

// internal
#include "Source/Sim/GPU/gpu_utils.h"
#include "neighbor_search_kernels.cuh"
#include "neighbor_search.cuh"



namespace GPU01
{
	__constant__ float c_kernelRadius;
	__constant__ float c_kernelRadiusSqu;
	__constant__ float c_cellSize;
	__constant__ float c_cellSizeInv;
	__constant__ float c_xBoxOrig;
	__constant__ float c_yBoxOrig;
	__constant__ float c_zBoxOrig;
	__constant__ uint32_t c_N_DimX;
	__constant__ uint32_t c_N_DimY;
	__constant__ uint32_t c_N_DimZ;
	__constant__ uint32_t c_N_Cells;
	__constant__ uint32_t c_EMPTY;

	inline void Sim::NS_CellLinkedList_Basic_GPU01::ReloadConstants(
		float kernelRadius, float cellSize, float xBoxOrig, float yBoxOrig, float zBoxOrig,
		uint32_t N_DimX, uint32_t N_DimY, uint32_t N_DimZ, uint32_t N_Cells,
		uint32_t EMPTY, cudaStream_t stream)
	{
		float kernelRadiusSqu = kernelRadius * kernelRadius;
		float cellSizeInv = (cellSize == 0.0f) ? 0.0f : (1.0f / cellSize);

		// store values in constant memory on the GPU
		GPU::Check(cudaMemcpyToSymbol(c_kernelRadius, &kernelRadius, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_kernelRadiusSqu, &kernelRadiusSqu, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_cellSize, &cellSize, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_cellSizeInv, &cellSizeInv, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_xBoxOrig, &xBoxOrig, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_yBoxOrig, &yBoxOrig, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_zBoxOrig, &zBoxOrig, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_N_DimX, &N_DimX, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_N_DimY, &N_DimY, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_N_DimZ, &N_DimZ, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_N_Cells, &N_Cells, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_EMPTY, &EMPTY, sizeof(uint32_t)));

		GPU::StreamSync(stream);
	}

	// -------------------------------------------------------------------------
	// --- [ helpers ] ---------------------------------------------------------
	// -------------------------------------------------------------------------

	__device__ __forceinline__
	uint32_t MapToCellId_Basic(uint32_t cx, uint32_t cy, uint32_t cz)
	{
		return cx + c_N_DimX * (cy + c_N_DimY * cz);
	}

	__device__ __forceinline__
	int FloorToInt(float v)
	{
		return static_cast<int>(floorf(v));
	}

	__device__ __forceinline__
	uint32_t ClampInt(int v, int low, int high)
	{
		const int t = (v < low) ? low : v;
		return static_cast<uint32_t>((t > high) ? high : t);
	}

	// -------------------------------------------------------------------------
	// --- [ kernels ] ---------------------------------------------------------
	// -------------------------------------------------------------------------

	__global__
	void K_FillCellIdAndIdxF(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const uint32_t N_F,
		// destinations
		uint32_t* __restrict__ cellIdF)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_F) { return; }

		// particle i data
		// (global memory reads)
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];

		// cell coordinates of particle i
		const int cxi = FloorToInt((xi - c_xBoxOrig) * c_cellSizeInv);
		const int cyi = FloorToInt((yi - c_yBoxOrig) * c_cellSizeInv);
		const int czi = FloorToInt((zi - c_zBoxOrig) * c_cellSizeInv);

		const uint32_t cx = ClampInt(cxi, 0, c_N_DimX - 1);
		const uint32_t cy = ClampInt(cyi, 0, c_N_DimY - 1);
		const uint32_t cz = ClampInt(czi, 0, c_N_DimZ - 1);

		cellIdF[i] = MapToCellId_Basic(static_cast<uint32_t>(cx),
									   static_cast<uint32_t>(cy),
									   static_cast<uint32_t>(cz));
	}

	__global__ void
	K_FillCellIdAndIdxB(
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const uint32_t N_B,
		// destinations
		uint32_t* __restrict__ cellIdB)
	{
		uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_B) { return; }

		// particle i data
		// (global memory reads)
		const float xi = xPosB[i];
		const float yi = yPosB[i];
		const float zi = zPosB[i];

		// compute cell coordinates
		const int cxi = FloorToInt((xi - c_xBoxOrig) * c_cellSizeInv);
		const int cyi = FloorToInt((yi - c_yBoxOrig) * c_cellSizeInv);
		const int czi = FloorToInt((zi - c_zBoxOrig) * c_cellSizeInv);

		const uint32_t cx = ClampInt(cxi, 0, c_N_DimX - 1);
		const uint32_t cy = ClampInt(cyi, 0, c_N_DimY - 1);
		const uint32_t cz = ClampInt(czi, 0, c_N_DimZ - 1);

		cellIdB[i] = MapToCellId_Basic(static_cast<uint32_t>(cx),
									   static_cast<uint32_t>(cy),
									   static_cast<uint32_t>(cz));
	}

	__global__
	void K_BuildRangesFromSortedKeys(
		const uint32_t* __restrict__ cellId,
		const uint32_t N,
		// destinations
		uint32_t* __restrict__ cellStartIdx,
		uint32_t* __restrict__ cellEndIdx)
	{
		uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N) { return; }

		uint32_t key = cellId[i];

		// first element or boundary start
		if (i == 0 || key != cellId[i - 1])
		{
			cellStartIdx[key] = i;

			if (i > 0) // end of previous run
			{
				uint32_t prev = cellId[i - 1];
				cellEndIdx[prev] = i;
			}
		}

		// last element -> close current run
		if (i == N - 1)
		{
			cellEndIdx[key] = N;
		}
	}

	__global__
	void K_ClearRanges(
		const uint32_t N_Cells,
		// destinations
		uint32_t* __restrict__ cellStartIdx,
		uint32_t* __restrict__ cellEndIdx)
	{
		uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_Cells) { return; }

		cellStartIdx[i] = c_EMPTY;
		cellEndIdx[i] = c_EMPTY;
	}

	__global__
	void K_QueryNeighbors(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const uint32_t* __restrict__ cellStartIdxF,
		const uint32_t* __restrict__ cellEndIdxF,
		const uint32_t* __restrict__ cellStartIdxB,
		const uint32_t* __restrict__ cellEndIdxB,
		const uint32_t* __restrict__ sortedParticleIdxF,
		const uint32_t* __restrict__ sortedParticleIdxB,
		const uint32_t N_F,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		// destinations
		uint32_t* __restrict__ neighCountFF,
		uint32_t* __restrict__ neighCountFB,
		uint32_t* __restrict__ neighListFF,
		uint32_t* __restrict__ neighListFB,
		uint32_t* __restrict__ overflowFlag)
	{
		uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_F) { return; }

		// particle i data
		// (global memory reads)
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];

		// cell coordinates of particle i
		const int cxi = static_cast<int>(floorf((xi - c_xBoxOrig) * c_cellSizeInv));
		const int cyi = static_cast<int>(floorf((yi - c_yBoxOrig) * c_cellSizeInv));
		const int czi = static_cast<int>(floorf((zi - c_zBoxOrig) * c_cellSizeInv));

		// TODO: use helper instead?
		/*
		const int cxi = FloorToInt((xi - c_xBoxOrig) * c_cellSizeInv);
		const int cyi = FloorToInt((yi - c_yBoxOrig) * c_cellSizeInv);
		const int czi = FloorToInt((zi - c_zBoxOrig) * c_cellSizeInv);
		*/

		// neighbor write bases
		const uint32_t baseFF = i * N_MAX_FF;
		const uint32_t baseFB = i * N_MAX_FB;
		uint32_t cntFF = 0;
		uint32_t cntFB = 0;

		// check all particles j in 27 neighboring grid cells
		for (int rz = -1; rz <= 1; rz++)
		{
			int cz = czi + rz; // target cell coordinate in z-direction
			if (cz < 0 || cz >= static_cast<int>(c_N_DimZ)) { continue; }

			for (int ry = -1; ry <= 1; ry++)
			{
				int cy = cyi + ry; // target cell coordinate in y-direction
				if (cy < 0 || cy >= static_cast<int>(c_N_DimY)) { continue; }

				for (int rx = -1; rx <= 1; rx++)
				{
					int cx = cxi + rx; // target cell coordinate in x-direction
					if (cx < 0 || cx >= static_cast<int>(c_N_DimX)) { continue; }

					// cell ID
					const uint32_t cid = MapToCellId_Basic(static_cast<uint32_t>(cx),
														   static_cast<uint32_t>(cy),
														   static_cast<uint32_t>(cz));

					// check fluid candidates
					const uint32_t sF = cellStartIdxF[cid];
					const uint32_t eF = cellEndIdxF[cid];
					if (sF != c_EMPTY && eF != c_EMPTY)
					{
						// check all neighbor candidates in the cell
						for (uint32_t k = sF; k < eF; k++)
						{
							// index of neighbor candidate
							const uint32_t j = sortedParticleIdxF[k];

							// position difference
							const float dx = xi - xPosF[j];
							const float dy = yi - yPosF[j];
							const float dz = zi - zPosF[j];

							// squared distance between particles i and j
							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= c_kernelRadiusSqu) // neighbor found
							{
								if (cntFF < N_MAX_FF)
								{
									// add fluid neighbor to the list
									// (global memory write)
									neighListFF[baseFF + cntFF] = j;
									cntFF++;
								}
								else // overflow
								{
									atomicExch(overflowFlag, 1u);
								}
							}
						}
					}

					// check boundary candidates
					const uint32_t sB = cellStartIdxB[cid];
					const uint32_t eB = cellEndIdxB[cid];
					if (sB != c_EMPTY && eB != c_EMPTY)
					{
						// check all neighbor candidates in the cell
						for (uint32_t k = sB; k < eB; k++)
						{
							// index of neighbor candidate
							const uint32_t j = sortedParticleIdxB[k];

							// position difference
							const float dx = xi - xPosB[j];
							const float dy = yi - yPosB[j];
							const float dz = zi - zPosB[j];

							// squared distance between particles i and j
							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= c_kernelRadiusSqu) // neighbor found
							{
								if (cntFB < N_MAX_FB)
								{
									// add boundary neighbor to the list
									// (global memory write)
									neighListFB[baseFB + cntFB] = j;
									cntFB++;
								}
								else // overflow
								{
									atomicExch(overflowFlag, 1u);
								}
							}
						}
					}
				}
			}
		}

		// update neighbor counts
		// (global memory write)
		neighCountFF[i] = cntFF;
		neighCountFB[i] = cntFB;
	}

} // namespace GPU01

// -----------------------------------------------------------------------------
// --- [ kernel launchers ] ----------------------------------------------------
// -----------------------------------------------------------------------------

namespace GPU01
{
	void Launch_FillCellIdAndIdxF(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const uint32_t N_F,
		cudaStream_t stream,
		// destinations
		uint32_t* cellIdF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_FillCellIdAndIdxF<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, N_F, cellIdF);

		GPU::CheckLastKernel("K_FillCellIdAndIdxF()");
	}

	void Launch_FillCellIdAndIdxB(
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		uint32_t* cellIdB)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_B + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_FillCellIdAndIdxB<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosB, yPosB, zPosB, N_B, cellIdB);

		GPU::CheckLastKernel("K_FillCellIdAndIdxB()");
	}

	void Launch_BuildRangesFromSortedKeys(
		const uint32_t* cellId,
		const uint32_t N,
		cudaStream_t stream,
		// destinations
		uint32_t* cellStartIdx, uint32_t* cellEndIdx)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_BuildRangesFromSortedKeys<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			cellId, N, cellStartIdx, cellEndIdx);

		GPU::CheckLastKernel("K_BuildRangesFromSortedKeys()");
	}

	void Launch_ClearRanges(
		const uint32_t N_Cells,
		cudaStream_t stream,
		// destinations
		uint32_t* cellStartIdx, uint32_t* cellEndIdx)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_Cells + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_ClearRanges<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			N_Cells, cellStartIdx, cellEndIdx);

		GPU::CheckLastKernel("K_ClearRanges()");
	}

	void Launch_QueryNeighbors(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* cellStartIdxF, const uint32_t* cellEndIdxF,
		const uint32_t* cellStartIdxB, const uint32_t* cellEndIdxB,
		const uint32_t* sortedParticleIdxF, const uint32_t* sortedParticleIdxB,
		const uint32_t N_F, const uint32_t N_B,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		cudaStream_t stream,
		// destinations
		uint32_t* neighCountFF, uint32_t* neighCountFB,
		uint32_t* neighListFF, uint32_t* neighListFB,
		uint32_t* overflowFlag)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_QueryNeighbors<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, cellStartIdxF, cellEndIdxF,
			cellStartIdxB, cellEndIdxB, sortedParticleIdxF, sortedParticleIdxB,
			N_F, N_MAX_FF, N_MAX_FB,
			neighCountFF, neighCountFB, neighListFF, neighListFB, overflowFlag);

		GPU::CheckLastKernel("K_QueryNeighbors()");
	}
	
} // namespace GPU01
