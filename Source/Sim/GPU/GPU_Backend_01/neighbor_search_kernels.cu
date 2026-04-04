// neighbor_search_kernels.cu

// external

// internal
#include "Source/Sim/GPU/gpu_utils.h"
#include "neighbor_search_kernels.cuh"
#include "neighbor_search.cuh"



namespace GPU01
{
	static constexpr uint32_t N_SellWidth = 32u;

	__constant__ uint32_t c_ns_N_F;
	__constant__ uint32_t c_ns_N_B;
	__constant__ float c_ns_kernelRadius;
	__constant__ float c_ns_kernelRadiusSqu;
	__constant__ float c_ns_cellSize;
	__constant__ float c_ns_cellSizeInv;
	__constant__ uint32_t c_ns_cellQueryReach;
	__constant__ float c_ns_xBoxOrig;
	__constant__ float c_ns_yBoxOrig;
	__constant__ float c_ns_zBoxOrig;
	__constant__ uint32_t c_ns_N_DimX;
	__constant__ uint32_t c_ns_N_DimY;
	__constant__ uint32_t c_ns_N_DimZ;
	__constant__ uint32_t c_ns_N_Cells;
	__constant__ uint32_t c_ns_EMPTY;

	void NS_ReloadConstants(
		const uint32_t N_F, const uint32_t N_B,
		const float kernelRadius, const float cellSize, const uint32_t cellQueryReach,
		const float xBoxOrig, const float yBoxOrig, const float zBoxOrig,
		const uint32_t N_DimX, const uint32_t N_DimY, const uint32_t N_DimZ,
		const uint32_t N_Cells, const uint32_t EMPTY,
		cudaStream_t stream)
	{
		float kernelRadiusSqu = kernelRadius * kernelRadius;
		float cellSizeInv = (cellSize == 0.0f) ? 0.0f : (1.0f / cellSize);

		// store values in constant memory on the GPU
		GPU::Check(cudaMemcpyToSymbol(c_ns_N_F, &N_F, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_N_B, &N_B, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_kernelRadius, &kernelRadius, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_kernelRadiusSqu, &kernelRadiusSqu, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_cellSize, &cellSize, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_cellSizeInv, &cellSizeInv, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_cellQueryReach, &cellQueryReach, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_xBoxOrig, &xBoxOrig, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_yBoxOrig, &yBoxOrig, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_zBoxOrig, &zBoxOrig, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_N_DimX, &N_DimX, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_N_DimY, &N_DimY, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_N_DimZ, &N_DimZ, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_N_Cells, &N_Cells, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_EMPTY, &EMPTY, sizeof(uint32_t)));

		GPU::StreamSync(stream);
	}

	// ------------------------------------------------------------------------------------------------
	// --- [ helpers ] --------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------

	__device__ __forceinline__
	uint32_t MapToCellId_Basic(uint32_t cx, uint32_t cy, uint32_t cz)
	{
		return cx + c_ns_N_DimX * (cy + c_ns_N_DimY * cz);
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

	__device__ __forceinline__
	void StoreNeighbor_SELL32(
		const uint64_t chunkBase,
		const uint64_t chunkPitch,
		const uint32_t lane,
		uint64_t& wr,
		const uint32_t j,
		uint32_t* __restrict__ nbrList,
		uint32_t* __restrict__ overflowFlag)
	{
		if (wr < chunkPitch)
		{
			nbrList[chunkBase + wr * static_cast<uint64_t>(N_SellWidth) + lane] = j;
			wr++;
		}
		else
		{
			atomicExch(overflowFlag, 1u);
		}
	}

	// ------------------------------------------------------------------------------------------------
	// --- [ kernels launched by host ] ---------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------

	__global__
	void K_BruteForce_Query_Count(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		// destinations
		uint32_t* __restrict__ nbrCountFF,
		uint32_t* __restrict__ nbrCountFB)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_ns_N_F) { return; }

		// particle i data
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];

		uint32_t cntFF = 0;
		uint32_t cntFB = 0;

		// check fluid candidates (all particles)
		for (uint32_t j = 0; j < c_ns_N_F; j++)
		{
			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// squared distance between particles i and j
			const float distSqu = dx * dx + dy * dy + dz * dz;

			if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
			{
				cntFF++;
			}
		}

		// check boundary candidates (all particles)
		for (uint32_t j = 0; j < c_ns_N_B; j++)
		{
			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// squared distance between particles i and j
			const float distSqu = dx * dx + dy * dy + dz * dz;

			if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
			{
				cntFB++;
			}
		}

		// update neighbor counts
		nbrCountFF[i] = cntFF;
		nbrCountFB[i] = cntFB;
	}

	__global__
	void K_BruteForce_Query_Fill(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		// destinations
		uint32_t* __restrict__ nbrListFF,
		uint32_t* __restrict__ nbrListFB,
		uint32_t* __restrict__ overflowFlag)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_ns_N_F) { return; }

		// particle i data
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];

		// write ranges of packed SELL32 neighbor lists
		const uint32_t chunk = i / N_SellWidth;
		const uint32_t lane = i % N_SellWidth;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint64_t pitchFF =
			(nbrChunkOffsetsFF[chunk + 1] - baseFF) / static_cast<uint64_t>(N_SellWidth);
		const uint64_t pitchFB =
			(nbrChunkOffsetsFB[chunk + 1] - baseFB) / static_cast<uint64_t>(N_SellWidth);
		uint64_t wrFF = 0;
		uint64_t wrFB = 0;

		// check fluid candidates (all particles)
		for (uint32_t j = 0; j < c_ns_N_F; j++)
		{
			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// squared distance between particles i and j
			const float distSqu = dx * dx + dy * dy + dz * dz;

			if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
			{
				// add fluid neighbor to the list
				StoreNeighbor_SELL32(baseFF, pitchFF, lane, wrFF, j, nbrListFF, overflowFlag);
			}
		}

		// check boundary candidates (all particles)
		for (uint32_t j = 0; j < c_ns_N_B; j++)
		{
			// position difference (gmr)
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// squared distance between particles i and j
			const float distSqu = dx * dx + dy * dy + dz * dz;

			if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
			{
				// add boundary neighbor to the list
				StoreNeighbor_SELL32(baseFB, pitchFB, lane, wrFB, j, nbrListFB, overflowFlag);
			}
		}
	}

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

		// particle i data (gmr)
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];

		// cell coordinates of particle i
		const int cxi = FloorToInt((xi - c_ns_xBoxOrig) * c_ns_cellSizeInv);
		const int cyi = FloorToInt((yi - c_ns_yBoxOrig) * c_ns_cellSizeInv);
		const int czi = FloorToInt((zi - c_ns_zBoxOrig) * c_ns_cellSizeInv);

		const uint32_t cx = ClampInt(cxi, 0, c_ns_N_DimX - 1);
		const uint32_t cy = ClampInt(cyi, 0, c_ns_N_DimY - 1);
		const uint32_t cz = ClampInt(czi, 0, c_ns_N_DimZ - 1);

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
		const float xi = xPosB[i];
		const float yi = yPosB[i];
		const float zi = zPosB[i];

		// compute cell coordinates
		const int cxi = FloorToInt((xi - c_ns_xBoxOrig) * c_ns_cellSizeInv);
		const int cyi = FloorToInt((yi - c_ns_yBoxOrig) * c_ns_cellSizeInv);
		const int czi = FloorToInt((zi - c_ns_zBoxOrig) * c_ns_cellSizeInv);

		const uint32_t cx = ClampInt(cxi, 0, c_ns_N_DimX - 1);
		const uint32_t cy = ClampInt(cyi, 0, c_ns_N_DimY - 1);
		const uint32_t cz = ClampInt(czi, 0, c_ns_N_DimZ - 1);

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

		cellStartIdx[i] = c_ns_EMPTY;
		cellEndIdx[i] = c_ns_EMPTY;
	}

	__global__
	void K_CellLinkedList_Basic_Query_Count(
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
		// destinations
		uint32_t* __restrict__ nbrCountFF,
		uint32_t* __restrict__ nbrCountFB)
	{
		uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_F) { return; }

		// particle i data
		// (global memory reads)
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];

		// cell coordinates of particle i
		const int cxi = FloorToInt((xi - c_ns_xBoxOrig) * c_ns_cellSizeInv);
		const int cyi = FloorToInt((yi - c_ns_yBoxOrig) * c_ns_cellSizeInv);
		const int czi = FloorToInt((zi - c_ns_zBoxOrig) * c_ns_cellSizeInv);


		uint32_t cntFF = 0;
		uint32_t cntFB = 0;

		// check all particles j in neighboring grid cells
		const int cellQueryReach = static_cast<int>(c_ns_cellQueryReach);
		for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
		{
			int cz = czi + rz; // target cell coordinate in z-direction
			if (cz < 0 || cz >= static_cast<int>(c_ns_N_DimZ)) { continue; }

			for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
			{
				int cy = cyi + ry; // target cell coordinate in y-direction
				if (cy < 0 || cy >= static_cast<int>(c_ns_N_DimY)) { continue; }

				for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
				{
					int cx = cxi + rx; // target cell coordinate in x-direction
					if (cx < 0 || cx >= static_cast<int>(c_ns_N_DimX)) { continue; }

					// cell ID
					const uint32_t cid = MapToCellId_Basic(static_cast<uint32_t>(cx),
														   static_cast<uint32_t>(cy),
														   static_cast<uint32_t>(cz));

					// check fluid candidates
					const uint32_t sF = cellStartIdxF[cid];
					const uint32_t eF = cellEndIdxF[cid];
					if (sF != c_ns_EMPTY && eF != c_ns_EMPTY)
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

							if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
							{
								cntFF++;
							}
						}
					}

					// check boundary candidates
					const uint32_t sB = cellStartIdxB[cid];
					const uint32_t eB = cellEndIdxB[cid];
					if (sB != c_ns_EMPTY && eB != c_ns_EMPTY)
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

							if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
							{
								cntFB++;
							}
						}
					}
				}
			}
		}

		// update neighbor counts
		// (global memory write)
		nbrCountFF[i] = cntFF;
		nbrCountFB[i] = cntFB;
	}

	__global__
	void K_CellLinkedList_Basic_Query_Fill(
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
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint32_t N_F,
		// destinations
		uint32_t* __restrict__ nbrListFF,
		uint32_t* __restrict__ nbrListFB,
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
		const int cxi = FloorToInt((xi - c_ns_xBoxOrig) * c_ns_cellSizeInv);
		const int cyi = FloorToInt((yi - c_ns_yBoxOrig) * c_ns_cellSizeInv);
		const int czi = FloorToInt((zi - c_ns_zBoxOrig) * c_ns_cellSizeInv);

		// write ranges of packed SELL32 neighbor lists
		const uint32_t chunk = i / N_SellWidth;
		const uint32_t lane = i % N_SellWidth;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint64_t pitchFF =
			(nbrChunkOffsetsFF[chunk + 1] - baseFF) / static_cast<uint64_t>(N_SellWidth);
		const uint64_t pitchFB =
			(nbrChunkOffsetsFB[chunk + 1] - baseFB) / static_cast<uint64_t>(N_SellWidth);
		uint64_t wrFF = 0;
		uint64_t wrFB = 0;

		// check all particles j in neighboring grid cells
		const int cellQueryReach = static_cast<int>(c_ns_cellQueryReach);
		for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
		{
			int cz = czi + rz; // target cell coordinate in z-direction
			if (cz < 0 || cz >= static_cast<int>(c_ns_N_DimZ)) { continue; }

			for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
			{
				int cy = cyi + ry; // target cell coordinate in y-direction
				if (cy < 0 || cy >= static_cast<int>(c_ns_N_DimY)) { continue; }

				for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
				{
					int cx = cxi + rx; // target cell coordinate in x-direction
					if (cx < 0 || cx >= static_cast<int>(c_ns_N_DimX)) { continue; }

					// cell ID
					const uint32_t cid = MapToCellId_Basic(static_cast<uint32_t>(cx),
														   static_cast<uint32_t>(cy),
														   static_cast<uint32_t>(cz));

					// check fluid candidates
					const uint32_t sF = cellStartIdxF[cid];
					const uint32_t eF = cellEndIdxF[cid];
					if (sF != c_ns_EMPTY && eF != c_ns_EMPTY)
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

							if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
							{
								// add fluid neighbor to the list
								// (global memory write)
								StoreNeighbor_SELL32(baseFF, pitchFF, lane, wrFF,
									j, nbrListFF, overflowFlag);
							}
						}
					}

					// check boundary candidates
					const uint32_t sB = cellStartIdxB[cid];
					const uint32_t eB = cellEndIdxB[cid];
					if (sB != c_ns_EMPTY && eB != c_ns_EMPTY)
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

							if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
							{
								// add boundary neighbor to the list
								// (global memory write)
								StoreNeighbor_SELL32(baseFB, pitchFB, lane, wrFB,
									j, nbrListFB, overflowFlag);
							}
						}
					}
				}
			}
		}
	}

} // namespace GPU01

// -------------------------------------------------------------------------------------------------
// --- [ cell linked list compact implementation ] -------------------------------------------------
// -------------------------------------------------------------------------------------------------

namespace GPU01
{
	__device__ __forceinline__
	uint32_t ZigZag(int v)
	{
		// shift left by 1 and XOR with sign bit
		uint32_t uv = static_cast<uint32_t>(v);
		uint32_t even = uv << 1;
		uint32_t sign = static_cast<uint32_t>(-(v < 0)); // 0 or 0xFFFFFFFF

		return even ^ sign;
	}

	__device__ __forceinline__
	uint64_t SpreadBitsWithGaps_21to63(uint32_t v)
	{
		uint64_t x = static_cast<uint64_t>(v & 0x1fffffU); // lower 21 bits

		x = (x | (x << 32)) & 0x1f00000000ffffULL;
		x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
		x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
		x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
		x = (x | (x << 2)) & 0x1249249249249249ULL;

		return x;
	}

	__device__ __forceinline__
	uint64_t MapToCellId_Morton(
		int cx, int cy, int cz)
	{
		// zig-zag encoding of coordinates to handle negative values
		const uint32_t ux = ZigZag(cx);
		const uint32_t uy = ZigZag(cy);
		const uint32_t uz = ZigZag(cz);

		const uint64_t xx = SpreadBitsWithGaps_21to63(ux); // bits at pos 0,3,6,...
		const uint64_t yy = SpreadBitsWithGaps_21to63(uy) << 1; // bits at pos 1,4,7,...
		const uint64_t zz = SpreadBitsWithGaps_21to63(uz) << 2; // bits at pos 2,5,8,...

		return xx | yy | zz; // interleaved 3D Morton/Z-order code
	}

	__device__ __forceinline__
	uint64_t MapToCellId(
		int cx, int cy, int cz)
	{
		return MapToCellId_Morton(cx, cy, cz);
	}

	__device__
	int BinarySearchUniqueCellId(
		const uint64_t* uniqueCellId,
		uint32_t N_NonEmptyCells, uint64_t cid)
	{
		int low = 0;
		int high = static_cast<int>(N_NonEmptyCells) - 1;

		while (low <= high)
		{
			int mid = (low + high) >> 1;
			uint64_t midVal = uniqueCellId[mid];

			if (midVal < cid) { low = mid + 1; }
			else if (midVal > cid) { high = mid - 1; }
			else { return mid; } // found
		}

		return -1; // not found
	}

	__global__
	void K_MarkCellStarts(
		const uint64_t* __restrict__ cellIdSorted,
		const uint32_t N,
		// destinations
		uint32_t* __restrict__ cellStartMarker)
	{
		uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N) { return; }

		if (i == 0) // first element has to be marked
		{
			cellStartMarker[0] = (N > 0) ? 1 : 0;
		}
		else
		{
			uint64_t prev = cellIdSorted[i - 1];
			uint64_t curr = cellIdSorted[i];
			cellStartMarker[i] = (curr != prev) ? 1 : 0;
		}
	}

	__global__
	void K_FillUniqueCellIdAndCellStartIdx(
		const uint64_t* __restrict__ cellIdSorted,
		const uint32_t* __restrict__ cellStartScan,
		const uint32_t N,
		const uint32_t N_NonEmptyCells,
		// destinations
		uint64_t* __restrict__ uniqueCellId,
		uint32_t* __restrict__ cellStartIdx)
	{
		uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N) { return; }

		// inclusive scan value at element i
		const uint32_t s = cellStartScan[i];
		const uint32_t ci = s - 1;

		if (i == 0)
		{
			// first element always defines cell 0
			uniqueCellId[0] = cellIdSorted[0];
			cellStartIdx[0] = 0;
		}
		else
		{
			uint32_t ci_prev = cellStartScan[i - 1];

			if (s != ci_prev)
			{
				// first element of a new cell cluster
				uniqueCellId[ci] = cellIdSorted[i];
				cellStartIdx[ci] = i;
			}
		}

		// one thread also writes the sentinel
		if (i == N - 1)
		{
			cellStartIdx[N_NonEmptyCells] = N;
		}
	}

	__global__
	void K_FillCellIdAndInitParticleIdx(
		const float* __restrict__ xPos,
		const float* __restrict__ yPos,
		const float* __restrict__ zPos,
		const uint32_t N,
		// destinations
		uint64_t* cellId,
		uint32_t* sortedParticleIdx)
	{
		uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N) { return; }

		// particle i data
		// (global memory reads)
		const float x = xPos[i];
		const float y = yPos[i];
		const float z = zPos[i];

		// compute cell coordinates
		const int cx = FloorToInt(x * c_ns_cellSizeInv);
		const int cy = FloorToInt(y * c_ns_cellSizeInv);
		const int cz = FloorToInt(z * c_ns_cellSizeInv);

		cellId[i] = MapToCellId(cx, cy, cz);
		sortedParticleIdx[i] = i;
	}

	__global__
	void K_CellLinkedList_Compact_Query_Count(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const uint64_t* __restrict__ uniqueCellIdF,
		const uint64_t* __restrict__ uniqueCellIdB,
		const uint32_t* __restrict__ cellStartIdxF,
		const uint32_t* __restrict__ cellStartIdxB,
		const uint32_t* __restrict__ sortedParticleIdxF,
		const uint32_t* __restrict__ sortedParticleIdxB,
		const uint32_t N_NonEmptyCellsF, const uint32_t N_NonEmptyCellsB,
		const uint32_t N_F,
		// destinations
		uint32_t* __restrict__ nbrCountFF,
		uint32_t* __restrict__ nbrCountFB)
	{
		uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_F) { return; }

		// particle i data
		// (global memory reads)
		const float x = xPosF[i];
		const float y = yPosF[i];
		const float z = zPosF[i];

		// compute cell coordinates
		const int cx = FloorToInt(x * c_ns_cellSizeInv);
		const int cy = FloorToInt(y * c_ns_cellSizeInv);
		const int cz = FloorToInt(z * c_ns_cellSizeInv);

		uint32_t cntFF = 0;
		uint32_t cntFB = 0;

		// check all particles j in the queried neighboring grid cells
		const int cellQueryReach = static_cast<int>(c_ns_cellQueryReach);
		for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
		{
			int cz_nbr = cz + rz; // target cell coordinate in z-direction

			for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
			{
				int cy_nbr = cy + ry; // target cell coordinate in y-direction

				for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
				{
					int cx_nbr = cx + rx; // target cell coordinate in x-direction

					const uint64_t cid_nbr = MapToCellId(cx_nbr, cy_nbr, cz_nbr);

					// fluid neighbor cell
					int cF_nbr = BinarySearchUniqueCellId(uniqueCellIdF, N_NonEmptyCellsF, cid_nbr);
					if (cF_nbr >= 0)
					{
						// range of all fluid particles in the neighbor cell
						const uint32_t sF_nbr = cellStartIdxF[cF_nbr]; // start
						const uint32_t eF_nbr = cellStartIdxF[cF_nbr + 1]; // end
						if (sF_nbr >= eF_nbr) { continue; }

						for (uint32_t k = sF_nbr; k < eF_nbr; k++)
						{
							// index of fluid neighbor candidate
							const uint32_t j = sortedParticleIdxF[k];

							// position difference
							const float dx = x - xPosF[j];
							const float dy = y - yPosF[j];
							const float dz = z - zPosF[j];

							// squared distance between particles i and j
							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
							{
								cntFF++;
							}
						}
					}

					// boundary neighbor cell
					int cB_nbr = BinarySearchUniqueCellId(uniqueCellIdB, N_NonEmptyCellsB, cid_nbr);
					if (cB_nbr >= 0)
					{
						// range of all boundary particles in the neighbor cell
						const uint32_t sB_nbr = cellStartIdxB[cB_nbr]; // start
						const uint32_t eB_nbr = cellStartIdxB[cB_nbr + 1]; // end
						if (sB_nbr >= eB_nbr) { continue; }

						for (uint32_t k = sB_nbr; k < eB_nbr; k++)
						{
							// index of boundary neighbor candidate
							const uint32_t j = sortedParticleIdxB[k];

							// position difference
							const float dx = x - xPosB[j];
							const float dy = y - yPosB[j];
							const float dz = z - zPosB[j];

							// squared distance between particles i and j
							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
							{
								cntFB++;
							}
						}
					}
				}
			}
		}

		// update neighbor counts
		// (global memory write)
		nbrCountFF[i] = cntFF;
		nbrCountFB[i] = cntFB;
	}

	__global__
	void K_CellLinkedList_Compact_Query_Fill(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const uint64_t* __restrict__ uniqueCellIdF,
		const uint64_t* __restrict__ uniqueCellIdB,
		const uint32_t* __restrict__ cellStartIdxF,
		const uint32_t* __restrict__ cellStartIdxB,
		const uint32_t* __restrict__ sortedParticleIdxF,
		const uint32_t* __restrict__ sortedParticleIdxB,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint32_t N_NonEmptyCellsF, const uint32_t N_NonEmptyCellsB,
		const uint32_t N_F,
		// destinations
		uint32_t* __restrict__ nbrListFF,
		uint32_t* __restrict__ nbrListFB,
		uint32_t* __restrict__ overflowFlag)
	{
		uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_F) { return; }

		// particle i data
		// (global memory reads)
		const float x = xPosF[i];
		const float y = yPosF[i];
		const float z = zPosF[i];

		// compute cell coordinates
		const int cx = FloorToInt(x * c_ns_cellSizeInv);
		const int cy = FloorToInt(y * c_ns_cellSizeInv);
		const int cz = FloorToInt(z * c_ns_cellSizeInv);

		// write ranges of packed SELL32 neighbor lists
		const uint32_t chunk = i / N_SellWidth;
		const uint32_t lane = i % N_SellWidth;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint64_t pitchFF =
			(nbrChunkOffsetsFF[chunk + 1] - baseFF) / static_cast<uint64_t>(N_SellWidth);
		const uint64_t pitchFB =
			(nbrChunkOffsetsFB[chunk + 1] - baseFB) / static_cast<uint64_t>(N_SellWidth);
		uint64_t wrFF = 0;
		uint64_t wrFB = 0;

		// check all particles j in the queried neighboring grid cells
		const int cellQueryReach = static_cast<int>(c_ns_cellQueryReach);
		for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
		{
			int cz_nbr = cz + rz; // target cell coordinate in z-direction

			for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
			{
				int cy_nbr = cy + ry; // target cell coordinate in y-direction

				for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
				{
					int cx_nbr = cx + rx; // target cell coordinate in x-direction

					const uint64_t cid_nbr = MapToCellId(cx_nbr, cy_nbr, cz_nbr);

					// fluid neighbor cell
					int cF_nbr = BinarySearchUniqueCellId(uniqueCellIdF, N_NonEmptyCellsF, cid_nbr);
					if (cF_nbr >= 0)
					{
						// range of all fluid particles in the neighbor cell
						const uint32_t sF_nbr = cellStartIdxF[cF_nbr]; // start
						const uint32_t eF_nbr = cellStartIdxF[cF_nbr + 1]; // end
						if (sF_nbr >= eF_nbr) { continue; }

						for (uint32_t k = sF_nbr; k < eF_nbr; k++)
						{
							// index of fluid neighbor candidate
							const uint32_t j = sortedParticleIdxF[k];

							// position difference
							const float dx = x - xPosF[j];
							const float dy = y - yPosF[j];
							const float dz = z - zPosF[j];

							// squared distance between particles i and j
							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
							{
								// add fluid neighbor to the list
								// (global memory write)
								StoreNeighbor_SELL32(baseFF, pitchFF, lane, wrFF,
									j, nbrListFF, overflowFlag);
							}
						}
					}

					// boundary neighbor cell
					int cB_nbr = BinarySearchUniqueCellId(uniqueCellIdB, N_NonEmptyCellsB, cid_nbr);
					if (cB_nbr >= 0)
					{
						// range of all boundary particles in the neighbor cell
						const uint32_t sB_nbr = cellStartIdxB[cB_nbr]; // start
						const uint32_t eB_nbr = cellStartIdxB[cB_nbr + 1]; // end
						if (sB_nbr >= eB_nbr) { continue; }

						for (uint32_t k = sB_nbr; k < eB_nbr; k++)
						{
							// index of boundary neighbor candidate
							const uint32_t j = sortedParticleIdxB[k];

							// position difference
							const float dx = x - xPosB[j];
							const float dy = y - yPosB[j];
							const float dz = z - zPosB[j];

							// squared distance between particles i and j
							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
							{
								// add boundary neighbor to the list
								// (global memory write)
								StoreNeighbor_SELL32(baseFB, pitchFB, lane, wrFB,
									j, nbrListFB, overflowFlag);
							}
						}
					}
				}
			}
		}
	}

} // namespace GPU01

// -------------------------------------------------------------------------------------------------
// --- [ kernel launchers ] ------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

namespace GPU01
{
	void Launch_BruteForce_Query_Count(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		uint32_t* nbrCountFF,
		uint32_t* nbrCountFB)
	{
		(void)N_B;

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_BruteForce_Query_Count<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB,
			nbrCountFF, nbrCountFB);

		GPU::CheckLastKernel("K_BruteForce_Query_Count()");
	}

	void Launch_BruteForce_Query_Fill(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		// destinations
		uint32_t* nbrListFF,
		uint32_t* nbrListFB,
		uint32_t* overflowFlag)
	{
		(void)N_B;

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_BruteForce_Query_Fill<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB,
			nbrListFF, nbrListFB, overflowFlag);

		GPU::CheckLastKernel("K_BruteForce_Query_Fill()");
	}

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

	void Launch_CellLinkedList_Basic_Query_Count(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* cellStartIdxF, const uint32_t* cellEndIdxF,
		const uint32_t* cellStartIdxB, const uint32_t* cellEndIdxB,
		const uint32_t* sortedParticleIdxF, const uint32_t* sortedParticleIdxB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		uint32_t* nbrCountFF, uint32_t* nbrCountFB)
	{
		(void)N_B;

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_CellLinkedList_Basic_Query_Count<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, cellStartIdxF, cellEndIdxF,
			cellStartIdxB, cellEndIdxB, sortedParticleIdxF, sortedParticleIdxB,
			N_F, nbrCountFF, nbrCountFB);

		GPU::CheckLastKernel("K_CellLinkedList_Basic_Query_Count()");
	}

	void Launch_CellLinkedList_Basic_Query_Fill(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* cellStartIdxF, const uint32_t* cellEndIdxF,
		const uint32_t* cellStartIdxB, const uint32_t* cellEndIdxB,
		const uint32_t* sortedParticleIdxF, const uint32_t* sortedParticleIdxB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		// destinations
		uint32_t* nbrListFF, uint32_t* nbrListFB,
		uint32_t* overflowFlag)
	{
		(void)N_B;

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_CellLinkedList_Basic_Query_Fill<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, cellStartIdxF, cellEndIdxF,
			cellStartIdxB, cellEndIdxB, sortedParticleIdxF, sortedParticleIdxB,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, N_F,
			nbrListFF, nbrListFB, overflowFlag);

		GPU::CheckLastKernel("K_CellLinkedList_Basic_Query_Fill()");
	}

	void Launch_MarkCellStarts(
		const uint64_t* cellIdSorted,
		const uint32_t N,
		cudaStream_t stream,
		// destinations
		uint32_t* cellStartMarker)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_MarkCellStarts<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			cellIdSorted, N, cellStartMarker);

		GPU::CheckLastKernel("K_MarkCellStarts()");
	}

	void Launch_FillUniqueCellIdAndCellStartIdx(
		const uint64_t* cellIdSorted,
		const uint32_t* cellStartScan,
		const uint32_t N,
		const uint32_t N_NonEmptyCells,
		cudaStream_t stream,
		// destinations
		uint64_t* uniqueCellId,
		uint32_t* cellStartIdx)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_FillUniqueCellIdAndCellStartIdx<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			cellIdSorted, cellStartScan, N, N_NonEmptyCells, uniqueCellId, cellStartIdx);

		GPU::CheckLastKernel("K_FillUniqueCellIdAndCellStartIdx()");
	}

	void Launch_FillCellIdAndInitParticleIdx(
		const float* xPos, const float* yPos, const float* zPos,
		const uint32_t N,
		cudaStream_t stream,
		// destinations
		uint64_t* cellId, uint32_t* sortedParticleIdx)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_FillCellIdAndInitParticleIdx<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPos, yPos, zPos, N, cellId, sortedParticleIdx);

		GPU::CheckLastKernel("K_FillCellIdAndInitParticleIdx()");
	}

	void Launch_CellLinkedList_Compact_Query_Count(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint64_t* uniqueCellIdF, const uint64_t* uniqueCellIdB,
		const uint32_t* cellStartIdxF, const uint32_t* cellStartIdxB,
		const uint32_t* sortedParticleIdxF, const uint32_t* sortedParticleIdxB,
		const uint32_t N_NonEmptyCellsF, const uint32_t N_NonEmptyCellsB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		uint32_t* nbrCountFF, uint32_t* nbrCountFB)
	{
		(void)N_B;

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_CellLinkedList_Compact_Query_Count<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, uniqueCellIdF, uniqueCellIdB,
			cellStartIdxF, cellStartIdxB, sortedParticleIdxF, sortedParticleIdxB,
			N_NonEmptyCellsF, N_NonEmptyCellsB, N_F,
			nbrCountFF, nbrCountFB);

		GPU::CheckLastKernel("K_CellLinkedList_Compact_Query_Count()");
	}

	void Launch_CellLinkedList_Compact_Query_Fill(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint64_t* uniqueCellIdF, const uint64_t* uniqueCellIdB,
		const uint32_t* cellStartIdxF, const uint32_t* cellStartIdxB,
		const uint32_t* sortedParticleIdxF, const uint32_t* sortedParticleIdxB,
		const uint32_t N_NonEmptyCellsF, const uint32_t N_NonEmptyCellsB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		// destinations
		uint32_t* nbrListFF, uint32_t* nbrListFB,
		uint32_t* overflowFlag)
	{
		(void)N_B;

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_CellLinkedList_Compact_Query_Fill<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, uniqueCellIdF, uniqueCellIdB,
			cellStartIdxF, cellStartIdxB, sortedParticleIdxF, sortedParticleIdxB,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB,
			N_NonEmptyCellsF, N_NonEmptyCellsB, N_F,
			nbrListFF, nbrListFB, overflowFlag);

		GPU::CheckLastKernel("K_CellLinkedList_Compact_Query_Fill()");
	}

} // namespace GPU01
