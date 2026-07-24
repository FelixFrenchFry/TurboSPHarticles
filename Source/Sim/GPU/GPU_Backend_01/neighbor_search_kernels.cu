// neighbor_search_kernels.cu

// external
#include <cuda_runtime.h>

// internal
#include "Source/Sim/GPU/gpu_utils.h"
#include "neighbor_search_kernels.cuh"
#include "neighbor_search.cuh"



namespace GPU01
{
	static constexpr uint32_t N_SellWidth = 32u;

	__constant__ uint32_t c_ns_N_F;
	__constant__ uint32_t c_ns_N_B;
	__constant__ uint32_t c_ns_N_R;
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
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
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
		GPU::Check(cudaMemcpyToSymbol(c_ns_N_R, &N_R, sizeof(uint32_t)));
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
	void CommitChunkNeighborIndexRange(
		const uint32_t chunk,
		const uint32_t localMin,
		const uint32_t localMax,
		uint32_t* __restrict__ chunkMinRef,
		uint32_t* __restrict__ chunkMaxRef)
	{
		if (localMin == 0xffffffffu) { return; }

		atomicMin(chunkMinRef + chunk, localMin);
		atomicMax(chunkMaxRef + chunk, localMax);
	}

	__device__ __forceinline__
	void StoreNeighbor_CompressedSell32(
		const uint64_t chunkBase,
		const uint32_t chunkPitch,
		const uint32_t lane,
		uint32_t& wr,
		const uint32_t j,
		const uint32_t chunkBaseIdx,
		const uint8_t chunkFormat,
		uint8_t* __restrict__ nbrListDelta8,
		uint16_t* __restrict__ nbrListDelta16,
		uint32_t* __restrict__ nbrListAbs32,
		uint32_t* __restrict__ overflowFlag)
	{
		if (wr < chunkPitch)
		{
			const uint64_t linearIdx = wr * N_SellWidth + lane;

			if (chunkFormat == static_cast<uint8_t>(NbrChunkFormat::Delta8))
			{
				#if defined(_DEBUG)
				if (j < chunkBaseIdx || j - chunkBaseIdx > 0xffu)
				{
					atomicOr(overflowFlag, NbrListError_DeltaRangeOverflow);
					return;
				}
				#endif

				nbrListDelta8[chunkBase + linearIdx] =
					static_cast<uint8_t>(j - chunkBaseIdx);
			}
			else if (chunkFormat == static_cast<uint8_t>(NbrChunkFormat::Delta16))
			{
				#if defined(_DEBUG)
				if (j < chunkBaseIdx || j - chunkBaseIdx > 0xffffu)
				{
					atomicOr(overflowFlag, NbrListError_DeltaRangeOverflow);
					return;
				}
				#endif

				nbrListDelta16[chunkBase + linearIdx] =
					static_cast<uint16_t>(j - chunkBaseIdx);
			}
			else
			{
				#if defined(_DEBUG)
				if (chunkFormat != static_cast<uint8_t>(NbrChunkFormat::Abs32))
				{
					atomicOr(overflowFlag, NbrListError_InvalidChunkFormat);
					return;
				}
				#endif

				nbrListAbs32[chunkBase + linearIdx] = j;
			}

			wr++;
		}
		else
		{
			atomicOr(overflowFlag, NbrListError_FillOverflow);
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
		const float* __restrict__ xPosR,
		const float* __restrict__ yPosR,
		const float* __restrict__ zPosR,
		// destinations
		uint16_t* __restrict__ nbrCountFF,
		uint16_t* __restrict__ nbrCountFB,
		uint16_t* __restrict__ nbrCountFR,
		uint32_t* __restrict__ nbrChunkMinRefFF,
		uint32_t* __restrict__ nbrChunkMaxRefFF,
		uint32_t* __restrict__ nbrChunkMinRefFB,
		uint32_t* __restrict__ nbrChunkMaxRefFB,
		uint32_t* __restrict__ nbrChunkMinRefFR,
		uint32_t* __restrict__ nbrChunkMaxRefFR)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_ns_N_F) { return; }

		// particle i data
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];
		const uint32_t chunk = i / N_SellWidth;

		uint32_t cntFF = 0;
		uint32_t cntFB = 0;
		uint32_t cntFR = 0;
		uint32_t minRefFF = 0xffffffffu;
		uint32_t maxRefFF = 0u;
		uint32_t minRefFB = 0xffffffffu;
		uint32_t maxRefFB = 0u;
		uint32_t minRefFR = 0xffffffffu;
		uint32_t maxRefFR = 0u;

		// check fluid candidates (all particles)
		for (uint32_t j = 0; j < c_ns_N_F; j++)
		{
			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// squared distance between particles i and j
			const float distSqu = dx * dx + dy * dy + dz * dz;

			if (distSqu <= c_ns_kernelRadiusSqu) [[unlikely]] // neighbor found
			{
				cntFF++;
				minRefFF = (j < minRefFF) ? j : minRefFF;
				maxRefFF = (j > maxRefFF) ? j : maxRefFF;
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

			if (distSqu <= c_ns_kernelRadiusSqu) [[unlikely]] // neighbor found
			{
				cntFB++;
				minRefFB = (j < minRefFB) ? j : minRefFB;
				maxRefFB = (j > maxRefFB) ? j : maxRefFB;
			}
		}

		// check rigid candidates (all particles)
		for (uint32_t j = 0; j < c_ns_N_R; j++)
		{
			// position difference
			const float dx = xi - xPosR[j];
			const float dy = yi - yPosR[j];
			const float dz = zi - zPosR[j];

			// squared distance between particles i and j
			const float distSqu = dx * dx + dy * dy + dz * dz;

			if (distSqu <= c_ns_kernelRadiusSqu) [[unlikely]] // neighbor found
			{
				cntFR++;
				minRefFR = (j < minRefFR) ? j : minRefFR;
				maxRefFR = (j > maxRefFR) ? j : maxRefFR;
			}
		}

		CommitChunkNeighborIndexRange(
			chunk, minRefFF, maxRefFF, nbrChunkMinRefFF, nbrChunkMaxRefFF);
		CommitChunkNeighborIndexRange(
			chunk, minRefFB, maxRefFB, nbrChunkMinRefFB, nbrChunkMaxRefFB);
		CommitChunkNeighborIndexRange(
			chunk, minRefFR, maxRefFR, nbrChunkMinRefFR, nbrChunkMaxRefFR);

		// update neighbor counts
		nbrCountFF[i] = static_cast<uint16_t>(cntFF);
		nbrCountFB[i] = static_cast<uint16_t>(cntFB);
		nbrCountFR[i] = static_cast<uint16_t>(cntFR);
	}

	__global__
	void K_BruteForce_Query_Fill(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const float* __restrict__ xPosR,
		const float* __restrict__ yPosR,
		const float* __restrict__ zPosR,
		const uint16_t* __restrict__ nbrChunkMaxCountFF,
		const uint16_t* __restrict__ nbrChunkMaxCountFB,
		const uint16_t* __restrict__ nbrChunkMaxCountFR,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFR,
		const uint32_t* __restrict__ nbrChunkBaseIdxFF,
		const uint32_t* __restrict__ nbrChunkBaseIdxFB,
		const uint32_t* __restrict__ nbrChunkBaseIdxFR,
		const uint8_t* __restrict__ nbrChunkFormatFF,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrChunkFormatFR,
		// destinations
		uint8_t* __restrict__ nbrListDelta8FF,
		uint8_t* __restrict__ nbrListDelta8FB,
		uint8_t* __restrict__ nbrListDelta8FR,
		uint16_t* __restrict__ nbrListDelta16FF,
		uint16_t* __restrict__ nbrListDelta16FB,
		uint16_t* __restrict__ nbrListDelta16FR,
		uint32_t* __restrict__ nbrListAbs32FF,
		uint32_t* __restrict__ nbrListAbs32FB,
		uint32_t* __restrict__ nbrListAbs32FR,
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
		const uint64_t baseFR = nbrChunkOffsetsFR[chunk];
		const uint32_t pitchFF = static_cast<uint32_t>(nbrChunkMaxCountFF[chunk]);
		const uint32_t pitchFB = static_cast<uint32_t>(nbrChunkMaxCountFB[chunk]);
		const uint32_t pitchFR = static_cast<uint32_t>(nbrChunkMaxCountFR[chunk]);
		const uint32_t baseIdxFF = nbrChunkBaseIdxFF[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint32_t baseIdxFR = nbrChunkBaseIdxFR[chunk];
		const uint8_t formatFF = nbrChunkFormatFF[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint8_t formatFR = nbrChunkFormatFR[chunk];
		uint32_t wrFF = 0;
		uint32_t wrFB = 0;
		uint32_t wrFR = 0;

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
				StoreNeighbor_CompressedSell32(
					baseFF, pitchFF, lane, wrFF, j,
					baseIdxFF, formatFF,
					nbrListDelta8FF, nbrListDelta16FF, nbrListAbs32FF,
					overflowFlag);
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
				StoreNeighbor_CompressedSell32(
					baseFB, pitchFB, lane, wrFB, j,
					baseIdxFB, formatFB,
					nbrListDelta8FB, nbrListDelta16FB, nbrListAbs32FB,
					overflowFlag);
			}
		}

		// check rigid candidates (all particles)
		for (uint32_t j = 0; j < c_ns_N_R; j++)
		{
			// position difference
			const float dx = xi - xPosR[j];
			const float dy = yi - yPosR[j];
			const float dz = zi - zPosR[j];

			// squared distance between particles i and j
			const float distSqu = dx * dx + dy * dy + dz * dz;

			if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
			{
				// add rigid neighbor to the list
				StoreNeighbor_CompressedSell32(
					baseFR, pitchFR, lane, wrFR, j,
					baseIdxFR, formatFR,
					nbrListDelta8FR, nbrListDelta16FR, nbrListAbs32FR,
					overflowFlag);
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

	__device__
	uint32_t LowerBoundUniqueCellId(
		const uint64_t* uniqueCellId,
		uint32_t N_NonEmptyCells, uint64_t cid)
	{
		uint32_t lo = 0;
		uint32_t hi = N_NonEmptyCells;

		while (lo < hi)
		{
			const uint32_t mid = lo + ((hi - lo) >> 1u);
			if (uniqueCellId[mid] < cid)
			{
				lo = mid + 1u;
			}
			else
			{
				hi = mid;
			}
		}

		return lo;
	}

	__device__
	uint32_t UpperBoundUniqueCellId(
		const uint64_t* uniqueCellId,
		uint32_t N_NonEmptyCells, uint64_t cid)
	{
		uint32_t lo = 0;
		uint32_t hi = N_NonEmptyCells;

		while (lo < hi)
		{
			const uint32_t mid = lo + ((hi - lo) >> 1u);
			if (uniqueCellId[mid] <= cid)
			{
				lo = mid + 1u;
			}
			else
			{
				hi = mid;
			}
		}

		return lo;
	}

	__device__
	int BinarySearchUniqueCellId(
		const uint64_t* uniqueCellId,
		uint32_t lo, uint32_t hi, uint64_t cid)
	{
		while (lo < hi)
		{
			const uint32_t mid = lo + ((hi - lo) >> 1u);
			const uint64_t midVal = uniqueCellId[mid];

			if (midVal < cid)
			{
				lo = mid + 1u;
			}
			else if (midVal > cid)
			{
				hi = mid;
			}
			else
			{
				return static_cast<int>(mid);
			}
		}

		return -1;
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
	void K_FillMergedCompactCellRefs(
		const uint64_t* __restrict__ uniqueCellIdAny,
		const uint32_t N_NonEmptyCellsAny,
		const uint64_t* __restrict__ uniqueCellIdF,
		const uint64_t* __restrict__ uniqueCellIdB,
		const uint32_t N_NonEmptyCellsF,
		const uint32_t N_NonEmptyCellsB,
		// destinations
		uint32_t* __restrict__ cellRefFByAny,
		uint32_t* __restrict__ cellRefBByAny)
	{
		const uint32_t cAny = blockIdx.x * blockDim.x + threadIdx.x;
		if (cAny >= N_NonEmptyCellsAny) { return; }

		const uint64_t cid = uniqueCellIdAny[cAny];
		const int cF = BinarySearchUniqueCellId(uniqueCellIdF, N_NonEmptyCellsF, cid);
		const int cB = BinarySearchUniqueCellId(uniqueCellIdB, N_NonEmptyCellsB, cid);

		cellRefFByAny[cAny] = (cF >= 0) ? static_cast<uint32_t>(cF) : 0xffffffffu;
		cellRefBByAny[cAny] = (cB >= 0) ? static_cast<uint32_t>(cB) : 0xffffffffu;
	}

	__global__
	void K_FillQueryLookupBoundsF(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const uint32_t* __restrict__ cellStartIdxF,
		const uint64_t* __restrict__ uniqueCellIdAny,
		const uint32_t N_NonEmptyCellsF,
		const uint32_t N_NonEmptyCellsAny,
		const uint32_t cellQueryReach,
		// destinations
		uint32_t* __restrict__ compactCellIdxByParticleF,
		uint32_t* __restrict__ queryLoAnyByCellF,
		uint32_t* __restrict__ queryHiAnyByCellF)
	{
		const uint32_t cF = blockIdx.x * blockDim.x + threadIdx.x;
		if (cF >= N_NonEmptyCellsF) { return; }

		const uint32_t srcBegin = cellStartIdxF[cF];
		const uint32_t srcEnd = cellStartIdxF[cF + 1];
		if (srcBegin >= srcEnd)
		{
			queryLoAnyByCellF[cF] = 0;
			queryHiAnyByCellF[cF] = 0;
			return;
		}

		const float x0 = xPosF[srcBegin];
		const float y0 = yPosF[srcBegin];
		const float z0 = zPosF[srcBegin];

		const int cx = FloorToInt(x0 * c_ns_cellSizeInv);
		const int cy = FloorToInt(y0 * c_ns_cellSizeInv);
		const int cz = FloorToInt(z0 * c_ns_cellSizeInv);

		uint64_t litMaxCellId = 0xffffffffffffffffull;
		uint64_t bigMinCellId = 0;
		const int iCellQueryReach = static_cast<int>(cellQueryReach);

		for (int rz = -iCellQueryReach; rz <= iCellQueryReach; rz++)
		{
			const int cz_nbr = cz + rz;

			for (int ry = -iCellQueryReach; ry <= iCellQueryReach; ry++)
			{
				const int cy_nbr = cy + ry;

				for (int rx = -iCellQueryReach; rx <= iCellQueryReach; rx++)
				{
					const int cx_nbr = cx + rx;
					const uint64_t cid_nbr = MapToCellId(cx_nbr, cy_nbr, cz_nbr);
					litMaxCellId = (cid_nbr < litMaxCellId) ? cid_nbr : litMaxCellId;
					bigMinCellId = (cid_nbr > bigMinCellId) ? cid_nbr : bigMinCellId;
				}
			}
		}

		const uint32_t loAny = LowerBoundUniqueCellId(
			uniqueCellIdAny, N_NonEmptyCellsAny, litMaxCellId);
		uint32_t hiAny = UpperBoundUniqueCellId(
			uniqueCellIdAny, N_NonEmptyCellsAny, bigMinCellId);
		if (hiAny < loAny) { hiAny = loAny; }

		queryLoAnyByCellF[cF] = loAny;
		queryHiAnyByCellF[cF] = hiAny;

		for (uint32_t i = srcBegin; i < srcEnd; i++)
		{
			compactCellIdxByParticleF[i] = cF;
		}
	}

	__global__
	void K_FillQueryCellRefsByCellF(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const uint32_t* __restrict__ cellStartIdxF,
		const uint64_t* __restrict__ uniqueCellIdAny,
		const uint32_t* __restrict__ queryLoAnyByCellF,
		const uint32_t* __restrict__ queryHiAnyByCellF,
		const uint32_t N_NonEmptyCellsF,
		const uint32_t queryCellSlotCount,
		// destinations
		uint32_t* __restrict__ queryCellRefAnyByCellF)
	{
		const uint32_t cF = blockIdx.x * blockDim.x + threadIdx.x;
		if (cF >= N_NonEmptyCellsF) { return; }

		const uint32_t srcBegin = cellStartIdxF[cF];
		const uint32_t srcEnd = cellStartIdxF[cF + 1];
		const uint64_t slotBase =
			static_cast<uint64_t>(cF) * static_cast<uint64_t>(queryCellSlotCount);

		if (srcBegin >= srcEnd)
		{
			for (uint32_t s = 0; s < queryCellSlotCount; s++)
			{
				queryCellRefAnyByCellF[slotBase + s] = 0xffffffffu;
			}
			return;
		}

		const float x0 = xPosF[srcBegin];
		const float y0 = yPosF[srcBegin];
		const float z0 = zPosF[srcBegin];

		const int cx = FloorToInt(x0 * c_ns_cellSizeInv);
		const int cy = FloorToInt(y0 * c_ns_cellSizeInv);
		const int cz = FloorToInt(z0 * c_ns_cellSizeInv);
		const uint32_t loAny = queryLoAnyByCellF[cF];
		const uint32_t hiAny = queryHiAnyByCellF[cF];
		const int cellQueryReach = static_cast<int>(c_ns_cellQueryReach);

		uint32_t s = 0;
		for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
		{
			const int cz_nbr = cz + rz;

			for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
			{
				const int cy_nbr = cy + ry;

				for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
				{
					const int cx_nbr = cx + rx;
					uint32_t cAny_nbr = 0xffffffffu;

					if (loAny < hiAny)
					{
						const uint64_t cid_nbr = MapToCellId(cx_nbr, cy_nbr, cz_nbr);
						const int cAny = BinarySearchUniqueCellId(
							uniqueCellIdAny, loAny, hiAny, cid_nbr);
						if (cAny >= 0) { cAny_nbr = static_cast<uint32_t>(cAny); }
					}

					if (s < queryCellSlotCount)
					{
						queryCellRefAnyByCellF[slotBase + s] = cAny_nbr;
					}
					s++;
				}
			}
		}

		for (; s < queryCellSlotCount; s++)
		{
			queryCellRefAnyByCellF[slotBase + s] = 0xffffffffu;
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
		const float* __restrict__ xPosR,
		const float* __restrict__ yPosR,
		const float* __restrict__ zPosR,
		const uint32_t* __restrict__ cellRefFByAny,
		const uint32_t* __restrict__ cellRefBByAny,
		const uint32_t* __restrict__ cellStartIdxF,
		const uint32_t* __restrict__ cellStartIdxB,
		const uint32_t* __restrict__ compactCellIdxByParticleF,
		const uint32_t* __restrict__ queryCellRefAnyByCellF,
		const uint32_t queryCellSlotCount,
		const uint64_t* __restrict__ uniqueCellIdR,
		const uint32_t* __restrict__ cellStartIdxR,
		const uint32_t* __restrict__ sortedParticleIdxR,
		const uint32_t N_NonEmptyCellsR,
		const uint32_t N_F,
		// destinations
		uint16_t* __restrict__ nbrCountFF,
		uint16_t* __restrict__ nbrCountFB,
		uint16_t* __restrict__ nbrCountFR,
		uint32_t* __restrict__ nbrChunkMinRefFF,
		uint32_t* __restrict__ nbrChunkMaxRefFF,
		uint32_t* __restrict__ nbrChunkMinRefFB,
		uint32_t* __restrict__ nbrChunkMaxRefFB,
		uint32_t* __restrict__ nbrChunkMinRefFR,
		uint32_t* __restrict__ nbrChunkMaxRefFR)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_F) { return; }

		// particle i data
		// (global memory reads)
		const float x = xPosF[i];
		const float y = yPosF[i];
		const float z = zPosF[i];
		const uint32_t chunk = i / N_SellWidth;
		const uint32_t cF_src = compactCellIdxByParticleF[i];
		const uint64_t querySlotBase =
			static_cast<uint64_t>(cF_src) * static_cast<uint64_t>(queryCellSlotCount);

		uint32_t cntFF = 0;
		uint32_t cntFB = 0;
		uint32_t cntFR = 0;
		uint32_t minRefFF = 0xffffffffu;
		uint32_t maxRefFF = 0u;
		uint32_t minRefFB = 0xffffffffu;
		uint32_t maxRefFB = 0u;
		uint32_t minRefFR = 0xffffffffu;
		uint32_t maxRefFR = 0u;

		// check all particles j in the cached neighboring grid cells
		for (uint32_t s = 0; s < queryCellSlotCount; s++)
		{
			const uint32_t cAny_nbr = queryCellRefAnyByCellF[querySlotBase + s];
			if (cAny_nbr == 0xffffffffu) { continue; }

			// fluid neighbor cell
			const uint32_t cF_nbr = cellRefFByAny[cAny_nbr];
			if (cF_nbr != 0xffffffffu)
			{
				// range of all fluid particles in the neighbor cell
				const uint32_t sF_nbr = cellStartIdxF[cF_nbr]; // start
				const uint32_t eF_nbr = cellStartIdxF[cF_nbr + 1]; // end
				if (sF_nbr >= eF_nbr) { continue; }

				for (uint32_t k = sF_nbr; k < eF_nbr; k++)
				{
					// index of fluid neighbor candidate
					const uint32_t j = k;

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
						minRefFF = (j < minRefFF) ? j : minRefFF;
						maxRefFF = (j > maxRefFF) ? j : maxRefFF;
					}
				}
			}

			// boundary neighbor cell
			const uint32_t cB_nbr = cellRefBByAny[cAny_nbr];
			if (cB_nbr != 0xffffffffu)
			{
				// range of all boundary particles in the neighbor cell
				const uint32_t sB_nbr = cellStartIdxB[cB_nbr]; // start
				const uint32_t eB_nbr = cellStartIdxB[cB_nbr + 1]; // end
				if (sB_nbr >= eB_nbr) { continue; }

				for (uint32_t k = sB_nbr; k < eB_nbr; k++)
				{
					// index of boundary neighbor candidate
					const uint32_t j = k;

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
						minRefFB = (j < minRefFB) ? j : minRefFB;
						maxRefFB = (j > maxRefFB) ? j : maxRefFB;
					}
				}
			}
		}

		// check all rigid particles in the neighboring grid cells
		if (c_ns_N_R > 0u)
		{
			const int cx = FloorToInt(x * c_ns_cellSizeInv);
			const int cy = FloorToInt(y * c_ns_cellSizeInv);
			const int cz = FloorToInt(z * c_ns_cellSizeInv);
			const int cellQueryReach = static_cast<int>(c_ns_cellQueryReach);

			for (int cellOffsetZ = -cellQueryReach;
				cellOffsetZ <= cellQueryReach; cellOffsetZ++)
			{
				for (int cellOffsetY = -cellQueryReach;
					cellOffsetY <= cellQueryReach; cellOffsetY++)
				{
					for (int cellOffsetX = -cellQueryReach;
						cellOffsetX <= cellQueryReach; cellOffsetX++)
					{
						const uint64_t cellId = MapToCellId(
							cx + cellOffsetX, cy + cellOffsetY, cz + cellOffsetZ);
						const int cellR = BinarySearchUniqueCellId(
							uniqueCellIdR, N_NonEmptyCellsR, cellId);
						if (cellR < 0) { continue; }

						const uint32_t begin = cellStartIdxR[cellR];
						const uint32_t end = cellStartIdxR[cellR + 1];
						for (uint32_t k = begin; k < end; k++)
						{
							const uint32_t j = sortedParticleIdxR[k];
							const float dx = x - xPosR[j];
							const float dy = y - yPosR[j];
							const float dz = z - zPosR[j];
							const float distSqu = dx * dx + dy * dy + dz * dz;

							if (distSqu <= c_ns_kernelRadiusSqu)
							{
								cntFR++;
								minRefFR = (j < minRefFR) ? j : minRefFR;
								maxRefFR = (j > maxRefFR) ? j : maxRefFR;
							}
						}
					}
				}
			}
		}

		// update neighbor counts
		// (global memory write)
		CommitChunkNeighborIndexRange(
			chunk, minRefFF, maxRefFF, nbrChunkMinRefFF, nbrChunkMaxRefFF);
		CommitChunkNeighborIndexRange(
			chunk, minRefFB, maxRefFB, nbrChunkMinRefFB, nbrChunkMaxRefFB);
		CommitChunkNeighborIndexRange(
			chunk, minRefFR, maxRefFR, nbrChunkMinRefFR, nbrChunkMaxRefFR);

		nbrCountFF[i] = static_cast<uint16_t>(cntFF);
		nbrCountFB[i] = static_cast<uint16_t>(cntFB);
		nbrCountFR[i] = static_cast<uint16_t>(cntFR);
	}

	__global__
	void K_CellLinkedList_Compact_Query_Fill(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const float* __restrict__ xPosR,
		const float* __restrict__ yPosR,
		const float* __restrict__ zPosR,
		const uint32_t* __restrict__ cellRefFByAny,
		const uint32_t* __restrict__ cellRefBByAny,
		const uint32_t* __restrict__ cellStartIdxF,
		const uint32_t* __restrict__ cellStartIdxB,
		const uint32_t* __restrict__ compactCellIdxByParticleF,
		const uint32_t* __restrict__ queryCellRefAnyByCellF,
		const uint32_t queryCellSlotCount,
		const uint64_t* __restrict__ uniqueCellIdR,
		const uint32_t* __restrict__ cellStartIdxR,
		const uint32_t* __restrict__ sortedParticleIdxR,
		const uint32_t N_NonEmptyCellsR,
		const uint16_t* __restrict__ nbrChunkMaxCountFF,
		const uint16_t* __restrict__ nbrChunkMaxCountFB,
		const uint16_t* __restrict__ nbrChunkMaxCountFR,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFR,
		const uint32_t* __restrict__ nbrChunkBaseIdxFF,
		const uint32_t* __restrict__ nbrChunkBaseIdxFB,
		const uint32_t* __restrict__ nbrChunkBaseIdxFR,
		const uint8_t* __restrict__ nbrChunkFormatFF,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrChunkFormatFR,
		const uint32_t N_F,
		// destinations
		uint8_t* __restrict__ nbrListDelta8FF,
		uint8_t* __restrict__ nbrListDelta8FB,
		uint8_t* __restrict__ nbrListDelta8FR,
		uint16_t* __restrict__ nbrListDelta16FF,
		uint16_t* __restrict__ nbrListDelta16FB,
		uint16_t* __restrict__ nbrListDelta16FR,
		uint32_t* __restrict__ nbrListAbs32FF,
		uint32_t* __restrict__ nbrListAbs32FB,
		uint32_t* __restrict__ nbrListAbs32FR,
		uint32_t* __restrict__ overflowFlag)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_F) { return; }

		// particle i data
		// (global memory reads)
		const float x = xPosF[i];
		const float y = yPosF[i];
		const float z = zPosF[i];
		const uint32_t cF_src = compactCellIdxByParticleF[i];
		const uint64_t querySlotBase =
			static_cast<uint64_t>(cF_src) * static_cast<uint64_t>(queryCellSlotCount);

		// write ranges of packed SELL32 neighbor lists
		const uint32_t chunk = i / N_SellWidth;
		const uint32_t lane = i % N_SellWidth;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint64_t baseFR = nbrChunkOffsetsFR[chunk];
		const uint32_t pitchFF = static_cast<uint32_t>(nbrChunkMaxCountFF[chunk]);
		const uint32_t pitchFB = static_cast<uint32_t>(nbrChunkMaxCountFB[chunk]);
		const uint32_t pitchFR = static_cast<uint32_t>(nbrChunkMaxCountFR[chunk]);
		const uint32_t baseIdxFF = nbrChunkBaseIdxFF[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint32_t baseIdxFR = nbrChunkBaseIdxFR[chunk];
		const uint8_t formatFF = nbrChunkFormatFF[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint8_t formatFR = nbrChunkFormatFR[chunk];
		uint32_t wrFF = 0;
		uint32_t wrFB = 0;
		uint32_t wrFR = 0;

		// check all particles j in the cached neighboring grid cells
		for (uint32_t s = 0; s < queryCellSlotCount; s++)
		{
			const uint32_t cAny_nbr = queryCellRefAnyByCellF[querySlotBase + s];
			if (cAny_nbr == 0xffffffffu) { continue; }

			// fluid neighbor cell
			const uint32_t cF_nbr = cellRefFByAny[cAny_nbr];
			if (cF_nbr != 0xffffffffu)
			{
				// range of all fluid particles in the neighbor cell
				const uint32_t sF_nbr = cellStartIdxF[cF_nbr]; // start
				const uint32_t eF_nbr = cellStartIdxF[cF_nbr + 1]; // end
				if (sF_nbr >= eF_nbr) { continue; }

				for (uint32_t k = sF_nbr; k < eF_nbr; k++)
				{
					// index of fluid neighbor candidate
					const uint32_t j = k;

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
						StoreNeighbor_CompressedSell32(
							baseFF, pitchFF, lane, wrFF, j,
							baseIdxFF, formatFF,
							nbrListDelta8FF, nbrListDelta16FF, nbrListAbs32FF,
							overflowFlag);
					}
				}
			}

			// boundary neighbor cell
			const uint32_t cB_nbr = cellRefBByAny[cAny_nbr];
			if (cB_nbr != 0xffffffffu)
			{
				// range of all boundary particles in the neighbor cell
				const uint32_t sB_nbr = cellStartIdxB[cB_nbr]; // start
				const uint32_t eB_nbr = cellStartIdxB[cB_nbr + 1]; // end
				if (sB_nbr >= eB_nbr) { continue; }

				for (uint32_t k = sB_nbr; k < eB_nbr; k++)
				{
					// index of boundary neighbor candidate
					const uint32_t j = k;

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
						StoreNeighbor_CompressedSell32(
							baseFB, pitchFB, lane, wrFB, j,
							baseIdxFB, formatFB,
							nbrListDelta8FB, nbrListDelta16FB, nbrListAbs32FB,
							overflowFlag);
					}
				}
			}
		}

		// store all rigid neighbors in the neighboring grid cells
		if (c_ns_N_R > 0u)
		{
			const int cx = FloorToInt(x * c_ns_cellSizeInv);
			const int cy = FloorToInt(y * c_ns_cellSizeInv);
			const int cz = FloorToInt(z * c_ns_cellSizeInv);
			const int cellQueryReach = static_cast<int>(c_ns_cellQueryReach);

			for (int cellOffsetZ = -cellQueryReach;
				cellOffsetZ <= cellQueryReach; cellOffsetZ++)
			{
				for (int cellOffsetY = -cellQueryReach;
					cellOffsetY <= cellQueryReach; cellOffsetY++)
				{
					for (int cellOffsetX = -cellQueryReach;
						cellOffsetX <= cellQueryReach; cellOffsetX++)
					{
						const uint64_t cellId = MapToCellId(
							cx + cellOffsetX, cy + cellOffsetY, cz + cellOffsetZ);
						const int cellR = BinarySearchUniqueCellId(
							uniqueCellIdR, N_NonEmptyCellsR, cellId);
						if (cellR < 0) { continue; }

						const uint32_t begin = cellStartIdxR[cellR];
						const uint32_t end = cellStartIdxR[cellR + 1];
						for (uint32_t k = begin; k < end; k++)
						{
							const uint32_t j = sortedParticleIdxR[k];
							const float dx = x - xPosR[j];
							const float dy = y - yPosR[j];
							const float dz = z - zPosR[j];
							const float distSqu = dx * dx + dy * dy + dz * dz;

							if (distSqu <= c_ns_kernelRadiusSqu)
							{
								StoreNeighbor_CompressedSell32(
									baseFR, pitchFR, lane, wrFR, j,
									baseIdxFR, formatFR,
									nbrListDelta8FR, nbrListDelta16FR, nbrListAbs32FR,
									overflowFlag);
							}
						}
					}
				}
			}
		}
	}

	// ------------------------------------------------------------------------------------------------
	// --- [ rigid-source kernels ] -------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------

	__global__
	void K_BruteForce_Query_Count_RF_RB_RR(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const float* __restrict__ xPosR,
		const float* __restrict__ yPosR,
		const float* __restrict__ zPosR,
		const uint32_t* __restrict__ bodyIdR,
		// destinations
		uint16_t* __restrict__ nbrCountRF,
		uint16_t* __restrict__ nbrCountRB,
		uint16_t* __restrict__ nbrCountRR)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_ns_N_R) { return; }

		uint32_t cntRF = 0;
		uint32_t cntRB = 0;
		uint32_t cntRR = 0;

		// count fluid neighbors
		for (uint32_t j = 0; j < c_ns_N_F; j++)
		{
			const float dx = xPosR[i] - xPosF[j];
			const float dy = yPosR[i] - yPosF[j];
			const float dz = zPosR[i] - zPosF[j];
			if (dx * dx + dy * dy + dz * dz <= c_ns_kernelRadiusSqu)
			{
				cntRF++;
			}
		}

		// count static boundary neighbors
		for (uint32_t j = 0; j < c_ns_N_B; j++)
		{
			const float dx = xPosR[i] - xPosB[j];
			const float dy = yPosR[i] - yPosB[j];
			const float dz = zPosR[i] - zPosB[j];
			if (dx * dx + dy * dy + dz * dz <= c_ns_kernelRadiusSqu)
			{
				cntRB++;
			}
		}

		// count rigid neighbors attached to other bodies
		for (uint32_t j = 0; j < c_ns_N_R; j++)
		{
			if (j == i || bodyIdR[j] == bodyIdR[i]) { continue; }
			const float dx = xPosR[i] - xPosR[j];
			const float dy = yPosR[i] - yPosR[j];
			const float dz = zPosR[i] - zPosR[j];
			if (dx * dx + dy * dy + dz * dz <= c_ns_kernelRadiusSqu)
			{
				cntRR++;
			}
		}
		nbrCountRF[i] = static_cast<uint16_t>(cntRF);
		nbrCountRB[i] = static_cast<uint16_t>(cntRB);
		nbrCountRR[i] = static_cast<uint16_t>(cntRR);
	}

	__global__
	void K_BruteForce_Query_Fill_RF_RB_RR(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const float* __restrict__ xPosR,
		const float* __restrict__ yPosR,
		const float* __restrict__ zPosR,
		const uint32_t* __restrict__ bodyIdR,
		const uint64_t* __restrict__ nbrOffsetsRF,
		const uint64_t* __restrict__ nbrOffsetsRB,
		const uint64_t* __restrict__ nbrOffsetsRR,
		// destinations
		uint32_t* __restrict__ nbrListRF,
		uint32_t* __restrict__ nbrListRB,
		uint32_t* __restrict__ nbrListRR)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_ns_N_R) { return; }

		uint64_t wrRF = nbrOffsetsRF[i];
		uint64_t wrRB = nbrOffsetsRB[i];
		uint64_t wrRR = nbrOffsetsRR[i];

		// store fluid neighbors
		for (uint32_t j = 0; j < c_ns_N_F; j++)
		{
			const float dx = xPosR[i] - xPosF[j];
			const float dy = yPosR[i] - yPosF[j];
			const float dz = zPosR[i] - zPosF[j];
			if (dx * dx + dy * dy + dz * dz <= c_ns_kernelRadiusSqu)
			{
				nbrListRF[wrRF++] = j;
			}
		}

		// store static boundary neighbors
		for (uint32_t j = 0; j < c_ns_N_B; j++)
		{
			const float dx = xPosR[i] - xPosB[j];
			const float dy = yPosR[i] - yPosB[j];
			const float dz = zPosR[i] - zPosB[j];
			if (dx * dx + dy * dy + dz * dz <= c_ns_kernelRadiusSqu)
			{
				nbrListRB[wrRB++] = j;
			}
		}

		// store rigid neighbors attached to other bodies
		for (uint32_t j = 0; j < c_ns_N_R; j++)
		{
			if (j == i || bodyIdR[j] == bodyIdR[i]) { continue; }
			const float dx = xPosR[i] - xPosR[j];
			const float dy = yPosR[i] - yPosR[j];
			const float dz = zPosR[i] - zPosR[j];
			if (dx * dx + dy * dy + dz * dz <= c_ns_kernelRadiusSqu)
			{
				nbrListRR[wrRR++] = j;
			}
		}
	}

	__global__
	void K_Compact_Query_Count_RF_RB_RR(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const float* __restrict__ xPosR,
		const float* __restrict__ yPosR,
		const float* __restrict__ zPosR,
		const uint32_t* __restrict__ bodyIdR,
		const uint64_t* __restrict__ uniqueCellIdF,
		const uint32_t* __restrict__ cellStartIdxF,
		const uint64_t* __restrict__ uniqueCellIdB,
		const uint32_t* __restrict__ cellStartIdxB,
		const uint64_t* __restrict__ uniqueCellIdR,
		const uint32_t* __restrict__ cellStartIdxR,
		const uint32_t* __restrict__ sortedParticleIdxR,
		const uint32_t N_NonEmptyCellsF,
		const uint32_t N_NonEmptyCellsB,
		const uint32_t N_NonEmptyCellsR,
		// destinations
		uint16_t* __restrict__ nbrCountRF,
		uint16_t* __restrict__ nbrCountRB,
		uint16_t* __restrict__ nbrCountRR)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_ns_N_R) { return; }

		const float x = xPosR[i];
		const float y = yPosR[i];
		const float z = zPosR[i];
		const int cx = FloorToInt(x * c_ns_cellSizeInv);
		const int cy = FloorToInt(y * c_ns_cellSizeInv);
		const int cz = FloorToInt(z * c_ns_cellSizeInv);
		const int cellQueryReach = static_cast<int>(c_ns_cellQueryReach);
		uint32_t cntRF = 0;
		uint32_t cntRB = 0;
		uint32_t cntRR = 0;

		for (int cellOffsetZ = -cellQueryReach;
			cellOffsetZ <= cellQueryReach; cellOffsetZ++)
		{
			for (int cellOffsetY = -cellQueryReach;
				cellOffsetY <= cellQueryReach; cellOffsetY++)
			{
				for (int cellOffsetX = -cellQueryReach;
					cellOffsetX <= cellQueryReach; cellOffsetX++)
				{
					const uint64_t cellId = MapToCellId(
						cx + cellOffsetX, cy + cellOffsetY, cz + cellOffsetZ);

					// count fluid neighbors in this cell
					const int cellF = BinarySearchUniqueCellId(
						uniqueCellIdF, N_NonEmptyCellsF, cellId);
					if (cellF >= 0)
					{
						const uint32_t beginF = cellStartIdxF[cellF];
						const uint32_t endF = cellStartIdxF[cellF + 1];
						for (uint32_t j = beginF; j < endF; j++)
						{
							const float dx = x - xPosF[j];
							const float dy = y - yPosF[j];
							const float dz = z - zPosF[j];
							if (dx * dx + dy * dy + dz * dz <= c_ns_kernelRadiusSqu)
							{
								cntRF++;
							}
						}
					}

					// count static boundary neighbors in this cell
					const int cellB = BinarySearchUniqueCellId(
						uniqueCellIdB, N_NonEmptyCellsB, cellId);
					if (cellB >= 0)
					{
						const uint32_t beginB = cellStartIdxB[cellB];
						const uint32_t endB = cellStartIdxB[cellB + 1];
						for (uint32_t j = beginB; j < endB; j++)
						{
							const float dx = x - xPosB[j];
							const float dy = y - yPosB[j];
							const float dz = z - zPosB[j];
							if (dx * dx + dy * dy + dz * dz <= c_ns_kernelRadiusSqu)
							{
								cntRB++;
							}
						}
					}

					// count rigid neighbors attached to other bodies in this cell
					const int cellR = BinarySearchUniqueCellId(
						uniqueCellIdR, N_NonEmptyCellsR, cellId);
					if (cellR >= 0)
					{
						const uint32_t beginR = cellStartIdxR[cellR];
						const uint32_t endR = cellStartIdxR[cellR + 1];
						for (uint32_t k = beginR; k < endR; k++)
						{
							const uint32_t j = sortedParticleIdxR[k];
							if (j == i || bodyIdR[j] == bodyIdR[i]) { continue; }
							const float dx = x - xPosR[j];
							const float dy = y - yPosR[j];
							const float dz = z - zPosR[j];
							if (dx * dx + dy * dy + dz * dz <= c_ns_kernelRadiusSqu)
							{
								cntRR++;
							}
						}
					}
				}
			}
		}
		nbrCountRF[i] = static_cast<uint16_t>(cntRF);
		nbrCountRB[i] = static_cast<uint16_t>(cntRB);
		nbrCountRR[i] = static_cast<uint16_t>(cntRR);
	}

	__global__
	void K_Compact_Query_Fill_RF_RB_RR(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const float* __restrict__ xPosR,
		const float* __restrict__ yPosR,
		const float* __restrict__ zPosR,
		const uint32_t* __restrict__ bodyIdR,
		const uint64_t* __restrict__ uniqueCellIdF,
		const uint32_t* __restrict__ cellStartIdxF,
		const uint64_t* __restrict__ uniqueCellIdB,
		const uint32_t* __restrict__ cellStartIdxB,
		const uint64_t* __restrict__ uniqueCellIdR,
		const uint32_t* __restrict__ cellStartIdxR,
		const uint32_t* __restrict__ sortedParticleIdxR,
		const uint32_t N_NonEmptyCellsF,
		const uint32_t N_NonEmptyCellsB,
		const uint32_t N_NonEmptyCellsR,
		const uint64_t* __restrict__ nbrOffsetsRF,
		const uint64_t* __restrict__ nbrOffsetsRB,
		const uint64_t* __restrict__ nbrOffsetsRR,
		// destinations
		uint32_t* __restrict__ nbrListRF,
		uint32_t* __restrict__ nbrListRB,
		uint32_t* __restrict__ nbrListRR)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_ns_N_R) { return; }

		const float x = xPosR[i];
		const float y = yPosR[i];
		const float z = zPosR[i];
		const int cx = FloorToInt(x * c_ns_cellSizeInv);
		const int cy = FloorToInt(y * c_ns_cellSizeInv);
		const int cz = FloorToInt(z * c_ns_cellSizeInv);
		const int cellQueryReach = static_cast<int>(c_ns_cellQueryReach);
		uint64_t wrRF = nbrOffsetsRF[i];
		uint64_t wrRB = nbrOffsetsRB[i];
		uint64_t wrRR = nbrOffsetsRR[i];

		for (int cellOffsetZ = -cellQueryReach;
			cellOffsetZ <= cellQueryReach; cellOffsetZ++)
		{
			for (int cellOffsetY = -cellQueryReach;
				cellOffsetY <= cellQueryReach; cellOffsetY++)
			{
				for (int cellOffsetX = -cellQueryReach;
					cellOffsetX <= cellQueryReach; cellOffsetX++)
				{
					const uint64_t cellId = MapToCellId(
						cx + cellOffsetX, cy + cellOffsetY, cz + cellOffsetZ);

					// store fluid neighbors in this cell
					const int cellF = BinarySearchUniqueCellId(
						uniqueCellIdF, N_NonEmptyCellsF, cellId);
					if (cellF >= 0)
					{
						const uint32_t beginF = cellStartIdxF[cellF];
						const uint32_t endF = cellStartIdxF[cellF + 1];
						for (uint32_t j = beginF; j < endF; j++)
						{
							const float dx = x - xPosF[j];
							const float dy = y - yPosF[j];
							const float dz = z - zPosF[j];
							if (dx * dx + dy * dy + dz * dz <= c_ns_kernelRadiusSqu)
							{
								nbrListRF[wrRF++] = j;
							}
						}
					}

					// store static boundary neighbors in this cell
					const int cellB = BinarySearchUniqueCellId(
						uniqueCellIdB, N_NonEmptyCellsB, cellId);
					if (cellB >= 0)
					{
						const uint32_t beginB = cellStartIdxB[cellB];
						const uint32_t endB = cellStartIdxB[cellB + 1];
						for (uint32_t j = beginB; j < endB; j++)
						{
							const float dx = x - xPosB[j];
							const float dy = y - yPosB[j];
							const float dz = z - zPosB[j];
							if (dx * dx + dy * dy + dz * dz <= c_ns_kernelRadiusSqu)
							{
								nbrListRB[wrRB++] = j;
							}
						}
					}

					// store rigid neighbors attached to other bodies in this cell
					const int cellR = BinarySearchUniqueCellId(
						uniqueCellIdR, N_NonEmptyCellsR, cellId);
					if (cellR >= 0)
					{
						const uint32_t beginR = cellStartIdxR[cellR];
						const uint32_t endR = cellStartIdxR[cellR + 1];
						for (uint32_t k = beginR; k < endR; k++)
						{
							const uint32_t j = sortedParticleIdxR[k];
							if (j == i || bodyIdR[j] == bodyIdR[i]) { continue; }
							const float dx = x - xPosR[j];
							const float dy = y - yPosR[j];
							const float dz = z - zPosR[j];
							if (dx * dx + dy * dy + dz * dz <= c_ns_kernelRadiusSqu)
							{
								nbrListRR[wrRR++] = j;
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
		const float* xPosR, const float* yPosR, const float* zPosR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		uint16_t* nbrCountFF,
		uint16_t* nbrCountFB,
		uint16_t* nbrCountFR,
		uint32_t* nbrChunkMinRefFF,
		uint32_t* nbrChunkMaxRefFF,
		uint32_t* nbrChunkMinRefFB,
		uint32_t* nbrChunkMaxRefFB,
		uint32_t* nbrChunkMinRefFR,
		uint32_t* nbrChunkMaxRefFR)
	{
		(void)N_B;
		(void)N_R;

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_BruteForce_Query_Count<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB,
			xPosR, yPosR, zPosR,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkMinRefFF, nbrChunkMaxRefFF,
			nbrChunkMinRefFB, nbrChunkMaxRefFB,
			nbrChunkMinRefFR, nbrChunkMaxRefFR);

		GPU::CheckLastKernel("K_BruteForce_Query_Count()");
	}

	void Launch_BruteForce_Query_Fill(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		const uint16_t* nbrChunkMaxCountFF,
		const uint16_t* nbrChunkMaxCountFB,
		const uint16_t* nbrChunkMaxCountFR,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint64_t* nbrChunkOffsetsFR,
		const uint32_t* nbrChunkBaseIdxFF,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint32_t* nbrChunkBaseIdxFR,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrChunkFormatFR,
		// destinations
		uint8_t* nbrListDelta8FF,
		uint8_t* nbrListDelta8FB,
		uint8_t* nbrListDelta8FR,
		uint16_t* nbrListDelta16FF,
		uint16_t* nbrListDelta16FB,
		uint16_t* nbrListDelta16FR,
		uint32_t* nbrListAbs32FF,
		uint32_t* nbrListAbs32FB,
		uint32_t* nbrListAbs32FR,
		uint32_t* overflowFlag)
	{
		(void)N_B;
		(void)N_R;

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_BruteForce_Query_Fill<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB,
			xPosR, yPosR, zPosR,
			nbrChunkMaxCountFF, nbrChunkMaxCountFB, nbrChunkMaxCountFR,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrChunkOffsetsFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR,
			nbrListDelta8FF, nbrListDelta8FB, nbrListDelta8FR,
			nbrListDelta16FF, nbrListDelta16FB, nbrListDelta16FR,
			nbrListAbs32FF, nbrListAbs32FB, nbrListAbs32FR,
			overflowFlag);

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

	void Launch_FillMergedCompactCellRefs(
		const uint64_t* uniqueCellIdAny,
		const uint32_t N_NonEmptyCellsAny,
		const uint64_t* uniqueCellIdF,
		const uint64_t* uniqueCellIdB,
		const uint32_t N_NonEmptyCellsF,
		const uint32_t N_NonEmptyCellsB,
		cudaStream_t stream,
		// destinations
		uint32_t* cellRefFByAny,
		uint32_t* cellRefBByAny)
	{
		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_NonEmptyCellsAny + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_FillMergedCompactCellRefs<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			uniqueCellIdAny, N_NonEmptyCellsAny,
			uniqueCellIdF, uniqueCellIdB,
			N_NonEmptyCellsF, N_NonEmptyCellsB,
			cellRefFByAny, cellRefBByAny);

		GPU::CheckLastKernel("K_FillMergedCompactCellRefs()");
	}

	void Launch_FillQueryLookupBoundsF(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const uint32_t* cellStartIdxF,
		const uint64_t* uniqueCellIdAny,
		const uint32_t N_NonEmptyCellsF,
		const uint32_t N_NonEmptyCellsAny,
		const uint32_t cellQueryReach,
		cudaStream_t stream,
		// destinations
		uint32_t* compactCellIdxByParticleF,
		uint32_t* queryLoAnyByCellF,
		uint32_t* queryHiAnyByCellF)
	{
		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_NonEmptyCellsF + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_FillQueryLookupBoundsF<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF,
			cellStartIdxF, uniqueCellIdAny,
			N_NonEmptyCellsF, N_NonEmptyCellsAny, cellQueryReach,
			compactCellIdxByParticleF,
			queryLoAnyByCellF, queryHiAnyByCellF);

		GPU::CheckLastKernel("K_FillQueryLookupBoundsF()");
	}

	void Launch_FillQueryCellRefsByCellF(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const uint32_t* cellStartIdxF,
		const uint64_t* uniqueCellIdAny,
		const uint32_t* queryLoAnyByCellF,
		const uint32_t* queryHiAnyByCellF,
		const uint32_t N_NonEmptyCellsF,
		const uint32_t queryCellSlotCount,
		cudaStream_t stream,
		// destinations
		uint32_t* queryCellRefAnyByCellF)
	{
		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_NonEmptyCellsF + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_FillQueryCellRefsByCellF<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF,
			cellStartIdxF, uniqueCellIdAny,
			queryLoAnyByCellF, queryHiAnyByCellF,
			N_NonEmptyCellsF, queryCellSlotCount,
			queryCellRefAnyByCellF);

		GPU::CheckLastKernel("K_FillQueryCellRefsByCellF()");
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
		const float* xPosR, const float* yPosR, const float* zPosR,
		const uint32_t* cellRefFByAny, const uint32_t* cellRefBByAny,
		const uint32_t* cellStartIdxF, const uint32_t* cellStartIdxB,
		const uint32_t* compactCellIdxByParticleF,
		const uint32_t* queryCellRefAnyByCellF,
		const uint64_t* uniqueCellIdR,
		const uint32_t* cellStartIdxR,
		const uint32_t* sortedParticleIdxR,
		const uint32_t N_NonEmptyCellsR,
		const uint32_t queryCellSlotCount,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		uint16_t* nbrCountFF,
		uint16_t* nbrCountFB,
		uint16_t* nbrCountFR,
		uint32_t* nbrChunkMinRefFF,
		uint32_t* nbrChunkMaxRefFF,
		uint32_t* nbrChunkMinRefFB,
		uint32_t* nbrChunkMaxRefFB,
		uint32_t* nbrChunkMinRefFR,
		uint32_t* nbrChunkMaxRefFR)
	{
		(void)N_B;
		(void)N_R;

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_CellLinkedList_Compact_Query_Count<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB,
			xPosR, yPosR, zPosR,
			cellRefFByAny, cellRefBByAny,
			cellStartIdxF, cellStartIdxB,
			compactCellIdxByParticleF,
			queryCellRefAnyByCellF, queryCellSlotCount,
			uniqueCellIdR, cellStartIdxR, sortedParticleIdxR,
			N_NonEmptyCellsR,
			N_F,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkMinRefFF, nbrChunkMaxRefFF,
			nbrChunkMinRefFB, nbrChunkMaxRefFB,
			nbrChunkMinRefFR, nbrChunkMaxRefFR);

		GPU::CheckLastKernel("K_CellLinkedList_Compact_Query_Count()");
	}

	void Launch_CellLinkedList_Compact_Query_Fill(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const uint32_t* cellRefFByAny, const uint32_t* cellRefBByAny,
		const uint32_t* cellStartIdxF, const uint32_t* cellStartIdxB,
		const uint32_t* compactCellIdxByParticleF,
		const uint32_t* queryCellRefAnyByCellF,
		const uint64_t* uniqueCellIdR,
		const uint32_t* cellStartIdxR,
		const uint32_t* sortedParticleIdxR,
		const uint32_t N_NonEmptyCellsR,
		const uint32_t queryCellSlotCount,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		const uint16_t* nbrChunkMaxCountFF,
		const uint16_t* nbrChunkMaxCountFB,
		const uint16_t* nbrChunkMaxCountFR,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint64_t* nbrChunkOffsetsFR,
		const uint32_t* nbrChunkBaseIdxFF,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint32_t* nbrChunkBaseIdxFR,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrChunkFormatFR,
		// destinations
		uint8_t* nbrListDelta8FF,
		uint8_t* nbrListDelta8FB,
		uint8_t* nbrListDelta8FR,
		uint16_t* nbrListDelta16FF,
		uint16_t* nbrListDelta16FB,
		uint16_t* nbrListDelta16FR,
		uint32_t* nbrListAbs32FF,
		uint32_t* nbrListAbs32FB,
		uint32_t* nbrListAbs32FR,
		uint32_t* overflowFlag)
	{
		(void)N_B;
		(void)N_R;

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_CellLinkedList_Compact_Query_Fill<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB,
			xPosR, yPosR, zPosR,
			cellRefFByAny, cellRefBByAny,
			cellStartIdxF, cellStartIdxB,
			compactCellIdxByParticleF,
			queryCellRefAnyByCellF, queryCellSlotCount,
			uniqueCellIdR, cellStartIdxR, sortedParticleIdxR,
			N_NonEmptyCellsR,
			nbrChunkMaxCountFF, nbrChunkMaxCountFB, nbrChunkMaxCountFR,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrChunkOffsetsFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR,
			N_F,
			nbrListDelta8FF, nbrListDelta8FB, nbrListDelta8FR,
			nbrListDelta16FF, nbrListDelta16FB, nbrListDelta16FR,
			nbrListAbs32FF, nbrListAbs32FB, nbrListAbs32FR,
			overflowFlag);

		GPU::CheckLastKernel("K_CellLinkedList_Compact_Query_Fill()");
	}

	void Launch_BruteForce_Query_Count_RF_RB_RR(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const uint32_t* bodyIdR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		uint16_t* nbrCountRF,
		uint16_t* nbrCountRB,
		uint16_t* nbrCountRR)
	{
		(void)N_F;
		(void)N_B;

		const uint32_t N_BLOCKSIZE = 256;
		if (N_R > 0)
		{
			const uint32_t N_GRIDSIZE = (N_R + N_BLOCKSIZE - 1) / N_BLOCKSIZE;
			K_BruteForce_Query_Count_RF_RB_RR<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
				xPosF, yPosF, zPosF,
				xPosB, yPosB, zPosB,
				xPosR, yPosR, zPosR, bodyIdR,
				nbrCountRF, nbrCountRB, nbrCountRR);
			GPU::CheckLastKernel("K_BruteForce_Query_Count_RF_RB_RR()");
		}
	}

	void Launch_BruteForce_Query_Fill_RF_RB_RR(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const uint32_t* bodyIdR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		const uint64_t* nbrOffsetsRF,
		const uint64_t* nbrOffsetsRB,
		const uint64_t* nbrOffsetsRR,
		// destinations
		uint32_t* nbrListRF,
		uint32_t* nbrListRB,
		uint32_t* nbrListRR)
	{
		(void)N_F;
		(void)N_B;

		const uint32_t N_BLOCKSIZE = 256;
		if (N_R > 0)
		{
			const uint32_t N_GRIDSIZE = (N_R + N_BLOCKSIZE - 1) / N_BLOCKSIZE;
			K_BruteForce_Query_Fill_RF_RB_RR<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
				xPosF, yPosF, zPosF,
				xPosB, yPosB, zPosB,
				xPosR, yPosR, zPosR, bodyIdR,
				nbrOffsetsRF, nbrOffsetsRB, nbrOffsetsRR,
				nbrListRF, nbrListRB, nbrListRR);
			GPU::CheckLastKernel("K_BruteForce_Query_Fill_RF_RB_RR()");
		}
	}

	void Launch_CellLinkedList_Compact_Query_Count_RF_RB_RR(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const uint32_t* bodyIdR,
		const uint64_t* uniqueCellIdF, const uint32_t* cellStartIdxF,
		const uint64_t* uniqueCellIdB, const uint32_t* cellStartIdxB,
		const uint64_t* uniqueCellIdR, const uint32_t* cellStartIdxR,
		const uint32_t* sortedParticleIdxR,
		const uint32_t N_NonEmptyCellsF, const uint32_t N_NonEmptyCellsB,
		const uint32_t N_NonEmptyCellsR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		uint16_t* nbrCountRF,
		uint16_t* nbrCountRB,
		uint16_t* nbrCountRR)
	{
		(void)N_F;
		(void)N_B;

		const uint32_t N_BLOCKSIZE = 256;
		if (N_R > 0)
		{
			const uint32_t N_GRIDSIZE = (N_R + N_BLOCKSIZE - 1) / N_BLOCKSIZE;
			K_Compact_Query_Count_RF_RB_RR<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
				xPosF, yPosF, zPosF,
				xPosB, yPosB, zPosB,
				xPosR, yPosR, zPosR, bodyIdR,
				uniqueCellIdF, cellStartIdxF,
				uniqueCellIdB, cellStartIdxB,
				uniqueCellIdR, cellStartIdxR, sortedParticleIdxR,
				N_NonEmptyCellsF, N_NonEmptyCellsB, N_NonEmptyCellsR,
				nbrCountRF, nbrCountRB, nbrCountRR);
			GPU::CheckLastKernel("K_Compact_Query_Count_RF_RB_RR()");
		}
	}

	void Launch_CellLinkedList_Compact_Query_Fill_RF_RB_RR(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const uint32_t* bodyIdR,
		const uint64_t* uniqueCellIdF, const uint32_t* cellStartIdxF,
		const uint64_t* uniqueCellIdB, const uint32_t* cellStartIdxB,
		const uint64_t* uniqueCellIdR, const uint32_t* cellStartIdxR,
		const uint32_t* sortedParticleIdxR,
		const uint32_t N_NonEmptyCellsF, const uint32_t N_NonEmptyCellsB,
		const uint32_t N_NonEmptyCellsR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		const uint64_t* nbrOffsetsRF,
		const uint64_t* nbrOffsetsRB,
		const uint64_t* nbrOffsetsRR,
		// destinations
		uint32_t* nbrListRF,
		uint32_t* nbrListRB,
		uint32_t* nbrListRR)
	{
		(void)N_F;
		(void)N_B;

		const uint32_t N_BLOCKSIZE = 256;
		if (N_R > 0)
		{
			const uint32_t N_GRIDSIZE = (N_R + N_BLOCKSIZE - 1) / N_BLOCKSIZE;
			K_Compact_Query_Fill_RF_RB_RR<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
				xPosF, yPosF, zPosF,
				xPosB, yPosB, zPosB,
				xPosR, yPosR, zPosR, bodyIdR,
				uniqueCellIdF, cellStartIdxF,
				uniqueCellIdB, cellStartIdxB,
				uniqueCellIdR, cellStartIdxR, sortedParticleIdxR,
				N_NonEmptyCellsF, N_NonEmptyCellsB, N_NonEmptyCellsR,
				nbrOffsetsRF, nbrOffsetsRB, nbrOffsetsRR,
				nbrListRF, nbrListRB, nbrListRR);
			GPU::CheckLastKernel("K_Compact_Query_Fill_RF_RB_RR()");
		}
	}

} // namespace GPU01
