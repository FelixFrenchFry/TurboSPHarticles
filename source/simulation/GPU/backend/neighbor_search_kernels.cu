// neighbor_search_kernels.cu

// external
#include <cuda_runtime.h>

// internal
#include "source/simulation/GPU/gpu_utils.h"
#include "neighbor_search_kernels.cuh"
#include "neighbor_search.cuh"



namespace GPU
{
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
	__constant__ int32_t c_ns_cellOrigX;
	__constant__ int32_t c_ns_cellOrigY;
	__constant__ int32_t c_ns_cellOrigZ;
	__constant__ uint32_t c_ns_N_dimX;
	__constant__ uint32_t c_ns_N_dimY;
	__constant__ uint32_t c_ns_N_dimZ;
	__constant__ uint32_t c_ns_N_cells;
	__constant__ uint32_t c_ns_EMPTY;

	void NS_ReloadConstants(
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		const float kernelRadius, const float cellSize, const uint32_t cellQueryReach,
		const float xBoxOrig, const float yBoxOrig, const float zBoxOrig,
		const Sim::CellCoordFrame& cellFrame,
		const uint32_t N_cells, const uint32_t EMPTY,
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
		GPU::Check(cudaMemcpyToSymbol(c_ns_cellOrigX, &cellFrame.cellOrigX, sizeof(int32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_cellOrigY, &cellFrame.cellOrigY, sizeof(int32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_cellOrigZ, &cellFrame.cellOrigZ, sizeof(int32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_N_dimX, &cellFrame.N_dimX, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_N_dimY, &cellFrame.N_dimY, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_N_dimZ, &cellFrame.N_dimZ, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_N_cells, &N_cells, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_ns_EMPTY, &EMPTY, sizeof(uint32_t)));

		GPU::StreamSync(stream);
	}

	// ------------------------------------------------------------------------------------------------
	// --- [ helpers ] --------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------

	__device__ __forceinline__
	uint32_t MapToCellId_Basic(uint32_t cx, uint32_t cy, uint32_t cz)
	{
		return cx + c_ns_N_dimX * (cy + c_ns_N_dimY * cz);
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

	static_assert(N_SELL_WIDTH == 32u,
		"Chunked SELL32 chunk meta reduction assumes one chunk per warp.");

	// every lane of the warp has to reach this, the shuffles span the full chunk
	__device__ __forceinline__
	void CommitChunkCompressionMeta(
		const uint32_t chunk,
		const bool isChunkOwner,
		uint32_t pitch,
		uint32_t minRef,
		uint32_t maxRef,
		uint16_t* __restrict__ chunkMaxCount,
		uint32_t* __restrict__ chunkBaseIdx,
		uint8_t* __restrict__ chunkFormat)
	{
		// match per-particle counts before reducing the chunk pitch
		pitch = static_cast<uint16_t>(pitch);

		for (uint32_t offset = N_SELL_WIDTH / 2u; offset != 0u; offset >>= 1u)
		{
			const uint32_t otherPitch = __shfl_down_sync(0xffffffffu, pitch, offset);
			const uint32_t otherMinRef = __shfl_down_sync(0xffffffffu, minRef, offset);
			const uint32_t otherMaxRef = __shfl_down_sync(0xffffffffu, maxRef, offset);
			pitch = (otherPitch > pitch) ? otherPitch : pitch;
			minRef = (otherMinRef < minRef) ? otherMinRef : minRef;
			maxRef = (otherMaxRef > maxRef) ? otherMaxRef : maxRef;
		}

		if (!isChunkOwner) { return; }

		chunkMaxCount[chunk] = static_cast<uint16_t>(pitch);

		if (pitch == 0u)
		{
			chunkBaseIdx[chunk] = 0u;
			chunkFormat[chunk] = static_cast<uint8_t>(NbrChunkFormat::Abs32);
			return;
		}

		const uint32_t span = (maxRef >= minRef) ? (maxRef - minRef) : 0xffffffffu;
		const bool canUseDelta16 =
			(maxRef >= minRef) && (span <= 0xffffu);
		const bool canUseDelta8 =
			(maxRef >= minRef) && (span <= 0xffu);

		chunkBaseIdx[chunk] = minRef;
		chunkFormat[chunk] = canUseDelta8
			? static_cast<uint8_t>(NbrChunkFormat::Delta8)
			: (canUseDelta16
				? static_cast<uint8_t>(NbrChunkFormat::Delta16)
				: static_cast<uint8_t>(NbrChunkFormat::Abs32));
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
			const uint64_t linearIdx = wr * N_SELL_WIDTH + lane;

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
		uint16_t* __restrict__ nbrChunkMaxCountFF,
		uint16_t* __restrict__ nbrChunkMaxCountFB,
		uint16_t* __restrict__ nbrChunkMaxCountFR,
		uint32_t* __restrict__ nbrChunkBaseIdxFF,
		uint32_t* __restrict__ nbrChunkBaseIdxFB,
		uint32_t* __restrict__ nbrChunkBaseIdxFR,
		uint8_t* __restrict__ nbrChunkFormatFF,
		uint8_t* __restrict__ nbrChunkFormatFB,
		uint8_t* __restrict__ nbrChunkFormatFR)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		const uint32_t chunk = i / N_SELL_WIDTH;
		// one chunk per warp, only lane 0 writes the metadata after the reduction
		// (lanes 1-31 only feed the reduction that produces it, but do not write)
		const bool isChunkOwner = (i < c_ns_N_F) && ((threadIdx.x % N_SELL_WIDTH) == 0u);

		uint32_t cntFF = 0;
		uint32_t cntFB = 0;
		uint32_t cntFR = 0;
		uint32_t minRefFF = 0xffffffffu;
		uint32_t maxRefFF = 0u;
		uint32_t minRefFB = 0xffffffffu;
		uint32_t maxRefFB = 0u;
		uint32_t minRefFR = 0xffffffffu;
		uint32_t maxRefFR = 0u;

		if (i < c_ns_N_F)
		{
			// particle i data
			const float xi = xPosF[i];
			const float yi = yPosF[i];
			const float zi = zPosF[i];

			// check fluid candidates (all particles)
			for (uint32_t j = 0; j < c_ns_N_F; j++)
			{
				const float dx = xi - xPosF[j];
				const float dy = yi - yPosF[j];
				const float dz = zi - zPosF[j];
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
				const float dx = xi - xPosB[j];
				const float dy = yi - yPosB[j];
				const float dz = zi - zPosB[j];
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
				const float dx = xi - xPosR[j];
				const float dy = yi - yPosR[j];
				const float dz = zi - zPosR[j];
				const float distSqu = dx * dx + dy * dy + dz * dz;

				if (distSqu <= c_ns_kernelRadiusSqu) [[unlikely]] // neighbor found
				{
					cntFR++;
					minRefFR = (j < minRefFR) ? j : minRefFR;
					maxRefFR = (j > maxRefFR) ? j : maxRefFR;
				}
			}

			// update neighbor counts
			nbrCountFF[i] = static_cast<uint16_t>(cntFF);
			nbrCountFB[i] = static_cast<uint16_t>(cntFB);
			nbrCountFR[i] = static_cast<uint16_t>(cntFR);
		}

		CommitChunkCompressionMeta(
			chunk, isChunkOwner, cntFF, minRefFF, maxRefFF,
			nbrChunkMaxCountFF, nbrChunkBaseIdxFF, nbrChunkFormatFF);
		CommitChunkCompressionMeta(
			chunk, isChunkOwner, cntFB, minRefFB, maxRefFB,
			nbrChunkMaxCountFB, nbrChunkBaseIdxFB, nbrChunkFormatFB);
		CommitChunkCompressionMeta(
			chunk, isChunkOwner, cntFR, minRefFR, maxRefFR,
			nbrChunkMaxCountFR, nbrChunkBaseIdxFR, nbrChunkFormatFR);
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
		const uint32_t chunk = i / N_SELL_WIDTH;
		const uint32_t lane = i % N_SELL_WIDTH;
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
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];
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
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];
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
			const float dx = xi - xPosR[j];
			const float dy = yi - yPosR[j];
			const float dz = zi - zPosR[j];
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

		// particle i data
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];

		// cell coordinates of particle i
		const int cxi = FloorToInt((xi - c_ns_xBoxOrig) * c_ns_cellSizeInv);
		const int cyi = FloorToInt((yi - c_ns_yBoxOrig) * c_ns_cellSizeInv);
		const int czi = FloorToInt((zi - c_ns_zBoxOrig) * c_ns_cellSizeInv);

		const uint32_t cx = ClampInt(cxi, 0, c_ns_N_dimX - 1);
		const uint32_t cy = ClampInt(cyi, 0, c_ns_N_dimY - 1);
		const uint32_t cz = ClampInt(czi, 0, c_ns_N_dimZ - 1);

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

		const uint32_t cx = ClampInt(cxi, 0, c_ns_N_dimX - 1);
		const uint32_t cy = ClampInt(cyi, 0, c_ns_N_dimY - 1);
		const uint32_t cz = ClampInt(czi, 0, c_ns_N_dimZ - 1);

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
		const uint32_t N_cells,
		// destinations
		uint32_t* __restrict__ cellStartIdx,
		uint32_t* __restrict__ cellEndIdx)
	{
		uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_cells) { return; }

		cellStartIdx[i] = c_ns_EMPTY;
		cellEndIdx[i] = c_ns_EMPTY;
	}

} // namespace GPU

// -------------------------------------------------------------------------------------------------
// --- [ cell linked list compact implementation ] -------------------------------------------------
// -------------------------------------------------------------------------------------------------

namespace GPU
{
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
		const uint32_t lx, const uint32_t ly, const uint32_t lz)
	{
		const uint64_t xx = SpreadBitsWithGaps_21to63(lx); // bits at pos 0,3,6,...
		const uint64_t yy = SpreadBitsWithGaps_21to63(ly) << 1; // bits at pos 1,4,7,...
		const uint64_t zz = SpreadBitsWithGaps_21to63(lz) << 2; // bits at pos 2,5,8,...

		return xx | yy | zz; // interleaved 3D Morton/Z-order code
	}

	__device__ __forceinline__
	uint64_t MapToCellId(
		const int cx, const int cy, const int cz)
	{
		const int lx = cx - c_ns_cellOrigX;
		const int ly = cy - c_ns_cellOrigY;
		const int lz = cz - c_ns_cellOrigZ;

		// particles outside the frame left the domain and stay out of the grid
		if (lx < 0 || static_cast<uint32_t>(lx) >= c_ns_N_dimX ||
			ly < 0 || static_cast<uint32_t>(ly) >= c_ns_N_dimY ||
			lz < 0 || static_cast<uint32_t>(lz) >= c_ns_N_dimZ)
		{
			return Sim::EMPTY_64;
		}

		return MapToCellId_Morton(
			static_cast<uint32_t>(lx),
			static_cast<uint32_t>(ly),
			static_cast<uint32_t>(lz));
	}

	__device__
	int BinarySearchUniqueCellId(
		const uint64_t* uniqueCellId,
		uint32_t N_nonEmptyCells, uint64_t cid)
	{
		int low = 0;
		int high = static_cast<int>(N_nonEmptyCells) - 1;

		while (low <= high)
		{
			int mid = low + ((high - low) >> 1);
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
		uint32_t N_nonEmptyCells, uint64_t cid)
	{
		uint32_t lo = 0;
		uint32_t hi = N_nonEmptyCells;

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
		uint32_t N_nonEmptyCells, uint64_t cid)
	{
		uint32_t lo = 0;
		uint32_t hi = N_nonEmptyCells;

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
	uint32_t BinarySearchUniqueCellId(
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
				return mid;
			}
		}

		return Sim::EMPTY_32;
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
		const uint32_t N_nonEmptyCells,
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
			cellStartIdx[N_nonEmptyCells] = N;
		}
	}

	__global__
	void K_FillMergedCompactCellRanges(
		const uint64_t* __restrict__ uniqueCellIdAny,
		const uint32_t N_nonEmptyCellsAny,
		const uint64_t* __restrict__ uniqueCellIdF,
		const uint64_t* __restrict__ uniqueCellIdB,
		const uint32_t* __restrict__ cellStartIdxF,
		const uint32_t* __restrict__ cellStartIdxB,
		const uint32_t N_nonEmptyCellsF,
		const uint32_t N_nonEmptyCellsB,
		// destinations
		CellRangeByAny* __restrict__ cellRangeByAny)
	{
		const uint32_t cAny = blockIdx.x * blockDim.x + threadIdx.x;
		if (cAny >= N_nonEmptyCellsAny) { return; }

		const uint64_t cid = uniqueCellIdAny[cAny];
		const int cF = BinarySearchUniqueCellId(uniqueCellIdF, N_nonEmptyCellsF, cid);
		const int cB = BinarySearchUniqueCellId(uniqueCellIdB, N_nonEmptyCellsB, cid);

		CellRangeByAny cellRange{ 0u, 0u, 0u, 0u };
		if (cF >= 0)
		{
			cellRange.beginF = cellStartIdxF[cF];
			cellRange.endF = cellStartIdxF[cF + 1];
		}
		if (cB >= 0)
		{
			cellRange.beginB = cellStartIdxB[cB];
			cellRange.endB = cellStartIdxB[cB + 1];
		}

		cellRangeByAny[cAny] = cellRange;
	}

	__global__
	void K_FillQueryLookupBoundsF(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const uint32_t* __restrict__ cellStartIdxF,
		const uint64_t* __restrict__ uniqueCellIdAny,
		const uint32_t N_nonEmptyCellsF,
		const uint32_t N_nonEmptyCellsAny,
		const uint32_t cellQueryReach,
		// destinations
		uint32_t* __restrict__ compactCellIdxByParticleF,
		uint32_t* __restrict__ queryLoAnyByCellF,
		uint32_t* __restrict__ queryHiAnyByCellF)
	{
		const uint32_t cF = blockIdx.x * blockDim.x + threadIdx.x;
		if (cF >= N_nonEmptyCellsF) { return; }

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

					// stencil cells outside the frame hold no particles
					if (cid_nbr == Sim::EMPTY_64) { continue; }

					litMaxCellId = (cid_nbr < litMaxCellId) ? cid_nbr : litMaxCellId;
					bigMinCellId = (cid_nbr > bigMinCellId) ? cid_nbr : bigMinCellId;
				}
			}
		}

		const uint32_t loAny = LowerBoundUniqueCellId(
			uniqueCellIdAny, N_nonEmptyCellsAny, litMaxCellId);
		uint32_t hiAny = UpperBoundUniqueCellId(
			uniqueCellIdAny, N_nonEmptyCellsAny, bigMinCellId);
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
		const uint32_t N_nonEmptyCellsF,
		const uint32_t N_queryCellSlots,
		// destinations
		uint32_t* __restrict__ queryCellRefAnyByCellF)
	{
		const uint32_t cF = blockIdx.x * blockDim.x + threadIdx.x;
		if (cF >= N_nonEmptyCellsF) { return; }

		const uint32_t srcBegin = cellStartIdxF[cF];
		const uint32_t srcEnd = cellStartIdxF[cF + 1];
		const uint64_t slotBase =
			static_cast<uint64_t>(cF) * static_cast<uint64_t>(N_queryCellSlots);

		if (srcBegin >= srcEnd)
		{
			for (uint32_t s = 0; s < N_queryCellSlots; s++)
			{
				queryCellRefAnyByCellF[slotBase + s] = Sim::EMPTY_32;
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
					uint32_t cAny_nbr = Sim::EMPTY_32;

					// cells outside the frame are never in the search range
					if (loAny < hiAny)
					{
						const uint64_t cid_nbr = MapToCellId(cx_nbr, cy_nbr, cz_nbr);
						const uint32_t cAny = BinarySearchUniqueCellId(
							uniqueCellIdAny, loAny, hiAny, cid_nbr);
						if (cAny != Sim::EMPTY_32) { cAny_nbr = cAny; }
					}

					if (s < N_queryCellSlots)
					{
						queryCellRefAnyByCellF[slotBase + s] = cAny_nbr;
					}
					s++;
				}
			}
		}

		for (; s < N_queryCellSlots; s++)
		{
			queryCellRefAnyByCellF[slotBase + s] = Sim::EMPTY_32;
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
		const CellRangeByAny* __restrict__ cellRangeByAny,
		const uint32_t* __restrict__ compactCellIdxByParticleF,
		const uint32_t* __restrict__ queryCellRefAnyByCellF,
		const uint32_t N_queryCellSlots,
		const uint64_t* __restrict__ uniqueCellIdR,
		const uint32_t* __restrict__ cellStartIdxR,
		const uint32_t* __restrict__ sortedParticleIdxR,
		const uint32_t N_nonEmptyCellsR,
		const uint32_t N_F,
		// destinations
		uint16_t* __restrict__ nbrCountFF,
		uint16_t* __restrict__ nbrCountFB,
		uint16_t* __restrict__ nbrCountFR,
		uint16_t* __restrict__ nbrChunkMaxCountFF,
		uint16_t* __restrict__ nbrChunkMaxCountFB,
		uint16_t* __restrict__ nbrChunkMaxCountFR,
		uint32_t* __restrict__ nbrChunkBaseIdxFF,
		uint32_t* __restrict__ nbrChunkBaseIdxFB,
		uint32_t* __restrict__ nbrChunkBaseIdxFR,
		uint8_t* __restrict__ nbrChunkFormatFF,
		uint8_t* __restrict__ nbrChunkFormatFB,
		uint8_t* __restrict__ nbrChunkFormatFR)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		const uint32_t chunk = i / N_SELL_WIDTH;
		// one chunk per warp, only lane 0 writes the metadata after the reduction
		// (lanes 1-31 only feed the reduction that produces it, but do not write)
		const bool isChunkOwner = (i < N_F) && ((threadIdx.x % N_SELL_WIDTH) == 0u);

		uint32_t cntFF = 0;
		uint32_t cntFB = 0;
		uint32_t cntFR = 0;
		uint32_t minRefFF = 0xffffffffu;
		uint32_t maxRefFF = 0u;
		uint32_t minRefFB = 0xffffffffu;
		uint32_t maxRefFB = 0u;
		uint32_t minRefFR = 0xffffffffu;
		uint32_t maxRefFR = 0u;

		if (i < N_F)
		{
			// particle i data
			const float x = xPosF[i];
			const float y = yPosF[i];
			const float z = zPosF[i];
			const uint32_t cF_src = compactCellIdxByParticleF[i];
			const uint64_t querySlotBase =
				static_cast<uint64_t>(cF_src) * static_cast<uint64_t>(N_queryCellSlots);

			// check all particles j in the cached neighboring grid cells
			for (uint32_t s = 0; s < N_queryCellSlots; s++)
			{
				const uint32_t cellAny = queryCellRefAnyByCellF[querySlotBase + s];
				if (cellAny == Sim::EMPTY_32) { continue; }

				// particle ranges of the neighbor cell, an absent species has an empty range
				const CellRangeByAny cellRange = cellRangeByAny[cellAny];

				// fluid neighbors
				for (uint32_t j = cellRange.beginF; j < cellRange.endF; j++)
				{
					const float dx = x - xPosF[j];
					const float dy = y - yPosF[j];
					const float dz = z - zPosF[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
					{
						cntFF++;
						minRefFF = (j < minRefFF) ? j : minRefFF;
						maxRefFF = (j > maxRefFF) ? j : maxRefFF;
					}
				}

				// boundary neighbors
				for (uint32_t j = cellRange.beginB; j < cellRange.endB; j++)
				{
					const float dx = x - xPosB[j];
					const float dy = y - yPosB[j];
					const float dz = z - zPosB[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
					{
						cntFB++;
						minRefFB = (j < minRefFB) ? j : minRefFB;
						maxRefFB = (j > maxRefFB) ? j : maxRefFB;
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

				for (int oz = cz - cellQueryReach; oz <= cz + cellQueryReach; oz++)
				{
					for (int oy = cy - cellQueryReach; oy <= cy + cellQueryReach; oy++)
					{
						for (int ox = cx - cellQueryReach; ox <= cx + cellQueryReach; ox++)
						{
							const uint64_t cellId = MapToCellId(ox, oy, oz);

							// stencil cells outside the frame hold no particles
							if (cellId == Sim::EMPTY_64) { continue; }

							// rigid neighbor cell
							const int cellR = BinarySearchUniqueCellId(
								uniqueCellIdR, N_nonEmptyCellsR, cellId);
							if (cellR < 0) { continue; }

							// range of all rigid particles in the neighbor cell
							const uint32_t begin = cellStartIdxR[cellR];
							const uint32_t end = cellStartIdxR[cellR + 1];
							for (uint32_t k = begin; k < end; k++)
							{
								const uint32_t j = sortedParticleIdxR[k];
								const float dx = x - xPosR[j];
								const float dy = y - yPosR[j];
								const float dz = z - zPosR[j];
								const float distSqu = dx * dx + dy * dy + dz * dz;

								if (distSqu <= c_ns_kernelRadiusSqu) // neighbor found
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
			nbrCountFF[i] = static_cast<uint16_t>(cntFF);
			nbrCountFB[i] = static_cast<uint16_t>(cntFB);
			nbrCountFR[i] = static_cast<uint16_t>(cntFR);
		}

		CommitChunkCompressionMeta(
			chunk, isChunkOwner, cntFF, minRefFF, maxRefFF,
			nbrChunkMaxCountFF, nbrChunkBaseIdxFF, nbrChunkFormatFF);
		CommitChunkCompressionMeta(
			chunk, isChunkOwner, cntFB, minRefFB, maxRefFB,
			nbrChunkMaxCountFB, nbrChunkBaseIdxFB, nbrChunkFormatFB);
		CommitChunkCompressionMeta(
			chunk, isChunkOwner, cntFR, minRefFR, maxRefFR,
			nbrChunkMaxCountFR, nbrChunkBaseIdxFR, nbrChunkFormatFR);
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
		const CellRangeByAny* __restrict__ cellRangeByAny,
		const uint32_t* __restrict__ compactCellIdxByParticleF,
		const uint32_t* __restrict__ queryCellRefAnyByCellF,
		const uint32_t N_queryCellSlots,
		const uint64_t* __restrict__ uniqueCellIdR,
		const uint32_t* __restrict__ cellStartIdxR,
		const uint32_t* __restrict__ sortedParticleIdxR,
		const uint32_t N_nonEmptyCellsR,
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
		const float x = xPosF[i];
		const float y = yPosF[i];
		const float z = zPosF[i];
		const uint32_t cF_src = compactCellIdxByParticleF[i];
		const uint64_t querySlotBase =
			static_cast<uint64_t>(cF_src) * static_cast<uint64_t>(N_queryCellSlots);

		// write ranges of packed SELL32 neighbor lists
		const uint32_t chunk = i / N_SELL_WIDTH;
		const uint32_t lane = i % N_SELL_WIDTH;
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

		// store all particles j in the cached neighboring grid cells
		for (uint32_t s = 0; s < N_queryCellSlots; s++)
		{
			const uint32_t cellAny = queryCellRefAnyByCellF[querySlotBase + s];
			if (cellAny == Sim::EMPTY_32) { continue; }

			// particle ranges of the neighbor cell, an absent species has an empty range
			const CellRangeByAny cellRange = cellRangeByAny[cellAny];

			// fluid neighbors
			for (uint32_t j = cellRange.beginF; j < cellRange.endF; j++)
			{
				const float dx = x - xPosF[j];
				const float dy = y - yPosF[j];
				const float dz = z - zPosF[j];
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

			// boundary neighbors
			for (uint32_t j = cellRange.beginB; j < cellRange.endB; j++)
			{
				const float dx = x - xPosB[j];
				const float dy = y - yPosB[j];
				const float dz = z - zPosB[j];
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
		}

		// store all rigid neighbors in the neighboring grid cells
		if (c_ns_N_R > 0u)
		{
			const int cx = FloorToInt(x * c_ns_cellSizeInv);
			const int cy = FloorToInt(y * c_ns_cellSizeInv);
			const int cz = FloorToInt(z * c_ns_cellSizeInv);
			const int cellQueryReach = static_cast<int>(c_ns_cellQueryReach);

			for (int oz = cz - cellQueryReach; oz <= cz + cellQueryReach; oz++)
			{
				for (int oy = cy - cellQueryReach; oy <= cy + cellQueryReach; oy++)
				{
					for (int ox = cx - cellQueryReach; ox <= cx + cellQueryReach; ox++)
					{
						const uint64_t cellId = MapToCellId(ox, oy, oz);

						// stencil cells outside the frame hold no particles
						if (cellId == Sim::EMPTY_64) { continue; }

						// rigid neighbor cell
						const int cellR = BinarySearchUniqueCellId(
							uniqueCellIdR, N_nonEmptyCellsR, cellId);
						if (cellR < 0) { continue; }

						// range of all rigid particles in the neighbor cell
						const uint32_t begin = cellStartIdxR[cellR];
						const uint32_t end = cellStartIdxR[cellR + 1];
						for (uint32_t k = begin; k < end; k++)
						{
							const uint32_t j = sortedParticleIdxR[k];
							const float dx = x - xPosR[j];
							const float dy = y - yPosR[j];
							const float dz = z - zPosR[j];
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
	void K_CellLinkedList_Compact_Query_Count_RF_RB_RR(
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
		const uint32_t N_nonEmptyCellsF,
		const uint32_t N_nonEmptyCellsB,
		const uint32_t N_nonEmptyCellsR,
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

		for (int oz = cz - cellQueryReach; oz <= cz + cellQueryReach; oz++)
		{
			for (int oy = cy - cellQueryReach; oy <= cy + cellQueryReach; oy++)
			{
				for (int ox = cx - cellQueryReach; ox <= cx + cellQueryReach; ox++)
				{
					const uint64_t cellId = MapToCellId(ox, oy, oz);

					// stencil cells outside the frame hold no particles
					if (cellId == Sim::EMPTY_64) { continue; }

					// count fluid neighbors in this cell
					const int cellF = BinarySearchUniqueCellId(
						uniqueCellIdF, N_nonEmptyCellsF, cellId);
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
						uniqueCellIdB, N_nonEmptyCellsB, cellId);
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
						uniqueCellIdR, N_nonEmptyCellsR, cellId);
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
	void K_CellLinkedList_Compact_Query_Fill_RF_RB_RR(
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
		const uint32_t N_nonEmptyCellsF,
		const uint32_t N_nonEmptyCellsB,
		const uint32_t N_nonEmptyCellsR,
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

		for (int oz = cz - cellQueryReach; oz <= cz + cellQueryReach; oz++)
		{
			for (int oy = cy - cellQueryReach; oy <= cy + cellQueryReach; oy++)
			{
				for (int ox = cx - cellQueryReach; ox <= cx + cellQueryReach; ox++)
				{
					const uint64_t cellId = MapToCellId(ox, oy, oz);

					// stencil cells outside the frame hold no particles
					if (cellId == Sim::EMPTY_64) { continue; }

					// store fluid neighbors in this cell
					const int cellF = BinarySearchUniqueCellId(
						uniqueCellIdF, N_nonEmptyCellsF, cellId);
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
						uniqueCellIdB, N_nonEmptyCellsB, cellId);
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
						uniqueCellIdR, N_nonEmptyCellsR, cellId);
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

} // namespace GPU

// -------------------------------------------------------------------------------------------------
// --- [ kernel launchers ] ------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

namespace GPU
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
		uint16_t* nbrChunkMaxCountFF,
		uint16_t* nbrChunkMaxCountFB,
		uint16_t* nbrChunkMaxCountFR,
		uint32_t* nbrChunkBaseIdxFF,
		uint32_t* nbrChunkBaseIdxFB,
		uint32_t* nbrChunkBaseIdxFR,
		uint8_t* nbrChunkFormatFF,
		uint8_t* nbrChunkFormatFB,
		uint8_t* nbrChunkFormatFR)
	{
		(void)N_B;
		(void)N_R;

		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_F + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_BruteForce_Query_Count<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB,
			xPosR, yPosR, zPosR,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkMaxCountFF, nbrChunkMaxCountFB, nbrChunkMaxCountFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR);

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

		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_F + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_BruteForce_Query_Fill<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
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
		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_F + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_FillCellIdAndIdxF<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
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
		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_B + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_FillCellIdAndIdxB<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
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
		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_BuildRangesFromSortedKeys<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			cellId, N, cellStartIdx, cellEndIdx);

		GPU::CheckLastKernel("K_BuildRangesFromSortedKeys()");
	}

	void Launch_ClearRanges(
		const uint32_t N_cells,
		cudaStream_t stream,
		// destinations
		uint32_t* cellStartIdx, uint32_t* cellEndIdx)
	{
		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_cells + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_ClearRanges<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			N_cells, cellStartIdx, cellEndIdx);

		GPU::CheckLastKernel("K_ClearRanges()");
	}

	void Launch_MarkCellStarts(
		const uint64_t* cellIdSorted,
		const uint32_t N,
		cudaStream_t stream,
		// destinations
		uint32_t* cellStartMarker)
	{
		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_MarkCellStarts<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			cellIdSorted, N, cellStartMarker);

		GPU::CheckLastKernel("K_MarkCellStarts()");
	}

	void Launch_FillUniqueCellIdAndCellStartIdx(
		const uint64_t* cellIdSorted,
		const uint32_t* cellStartScan,
		const uint32_t N,
		const uint32_t N_nonEmptyCells,
		cudaStream_t stream,
		// destinations
		uint64_t* uniqueCellId,
		uint32_t* cellStartIdx)
	{
		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_FillUniqueCellIdAndCellStartIdx<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			cellIdSorted, cellStartScan, N, N_nonEmptyCells, uniqueCellId, cellStartIdx);

		GPU::CheckLastKernel("K_FillUniqueCellIdAndCellStartIdx()");
	}

	void Launch_FillMergedCompactCellRanges(
		const uint64_t* uniqueCellIdAny,
		const uint32_t N_nonEmptyCellsAny,
		const uint64_t* uniqueCellIdF,
		const uint64_t* uniqueCellIdB,
		const uint32_t* cellStartIdxF,
		const uint32_t* cellStartIdxB,
		const uint32_t N_nonEmptyCellsF,
		const uint32_t N_nonEmptyCellsB,
		cudaStream_t stream,
		// destinations
		CellRangeByAny* cellRangeByAny)
	{
		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = N_nonEmptyCellsAny == 0
			? 0
			: 1u + (N_nonEmptyCellsAny - 1u) / N_BLOCK_SIZE;
		if (N_GRID_SIZE == 0) { return; }

		K_FillMergedCompactCellRanges<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			uniqueCellIdAny, N_nonEmptyCellsAny,
			uniqueCellIdF, uniqueCellIdB,
			cellStartIdxF, cellStartIdxB,
			N_nonEmptyCellsF, N_nonEmptyCellsB,
			cellRangeByAny);

		GPU::CheckLastKernel("K_FillMergedCompactCellRanges()");
	}

	void Launch_FillQueryLookupBoundsF(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const uint32_t* cellStartIdxF,
		const uint64_t* uniqueCellIdAny,
		const uint32_t N_nonEmptyCellsF,
		const uint32_t N_nonEmptyCellsAny,
		const uint32_t cellQueryReach,
		cudaStream_t stream,
		// destinations
		uint32_t* compactCellIdxByParticleF,
		uint32_t* queryLoAnyByCellF,
		uint32_t* queryHiAnyByCellF)
	{
		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_nonEmptyCellsF + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_FillQueryLookupBoundsF<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			xPosF, yPosF, zPosF,
			cellStartIdxF, uniqueCellIdAny,
			N_nonEmptyCellsF, N_nonEmptyCellsAny, cellQueryReach,
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
		const uint32_t N_nonEmptyCellsF,
		const uint32_t N_queryCellSlots,
		cudaStream_t stream,
		// destinations
		uint32_t* queryCellRefAnyByCellF)
	{
		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_nonEmptyCellsF + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_FillQueryCellRefsByCellF<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			xPosF, yPosF, zPosF,
			cellStartIdxF, uniqueCellIdAny,
			queryLoAnyByCellF, queryHiAnyByCellF,
			N_nonEmptyCellsF, N_queryCellSlots,
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
		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_FillCellIdAndInitParticleIdx<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			xPos, yPos, zPos, N, cellId, sortedParticleIdx);

		GPU::CheckLastKernel("K_FillCellIdAndInitParticleIdx()");
	}

	void Launch_CellLinkedList_Compact_Query_Count(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const CellRangeByAny* cellRangeByAny,
		const uint32_t* compactCellIdxByParticleF,
		const uint32_t* queryCellRefAnyByCellF,
		const uint64_t* uniqueCellIdR,
		const uint32_t* cellStartIdxR,
		const uint32_t* sortedParticleIdxR,
		const uint32_t N_nonEmptyCellsR,
		const uint32_t N_queryCellSlots,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		uint16_t* nbrCountFF,
		uint16_t* nbrCountFB,
		uint16_t* nbrCountFR,
		uint16_t* nbrChunkMaxCountFF,
		uint16_t* nbrChunkMaxCountFB,
		uint16_t* nbrChunkMaxCountFR,
		uint32_t* nbrChunkBaseIdxFF,
		uint32_t* nbrChunkBaseIdxFB,
		uint32_t* nbrChunkBaseIdxFR,
		uint8_t* nbrChunkFormatFF,
		uint8_t* nbrChunkFormatFB,
		uint8_t* nbrChunkFormatFR)
	{
		(void)N_B;
		(void)N_R;

		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_F + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_CellLinkedList_Compact_Query_Count<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR,
			cellRangeByAny,
			compactCellIdxByParticleF,
			queryCellRefAnyByCellF, N_queryCellSlots,
			uniqueCellIdR, cellStartIdxR, sortedParticleIdxR,
			N_nonEmptyCellsR,
			N_F,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkMaxCountFF, nbrChunkMaxCountFB, nbrChunkMaxCountFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR);

		GPU::CheckLastKernel("K_CellLinkedList_Compact_Query_Count()");
	}

	void Launch_CellLinkedList_Compact_Query_Fill(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const CellRangeByAny* cellRangeByAny,
		const uint32_t* compactCellIdxByParticleF,
		const uint32_t* queryCellRefAnyByCellF,
		const uint64_t* uniqueCellIdR,
		const uint32_t* cellStartIdxR,
		const uint32_t* sortedParticleIdxR,
		const uint32_t N_nonEmptyCellsR,
		const uint32_t N_queryCellSlots,
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

		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_F + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_CellLinkedList_Compact_Query_Fill<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR,
			cellRangeByAny,
			compactCellIdxByParticleF,
			queryCellRefAnyByCellF, N_queryCellSlots,
			uniqueCellIdR, cellStartIdxR, sortedParticleIdxR,
			N_nonEmptyCellsR,
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

		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_R + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		if (N_R > 0)
		{
			K_BruteForce_Query_Count_RF_RB_RR<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
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

		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_R + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		if (N_R > 0)
		{
			K_BruteForce_Query_Fill_RF_RB_RR<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
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
		const uint32_t N_nonEmptyCellsF,
		const uint32_t N_nonEmptyCellsB,
		const uint32_t N_nonEmptyCellsR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		uint16_t* nbrCountRF,
		uint16_t* nbrCountRB,
		uint16_t* nbrCountRR)
	{
		(void)N_F;
		(void)N_B;

		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_R + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		if (N_R > 0)
		{
			K_CellLinkedList_Compact_Query_Count_RF_RB_RR<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
				xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR, bodyIdR,
				uniqueCellIdF, cellStartIdxF,
				uniqueCellIdB, cellStartIdxB,
				uniqueCellIdR, cellStartIdxR, sortedParticleIdxR,
				N_nonEmptyCellsF, N_nonEmptyCellsB, N_nonEmptyCellsR,
				nbrCountRF, nbrCountRB, nbrCountRR);
			GPU::CheckLastKernel("K_CellLinkedList_Compact_Query_Count_RF_RB_RR()");
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
		const uint32_t N_nonEmptyCellsF,
		const uint32_t N_nonEmptyCellsB,
		const uint32_t N_nonEmptyCellsR,
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

		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_R + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		if (N_R > 0)
		{
			K_CellLinkedList_Compact_Query_Fill_RF_RB_RR<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
				xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR,
				bodyIdR,
				uniqueCellIdF, cellStartIdxF,
				uniqueCellIdB, cellStartIdxB,
				uniqueCellIdR, cellStartIdxR, sortedParticleIdxR,
				N_nonEmptyCellsF, N_nonEmptyCellsB, N_nonEmptyCellsR,
				nbrOffsetsRF, nbrOffsetsRB, nbrOffsetsRR,
				nbrListRF, nbrListRB, nbrListRR);
			GPU::CheckLastKernel("K_CellLinkedList_Compact_Query_Fill_RF_RB_RR()");
		}
	}

} // namespace GPU
