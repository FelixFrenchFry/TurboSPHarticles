// neighbor_search_kernels.cuh

#pragma once

// external
#include <cuda_runtime.h>
#include <cstdint>

// internal
#include "source/simulation/core/cell_frame.h"



namespace GPU
{
	void NS_ReloadConstants(
		uint32_t N_F, uint32_t N_B, uint32_t N_R,
		float kernelRadius, float cellSize, uint32_t cellQueryReach,
		float xBoxOrig, float yBoxOrig, float zBoxOrig,
		const Sim::CellCoordFrame& cellFrame,
		uint32_t N_cells, uint32_t EMPTY,
		cudaStream_t stream);

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
		uint32_t* nbrChunkMaxRefFR);

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
		uint32_t* overflowFlag);

	void Launch_FillCellIdAndIdxF(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const uint32_t N_F,
		cudaStream_t stream,
		// destinations
		uint32_t* cellIdF);

	void Launch_FillCellIdAndIdxB(
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		uint32_t* cellIdB);

	void Launch_BuildRangesFromSortedKeys(
		const uint32_t* cellId,
		const uint32_t N,
		cudaStream_t stream,
		// destinations
		uint32_t* cellStartIdx, uint32_t* cellEndIdx);

	void Launch_ClearRanges(
		const uint32_t N_cells,
		cudaStream_t stream,
		// destinations
		uint32_t* cellStartIdx, uint32_t* cellEndIdx);

	void Launch_MarkCellStarts(
		const uint64_t* cellIdSorted,
		const uint32_t N,
		cudaStream_t stream,
		// destinations
		uint32_t* cellStartMarker);

	void Launch_FillUniqueCellIdAndCellStartIdx(
		const uint64_t* cellIdSorted,
		const uint32_t* cellStartScan,
		const uint32_t N,
		const uint32_t N_nonEmptyCells,
		cudaStream_t stream,
		// destinations
		uint64_t* uniqueCellId,
		uint32_t* cellStartIdx);

	void Launch_FillMergedCompactCellRefs(
		const uint64_t* uniqueCellIdAny,
		const uint32_t N_nonEmptyCellsAny,
		const uint64_t* uniqueCellIdF,
		const uint64_t* uniqueCellIdB,
		const uint32_t N_nonEmptyCellsF,
		const uint32_t N_nonEmptyCellsB,
		cudaStream_t stream,
		// destinations
		uint32_t* cellRefFByAny,
		uint32_t* cellRefBByAny);

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
		uint32_t* queryHiAnyByCellF);

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
		uint32_t* queryCellRefAnyByCellF);

	void Launch_FillCellIdAndInitParticleIdx(
		const float* xPos, const float* yPos, const float* zPos,
		const uint32_t N,
		cudaStream_t stream,
		// destinations
		uint64_t* cellId, uint32_t* sortedParticleIdx);

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
		const uint32_t N_nonEmptyCellsR,
		const uint32_t N_queryCellSlots,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		uint16_t* nbrCountFF, uint16_t* nbrCountFB,
		uint16_t* nbrCountFR,
		uint32_t* nbrChunkMinRefFF,
		uint32_t* nbrChunkMaxRefFF,
		uint32_t* nbrChunkMinRefFB,
		uint32_t* nbrChunkMaxRefFB,
		uint32_t* nbrChunkMinRefFR,
		uint32_t* nbrChunkMaxRefFR);

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
		uint32_t* overflowFlag);

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
		uint16_t* nbrCountRR);

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
		uint32_t* nbrListRR);

	void Launch_CellLinkedList_Compact_Query_Count_RF_RB_RR(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const uint32_t* bodyIdR,
		const uint64_t* uniqueCellIdF, const uint32_t* cellStartIdxF,
		const uint64_t* uniqueCellIdB, const uint32_t* cellStartIdxB,
		const uint64_t* uniqueCellIdR, const uint32_t* cellStartIdxR,
		const uint32_t* sortedParticleIdxR,
		const uint32_t N_nonEmptyCellsF, const uint32_t N_nonEmptyCellsB,
		const uint32_t N_nonEmptyCellsR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		uint16_t* nbrCountRF,
		uint16_t* nbrCountRB,
		uint16_t* nbrCountRR);

	void Launch_CellLinkedList_Compact_Query_Fill_RF_RB_RR(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const uint32_t* bodyIdR,
		const uint64_t* uniqueCellIdF, const uint32_t* cellStartIdxF,
		const uint64_t* uniqueCellIdB, const uint32_t* cellStartIdxB,
		const uint64_t* uniqueCellIdR, const uint32_t* cellStartIdxR,
		const uint32_t* sortedParticleIdxR,
		const uint32_t N_nonEmptyCellsF, const uint32_t N_nonEmptyCellsB,
		const uint32_t N_nonEmptyCellsR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		const uint64_t* nbrOffsetsRF,
		const uint64_t* nbrOffsetsRB,
		const uint64_t* nbrOffsetsRR,
		// destinations
		uint32_t* nbrListRF,
		uint32_t* nbrListRB,
		uint32_t* nbrListRR);

} // namespace GPU
