// neighbor_search_kernels.cuh
//
// TODO: comment

#pragma once

// external

// internal



namespace GPU01
{
	void NS_ReloadConstants(
		uint32_t N_F, uint32_t N_B,
		float kernelRadius, float cellSize,
		float xBoxOrig, float yBoxOrig, float zBoxOrig,
		uint32_t N_DimX, uint32_t N_DimY, uint32_t N_DimZ,
		uint32_t N_Cells, uint32_t EMPTY,
		cudaStream_t stream);

	void Launch_BruteForce_Query(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t N_F, const uint32_t N_B,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		cudaStream_t stream,
		// destinations
		uint8_t* neighCountFF,
		uint8_t* neighCountFB,
		uint32_t* neighListFF,
		uint32_t* neighListFB,
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
		const uint32_t N_Cells,
		cudaStream_t stream,
		// destinations
		uint32_t* cellStartIdx, uint32_t* cellEndIdx);

	void Launch_CellLinkedList_Basic_Query(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* cellStartIdxF, const uint32_t* cellEndIdxF,
		const uint32_t* cellStartIdxB, const uint32_t* cellEndIdxB,
		const uint32_t* sortedParticleIdxF, const uint32_t* sortedParticleIdxB,
		const uint32_t N_F, const uint32_t N_B,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		cudaStream_t stream,
		// destinations
		uint8_t* neighCountFF, uint8_t* neighCountFB,
		uint32_t* neighListFF, uint32_t* neighListFB,
		uint32_t* overflowFlag);

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
		const uint32_t N_NonEmptyCells,
		cudaStream_t stream,
		// destinations
		uint64_t* uniqueCellId,
		uint32_t* cellStartIdx);

	void Launch_FillCellIdAndInitParticleIdx(
		const float* xPos, const float* yPos, const float* zPos,
		const uint32_t N,
		cudaStream_t stream,
		// destinations
		uint64_t* cellId, uint32_t* sortedParticleIdx);

	void Launch_CellLinkedList_Compact_Query_Basic(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint64_t* uniqueCellIdF, const uint64_t* uniqueCellIdB,
		const uint32_t* cellStartIdxF, const uint32_t* cellStartIdxB,
		const uint32_t* sortedParticleIdxF, const uint32_t* sortedParticleIdxB,
		const uint32_t N_NonEmptyCellsF, const uint32_t N_NonEmptyCellsB,
		const uint32_t N_F, const uint32_t N_B,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		cudaStream_t stream,
		// destinations
		uint8_t* neighCountFF, uint8_t* neighCountFB,
		uint32_t* neighListFF, uint32_t* neighListFB,
		uint32_t* overflowFlag);

} // namespace GPU01
