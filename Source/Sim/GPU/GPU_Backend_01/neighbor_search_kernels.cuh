// neighbor_search_kernels.cuh
//
// TODO: comment

#pragma once

// external
#include <cuda_runtime.h>
#include <cstdint>

// internal



namespace GPU01
{
	struct LBVH_AABB
	{
		float minX = 0.0f;
		float minY = 0.0f;
		float minZ = 0.0f;
		float maxX = 0.0f;
		float maxY = 0.0f;
		float maxZ = 0.0f;
	};

	struct LBVH_InternalNode
	{
		LBVH_AABB box{};
		uint32_t leftChild = 0;
		uint32_t rightChild = 0;
		uint32_t leftIsLeaf = 0;
		uint32_t rightIsLeaf = 0;
	};

	void NS_ReloadConstants(
		uint32_t N_F, uint32_t N_B,
		float kernelRadius, float cellSize, uint32_t cellQueryReach, uint32_t gridCellIndexing,
		float xBoxOrig, float yBoxOrig, float zBoxOrig,
		uint32_t N_DimX, uint32_t N_DimY, uint32_t N_DimZ,
		uint32_t N_Cells, uint32_t EMPTY,
		cudaStream_t stream);

	void Launch_BruteForce_Query_Count(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		uint32_t* nbrCountFF,
		uint32_t* nbrCountFB);

	void Launch_BruteForce_Query_Fill(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		const uint64_t* nbrOffsetsFF,
		const uint64_t* nbrOffsetsFB,
		// destinations
		uint32_t* nbrListFF,
		uint32_t* nbrListFB,
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

	void Launch_CellLinkedList_Basic_Query_Count(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* cellStartIdxF, const uint32_t* cellEndIdxF,
		const uint32_t* cellStartIdxB, const uint32_t* cellEndIdxB,
		const uint32_t* sortedParticleIdxF, const uint32_t* sortedParticleIdxB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		uint32_t* nbrCountFF, uint32_t* nbrCountFB);

	void Launch_CellLinkedList_Basic_Query_Fill(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* cellStartIdxF, const uint32_t* cellEndIdxF,
		const uint32_t* cellStartIdxB, const uint32_t* cellEndIdxB,
		const uint32_t* sortedParticleIdxF, const uint32_t* sortedParticleIdxB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		const uint64_t* nbrOffsetsFF,
		const uint64_t* nbrOffsetsFB,
		// destinations
		uint32_t* nbrListFF, uint32_t* nbrListFB,
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
		uint32_t* nbrCountFF, uint32_t* nbrCountFB);

	void Launch_CellLinkedList_Compact_Query_Fill(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint64_t* uniqueCellIdF, const uint64_t* uniqueCellIdB,
		const uint32_t* cellStartIdxF, const uint32_t* cellStartIdxB,
		const uint32_t* sortedParticleIdxF, const uint32_t* sortedParticleIdxB,
		const uint32_t N_NonEmptyCellsF, const uint32_t N_NonEmptyCellsB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		const uint64_t* nbrOffsetsFF,
		const uint64_t* nbrOffsetsFB,
		// destinations
		uint32_t* nbrListFF, uint32_t* nbrListFB,
		uint32_t* overflowFlag);

	void Launch_LBVH_FillMortonAndAABB(
		const float* xPos, const float* yPos, const float* zPos,
		const uint32_t N,
		const float sceneMinX, const float sceneMinY, const float sceneMinZ,
		const float invExtX, const float invExtY, const float invExtZ,
		const float leafAabbRadius,
		cudaStream_t stream,
		// destinations
		uint32_t* morton,
		uint32_t* sortedParticleIdx,
		LBVH_AABB* leafBox);

	void Launch_LBVH_BuildInternalNodes(
		const uint32_t* sortedMorton,
		const uint32_t N,
		cudaStream_t stream,
		// destinations
		LBVH_InternalNode* nodes);

	void Launch_LBVH_BuildParentLookupAndResetVisits(
		const LBVH_InternalNode* nodes,
		const uint32_t N_internal,
		const uint32_t N,
		cudaStream_t stream,
		// destinations
		uint32_t* parentOf,
		uint32_t* visitCount);

	void Launch_LBVH_PropagateAABBs(
		const LBVH_AABB* sortedLeafBox,
		const uint32_t* parentOf,
		const uint32_t N,
		const uint32_t N_internal,
		cudaStream_t stream,
		// destinations
		LBVH_InternalNode* nodes,
		uint32_t* visitCount);

	void Launch_LBVH_Query_Count(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* sortedParticleIdxF, const uint32_t* sortedParticleIdxB,
		const LBVH_InternalNode* nodesF, const LBVH_InternalNode* nodesB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		uint32_t* nbrCountFF, uint32_t* nbrCountFB);

	void Launch_LBVH_Query_Fill(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* sortedParticleIdxF, const uint32_t* sortedParticleIdxB,
		const LBVH_InternalNode* nodesF, const LBVH_InternalNode* nodesB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		const uint64_t* nbrOffsetsFF,
		const uint64_t* nbrOffsetsFB,
		// destinations
		uint32_t* nbrListFF, uint32_t* nbrListFB,
		uint32_t* overflowFlag);

} // namespace GPU01
