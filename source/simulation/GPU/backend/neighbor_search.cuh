// neighbor_search.cuh
#pragma once

// external
#include <cstdint>

// internal
#include "common.cuh"
#include "gpu_backend.h"
#include "source/simulation/core/backend_sim.h"
#include "source/simulation/core/cell_frame.h"



namespace Sim
{
	// ---------------------------------------------------------------------------------------------
	// --- [ cell linked list compact implementation ] ---------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NeighborSearch_CellLinkedList_Compact_GPU
	{
	public:

		explicit NeighborSearch_CellLinkedList_Compact_GPU(SimBackend_GPU& backend)
			: be_{ backend } {}

		void Init();

		void Construct(const SimParams& params);

		void Query_Count(const SimParams& params);

		void Query_Fill(const SimParams& params);

		void DataReOrder(const SimParams& params);

		// re-orders particle fields ahead of a full solver step, which re-initializes some of them,
		// so permutations of those fields are skipped (TODO: update if solver step behavior changes)
		// (currently omitted are accPress, sourceTerm, diagElem, density, and pressure without warm start)
		void DataReOrder_SolverStep(const SimParams& params);

		Mem::ParticleClassMemUsage GetMemUsage() const;

	private:

		SimBackend_GPU& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;
		uint32_t cellQueryReach_ = 1;
		CellCoordFrame cellFrame_ = {};
		uint32_t N_nonEmptyCellsF_ = 0;
		uint32_t N_nonEmptyCellsB_ = 0;
		uint32_t N_nonEmptyCellsR_ = 0;
		uint32_t N_nonEmptyCellsAny_ = 0;
		uint32_t N_queryCellSlots_ = 0;
		bool particlesMatchCurrentCellOrder_ = false;
		bool cellIdsAvailableForReuse_ = false;

		// data structures
		DeviceBuffer<uint64_t> d_uniqueCellIdF_; // unique cell IDs of non-empty cells
		DeviceBuffer<uint32_t> d_cellStartIdxF_; // per-cell start index in the re-ordered particle arrays
		DeviceBuffer<uint32_t> d_sortedParticleIdxF_; // fluid particle permutation / per-particle compact cell index

		DeviceBuffer<uint64_t> d_uniqueCellIdB_;
		DeviceBuffer<uint32_t> d_cellStartIdxB_;

		DeviceBuffer<uint64_t> d_uniqueCellIdR_;
		DeviceBuffer<uint32_t> d_cellStartIdxR_;
		DeviceBuffer<uint32_t> d_sortedParticleIdxR_;

		DeviceBuffer<uint64_t> d_uniqueCellIdAny_; // unique cell IDs of all non-empty fluid/boundary cells
		DeviceBuffer<GPU::CellRangeByAny> d_cellRangeByAny_; // merged-cell to fluid/boundary particle ranges

		// helpers
		void EnsureStorageAllocation();
		void UpdateCellFrame();
		void ReOrderParticleData(const SimParams& params, bool permuteSolverFields);
		void BuildCompactCellList(
			DeviceBuffer<uint64_t>& d_uniqueCellId,
			DeviceBuffer<uint32_t>& d_cellStartIdx,
			uint32_t* d_markerScanScratch,
			const uint64_t* d_cellIdSorted,
			uint32_t N, uint32_t& N_nonEmptyCells);

		void BuildMergedCompactCellRanges();
		void BuildQueryLookupBounds();
		void BuildQueryCellRefCache();
	};

} // namespace Sim
