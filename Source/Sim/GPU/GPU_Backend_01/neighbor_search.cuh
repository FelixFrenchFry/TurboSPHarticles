// neighbor_search.cuh
#pragma once

// external
#include <limits>

// internal
#include "common.cuh"
#include "gpu_backend_01.h"
#include "neighbor_search_kernels.cuh"
#include "Source/Sim/Core/backend_sim.h"



namespace Sim
{
	// ---------------------------------------------------------------------------------------------
	// --- [ brute force implementation ] ----------------------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NS_BruteForce_GPU01
		: public I_NeighborSearch
	{
	public: // -------------------------------------------------------------------------------------

		explicit NS_BruteForce_GPU01(SimBackend_GPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "BruteForce_GPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query_Count(const SimParams& params) override;

		void Query_Fill(const SimParams& params) override;

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage>
			GetMemUsage(const SimParams& params) const override;

	private: // ------------------------------------------------------------------------------------

		SimBackend_GPU01& be_;
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ cell linked list compact implementation ] ---------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NS_CellLinkedList_Compact_GPU01
		: public I_NeighborSearch
	{
	public: // -------------------------------------------------------------------------------------

		explicit NS_CellLinkedList_Compact_GPU01(SimBackend_GPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "CellLinkedList_Compact_GPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query_Count(const SimParams& params) override;

		void Query_Fill(const SimParams& params) override;

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage>
			GetMemUsage(const SimParams& params) const override;

	private: // ------------------------------------------------------------------------------------

		SimBackend_GPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;
		uint32_t cellQueryReach_ = 1;
		uint32_t N_NonEmptyCellsF_ = 0;
		uint32_t N_NonEmptyCellsB_ = 0;
		uint32_t N_NonEmptyCellsAny_ = 0;
		bool cellIdsMatchCurrentParticleOrder_ = false;

		// data structures
		DeviceBuffer<uint64_t> d_uniqueCellIdF_; // unique cell IDs of non-empty cells
		DeviceBuffer<uint32_t> d_cellStartIdxF_; // per-cell start index in the re-ordered particle arrays
		DeviceBuffer<uint64_t> d_cellIdF_; // per-particle cell ID array
		DeviceBuffer<uint32_t> d_sortedParticleIdxF_; // sorted fluid particle index array / per-particle compact cell index

		DeviceBuffer<uint64_t> d_uniqueCellIdB_;
		DeviceBuffer<uint32_t> d_cellStartIdxB_;
		DeviceBuffer<uint64_t> d_cellIdB_;
		DeviceBuffer<uint32_t> d_sortedParticleIdxB_;

		DeviceBuffer<uint64_t> d_uniqueCellIdAny_; // unique cell IDs of all non-empty fluid/boundary cells
		DeviceBuffer<uint32_t> d_cellRefFByAny_; // merged-cell to compact fluid-cell index
		DeviceBuffer<uint32_t> d_cellRefBByAny_; // merged-cell to compact boundary-cell index
		DeviceBuffer<uint32_t> d_queryLoAnyByCellF_; // lower search bound in merged-cell table per source fluid cell
		DeviceBuffer<uint32_t> d_queryHiAnyByCellF_; // upper search bound in merged-cell table per source fluid cell

		// re-usable work (scratch) buffer for temp storage
		DeviceBuffer<uint32_t> d_cellStartMarkerScan_;

		static constexpr uint32_t EMPTY_32 = std::numeric_limits<uint32_t>::max();
		static constexpr uint64_t EMPTY_64 = std::numeric_limits<uint64_t>::max();

		// helpers
		void EnsureStorageAllocation();
		void BuildCompactCellList(
			DeviceBuffer<uint64_t>& d_uniqueCellId,
			DeviceBuffer<uint32_t>& d_cellStartIdx,
			const uint64_t* d_cellIdSorted,
			uint32_t N, uint32_t& N_NonEmptyCells);

		void BuildMergedCompactCellLookup();
		void BuildQueryLookupBounds();
	};

} // namespace Sim
