// neighbor_search.cuh
//
// TODO: comment

#pragma once

// external
#include <limits>

// internal
#include "common.cuh"
#include "gpu_backend_01.h"
#include "Source/Sim/Core/backend_sim.h"



namespace Sim
{
	class NS_BruteForce_GPU01 : public I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		explicit NS_BruteForce_GPU01(SimBackend_GPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "BruteForce_GPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query(const SimParams& params) override;

		void DataReOrder(const SimParams& params) override;

	private: // ----------------------------------------------------------------

		SimBackend_GPU01& be_;
	};

	class NS_CellLinkedList_Basic_GPU01 : public I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		explicit NS_CellLinkedList_Basic_GPU01(SimBackend_GPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "CellLinkedList_Basic_GPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query(const SimParams& params) override;

		void DataReOrder(const SimParams& params) override;

	private: // ----------------------------------------------------------------

		SimBackend_GPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;

		// grid AABB origin
		float xBoxOrig_ = 0.0f;
		float yBoxOrig_ = 0.0f;
		float zBoxOrig_ = 0.0f;

		// grid dimensions
		uint32_t N_DimX_ = 0;
		uint32_t N_DimY_ = 0;
		uint32_t N_DimZ_ = 0;
		uint32_t N_Cells_ = 0;

		// data structures
		DeviceBuffer<uint32_t> d_cellStartIdxF_;
		DeviceBuffer<uint32_t> d_cellEndIdxF_;
		DeviceBuffer<uint32_t> d_cellIdF_;
		DeviceBuffer<uint32_t> d_sortedParticleIdxF_;

		DeviceBuffer<uint32_t> d_cellStartIdxB_;
		DeviceBuffer<uint32_t> d_cellEndIdxB_;
		DeviceBuffer<uint32_t> d_cellIdB_;
		DeviceBuffer<uint32_t> d_sortedParticleIdxB_;

		static constexpr uint32_t EMPTY = std::numeric_limits<uint32_t>::max();

		using MinMax3 = GPU01::MinMax3HD;
		MinMax3 h_aabbF_{};
		MinMax3 h_aabbB_{};

		// resizes all device buffers when particle counts change
		// (does not preserve data)
		void EnsureCapacity(uint32_t N_F, uint32_t N_B, uint32_t N_Cells)
		{
			// fluid
			if (d_cellStartIdxF_.Size() != N_Cells) { d_cellStartIdxF_.Resize(N_Cells); }
			if (d_cellEndIdxF_.Size() != N_Cells) { d_cellEndIdxF_.Resize(N_Cells); }
			if (d_cellIdF_.Size() != N_F) { d_cellIdF_.Resize(N_F); }
			if (d_sortedParticleIdxF_.Size() != N_F) { d_sortedParticleIdxF_.Resize(N_F); }

			// boundary
			if (d_cellStartIdxB_.Size() != N_Cells) { d_cellStartIdxB_.Resize(N_Cells); }
			if (d_cellEndIdxB_.Size() != N_Cells) { d_cellEndIdxB_.Resize(N_Cells); }
			if (d_cellIdB_.Size() != N_B) { d_cellIdB_.Resize(N_B); }
			if (d_sortedParticleIdxB_.Size() != N_B) { d_sortedParticleIdxB_.Resize(N_B); }
		}
	};

	class NS_CellLinkedList_Compact_GPU01 : public I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		explicit NS_CellLinkedList_Compact_GPU01(SimBackend_GPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "CellLinkedList_Compact_GPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query(const SimParams& params) override;

		void DataReOrder(const SimParams& params) override;

	private: // ----------------------------------------------------------------

		SimBackend_GPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;
		uint32_t N_NonEmptyCellsF_ = 0;
		uint32_t N_NonEmptyCellsB_ = 0;

		// data structures
		DeviceBuffer<uint64_t> d_uniqueCellIdF_; // unique cell IDs of non-empty cells
		DeviceBuffer<uint32_t> d_cellStartIdxF_; // per-cell start index in the sorted particle index array
		DeviceBuffer<uint64_t> d_cellIdF_; // per-particle cell ID array
		DeviceBuffer<uint32_t> d_sortedParticleIdxF_; // sorted fluid particle index array (permutation)

		DeviceBuffer<uint64_t> d_uniqueCellIdB_;
		DeviceBuffer<uint32_t> d_cellStartIdxB_;
		DeviceBuffer<uint64_t> d_cellIdB_;
		DeviceBuffer<uint32_t> d_sortedParticleIdxB_;

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
	};

} // namespace Sim
