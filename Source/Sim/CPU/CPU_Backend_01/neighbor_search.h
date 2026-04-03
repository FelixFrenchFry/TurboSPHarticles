// neighbor_search.h

#pragma once

// external
#include <cstdint>
#include <limits>
#include <tbb/enumerable_thread_specific.h>
#include <vector>

// internal
#include "Source/Sim/Core/backend_sim.h"



// TODO: make these used the params based on heuristic expected values?
//static constexpr uint32_t N_perScratchInlineFF = 64;
//static constexpr uint32_t N_perScratchInlineFB = 48;

namespace Sim
{
	// forward declarations
	class SimBackend_CPU01;

	// ---------------------------------------------------------------------------------------------
	// --- [ brute force implementation ] ----------------------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NS_BruteForce_CPU01
		: public I_NeighborSearch
	{
	public: // -------------------------------------------------------------------------------------

		explicit NS_BruteForce_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "BruteForce_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query_Count(const SimParams& params) override;

		void Query_Fill(const SimParams& params) override;

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

	private: // ------------------------------------------------------------------------------------
		
		SimBackend_CPU01& be_;

		// query scratch
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> scratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> scratchFB_;

		float kernelRadiusSqu_ = 0.0f;
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ cell linked list compact implementation ] ---------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NS_CellLinkedList_Compact_CPU01
		: public I_NeighborSearch
	{
	public: // -------------------------------------------------------------------------------------

		explicit NS_CellLinkedList_Compact_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "CellLinkedList_Compact_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query_Count(const SimParams& params) override;

		void Query_Fill(const SimParams& params) override;

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

		// TODO: implement query with BigMin-LitMax and ternary search and caching of results?

	private: // ------------------------------------------------------------------------------------

		SimBackend_CPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;
		float kernelRadiusSqu_ = 0.0f;
		uint32_t cellQueryReach_ = 1; // query range in cells around each particle
		static constexpr uint32_t N_CellFrameRebuildMargin_ = 8; // hysteresis buffer

		// local grid frame in cell coordinates
		int cellOrigX_ = 0;
		int cellOrigY_ = 0;
		int cellOrigZ_ = 0;
		uint32_t N_DimX_ = 0;
		uint32_t N_DimY_ = 0;
		uint32_t N_DimZ_ = 0;
		bool cellFrameValid_ = false;
		bool cellIdsMatchCurrentParticleOrder_ = false;

		// query scratch
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> scratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> scratchFB_;

		// data structures
		std::vector<uint64_t> uniqueCellIdF_; // unique cell IDs of non-empty cells
		std::vector<uint32_t> cellStartIdxF_; // per-cell start index in the re-ordered particle arrays
		std::vector<uint64_t> cellIdF_; // per-particle cell ID array
		std::vector<uint32_t> particleOrderF_; // re-order scratch permutation for fluid particles

		std::vector<uint64_t> uniqueCellIdB_;
		std::vector<uint32_t> cellStartIdxB_;
		std::vector<uint64_t> cellIdB_;
		std::vector<uint32_t> particleOrderB_;

		static constexpr uint32_t EMPTY_32 = std::numeric_limits<uint32_t>::max();
		static constexpr uint64_t EMPTY_64 = std::numeric_limits<uint64_t>::max();

		// helpers
		void UpdateCellFrame();
		inline uint32_t NeighborCellCount() const noexcept;
		inline uint64_t MapToCellId(int cx, int cy, int cz) const noexcept;
		inline uint64_t MapToCellId_Morton(int cx, int cy, int cz) const noexcept;
		static inline uint64_t SpreadBitsWithGaps_21to63(uint32_t v) noexcept;
		inline int NeighborOffset(int rx, int ry, int rz) const noexcept;
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ bounding volume hierarchy linear (LBVH) implementation ] ------------------------------
	// ---------------------------------------------------------------------------------------------

	// TODO: implement

} // namespace Sim
