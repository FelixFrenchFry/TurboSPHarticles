// neighbor_search.h

#pragma once

// external
#include <boost/container/small_vector.hpp>
#include <cstdint>
#include <limits>
#include <span>
#include <tbb/enumerable_thread_specific.h>
#include <vector>

// internal
#include "source/simulation/core/backend_sim.h"



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

		Mem::ParticleClassMemUsage GetMemUsage(const SimParams& params) const override;

	private: // ------------------------------------------------------------------------------------
		
		SimBackend_CPU01& be_;

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

		void DataReOrder_SolverStep(const SimParams& params) override;

		bool TryGetLastParticleReOrder(
			std::span<const uint32_t>& oldFluidLocalIdx_FromNewLocalIdx,
			std::span<const uint32_t>& oldBoundaryLocalIdx_FromNewLocalIdx) const override;

		Mem::ParticleClassMemUsage GetMemUsage(const SimParams& params) const override;

		// TODO: implement query with BigMin-LitMax and ternary search and caching of results?

	private: // ------------------------------------------------------------------------------------

		SimBackend_CPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;
		float kernelRadiusSqu_ = 0.0f;
		uint32_t cellQueryReach_ = 1; // query range in cells around each particle
		static constexpr uint32_t N_CELL_FRAME_REBUILD_MARGIN = 8; // hysteresis buffer
		static constexpr uint32_t N_QUERY_CELL_INLINE = 96;

		// local grid frame in cell coordinates
		int cellOrigX_ = 0;
		int cellOrigY_ = 0;
		int cellOrigZ_ = 0;
		uint32_t N_dimX_ = 0;
		uint32_t N_dimY_ = 0;
		uint32_t N_dimZ_ = 0;
		bool cellFrameValid_ = false;

		// also marks the cell frame as current, and only holds until a particle moves
		bool cellIdsMatchCurrentParticleOrder_ = false;

		struct ActiveCellRange
		{
			uint32_t begin = 0;
			uint32_t end = 0;
		};

		struct QueryMetaScratch
		{
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE> fluidRanges;
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE> boundaryRanges;
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE> rigidRanges;
		};
		mutable tbb::enumerable_thread_specific<QueryMetaScratch> queryMetaScratch_;

		// data structures
		std::vector<uint32_t> cellStartIdxF_; // per-cell start index in the re-ordered particle arrays
		std::vector<uint64_t> cellIdF_; // per-particle cell ID array
		std::vector<uint64_t> cellIdScratchF_; // resident permutation scratch for the cell ID array
		std::vector<uint32_t> particleOrderF_; // re-order scratch permutation for fluid particles
		std::vector<uint32_t> particleOrderScratchF_; // resident sort scratch for the permutation
		std::vector<uint32_t> cellTableF_; // dense local-frame lookup table for compact cell indices

		std::vector<uint32_t> cellStartIdxB_;
		std::vector<uint64_t> cellIdB_;
		std::vector<uint64_t> cellIdScratchB_;
		std::vector<uint32_t> particleOrderB_;
		std::vector<uint32_t> particleOrderScratchB_;
		std::vector<uint32_t> cellTableB_;

		std::vector<uint32_t> cellStartIdxR_;
		std::vector<uint64_t> cellIdR_;
		std::vector<uint64_t> cellIdScratchR_;
		std::vector<uint32_t> particleOrderR_;
		std::vector<uint32_t> particleOrderScratchR_;
		std::vector<uint32_t> cellTableR_;

		// scratch shared by the compact cell table builds, holding per-chunk cell run start
		// counts that are prefixed into per-chunk write offsets
		// (only valid because the fluid, boundary, and rigid builds run one after another)
		std::vector<uint32_t> cellRunStartCounts_ByChunk_;

		// scratch shared by the cell ID sorts, holding per-chunk digit counts that are
		// prefixed into per-chunk write offsets
		// (only valid because the fluid, boundary, and rigid sorts run one after another)
		std::vector<uint32_t> radixDigitCounts_ByChunk_;

		// dense cell table slots hold the compact cell index biased by one, so an empty slot is
		// zero and a whole table is cleared with a plain zero fill (which lowers to memset)
		static constexpr uint32_t CELL_TABLE_EMPTY = 0u;
		static constexpr uint64_t EMPTY_64 = std::numeric_limits<uint64_t>::max();

		// helpers
		void UpdateCellFrame();
		void ReOrderParticleData(const SimParams& params, bool permuteSolverFields);
		void BuildDenseCellTables();

		// appends the non-empty cell ranges of an inclusive cell-coordinate box,
		// visiting every cell once to only require dropping duplicates when boxes overlap
		void GatherBoxQueryRanges(
			int cxBegin, int cxEnd, int cyBegin, int cyEnd, int czBegin, int czEnd,
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& fluidRanges,
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& boundaryRanges,
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& rigidRanges) const;
		void GatherCellQueryRanges(
			int cx, int cy, int cz,
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& fluidRanges,
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& boundaryRanges,
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& rigidRanges) const;
		void GatherChunkQueryRanges(
			uint32_t begin, uint32_t end,
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& fluidRanges,
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& boundaryRanges,
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& rigidRanges) const;

		inline bool IsInsideSimDomain(float x, float y, float z) const noexcept;
		inline bool IsInsideCellFrame(int cx, int cy, int cz) const noexcept;
		inline uint64_t MapToCellId(int cx, int cy, int cz) const noexcept;
		inline uint64_t MapToCellId_Morton(int cx, int cy, int cz) const noexcept;
		inline uint64_t MapToDenseCellIdx(int cx, int cy, int cz) const noexcept;
		static inline uint64_t SpreadBitsWithGaps_21to63(uint32_t v) noexcept;
	};

} // namespace Sim
