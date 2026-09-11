// neighbor_search.h
#pragma once

// external
#include <array>
#include <bit>
#include <boost/container/small_vector.hpp>
#include <cstdint>
#include <span>
#include <tbb/enumerable_thread_specific.h>
#include <vector>

// internal
#include "source/simulation/core/backend_sim.h"
#include "source/simulation/core/cell_frame.h"



namespace Sim
{
	// forward declarations
	class SimBackend_CPU;

	// ---------------------------------------------------------------------------------------------
	// --- [ brute force implementation ] ----------------------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NS_BruteForce_CPU
		: public I_NeighborSearch
	{
	public:

		explicit NS_BruteForce_CPU(SimBackend_CPU& backend)
			: be_{ backend } {}

		const char* Name() const override { return "BruteForce_CPU"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query_Count(const SimParams& params) override;

		void Query_Fill(const SimParams& params) override;

		void DataReOrder(const SimParams& params) override;

		Mem::ParticleClassMemUsage GetMemUsage(const SimParams& params) const override;

	private:
		
		SimBackend_CPU& be_;

		float kernelRadiusSqu_ = 0.0f;
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ cell linked list compact implementation ] ---------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NS_CellLinkedList_Compact_CPU
		: public I_NeighborSearch
	{
	public:

		explicit NS_CellLinkedList_Compact_CPU(SimBackend_CPU& backend)
			: be_{ backend } {}

		const char* Name() const override { return "CellLinkedList_Compact_CPU"; }

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

	private:

		SimBackend_CPU& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;
		float kernelRadiusSqu_ = 0.0f;
		uint32_t cellQueryReach_ = 1; // query range in cells around each particle
		static constexpr uint32_t N_CELL_FRAME_REBUILD_MARGIN = 8; // hysteresis buffer
		static constexpr uint64_t N_CELL_SEARCH_LINEAR_TAIL = 32; // max elemnt count for linear search
		static constexpr uint32_t N_QUERY_CELL_INLINE = 96;
		static constexpr uint32_t N_QUERY_LOOKUP_CACHE = 256;
		static_assert(std::has_single_bit(N_QUERY_LOOKUP_CACHE),
			"Query lookup cache size must be a power of two.");

		// local grid frame in cell coordinates
		CellCoordFrame cellFrame_ = {};
		bool cellFrameValid_ = false;

		// also marks the cell frame as current, and only holds until a particle moves
		bool cellIdsMatchCurrentParticleOrder_ = false;

		// half-open particle index range of one occupied cell
		struct ActiveCellRange
		{
			uint32_t begin = 0;
			uint32_t end = 0;
		};

		// resolved cell ranges, shared by every query work until the next cell directory build
		struct CellRangeCache
		{
			struct Entry
			{
				uint64_t cellId = EMPTY_64;
				uint32_t generation = 0;
				ActiveCellRange fluid = {};
				ActiveCellRange haloFluid = {};
				ActiveCellRange boundary = {};
				ActiveCellRange haloBoundary = {};
				ActiveCellRange rigid = {};
			};

			std::array<Entry, N_QUERY_LOOKUP_CACHE> entries{};

			Entry* Find(const uint64_t cellId, const uint32_t generation) noexcept
			{
				const uint32_t cacheSlot = static_cast<uint32_t>(cellId ^ (cellId >> 32))
					& (N_QUERY_LOOKUP_CACHE - 1u);
				Entry& entry = entries[cacheSlot];
				if (entry.cellId != cellId || entry.generation != generation)
				{
					return nullptr;
				}

				return &entry;
			}

			Entry& Insert(const uint64_t cellId, const uint32_t generation) noexcept
			{
				const uint32_t cacheSlot = static_cast<uint32_t>(cellId ^ (cellId >> 32))
					& (N_QUERY_LOOKUP_CACHE - 1u);
				Entry& entry = entries[cacheSlot];
				entry = {};
				entry.cellId = cellId;
				entry.generation = generation;
				return entry;
			}
		};

		// per-thread query scratch, reused by every query work a thread runs
		struct QueryMetaScratch
		{
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE> fluidRanges;
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE> boundaryRanges;
			boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE> rigidRanges;
			CellRangeCache cellRangeCache;
		};
		mutable tbb::enumerable_thread_specific<QueryMetaScratch> queryMetaScratch_;

		// stamped onto the cached cell ranges and advanced by every cell directory build
		// (never reset, because the thread-local caches outlive an Init())
		uint32_t cellDirectoryGeneration_ = 0;

		// scratch buffers shared by the serial compact-cell sorting and directory-build phases
		// (radix sorting uses all three buffers, directory building reuses chunkData afterward)
		struct CompactCellBuildScratch
		{
			std::vector<uint64_t> cellIds;
			std::vector<uint32_t> particleOrder;
			std::vector<uint32_t> chunkData;
		};

		// data structures
		std::vector<uint64_t> cellIdF_; // per-particle cell ID
		std::vector<uint32_t> cellStartIdxF_; // per-cell start index into the re-ordered particle SoAs
		std::vector<uint32_t> particleOrderF_; // re-order scratch permutation for fluid particles
		std::vector<uint64_t> uniqueCellIdF_; // per-cell IDs of sorted non-empty cells
		std::vector<uint32_t> cellStartIdxHaloF_;
		std::vector<uint64_t> uniqueCellIdHaloF_;

		std::vector<uint64_t> cellIdB_;
		std::vector<uint32_t> cellStartIdxB_;
		std::vector<uint32_t> particleOrderB_;
		std::vector<uint64_t> uniqueCellIdB_;
		std::vector<uint32_t> cellStartIdxHaloB_;
		std::vector<uint64_t> uniqueCellIdHaloB_;

		std::vector<uint64_t> cellIdR_;
		std::vector<uint32_t> cellStartIdxR_;
		std::vector<uint32_t> particleOrderR_;
		std::vector<uint64_t> uniqueCellIdR_;

		CompactCellBuildScratch compactCellBuildScratch_;

		// helpers
		void UpdateCellFrame();
		void ReOrderParticleData(const SimParams& params, bool permuteSolverFields);
		const CellRangeCache::Entry& ResolveCellQueryRanges(
			uint64_t cellId, CellRangeCache& cellRangeCache) const noexcept;

		// appends the non-empty cell ranges of an inclusive cell-coordinate box,
		// using cached binary searches over the sorted occupied-cell IDs
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

		// finds the compact cell index of a cell ID among the sorted occupied-cell IDs,
		// narrowing with independent probe pairs and scanning the remaining tail branch-free
		static uint32_t FindCompactCellIdx(
			const std::vector<uint64_t>& uniqueCellIds, uint64_t cellId) noexcept;

		inline bool IsInsideSimDomain(float x, float y, float z) const noexcept;
		inline bool IsInsideCellFrame(int cx, int cy, int cz) const noexcept;
		inline uint64_t MapToCellId_Morton(int cx, int cy, int cz) const noexcept;
		static inline uint64_t SpreadBitsWithGaps_21to63(uint32_t v) noexcept;
	};

} // namespace Sim
