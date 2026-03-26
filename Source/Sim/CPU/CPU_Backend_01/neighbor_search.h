// neighbor_search.h
//
// TODO: comment

#pragma once

// external
#include <ankerl/unordered_dense.h>
#include <array>
#include <boost/container/small_vector.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <tbb/enumerable_thread_specific.h>
#include <vector>

// internal
#include "Source/Sim/Core/backend_sim.h"



// TODO: make these used the params based on heuristic expected values?
static constexpr uint32_t N_perScratchInlineFF = 64;
static constexpr uint32_t N_perScratchInlineFB = 48;

namespace Sim
{
	// forward declarations
	class SimBackend_CPU01;

	// ---------------------------------------------------------------------------------------------
	// --- [ brute force implementation ] ----------------------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NS_BruteForce_CPU01 : public I_NeighborSearchStrat
	{
	public: // -------------------------------------------------------------------------------------

		explicit NS_BruteForce_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "BruteForce_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query_Count(const SimParams& params) override;

		void Query_Fill(const SimParams& params) override;

		NeighborSpan QueryParticleFF(uint32_t i) const override;
		NeighborSpan QueryParticleFB(uint32_t i) const override;

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
	// --- [ spatial hashing basic implementation ] ------------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NS_SpatialHashing_Basic_CPU01 : public I_NeighborSearchStrat
	{
	public: // -------------------------------------------------------------------------------------

		explicit NS_SpatialHashing_Basic_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "SpatialHashing_Basic_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query_Count(const SimParams& params) override;

		void Query_Fill(const SimParams& params) override;

		NeighborSpan QueryParticleFF(uint32_t i) const override;
		NeighborSpan QueryParticleFB(uint32_t i) const override;

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

	private: // ------------------------------------------------------------------------------------

		SimBackend_CPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;
		float kernelRadiusSqu_ = 0.0f;
		uint32_t cellQueryReach_ = 1; // query range in cells around each particle

		// local grid frame in cell coordinates
		int cellOrigX_ = 0;
		int cellOrigY_ = 0;
		int cellOrigZ_ = 0;
		uint32_t N_DimX_ = 0;
		uint32_t N_DimY_ = 0;
		uint32_t N_DimZ_ = 0;

		// grid of cells, each made up of a key and a bucket
		using Key = uint64_t;
		using Bucket = std::vector<uint32_t>;

		// custom hash for packed cell keys and final avalanche mix for better bit diffusion
		struct CellKeyHash
		{
			std::size_t operator()(Key key) const noexcept
			{
				// packed key layout [z:21][y:21][x:21]
				constexpr uint64_t mask21 = (1ull << 21) - 1ull;

				// prime-based component mix
				constexpr uint64_t p1 = 11400714785074694791ull; // PRIME64_1 (xxHash)
				constexpr uint64_t p2 = 14029467366897019727ull; // PRIME64_2 (xxHash)
				constexpr uint64_t p3 = 1609587929392839161ull;  // PRIME64_3 (xxHash)

				const uint64_t x = key & mask21;
				const uint64_t y = (key >> 21) & mask21;
				const uint64_t z = (key >> 42) & mask21;

				return (p1 * x) ^ (p2 * y) ^ (p3 * z);
			}
		};

		// query scratch
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> scratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> scratchFB_;

		// data structures
		std::unordered_map<Key, Bucket, CellKeyHash> cellGridF_{};
		std::unordered_map<Key, Bucket, CellKeyHash> cellGridB_{};

		static constexpr float hashTableMaxLoadFactor_ = 0.67f;

		// helpers
		static inline uint64_t PackKey(uint32_t cx, uint32_t cy, uint32_t cz) noexcept;
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ spatial hashing compact implementation ] ----------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NS_SpatialHashing_Compact_CPU01 : public I_NeighborSearchStrat
	{
	public: // -------------------------------------------------------------------------------------

		explicit NS_SpatialHashing_Compact_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "SpatialHashing_Compact_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query_Count(const SimParams& params) override;

		void Query_Fill(const SimParams& params) override;

		NeighborSpan QueryParticleFF(uint32_t i) const override;
		NeighborSpan QueryParticleFB(uint32_t i) const override;

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

	private: // ------------------------------------------------------------------------------------

		SimBackend_CPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;

		// local grid frame in cell coordinates
		int cellOrigX_ = 0;
		int cellOrigY_ = 0;
		int cellOrigZ_ = 0;
		uint32_t N_DimX_ = 0;
		uint32_t N_DimY_ = 0;
		uint32_t N_DimZ_ = 0;

		// grid of cells, each made up of a key and a bucket
		using Key = uint64_t;
		using Bucket = std::vector<uint32_t>;

		// query scratch
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> scratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> scratchFB_;

		// data structures
		ankerl::unordered_dense::map<Key, Bucket> cellGridF_{};
		ankerl::unordered_dense::map<Key, Bucket> cellGridB_{};

		// helpers
		static inline uint64_t PackKey(uint32_t cx, uint32_t cy, uint32_t cz) noexcept;
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ spatial hashing explicit implementation ] ---------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NS_SpatialHashing_Explicit_CPU01 : public I_NeighborSearchStrat
	{
	public: // -------------------------------------------------------------------------------------

		explicit NS_SpatialHashing_Explicit_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "SpatialHashing_Explicit_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query_Count(const SimParams& params) override;

		void Query_Fill(const SimParams& params) override;

		NeighborSpan QueryParticleFF(uint32_t i) const override;
		NeighborSpan QueryParticleFB(uint32_t i) const override;

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

	private: // ------------------------------------------------------------------------------------

		SimBackend_CPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;

		// grid AABB origin
		float xBoxOrig_ = 0.0f;
		float yBoxOrig_ = 0.0f;
		float zBoxOrig_ = 0.0f;

		// grid AABB max corner
		float xBoxMax_ = 0.0f;
		float yBoxMax_ = 0.0f;
		float zBoxMax_ = 0.0f;

		// grid dimensions
		uint32_t N_DimX_ = 0;
		uint32_t N_DimY_ = 0;
		uint32_t N_DimZ_ = 0;
		uint64_t N_Cells_ = 0;

		// grid of cells, each made up of a key and a bucket
		using Key = uint64_t;
		using Bucket = std::vector<uint32_t>;

		// query scratch
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> scratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> scratchFB_;

		// data structures
		std::vector<Bucket> cellGridF_{};
		std::vector<Bucket> cellGridB_{};

		static constexpr uint32_t N_gridMarginCells_ = 2;
		static constexpr float AABB_pad_ = 1e-7f;

		// helpers
		inline Key PackKey(uint32_t cx, uint32_t cy, uint32_t cz) const noexcept;
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ cell linked list basic implementation ] -----------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NS_CellLinkedList_Basic_CPU01 : public I_NeighborSearchStrat
	{
	public: // -------------------------------------------------------------------------------------

		explicit NS_CellLinkedList_Basic_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "CellLinkedList_Basic_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query_Count(const SimParams& params) override;

		void Query_Fill(const SimParams& params) override;

		NeighborSpan QueryParticleFF(uint32_t i) const override;
		NeighborSpan QueryParticleFB(uint32_t i) const override;

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

	private: // ------------------------------------------------------------------------------------

		SimBackend_CPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;
		float kernelRadiusSqu_ = 0.0f;
		uint32_t cellQueryReach_ = 1; // query range in cells around each particle

		// grid AABB origin
		float xBoxOrig_ = 0.0f;
		float yBoxOrig_ = 0.0f;
		float zBoxOrig_ = 0.0f;

		// grid dimensions
		uint32_t N_DimX_ = 0;
		uint32_t N_DimY_ = 0;
		uint32_t N_DimZ_ = 0;
		uint32_t N_Cells_ = 0;

		// query scratch
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> scratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> scratchFB_;

		// data structures
		std::vector<uint32_t> cellStartIdxF_; // per-cell start index in the sorted particle index array
		std::vector<uint32_t> cellEndIdxF_; // per-cell end index in the sorted particle index array
		std::vector<uint32_t> cellIdF_; // per-particle cell ID array
		std::vector<uint32_t> sortedParticleIdxF_; // sorted fluid particle index array (permutation)

		std::vector<uint32_t> cellStartIdxB_;
		std::vector<uint32_t> cellEndIdxB_;
		std::vector<uint32_t> cellIdB_;
		std::vector<uint32_t> sortedParticleIdxB_;

		static constexpr uint32_t EMPTY = std::numeric_limits<uint32_t>::max();

		// helpers
		inline uint32_t MapToCellId_Basic(uint32_t cx, uint32_t cy, uint32_t cz) const noexcept;
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ cell linked list compact implementation ] ---------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NS_CellLinkedList_Compact_CPU01 : public I_NeighborSearchStrat
	{
	public: // -------------------------------------------------------------------------------------

		explicit NS_CellLinkedList_Compact_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "CellLinkedList_Compact_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query_Count(const SimParams& params) override;

		void Query_Fill(const SimParams& params) override;

		NeighborSpan QueryParticleFF(uint32_t i) const override;
		NeighborSpan QueryParticleFB(uint32_t i) const override;

		void Query_Fill_Basic(const SimParams& params);

		void Query_Fill_Smart(const SimParams& params);

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

		// TODO: implement query with BigMin-LitMax and ternary search and caching of results?

	private: // ------------------------------------------------------------------------------------

		SimBackend_CPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;
		float kernelRadiusSqu_ = 0.0f;
		Base::GridCellIndexing gridCellIndexing_ = Base::GridCellIndexing::Morton;
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

		// query scratch
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> scratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> scratchFB_;

		// data structures
		std::vector<uint64_t> uniqueCellIdF_; // unique cell IDs of non-empty cells
		std::vector<uint32_t> cellStartIdxF_; // per-cell start index in the sorted particle index array
		std::vector<uint64_t> cellIdF_; // per-particle cell ID array
		std::vector<uint32_t> sortedParticleIdxF_; // sorted fluid particle index array (permutation)

		std::vector<uint64_t> uniqueCellIdB_;
		std::vector<uint32_t> cellStartIdxB_;
		std::vector<uint64_t> cellIdB_;
		std::vector<uint32_t> sortedParticleIdxB_;

		static constexpr uint32_t EMPTY_32 = std::numeric_limits<uint32_t>::max();
		static constexpr uint64_t EMPTY_64 = std::numeric_limits<uint64_t>::max();

		// helpers
		void UpdateCellFrame();
		inline uint32_t NeighborCellCount() const noexcept;
		inline uint64_t MapToCellId(int cx, int cy, int cz) const noexcept;
		inline uint64_t MapToCellId_XYZ(int cx, int cy, int cz) const noexcept;
		inline uint64_t MapToCellId_Morton(int cx, int cy, int cz) const noexcept;
		static inline uint64_t SpreadBitsWithGaps_21to63(uint32_t v) noexcept;
		inline int NeighborOffset(int rx, int ry, int rz) const noexcept;
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ bounding volume hierarchy linear (LBVH) implementation ] ------------------------------
	// ---------------------------------------------------------------------------------------------

	class NS_BoundVolHier_Linear_LBVH_CPU01 : public I_NeighborSearchStrat
	{
	public: // -------------------------------------------------------------------------------------

		explicit NS_BoundVolHier_Linear_LBVH_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "BoundVolHier_Linear_LBVH_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query_Count(const SimParams& params) override;

		void Query_Fill(const SimParams& params) override;

		NeighborSpan QueryParticleFF(uint32_t i) const override;
		NeighborSpan QueryParticleFB(uint32_t i) const override;

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

	private: // ------------------------------------------------------------------------------------

		SimBackend_CPU01& be_;

		// AABB represented by min and max corners
		struct AABB
		{
			float minX = 0.0f, minY = 0.0f, minZ = 0.0f;
			float maxX = 0.0f, maxY = 0.0f, maxZ = 0.0f;
		};

		// internal BVH node (non-leaf)
		struct InternalNode
		{
			AABB box{};
			uint32_t leftChild = 0; // index into nodes_ (internal) or leaves (leaf)
			uint32_t rightChild = 0;
			bool leftIsLeaf = false;
			bool rightIsLeaf = false;
		};

		// traversal stack depth (64 levels supports trees with up to 2^64 leaves)
		static constexpr uint32_t MAX_STACK_DEPTH = 64;

		// query scratch
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> scratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> scratchFB_;

		// --- [ fluid BVH data structures ]

		std::vector<uint32_t> mortonF_; // per-particle 30-bit Morton code
		std::vector<uint32_t> sortedIdxF_; // particle indices sorted by Morton code
		std::vector<InternalNode> nodesF_; // internal nodes (N-1 for N leaves)

		// --- [ boundary BVH data structures ]

		std::vector<uint32_t> mortonB_;
		std::vector<uint32_t> sortedIdxB_;
		std::vector<InternalNode> nodesB_;

		// reusable construction scratch (to reduce per-step allocations)
		std::vector<uint32_t> sortedMortonScratch_;
		std::vector<AABB> leafBoxScratch_;
		std::vector<AABB> sortedLeafBoxScratch_;
		std::vector<uint32_t> parentOfScratch_;

		// boundary BVH cache (boundaries are static in the current CPU01 solver path)
		bool boundaryBVHValid_ = false;
		uint32_t cachedBoundaryCount_ = 0;
		const float* cachedBoundaryXPosPtr_ = nullptr;
		const float* cachedBoundaryYPosPtr_ = nullptr;
		const float* cachedBoundaryZPosPtr_ = nullptr;
		float kernelRadiusSqu_ = 0.0f;
		float leafAabbRadius_ = 0.0f;

		// helpers
		void BuildBVH(const float* xPos, const float* yPos,
			const float* zPos, uint32_t N,
			std::vector<uint32_t>& morton, std::vector<uint32_t>& sortedIdx,
			std::vector<InternalNode>& nodes);

		void QueryBVH_Count(float xi, float yi, float zi,
			float kernelRadiusSqu, const float* xPos, const float* yPos,
			const float* zPos, const std::vector<uint32_t>& sortedIdx,
			const std::vector<InternalNode>& nodes, uint32_t N, uint32_t& count) const;

		template <typename ScratchVec>
		void QueryBVH_Fill(float xi, float yi, float zi,
			float kernelRadiusSqu, const float* xPos, const float* yPos,
			const float* zPos, const std::vector<uint32_t>& sortedIdx,
			const std::vector<InternalNode>& nodes, uint32_t N, ScratchVec& scratch,
			uint32_t* traversedNodeCount = nullptr,
			uint32_t* candCount = nullptr,
			uint32_t* posCount = nullptr) const
		{
			// single leaf BVH: test once and return
			if (N == 1)
			{
				const uint32_t j = sortedIdx[0];
				const float dx = xi - xPos[j];
				const float dy = yi - yPos[j];
				const float dz = zi - zPos[j];
				const float distSqu = dx * dx + dy * dy + dz * dz;

				if (candCount != nullptr)
				{
					(*candCount)++;
				}

				if (distSqu <= kernelRadiusSqu) [[unlikely]]
				{
					if (posCount != nullptr)
					{
						(*posCount)++;
					}

					scratch.push_back(j);
				}
				return;
			}

			std::array<uint32_t, MAX_STACK_DEPTH> stack{};
			constexpr uint32_t LEAF_FLAG = 0x80000000u;
			uint32_t stackTop = 0;

			stack[stackTop++] = 0;

			while (stackTop > 0)
			{
				const uint32_t entry = stack[--stackTop];

				if (entry & LEAF_FLAG)
				{
					const uint32_t leafIdx = entry & ~LEAF_FLAG;
					const uint32_t j = sortedIdx[leafIdx];

					const float dx = xi - xPos[j];
					const float dy = yi - yPos[j];
					const float dz = zi - zPos[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (candCount != nullptr)
					{
						(*candCount)++;
					}

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						if (posCount != nullptr)
						{
							(*posCount)++;
						}

						scratch.push_back(j);
					}
				}
				else
				{
					if (traversedNodeCount != nullptr)
					{
						(*traversedNodeCount)++;
					}

					const auto& node = nodes[entry];

					if (!OverlapsAABB(node.box, xi, yi, zi, kernelRadiusSqu))
					{
						continue;
					}

					if (node.rightIsLeaf)
					{
						stack[stackTop++] = node.rightChild | LEAF_FLAG;
					}
					else
					{
						stack[stackTop++] = node.rightChild;
					}

					if (node.leftIsLeaf)
					{
						stack[stackTop++] = node.leftChild | LEAF_FLAG;
					}
					else
					{
						stack[stackTop++] = node.leftChild;
					}
				}
			}
		}

		static inline AABB MakeParticleAABB(float x, float y, float z, float r) noexcept;
		static inline AABB MergeAABB(const AABB& a, const AABB& b) noexcept;
		static inline bool OverlapsAABB(const AABB& box, float x, float y, float z,
			float radiusSqu) noexcept;
		static inline uint32_t ExpandBits10(uint32_t v) noexcept;
		static inline uint32_t Morton3D_30bit(float nx, float ny, float nz) noexcept;
		static inline int CountLeadingZeros(uint32_t v) noexcept;
		static inline int Delta(const std::vector<uint32_t>& sortedMorton,
			uint32_t N, int i, int j) noexcept;
	};

} // namespace Sim
