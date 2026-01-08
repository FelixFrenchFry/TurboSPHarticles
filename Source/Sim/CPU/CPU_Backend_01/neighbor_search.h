// neighbor_search.h
//
// TODO: comment

#pragma once

// external
#include <ankerl/unordered_dense.h>
#include <boost/container/small_vector.hpp>
#include <cstdint>
#include <limits>
#include <tbb/enumerable_thread_specific.h>
#include <vector>

// internal
#include "Source/Sim/Core/backend_sim.h"



namespace Sim
{
	// forward declarations
	class SimBackend_CPU01;

	// -------------------------------------------------------------------------
	// --- [ brute force implementation ] --------------------------------------
	// -------------------------------------------------------------------------

	class NS_BruteForce_CPU01 : public I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		explicit NS_BruteForce_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "BruteForce_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query(const SimParams& params) override;

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

	private: // ----------------------------------------------------------------
		
		SimBackend_CPU01& be_;

		// query scratch
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_expctMaxNbrCountF>> scratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_expctMaxNbrCountB>> scratchFB_;
	};

	// -------------------------------------------------------------------------
	// --- [ spatial hashing basic implementation ] ----------------------------
	// -------------------------------------------------------------------------

	class NS_SpatialHashing_Basic_CPU01 : public I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		explicit NS_SpatialHashing_Basic_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "SpatialHashing_Basic_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query(const SimParams& params) override;

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

	private: // ----------------------------------------------------------------

		SimBackend_CPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;

		// grid of cells, each made up of a key and a bucket
		using Key = uint64_t;
		using Bucket = std::vector<uint32_t>;

		// query scratch
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_expctMaxNbrCountF>> scratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_expctMaxNbrCountB>> scratchFB_;

		// data structures
		ankerl::unordered_dense::map<Key, Bucket> cellGridF_{};
		ankerl::unordered_dense::map<Key, Bucket> cellGridB_{};
		//std::unordered_map<Key, Bucket> cellGridF_{};
		//std::unordered_map<Key, Bucket> cellGridB_{};

		// helpers
		static inline uint32_t ZigZag(int v) noexcept;
		static inline uint64_t PackKey(int cx, int cy, int cz) noexcept;
		static inline int FloorToInt(float v) noexcept;
	};

	// -------------------------------------------------------------------------
	// --- [ cell linked list basic implementation ] ---------------------------
	// -------------------------------------------------------------------------

	class NS_CellLinkedList_Basic_CPU01 : public I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		explicit NS_CellLinkedList_Basic_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "CellLinkedList_Basic_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query(const SimParams& params) override;

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

	private: // ----------------------------------------------------------------

		SimBackend_CPU01& be_;

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

		// query scratch
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_expctMaxNbrCountF>> scratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_expctMaxNbrCountB>> scratchFB_;

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
		static inline int FloorToInt(float v) noexcept;
		static inline uint32_t ClampInt(int v, int low, int high) noexcept;
	};

	// -------------------------------------------------------------------------
	// --- [ cell linked list compact implementation ] -------------------------
	// -------------------------------------------------------------------------

	class NS_CellLinkedList_Compact_CPU01 : public I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		explicit NS_CellLinkedList_Compact_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "CellLinkedList_Compact_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query(const SimParams& params) override;

		void Query_Basic(const SimParams& params);

		void Query_Smart(const SimParams& params);

		void DataReOrder(const SimParams& params) override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

		// TODO: implement query with BigMin-LitMax and ternary search and caching of results?

	private: // ----------------------------------------------------------------

		SimBackend_CPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;
		static constexpr uint32_t N_NbrDim_ = 3;
		static constexpr uint32_t N_NbrCells_ = N_NbrDim_ * N_NbrDim_ * N_NbrDim_;

		// query scratch
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_expctMaxNbrCountF>> scratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_expctMaxNbrCountB>> scratchFB_;

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
		static inline uint64_t MapToCellId_XYZ(int cx, int cy, int cz) noexcept;
		static inline uint64_t MapToCellId_Morton(int cx, int cy, int cz) noexcept;
		static inline uint64_t SpreadBitsWithGaps_21to63(uint32_t v) noexcept;
		static inline uint32_t ZigZag(int v) noexcept;
		static inline int NeighborOffset(int rx, int ry, int rz) noexcept;
		static inline int FloorToInt(float v) noexcept;
	};

} // namespace Sim
