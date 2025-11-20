// neighbor_search.h
//
// TODO: comment

#pragma once

// external
#include <ankerl/unordered_dense.h>
#include <cstdint>
#include <limits>
#include <vector>

// internal
#include "Source/Sim/Core/backend_sim.h"



namespace Sim
{
	// forward declarations
	class SimBackend_CPU01;

	class NS_BruteForce_CPU01 : public I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		explicit NS_BruteForce_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "BruteForce_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query(const SimParams& params) override;

	private: // ----------------------------------------------------------------
		
		SimBackend_CPU01& be_;
	};

	class NS_SpatialHashing_Basic_CPU01 : public I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		explicit NS_SpatialHashing_Basic_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "SpatialHashing_Basic_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query(const SimParams& params) override;

	private: // ----------------------------------------------------------------

		SimBackend_CPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;

		// grid of cells, each made up of a key and a bucket
		using Key = uint64_t;
		using Bucket = std::vector<uint32_t>;

		// data structures
		ankerl::unordered_dense::map<Key, Bucket> cellGridF_{};
		ankerl::unordered_dense::map<Key, Bucket> cellGridB_{};

		// helpers
		inline uint32_t ZigZag(int v) noexcept;
		inline uint64_t PackKey(int cx, int cy, int cz) noexcept;
		inline int FloorToInt(float v) const noexcept;
	};

	class NS_CellLinkedList_Basic_CPU01 : public I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		explicit NS_CellLinkedList_Basic_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "CellLinkedList_Basic_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query(const SimParams& params) override;

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
		// TODO: make them const or static?
		inline uint32_t MapToCellId_Basic(uint32_t cx, uint32_t cy, uint32_t cz) const noexcept;
		inline int FloorToInt(float v) const noexcept;
		inline uint32_t ClampInt(int v, int low, int high) const noexcept;
	};

	class NS_CellLinkedList_Compact_CPU01 : public I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		explicit NS_CellLinkedList_Compact_CPU01(SimBackend_CPU01& backend)
			: be_{ backend } {}

		const char* Name() const override { return "CellLinkedList_Compact_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query(const SimParams& params) override;

	private: // ----------------------------------------------------------------

		SimBackend_CPU01& be_;

		// grid metadata
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;

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
		static inline uint64_t MapToCellId_Basic(uint32_t cx, uint32_t cy, uint32_t cz) noexcept;
		static inline uint64_t MapToCellId_Morton(uint32_t cx, uint32_t cy, uint32_t cz) noexcept;
		static inline uint64_t SpreadBitsWithGaps_21to63(uint32_t v) noexcept;
		static inline uint32_t ZigZag(int v) noexcept;
		static inline int FloorToInt(float v) noexcept;
	};

} // namespace Sim
