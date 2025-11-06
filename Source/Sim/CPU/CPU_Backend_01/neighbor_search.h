// neighbor_search.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>
#include <ankerl/unordered_dense.h>
#include <vector>

// internal
#include "Source/Sim/Core/backend_sim.h"



namespace Sim
{
	// forward declarations
	class SimBackend_CPU01;

	// TODO: add this as a friend class and let it access private members
	// TODO: instead of neighbor search ctx structs that have to be adapted per backend?
	class NS_BruteForce_CPU01 : public I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		explicit NS_BruteForce_CPU01(SimBackend_CPU01& backend);

		const char* Name() const override { return "BruteForce_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query(const SimParams& params) override;

	private: // ----------------------------------------------------------------
		
		SimBackend_CPU01& be_;
	};

	// TODO: add this as a friend class and let it access private members
	// TODO: instead of neighbor search ctx structs that have to be adapted per backend?
	class NS_SpatialHashingBasic_CPU01 : public I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		explicit NS_SpatialHashingBasic_CPU01(SimBackend_CPU01& backend);

		const char* Name() const override { return "SpatialHashing_CPU01"; }

		void Init(const SimParams& params) override;

		void Construct(const SimParams& params) override;

		void Query(const SimParams& params) override;

	private: // ----------------------------------------------------------------

		SimBackend_CPU01& be_;

		// grid of cells, each made up of a key and a bucket
		using Key = uint64_t;
		using Bucket = std::vector<uint32_t>;
		using CellGrid = ankerl::unordered_dense::map<Key, Bucket>;

		// data structures
		CellGrid cellGridF_{};
		CellGrid cellGridB_{};

		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;

		// helpers
		inline int FloorToInt(float v) noexcept;
		inline uint32_t ZigZag(int v) noexcept;
		inline uint64_t PackKey(int cx, int cy, int cz) noexcept;
	};

} // namespace Sim
