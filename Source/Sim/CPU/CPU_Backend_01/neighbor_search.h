// neighbor_search.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>
#include <unordered_map>
#include <vector>

// internal
#include "Source/Sim/Core/backend_sim.h"



namespace Sim
{
	class BruteForce final : public ISimNeighborSearch
	{
	public: // -----------------------------------------------------------------

		BruteForce(Bench::BenchmarksMgr& benchmarksMgr);

		const char* Name() const override { return "BruteForce_CPU_01"; }

		void Init() override;

		void Construct(const NeighborSearchContext& ctx) override;

		void Query(const NeighborSearchContext& ctx,
				   const NeighborSearchResult& out) override;

	private: // ----------------------------------------------------------------

		// managers
		Bench::BenchmarksMgr& benchmarksMgr_;
	};

	class SpatialHashing_Simple final : public ISimNeighborSearch
	{
	public: // -----------------------------------------------------------------

		SpatialHashing_Simple(Bench::BenchmarksMgr& benchmarksMgr);

		const char* Name() const override { return "SpatialHashing_CPU_01"; }

		void Init() override;

		void Construct(const NeighborSearchContext& ctx) override;

		void Query(const NeighborSearchContext& ctx,
				   const NeighborSearchResult& out) override;

	private: // ----------------------------------------------------------------

		// grid of cells, each made up of a key and a bucket
		using Key = uint64_t;
		using Bucket = std::vector<uint32_t>;
		using CellGrid = std::unordered_map<Key, Bucket>;

		// managers
		Bench::BenchmarksMgr& benchmarksMgr_;

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
