// backend_sim.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>
#include <memory>
#include <vector>

// internal
#include "Source/Base/enum_hub.h"
#include "Source/Eval/Bench/benchmarks.h"
#include "Source/Eval/Bench/operations.h"
#include "Source/Eval/Stats/statistics.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Sync/snapshots.h"
#include "Source/Sim/Sync/state_sim.h"



namespace Sim
{
	// forward declarations
	class I_NeighborSearchStrat;

	// -------------------------------------------------------------------------
	// --- [ simulation backend interface ] ------------------------------------
	// -------------------------------------------------------------------------

	class I_SimBackend
	{
	public: // -----------------------------------------------------------------

		virtual ~I_SimBackend() = default;

		virtual const char* Name() const = 0;

		virtual void SetNeighborSearchStrategy(std::unique_ptr<I_NeighborSearchStrat> strat) = 0;

		// copies & translates a universal simulation state into -> backend-specific data structures
		virtual void PullFromUniversalSimState(const Sync::UniversalSimState& src) = 0;

		// copies & translates backend-specific data structures into -> universal simulation state
		virtual void PushToUniversalSimState(Sync::UniversalSimState& dst) const = 0;

		// packs the current simulation state into a snapshot used for exporting or visualization
		// (implementation is backend-specific for better performance)
		virtual void PackExportSnapshot(Sync::ExportSnapshot& snap, const SimParams& simParams) const = 0;
		virtual void PackVisualSnapshot(Sync::VisualSnapshot& snap) const = 0;

		// computes & fills in the scalar data based on different color coding modes
		// (backend-specific so that possible transformations can be applied efficiently)
		virtual void FillScalarData(Sync::VisualSnapshot& snap, Base::ColorCodingMode mode) const = 0;

		// computes & fills in the stat data for the current sim step
		// (backend-specific so that the stats can be computed from the custom data structures efficiently)
		virtual void FillStatisticsRecord(Stats::StatRecordStep& record, const SimParams& simParams) const = 0;

		// advances the simulation by N_steps
		virtual void ExecuteSimStep(const SimParams&, uint32_t N_steps = 1) = 0;

		// IISPH fluid solver step
		virtual void Step_IISPH(const SimParams& params) = 0;

		// provides the list defining how computation operations are benchmarked in groups and/or individually
		// (backend-specific so that important cache-efficiency optimization can be applied)
		virtual void GetBenchOpGroups(std::vector<Bench::BenchOpGroupDef>& out) const = 0;
	};

	// backend factory
	std::unique_ptr<I_SimBackend> MakeSimBackend(Base::BackendVersion version,
												 Bench::BenchmarksMgr& benchmarksMgr,
												 const SimParams& params);

	// -------------------------------------------------------------------------
	// --- [ neighbor search and list interface ] ------------------------------
	// -------------------------------------------------------------------------

	class I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		virtual ~I_NeighborSearchStrat() = default;

		virtual const char* Name() const = 0;

		virtual void Init(const SimParams& params) = 0;

		virtual void Construct(const SimParams& params) = 0;

		virtual void Query(const SimParams& params) = 0;

		virtual void DataReOrder(const SimParams& params) = 0;
	};

	struct NeighborSpan
	{
		const uint32_t* data = nullptr;
		uint32_t count = 0;
	};

	class I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		virtual ~I_NeighborLists() = default;

		virtual void Resize(uint32_t N_F, uint32_t N_B) = 0;

		// clears the fluid and boundary neighbor list of a single fluid particle i
		virtual void ClearOne(uint32_t i) = 0;

		// clears all neighbor lists
		virtual void ClearAll() = 0;

		virtual void Finalize() = 0;

		// adds single fluid/boundary neighbor j of fluid particle i to its neighbor list
		virtual void AddNbrFF(uint32_t i, uint32_t j) = 0;
		virtual void AddNbrFB(uint32_t i, uint32_t j) = 0;

		// adds bulk of fluid/boundary neighbors js of fluid particle i by replacing its neighbor list
		virtual void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) = 0;
		virtual void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) = 0;

		// gets a "span" of the fluid/boundary neighbor list of fluid particle i
		virtual NeighborSpan FF(uint32_t i) const = 0;
		virtual NeighborSpan FB(uint32_t i) const = 0;
	};

} // namespace Sim
