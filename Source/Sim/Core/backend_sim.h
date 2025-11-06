// backend_sim.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

// internal
#include "parameters_sim.h"
#include "Source/Base/enum_hub.h"
#include "Source/Eval/Bench/benchmarks.h"
#include "Source/Eval/Bench/operations.h"
#include "Source/Eval/Stats/statistics.h"
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
		virtual void PackExportSnapshot(Sync::ExportSnapshot& snap) const = 0;
		virtual void PackVisualSnapshot(Sync::VisualSnapshot& snap) const = 0;

		// computes & fills in the scalar data based on different color coding modes
		// (backend-specific so that possible transformations can be applied efficiently)
		virtual void FillScalarData(Sync::VisualSnapshot& snap, Base::ColorCodingMode mode) const = 0;

		// computes & fills in the stat data for the current sim step
		// (backend-specific so that the stats can be computed from the custom data structures efficiently)
		virtual void FillStatisticsRecord(Stats::StatRecordStep& record, const SimParams& simParams) const = 0;

		// advances the simulation by N_steps
		virtual void ExecuteSimStep(SimParams params, int64_t N_steps = 1) = 0;

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
	// --- [ neighbor search interface ] ---------------------------------------
	// -------------------------------------------------------------------------

	class I_NeighborSearchStrat
	{
	public: // -----------------------------------------------------------------

		virtual ~I_NeighborSearchStrat() = default;

		virtual const char* Name() const = 0;

		virtual void Init(const SimParams& params) = 0;

		virtual void Construct(const SimParams& params) = 0;

		virtual void Query(const SimParams& params) = 0;

		// TODO: add re-order and sorting stuff?
	};

} // namespace Sim
