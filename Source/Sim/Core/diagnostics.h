// diagnostics.h
//
// TODO: comment

#pragma once

// external
#include <chrono>
#include <cstdint>

// internal
#include "parameters_sim.h"
#include "simulation.h"



namespace Sim
{
	class DiagnosticsMgr
	{
	public: // ---------------------------------------------------------------------------------------

		DiagnosticsMgr() = delete;

		static void PrintHardwareSpecs(uint32_t N_totalThreads,
		                               uint32_t N_simThreads,
		                               uint32_t N_visThreads);

		static void PrintSimParams(const SimParams& params);

		static void PrintSimSetup(const SimulationMgr& simMgr,
								  const SimParams& params);

		static void PrintLatestPerformanceSummaryAndProgress(
			const Bench::BenchmarksTracker& benchmarksTracker,
			const std::vector<Bench::BenchOpGroupDef>& opGroups,
			Bench::TimeUnit unit,
			uint32_t step,
			int64_t maxSteps = -1);

		static void PrintMemorySummary_Standalone(
			const Mem::MemoryTracker& memTracker,
			Mem::ByteUnit unit,
			Base::BackendVersion backendVersion);

		static void PrintStatisticsSummary_Standalone(
			const Stats::StatisticsTracker& statisticsTracker,
			const SimParams& params);

		static void PrintNeighborListByteBreakdown_Standalone(
			const I_SimBackend& simBackend,
			const SimParams& params,
			Mem::ByteUnit unit);

		static void PrintNeighborListByteBreakdownAverages_Standalone(
			const NeighborListByteBreakdownAverages& avgs,
			const SimParams& params,
			Mem::ByteUnit unit);

		static void PrintNeighborListDeltaDistribution_Standalone(
			const I_SimBackend& simBackend,
			const SimParams& params);
	};

} // namespace Sim
