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
	public: // -----------------------------------------------------------------

		DiagnosticsMgr() = delete;

		static void PrintHardwareSpecs(uint32_t N_totalThreads,
		                               uint32_t N_simThreads,
		                               uint32_t N_visThreads);

		static void PrintSimParams(const SimParams& params);

		static void PrintSimSetup(const SimulationMgr& simMgr,
								  const SimParams& params);

		static void PrintMemoryUsage();

		static void PrintLatestPerformanceSummaryAndProgress(
			const Bench::BenchmarksMgr& benchMgr,
			const std::vector<Bench::BenchOpGroupDef>& opGroups,
			Bench::TimeUnit unit,
			uint32_t step,
			int64_t maxSteps = -1);

		static void PrintAveragePerformanceSummary(
			const Bench::BenchmarksMgr& benchMgr,
			const std::vector<Bench::BenchOpGroupDef>& opGroups,
			Bench::TimeUnit unit,
			uint32_t step);

		// TODO: print big performance and statistics tables in-place after every step?
	};

} // namespace Sim
