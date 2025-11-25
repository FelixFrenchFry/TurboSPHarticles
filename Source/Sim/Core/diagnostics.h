// diagnostics.h
//
// TODO: comment

#pragma once

// external
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

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

		static void PrintSimSetup(const SimulationMgr& simMgr);

		static void PrintMemoryUsage();

		static void PrintPerformanceSummaryAndProgress(
			const Bench::BenchmarksMgr& benchMgr,
			const std::vector<Bench::BenchOpGroupDef>& opGroups,
			Bench::TimeUnit unit,
			int64_t step,
			int64_t totalSteps = -1);

		// TODO: print big performance and statistics tables in-place after every step?

		// TODO: print current step and steps/sec periodically?
	};

} // namespace Sim
