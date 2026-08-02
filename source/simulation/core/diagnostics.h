// diagnostics.h

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
	public:

		DiagnosticsMgr() = delete;

		static void PrintBuildConfig();

		static void PrintHardwareSpecs(uint32_t N_totalThreads,
		                               uint32_t N_simThreads,
		                               uint32_t N_visThreads);

		static void PrintSimParams(const SimParams& params);

		static void PrintSimSetup(const SimulationMgr& simMgr,
								  const SimParams& params);

		static void ResetProgress();

		static void PrintFinalProgressPeakRam(uint64_t N_peakResidentBytes);

		static void PrintLatestPerformanceSummaryAndProgress(
			const Bench::BenchmarksTracker& benchmarksTracker,
			const Bench::BenchOpList& benchOps,
			Bench::TimeUnit unit,
			uint64_t step,
			int64_t maxSteps = -1,
			uint64_t N_peakResidentBytes = 0);

		static void PrintMemorySummary_Standalone(
			const Mem::MemoryTracker& memTracker,
			Mem::ByteUnit unit,
			Base::BackendVersion backendVersion);
	};

} // namespace Sim
