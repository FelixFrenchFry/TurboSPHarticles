// diagnostics.h
#pragma once

// external
#include <cstdint>
#include <string>

// internal
#include "parameters_sim.h"
#include "simulation.h"



namespace Sim
{
	inline constexpr int N_CLI_TABLE_CHARS = 90;
	inline constexpr int N_CLI_BENCH_LABEL_CHARS = N_CLI_TABLE_CHARS - 50;
	inline const std::string CLI_SEPARATOR(N_CLI_TABLE_CHARS, '-');

	class DiagnosticsMgr
	{
	public:

		DiagnosticsMgr() = delete;

		static void PrintCliSeparator();
		static void PrintCliSectionHeader(const char* title);

		static void PrintBuildConfig();

		static void PrintHardwareSpecs(uint32_t N_totalThreads,
		                               uint32_t N_simThreads,
		                               uint32_t N_visThreads);

		static void PrintSimParams(const SimParams& params);

		static void PrintSimSetup(const SimulationMgr& simMgr,
								  const SimParams& params);

		static bool ShouldOverwriteProgress(Base::ProgressReportMode mode);

		static void ResetProgress(uint64_t step);
		static void SetProgressRunning(bool running);

		static void PrintFinalProgressPeakRam(uint64_t N_residentBytes,
											  uint64_t N_peakResidentBytes,
											  uint64_t N_deviceResidentBytes = 0,
											  uint64_t N_peakDeviceResidentBytes = 0);

		static void PrintLatestPerformanceSummaryAndProgress(
			const Bench::BenchmarksTracker& benchmarksTracker,
			const Bench::BenchOpList& benchOps,
			Bench::TimeUnit unit,
			Base::ProgressReportMode progressReportMode,
			uint64_t step,
			int64_t maxSteps = -1,
			uint64_t N_residentBytes = 0,
			uint64_t N_peakResidentBytes = 0,
			uint64_t N_deviceResidentBytes = 0,
			uint64_t N_peakDeviceResidentBytes = 0);

		static void PrintMemorySummary_Standalone(
			const Mem::MemoryTracker& memTracker,
			Mem::ByteUnit unit,
			Base::BackendType backendType);
	};

} // namespace Sim
