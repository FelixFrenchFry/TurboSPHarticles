// dist_diagnostics.h

#pragma once

// external
#include <array>
#include <cstdint>
#include <vector>

// internal
#include "source/eval/bench/operations.h"



namespace Sim
{
	struct SimParams;
}

namespace Dist
{
	class MpiContext;
	struct ParticleRedistributionTimingReport;

	// MPI tags for distributed diagnostic reports
	inline constexpr int DIST_DIAGNOSTICS_TAG_HARDWARE_REPORT = 8100;
	inline constexpr int DIST_DIAGNOSTICS_TAG_STEP_REPORT = 8101;
	inline constexpr int DIST_DIAGNOSTICS_TAG_STARTUP_REPORT = 8102;
	inline constexpr int DIST_DIAGNOSTICS_TAG_BENCH_REPORT = 8103;
	inline constexpr int DIST_DIAGNOSTICS_TAG_REDISTRIBUTION_TIMING_REPORT = 8104;

	constexpr int DIST_STARTUP_REPORT_TEXT_CHARS = 128;

	// root-only summary of the distributed setup after initialization
	struct DistSimSetupReport
	{
		int mpiThreadSupport = 0;

		const char* domainDecompMethod = "";
		const char* distHaloMode = "";
		const char* loadBalanceMetric = "";
		const char* loadBalanceTrigger = "";
		const char* loadBalanceWeightModel = "";
		const char* backendVersion = "";
		const char* scenarioType = "";
		const char* neighborSearchMethod = "";
		const char* neighborListStorage = "";

		uint32_t N_globalF = 0;
		uint32_t N_globalB = 0;
		uint32_t N_globalR = 0;
		uint32_t N_globalRB = 0;
	};

	// rank-local report gathered by root for startup tables
	struct DistStartupRankReport
	{
		int rank = 0;

		char neighborRanks[DIST_STARTUP_REPORT_TEXT_CHARS] = {};

		uint32_t N_ownedF = 0;
		uint32_t N_ownedB = 0;
		uint32_t N_replicatedR = 0;
	};

	// rank-local report gathered by root for distributed progress output
	struct DistRankStepReport
	{
		int rank = 0;

		uint64_t step = 0;
		int64_t maxSteps = -1;

		uint32_t N_ownedF = 0;
		uint32_t N_innerF = 0;
		uint32_t N_outerF = 0;
		uint32_t N_haloF = 0;
		uint32_t N_ownedB = 0;
		uint32_t N_innerB = 0;
		uint32_t N_outerB = 0;
		uint32_t N_haloB = 0;

		uint32_t N_sentMigratedF = 0;
		uint32_t N_recvMigratedF = 0;
		uint32_t N_sentMigratedB = 0;
		uint32_t N_recvMigratedB = 0;

		uint32_t N_sentHaloF = 0;
		uint32_t N_recvHaloF = 0;
		uint32_t N_sentHaloB = 0;
		uint32_t N_recvHaloB = 0;

		uint64_t totalStepLatestNs = 0;
		uint64_t totalStepAvgNs = 0;

		uint64_t N_latestMemUsedBytes = 0;
		uint64_t N_latestMemAllocBytes = 0;
		uint64_t N_maxMemUsedBytes = 0;
		uint64_t N_maxMemAllocBytes = 0;

		uint64_t N_maxBackendMemAllocBytes = 0;
		uint64_t N_maxDistGlobalStateMemAllocBytes = 0;
		uint64_t N_maxDistLocalStateMemAllocBytes = 0;
		uint64_t N_maxDistParticleMetadataMemAllocBytes = 0;
		uint64_t N_maxDistExchangeBuffersMemAllocBytes = 0;
		uint64_t N_maxDistMiscMemAllocBytes = 0;
		uint64_t N_maxTotalTrackedMemAllocBytes = 0;

		uint8_t exportedStep = 0;
		uint64_t N_processResidentBytes = 0;
		uint64_t N_processPeakResidentBytes = 0;
	};

	// rank-local benchmark report gathered by root for distributed timing rows
	struct DistRankBenchReport
	{
		int rank = 0;
		uint64_t step = 0;

		std::array<uint64_t, Bench::GetIdx(Bench::BenchOp::Count)> latestNs{};
		std::array<uint64_t, Bench::GetIdx(Bench::BenchOp::Count)> avgNs{};
	};

	class DistDiagnosticsMgr
	{
	public:

		DistDiagnosticsMgr() = delete;

		static void PrintBuildConfig(const MpiContext& mpiContext);

		static void PrintHardwareSpecs(
			const MpiContext& mpiContext,
			uint32_t N_totalThreads,
			uint32_t N_simThreads,
			uint32_t N_visThreads);

		static void PrintSimParams(
			const MpiContext& mpiContext,
			const Sim::SimParams& params);

		static void PrintDistSimSetup(
			const MpiContext& mpiContext,
			const DistSimSetupReport& setup,
			const DistStartupRankReport& localReport);

		static void ResetProgress();

		static void PrintFinalProgressPeakRam(
			const MpiContext& mpiContext,
			const DistRankStepReport& localStepReport);

		static void PrintLatestPerformanceSummaryAndProgress(
			const MpiContext& mpiContext,
			const Sim::SimParams& params,
			const Bench::BenchOpList& benchOps,
			const DistRankStepReport& localStepReport,
			const DistRankBenchReport& localBenchReport,
			const ParticleRedistributionTimingReport&
				localParticleRedistributionTimingReport);

		static void PrintMemorySummary(
			const MpiContext& mpiContext,
			const Sim::SimParams& params,
			const DistRankStepReport& localStepReport);
	};

} // namespace Dist
