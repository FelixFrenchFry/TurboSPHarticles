// dist_diagnostics.h

#pragma once

// external
#include <array>
#include <cstdint>
#include <vector>

// internal
#include "Source/Eval/Bench/operations.h"



namespace Sim
{
	struct SimParams;
}

namespace Dist
{
	class MpiContext;
	struct DistSyncTimingReport;

	constexpr int DIST_STARTUP_REPORT_TEXT_CHARS = 128;

	// root-only summary of the distributed setup after initialization
	struct DistSimSetupReport
	{
		int mpiThreadSupport = 0;

		const char* domainDecompMethod = "";
		const char* backendVersion = "";
		const char* scenarioType = "";
		const char* neighborSearchMethod = "";
		const char* neighborListStorage = "";

		uint32_t globalFluidParticleCount = 0;
		uint32_t globalBoundaryParticleCount = 0;
		uint32_t globalRigidParticleCount = 0;
		uint32_t globalRigidBodyCount = 0;
	};

	// rank-local report gathered by root for startup tables
	struct DistStartupRankReport
	{
		int rank = 0;

		char neighborRanks[DIST_STARTUP_REPORT_TEXT_CHARS] = {};

		uint32_t ownedFluidParticleCount = 0;
		uint32_t ownedBoundaryParticleCount = 0;
		uint32_t replicatedRigidParticleCount = 0;
	};

	// rank-local report gathered by root for distributed progress output
	struct DistRankStepReport
	{
		int rank = 0;

		uint64_t step = 0;
		int64_t maxSteps = -1;

		uint32_t ownedFluidParticleCount = 0;
		uint32_t innerFluidParticleCount = 0;
		uint32_t outerFluidParticleCount = 0;
		uint32_t haloFluidParticleCount = 0;
		uint32_t ownedBoundaryParticleCount = 0;
		uint32_t innerBoundaryParticleCount = 0;
		uint32_t outerBoundaryParticleCount = 0;
		uint32_t haloBoundaryParticleCount = 0;

		uint32_t migratedFluidParticleSendCount = 0;
		uint32_t migratedFluidParticleRecvCount = 0;
		uint32_t migratedBoundaryParticleSendCount = 0;
		uint32_t migratedBoundaryParticleRecvCount = 0;

		uint32_t haloFluidParticleSendCount = 0;
		uint32_t haloFluidParticleRecvCount = 0;
		uint32_t haloBoundaryParticleSendCount = 0;
		uint32_t haloBoundaryParticleRecvCount = 0;

		uint64_t totalStepLatestNs = 0;
		uint64_t totalStepAvgNs = 0;

		uint64_t latestMemUsedBytes = 0;
		uint64_t latestMemAllocBytes = 0;
		uint64_t maxMemUsedBytes = 0;
		uint64_t maxMemAllocBytes = 0;

		uint64_t maxBackendMemAllocBytes = 0;
		uint64_t maxDistGlobalStateMemAllocBytes = 0;
		uint64_t maxDistLocalStateMemAllocBytes = 0;
		uint64_t maxDistParticleMetadataMemAllocBytes = 0;
		uint64_t maxDistExchangeBuffersMemAllocBytes = 0;
		uint64_t maxDistMiscMemAllocBytes = 0;
		uint64_t maxTotalTrackedMemAllocBytes = 0;

		uint8_t exportedStep = 0;
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

		static void PrintLatestPerformanceSummaryAndProgress(
			const MpiContext& mpiContext,
			const Sim::SimParams& params,
			const Bench::BenchOpList& benchOps,
			const DistRankStepReport& localStepReport,
			const DistRankBenchReport& localBenchReport,
			const DistSyncTimingReport& localDistSyncTimingReport);

		static void PrintMemorySummary(
			const MpiContext& mpiContext,
			const Sim::SimParams& params,
			const DistRankStepReport& localStepReport);
	};

} // namespace Dist
