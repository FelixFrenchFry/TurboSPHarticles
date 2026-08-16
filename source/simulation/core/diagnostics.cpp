// diagnostics.cpp

// external
#include <algorithm>
#include <cinttypes>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include <string>
#ifdef _WIN32
	#include <windows.h>
#else
	#include <unistd.h>
#endif

// internal
#include "build_config.h"
#include "diagnostics.h"
#include "source/base/helpers.h"
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/core/simulation.h"
#include "source/eval/bench/benchmarks.h"



namespace
{
	using steady_clock = std::chrono::steady_clock;

	constexpr int N_CLI_SEPARATOR_CHARS = 80;
	constexpr int N_CLI_SIM_PARAM_LABEL_CHARS = 38;
	constexpr const char* CLI_SEPARATOR =
		"--------------------------------------------------------------------------------";

	struct ProgressState
	{
		bool initialized = false;
		uint64_t lastStep = 0;
		steady_clock::time_point startSteady{};
		int N_lastLines = 0;
		std::size_t lastProgressLen = 0;
		std::string lastProgressLine{};
	};

	ProgressState& GetProgressState()
	{
		static ProgressState state{};
		return state;
	}

	std::string FormatRamUsage(
		const uint64_t N_residentBytes,
		const uint64_t N_peakResidentBytes)
	{
		char buffer[64];
		std::snprintf(buffer, sizeof(buffer), " | RSS %.2f GB | HWM %.2f GB",
			Mem::ConvertFromBytes(Mem::ByteUnit::GB, N_residentBytes),
			Mem::ConvertFromBytes(Mem::ByteUnit::GB, N_peakResidentBytes));
		return buffer;
	}

	void PrintCliSeparator()
	{
		std::printf("%s\n", CLI_SEPARATOR);
	}

	void PrintCliSectionHeader(const char* title)
	{
		int N_chars = std::printf("--- [ %s ] ", title);
		N_chars = std::max(0, N_chars);

		for (int i = N_chars; i < N_CLI_SEPARATOR_CHARS; i++)
		{
			std::putchar('-');
		}

		std::putchar('\n');
	}

	std::string MakeDisplayExportDir(const std::string& exportDir)
	{
		if (exportDir.empty()) { return "/"; }
		if (exportDir[0] == '/' || exportDir[0] == '\\') { return exportDir; }
		if (exportDir.size() >= 2 && exportDir[1] == ':') { return exportDir; }

		return "/" + exportDir;
	}

} // namespace anonymous

namespace Sim
{
	void DiagnosticsMgr::ResetProgress()
	{
		ProgressState& state = GetProgressState();
		state = ProgressState{};
		state.initialized = true;
		state.startSteady = steady_clock::now();
	}

	void DiagnosticsMgr::PrintFinalProgressPeakRam(
		const uint64_t N_residentBytes,
		const uint64_t N_peakResidentBytes)
	{
		ProgressState& state = GetProgressState();
		if (state.N_lastLines == 0 || state.lastProgressLine.empty()) { return; }

		const std::string marker = " | RSS ";
		const std::size_t markerPos = state.lastProgressLine.find(marker);
		if (markerPos == std::string::npos) { return; }

		std::string line = state.lastProgressLine.substr(0, markerPos);
		line += FormatRamUsage(N_residentBytes, N_peakResidentBytes);
		SPDLOG_INFO("Final progress report:");
		std::printf("\n%s\n\n", line.c_str());
		std::fflush(stdout);
	}

	void DiagnosticsMgr::PrintBuildConfig()
	{
		printf("\n");
		PrintCliSectionHeader("build configuration");
		PrintCliSeparator();

		printf("%-30s = %s\n", "git revision",
            BuildConfig::gitCommitHash);
		printf("%-30s = %s\n", "build type",
            BuildConfig::buildType);
		printf("%-30s = %s %s\n", "compiler",
			BuildConfig::cxxCompilerId,
			BuildConfig::cxxCompilerVersion);
		printf("%-30s = %s (%u lanes)\n", "cpu simd arch",
			BuildConfig::cpuSimdArch,
			BuildConfig::cpuSimdWidth);
		printf("%-30s = %s\n", "cpu isa compiler flags",
			BuildConfig::cpuIsaCompilerFlags);
	}

    void DiagnosticsMgr::PrintHardwareSpecs(uint32_t N_totalThreads,
											uint32_t N_simThreads,
											uint32_t N_visThreads)
    {
        printf("\n");
		PrintCliSectionHeader("hardware specifications");
		PrintCliSeparator();

        // hostname
        char hostname[256] = "unknown";
		#if defined(_WIN32)
			// windows version
			DWORD size = sizeof(hostname);
			GetComputerNameA(hostname, &size);
		#else
			// linux version
			gethostname(hostname, sizeof(hostname));
		#endif

        printf("%-30s = %s\n", "host name", hostname);
        printf("%-30s = %u\n", "avail threads", N_totalThreads);
        printf("%-30s = %u\n", "sim budget", N_simThreads);
        printf("%-30s = %u\n", "vis budget", N_visThreads);
        printf("\n");
    }

    void DiagnosticsMgr::PrintSimParams(const SimParams& params)
    {
        printf("\n");
		PrintCliSectionHeader("simulation parameters");
		PrintCliSeparator();

        // print table with aligned columns of sim parameters
        printf("%-*s = %.6f\n", N_CLI_SIM_PARAM_LABEL_CHARS, "timeStep", params.timeStep);
        printf("%-*s = [ %.6f / %.6f ]\n", N_CLI_SIM_PARAM_LABEL_CHARS, "particleSize [F / B]",
			params.particleSizeF, params.particleSizeB);
        printf("%-*s = [ %.6f / %.6f ]\n", N_CLI_SIM_PARAM_LABEL_CHARS, "particleSizeInv [F / B]",
			params.particleSizeInvF, params.particleSizeInvB);
        printf("%-*s = %.6f\n", N_CLI_SIM_PARAM_LABEL_CHARS, "kernelSuppRatio", params.kernelSuppRatio);
        printf("%-*s = %.6f\n", N_CLI_SIM_PARAM_LABEL_CHARS, "kernelRadius", params.kernelRadius);
        printf("%-*s = %.6f\n", N_CLI_SIM_PARAM_LABEL_CHARS, "kernelRadiusSqu", params.kernelRadiusSqu);
        printf("%-*s = [ %u / %u ]\n", N_CLI_SIM_PARAM_LABEL_CHARS, "expctdNbrCountSoftMax [FF / FB]",
			params.expctdNbrCountSoftMax_FF, params.expctdNbrCountSoftMax_FB);
        printf("%-*s = [ %u / %u ]\n", N_CLI_SIM_PARAM_LABEL_CHARS, "expctdNbrCountHardMax [FF / FB]",
			params.expctdNbrCountHardMax_FF, params.expctdNbrCountHardMax_FB);
        printf("%-*s = [ %u / %u ]\n", N_CLI_SIM_PARAM_LABEL_CHARS, "expctdAvgNbrCountHardMax [FF / FB]",
			params.expctdAvgNbrCountHardMax_FF, params.expctdAvgNbrCountHardMax_FB);
        printf("%-*s = %.6f\n", N_CLI_SIM_PARAM_LABEL_CHARS, "restDensity",
			params.restDensity);
        printf("%-*s = [ %.6f / %.6f ]\n", N_CLI_SIM_PARAM_LABEL_CHARS, "mass [F / B]",
			params.massF, params.massB);
        printf("%-*s = [ %.6f / %.6f ]\n", N_CLI_SIM_PARAM_LABEL_CHARS, "viscosity [F / B]",
			params.viscosityF, params.viscosityB);
        printf("%-*s = (%.3f, %.3f, %.3f)\n", N_CLI_SIM_PARAM_LABEL_CHARS, "gravity",
            params.gravity.x, params.gravity.y, params.gravity.z);
        printf("%-*s = %.6f\n", N_CLI_SIM_PARAM_LABEL_CHARS, "pressGamma", params.pressGamma);
        printf("%-*s = %.6f\n", N_CLI_SIM_PARAM_LABEL_CHARS, "jacobiOmega", params.jacobiOmega);
        printf("%-*s = %.6f\n", N_CLI_SIM_PARAM_LABEL_CHARS, "densityDevTarget", params.densityDevTarget);
        printf("%-*s = %u\n", N_CLI_SIM_PARAM_LABEL_CHARS, "fixedIters", params.fixedIters);
        printf("%-*s = %u\n", N_CLI_SIM_PARAM_LABEL_CHARS, "minSolverIters", params.minSolverIters);
        printf("%-*s = %u\n", N_CLI_SIM_PARAM_LABEL_CHARS, "maxSolverIters", params.maxSolverIters);
        printf("%-*s = %s\n", N_CLI_SIM_PARAM_LABEL_CHARS, "solverTerminationCond",
            Base::ToCString(params.solverTerminationCond));
        printf("%-*s = %.6f\n", N_CLI_SIM_PARAM_LABEL_CHARS, "gridCellSizeRatio", params.gridCellSizeRatio);
        printf("%-*s = [ %.6f / %.6f ]\n", N_CLI_SIM_PARAM_LABEL_CHARS, "jitterFrac [F / B]",
			params.jitterFracF, params.jitterFracB);
		printf("%-*s = [ %" PRId64 " / %" PRId64 " ]\n", N_CLI_SIM_PARAM_LABEL_CHARS, "jitterSeed [F / B]",
			params.jitterSeedF, params.jitterSeedB);
		printf("%-*s = %u\n", N_CLI_SIM_PARAM_LABEL_CHARS, "boundaryLayers", params.boundaryLayers);
		printf("%-*s = %.6f\n", N_CLI_SIM_PARAM_LABEL_CHARS,
			"loadBalanceThreshold", params.loadBalanceThreshold);
		printf("%-*s = %u\n", N_CLI_SIM_PARAM_LABEL_CHARS,
			"loadBalanceSustainSteps", params.loadBalanceSustainSteps);
		printf("%-*s = %u\n", N_CLI_SIM_PARAM_LABEL_CHARS,
			"loadBalanceCooldownSteps", params.loadBalanceCooldownSteps);
		printf("%-*s = %u\n", N_CLI_SIM_PARAM_LABEL_CHARS,
			"loadBalanceIntervalSteps", params.loadBalanceIntervalSteps);
		printf("%-*s = [ %.3f / %.3f / %.3f ]\n", N_CLI_SIM_PARAM_LABEL_CHARS,
			"sampling ratio [min/max/probe]",
			params.surfaceSamplingMinDistanceRatio,
			params.surfaceSamplingMaxDistanceRatio,
			params.surfaceSamplingCoverageProbeSpacingRatio);
		printf("%-*s = [ %.6f / %.6f / %.6f ]\n", N_CLI_SIM_PARAM_LABEL_CHARS,
			"sampling dist [min/max/probe]",
			params.surfaceSamplingMinDistance,
			params.surfaceSamplingMaxDistance,
			params.surfaceSamplingCoverageProbeSpacing);
		printf("%-*s = [ %u / %u ]\n", N_CLI_SIM_PARAM_LABEL_CHARS,
			"sampling candidates / trials",
			params.surfaceSamplingCandidateMultiplier,
			params.surfaceSamplingMaxTrialsPerCell);
		printf("%-*s = %" PRIu64 "\n", N_CLI_SIM_PARAM_LABEL_CHARS,
			"sampling seed", params.surfaceSamplingSeed);
		printf("%-*s = [ %.3f / %u ]\n", N_CLI_SIM_PARAM_LABEL_CHARS,
			"surface nbr [ratio/count]",
			params.surfaceNbrCountRatio,
			params.surfaceNbrCountThreshold);

		printf("%-*s = %s\n", N_CLI_SIM_PARAM_LABEL_CHARS, "exportEnabled", params.exportEnabled ? "true" : "false");
		printf("%-*s = %s\n", N_CLI_SIM_PARAM_LABEL_CHARS, "saveFinalState", params.saveFinalState ? "true" : "false");
		const std::string exportDir = MakeDisplayExportDir(params.exportDir);
        printf("%-*s = %s\n", N_CLI_SIM_PARAM_LABEL_CHARS, "exportDir", exportDir.c_str());
		printf("%-*s = %u\n", N_CLI_SIM_PARAM_LABEL_CHARS, "exportInterval", params.exportInterval);
    }

    void DiagnosticsMgr::PrintSimSetup(const Sim::SimulationMgr& simMgr,
									   const SimParams& params)
    {
        printf("\n");
		PrintCliSectionHeader("simulation setup");
		PrintCliSeparator();

        // modes
		printf("%-30s = %s\n", "sim backend",
			Base::ToCString(params.backendVersion));
        printf("%-30s = %s\n", "sim scenario",
            Base::ToCString(params.scenarioType));
        printf("%-30s = %s\n", "neighbor search method",
			Base::ToCString(params.neighborSearchMethod));
        printf("%-30s = %s\n", "neighbor list storage",
            Base::ToEffectiveNeighborListStorageCString(
				params.backendVersion,
				params.neighborListMethod));

        // particle counts
        printf("%-30s = [ %u / %u / %u ]\n", "particles [F / B / R]",
            simMgr.simBackend_->GetNumFluidParticles(),
            simMgr.simBackend_->GetNumBoundaryParticles(),
            simMgr.simBackend_->GetNumRigidParticles());
    }

    void DiagnosticsMgr::PrintLatestPerformanceSummaryAndProgress(
        const Bench::BenchmarksTracker& benchmarksTracker,
        const Bench::BenchOpList& benchOps,
        Bench::TimeUnit unit,
        uint64_t step,
        int64_t maxSteps,
        const uint64_t N_residentBytes,
        const uint64_t N_peakResidentBytes)
    {
        ProgressState& state = GetProgressState();

        const bool hasTotal = (maxSteps > 0);
        const uint64_t maxStepsU = hasTotal ? static_cast<uint64_t>(maxSteps) : 0;
        const auto now = steady_clock::now();

        // re-initialize at start or when step == 0
        if (!state.initialized || step == 0)
        {
            state = ProgressState{};
            state.initialized = true;
            state.startSteady = now;
        }

        if (step <= state.lastStep) { return; }
        state.lastStep = step;

        // total elapsed time since start
        double elapsedS = std::chrono::duration<double>(now - state.startSteady).count();
        elapsedS = std::max(elapsedS, 1e-14);

        const double stepsPerSec = static_cast<double>(step) / elapsedS;

        // remaining time based on total simulation speed (only if total steps known)
        double remainingS = 0.0;
        if (hasTotal && stepsPerSec > 1e-14 && step < maxStepsU)
        {
            remainingS = static_cast<double>(maxStepsU - step) / stepsPerSec;
        }

        auto fmt_hms = [](double seconds, int& h, int& m, int& s)
            {
                seconds = std::max(seconds, 0.0);
                int N_totalSeconds = static_cast<int>(seconds + 0.5);
                h = N_totalSeconds / 3600;
                m = (N_totalSeconds % 3600) / 60;
                s = N_totalSeconds % 60;
            };

        int elH, elM, elS;
        int etaH = 0, etaM = 0, etaS = 0;
        fmt_hms(elapsedS, elH, elM, elS);
        if (hasTotal)
        {
            fmt_hms(remainingS, etaH, etaM, etaS);
        }

        double percent = 0.0;
        if (hasTotal)
        {
            double progress = std::min(1.0, std::max(0.0,
                static_cast<double>(step) / static_cast<double>(maxStepsU)));
            percent = progress * 100.0;
        }

        // move cursor up to overwrite previous block
        if (state.N_lastLines > 0)
        {
            std::printf("\x1b[%dA\r", state.N_lastLines);
        }

        int N_lines = 0;

        // per-operation timing
        std::vector<uint64_t> opLatest(benchOps.size(), 0);
        std::vector<uint64_t> opAvg(benchOps.size(), 0);

        // sum of displayed non-total rows, used to derive unaccounted time
        uint64_t accountedNsLatest = 0;
        uint64_t accountedNsAvg = 0;
        int totalIdx = -1;

        auto SaturatingSubtract = [](const uint64_t a, const uint64_t b)
        {
            return a > b ? a - b : 0;
        };

        for (uint32_t i = 0; i < benchOps.size(); i++)
        {
            const Bench::BenchOp op = benchOps[i];
            const uint64_t opNsLatest = benchmarksTracker.GetBench_Latest(op);
            const uint64_t opNsAvg = benchmarksTracker.GetBench_Avg_OverAll(op);

            opLatest[i] = opNsLatest;
            opAvg[i] = opNsAvg;

            if (op == Bench::BenchOp::Total)
            {
                totalIdx = static_cast<int>(i);
            }
            else
            {
                accountedNsLatest += opNsLatest;
                accountedNsAvg += opNsAvg;
            }
        }

        const uint64_t totalLatestNs = totalIdx >= 0
            ? opLatest[static_cast<std::size_t>(totalIdx)]
            : accountedNsLatest;
        const uint64_t totalAvgNs = totalIdx >= 0
            ? opAvg[static_cast<std::size_t>(totalIdx)]
            : accountedNsAvg;

        const char* unitStr = Bench::UnitLabel(unit);

        char latestLabel[32];
        char avgLabel[32];
        snprintf(latestLabel, sizeof(latestLabel), "latest [%s]", unitStr);
        snprintf(avgLabel, sizeof(avgLabel), "avg [%s]", unitStr);

        std::printf("\n");
        N_lines++;
        // header
        printf("%-30.30s | %12s  %8s | %12s  %8s\n",
            "benchmark summary", latestLabel, "% total", avgLabel, "% total");
        N_lines++;
		PrintCliSeparator();
        N_lines++;

        auto PrintTimingRow =
            [&](const char* name,
                const uint64_t nsLatest,
                const uint64_t nsAvg,
                const bool isTotal)
            {
                const double t_latest = Bench::ConvertFromNs(unit, nsLatest);
                const double t_avg = Bench::ConvertFromNs(unit, nsAvg);

                double pctLatest = 0.0;
                double pctAvg = 0.0;

                if (!isTotal)
                {
                    if (totalLatestNs > 0)
                    {
                        pctLatest = 100.0 * static_cast<double>(nsLatest)
                            / static_cast<double>(totalLatestNs);
                    }
                    if (totalAvgNs > 0)
                    {
                        pctAvg = 100.0 * static_cast<double>(nsAvg)
                            / static_cast<double>(totalAvgNs);
                    }
                }

                if (isTotal)
                {
                    std::printf("%-30.30s | %12.2f  %8s | %12.2f  %8s\n",
                        name, t_latest, "---", t_avg, "---");
                }
                else
                {
					char pctLatestLabel[16];
					char pctAvgLabel[16];
					std::snprintf(pctLatestLabel, sizeof(pctLatestLabel),
						"%.1f %%", pctLatest);
					std::snprintf(pctAvgLabel, sizeof(pctAvgLabel),
						"%.1f %%", pctAvg);

                    std::printf("%-30.30s | %12.2f  %8s | %12.2f  %8s\n",
                        name, t_latest, pctLatestLabel, t_avg, pctAvgLabel);
                }

                N_lines++;
            };

        auto PrintRow = [&](const uint32_t i, const bool isTotal)
            {
                const Bench::BenchOp op = benchOps[i];
                const auto& def = Bench::Definitions[Bench::GetIdx(op)];
                const char* name = isTotal ? "total" : def.name.data();

                PrintTimingRow(name, opLatest[i], opAvg[i], isTotal);
            };

        // regular rows
        for (uint32_t i = 0; i < benchOps.size(); i++)
        {
            if (static_cast<int>(i) == totalIdx) { continue; }
            PrintRow(i, false);
        }

        // total row last
        if (totalIdx >= 0)
        {
            const uint64_t otherLatestNs = SaturatingSubtract(totalLatestNs, accountedNsLatest);
            const uint64_t otherAvgNs = SaturatingSubtract(totalAvgNs, accountedNsAvg);

            PrintTimingRow("other", otherLatestNs, otherAvgNs, false);
			PrintCliSeparator();
            N_lines++;
            PrintRow(static_cast<uint32_t>(totalIdx), true);
        }

        // blank line separating table and progress line
        printf("\n");
        N_lines++;

        // progress line, using printf and padding to overwrite old text cleanly
        char buf[512];

        if (hasTotal)
        {
            snprintf(
                buf, sizeof(buf),
                "%02d:%02d:%02d time | %" PRIu64 "/%" PRId64 " steps (%.1f %%) | %.3f steps/s",
                elH, elM, elS,
                step,
                maxSteps,
                percent,
                stepsPerSec
            );
        }
        else
        {
            snprintf(
                buf, sizeof(buf),
                "%02d:%02d:%02d time | %" PRIu64 " steps | %.3f steps/s",
                elH, elM, elS,
                step,
                stepsPerSec
            );
        }

		std::string line(buf);
		line += FormatRamUsage(N_residentBytes, N_peakResidentBytes);
		if (line.size() < state.lastProgressLen)
		{
			line.append(state.lastProgressLen - line.size(), ' ');
		}
		state.lastProgressLen = line.size();
		state.lastProgressLine = line;

        printf("%s\n", line.c_str());
        N_lines++;

        fflush(stdout);

        // remember how many lines were printed
        state.N_lastLines = N_lines;
    }

	void DiagnosticsMgr::PrintMemorySummary_Standalone(
        const Mem::MemoryTracker& memTracker,
        Mem::ByteUnit unit,
        Base::BackendVersion backendVersion)
    {
        // pack snapshot
        Sync::MemSnapshot snap{};
        memTracker.PackMemSnapshot(snap);

        constexpr uint8_t BUCKET_COUNT = Mem::GetIdx(Mem::MemBucket::Count);

        const char* unitStr = Mem::UnitLabel(unit);

        char usedLabel[32];
        char allocLabel[32];
        std::snprintf(usedLabel, sizeof(usedLabel), "used [%s]", unitStr);
        std::snprintf(allocLabel, sizeof(allocLabel), "alloc [%s]", unitStr);

		auto PrintSpaceTable = [&](const Mem::MemSpace space,
								   const char* title,
								   const char* totalLabel)
			{
				const uint8_t spaceIdx = Mem::GetIdx(space);

				uint64_t N_totalMaxUsedBytes = 0;
				uint64_t N_totalMaxAllocBytes = 0;
				for (uint8_t i = 0; i < BUCKET_COUNT; i++)
				{
					N_totalMaxUsedBytes += snap.max[spaceIdx][i].N_usedBytes;
					N_totalMaxAllocBytes += snap.max[spaceIdx][i].N_allocBytes;
				}

				std::printf("\n");
				std::printf("%-30.30s | %12s  %8s | %12s  %8s\n",
					title, usedLabel, "% fluid", allocLabel, "% fluid");
				PrintCliSeparator();

				auto GetMemBytes = [&](const Mem::MemBucket bucket, const bool alloc)
					{
						const auto& mem = snap.max[spaceIdx][Mem::GetIdx(bucket)];
						return alloc ? mem.N_allocBytes : mem.N_usedBytes;
					};

				auto PrintMemRow = [&](const char* name,
									   const uint64_t N_maxUsedBytes,
									   const char* pctUsedLabel,
									   const uint64_t N_maxAllocBytes,
									   const char* pctAllocLabel)
					{
						const double vMaxUsed = Mem::ConvertFromBytes(unit, N_maxUsedBytes);
						const double vMaxAlloc = Mem::ConvertFromBytes(unit, N_maxAllocBytes);

						std::printf("%-30.30s | %12.2f  %8s | %12.2f  %8s\n",
							name, vMaxUsed, pctUsedLabel, vMaxAlloc, pctAllocLabel);
					};

				auto PrintParticleClassMem = [&](const char* name,
											 const Mem::MemBucket bucketF,
											 const Mem::MemBucket bucketB,
											 const Mem::MemBucket bucketR)
					{
						const uint64_t N_maxUsedBytesF = GetMemBytes(bucketF, false);
						const uint64_t N_maxUsedBytesB = GetMemBytes(bucketB, false);
						const uint64_t N_maxUsedBytesR = GetMemBytes(bucketR, false);
						const uint64_t N_maxAllocBytesF = GetMemBytes(bucketF, true);
						const uint64_t N_maxAllocBytesB = GetMemBytes(bucketB, true);
						const uint64_t N_maxAllocBytesR = GetMemBytes(bucketR, true);

						const uint64_t N_maxUsedBytes = N_maxUsedBytesF + N_maxUsedBytesB + N_maxUsedBytesR;
						const uint64_t N_maxAllocBytes = N_maxAllocBytesF + N_maxAllocBytesB + N_maxAllocBytesR;

						char pctUsedLabel[16];
						char pctAllocLabel[16];
						if (N_maxUsedBytes > 0ull)
						{
							const double pctUsed = 100.0 * static_cast<double>(N_maxUsedBytesF)
								/ static_cast<double>(N_maxUsedBytes);
							std::snprintf(pctUsedLabel, sizeof(pctUsedLabel),
								"%.1f %%", pctUsed);
						}
						else
						{
							std::snprintf(pctUsedLabel, sizeof(pctUsedLabel), "---");
						}
						if (N_maxAllocBytes > 0ull)
						{
							const double pctAlloc = 100.0 * static_cast<double>(N_maxAllocBytesF)
								/ static_cast<double>(N_maxAllocBytes);
							std::snprintf(pctAllocLabel, sizeof(pctAllocLabel),
								"%.1f %%", pctAlloc);
						}
						else
						{
							std::snprintf(pctAllocLabel, sizeof(pctAllocLabel), "---");
						}

						PrintMemRow(name, N_maxUsedBytes, pctUsedLabel, N_maxAllocBytes, pctAllocLabel);
					};

				// combine particle class buckets for concise CLI output
				PrintParticleClassMem("particle properties",
					Mem::MemBucket::ParticlePropertiesF,
					Mem::MemBucket::ParticlePropertiesB,
					Mem::MemBucket::ParticlePropertiesR);
				PrintParticleClassMem("neighbor search data struct",
					Mem::MemBucket::Neighbors_SearchDataStructF,
					Mem::MemBucket::Neighbors_SearchDataStructB,
					Mem::MemBucket::Neighbors_SearchDataStructR);
				PrintParticleClassMem("neighbor list storage",
					Mem::MemBucket::Neighbors_ListStorageF,
					Mem::MemBucket::Neighbors_ListStorageB,
					Mem::MemBucket::Neighbors_ListStorageR);
				PrintMemRow("misc",
					GetMemBytes(Mem::MemBucket::Misc, false),
					"---",
					GetMemBytes(Mem::MemBucket::Misc, true),
					"---");

				PrintCliSeparator();

				const double tMaxUsed = Mem::ConvertFromBytes(unit, N_totalMaxUsedBytes);
				const double tMaxAlloc = Mem::ConvertFromBytes(unit, N_totalMaxAllocBytes);

				char totalUsedValueLabel[32];
				char totalAllocValueLabel[32];
				std::snprintf(totalUsedValueLabel, sizeof(totalUsedValueLabel), "> %.2f", tMaxUsed);
				std::snprintf(totalAllocValueLabel, sizeof(totalAllocValueLabel), "> %.2f", tMaxAlloc);
				std::printf("%-30.30s | %12s  %8s | %12s  %8s\n",
					totalLabel, totalUsedValueLabel, "---", totalAllocValueLabel, "---");
			};

		switch (backendVersion)
		{
		case Base::BackendVersion::GPU01:
			PrintSpaceTable(Mem::MemSpace::VRAM, "memory usage (VRAM)", "total (VRAM)");
			break;
		case Base::BackendVersion::CPU01:
			[[fallthrough]];
		default:
			PrintSpaceTable(Mem::MemSpace::RAM, "memory usage (RAM)", "total (RAM)");
			break;
		}

        std::fflush(stdout);
    }

} // namespace Sim
