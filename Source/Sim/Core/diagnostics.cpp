// diagnostics.cpp

// external
#include <algorithm>
#include <cinttypes>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <spdlog/spdlog.h>
#include <string>
#ifdef _WIN32
	#include <windows.h>
#else
	#include <unistd.h>
#endif

// internal
#include "diagnostics.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Core/simulation.h"
#include "Source/Eval/Bench/benchmarks.h"



namespace Sim
{
    void DiagnosticsMgr::PrintHardwareSpecs(uint32_t N_totalThreads,
											uint32_t N_simThreads,
											uint32_t N_visThreads)
    {
        printf("\n");
        printf("--- [ hardware specifications ] ------------------------------------------\n");
        printf("--------------------------------------------------------------------------\n");

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

        printf("%-24s = %s\n", "host", hostname);
        printf("%-24s = %u\n", "threads total avail", N_totalThreads);
        printf("%-24s = %u\n", "threads used for sim", N_simThreads);
        printf("%-24s = %u\n", "threads used for vis", N_visThreads);
    }

    void DiagnosticsMgr::PrintSimParams(const SimParams& params)
    {
        printf("\n");
        printf("--- [ simulation parameters ] --------------------------------------------\n");
        printf("--------------------------------------------------------------------------\n");

        // print table with aligned columns of sim parameters
        printf("%-24s = %.6f\n", "timeStep", params.timeStep);
        printf("%-24s = %.6f\n", "particleSizeF", params.particleSizeF);
        printf("%-24s = %.6f\n", "particleSizeB", params.particleSizeB);
        printf("%-24s = %.6f\n", "particleSizeInvF", params.particleSizeInvF);
        printf("%-24s = %.6f\n", "particleSizeInvB", params.particleSizeInvB);
        printf("%-24s = %.6f\n", "kernelSuppRatio", params.kernelSuppRatio);
        printf("%-24s = %.6f\n", "kernelRadius", params.kernelRadius);
        printf("%-24s = %.6f\n", "kernelRadiusSqu", params.kernelRadiusSqu);
        printf("%-24s = %u\n", "expctdNbrCountSoftMax_FF", params.expctdNbrCountSoftMax_FF);
        printf("%-24s = %u\n", "expctdNbrCountSoftMax_FB", params.expctdNbrCountSoftMax_FB);
        printf("%-24s = %u\n", "expctdNbrCountHardMax_FF", params.expctdNbrCountHardMax_FF);
        printf("%-24s = %u\n", "expctdNbrCountHardMax_FB", params.expctdNbrCountHardMax_FB);
        printf("%-24s = %u\n", "expctdAvgNbrCountHardMax_FF", params.expctdAvgNbrCountHardMax_FF);
        printf("%-24s = %u\n", "expctdAvgNbrCountHardMax_FB", params.expctdAvgNbrCountHardMax_FB);
        printf("%-24s = %.6f\n", "restDensityF", params.restDensityF);
        printf("%-24s = %.6f\n", "restDensityB", params.restDensityB);
        printf("%-24s = %.6f\n", "massF", params.massF);
        printf("%-24s = %.6f\n", "massB", params.massB);
        printf("%-24s = %.6f\n", "viscosityF", params.viscosityF);
        printf("%-24s = %.6f\n", "viscosityB", params.viscosityB);
        printf("%-24s = (%.3f, %.3f, %.3f)\n", "gravity",
            params.gravity.x, params.gravity.y, params.gravity.z);
        printf("%-24s = %.6f\n", "pressGamma", params.pressGamma);
        printf("%-24s = %.6f\n", "jacobiOmega", params.jacobiOmega);
        printf("%-24s = %.6f\n", "densityDevTarget", params.densityDevTarget);
        printf("%-24s = %u\n", "fixedIters", params.fixedIters);
        printf("%-24s = %u\n", "minSolverIters", params.minSolverIters);
        printf("%-24s = %u\n", "maxSolverIters", params.maxSolverIters);
        printf("%-24s = %s\n", "solverTerminationCond",
            Base::ToCString(params.solverTerminationCond));
        printf("%-24s = %u\n", "dataReOrderInterval", params.dataReOrderInterval);
        printf("%-24s = %.6f\n", "gridCellSizeRatio", params.gridCellSizeRatio);
        printf("%-24s = %s\n", "gridCellIndexing",
            Base::ToCString(params.gridCellIndexing));
        printf("%-24s = %.6f\n", "jitterFracF", params.jitterFracF);
        printf("%-24s = %" PRId64 "\n", "jitterSeedF", params.jitterSeedF);
        printf("%-24s = %.6f\n", "jitterFracB", params.jitterFracB);
        printf("%-24s = %" PRId64 "\n", "jitterSeedB", params.jitterSeedB);
        printf("%-24s = %u\n", "boundaryLayers", params.boundaryLayers);
        printf("%-24s = %s\n", "exportEnabled", params.exportEnabled ? "true" : "false");
        printf("%-24s = %s\n", "exportDir", params.exportDir.c_str());
		printf("%-24s = %u\n", "exportInterval", params.exportInterval);
    }

    void DiagnosticsMgr::PrintSimSetup(const Sim::SimulationMgr& simMgr,
									   const SimParams& params)
    {
        printf("\n");
        printf("--- [ simulation setup ] -------------------------------------------------\n");
        printf("--------------------------------------------------------------------------\n");

        // modes
		printf("%-24s = %s\n", "sim backend",
			Base::ToCString(params.backendVersion));
        printf("%-24s = %s\n", "sim scenario",
            Base::ToCString(params.scenarioType));
        printf("%-24s = %s\n", "neighbor search method",
			Base::ToCString(params.neighborSearchMethod));
        printf("%-24s = %s\n", "neighbor list storage",
            Base::ToCString(params.neighborListStorage));

        // particle counts
        printf("%-24s = %u\n", "fluid particles", 
            simMgr.simBackend_->GetNumFluidParticles());
        printf("%-24s = %u\n", "boundary particles", 
            simMgr.simBackend_->GetNumBoundaryParticles());
    }

    void DiagnosticsMgr::PrintLatestPerformanceSummaryAndProgress(
        const Bench::BenchmarksTracker& benchmarksTracker,
        const std::vector<Bench::BenchOpGroupDef>& opGroups,
        Bench::TimeUnit unit,
        uint32_t step,
        int64_t maxSteps)
    {
        using steady_clock = std::chrono::steady_clock;

        // static state for progress, benchmarks, and in-place redraw
        static bool initialized = false;
        static uint32_t lastStep = 0;
        static steady_clock::time_point startSteady;

        // history for moving-average steps/s over last few samples
        using Sample = std::pair<uint32_t, steady_clock::time_point>;
        static std::deque<Sample> history;
        static int lastNumLines = 0;

        const bool hasTotal = (maxSteps > 0);
        const auto now = steady_clock::now();

        // re-initialize at start or when step == 0
        if (!initialized || step == 0)
        {
            initialized = true;
            lastStep = 0;
            startSteady = now;
            history.clear();
            lastNumLines = 0;
        }

        if (step <= lastStep) { return; }
        lastStep = step;

        // update history for moving-average steps/s
        history.emplace_back(step, now);
        constexpr uint32_t MAX_HISTORY = 10;
        if (history.size() > MAX_HISTORY) { history.pop_front(); }

        // total elapsed time since start
        double elapsedS = std::chrono::duration<double>(now - startSteady).count();
        elapsedS = std::max(elapsedS, 1e-14);

        // compute outputs on every call (no throttling)
        double stepsPerSec = 0.0;
        if (history.size() >= 2)
        {
            const auto& oldest = history.front();
            const auto& newest = history.back();
            double dt = std::chrono::duration<double>(newest.second - oldest.second).count();
            uint32_t ds = newest.first - oldest.first;

            if (dt > 1e-14 && ds > 0)
            {
                stepsPerSec = static_cast<double>(ds) / dt;
            }
        }

        // global average if window is too small / degenerate
        if (stepsPerSec <= 0.0)
        {
            stepsPerSec = static_cast<double>(step) / elapsedS;
        }

        // remaining time based on moving-average speed (only if total steps known)
        double remainingS = 0.0;
        if (hasTotal && stepsPerSec > 1e-14 && step < maxSteps)
        {
            remainingS = static_cast<double>(maxSteps - step) / stepsPerSec;
        }

        auto fmt_hms = [](double seconds, int& h, int& m, int& s)
            {
                seconds = std::max(seconds, 0.0);
                int total = static_cast<int>(seconds + 0.5);
                h = total / 3600;
                m = (total % 3600) / 60;
                s = total % 60;
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
                static_cast<double>(step) / static_cast<double>(maxSteps)));
            percent = progress * 100.0;
        }

        // move cursor up to overwrite previous block
        if (lastNumLines > 0)
        {
            std::printf("\x1b[%dA\r", lastNumLines);
        }

        int lineCount = 0;

        // per-group timing
        std::vector<uint64_t> groupLatest(opGroups.size(), 0);
        std::vector<uint64_t> groupAvg(opGroups.size(), 0);

        // sums for percentage calculation (excluding Total_Sim)
        uint64_t totalNsLatest = 0;
        uint64_t totalNsAvg = 0;
        int totalSimIdx = -1;

        for (uint32_t i = 0; i < opGroups.size(); i++)
        {
            const auto& group = opGroups[i];

            auto MinOrZero = [](uint64_t a, uint64_t b)
        	{
                if (a == 0) { return b; }
                if (b == 0) { return a; }

                return std::min(a, b);
        	};

            uint64_t groupNsLatest = 0;
            uint64_t groupNsAvg = 0;

            for (auto op : group.ops)
            {
                groupNsLatest = MinOrZero(groupNsLatest, benchmarksTracker.GetBench_Latest(op));
                groupNsAvg = MinOrZero(groupNsAvg, benchmarksTracker.GetBench_Avg_OverAll(op));
            }

            groupLatest[i] = groupNsLatest;
            groupAvg[i] = groupNsAvg;

            if (group.ops.size() == 1 && group.ops[0] == Bench::BenchOp::Total_Sim)
            {
                totalSimIdx = static_cast<int>(i);
            }
            else
            {
                totalNsLatest += groupNsLatest;
                totalNsAvg += groupNsAvg;
            }
        }

        const char* unitStr = Bench::UnitLabel(unit);

        char latestLabel[32];
        char avgLabel[32];
        snprintf(latestLabel, sizeof(latestLabel), "latest [%s]", unitStr);
        snprintf(avgLabel, sizeof(avgLabel), "avg [%s]", unitStr);

        std::printf("\n");
        lineCount++;
        // header
        printf("%-30.30s | %12s %6s | %12s %6s\n",
            "benchmark summary", latestLabel, "%", avgLabel, "%");
        lineCount++;
        printf("--------------------------------------------------------------------------\n");
        lineCount++;

        auto PrintRow = [&](uint32_t i, bool isTotalSim)
            {
                const auto& group = opGroups[i];

                const uint64_t nsLatest = groupLatest[i];
                const uint64_t nsAvg = groupAvg[i];

                const double t_latest = Bench::ConvertFromNs(unit, nsLatest);
                const double t_avg = Bench::ConvertFromNs(unit, nsAvg);

                std::string alias;
                if (isTotalSim)
                {
                    alias = "total";
                }
                else if (!group.ops.empty())
                {
                    for (std::size_t k = 0; k < group.ops.size(); ++k)
                    {
                        const auto op = group.ops[k];
                        const auto& def = Bench::Definitions[Bench::GetIdx(op)];
                        if (k > 0) { alias += " + "; }
                        alias += def.alias.data();
                    }
                }
                else
                {
                    alias = "<empty>";
                }

                double pctLatest = 0.0;
                double pctAvg = 0.0;

                if (!isTotalSim)
                {
                    if (totalNsLatest > 0)
                    {
                        pctLatest = 100.0 * static_cast<double>(nsLatest)
                            / static_cast<double>(totalNsLatest);
                    }
                    if (totalNsAvg > 0)
                    {
                        pctAvg = 100.0 * static_cast<double>(nsAvg)
                            / static_cast<double>(totalNsAvg);
                    }
                }

                if (isTotalSim)
                {
                    std::printf("%-30.30s | %12.2f %6s | %12.2f %6s\n",
                        alias.c_str(), t_latest, "---", t_avg, "---");
                }
                else
                {
                    std::printf("%-30.30s | %12.2f %6.1f | %12.2f %6.1f\n",
                        alias.c_str(), t_latest, pctLatest, t_avg, pctAvg);
                }

                lineCount++;
            };

        // regular rows
        for (uint32_t i = 0; i < opGroups.size(); i++)
        {
            if (static_cast<int>(i) == totalSimIdx) { continue; }
            PrintRow(i, false);
        }

        // total simulation row last
        if (totalSimIdx >= 0)
        {
            printf("--------------------------------------------------------------------------\n");
            lineCount++;
            PrintRow(static_cast<std::size_t>(totalSimIdx), true);
        }

        // blank line separating table and progress line
        printf("\n");
        lineCount++;

        // progress line, using printf and padding to overwrite old text cleanly
        static std::size_t lastProgressLen = 0;
        char buf[512];

        if (hasTotal)
        {
            snprintf(
                buf, sizeof(buf),
                "%02d:%02d:%02d elapsed | %lld/%lld steps (%.1f %%) | %.3f steps/s (recently)",
                elH, elM, elS,
                static_cast<long long>(step),
                static_cast<long long>(maxSteps),
                percent,
                stepsPerSec
            );
        }
        else
        {
            snprintf(
                buf, sizeof(buf),
                "%02d:%02d:%02d elapsed | %lld steps | %.3f steps/s (recently)",
                elH, elM, elS,
                static_cast<long long>(step),
                stepsPerSec
            );
        }

        std::string line(buf);
        if (line.size() < lastProgressLen)
        {
            line.append(lastProgressLen - line.size(), ' ');
        }
        lastProgressLen = line.size();

        printf("%s\n", line.c_str());
        lineCount++;

        fflush(stdout);

        // remember how many lines were printed
        lastNumLines = lineCount;
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

        uint64_t totalMaxUsedBytes = 0;
        uint64_t totalMaxAllocBytes = 0;
        for (uint8_t i = 0; i < BUCKET_COUNT; i++)
        {
            totalMaxUsedBytes += snap.max[i].usedBytes;
            totalMaxAllocBytes += snap.max[i].allocBytes;
        }

        const char* unitStr = Mem::UnitLabel(unit);

        char maxUsedLabel[32];
        char maxAllocLabel[32];
        std::snprintf(maxUsedLabel, sizeof(maxUsedLabel), "max used [%s]", unitStr);
        std::snprintf(maxAllocLabel, sizeof(maxAllocLabel), "max allocated [%s]", unitStr);

        std::printf("\n");
        std::printf("%-30.30s | %19s | %19s\n",
            "memory usage", maxUsedLabel, maxAllocLabel);
        std::printf("--------------------------------------------------------------------------\n");

        for (uint8_t i = 0; i < BUCKET_COUNT; i++)
        {
            const auto& def = Mem::Definitions[i];
            const double vMaxUsed = Mem::ConvertFromBytes(unit, snap.max[i].usedBytes);
            const double vMaxAlloc = Mem::ConvertFromBytes(unit, snap.max[i].allocBytes);

            std::printf("%-30.30s | %19.2f | %19.2f\n",
                def.name.data(), vMaxUsed, vMaxAlloc);
        }

        std::printf("--------------------------------------------------------------------------\n");

        const double tMaxUsed = Mem::ConvertFromBytes(unit, totalMaxUsedBytes);
        const double tMaxAlloc = Mem::ConvertFromBytes(unit, totalMaxAllocBytes);

		const char* totalLabel = "total";
		switch (backendVersion)
		{
		case Base::BackendVersion::CPU01:
			totalLabel = "total (RAM)";
			break;
		case Base::BackendVersion::GPU01:
			totalLabel = "total (VRAM)";
			break;
		default:
			break;
		}

        char totalUsedValueLabel[32];
        char totalAllocValueLabel[32];
        std::snprintf(totalUsedValueLabel, sizeof(totalUsedValueLabel), "> %.2f", tMaxUsed);
        std::snprintf(totalAllocValueLabel, sizeof(totalAllocValueLabel), "> %.2f", tMaxAlloc);
        std::printf("%-30.30s | %19s | %19s\n",
            totalLabel, totalUsedValueLabel, totalAllocValueLabel);

        std::fflush(stdout);
    }

	void DiagnosticsMgr::PrintStatisticsSummary_Standalone(
		const Stats::StatisticsTracker& statisticsTracker,
		const SimParams& params)
	{
		if (!params.statGroupsEnabled[Stats::GetIdx(Stats::StatGroup::Particles)]) { return; }
		if (statisticsTracker.GetHistory().empty()) { return; }

		const float negNbrCandFF_avg = statisticsTracker.GetStat_Avg(
			Stats::StatMetric::NegNbrCandFF_Pct);

		std::printf("\n");
		std::printf("%-30.30s | %19s\n",
			"statistics summary", "avg [%]");
		std::printf("----------------------------------------------------\n");
		std::printf("%-30.30s | %19.2f\n",
			"neg fluid candidates", negNbrCandFF_avg * 100.0f);

		std::fflush(stdout);
	}
	
} // namespace Sim
