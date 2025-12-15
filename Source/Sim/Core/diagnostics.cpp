// diagnostics.cpp

// external
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <deque>
#include <spdlog/spdlog.h>
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
        printf("--- [ hardware specifications ] ------------------------------------------------\n");
        printf("--------------------------------------------------------------------------------\n");

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

        printf("\n");
    }

    void DiagnosticsMgr::PrintSimParams(const SimParams& params)
    {
        printf("--- [ simulation parameters ] --------------------------------------------------\n");
        printf("--------------------------------------------------------------------------------\n");

        // print table with aligned columns of sim parameters
        printf("%-24s = %.6f\n", "timeStep", params.timeStep);
        printf("%-24s = %.6f\n", "particleSizeF", params.particleSizeF);
        printf("%-24s = %.6f\n", "particleSizeB", params.particleSizeB);
        printf("%-24s = %.6f\n", "restDensityF", params.restDensityF);
        printf("%-24s = %.6f\n", "restDensityB", params.restDensityB);
        printf("%-24s = %.6f\n", "massF", params.massF);
        printf("%-24s = %.6f\n", "massB", params.massB);
        printf("%-24s = %.6f\n", "viscosityF", params.viscosityF);
        printf("%-24s = %.6f\n", "viscosityB", params.viscosityB);
        printf("%-24s = %.6f\n", "gammaPress", params.gammaPress);
        printf("%-24s = %.6f\n", "jacobiOmega", params.jacobiOmega);
        printf("%-24s = %.6f\n", "kernelSuppRatio", params.kernelSuppRatio);
        printf("%-24s = %.6f\n", "kernelRadius", params.kernelRadius);
        printf("%-24s = %.6f\n", "jitterFracF", params.jitterFracF);
        printf("%-24s = %.6f\n", "jitterFracB", params.jitterFracB);
        printf("%-24s = %u\n", "boundaryLayers", params.boundaryLayers);
        printf("%-24s = (%.3f, %.3f, %.3f)\n", "gravity",
            params.gravity.x, params.gravity.y, params.gravity.z);

        printf("\n");
    }

    void DiagnosticsMgr::PrintSimSetup(const Sim::SimulationMgr& simMgr)
    {
        printf("--- [ simulation setup ] -------------------------------------------------------\n");
        printf("--------------------------------------------------------------------------------\n");

        const auto& backend = simMgr.simBackend_;
        const auto& scenario = simMgr.simScenario_;

        // backend
		std::string backendName = "<unknown>";
		if (backend)
		{
			backendName = simMgr.simBackend_->Name();
		}
        printf("%-24s = %s\n", "sim backend", backendName.c_str());

        // scenario
        std::string scenarioName = "<unknown>";
        if (scenario)
        {
            scenarioName = simMgr.simScenario_->Name();
        }
        printf("%-24s = %s\n", "sim scenario", scenarioName.c_str());

        // particle counts
        printf("%-24s = %u\n", "fluid particles", 
            simMgr.universalSimState_.fluidParticles.N_);
        printf("%-24s = %u\n", "boundary particles", 
            simMgr.universalSimState_.boundaryParticles.N_);

        printf("\n");
    }

    void DiagnosticsMgr::PrintMemoryUsage()
    {
        printf("--- [ memory usage ] -----------------------------------------------------------\n");
        printf("--------------------------------------------------------------------------------\n");

        // TODO: implement
    }

    void DiagnosticsMgr::PrintLatestPerformanceSummaryAndProgress(
        const Bench::BenchmarksMgr& benchMgr,
        const std::vector<Bench::BenchOpGroupDef>& opGroups,
        Bench::TimeUnit unit,
        uint32_t step,
        uint32_t totalSteps)
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

        const bool hasTotal = (totalSteps > 0);
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
        elapsedS = std::max(elapsedS, 1e-9);

        // compute outputs on every call (no throttling)
        double stepsPerSec = 0.0;
        if (history.size() >= 2)
        {
            const auto& oldest = history.front();
            const auto& newest = history.back();
            double dt = std::chrono::duration<double>(newest.second - oldest.second).count();
            uint32_t ds = newest.first - oldest.first;

            if (dt > 1e-6 && ds > 0)
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
        if (hasTotal && stepsPerSec > 1e-9 && step < totalSteps)
        {
            remainingS = static_cast<double>(totalSteps - step) / stepsPerSec;
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
                static_cast<double>(step) / static_cast<double>(totalSteps)));
            percent = progress * 100.0;
        }

        // move cursor up to overwrite previous block
        if (lastNumLines > 0)
        {
            std::printf("\x1b[%dA\r", lastNumLines);
        }

        int lineCount = 0;

        std::printf("--- [ benchmark summary ] ------------------------------------------------------\n");
        lineCount++;
        std::printf("--------------------------------------------------------------------------------\n");
        lineCount++;

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

            uint64_t groupNsLatest = 0;
            uint64_t groupNsAvg = 0;

            for (auto op : group.ops)
            {
                groupNsLatest += benchMgr.GetBench_Latest(op);
                groupNsAvg += benchMgr.GetBench_Avg_OverAll(op);
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

        const char* unitStr = Bench::BenchmarksMgr::UnitLabel(unit);

        char latestLabel[32];
        char avgLabel[32];
        std::snprintf(latestLabel, sizeof(latestLabel), "latest [%s]", unitStr);
        std::snprintf(avgLabel, sizeof(avgLabel), "avg [%s]", unitStr);

        // header: 30 (name) + latest pair + gap + avg pair
        std::printf("%-30.30s | %12s  %6s    | %12s  %6s\n",
            "operation group", latestLabel, "%", avgLabel, "%");
        lineCount++;
        std::printf("--------------------------------------------------------------------------------\n");
        lineCount++;

        auto PrintRow = [&](uint32_t i, bool isTotalSim)
            {
                const auto& group = opGroups[i];

                const uint64_t nsLatest = groupLatest[i];
                const uint64_t nsAvg = groupAvg[i];

                const double t_latest = Bench::BenchmarksMgr::ConvertFromNs(unit, nsLatest);
                const double t_avg = Bench::BenchmarksMgr::ConvertFromNs(unit, nsAvg);

                std::string alias;
                if (isTotalSim)
                {
                    alias = "total simulation time";
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
                    std::printf("%-30.30s | %12.2f  %6s    | %12.2f  %6s\n",
                        alias.c_str(), t_latest, "---", t_avg, "---");
                }
                else
                {
                    std::printf("%-30.30s | %12.2f  %6.1f    | %12.2f  %6.1f\n",
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
            std::printf("--------------------------------------------------------------------------------\n");
            lineCount++;
            PrintRow(static_cast<std::size_t>(totalSimIdx), true);
        }

        // blank line separating table and progress line
        std::printf("\n");
        lineCount++;

        // progress line, using printf and padding to overwrite old text cleanly
        static std::size_t lastProgressLen = 0;
        char buf[512];

        if (hasTotal)
        {
            std::snprintf(
                buf, sizeof(buf),
                "%02d:%02d:%02d elapsed | %lld/%lld steps (%.1f %%) | %.3f steps/s (recently)",
                elH, elM, elS,
                static_cast<long long>(step),
                static_cast<long long>(totalSteps),
                percent,
                stepsPerSec
            );
        }
        else
        {
            std::snprintf(
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

        std::printf("%s\n", line.c_str());
        lineCount++;

        std::fflush(stdout);

        // remember how many lines were printed
        lastNumLines = lineCount;
    }

    void DiagnosticsMgr::PrintAveragePerformanceSummary(
        const Bench::BenchmarksMgr& benchMgr,
        const std::vector<Bench::BenchOpGroupDef>& opGroups,
        Bench::TimeUnit unit,
        uint32_t step)
    {
        // per-group timing
        std::vector<uint64_t> groupTimings(opGroups.size(), 0);

        // sum of all groups except Total_Sim (for percentages)
        uint64_t totalNs = 0;
        int totalSimIdx = -1;

        for (uint32_t i = 0; i < opGroups.size(); i++)
        {
            const auto& group = opGroups[i];

            uint64_t groupNs = 0;
            for (auto op : group.ops)
            {
                groupNs += benchMgr.GetBench_Avg_OverAll(op);
            }
            groupTimings[i] = groupNs;

            if (group.ops.size() == 1 && group.ops[0] == Bench::BenchOp::Total_Sim)
            {
                totalSimIdx = static_cast<int>(i);
            }
            else
            {
                totalNs += groupNs;
            }
        }

        const char* unitStr = Bench::BenchmarksMgr::UnitLabel(unit);

        std::printf("--- [ benchmark summary (AVERAGE measurement over all executed steps) ----------\n");
        std::printf("--------------------------------------------------------------------------------\n");

        // header (same widths as rows: 30 / 10 / 5)
        std::printf("%-30.30s  %10s  %5s\n", "operation group", unitStr, "%");
        std::printf("--------------------------------------------------------------------------------\n");

        auto PrintRow = [&](uint32_t i, bool isTotalSim)
            {
                const auto& group = opGroups[i];
                const uint64_t ns = groupTimings[i];
                const double t_unit = Bench::BenchmarksMgr::ConvertFromNs(unit, ns);

                std::string alias;
                if (isTotalSim)
                {
                    alias = "total simulation time";
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

                double pct = 0.0;
                if (!isTotalSim && totalNs > 0)
                {
                    pct = 100.0 * static_cast<double>(ns) / static_cast<double>(totalNs);
                }

                if (isTotalSim)
                {
                    std::printf("%-30.30s  %10.2f  %5s  (avg over %lld steps)\n",
                        alias.c_str(), t_unit, "--",
                        static_cast<long long>(step));
                }
                else
                {
                    std::printf("%-30.30s  %10.2f  %5.1f\n",
                        alias.c_str(), t_unit, pct);
                }
            };

        // regular rows
        for (uint32_t i = 0; i < opGroups.size(); ++i)
        {
            if (static_cast<int>(i) == totalSimIdx) { continue; }
            PrintRow(i, false);
        }

        // total simulation row last
        if (totalSimIdx >= 0)
        {
            std::printf("--------------------------------------------------------------------------------\n");
            PrintRow(static_cast<std::size_t>(totalSimIdx), true);
        }

        std::printf("--------------------------------------------------------------------------------\n");
    }
	
} // namespace Sim
