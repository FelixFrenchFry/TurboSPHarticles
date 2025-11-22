// diagnostics.cpp

// external
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <spdlog/spdlog.h>
#include <sstream>
#ifdef _WIN32
	#include <windows.h>
#else
	#include <unistd.h>
#endif

// internal
#include "diagnostics.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Core/simulation.h"



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
        printf("%-24s = %d\n", "boundaryLayers", params.boundaryLayers);
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

    void DiagnosticsMgr::PrintProgressMilestone(int64_t step, int64_t totalSteps)
    {
        using steady_clock = std::chrono::steady_clock;

        // static state to persist across calls
        static bool initialized = false;
        static int64_t lastStep = 0;
        static steady_clock::time_point startSteady;

        // history for moving-average steps/s over the last ~100 steps
        using Sample = std::pair<int64_t, steady_clock::time_point>;
        static std::deque<Sample> history;

        // last time when elapsed/remaining/stepsPerSec was re-computed
        static steady_clock::time_point lastUpdateSteady;

        // cached derived values (updated at most once per second)
        static double cachedElapsedS = 0.0;
        static double cachedRemainingS = 0.0;
        static double cachedStepsPerSec = 0.0;

        const bool hasTotal = (totalSteps > 0);

        // re-initialize on first call or when step == 0
        if (!initialized || step == 0)
        {
            initialized = true;
            lastStep = 0;
            startSteady = steady_clock::now();
            lastUpdateSteady = startSteady;
            history.clear();
            cachedElapsedS = 0.0;
            cachedRemainingS = 0.0;
            cachedStepsPerSec = 0.0;
        }

        if (step <= lastStep) { return; }
        lastStep = step;

        auto nowSteady = steady_clock::now();

        // update history for moving average (last up to 100 steps)
        history.emplace_back(step, nowSteady);
        constexpr std::size_t MAX_HISTORY = 10;
        if (history.size() > MAX_HISTORY) { history.pop_front(); }

        // total elapsed time since start
        double elapsedS = std::chrono::duration<double>(nowSteady - startSteady).count();
        elapsedS = std::max(elapsedS, 1e-9); // avoid div-by-zero

        const bool isLastStep = (hasTotal && step >= totalSteps);
        const auto UPDATE_INTERVAL = std::chrono::seconds(1);

        bool doUpdate = false;
        if (isLastStep)
        {
            // force update for final line
            doUpdate = true;
        }
        else if ((nowSteady - lastUpdateSteady) >= UPDATE_INTERVAL)
        {
            // refresh once per second
            doUpdate = true;
        }

        if (doUpdate)
        {
            lastUpdateSteady = nowSteady;

            // compute moving-average steps/s over history window
            double stepsPerSec = 0.0;
            if (history.size() >= 2)
            {
                const auto& oldest = history.front();
                const auto& newest = history.back();
                double   dt = std::chrono::duration<double>(newest.second - oldest.second).count();
                int64_t  ds = newest.first - oldest.first;

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

            cachedElapsedS = elapsedS;
            cachedRemainingS = remainingS;
            cachedStepsPerSec = stepsPerSec;
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
        fmt_hms(cachedElapsedS, elH, elM, elS);
        if (hasTotal)
        {
            fmt_hms(cachedRemainingS, etaH, etaM, etaS);
        }

        // progress percentage (only if total steps known)
        double percent = 0.0;
        if (hasTotal)
        {
            double progress = std::min(1.0, std::max(0.0,
                static_cast<double>(step) / static_cast<double>(totalSteps)));
            percent = progress * 100.0;
        }

		// overwrite of previous line
        std::ostringstream oss;
        oss << '\r';

        // elapsed time
        oss << std::setfill('0') << std::setw(2) << elH << ':'
            << std::setw(2) << elM << ':'
            << std::setw(2) << elS << " elapsed";

        if (hasTotal)
        {
            // remaining time
            oss << " | "
        		<< std::setw(2) << etaH << ':'
                << std::setw(2) << etaM << ':'
                << std::setw(2) << etaS << " remaining";

            // steps so far and percentage
            oss << " | "
                << step << '/' << totalSteps << " steps ("
                << std::fixed << std::setprecision(1)
                << percent << " %)";
        }
        else // total steps unknown
        {
            // steps so far
            oss << " | " << step << " steps";
        }

        // steps per second (moving average)
        oss << " | "
            << std::fixed << std::setprecision(2)
            << cachedStepsPerSec << " steps/s";

        std::cout << oss.str() << std::flush;

        if (isLastStep) { std::cout << std::endl; }
    }

    void DiagnosticsMgr::PrintPerformanceSummary(int64_t totalSteps,
												 double totalSimTime,
												 double wallSeconds)
    {
        // TODO: implement

        return;
    }
	
} // namespace Sim
