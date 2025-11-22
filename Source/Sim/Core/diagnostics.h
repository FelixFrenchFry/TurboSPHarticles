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
	void PrintHardwareSpecs(uint32_t N_totalThreads,
						    uint32_t N_simThreads, 
						    uint32_t N_visThreads);

	void PrintSimParams(const SimParams& params);

	void PrintSetupSummary(const SimulationMgr& simMgr);

	void PrintProgressMilestone(int64_t step, int64_t totalSteps = -1);

	void PrintPerformanceSummary(int64_t totalSteps,
								 double totalSimTime,
								 double wallSeconds);

	// TODO: print big performance and statistics tables in-place after every step?

	// TODO: print current step and steps/sec periodically?

} // namespace Sim
