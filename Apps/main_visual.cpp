// main_visual.cpp

// external
#include <CLI/CLI.hpp>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <spdlog/spdlog.h>
#include <tbb/global_control.h>

// internal
#include "Source/Sim/Core/diagnostics.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Core/simulation.h"
#include "Source/Vis/Core/parameters_vis.h"
#include "Source/Vis/Core/visualization.h"



int main(int argc, char** argv)
{
	#if defined(_WIN32)
	printf("Running on Windows\n");
	#elif defined(__linux__)
	printf("Running on Linux\n");
	#endif

	// [year-month-day 12:34:56.789] [file.cpp:line] [type] message
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%s:%#] [%^%l%$] %v");
	spdlog::flush_on(spdlog::level::err);

	try // application setup
	{
		SPDLOG_INFO("Launching TSP (visual application)...");

		CLI::App application{ "TSP Visual" };
		std::string configFile; // TODO: load default config file?

		// TODO: make this required or use default config file, if this is not provided?
		application.add_option("--config", configFile, "Configuration file");
		// TODO: add --version, --threads, --backend, etc... ?

		CLI11_PARSE(application, argc, argv);

		// --- [ hardware budgets ] --------------------------------------------

		const std::uint32_t N_totalThreads = std::max<uint32_t>(1, std::thread::hardware_concurrency());
		const std::uint32_t N_visThreads = 1; // TODO: make configurable?
		const std::uint32_t N_simThreads = N_totalThreads > N_visThreads ?
			N_totalThreads - N_visThreads : 1;

		Sim::DiagnosticsMgr::PrintHardwareSpecs(N_totalThreads, N_simThreads, N_visThreads);

		// limit TBB to N_simThreads worker threads
		tbb::global_control gc(tbb::global_control::max_allowed_parallelism, N_simThreads);

		// --- [ initialization of manager systems ] ---------------------------

		// simulation parameters
		Sim::SimParamsMgr simParamsMgr{};
		simParamsMgr.PresetParams_B();
		simParamsMgr.AssertParamDerivationCorrectness();

		// sub-systems
		Vis::VisParamsMgr visParamsMgr{};
		Scene::ScenarioMgr scenarioMgr{};
		Bench::BenchmarksMgr benchmarksMgr{ simParamsMgr.SnapshotParams().first };
		Stats::StatisticsMgr statisticsMgr{ simParamsMgr.SnapshotParams().first };

		// main systems
		Sim::SimulationMgr simulationMgr{ simParamsMgr, scenarioMgr, benchmarksMgr, statisticsMgr };
		Vis::VisualizationMgr visualizationMgr{ simulationMgr, benchmarksMgr, simParamsMgr, visParamsMgr };

		// visual mailbox
		Sync::VisualSnapMailbox visualMailbox{};
		simulationMgr.SetVisualMailbox(&visualMailbox);
		visualizationMgr.SetVisualMailbox(&visualMailbox);

		// bench mailbox
		Sync::BenchSnapMailbox benchMailbox{};
		simulationMgr.SetBenchMailbox(&benchMailbox);
		visualizationMgr.SetBenchMailbox(&benchMailbox);

		// stat mailbox
		Sync::StatSnapMailbox statMailbox{};
		simulationMgr.SetStatMailbox(&statMailbox);
		visualizationMgr.SetStatMailbox(&statMailbox);

		if (!simulationMgr.Init())
		{
			SPDLOG_ERROR("Failed to initialize SimulationMgr");
			return EXIT_FAILURE;
		}

		auto tempVisParams = visParamsMgr.SnapshotParams().first;
		if (!visualizationMgr.Init(tempVisParams.window))
		{
			SPDLOG_ERROR("Failed to initialize VisualizationMgr");
			return EXIT_FAILURE;
		}

		// --- [ main loop ] ---------------------------------------------------

		// simulation runs on background controller thread
		// (spawn N-1 worker threads within)
		simulationMgr.Launch_Async();

		// visualization runs on main thread
		visualizationMgr.RunVis();

		// --- [ shutdown ] ----------------------------------------------------

		simulationMgr.RequestStop();
		simulationMgr.JoinThreads();
		visualizationMgr.Shutdown();
		simulationMgr.Shutdown();
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("Exception: {}", ex.what());
		return EXIT_FAILURE;
	}
	catch (...)
	{
		SPDLOG_ERROR("Exception: UNKNOWN");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
