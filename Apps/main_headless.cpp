// main_headless.cpp

// external
#include <CLI/CLI.hpp>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <spdlog/spdlog.h>
#include <tbb/global_control.h>

// internal
#include "Source/Eval/Serial/serialization.h"
#include "Source/Sim/Core/diagnostics.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Core/simulation.h"



int main(int argc, char** argv)
{
	#if defined(_WIN32)
	printf("Running on Windows\n");
	#elif defined(__linux__)
	printf("Running on Linux\n");
	#endif

	// [year-month-day 12:34:56.789] [file.cpp:line] [type] message
	/*
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%s:%#] [%^%l%$] %v");
	spdlog::flush_on(spdlog::level::err);
	*/

	// [year-month-day 12:34:56.789] [type] message
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
	spdlog::flush_on(spdlog::level::err);

	try // application setup
	{
		SPDLOG_INFO("Launching TSP (headless application)...");

		CLI::App application{ "TSP Headless" };
		std::string configFile; // TODO: load default config file?

		// TODO: make this required or use default config file, if this is not provided?
		application.add_option("--config", configFile, "Configuration file");
		// TODO: add --version, --threads, --backend, etc... ?

		CLI11_PARSE(application, argc, argv);

		// --- [ hardware budgets ] --------------------------------------------

		const std::uint32_t N_totalThreads = std::max<uint32_t>(1, std::thread::hardware_concurrency());

		Sim::DiagnosticsMgr::PrintHardwareSpecs(N_totalThreads, N_totalThreads, 0);

		// limit TBB to N_simThreads worker threads
		tbb::global_control gc(tbb::global_control::max_allowed_parallelism, N_totalThreads);

		// --- [ initialization of manager systems ] ---------------------------

		// simulation parameters
		Sim::SimParamsMgr simParamsMgr{};
		// TODO: make adjustable via command line argument?
		simParamsMgr.LoadSimParamsFromJsonFile("SimParams_B.json");

		// sub-systems
		Scene::ScenarioMgr scenarioMgr{};
		Bench::BenchmarksMgr benchmarksMgr{ simParamsMgr.SnapshotParams().first };
		Stats::StatisticsMgr statisticsMgr{ simParamsMgr.SnapshotParams().first };
		Export::SerializationMgr serializationMgr{};

		// main systems
		Sim::SimulationMgr simulationMgr{ simParamsMgr, scenarioMgr, benchmarksMgr, statisticsMgr, serializationMgr };

		// bench mailbox
		Sync::BenchSnapMailbox benchMailbox{};
		simulationMgr.SetBenchMailbox(&benchMailbox);

		// stat mailbox
		Sync::StatSnapMailbox statMailbox{};
		simulationMgr.SetStatMailbox(&statMailbox);

		if (!simulationMgr.Init())
		{
			SPDLOG_ERROR("Failed to initialize SimulationMgr");
			return EXIT_FAILURE;
		}
		simParamsMgr.Set_SimRunning(true);

		// --- [ main loop ] ---------------------------------------------------

		// simulation runs on all available threads
		simulationMgr.Launch_Blocking(simParamsMgr.SnapshotParams().first.maxSteps);

		// --- [ shutdown ] ----------------------------------------------------

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
