// main_headless.cpp

// external
#include <cstdlib>
#include <exception>
#include <spdlog/spdlog.h>
#include <tbb/global_control.h>

// internal
#include "apps/app_runtime.h"
#include "source/eval/serial/serialization.h"
#include "source/simulation/core/diagnostics.h"
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/core/simulation.h"



int main(int argc, char** argv)
{
	App::PrintPlatformBanner();
	App::SetupLogging();

	try // application setup
	{
		SPDLOG_INFO("Launching TSP (headless application)...");

		auto parse = App::ParseMainOptions(argc, argv, "TSP Headless");
		if (parse.shouldExit) { return parse.exitCode; }

		std::string profile = parse.options.profile;
		const auto& simOverrides = parse.options.simOverrides;

        // --- [ initialization of manager systems ] -----------------------------------------------

		// config profile and simulation parameters
		Sim::SimParamsMgr simParamsMgr{};
		if (!App::LoadSimParams(simParamsMgr, profile, simOverrides)) { return EXIT_FAILURE; }

        // --- [ hardware budgets ] ----------------------------------------------------------------

		const App::ThreadBudget threadBudget = App::ResolveThreadBudget(
			simParamsMgr.SnapshotParams().first.maxThreadsSim, 0);

		Sim::DiagnosticsMgr::PrintBuildConfig();
		Sim::DiagnosticsMgr::PrintHardwareSpecs(
			threadBudget.N_totalThreads,
			threadBudget.N_simThreads,
			threadBudget.N_visThreads);

		// limit TBB to the simulation thread budget
		tbb::global_control gc(tbb::global_control::max_allowed_parallelism, threadBudget.N_simThreads);

		// sub-systems
		Scene::ScenarioMgr scenarioMgr{};
		Bench::BenchmarksTracker benchmarksTracker{ simParamsMgr.SnapshotParams().first };
		Mem::MemoryTracker memoryTracker{ simParamsMgr.SnapshotParams().first };
		Stats::StatisticsTracker statisticsTracker{ simParamsMgr.SnapshotParams().first };
		Export::SerializationMgr serializationMgr{};

		// main systems
		Sim::SimulationMgr simulationMgr{
			simParamsMgr,
			scenarioMgr,
			benchmarksTracker,
			memoryTracker,
			statisticsTracker,
			serializationMgr };
		simulationMgr.SetConfigProfile(profile);

		// bench mailbox
		Sync::BenchSnapMailbox benchMailbox{};
		simulationMgr.SetBenchMailbox(&benchMailbox);

		// memory mailbox
		Sync::MemSnapMailbox memMailbox{};
		simulationMgr.SetMemMailbox(&memMailbox);

		// stat mailbox
		Sync::StatSnapMailbox statMailbox{};
		simulationMgr.SetStatMailbox(&statMailbox);

		simulationMgr.Init();
		simParamsMgr.Set_SimRunning(true);

		// --- [ main loop ] -----------------------------------------------------------------------

		App::AppStopWatcher stopWatcher{};
		stopWatcher.Start([&]()
			{
				simulationMgr.RequestStop();
			});

		// simulation runs on the configured simulation thread budget
		simulationMgr.Launch_Blocking();
		stopWatcher.Stop();

		// --- [ shutdown ] ------------------------------------------------------------------------

		simulationMgr.Shutdown();
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("Fatal error: {}", ex.what());
		return EXIT_FAILURE;
	}
	catch (...)
	{
		SPDLOG_ERROR("Fatal error: Unknown exception.");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
