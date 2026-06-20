// main_distributed.cpp

// external
#include <cstdlib>
#include <exception>
#include <spdlog/spdlog.h>
#include <tbb/global_control.h>

// internal
#include "Apps/app_runtime.h"
#include "Source/Sim/Core/diagnostics.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Dist/dist_helpers.h"
#include "Source/Sim/Dist/dist_simulation.h"
#include "Source/Sim/Dist/mpi_context.h"



int main(int argc, char** argv)
{
	App::SetupLogging();

	try // application setup
	{
		// init of MPI context
		Dist::MpiContext mpiContext{ argc, argv };

		if (mpiContext.IsRoot())
		{
			App::PrintPlatformBanner();
		}
		SPDLOG_INFO_MPI("Launching TSP (distributed application)...", mpiContext);
		SPDLOG_INFO_MPI("MPI world size: {}, thread support: {}", mpiContext,
			mpiContext.Size(), Dist::MpiContext::ThreadSupportName(mpiContext.ThreadSupport()));

		auto parse = App::ParseMainOptions(argc, argv, "TSP Distributed", mpiContext.IsRoot());
		if (parse.shouldExit) { return parse.exitCode; }

		std::string profile = parse.options.profile;
		const auto& simOverrides = parse.options.simOverrides;

		// --- [ initialization of manager systems ] -----------------------------------------------

		Sim::SimParamsMgr simParamsMgr{};
		if (!App::LoadSimParams(simParamsMgr, profile, simOverrides)) { return EXIT_FAILURE; }

		// --- [ hardware budgets ] ----------------------------------------------------------------

		const App::ThreadBudget threadBudget = App::ResolveThreadBudget(
			simParamsMgr.SnapshotParams().first.maxThreadsSim, 0);

		if (mpiContext.IsRoot())
		{
			Sim::DiagnosticsMgr::PrintHardwareSpecs(
				threadBudget.N_totalThreads,
				threadBudget.N_simThreads,
				threadBudget.N_visThreads);
		}

		// limit TBB to the simulation thread budget on each MPI rank
		tbb::global_control gc(tbb::global_control::max_allowed_parallelism, threadBudget.N_simThreads);

		// main systems
		Dist::DistSimulationMgr distSimulationMgr{
			mpiContext,
			simParamsMgr};

		if (!distSimulationMgr.Init(profile, simOverrides))
		{
			SPDLOG_ERROR_MPI("Failed to initialize DistSimulationMgr", mpiContext);
			return EXIT_FAILURE;
		}
		if (mpiContext.IsRoot())
		{
			Sim::DiagnosticsMgr::PrintBuildConfig();
		}

		simParamsMgr.Set_SimRunning(true);

		// --- [ main loop ] -----------------------------------------------------------------------

		distSimulationMgr.RunStartupCheck();

		App::AppStopWatcher stopWatcher{};
		stopWatcher.Start([&]()
			{
				distSimulationMgr.RequestStop();
			});

		// simulation runs on the configured simulation thread budget
		distSimulationMgr.RunSim_Blocking();
		stopWatcher.Stop();

        // --- [ shutdown ] ------------------------------------------------------------------------

		distSimulationMgr.Shutdown();
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
