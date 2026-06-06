// main_distributed.cpp

// external
#include <cstdlib>
#include <exception>
#include <spdlog/spdlog.h>

// internal
#include "Apps/app_runtime.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Dist/distributed_simulation.h"
#include "Source/Sim/Dist/mpi_context.h"



int main(int argc, char** argv)
{
	App::SetupLogging();

	try // application setup
	{
		Dist::MpiContext mpiContext{ argc, argv };

		if (mpiContext.IsRoot())
		{
			App::PrintPlatformBanner();
			SPDLOG_INFO("Launching TSP (distributed MPI application)...");
			SPDLOG_INFO("MPI world size: {}, thread support: {}",
				mpiContext.Size(),
				Dist::MpiContext::ThreadSupportName(mpiContext.ThreadSupport()));
		}

		auto parse = App::ParseMainOptions(
			argc, argv, "TSP Distributed", mpiContext.IsRoot());
		if (parse.shouldExit) { return parse.exitCode; }

		std::string profile = parse.options.profile;
		const auto& simOverrides = parse.options.simOverrides;

		// --- [ initialization of manager systems ] ------------------------------------------------

		Sim::SimParamsMgr simParamsMgr{};
		Dist::DistributedSimulation distributedSimulation{
			mpiContext,
			simParamsMgr };

		if (!distributedSimulation.Init(profile, simOverrides))
		{
			if (mpiContext.IsRoot())
			{
				SPDLOG_ERROR("Failed to initialize DistributedSimulation");
			}
			return EXIT_FAILURE;
		}

		// --- [ main loop ] ------------------------------------------------------------------------

		distributedSimulation.RunStartupCheck();

		// --- [ shutdown ] -------------------------------------------------------------------------

		distributedSimulation.Shutdown();
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
