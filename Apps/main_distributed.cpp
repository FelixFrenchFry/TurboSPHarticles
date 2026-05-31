// main_distributed.cpp

// external
#include <CLI/CLI.hpp>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

// internal
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Dist/distributed_simulation.h"
#include "Source/Sim/Dist/mpi_context.h"



int main(int argc, char** argv)
{
	// [year-month-day 12:34:56.789] [type] message
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
	spdlog::flush_on(spdlog::level::err);

	try // application setup
	{
		Dist::MpiContext mpiContext{ argc, argv };

		if (mpiContext.IsRoot())
		{
			#if defined(_WIN32)
			printf("(Running on Windows)\n");
			#elif defined(__linux__)
			printf("(Running on Linux)\n");
			#endif

			SPDLOG_INFO("Launching TSP (distributed MPI application)...");
			SPDLOG_INFO("MPI world size: {}, thread support: {}",
				mpiContext.Size(),
				Dist::MpiContext::ThreadSupportName(mpiContext.ThreadSupport()));
		}

		CLI::App application{ "TSP Distributed" };
		std::string profile = "Default"; // fallback config profile
		std::vector<std::string> simOverrides{};
		application.add_option("-p,--profile", profile,
			"Name of configuration profile /Configs/<...> with 'A-Z'");
		application.add_option("--sim", simOverrides,
			"Master override simulation parameter via key=value "
			"(repeatable, CLI wins over config file)");

		// allow one-dash alias "-sim" by normalizing to "--sim" before CLI11 parsing
		std::vector<std::string> cliArgs{};
		cliArgs.reserve((argc > 0) ? static_cast<size_t>(argc) : 0);
		if (argc > 0)
		{
			cliArgs.push_back(argv[0]);
		}
		for (int i = 1; i < argc; i++)
		{
			std::string arg = argv[i];
			if (arg == "-sim")
			{
				arg = "--sim";
			}
			else if (arg.rfind("-sim=", 0) == 0)
			{
				arg.replace(0, 4, "--sim");
			}

			cliArgs.push_back(arg);
		}

		std::vector<char*> cliArgv{};
		cliArgv.reserve(cliArgs.size());
		for (auto& arg : cliArgs)
		{
			cliArgv.push_back(arg.data());
		}

		try
		{
			application.parse(static_cast<int>(cliArgv.size()), cliArgv.data());
		}
		catch (const CLI::ParseError& e)
		{
			return mpiContext.IsRoot()
				? application.exit(e)
				: e.get_exit_code();
		}

		// normalize config profile names (e.g. "a" -> "A", "" -> "Default")
		auto NormalizeProfile = [](std::string& p) -> void
			{
				if (p.empty()) { p = "Default"; return; }

				// single letter -> uppercase
				if (p.size() == 1 && std::isalpha(static_cast<unsigned char>(p[0])))
				{
					p[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(p[0])));
				}
			};
		// hard check for valid profile name
		auto IsValidProfile = [](const std::string& p) -> bool
			{
				if (p == "Default") { return true; }
				return (p.size() == 1 && p[0] >= 'A' && p[0] <= 'Z');
			};

		NormalizeProfile(profile);
		if (!IsValidProfile(profile))
		{
			if (mpiContext.IsRoot())
			{
				SPDLOG_ERROR("Invalid --profile arg '{}'. Use a single letter 'A-Z' "
					"or leave empty for 'Default'.", profile);
			}
			return EXIT_FAILURE;
		}

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
