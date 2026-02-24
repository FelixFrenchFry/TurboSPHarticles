// main_headless.cpp

// external
#include <CLI/CLI.hpp>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <spdlog/spdlog.h>
#include <tbb/global_control.h>
#include <vector>

// internal
#include "Source/Eval/Serial/serialization.h"
#include "Source/Sim/Core/diagnostics.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Core/simulation.h"



int main(int argc, char** argv)
{
	#if defined(_WIN32)
	printf("(Running on Windows)\n");
	#elif defined(__linux__)
	printf("(Running on Linux)\n");
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
			return application.exit(e);
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
			SPDLOG_ERROR("Invalid --profile arg '{}'. Use a single letter 'A-Z' or leave empty for 'Default'.",
				profile);
			return EXIT_FAILURE;
		}

		// --- [ initialization of manager systems ] ----------------------------------------------------

		// config profile and simulation parameters
		Sim::SimParamsMgr simParamsMgr{};
		if (!simParamsMgr.LoadSimParamsFromProfile(profile))
		{
			if (profile != "Default")
			{
				SPDLOG_WARN("Failed to load '/{}/params.json'. Falling back to '/Default/params.json'.", profile);
				profile = "Default";

				if (!simParamsMgr.LoadSimParamsFromProfile(profile))
				{
					SPDLOG_ERROR("Failed to load '/Default/params.json' too.");
					return EXIT_FAILURE;
				}
			}
			else
			{
				SPDLOG_ERROR("Failed to load '/Default/params.json'.");
				return EXIT_FAILURE;
			}
		}
		printf("\n");
		SPDLOG_INFO("Using config profile '{}'", profile);

		if (!simOverrides.empty())
		{
			if (!simParamsMgr.LoadSimParamsFromCliOverrides(simOverrides, true))
			{
				SPDLOG_ERROR("Failed to apply one or more '-sim/--sim' overrides.");
				return EXIT_FAILURE;
			}

			SPDLOG_INFO("Applied {} simulation override(s) from command line",
				simOverrides.size());
		}

		// --- [ hardware budgets ] ---------------------------------------------------------------------

		auto ResolveSimThreadBudget = [](const std::uint32_t defaultSimThreads,
										 const std::int64_t maxThreadsSim) -> std::uint32_t
			{
				if (maxThreadsSim == -1)
				{
					return defaultSimThreads;
				}
				if (maxThreadsSim == 0)
				{
					SPDLOG_WARN("Invalid 'maxThreadsSim' value 0 (must be -1 or > 0). "
						"Falling back to default simulation thread budget {}",
						defaultSimThreads);
					return defaultSimThreads;
				}
				if (maxThreadsSim < -1)
				{
					SPDLOG_WARN("Invalid 'maxThreadsSim' value {} (must be -1 or > 0). "
						"Falling back to default simulation thread budget {}",
						maxThreadsSim, defaultSimThreads);
					return defaultSimThreads;
				}

				const std::uint64_t requested = static_cast<std::uint64_t>(maxThreadsSim);
				if (requested > static_cast<std::uint64_t>(defaultSimThreads))
				{
					return defaultSimThreads;
				}

				return static_cast<std::uint32_t>(requested);
			};

		const std::uint32_t N_totalThreads = std::max<uint32_t>(1, std::thread::hardware_concurrency());
		const std::uint32_t N_simThreadsDefault = N_totalThreads;
		const std::int64_t maxThreadsSim = simParamsMgr.SnapshotParams().first.maxThreadsSim;
		const std::uint32_t N_simThreads = ResolveSimThreadBudget(
			N_simThreadsDefault, maxThreadsSim);

		Sim::DiagnosticsMgr::PrintHardwareSpecs(N_totalThreads, N_simThreads, 0);

		// limit TBB to the simulation thread budget
		tbb::global_control gc(tbb::global_control::max_allowed_parallelism, N_simThreads);

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

		if (!simulationMgr.Init())
		{
			SPDLOG_ERROR("Failed to initialize SimulationMgr");
			return EXIT_FAILURE;
		}
		simParamsMgr.Set_SimRunning(true);

		// --- [ main loop ] ----------------------------------------------------------------------------

		// simulation runs on the configured simulation thread budget
		simulationMgr.Launch_Blocking();

		// --- [ shutdown ] -----------------------------------------------------------------------------

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
