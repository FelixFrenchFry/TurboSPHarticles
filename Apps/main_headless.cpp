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
		application.add_option("-p,--profile", profile,
			"Name of configuration profile /Configs/<...> with 'A-Z'");

		CLI11_PARSE(application, argc, argv);

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

		// --- [ hardware budgets ] --------------------------------------------

		const std::uint32_t N_totalThreads = std::max<uint32_t>(1, std::thread::hardware_concurrency());

		Sim::DiagnosticsMgr::PrintHardwareSpecs(N_totalThreads, N_totalThreads, 0);

		// limit TBB to N_simThreads worker threads
		tbb::global_control gc(tbb::global_control::max_allowed_parallelism, N_totalThreads);

		// --- [ initialization of manager systems ] ---------------------------

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

		// --- [ main loop ] ---------------------------------------------------

		// simulation runs on all available threads
		simulationMgr.Launch_Blocking();

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
