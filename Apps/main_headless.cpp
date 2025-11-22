// main_headless.cpp

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
		SPDLOG_INFO("Launching TSP (headless application)...");

		CLI::App application{ "TSP Headless" };
		std::string configFile; // TODO: load default config file?

		// TODO: make this required or use default config file, if this is not provided?
		application.add_option("--config", configFile, "Configuration file");
		// TODO: add --version, --threads, --backend, etc... ?

		CLI11_PARSE(application, argc, argv);

		// --- [ hardware budgets ] --------------------------------------------

		const std::uint32_t N_totalThreads = std::max<uint32_t>(1, std::thread::hardware_concurrency());

		Sim::PrintHardwareSpecs(N_totalThreads, N_totalThreads, 0);

		// limit TBB to N_simThreads worker threads
		tbb::global_control gc(tbb::global_control::max_allowed_parallelism, N_totalThreads);

		// --- [ initialization of manager systems ] ---------------------------
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
