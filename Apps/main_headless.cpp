// main_headless.cpp

// external
#include <CLI/CLI.hpp>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <spdlog/spdlog.h>

// internal
#include "../Source/Sim/Core/parameters_sim.h"
#include "../Source/Sim/Core/simulation.h"



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

		// set OpenMP environment variables
		#ifdef _WIN32
		_putenv_s("OMP_PLACES", "cores"); // use cores as places
		_putenv_s("OMP_PROC_BIND", "close"); // bind threads close to master thread
		_putenv_s("OMP_WAIT_POLICY", "ACTIVE"); // explicit busy-waiting for max sim performance
		#else
		setenv("OMP_PLACES", "cores", 1); // use cores as places
		setenv("OMP_PROC_BIND", "close", 1); // bind threads close to master thread
		setenv("OMP_WAIT_POLICY", "ACTIVE", 1); // explicit busy-waiting for max sim performance
		#endif
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
