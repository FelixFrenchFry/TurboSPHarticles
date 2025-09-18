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
