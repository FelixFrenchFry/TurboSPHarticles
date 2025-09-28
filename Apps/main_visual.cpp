// main_visual.cpp

// external
#include <CLI/CLI.hpp>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <spdlog/spdlog.h>

// internal
#include "../Source/Sim/Core/parameters_sim.h"
#include "../Source/Sim/Core/simulation.h"
#include "../Source/Vis/Core/parameters_vis.h"
#include "../Source/Vis/Core/visualization.h"



int main(int argc, char** argv)
{
	// [year-month-day 12:34:56.789] [file.cpp:line] [type] message
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%s:%#] [%^%l%$] %v");
	spdlog::flush_on(spdlog::level::err);

	try // application setup
	{
		SPDLOG_INFO("Launching TSP (visual application)...");

		CLI::App application{ "TSP Visual" };
		std::string configFile; // TODO: load default config file?

		// TODO: make this required or use default config file, if this is not provided?
		application.add_option("--config", configFile, "Configuration file");
		// TODO: add --version, --threads, --backend, etc... ?

		CLI11_PARSE(application, argc, argv);

		// set OpenMP environment variables
		#ifdef _WIN32
		_putenv_s("OMP_PLACES", "cores"); // use cores as places
		_putenv_s("OMP_PROC_BIND", "close"); // bind threads close to master thread
		_putenv_s("OMP_WAIT_POLICY", "PASSIVE"); // avoid busy-waiting for smoother vis performance
		#else
		setenv("OMP_PLACES", "cores", 1); // use cores as places
		setenv("OMP_PROC_BIND", "close", 1); // bind threads close to master thread
		setenv("OMP_WAIT_POLICY", "PASSIVE", 1); // avoid busy-waiting for smoother vis performance
		#endif

		// ===== initialization of manager systems =====

		Sim::SimParamsMgr simParamsMgr{};
		Vis::VisParamsMgr visParamsMgr{};
		Scene::ScenarioMgr scenarioMgr{};
		Sim::SimulationMgr simulationMgr{ simParamsMgr, scenarioMgr };
		Vis::VisualizationMgr visualizationMgr{ simulationMgr, simParamsMgr, visParamsMgr };

		Sync::VisualSnapMailbox visualMailbox{};
		simulationMgr.SetVisualMailbox(&visualMailbox);
		visualizationMgr.SetVisualMailbox(&visualMailbox);

		if (!simulationMgr.Init())
		{
			SPDLOG_ERROR("Failed to initialize SimulationMgr");
			return EXIT_FAILURE;
		}

		auto tempVisParams = visParamsMgr.SnapshotParams().first;
		if (!visualizationMgr.Init(tempVisParams.window))
		{
			SPDLOG_ERROR("Failed to initialize VisualizationMgr");
			return EXIT_FAILURE;
		}

		simParamsMgr.PresetParams_A();

		// ===== hardware budgets =====

		const std::uint32_t N_totalThreads = std::max<uint32_t>(1, std::thread::hardware_concurrency());
		const std::uint32_t N_visThreads = 1; // TODO: make configurable?
		const std::uint32_t N_simThreads = N_totalThreads > N_visThreads ?
										   N_totalThreads - N_visThreads : 1;

		SPDLOG_INFO("Number of threads [total/sim/vis]: [ {} / {} / {} ]",
			N_totalThreads, N_simThreads, N_visThreads);

		// ===== main loop =====

		// TODO: run simulation with separate thread(s)

		// simulation runs on background controller thread
		// (using OpenMP parallel regions to spawn N-1 worker threads within)
		simulationMgr.Launch_Async(N_simThreads);

		// visualization runs on main thread
		visualizationMgr.RunVis();

		// ===== shutdown =====

		simulationMgr.RequestStop();
		simulationMgr.JoinThreads();
		visualizationMgr.Shutdown();
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
