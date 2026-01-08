// simulation.cpp

// external
#include <chrono>
#include <iostream>
#include <mutex>
#include <spdlog/spdlog.h>

// internal
#include "Source/Eval/Serial/serialization.h"
#include "Source/Sim/Core/diagnostics.h"
#include "Source/Sim/CPU/CPU_Backend_01/cpu_backend_01.h"
#ifdef TURBOSPHARTICLES_WITH_CUDA
	#include "Source/Sim/GPU/GPU_Backend_01/gpu_backend_01.h"
#endif
#include "simulation.h"



using namespace Sim;

SimulationMgr::SimulationMgr(Sim::SimParamsMgr& simParamsMgr,
							 Scene::ScenarioMgr& scenarioMgr,
							 Bench::BenchmarksMgr& benchmarksMgr,
							 Mem::MemoryTracker& memoryTracker,
							 Stats::StatisticsMgr& statisticsMgr,
							 Export::SerializationMgr& serializationMgr)
	: simParamsMgr_{ simParamsMgr }
	, scenarioMgr_{ scenarioMgr }
	, benchmarksMgr_{ benchmarksMgr }
	, memoryTracker_{ memoryTracker }
	, statisticsMgr_{ statisticsMgr }
	, serializationMgr_{ serializationMgr }
{
	// nothing
}

bool SimulationMgr::Init()
{
	// initial snapshot
	SyncParameters(true);

	// default sim state (empty)
	universalSimState_ = Sync::UniversalSimState{};

	// TODO: load initial sim state from a file?

	SPDLOG_INFO("Initializing simulation state...");
	SelectSimScenario(simParams_.scenarioType, configProfile_);
	SelectSimBackend(simParams_.backendVersion);

	SPDLOG_INFO("Initializing backend data structures...");
	ReBuildInitialSimState();

	initialized_.store(true, std::memory_order_release);

	// updated snapshot
	SyncParameters(true);

	DiagnosticsMgr::PrintSimParams(simParams_);
	DiagnosticsMgr::PrintSimSetup(*this, simParams_);

	// TODO: reset universal sim state until it is needed again?
	//universalSimState_ = Sync::UniversalSimState{};

	return true;
}

void SimulationMgr::Shutdown()
{
	RequestStop();
	JoinThreads();

	// TODO: free memory occupied by particles, grids, etc...

	initialized_.store(false, std::memory_order_release);
}

void SimulationMgr::SyncParameters(bool force)
{
	uint64_t latestSimParamsVersion = simParamsMgr_.StableVersion();
	if (latestSimParamsVersion != simParamsVersion_ || force)
	{
		auto snapshot = simParamsMgr_.SnapshotParams();

		simParams_ = std::get<0>(snapshot);
		simParamsVersion_ = std::get<1>(snapshot);
	}
}

void SimulationMgr::SelectSimScenario(Base::ScenarioType type, std::string& profile)
{
	if (running_.load(std::memory_order_acquire))
	{
		SPDLOG_ERROR("Stop the simulation before switching scenario.");
		return;
	}

	// create new scenario
	simScenario_ = Scene::MakeSimScenario(scenarioMgr_, type, profile);
	if (!simScenario_)
	{
		SPDLOG_ERROR("Failed to create sim scenario ({})",
			static_cast<int>(type));
		return;
	}
}

void SimulationMgr::SelectSimBackend(Base::BackendVersion version)
{
	if (running_.load(std::memory_order_acquire))
	{
		SPDLOG_ERROR("Stop the simulation before switching backend");
		return;
	}

	if (simBackend_) // a backend already exists
	{
		std::scoped_lock lock(stateMutex_);

		// transfer its current data into the universal state to save it
		simBackend_->PushToUniversalSimState(universalSimState_);
	}

	// create new backend and initialize it with the universal state
	simBackend_ = MakeSimBackend(version, benchmarksMgr_, simParams_);
	if (!simBackend_)
	{
		SPDLOG_ERROR("Failed to create sim backend ({})",
			static_cast<int>(version));
		return;
	}

	{
		std::scoped_lock lock(stateMutex_);

		// transfer universal state into backend-specific data structures
		simBackend_->PullFromUniversalSimState(universalSimState_);
	}

	// clear benchmarks from previous backend
	benchmarksMgr_.ClearRecordHistory();

	// get and validate the new backend's benchmark operation groups
	simBackend_->GetBenchOpGroups(curBenchOpGroups_);
	if (!Bench::ValidateOpGroups(curBenchOpGroups_))
	{
		SPDLOG_ERROR("Invalid benchmark operation groups for sim backend ({})",
			static_cast<int>(version));
		return;
	}
}

void SimulationMgr::ReBuildInitialSimState()
{
	if (!simScenario_)
	{
		SPDLOG_ERROR("Failed to re-build initial sim state (no scenario selected)");
		return;
	}
	if (!simBackend_)
	{
		SPDLOG_ERROR("Failed to re-build initial sim state (no backend selected)");
		return;
	}

	// TODO: make sure simulation is stopped first to fix bug

	// stop simulation
	simParamsMgr_.Set_SimRunning(false);
	simParamsMgr_.Set_SimRunNextStep(false);

	// lock state during re-build
	std::scoped_lock lock(stateMutex_);

	// reset metadata
	simStep_ = 0;
	simTime_ = 0.0;

	// reset scenario and all its data
	universalSimState_ = {};
	universalSimState_.schemaVersion = 1;
	universalSimState_.step = 0;
	universalSimState_.time = 0.0;

	SyncParameters();

	// re-build initial scenario state by transferring into universal state
	simScenario_->Build(universalSimState_, simParams_);

	// transfer universal state into backend-specific data structures
	simBackend_->PullFromUniversalSimState(universalSimState_);
}

void SimulationMgr::Launch_Async()
{
	if (!initialized_.load(std::memory_order_acquire))
	{
		SPDLOG_ERROR("Failed to launch async SimulationMgr (not initialized)");
		return;
	}
	if (!simBackend_)
	{
		SPDLOG_ERROR("Failed to launch async SimulationMgr (no backend selected)");
		return;
	}
	if (running_.exchange(true))
	{
		SPDLOG_ERROR("Failed to launch async SimulationMgr (already running)");
		return;
	}

	// launch the controller thread
	controllerThread_ = std::jthread([this](const std::stop_token& stopToken)
		{
			RunSim_Async(stopToken);
		});
}

void SimulationMgr::Launch_Blocking()
{
	if (!initialized_.load(std::memory_order_acquire))
	{
		SPDLOG_ERROR("Failed to launch blocking SimulationMgr (not initialized)");
		return;
	}
	if (!simBackend_)
	{
		SPDLOG_ERROR("Failed to launch blocking SimulationMgr (no backend selected)");
		return;
	}
	if (running_.exchange(true))
	{
		SPDLOG_ERROR("Failed to launch blocking SimulationMgr (already running)");
		return;
	}

	// launch threads
	RunSim_Blocking();
}

void SimulationMgr::RunSim_Async(const std::stop_token& stopToken)
{
	printf("\n");
	SPDLOG_INFO("Simulation started running...");

	// initialize before entering the simulation loop --------------------------

	// local copy for thread safety
	const auto paramsInit = simParams_;

	// initial neighbor search for correct statistics
	simBackend_->ExecuteNeighborSearch(simParams_);

	{
		// create visual snapshot and push to its mailbox
		auto visSnap = std::make_shared<Sync::VisualSnapshot>();
		simBackend_->PackVisualSnapshot(*visSnap);

		// pack misc data into snapshot that is not handled by the backend
		visSnap->simStep = simStep_;
		visSnap->boxesB = simScenario_->GetBoundaryBoxes();
		// TODO: add this to vis params instead?
		//visSnap->scalarModeF = Base::ColorCodingMode::VelocityMag;

		visualSnapMailbox_->StoreSnapshot(std::move(visSnap));
	}

	// vis throttle starting point
	lastVisWall_ = std::chrono::steady_clock::now();

	// pack and serialize export snapshot of initial state
	if (paramsInit.exportEnabled)
	{
		// create export snapshot and save to file
		Sync::ExportSnapshot expSnap{};
		simBackend_->PackExportSnapshot(expSnap, simParams_);

		serializationMgr_.SaveExportSnapshot(expSnap, paramsInit.exportDir);
		serializationMgr_.SaveExportSnapXdmfInfo(expSnap, paramsInit.exportDir);
	}

	// enter the simulation loop -----------------------------------------------

	while (!stopToken.stop_requested())
	{
		SyncParameters(true);

		// local copy for thread safety
		const auto params = simParams_;
		const bool doSimStep = (params.simRunning || params.simRunNextStep);

		// explicit maximum number of steps reached
		if (static_cast<int64_t>(simStep_) >= paramsInit.maxSteps &&
			paramsInit.maxSteps > 0) { break; }

		// small sleep to avoid busy spin when nothing to do yet
		if (!doSimStep)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		// execute one simulation step
		if (doSimStep)
		{
			// reset data records for the new step
			benchmarksMgr_.BeginStep();
			memoryTracker_.BeginStep();
			statisticsMgr_.BeginStep();

			// starting point of total sim benchmark
			benchmarksMgr_.StartBench({ Bench::BenchOp::Total_Sim });

			// perform the actual simulation step
			simBackend_->ExecuteFullSimStep(params);

			// update metadata
			simStep_ += 1;
			simTime_ += params.timeStep;

			// reset single-step flag
			if (simParams_.simRunNextStep) { simParamsMgr_.Set_SimRunNextStep(false); }
		}

		// TODO: adjust update frequency with throttle
		// TODO: add upper bound via a "new frame" flag from the visualizer
		// TODO: add lower bound via ...
		if (doSimStep)
		{
			const auto now = std::chrono::steady_clock::now();

			// compute minimum interval from the target rate
			const auto minInterval = std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::duration<double>(1.0 / fmaxf(1e-6f, visMaxHz_)));

			if (now - lastVisWall_ >= minInterval)
			{
				benchmarksMgr_.StartBench({ Bench::BenchOp::Snapshot_Visual });

				// create visual snapshot and push to its mailbox
				auto visSnap = std::make_shared<Sync::VisualSnapshot>();
				simBackend_->PackVisualSnapshot(*visSnap);

				// pack misc data into snapshot that is not handled by the backend
				visSnap->simStep = simStep_;
				visSnap->boxesB = simScenario_->GetBoundaryBoxes();
				// TODO: add this to vis params instead?
				//visSnap->scalarModeF = Base::ColorCodingMode::VelocityMag;

				visualSnapMailbox_->StoreSnapshot(std::move(visSnap));

				benchmarksMgr_.StopBench({ Bench::BenchOp::Snapshot_Visual });

				lastVisWall_ = now;
			}
		}

		// pack and serialize export snapshot at given frequency
		if (simStep_ % params.exportInterval == 0 &&
			params.exportEnabled) { HandleExportSnapshot(params); }

		// evaluate the benchmarks, memory usage, and statistics for this step
		if (doSimStep) { FinalizeBenchMemStatMeasurements(); }

		// pack and send benchmark and statistics snapshots
		if (doSimStep) { HandleBenchMemStatSnapshots(); }

		SyncParameters();

		DisplayMemoryUsageStats(params);
		DisplayPerformanceStats(params);

		// TODO: check if scenario or backend need to be switched?
	}

	// finalize after exiting the simulation loop ------------------------------

	// local copy for thread safety
	const auto paramsFinal = simParams_;

	DisplayMemoryUsageStats(paramsFinal, true);

	{
		// transfer last sim state into the universal state to save it
		std::scoped_lock lock(stateMutex_);
		simBackend_->PushToUniversalSimState(universalSimState_);
	}

	printf("\n");
	SPDLOG_INFO("Simulation stopped running");
}

void SimulationMgr::RunSim_Blocking()
{
	printf("\n");
	SPDLOG_INFO("Simulation started running...");

	// initialize before entering the simulation loop --------------------------

	// local copy for thread safety
	const auto paramsInit = simParams_;

	// initial neighbor search for correct statistics
	simBackend_->ExecuteNeighborSearch(simParams_);

	// pack and serialize export snapshot of initial state
	if (paramsInit.exportEnabled)
	{
		// create export snapshot and save to file
		Sync::ExportSnapshot expSnap{};
		simBackend_->PackExportSnapshot(expSnap, simParams_);

		serializationMgr_.SaveExportSnapshot(expSnap, paramsInit.exportDir);
		serializationMgr_.SaveExportSnapXdmfInfo(expSnap, paramsInit.exportDir);
	}

	// enter the simulation loop -----------------------------------------------

	while (true)
	{
		SyncParameters(true);

		// local copy for thread safety
		const auto params = simParams_;
		const bool doSimStep = (params.simRunning || params.simRunNextStep);

		// explicit maximum number of steps reached
		if (static_cast<int64_t>(simStep_) >= paramsInit.maxSteps &&
			paramsInit.maxSteps > 0) { break; }

		// small sleep to avoid busy spin when nothing to do yet
        if (!doSimStep)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

		// execute one simulation step
		if (doSimStep)
		{
			// reset data records for the new step
			benchmarksMgr_.BeginStep();
			memoryTracker_.BeginStep();
			statisticsMgr_.BeginStep();

			// starting point of total sim benchmark
			benchmarksMgr_.StartBench({ Bench::BenchOp::Total_Sim });

			// perform the actual simulation step
			simBackend_->ExecuteFullSimStep(params);

			// update metadata
			simStep_ += 1;
			simTime_ += params.timeStep;

			// reset single-step flag
			if (simParams_.simRunNextStep) { simParamsMgr_.Set_SimRunNextStep(false); }
		}

		// pack and serialize export snapshot at given frequency
		if (simStep_ % params.exportInterval == 0 &&
			params.exportEnabled) { HandleExportSnapshot(params); }

		// evaluate the benchmarks, memory usage, and statistics for this step
		if (doSimStep) { FinalizeBenchMemStatMeasurements(); }

		// pack and send benchmark and statistics snapshots
		if (doSimStep) { HandleBenchMemStatSnapshots(); }

		SyncParameters();

		DisplayMemoryUsageStats(params);
		DisplayPerformanceStats(params);

		// TODO: check if scenario or backend need to be switched?
	}

	// finalize after exiting the simulation loop ------------------------------

	// local copy for thread safety
	const auto paramsFinal = simParams_;

	DisplayMemoryUsageStats(paramsFinal, true);

	{
		// transfer last sim state into the universal state to save it
		std::scoped_lock lock(stateMutex_);
		simBackend_->PushToUniversalSimState(universalSimState_);
	}

	running_.store(false, std::memory_order_release);

	printf("\n");
	SPDLOG_INFO("Simulation stopped running");
}

// requests the controller thread to stop cleanly
void SimulationMgr::RequestStop()
{
	if (controllerThread_.joinable())
	{
		controllerThread_.request_stop(); // non-blocking
	}
}

// blocks until the controller thread has stopped
void SimulationMgr::JoinThreads()
{
	if (controllerThread_.joinable())
	{
		controllerThread_.join();
		running_.store(false, std::memory_order_release);
	}
}

// sets the configuration profile and updates derived file paths
void SimulationMgr::SetConfigProfile(std::string profile)
{
	configProfile_ = std::move(profile);
	paramsFile_ = configProfile_ + "/params.json";
	scenariosFile_ = configProfile_ + "/scenarios.json";
}

void SimulationMgr::HandleExportSnapshot(const SimParams& params) const
{
	benchmarksMgr_.StartBench({ Bench::BenchOp::Snapshot_Export });
	// fill export snapshot
	Sync::ExportSnapshot expSnap{};
	simBackend_->PackExportSnapshot(expSnap, simParams_);
	serializationMgr_.SaveExportSnapshot(expSnap, params.exportDir);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Snapshot_Export });

	benchmarksMgr_.StartBench({ Bench::BenchOp::Serialization });
	// write export snapshot to file
	serializationMgr_.SaveExportSnapXdmfInfo(expSnap, params.exportDir);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Serialization });
}

void SimulationMgr::HandleBenchMemStatSnapshots() const
{
	// create bench snapshot and push to its mailbox
	auto benchSnap = std::make_shared<Sync::BenchSnapshot>();
	benchmarksMgr_.PackBenchSnapshot(*benchSnap);
	benchSnapMailbox_->StoreSnapshot(std::move(benchSnap));

	// create mem snapshot and push to its mailbox
	auto memSnap = std::make_shared<Sync::MemSnapshot>();
	memoryTracker_.PackMemSnapshot(*memSnap);
	memSnapMailbox_->StoreSnapshot(std::move(memSnap));

	// create stat snapshot and push to its mailbox
	auto statSnap = std::make_shared<Sync::StatSnapshot>();
	statisticsMgr_.PackStatSnapshot(*statSnap);
	statSnapMailbox_->StoreSnapshot(std::move(statSnap));
}

void SimulationMgr::FinalizeBenchMemStatMeasurements() const
{
	// record statistics, archive data and update aggregates
	simBackend_->FillStatisticsRecord(statisticsMgr_.GetCurrentRecord(), simParams_);
	statisticsMgr_.EndStep(simStep_, simTime_);

	// record memory usage, archive data and update aggregates
	simBackend_->FillMemoryRecord(memoryTracker_.GetCurrentRecord(), simParams_);
	memoryTracker_.EndStep(simStep_, simTime_);

	// ending point of total sim benchmark, archive data and update aggregates
	benchmarksMgr_.StopBench({ Bench::BenchOp::Total_Sim });
	benchmarksMgr_.EndStep(simStep_, simTime_);
}

void SimulationMgr::DisplayPerformanceStats(
	const SimParams& params) const
{
	DiagnosticsMgr::PrintLatestPerformanceSummaryAndProgress(
		benchmarksMgr_, curBenchOpGroups_, params.timeUnit, simStep_, params.maxSteps);
}

void SimulationMgr::DisplayMemoryUsageStats(
	const SimParams& params, const bool force) const
{
	if (!force && simStep_ != 1) { return; }

	DiagnosticsMgr::PrintMemorySummary_Standalone(
		memoryTracker_, params.byteUnit, Mem::MemUsageType::Used);

	DiagnosticsMgr::PrintMemorySummary_Standalone(
		memoryTracker_, params.byteUnit, Mem::MemUsageType::Alloc);

	DiagnosticsMgr::PrintMemorySummary_Standalone(
		memoryTracker_, params.byteUnit, Mem::MemUsageType::Occupied);
}
