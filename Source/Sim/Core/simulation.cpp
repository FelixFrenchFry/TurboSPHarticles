// simulation.cpp

// external
#include <chrono>
#include <mutex>
#include <omp.h>
#include <spdlog/spdlog.h>

// internal
#include "simulation.h"



using namespace Sim;

SimulationMgr::SimulationMgr(Sim::SimParamsMgr& simParamsMgr,
							 Scene::ScenarioMgr& scenarioMgr,
							 Bench::BenchmarksMgr& benchmarksMgr,
							 Stats::StatisticsMgr& statisticsMgr)
	: simParamsMgr_{ simParamsMgr }
	, scenarioMgr_{ scenarioMgr }
	, benchmarksMgr_{ benchmarksMgr }
	, statisticsMgr_{ statisticsMgr }
{
	// nothing
}

bool SimulationMgr::Init()
{
	// initial snapshot
	SyncParameters(true);

	// default sim state (empty)
	universalSimState_ = Sync::UniversalSimState{};

	// TODO: load initial sim state from a file

	// default sim scenario (BreakingDam)
	SelectSimScenario(Base::ScenarioType::BreakingDam);

	// default sim backend (CPU01)
	SelectSimBackend(Base::BackendVersion::CPU01);

	ReBuildInitialSimState();

	initialized_.store(true, std::memory_order_release);

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
	int64_t latestSimParamsVersion = simParamsMgr_.StableVersion();
	if (latestSimParamsVersion != simParamsVersion_ ||
		force)
	{
		auto snapshot = simParamsMgr_.SnapshotParams();

		// TODO: remove debug log
		// SPDLOG_INFO("Synced 'simParams' snapshot from version {} to {}",
		// simParamsVersion_, std::get<1>(snapshot));

		simParams_ = std::get<0>(snapshot);
		simParamsVersion_ = std::get<1>(snapshot);
	}
}

void SimulationMgr::SelectSimScenario(Base::ScenarioType type)
{
	if (running_.load(std::memory_order_acquire))
	{
		SPDLOG_ERROR("Stop the simulation before switching scenario.");
		return;
	}

	// create new scenario
	simScenario_ = Scene::MakeSimScenario(scenarioMgr_, type, simParams_);
	if (!simScenario_)
	{
		SPDLOG_ERROR("Failed to create sim scenario ({})",
			static_cast<int>(type));
		return;
	}

	SPDLOG_INFO("Selected sim scenario ({})",
		static_cast<int>(type));
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

	SPDLOG_INFO("Selected sim backend ({})",
		static_cast<int>(version));
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

void SimulationMgr::Launch_Async(std::uint32_t N_simThreads)
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
	controllerThread_ = std::jthread([this, N_simThreads](const std::stop_token& stopToken)
		{
			RunSim_Async(N_simThreads, stopToken);
		});
}

void SimulationMgr::RunSim_Async(std::uint32_t N_simThreads, const std::stop_token& stopToken)
{
	SPDLOG_INFO("Simulation started running with {} worker threads",
		N_simThreads);

	// initialize before entering the simulation loop --------------------------

	// create visual snapshot and push to mailbox
	auto snap = std::make_shared<Sync::VisualSnapshot>();
	simBackend_->PackVisualSnapshot(*snap);

	// pack misc data into snapshot that is not handled by the backend
	snap->simStep = simStep_;
	snap->boxesB = simScenario_->GetBoundaryBoxes();
	// TODO: add this to vis params instead?
	//snap->scalarModeF = Base::ColorCodingMode::VelocityMag;

	// move snapshot into mailbox (non-blocking)
	visualSnapMailbox_->StoreSnapshot(std::move(snap));

	// vis throttle starting point
	lastVisWall_ = std::chrono::steady_clock::now();

	// enter the simulation loop -----------------------------------------------

	while (!stopToken.stop_requested())
	{
		// local copy for thread safety
		const auto params = simParams_;
		const bool doSimStep = (params.simRunning || params.simRunNextStep);

		// execute one simulation step
		if (doSimStep)
		{
			// reset data records for the new step
			benchmarksMgr_.BeginStep();
			statisticsMgr_.BeginStep();

			// starting point of total sim benchmark
			benchmarksMgr_.StartBench({ Bench::BenchOp::Total_Sim });

			// perform the actual simulation step
			simBackend_->ExecuteSimStep(params);

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
				auto snap = std::make_shared<Sync::VisualSnapshot>();
				simBackend_->PackVisualSnapshot(*snap);

				// pack misc data into snapshot that is not handled by the backend
				snap->simStep = simStep_;
				snap->boxesB = simScenario_->GetBoundaryBoxes();
				// TODO: add this to vis params instead?
				//snap->scalarModeF = Base::ColorCodingMode::VelocityMag;

				visualSnapMailbox_->StoreSnapshot(std::move(snap));

				benchmarksMgr_.StopBench({ Bench::BenchOp::Snapshot_Visual });

				lastVisWall_ = now;
			}
		}

		// evaluate the step
		if (doSimStep)
		{
			// record statistics, archive data and update aggregates
			simBackend_->FillStatisticsRecord(statisticsMgr_.GetCurrentRecord(), simParams_);
			statisticsMgr_.EndStep(simStep_, simTime_);

			// ending point of total sim benchmark, archive data and update aggregates
			benchmarksMgr_.StopBench({ Bench::BenchOp::Total_Sim });
			benchmarksMgr_.EndStep(simStep_, simTime_);
		}

		// pack and send benchmark and statistics snapshots
		// (to be retrieved by the visualization manager)
		if (doSimStep)
		{
			// create bench snapshot and push to its mailbox
			auto benchSnap = std::make_shared<Sync::BenchSnapshot>();
			benchmarksMgr_.PackBenchSnapshot(*benchSnap);
			benchSnapMailbox_->StoreSnapshot(std::move(benchSnap));

			// create stat snapshot and push to its mailbox
			auto statSnap = std::make_shared<Sync::StatSnapshot>();
			statisticsMgr_.PackStatSnapshot(*statSnap);
			statSnapMailbox_->StoreSnapshot(std::move(statSnap));
		}

		SyncParameters();

		// TODO: check if scenario or backend need to be switched?
	}

	// finalize after exiting the simulation loop ------------------------------

	{
		// transfer last sim state into the universal state to save it
		std::scoped_lock lock(stateMutex_);
		simBackend_->PushToUniversalSimState(universalSimState_);
	}

	SPDLOG_INFO("Simulation stopped running");
}

// requests the controller thread to stop cleanly
void SimulationMgr::RequestStop()
{
	if (controllerThread_.joinable())
	{
		SPDLOG_INFO("Requesting stop of SimulationMgr...");
		controllerThread_.request_stop(); // non-blocking
	}
}

// blocks until the controller thread has stopped
void SimulationMgr::JoinThreads()
{
	if (controllerThread_.joinable())
	{
		SPDLOG_INFO("Joining SimulationMgr thread...");
		controllerThread_.join();
		running_.store(false, std::memory_order_release);
	}
}
