// simulation.cpp

// external
#include <omp.h>
#include "spdlog/spdlog.h"

// internal
#include "Source/Eval/Export/serialization.h"
#include "simulation.h"



using namespace Sim;

SimulationMgr::SimulationMgr(Sim::SimParamsMgr& simParamsMgr)
	: simParamsMgr_{ simParamsMgr }
{
	// TODO: implement
}

bool SimulationMgr::Init()
{
	// initial snapshot
	SyncParameters(true);

	// default sim state (empty)
	universalSimState_ = Sync::UniversalSimState{};

	// TODO: load initial sim state from a file
	// TODO: load sim scenario

	// default sim backend (CPU_01)
	SelectSimBackend(Base::BackendVersion::CPU_01);

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

void SimulationMgr::SelectSimBackend(Base::BackendVersion version)
{
	if (simBackend_) // a backend already exists
	{
		// transfer its current data into the universal state to save it
		std::scoped_lock lock(stateMutex_);
		simBackend_->PushToUniversalSimState(universalSimState_);
		simBackend_.reset();
	}

	// create new backend and initialize it with the universal state
	simBackend_ = MakeSimBackend(version, simParams_);
	if (!simBackend_)
	{
		SPDLOG_ERROR("Failed to create sim backend {}",
			static_cast<int>(version));
		return;
	}

	{
		// transfer universal state into backend-specific data structures
		std::scoped_lock lock(stateMutex_);
		simBackend_->PullFromUniversalSimState(universalSimState_);
	}

	SPDLOG_INFO("Selected sim backend {}",
		static_cast<int>(version));
}

void SimulationMgr::LoadInitialSimState(const Sync::UniversalSimState& state)
{
	// TODO: implement
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
	// configure pool size for any OpenMP usage within the selected backend
	omp_set_dynamic(0); // disable dynamic adjustment of threads
	omp_set_num_threads(N_simThreads); // set fixed number of worker threads

	SPDLOG_INFO("Simulation started running with {} worker threads",
		N_simThreads);

	// main simulation step loop
	while (!stopToken.stop_requested())
	{
		// execute one simulation step
		simBackend_->SimStep(1);

		// TODO: adjust update frequency
		if (visualSnapMailbox_)
		{
			// create visual snapshot and push to mailbox
			auto snap = std::make_shared<Sync::VisualSnapshot>();
			simBackend_->PackVisualSnapshot(*snap);
			visualSnapMailbox_->StoreSnapshot(std::move(snap)); // non-blocking
		}

		SyncParameters();
	}

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

void SimulationMgr::SetVisualMailbox(Sync::VisualSnapMailbox* visualSnapMailbox)
{
	visualSnapMailbox_ = visualSnapMailbox;
}
