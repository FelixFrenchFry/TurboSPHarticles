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

	// empty default state
	universalSimState_ = Sync::UniversalSimState{};

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
	uint64_t latestSimParamsVersion = simParamsMgr_.StableVersion();
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
	simBackend_ = MakeSimBackend(version, simParams_);

	if (!simBackend_)
	{
		SPDLOG_ERROR("Failed to create sim backend");
	}
}

void SimulationMgr::Launch_Async(std::uint32_t N_simThreads)
{
	if (!initialized_.load(std::memory_order_acquire))
	{
		SPDLOG_ERROR("Failed to launch async SimulationMgr (not initialized)");
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
	// configure OpenMP for the simulation pool
	omp_set_dynamic(0); // disable dynamic adjustment of threads
	omp_set_num_threads(N_simThreads); // set fixed number of worker threads

	SPDLOG_INFO("Simulation started running with {} worker threads",
		N_simThreads);

	// main simulation step loop
	while (!stopToken.stop_requested())
	{
		// one simulation step with one parallel region
		#pragma omp parallel
		{
			// TODO: use OpenMP for/single/etc... for SPH computations
		}
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
		SPDLOG_INFO("Stopping completed");
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
		SPDLOG_INFO("Joining completed");
	}
}

void SimulationMgr::SetVisualMailbox(Sync::VisualSnapMailbox* visualSnapMailbox)
{
	visualSnapMailbox_ = visualSnapMailbox;
}

// convert simulation state from data structures used for the computations
// to portable data structures used by the visual snapshot and push to mailbox
void SimulationMgr::PushToVisualMailbox(std::int64_t step)
{
	if (!visualSnapMailbox_) { return; }

	auto snap = std::make_shared<Sync::VisualSnapshot>();
	// TODO: translate simulation state into visual snapshot
	/*
	snap->position.resize(TODO);
	snap->scalarData.resize(TODO);
	snap->scalarMode = TODO;
	snap->N_particles = TODO;
	snap->step = step;
	*/

	// TODO: fill in data from simulation state
	/*
	#pragma omp parallel for schedule(static)
	for (std::int64_t i; i < TODO; i++)
	{
		snap->position[i] = TODO;
		snap->scalarData[i] = TODO;
	}
	*/

	visualSnapMailbox_->StoreSnapshot(std::move(snap)); // non-blocking
}
