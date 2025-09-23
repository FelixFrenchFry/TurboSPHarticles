// simulation.h
//
// TODO: comment

#pragma once

// external
#include <atomic>
#include <cstdint>
#include <span>
#include <stop_token>
#include <thread>

// internal
#include "backend_sim.h"
#include "parameters_sim.h"
#include "../Sync/mailboxes.h"
#include "../Sync/snapshots.h"



namespace Sim
{
	// TODO: prototype!!!
	struct SimStateView
	{
		std::uint32_t N_particles{ 0 };
		std::span<const float> posX, posY, posZ;
		std::span<const float> velX, velY, velZ;
		std::span<const float> density;
		std::span<const float> pressure;

		// TODO: finish implementation
	};

	class SimulationMgr
	{
	public: // ==============================

		SimulationMgr(Sim::SimParamsMgr& simParamsMgr);

		// initialize sub-systems
		bool Init();

		// destroy sub-systems and free up resources
		void Shutdown();

		// pulls latest parameter snapshot and version if they have changed
		void SyncParameters(bool force = false);

		// create or switch backend
		// (destroys previous backend and all its data)
		void SelectSimBackend(Base::BackendVersion version);

		void LoadInitialSimState(const Sync::UniversalSimState& state);

		// TODO: void Launch(std::uint32_t N_simThreads);
		// TODO: void RunSim(std::uint32_t N_simThreads);

		void Launch_Async(std::uint32_t N_simThreads);

		// main simulation loop
		void RunSim_Async(std::uint32_t N_simThreads, const std::stop_token& stopToken);

		void RequestStop();

		void JoinThreads();

		void SetVisualMailbox(Sync::VisualSnapMailbox* visualSnapMailbox);

	private: // ==============================

		// parameter manager and snapshot to be managed and passed to sub-systems
		Sim::SimParamsMgr& simParamsMgr_;
		Sim::SimParams simParams_;
		int64_t simParamsVersion_ = 0;

		// universal simulation state and backend for compute
		Sync::UniversalSimState universalSimState_;
		std::unique_ptr<ISimBackend> simBackend_{ nullptr };

		// TODO: comment
		Sync::VisualSnapMailbox* visualSnapMailbox_{ nullptr }; // not owned

		// controller thread spawns worker threads within OpenMP parallel regions
		std::jthread controllerThread_;
		std::atomic<bool> running_{ false };
		std::atomic<bool> initialized_{ false };

		// misc
		std::mutex stateMutex_; // protects universalSimState_
	};

} // namespace Sim
