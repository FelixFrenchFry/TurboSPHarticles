// simulation.h
//
// TODO: comment

#pragma once

// external
#include <atomic>
#include <cstdint>
#include <stop_token>
#include <thread>

// internal
#include "Source/Sim/Scene/builder.h"
#include "Source/Sim/Scene/scenario_sim.h"
#include "Source/Sim/Sync/mailboxes.h"
#include "Source/Sim/Sync/snapshots.h"
#include "backend_sim.h"
#include "parameters_sim.h"



namespace Sim
{
	class SimulationMgr
	{
	public: // ==============================

		SimulationMgr(Sim::SimParamsMgr& simParamsMgr,
					  Scene::ScenarioMgr& scenarioMgr);

		// initialize sub-systems
		bool Init();

		// destroy sub-systems and free up resources
		void Shutdown();

		// pulls latest parameter snapshot and version if they have changed
		void SyncParameters(bool force = false);

		// create or switch scenario
		// (destroys previous scenario and all its data)
		void SelectSimScenario(Base::ScenarioType type);

		// create or switch backend
		// (destroys previous backend and all its data)
		void SelectSimBackend(Base::BackendVersion version);

		void ReBuildInitialSimState();

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

		// universal simulation state, and backend for compute
		Sync::UniversalSimState universalSimState_;
		std::unique_ptr<ISimBackend> simBackend_{ nullptr };

		// scenario management
		Scene::ScenarioMgr scenarioMgr_;
		std::unique_ptr<Scene::ISimScenario> simScenario_{ nullptr };

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
