// simulation.h
//
// TODO: comment

#pragma once

// external
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <stop_token>
#include <thread>

// internal
#include "Source/Eval/Bench/benchmarks.h"
#include "Source/Eval/Serial/serialization.h"
#include "Source/Eval/Stats/statistics.h"
#include "Source/Sim/Scene/builder.h"
#include "Source/Sim/Scene/scenario_sim.h"
#include "Source/Sim/Sync/mailboxes.h"
#include "backend_sim.h"
#include "parameters_sim.h"



namespace Sim
{
	class SimulationMgr
	{
	public: // -----------------------------------------------------------------

		SimulationMgr(Sim::SimParamsMgr& simParamsMgr,
					  Scene::ScenarioMgr& scenarioMgr,
					  Bench::BenchmarksMgr& benchmarksMgr,
					  Stats::StatisticsMgr& statisticsMgr,
					  Export::SerializationMgr& serializationMgr);

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

		void Launch_Async(int64_t maxSteps = -1);
		void Launch_Blocking(int64_t maxSteps = -1);

		// main simulation loop
		void RunSim_Async(const std::stop_token& stopToken, int64_t maxSteps);
		void RunSim_Blocking(int64_t maxSteps);

		void RequestStop();

		void JoinThreads();

		void SetVisualMailbox(Sync::VisualSnapMailbox* visualSnapMailbox) { visualSnapMailbox_ = visualSnapMailbox; }
		void SetBenchMailbox(Sync::BenchSnapMailbox* benchSnapMailbox) { benchSnapMailbox_ = benchSnapMailbox; }
		void SetStatMailbox(Sync::StatSnapMailbox* statSnapMailbox) { statSnapMailbox_ = statSnapMailbox; }

		void GetBenchOpGroups(std::vector<Bench::BenchOpGroupDef>& out) const { out = curBenchOpGroups_; }

		// metadata
		int64_t simStep_ = 0;
		double simTime_ = 0.0;

	private: // ----------------------------------------------------------------

		friend class DiagnosticsMgr;

		// parameter manager and snapshot to be managed and passed to sub-systems
		Sim::SimParamsMgr& simParamsMgr_;
		Sim::SimParams simParams_;
		int64_t simParamsVersion_ = 0;

		// universal simulation state, and backend for compute
		Sync::UniversalSimState universalSimState_;
		std::unique_ptr<I_SimBackend> simBackend_{ nullptr };

		// scenario management
		Scene::ScenarioMgr& scenarioMgr_;
		std::unique_ptr<Scene::ISimScenario> simScenario_{ nullptr };

		// mailbox pointers
		Sync::VisualSnapMailbox* visualSnapMailbox_{ nullptr };
		Sync::BenchSnapMailbox* benchSnapMailbox_{ nullptr };
		Sync::StatSnapMailbox* statSnapMailbox_{ nullptr };

		// controller thread spawns worker threads within OpenMP parallel regions
		std::jthread controllerThread_;
		std::atomic<bool> running_{ false };
		std::atomic<bool> initialized_{ false };

		// benchmark management
		Bench::BenchmarksMgr& benchmarksMgr_;
		std::vector<Bench::BenchOpGroupDef> curBenchOpGroups_;

		// statistics management
		Stats::StatisticsMgr& statisticsMgr_;

		// serialization management
		Export::SerializationMgr serializationMgr_;

		// visualization management
		std::chrono::steady_clock::time_point lastVisWall_{};
		float visMaxHz_ = 60.0f;

		// misc
		std::mutex stateMutex_; // protects universalSimState_
	};

} // namespace Sim
