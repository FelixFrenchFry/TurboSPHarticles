// simulation.h

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
#include "Source/Eval/Serial/metrics_logger.h"
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
	public: // ---------------------------------------------------------------------------------------

		SimulationMgr(Sim::SimParamsMgr& simParamsMgr,
					  Scene::ScenarioMgr& scenarioMgr,
					  Bench::BenchmarksTracker& benchmarksTracker,
					  Mem::MemoryTracker& memoryTracker,
					  Stats::StatisticsTracker& statisticsTracker,
					  Export::SerializationMgr& serializationMgr);

		// initialize sub-systems
		void Init();

		// destroy sub-systems and free up resources
		void Shutdown();

		// pulls latest parameter snapshot and version if they have changed
		void SyncParameters(bool force = false);

		// create or switch scenario
		// (destroys previous scenario and all its data)
		void SelectSimScenario(Base::ScenarioType type, const std::string& profile);

		// create or switch backend
		// (destroys previous backend and all its data)
		void SelectSimBackend(Base::BackendVersion version);

		void ReBuildInitialSimState();

		// launch the simulation in async (for running live visualization)
		// or blocking mode (for running headless)
		void Launch_Async();
		void Launch_Blocking();

		// main simulation loop
		void RunSim_Async(const std::stop_token& stopToken);
		void RunSim_Blocking();

		void RequestStop();
		void JoinThreads();

		// helpers
		void SetConfigProfile(std::string profile);

		void HandleExportSnapshot(const SimParams& params) const;
		void HandleBenchMemStatSnapshots() const;
		void FinalizeBenchMemStatMeasurements() const;
		void BeginRunLogging(const SimParams& params);
		void EndRunLogging();
		void AppendLatestBenchmarkMetricsLog() const;
		void AppendInitialMemoryMetricsLog(const SimParams& params) const;
		void AppendLatestMemoryMetricsLog() const;
		void DisplayPerformanceStats(const SimParams& params) const;
		void DisplayMemoryUsageStats(const SimParams& params, bool force = false) const;

		void SetVisualMailbox(Sync::VisualSnapMailbox* visualSnapMailbox) { visualSnapMailbox_ = visualSnapMailbox; }
		void SetBenchMailbox(Sync::BenchSnapMailbox* benchSnapMailbox) { benchSnapMailbox_ = benchSnapMailbox; }
		void SetMemMailbox(Sync::MemSnapMailbox* memSnapMailbox) { memSnapMailbox_ = memSnapMailbox; }
		void SetStatMailbox(Sync::StatSnapMailbox* statSnapMailbox) { statSnapMailbox_ = statSnapMailbox; }
		void GetBenchOps(Bench::BenchOpList& out) const { out = curBenchOps_; }

		// metadata
		uint32_t simStep_ = 0;
		double simTime_ = 0.0;

	private: // --------------------------------------------------------------------------------------

		friend class DiagnosticsMgr;

		// configuration profile
		std::string configProfile_ = "Default";
		std::string paramsFile_ = "Default/params.json";
		std::string scenariosFile_ = "Default/scenarios.json";

		// parameter manager and snapshot to be managed and passed to sub-systems
		Sim::SimParamsMgr& simParamsMgr_;
		Sim::SimParams simParams_;
		uint64_t simParamsVersion_ = 0;

		// universal simulation state, and backend for compute
		Sync::UniversalSimState universalSimState_;
		std::unique_ptr<I_SimBackend> simBackend_{ nullptr };

		// scenario management
		Scene::ScenarioMgr& scenarioMgr_;
		std::unique_ptr<Scene::ISimScenario> simScenario_{ nullptr };

		// mailbox pointers
		Sync::VisualSnapMailbox* visualSnapMailbox_{ nullptr };
		Sync::BenchSnapMailbox* benchSnapMailbox_{ nullptr };
		Sync::MemSnapMailbox* memSnapMailbox_{ nullptr };
		Sync::StatSnapMailbox* statSnapMailbox_{ nullptr };

		// controller thread spawns worker threads within OpenMP parallel regions
		std::jthread controllerThread_;
		std::atomic<bool> running_{ false };
		// explicit stop request shared by blocking and async simulation loops
		std::atomic<bool> stopRequested_{ false };
		std::atomic<bool> initialized_{ false };

		// benchmark tracking
		Bench::BenchmarksTracker& benchmarksTracker_;
		Bench::BenchOpList curBenchOps_;

		// memory tracking
		Mem::MemoryTracker& memoryTracker_;

		// statistics tracking
		Stats::StatisticsTracker& statisticsTracker_;

		// serialization management
		Export::SerializationMgr serializationMgr_;
		mutable Export::LoggingMgr loggingMgr_;
		mutable Sync::ExportSnapshot exportSnapScratch_;

		// visualization management
		std::chrono::steady_clock::time_point lastVisWall_{};
		float visMaxHz_ = 60.0f;

		// misc
		std::mutex stateMutex_; // protects universalSimState_
	};

} // namespace Sim
