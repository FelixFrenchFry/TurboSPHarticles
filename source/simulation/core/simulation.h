// simulation.h
#pragma once

// external
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

// internal
#include "source/eval/bench/benchmarks.h"
#include "source/eval/serial/metrics_logger.h"
#include "source/eval/serial/serialization.h"
#include "source/eval/stats/statistics.h"
#include "source/simulation/scene/builder.h"
#include "source/simulation/scene/scenario_sim.h"
#include "source/simulation/sync/mailboxes.h"
#include "backend_sim.h"
#include "parameters_sim.h"



namespace Sim
{
	class SimulationMgr
	{
	public:

		SimulationMgr(Sim::SimParamsMgr& simParamsMgr,
					  Scene::ScenarioMgr& scenarioMgr,
					  Bench::BenchmarksTracker& benchmarksTracker,
					  Mem::MemoryTracker& memoryTracker,
					  Stats::StatisticsTracker& statisticsTracker,
					  Export::SerializationMgr& serializationMgr);

		// initialize sub-systems
		void Init(const std::string& restartCheckpoint = {});

		// destroy sub-systems and free up resources
		void Shutdown();

		// pulls latest parameter snapshot and version if they have changed
		void SyncParameters(bool force = false);

		// create or switch scenario
		// (destroys previous scenario and all its data)
		void SelectSimScenario(Base::ScenarioType type, const std::string& profile);

		// create or re-create backend when no backend is active
		void CreateAndInitializeSimBackend(Base::BackendType backendType);

		void ResetAndRebuildSimState();

		// launch the simulation in async (for running live visualization)
		// or blocking mode (for running headless)
		void Launch_Async();
		void Launch_Blocking();

		void RequestStop();
		void JoinThreads();

		// helpers
		void SetConfigProfile(std::string profile);

		void WriteExportSnapshot(const SimParams& params) const;
		void FinalizeRunState(const SimParams& params, bool checkpointDue = false);
		bool PublishVisualSnapshot(bool force = false);
		void HandleBenchMemStatSnapshots() const;
		void FinalizeBenchMemStatMeasurements() const;
		void AppendInitialMemoryMetricsLog() const;
		void AppendLatestMemoryMetricsLog() const;
		void AppendLatestBenchmarkMetricsLog() const;
		void BeginRunLogging(const SimParams& params);
		void EndRunLogging();
		void DisplayPerformanceStats(const SimParams& params) const;
		void DisplayMemoryUsageStats(const SimParams& params, bool force = false) const;

		void SetVisualMailbox(Sync::VisualSnapMailbox* visualSnapMailbox) { visualSnapMailbox_ = visualSnapMailbox; }
		void SetVisualScalarMode(Base::ColorCodingMode scalarMode) { visualScalarMode_.store(scalarMode, std::memory_order_release); }
		void SetBenchMailbox(Sync::BenchSnapMailbox* benchSnapMailbox) { benchSnapMailbox_ = benchSnapMailbox; }
		void SetMemMailbox(Sync::MemSnapMailbox* memSnapMailbox) { memSnapMailbox_ = memSnapMailbox; }
		void SetStatMailbox(Sync::StatSnapMailbox* statSnapMailbox) { statSnapMailbox_ = statSnapMailbox; }
		void GetBenchOps(Bench::BenchOpList& out) const { out = curBenchOps_; }

		// metadata
		uint64_t simStep_ = 0;
		double simTime_ = 0.0;

	private:

		friend class DiagnosticsMgr;

		// simulation loop, shared by blocking and async launch modes
		void RunSim(const std::stop_token& stopToken, bool publishVis);

		// captures the final backend state and publishes a completed checkpoint
		void SaveCheckpoint(const SimParams& params);

		// loads state and scene metadata before backend initialization
		void LoadCheckpoint(const std::string& manifestPath);

		// sets the spatial domain bounds of the backend, or clears them if none are available
		void ApplySimDomainBoundsToBackend();

		// configuration profile
		std::string configProfile_ = "default";
		std::string paramsFile_ = "params_def.json";
		std::string scenariosFile_ = "scenarios_def.json";

		// parameter manager and snapshot to be managed and passed to sub-systems
		Sim::SimParamsMgr& simParamsMgr_;
		Sim::SimParams simParams_;
		uint64_t simParamsVersion_ = 0;
		std::chrono::steady_clock::time_point initTime_{};

		// universal simulation state, and backend for compute
		Sync::UniversalSimState universalSimState_;
		std::unique_ptr<I_SimBackend> simBackend_{ nullptr };

		// scenario management
		Scene::ScenarioMgr& scenarioMgr_;
		std::unique_ptr<Scene::ISimScenario> simScenario_{ nullptr };
		std::vector<Scene::OBB> boundaryBoxes_{};
		struct DomainBounds
		{
			glm::vec3 min{};
			glm::vec3 max{};
		};
		std::optional<DomainBounds> domainBounds_{};

		// mailbox pointers
		Sync::VisualSnapMailbox* visualSnapMailbox_{ nullptr };
		Sync::BenchSnapMailbox* benchSnapMailbox_{ nullptr };
		Sync::MemSnapMailbox* memSnapMailbox_{ nullptr };
		Sync::StatSnapMailbox* statSnapMailbox_{ nullptr };
		std::atomic<Base::ColorCodingMode> visualScalarMode_{ Base::ColorCodingMode::None };

		// controller thread spawns worker threads within OpenMP parallel regions
		std::jthread controllerThread_;
		std::atomic<bool> running_{ false };
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
		Export::SerializationMgr& serializationMgr_;
		mutable Export::LoggingMgr loggingMgr_;
		mutable Sync::ExportSnapshot exportSnapScratch_;

		// visualization management
		std::optional<uint64_t> lastVisStep_{};
		Base::ColorCodingMode lastVisScalarMode_{ Base::ColorCodingMode::None };

		// misc
		std::mutex stateMutex_; // serializes reset with simulation and snapshot reads
	};

} // namespace Sim
