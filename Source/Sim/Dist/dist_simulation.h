// dist_simulation.h

#pragma once

// external
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// internal
#include "Source/Eval/Bench/benchmarks.h"
#include "Source/Eval/Mem/memory.h"
#include "Source/Eval/Serial/metrics_logger.h"
#include "Source/Eval/Serial/serialization.h"
#include "Source/Sim/Core/backend_sim.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Scene/scenario_sim.h"
#include "domain_decomp.h"
#include "dist_particle_state.h"
#include "mpi_context.h"



namespace Dist
{
	class DistSimulationMgr final
	{
	public:

		DistSimulationMgr(MpiContext& mpiContext,
						  Sim::SimParamsMgr& simParamsMgr);

		// initialize sub-systems
		bool Init(std::string& profile, const std::vector<std::string>& simOverrides);

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

		bool ReBuildInitialSimState();

		// prints and validates current distributed startup state
		void RunStartupCheck() const;

		// main simulation loop
		void RunSim_Blocking();

		void RequestStop();

		// metadata
		uint32_t simStep_ = 0;
		double simTime_ = 0.0;

	private:

		// root reads params.json, then every rank loads the broadcast copy
		bool LoadParamsFromProfileBroadcast(std::string& profile);

		// applies the same command-line overrides on every rank
		bool LoadParamsFromCliOverrides(const std::vector<std::string>& simOverrides);

		// creates the ownership / halo decomp from the selected scenario bounds
		bool BuildDomainDecomp();

		// builds the full scenario state before rank-local compaction
		bool BuildGlobalSimState();

		// keeps owned + halo particles for this rank
		bool BuildRankLocalSimState();

		// validates rank-local [ owned ][ halo ] state
		bool ValidateRankLocalSimState() const;

		// prints rank-local particle ownership / halo counts
		void LogRankLocalSimState() const;

		// transfers rank-local state into the selected backend
		bool InitRankLocalBackend();

		// runs a backend startup smoke test on rank-local data
		bool RunRankLocalBackendStartupCheck();

		// gathers the authoritative owned state on the root rank
		bool GatherOwnedGlobalSimStateOnRoot(const char* phaseName) const;

		// gathers and serializes an export snapshot on the root rank
		bool HandleExportSnapshot(const Sim::SimParams& params) const;

		// mirrors SimulationMgr run logging for rank-local distributed records
		void BeginRunLogging(const Sim::SimParams& params);
		void EndRunLogging();
		void AppendInitialMemoryMetricsLog(const Sim::SimParams& params) const;
		void AppendLatestMemoryMetricsLog() const;
		void AppendLatestBenchmarkMetricsLog() const;
		void FinalizeBenchMemMeasurements(const Sim::SimParams& params) const;
		void DisplayPerformanceStats(const Sim::SimParams& params) const;
		void DisplayMemoryUsageStats(const Sim::SimParams& params, bool force = false) const;

		// runs one distributed step with data movement around the rank-local backend
		bool ExecuteDistributedStep(const Sim::SimParams& params);

		// migrates owned fluid particles to their current owner ranks
		bool MigrateOwnedFluidParticlesToOwnerRanks();

		// temporary guard until boundary particle migration exists
		bool ValidateOwnedBoundaryParticlesStillOnRank() const;

		// reloads backend from rank-local state, runs neighbor search, and syncs order metadata
		bool ReloadRankLocalBackendAndRunNeighborSearch(const char* phaseName);

		// validates particle identity metadata and payload against current backend state
		bool ValidateRankLocalBackendParticleOrder(const char* phaseName) const;

		// validates particle identity metadata without comparing changed solver payload
		bool ValidateRankLocalParticleMetadata(const char* phaseName) const;

		// exchanges halo particles and rebuilds compact rank-local state
		bool ExchangeRankLocalHalos();

		// keeps particle identity metadata in sync with backend data ordering
		bool ApplyLastBackendParticleReOrder();

		// reads raw params.json text, parsing happens after broadcast
		static bool TryReadProfileJson(const std::string& profile, std::string& jsonText);

		MpiContext& mpiContext_;

		// parameter manager and snapshot to be managed and passed to sub-systems
		Sim::SimParamsMgr& simParamsMgr_;
		Sim::SimParams simParams_;
		uint64_t simParamsVersion_ = 0;

		// configuration profile
		std::string configProfile_ = "Default";
		std::string paramsFile_ = "Default/params.json";
		std::string scenariosFile_ = "Default/scenarios.json";

		// scenario management
		Scene::ScenarioMgr scenarioMgr_;
		std::unique_ptr<Scene::ISimScenario> simScenario_{ nullptr };

		// distributed domain and simulation state
		std::unique_ptr<IDomainDecomp> domainDecomp_{ nullptr };
		Sync::UniversalSimState globalSimState_{};
		Sync::UniversalSimState localSimState_{};
		DistParticleState fluidParticleState_{};
		DistParticleState boundaryParticleState_{};

		// backend over compact rank-local state
		std::unique_ptr<Bench::BenchmarksTracker> benchmarksTracker_{ nullptr };
		std::unique_ptr<Mem::MemoryTracker> memoryTracker_{ nullptr };
		std::vector<Bench::BenchOpGroupDef> curBenchOpGroups_;
		std::unique_ptr<Sim::I_SimBackend> simBackend_{ nullptr };

		// serialization management
		Export::SerializationMgr serializationMgr_{};
		mutable Export::LoggingMgr loggingMgr_;
		mutable Sync::ExportSnapshot exportSnapScratch_{};

		std::atomic<bool> running_{ false };
		std::atomic<bool> stopRequested_{ false };
		std::atomic<bool> initialized_{ false };
	};

} // namespace Dist
