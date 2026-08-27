// dist_simulation.h

#pragma once

// external
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// internal
#include "source/eval/bench/benchmarks.h"
#include "source/eval/memory/memory.h"
#include "source/eval/serial/metrics_logger.h"
#include "source/eval/serial/serialization.h"
#include "source/simulation/core/backend_sim.h"
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/scene/scenario_sim.h"
#include "domain_decomp.h"
#include "dist_diagnostics.h"
#include "dist_particle_state.h"
#include "dist_snapshots.h"
#include "fluid_solver_sync.h"
#include "load_balance.h"
#include "mpi_context.h"
#include "particle_redistribution.h"
#include "rigid_body_sync.h"



namespace Dist
{
	inline constexpr float INITIAL_DOMAIN_DECOMP_PROFILE_PARTICLE_SIZE_MULTIPLIER = 8.0f;

	class DistSimulationMgr final
	{
	public:

		DistSimulationMgr(MpiContext& mpiContext, Sim::SimParamsMgr& simParamsMgr);

		// initialize sub-systems
		bool Init(std::string& profile, const std::vector<std::string>& simOverrides, bool logParamStatus = true);

		// destroy sub-systems and free up resources
		void Shutdown();

		// pulls latest parameter snapshot and version if they have changed
		void SyncParameters(bool force = false);

		// create or switch scenario
		// (destroys previous scenario and all its data)
		void SelectSimScenario(Base::ScenarioType type, const std::string& profile);

		// create backend when no backend is active
		void SelectSimBackend(Base::BackendType backendType);

		// create or switch domain decomp
		// (destroys previous domain decomp, so it has to be built again afterwards)
		void SelectDomainDecomp(Base::DomainDecompMethod method);

		bool ReBuildInitialSimState();

		// prints root-gathered distributed setup diagnostics
		void RunStartupCheck() const;

		// builds root-level startup diagnostics after initialization
		DistSimSetupReport BuildDistSimSetupReport() const;

		// main simulation loop
		bool RunSim_Blocking();

		void RequestStop();

		// metadata
		uint64_t simStep_ = 0;
		double simTime_ = 0.0;

	private:

		// helpers marked [ rank-reduced ] run their own collective before returning, so they hand back
		// the same result on every rank and callers branch on it directly instead of reducing it again

		// [ rank-reduced ] root reads params.json, then every rank loads the broadcast copy
		bool LoadParamsFromProfileBroadcast(std::string& profile, bool logStatus = true);

		// [ rank-reduced ] applies the same command-line overrides on every rank
		bool LoadParamsFromCliOverrides(const std::vector<std::string>& simOverrides, bool logStatus = true);

		// [ rank-reduced ] creates the ownership / halo decomp from the selected scenario bounds
		bool BuildDomainDecomp();

		// [ rank-reduced ] profiles a coarse probe scenario state to balance the initial domain decomposition
		bool BuildInitialDomainDecomp();

		// [ rank-reduced ] selects load metric and trigger policies for the current domain decomp
		bool ConfigureLoadBalancing();

		// builds the full scenario state before rank-local compaction
		bool BuildGlobalSimState();

		// keeps owned + halo particles for this rank and releases the replicated scenario payload
		bool BuildRankLocalSimState();

		// validates rank-local [ owned ][ halo ] state
		bool ValidateRankLocalSimState(bool includeRigidState) const;

		// [ rank-reduced ] validates that the owned particle sums still add up to the global counts established during initialization
		bool ValidateGlobalOwnedParticleCounts() const;

		// [ rank-reduced ] transfers rank-local state into the selected backend
		bool InitRankLocalBackend();

		// [ rank-reduced ] runs a backend startup smoke test on rank-local data
		bool RunRankLocalBackendStartupCheck();

		// rebuilds the fluid solver synchronization plan from the current rank-local particle state
		bool RebuildFluidSolverSyncPlan();

		// [ rank-reduced ] gathers and serializes an export snapshot on the root rank
		bool WriteExportSnapshot(const Sim::SimParams& params);

		// [ rank-reduced ] materializes, gathers, and serializes the final full state
		bool WriteFinalState(const Sim::SimParams& params, bool enabled);

		// [ rank-reduced ] rebuilds the regular CPU fluid surface selection from authoritative rank data
		bool BuildExportFluidParticleLocalIdxs(
			const Sync::FluidConstView& particlesF,
			std::vector<uint32_t>& localIdxs);

		// builds rank-local startup diagnostics gathered by the root rank
		DistStartupRankReport BuildStartupRankReport() const;

		// manages per-rank CSV metric logging
		void BeginRunLogging(const Sim::SimParams& params);
		void EndRunLogging();
		void AppendInitialMemoryMetricsLog(const Sim::SimParams& params) const;
		void AppendLatestMemoryMetricsLog() const;
		void AppendLatestBenchmarkMetricsLog() const;

		// archives step metrics and reports root-level distributed diagnostics
		void FillDistributedMemoryRecord(Mem::MemRecordStep& record) const;

		// [ rank-reduced ] reduces rank-local partial stats into the global stat record
		bool FillDistributedStatisticsRecord(const Sim::SimParams& params) const;
		void FinalizeBenchMemMeasurements(const Sim::SimParams& params) const;
		DistRankStepReport BuildRankStepReport(const Sim::SimParams& params, bool exportedStep) const;
		DistRankBenchReport BuildRankBenchReport() const;
		void DisplayMemoryUsageStats(const Sim::SimParams& params, bool force = false) const;
		void DisplayDistributedProgress(const Sim::SimParams& params, bool exportedStep) const;

		// [ rank-reduced ] runs one distributed step with backend-resident particle storage
		bool ExecuteDistributedStep(const Sim::SimParams& params);

		// [ rank-reduced ] evaluates load balancing after a completed and measured step
		bool UpdateLoadBalancing();

		// [ rank-reduced ] routes owned particles and replaces backend [ owned ][ halo ] storage
		bool RedistributeRankLocalParticles(const Sim::SimParams& params);

		// [ rank-reduced ] copies backend state into localSimState_ for full final-state gathering
		bool MaterializeRankLocalState(bool backendWillBeRetired = false);

		// [ rank-reduced ] prepares rank-authoritative particle work and backend state for the next step
		bool PrepareRankLocalBackendStep();

		// [ rank-reduced ] reduces rigid loads, integrates on root, and broadcasts body dynamics
		bool FinishRankLocalBackendStep(const Sim::SimParams& params);

		// validates particle identity metadata and rigid state against the materialized backend state
		bool ValidateRankLocalBackendParticleState() const;

		// validates particle identity metadata without comparing changed solver payload
		bool ValidateRankLocalParticleMetadata() const;

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
		std::string configProfile_ = "default";
		std::string paramsFile_ = "default/params.json";
		std::string scenariosFile_ = "default/scenarios.json";

		// scenario management
		Scene::ScenarioMgr scenarioMgr_;
		std::unique_ptr<Scene::ISimScenario> simScenario_{ nullptr };

		// distributed domain and simulation state
		std::unique_ptr<IDomainDecomp> domainDecomp_{ nullptr };
		Sync::UniversalSimState globalSimState_{};
		Sync::UniversalSimState localSimState_{};
		uint64_t N_expectedGlobalF_ = 0; // expected global fluid particle count
		uint64_t N_expectedGlobalB_ = 0; // expected global boundary particle count
		uint64_t expectedOwnedIdxChecksumF_ = 0; // checksum over the global fluid index range
		uint64_t expectedOwnedIdxChecksumB_ = 0; // checksum over the global boundary index range
		DistParticleState particleStateF_{};
		DistParticleState particleStateB_{};
		ParticleReOrderWorkBuffers fluidParticleReOrderWorkBuffers_{};
		ParticleReOrderWorkBuffers boundaryParticleReOrderWorkBuffers_{};
		ParticleRedistributionMgr particleRedistributionMgr_{};
		bool boundaryRedistributionRequired_ = true;
		FluidSolverSyncMgr fluidSolverSyncMgr_{};
		LoadBalanceMgr loadBalanceMgr_{};
		std::vector<uint32_t> pressureAccFluidParticleLocalIdxs_{};
		std::vector<uint32_t> activeRigidParticleLocalIdxs_{};
		std::vector<uint32_t> activeRigidParticleRankCountScratch_{};
		Sim::RigidBodyForceTorque rigidBodyForceTorque_{};
		RigidBodySyncBuffers rigidBodySyncBuffers_{};

		// backend over compact rank-local state
		std::unique_ptr<Bench::BenchmarksTracker> benchmarksTracker_{ nullptr };
		std::unique_ptr<Mem::MemoryTracker> memoryTracker_{ nullptr };
		std::unique_ptr<Stats::StatisticsTracker> statisticsTracker_{ nullptr };
		Bench::BenchOpList curBenchOps_;
		std::unique_ptr<Sim::I_SimBackend> simBackend_{ nullptr };
		Sim::I_DistributedParticleBackend* distParticleBackend_ = nullptr;
		mutable Stats::StatisticsPartial statisticsPartialScratch_{};
		mutable std::vector<uint8_t> localNearSurfaceFlagsScratch_{};
		mutable std::vector<uint32_t> exportFluidParticleLocalIdxsScratch_{};
		mutable DistExportSnapshotBuffer exportSnapGatherScratch_{};

		// serialization management
		Export::SerializationMgr serializationMgr_{};
		mutable Export::LoggingMgr loggingMgr_;
		mutable Sync::ExportSnapshot exportSnapScratch_{};

		std::atomic<bool> running_{ false };
		std::atomic<bool> stopRequested_{ false };
		std::atomic<bool> initialized_{ false };
	};

} // namespace Dist
