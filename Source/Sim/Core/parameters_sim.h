// parameters_sim.h
//
// TODO: comment

#pragma once

// external
#include <ankerl/unordered_dense.h>
#include <atomic>
#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>
#include <string>

// internal
#include "Source/Base/enum_hub.h"
#include "Source/Base/helpers.h"
#include "Source/Eval/Stats/metrics.h"



namespace Sim
{
	// snapshot of simulation parameters (passed to compute-heavy functions)
	struct SimParams
	{
		// modes
		Base::BackendVersion backendVersion =
			Base::BackendVersion::CPU01;
		Base::ScenarioType scenarioType	=
			Base::ScenarioType::BreakingDam;
		Base::NeighborSearchMethod neighborSearchMethod =
			Base::NeighborSearchMethod::CellLinkedList_Compact;
		Base::NeighborListStorage neighborListStorage =
			Base::NeighborListStorage::CSRVector_FinalConcat;

		// mode flags
		bool runCountQuery		= false; // depends on the neighbor list storage
		bool runFillQuery		= true; // always true for now

		// simulation state
		bool simRunning			= false;
		bool simRunNextStep		= false;
		int64_t maxSteps		= -1;

		// time
		float timeStep			= 0.0001f;

		// particle sizes
		// (is treated as smoothing length and spacing "h")
		float particleSizeF		= 0.01f;
		float particleSizeB		= 0.01f;
		float particleSizeInvF	= 100.0f;
		float particleSizeInvB	= 100.0f;

		// kernel
		float kernelSuppRatio	= 2.0f;
		float kernelRadius		= 0.02f; // radius = size * supp ratio
		float kernelRadiusSqu	= 0.0004f;

		// neighbor count heuristics
		// TODO: use p95 and p99 values for heuristics?
		uint32_t expctdNbrCountSoftMax_FF = 40;
		uint32_t expctdNbrCountSoftMax_FB = 20;
		uint32_t expctdNbrCountHardMax_FF = 64;
		uint32_t expctdNbrCountHardMax_FB = 32;

		// density and mass
		float restDensityF		= 1'000.0f;
		float restDensityB		= 1'000.0f;
		float massF				= 0.001f;
		float massB				= 0.001f;

		// non-pressure accelerations
		float viscosityF		= 0.000003f;
		float viscosityB		= 0.000003f;
		glm::vec3 gravity		= { 0.0f, -9.81f, 0.0f };

		// pressure solving
		float pressGamma		= 0.5f;
		float jacobiOmega		= 0.6f;
		float densityDevTarget	= 0.01f;
		uint32_t fixedIters		= 10;
		uint32_t minSolverIters	= 5;
		uint32_t maxSolverIters	= 50;
		Base::SolverTerminationCond solverTerminationCond =
			Base::SolverTerminationCond::FixedIters;

		// data structuring
		uint32_t dataReOrderInterval = 50;

		// fluid conditions
		float jitterFracF		= 0.0f; // % of particle size
		int64_t jitterSeedF		= 42; // RNG seed for jitter

		// boundary conditions
		float jitterFracB		= 0.0f; // % of particle size
		int64_t jitterSeedB		= 42; // RNG seed for jitter
		uint32_t boundaryLayers	= 2;

		// benchmarks
		uint32_t maxBenchRecs = 1'000'000;
		Bench::TimeUnit timeUnit = Bench::TimeUnit::ms;

		// memory
		uint32_t maxMemRecs	= 1'000'000;
		Mem::ByteUnit byteUnit = Mem::ByteUnit::MiB;

		// statistics
		uint32_t maxStatRecs = 1'000'000;
		std::array<bool, static_cast<uint8_t>(Stats::StatGroup::Count)> statGroupsEnabled =
		{
			false,	// particles
			false,	// density
			false	// velocity
		};

		// export
		bool exportEnabled		= true;
		std::string exportDir	= "A";
		uint32_t exportInterval	= 10;
	};

	class SimParamsMgr
	{
	public: // -----------------------------------------------------------------

		SimParamsMgr() = default;

		// returns snapshot and version of current simulation parameters
		std::pair<SimParams, uint64_t> SnapshotParams() const;

		// returns current stable version number to check for changes
		uint64_t StableVersion() const;

		bool LoadSimParamsFromJson(const nlohmann::json& j, bool verbose = false);
		bool LoadSimParamsFromProfile(const std::string& profile, bool verbose = false);

		// setters for individual parameters and their dependent values
		void Set_RunCountQuery(bool runCountQuery, bool verbose = false);
		void Set_RunFillQuery(bool runFillQuery, bool verbose = false);
		void Set_SimRunning(bool simRunning, bool verbose = false);
		void Set_SimRunNextStep(bool simRunNextStep, bool verbose = false);
		void Set_TimeStep(float timeStep, bool verbose = false);
		void Set_ParticleSize(float particleSize, Base::ParticleType type, bool verbose = false);
		void Set_KernelSuppRatio(float kernelSuppRatio, bool verbose = false);
		void Set_KernelRadius(float kernelRadius, bool verbose = false);
		void Set_RestDensity(float restDensity, Base::ParticleType type, bool verbose = false);
		void Set_Mass(float mass, Base::ParticleType type, bool verbose = false);
		void Set_Viscosity(float viscosity, Base::ParticleType type, bool verbose = false);
		void Set_Gravity(glm::vec3 gravity, bool verbose = false);
		void Set_PressGamma(float pressGamma, bool verbose = false);
		void Set_JacobiOmega(float jacobiOmega, bool verbose = false);
		void Set_DensityDevTarget(float densityDevTarget, bool verbose = false);
		void Set_MinSolverIters(uint32_t minSolverIters, bool verbose = false);
		void Set_MaxSolverIters(uint32_t maxSolverIters, bool verbose = false);
		void Set_DataReOrderInterval(uint32_t dataReOrderInterval, bool verbose = false);
		void Set_BoundaryLayers(uint32_t boundaryLayers, bool verbose = false);

		// helpers
		bool AssertParamDerivationCorrectness() const;
		uint32_t HeuristicExpectedNeighborCount_SoftMax(
			float kernelSuppRatio, Base::ParticleType type) const;
		uint32_t HeuristicExpectedNeighborCount_HardMax(
			float kernelSuppRatio, Base::ParticleType type) const;

	private: // ----------------------------------------------------------------

		alignas(64) SimParams params_{}; // current active copy of visualization parameters
		alignas(64) std::atomic<uint64_t> version_{ 0 }; // even = stable, odd = writing

		inline void BeginWrite(); // mark beginning of write operation
		inline void EndWrite(); // mark end of write operation
	};

} // namespace Sim
