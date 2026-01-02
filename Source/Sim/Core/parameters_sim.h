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
#include "Source/Eval/Stats/metrics.h"

static constexpr uint32_t N_expctNbrCountF = 42;
static constexpr uint32_t N_expctNbrCountB = 20;
static constexpr uint32_t N_expctMaxNbrCountF = 64;
static constexpr uint32_t N_expctMaxNbrCountB = 32;



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
			Base::NeighborListStorage::CompactVectors;

		// simulation state
		bool simRunning			= false;
		bool simRunNextStep		= false;
		int64_t maxSteps		= -1;

		// particles
		uint32_t N_F			= 0;
		uint32_t N_B			= 0;

		// time
		float timeStep			= 0.0001f;

		// particle sizes
		// (size of particles is treated as smoothing length and spacing)
		float particleSizeF		= 0.01f;
		float particleSizeB		= 0.01f;
		float particleSizeInvF	= 100.0f;
		float particleSizeInvB  = 100.0f;
		float particleRadiusF	= 0.005f;
		float particleRadiusB	= 0.005f;

		// kernel
		float kernelSuppRatio	= 2.0f;
		float kernelRadius		= 0.02f; // radius = size * supp ratio
		float kernelRadiusSqu	= 0.0004f;

		// density and mass
		float restDensityF		= 1'000.0f;
		float restDensityB		= 1'000.0f;
		float massF				= 0.001f;
		float massB				= 0.001f;

		// viscosity
		float viscosityF		= 0.000003f;
		float viscosityB		= 0.000003f;

		// pressure solving
		float gammaPress		= 0.5f;
		float jacobiOmega		= 0.6f;
		float densityDevTarget  = 0.01f;
		uint32_t fixedIters		= 10;
		uint32_t minSolverIters = 5;
		uint32_t maxSolverIters	= 50;
		Base::SolverTerminationCond solverTerminationCond =
			Base::SolverTerminationCond::FixedIters;

		// gravity
		glm::vec3 gravity		= { 0.0f, -9.81f, 0.0f };

		// data structuring
		uint32_t dataReOrderInterval = 50;

		// fluid conditions
		float jitterFracF		= 0.0f; // % of particle size
		int64_t jitterSeedF		= 42; // RNG seed for jitter

		// boundary conditions
		float jitterFracB		= 0.0f; // % of particle size
		int64_t jitterSeedB		= 42; // RNG seed for jitter
		uint32_t boundaryLayers	= 2;

		// statistics
		uint32_t maxStatRecs	= 1'000'000;
		std::array<bool, static_cast<uint8_t>(Stats::StatGroup::Count)> statGroupsEnabled =
		{
			false,	// particles
			false,	// density
			false	// velocity
		};

		// benchmarks
		uint32_t maxBenchRecs	= 1'000'000;

		// export
		bool exportEnabled		= true;
		std::string exportDir   = "A";
		uint32_t exportInterval = 10;
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
		void Set_SimRunning(bool simRunning, bool verbose = false);
		void Set_SimRunNextStep(bool simRunNextStep, bool verbose = false);
		void Set_NumParticles(uint32_t N, Base::ParticleType type, bool verbose = false);
		void Set_TimeStep(float timeStep, bool verbose = false);
		void Set_ParticleSize(float particleSize, Base::ParticleType type, bool verbose = false);
		void Set_KernelSuppRatio(float kernelSuppRatio, bool verbose = false);
		void Set_KernelRadius(float kernelRadius, bool verbose = false);
		void Set_RestDensity(float restDensity, Base::ParticleType type, bool verbose = false);
		void Set_Mass(float mass, Base::ParticleType type, bool verbose = false);
		void Set_Viscosity(float viscosity, Base::ParticleType type, bool verbose = false);
		void Set_GammaPress(float gammaPress, bool verbose = false);
		void Set_JacobiOmega(float jacobiOmega, bool verbose = false);
		void Set_DensityDevTarget(float densityDevTarget, bool verbose = false);
		void Set_MinSolverIters(uint32_t minSolverIters, bool verbose = false);
		void Set_MaxSolverIters(uint32_t maxSolverIters, bool verbose = false);
		void Set_Gravity(glm::vec3 gravity, bool verbose = false);
		void Set_DataReOrderInterval(uint32_t dataReOrderInterval, bool verbose = false);
		void Set_BoundaryLayers(uint32_t boundaryLayers, bool verbose = false);

		bool AssertParamDerivationCorrectness() const;

	private: // ----------------------------------------------------------------

		alignas(64) SimParams params_{}; // current active copy of visualization parameters
		alignas(64) std::atomic<uint64_t> version_{ 0 }; // even = stable, odd = writing

		inline void BeginWrite(); // mark beginning of write operation
		inline void EndWrite(); // mark end of write operation
	};

} // namespace Sim
