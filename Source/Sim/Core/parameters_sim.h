// parameters_sim.h
//
// TODO: comment

#pragma once

// external
#include <atomic>
#include <glm/vec3.hpp>

// internal
#include "Source/Base/enum_hub.h"



namespace Sim
{
	// snapshot of simulation parameters (passed to compute-heavy functions)
	struct SimParams
	{
		// TODO: add more parameters as needed

		// simulation
		bool simRunning			= false;
		bool simRunNextStep		= false;

		// particles
		uint32_t N_F			= 0;
		uint32_t N_B			= 0;

		// time
		float timeStep			= 0.0001f;

		// particle sizes
		// (size of particles is treated as smoothing length and spacing)
		float particleSizeF		= 0.01f;
		float particleSizeB		= 0.01f;
		float particleRadiusF	= 0.005f;
		float particleRadiusB	= 0.005f;

		// density and mass
		float restDensityF		= 1'000.0f;
		float restDensityB		= 1'000.0f;
		float massF				= 0.001f;
		float massB				= 0.001f;

		// pressure
		float gammaPress		= 0.5f;
		float jacobiOmega		= 0.6f;

		// viscosity
		float viscosityF		= 0.000003f;
		float viscosityB		= 0.000003f;

		float taitExponent		= 7.0f; // TODO: to be removed
		float stiffness			= 57000.0f; // TODO: to be removed

		// gravity
		glm::vec3 gravity		= { 0.0f, -9.81f, 0.0f };

		// kernel
		float kernelSuppRatio	= 2.0f;
		float kernelRadius		= 0.02f; // radius = size * supp ratio

		// fluid conditions
		float jitterFracF		= 0.0f; // % of particle size
		int64_t jitterSeedF		= 42; // RNG seed for jitter

		// boundary conditions
		float jitterFracB		= 0.0f; // % of particle size
		int64_t jitterSeedB		= 42; // RNG seed for jitter
		int boundaryLayers		= 2;

		// statistics
		int64_t maxStatRecCount = 100'000'000;

		// benchmarks
		int64_t maxBenchRecCount = 100'000'000;
	};

	class SimParamsMgr
	{
	public: // ==============================

		SimParamsMgr() = default;

		// returns snapshot and version of current simulation parameters
		std::pair<SimParams, int64_t> SnapshotParams() const;

		// returns current stable version number to check for changes
		int64_t StableVersion() const;

		// setters for individual parameters and their dependent values
		void Set_SimRunning(bool simRunning);
		void Set_SimRunNextStep(bool simRunNextStep);
		void Set_NumParticles(uint32_t N, Base::ParticleType type);
		void Set_TimeStep(float timeStep);
		void Set_ParticleSize(float particleSize, Base::ParticleType type);
		void Set_RestDensity(float restDensity, Base::ParticleType type);
		void Set_Mass(float mass, Base::ParticleType type);
		void Set_GammaPress(float gammaPress);
		void Set_JacobiOmega(float jacobiOmega);
		void Set_Viscosity(float viscosity, Base::ParticleType type);
		void Set_Gravity(glm::vec3 gravity);
		void Set_KernelSuppRatio(float kernelSuppRatio);
		void Set_KernelRadius(float kernelRadius);
		void Set_BoundaryLayers(int boundaryLayers);

		bool AssertParamDerivationCorrectness();

		// different parameter presets for quick testing
		void PresetParams_A();
		void PresetParams_B();

	private: // ==============================

		alignas(64) SimParams params_{}; // current active copy of visualization parameters
		alignas(64) std::atomic<int64_t> version_{ 0 }; // even = stable, odd = writing

		inline void BeginWrite(); // mark beginning of write operation
		inline void EndWrite(); // mark end of write operation
	};

} // namespace Sim
