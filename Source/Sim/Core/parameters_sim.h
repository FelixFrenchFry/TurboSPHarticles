// parameters_sim.h
//
// TODO: comment

#pragma once

// external
#include <array>
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
		std::int64_t step		= 0;
		bool simRunning			= false;
		bool simRunNextStep		= false;

		// time
		float timeStep			= 0.01f;

		// particle sizes
		float particleSizeF		= 0.1f;
		float particleSizeB		= 0.1f;
		float particleRadiusF	= 0.05f;
		float particleRadiusB	= 0.05f;

		// kernel
		float kernelRadius		= 0.2f;

		// density and mass
		float restDensityF		= 1'000.0f;
		float restDensityB		= 1'000.0f;
		float massF				= 100.0f;
		float massB				= 100.0f;

		// pressure
		float taitExponent		= 5.0f;
		float stiffness			= 100.0f;

		// viscosity
		float viscosityF		= 0.1f;
		float viscosityB		= 0.1f;

		// gravity
		glm::vec3 gravity		= { 0.0f, -9.81f, 0.0f };

		// fluid conditions
		float jitFracF			= 0.0f; // % of particle size
		std::int64_t jitSeedF	= 42; // RNG seed for jitter

		// boundary conditions
		float jitFracB			= 0.0f; // % of particle size
		std::int64_t jitSeedB	= 42; // RNG seed for jitter
		int boundaryLayers		= 2;
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
		void Set_TimeStep(float timeStep);
		void Set_ParticleSize(float particleSize, Base::ParticleType type);
		void Set_KernelRadius(float kernelRadius);
		void Set_RestDensity(float restDensity, Base::ParticleType type);
		void Set_Viscosity(float viscosity, Base::ParticleType type);
		void Set_Gravity(glm::vec3 gravity);
		void Set_BoundaryLayers(int boundaryLayers);

	private: // ==============================

		alignas(64) SimParams params_{}; // current active copy of visualization parameters
		alignas(64) std::atomic<int64_t> version_{ 0 }; // even = stable, odd = writing

		inline void BeginWrite(); // mark beginning of write operation
		inline void EndWrite(); // mark end of write operation
	};

} // namespace Sim
