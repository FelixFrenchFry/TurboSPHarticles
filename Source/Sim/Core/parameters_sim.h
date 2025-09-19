// parameters_sim.h
//
// TODO: comment

#pragma once

// external
#include <atomic>

// internal



namespace Sim
{
	// snapshot of simulation parameters (passed to compute-heavy functions)
	struct SimParams
	{
		// TODO: add more parameters as needed

		// simulation
		std::uint64_t step		= 0;

		// time
		float timeStep			= 0.01f;
		float timeStepInv		= 100.0f;
		float timeStepSqu		= 0.0001f;

		// particles
		float particleSize		= 1.0f;
		float massFluid			= 0.2f;
		float massBound			= 0.2f;

		// kernel
		float kernelRadius		= 2.0f;
		float kernelRadiusSqu	= 4.0f;

		// density
		float restDensity		= 1'000.0f;
		float restDensityInv	= 0.001f;
		float restDensitySqu	= 1'000'000.0f;

		// viscosity
		float viscosityFluid	= 0.1f;
		float viscosityBound	= 0.1f;

		// gravity
		float gravity[3]		= { 0.0f, -9.81f, 0.0f };

		// flags

		// misc
	};

	class SimParamsMgr
	{
	public: // ==============================

		SimParamsMgr() = default;

		// returns snapshot and version of current simulation parameters
		std::pair<SimParams, uint64_t> SnapshotParams() const;

		// returns current stable version number to check for changes
		uint64_t StableVersion() const;

		// setters for individual parameters and their dependent values
		void Set_TimeStep(float newTimeStep);
		void Set_ParticleSize(float newParticleSize);
		void Set_MassFluid(float newMassFluid);
		void Set_MassBound(float newMassBound);
		void Set_KernelRadius(float newKernelRadius);
		void Set_RestDensity(float newRestDensity);
		void Set_ViscosityFluid(float newViscosityFluid);
		void Set_ViscosityBound(float newViscosityBound);

	private: // ==============================

		alignas(64) SimParams params_{}; // current active copy of visualization parameters
		alignas(64) std::atomic<uint64_t> version_{ 0 }; // even = stable, odd = writing

		inline void BeginWrite(); // mark beginning of write operation
		inline void EndWrite(); // mark end of write operation
	};

} // namespace Sim
