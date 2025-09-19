// simulation.h
//
// TODO: comment

#pragma once

// external
#include <span>

// internal
#include "parameters_sim.h"



namespace Sim
{
	// TODO: prototype!!!
	struct SimStateView
	{
		std::size_t N_particles{ 0 };
		std::span<const float> posX, posY, posZ;
		std::span<const float> velX, velY, velZ;
		std::span<const float> density;
		std::span<const float> pressure;

		// TODO: finish implementation
	};

	class SimulationMgr
	{
	public: // ==============================

		SimulationMgr(Sim::SimParamsMgr& simParamsMgr);

		// initialize sub-systems
		bool Init();

		// destroy contexts, sub-systems and free up resources
		void Shutdown();

		// pulls latest parameter snapshot and version if they have changed
		void SyncParameters(bool force = false);

	private: // ==============================

		// parameter manager and snapshot to be managed and passed to sub-systems
		Sim::SimParamsMgr& simParamsMgr_;
		Sim::SimParams simParams_;
		uint64_t simParamsVersion_ = 0;
	};

} // namespace Sim
