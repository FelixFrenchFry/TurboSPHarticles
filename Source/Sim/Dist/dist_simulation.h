// dist_simulation.h

#pragma once

// external
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// internal
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Scene/scenario_sim.h"
#include "domain_decomp.h"
#include "dist_particle_state.h"
#include "mpi_context.h"



namespace Dist
{
	class DistSimulationMgr final
	{
	public: // ---------------------------------------------------------------------------------------

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

		bool ReBuildInitialSimState();

		// temporary startup validation until the distributed step loop exists
		void RunStartupCheck() const;

	private: // --------------------------------------------------------------------------------------

		// root reads params.json, then every rank loads the broadcast copy
		bool LoadParamsFromProfileBroadcast(std::string& profile);

		// applies the same command-line overrides on every rank
		bool LoadParamsFromCliOverrides(const std::vector<std::string>& simOverrides);

		// creates the ownership / halo decomp from the selected scenario bounds
		bool BuildDomainDecomp();

		// reads raw params.json text, parsing happens after broadcast
		static bool TryReadProfileJson(const std::string& profile, std::string& jsonText);

		MpiContext& mpiContext_;

		// parameter manager and snapshot to be managed and passed to sub-systems
		Sim::SimParamsMgr& simParamsMgr_;
		Sim::SimParams simParams_;
		uint64_t simParamsVersion_ = 0;

		// configuration profile
		std::string configProfile_ = "Default";

		// scenario management
		Scene::ScenarioMgr scenarioMgr_;
		std::unique_ptr<Scene::ISimScenario> simScenario_{ nullptr };

		// distributed domain and rank-local simulation state
		std::unique_ptr<IDomainDecomp> domainDecomp_{ nullptr };
		Sync::UniversalSimState localSimState_{};
		DistParticleState fluidParticleState_{};
		DistParticleState boundaryParticleState_{};

		bool initialized_ = false;
	};

} // namespace Dist
