// dist_simulation.h

#pragma once

// external
#include <memory>
#include <string>
#include <vector>

// internal
#include "Source/Sim/Core/parameters_sim.h"
#include "domain_decomp.h"
#include "mpi_context.h"



namespace Dist
{
	class DistSimulationMgr final
	{
	public: // ---------------------------------------------------------------------------------------

		DistSimulationMgr(MpiContext& mpiContext,
						  Sim::SimParamsMgr& simParamsMgr);

		// loads shared params and creates the first rank-local domain view
		bool Init(std::string& profile, const std::vector<std::string>& simOverrides);

		// temporary startup validation until the distributed step loop exists
		void RunStartupCheck() const;
		void Shutdown();

	private: // --------------------------------------------------------------------------------------

		// root reads params.json, then every rank loads the broadcast copy
		bool LoadParamsFromProfileBroadcast(std::string& profile);

		// applies the same command-line overrides on every rank
		bool LoadParamsFromCliOverrides(const std::vector<std::string>& simOverrides);

		// creates the ownership / halo decomp from the selected scenario bounds
		bool BuildDomainDecomp(const std::string& profile);

		// reads raw params.json text, parsing happens after broadcast
		static bool TryReadProfileJson(const std::string& profile, std::string& jsonText);

		MpiContext& mpiContext_;
		Sim::SimParamsMgr& simParamsMgr_;
		std::unique_ptr<IDomainDecomp> domainDecomp_{ nullptr };
		bool initialized_ = false;
	};

} // namespace Dist
