// distributed_simulation.h

#pragma once

// external
#include <string>
#include <vector>

// internal
#include "Source/Sim/Core/parameters_sim.h"
#include "mpi_context.h"



namespace Dist
{
	class DistributedSimulation final
	{
	public: // ---------------------------------------------------------------------------------------

		DistributedSimulation(MpiContext& mpiContext,
							  Sim::SimParamsMgr& simParamsMgr);

		bool Init(std::string& profile, const std::vector<std::string>& simOverrides);
		void RunStartupCheck() const;
		void Shutdown();

	private: // --------------------------------------------------------------------------------------

		bool LoadParamsFromProfileBroadcast(std::string& profile);
		bool LoadParamsFromCliOverrides(const std::vector<std::string>& simOverrides);

		static bool TryReadProfileJson(const std::string& profile, std::string& jsonText);

		MpiContext& mpiContext_;
		Sim::SimParamsMgr& simParamsMgr_;
		bool initialized_ = false;
	};

} // namespace Dist
