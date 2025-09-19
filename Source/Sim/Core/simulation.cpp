// simulation.cpp

// external

// internal
#include "simulation.h"



using namespace Sim;

SimulationMgr::SimulationMgr(Sim::SimParamsMgr& simParamsMgr)
	: simParamsMgr_{ simParamsMgr }
{
	// TODO: implement
}

bool SimulationMgr::Init()
{
	// initial snapshot
	SyncParameters(true);

	return true;
}


void SimulationMgr::SyncParameters(bool force)
{
	uint64_t latestSimParamsVersion = simParamsMgr_.StableVersion();
	if (latestSimParamsVersion != simParamsVersion_ ||
		force)
	{
		auto snapshot = simParamsMgr_.SnapshotParams();

		// TODO: remove debug log
		// SPDLOG_INFO("Synced 'simParams' snapshot from version {} to {}",
		// simParamsVersion_, std::get<1>(snapshot));

		simParams_ = std::get<0>(snapshot);
		simParamsVersion_ = std::get<1>(snapshot);
	}
}
