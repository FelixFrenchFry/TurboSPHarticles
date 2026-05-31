// distributed_simulation.cpp

// external
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

// internal
#include "Source/Base/enum_hub.h"
#include "distributed_simulation.h"



using namespace Dist;

DistributedSimulation::DistributedSimulation(
	MpiContext& mpiContext,
	Sim::SimParamsMgr& simParamsMgr)
	: mpiContext_(mpiContext),
	  simParamsMgr_(simParamsMgr)
{
}

bool DistributedSimulation::Init(
	std::string& profile,
	const std::vector<std::string>& simOverrides)
{
	if (!LoadParamsFromProfileBroadcast(profile))
	{
		return false;
	}

	if (!LoadParamsFromCliOverrides(simOverrides))
	{
		return false;
	}

	initialized_ = true;
	return true;
}

void DistributedSimulation::RunStartupCheck() const
{
	if (!initialized_)
	{
		SPDLOG_ERROR("DistributedSimulation was not initialized.");
		return;
	}

	const Sim::SimParams params = simParamsMgr_.SnapshotParams().first;

	// print one rank at a time to keep MPI startup logs readable
	for (int r = 0; r < mpiContext_.Size(); r++)
	{
		mpiContext_.Barrier();
		if (mpiContext_.Rank() == r)
		{
			SPDLOG_INFO(
				"[rank {}/{}] Distributed startup validated "
				"(backend={}, maxSteps={}, exportDir='{}')",
				mpiContext_.Rank(),
				mpiContext_.Size(),
				Base::ToCString(params.backendVersion),
				params.maxSteps,
				params.exportDir);
		}
	}
	mpiContext_.Barrier();

	if (mpiContext_.IsRoot())
	{
		SPDLOG_WARN("Distributed simulation stepping is not implemented yet; "
			"exiting after MPI startup validation.");
	}
}

void DistributedSimulation::Shutdown()
{
	initialized_ = false;
}

bool DistributedSimulation::LoadParamsFromProfileBroadcast(std::string& profile)
{
	std::string jsonText{};
	int loadOk = 1;

	if (mpiContext_.IsRoot())
	{
		if (!TryReadProfileJson(profile, jsonText))
		{
			if (profile != "Default")
			{
				SPDLOG_WARN("Failed to load '/{}/params.json'. Falling back to '/Default/params.json'.",
					profile);
				profile = "Default";
			}

			if (!TryReadProfileJson(profile, jsonText))
			{
				SPDLOG_ERROR("Failed to load '/{}/params.json'.", profile);
				loadOk = 0;
			}
		}
	}

	mpiContext_.BroadcastInt(loadOk);
	mpiContext_.BroadcastString(profile);
	if (loadOk == 0) { return false; }

	mpiContext_.BroadcastString(jsonText);

	bool localOk = true;
	try
	{
		const nlohmann::json j = nlohmann::json::parse(jsonText);
		localOk = simParamsMgr_.LoadSimParamsFromJson(j);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to parse broadcast params JSON: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	const bool globalOk = mpiContext_.AllTrue(localOk);
	if (globalOk && mpiContext_.IsRoot())
	{
		SPDLOG_INFO("Using config profile '{}'", profile);
	}

	return globalOk;
}

bool DistributedSimulation::LoadParamsFromCliOverrides(
	const std::vector<std::string>& simOverrides)
{
	bool localOk = true;
	if (!simOverrides.empty())
	{
		localOk = simParamsMgr_.LoadSimParamsFromCliOverrides(simOverrides, mpiContext_.IsRoot());
	}

	const bool globalOk = mpiContext_.AllTrue(localOk);
	if (globalOk && mpiContext_.IsRoot() && !simOverrides.empty())
	{
		SPDLOG_INFO("Applied {} simulation override(s) from command line",
			simOverrides.size());
	}

	return globalOk;
}

bool DistributedSimulation::TryReadProfileJson(
	const std::string& profile,
	std::string& jsonText)
{
	const std::filesystem::path paramsPath =
		std::filesystem::path(TSP_CONFIG_DIR) / profile / "params.json";

	std::ifstream file(paramsPath, std::ios::in | std::ios::binary);
	if (!file) { return false; }

	jsonText.assign(
		std::istreambuf_iterator<char>(file),
		std::istreambuf_iterator<char>());
	return true;
}
