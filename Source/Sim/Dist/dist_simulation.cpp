// dist_simulation.cpp

// external
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>

// internal
#include "Source/Base/enum_hub.h"
#include "Source/Sim/Scene/scenario_sim.h"
#include "dist_simulation.h"
#include "Layouts/decomp_AABB_uniform.h"



using namespace Dist;

namespace
{
	void IncludePoint(AABB& box, const glm::vec3& point)
	{
		box.min.x = std::min(box.min.x, point.x);
		box.min.y = std::min(box.min.y, point.y);
		box.min.z = std::min(box.min.z, point.z);

		box.max.x = std::max(box.max.x, point.x);
		box.max.y = std::max(box.max.y, point.y);
		box.max.z = std::max(box.max.z, point.z);
	}

	// scenarios use OBBs, while decomp starts from one global AABB
	AABB ComputeAABBFromOBB(const Scene::OBB& box)
	{
		const glm::vec3 u = box.uHat * box.uLen;
		const glm::vec3 v = box.vHat * box.vLen;
		const glm::vec3 w = box.wHat * box.wLen;

		const std::array<glm::vec3, 8> corners =
		{
			box.origin,
			box.origin + u,
			box.origin + v,
			box.origin + w,
			box.origin + u + v,
			box.origin + u + w,
			box.origin + v + w,
			box.origin + u + v + w
		};

		AABB boxAABB
		{
			.min = corners[0],
			.max = corners[0]
		};

		for (const glm::vec3& corner : corners)
		{
			IncludePoint(boxAABB, corner);
		}

		return boxAABB;
	}

	// build the global domain AABB that will be split between MPI ranks
	AABB ComputeGlobalDomainBounds(
		const std::vector<Scene::OBB>& boundaryBoxes,
		const Sim::SimParams& params)
	{
		if (boundaryBoxes.empty())
		{
			throw std::runtime_error("Scenario does not provide boundary boxes.");
		}

		AABB globalBounds = ComputeAABBFromOBB(boundaryBoxes.front());

		for (uint64_t i = 1; i < boundaryBoxes.size(); i++)
		{
			const AABB boxBounds = ComputeAABBFromOBB(boundaryBoxes[i]);
			IncludePoint(globalBounds, boxBounds.min);
			IncludePoint(globalBounds, boxBounds.max);
		}

		const float boundaryPad = params.boundaryLayers > 0
			? (static_cast<float>(params.boundaryLayers) - 0.5f) * params.particleSizeB
			: 0.0f;

		return globalBounds.Expanded(boundaryPad);
	}

	std::string FormatVec3(const glm::vec3& v)
	{
		std::ostringstream oss;
		oss << '(' << v.x << ", " << v.y << ", " << v.z << ')';
		return oss.str();
	}

	std::string FormatAABB(const AABB& box)
	{
		std::ostringstream oss;
		oss << '[' << FormatVec3(box.min) << " -> " << FormatVec3(box.max) << ']';
		return oss.str();
	}

	std::string FormatNeighborRanks(const std::vector<int>& neighborRanks)
	{
		std::ostringstream oss;
		oss << '[';
		for (uint64_t i = 0; i < neighborRanks.size(); i++)
		{
			if (i > 0) { oss << ", "; }
			oss << neighborRanks[i];
		}
		oss << ']';
		return oss.str();
	}
}

DistSimulationMgr::DistSimulationMgr(
	MpiContext& mpiContext,
	Sim::SimParamsMgr& simParamsMgr)
	: mpiContext_{ mpiContext }
	, simParamsMgr_{ simParamsMgr }
	, domainDecomp_{ std::make_unique<Decomp_AABB_Uniform>() }
{
}

bool DistSimulationMgr::Init(
	std::string& profile,
	const std::vector<std::string>& simOverrides)
{
	if (!LoadParamsFromProfileBroadcast(profile))
	{
		return false;
	}

	configProfile_ = profile;
	SyncParameters(true);

	if (!LoadParamsFromCliOverrides(simOverrides))
	{
		return false;
	}
	SyncParameters(true);

	if (mpiContext_.IsRoot())
	{
		SPDLOG_INFO("Initializing distributed simulation state...");
	}
	SelectSimScenario(simParams_.scenarioType, configProfile_);
	if (!mpiContext_.AllTrue(simScenario_ != nullptr)) { return false; }

	if (mpiContext_.IsRoot())
	{
		SPDLOG_INFO("Initializing distributed domain decomp...");
	}
	if (!BuildDomainDecomp()) { return false; }

	if (mpiContext_.IsRoot())
	{
		SPDLOG_INFO("Initializing rank-local distributed state...");
	}
	if (!ReBuildInitialSimState()) { return false; }

	initialized_ = true;
	SyncParameters(true);
	return true;
}

void DistSimulationMgr::Shutdown()
{
	initialized_ = false;
}

void DistSimulationMgr::SyncParameters(const bool force)
{
	uint64_t latestSimParamsVersion = simParamsMgr_.StableVersion();
	if (latestSimParamsVersion != simParamsVersion_ || force)
	{
		auto snapshot = simParamsMgr_.SnapshotParams();

		simParams_ = std::get<0>(snapshot);
		simParamsVersion_ = std::get<1>(snapshot);
	}
}

void DistSimulationMgr::SelectSimScenario(
	const Base::ScenarioType type,
	const std::string& profile)
{
	// create new scenario
	simScenario_ = Scene::MakeSimScenario(scenarioMgr_, type, profile);
	if (!simScenario_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to create sim scenario ({})",
			mpiContext_.Rank(),
			mpiContext_.Size(),
			static_cast<int>(type));
		return;
	}
}

bool DistSimulationMgr::ReBuildInitialSimState()
{
	bool localOk = true;

	try
	{
		if (!simScenario_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed to re-build initial sim state (no scenario selected)",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		if (!domainDecomp_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed to re-build initial sim state (no domain decomp)",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}

		if (localOk)
		{
			SyncParameters();

			// reset scenario and all its rank-local data
			localSimState_ = {};
			fluidParticleState_.Clear();
			boundaryParticleState_.Clear();

			Sync::UniversalSimState globalSimState{};
			globalSimState.schemaVersion = 1;
			globalSimState.step = 0;
			globalSimState.time = 0.0;

			// all ranks build the same global scenario, then keep owned + halo particles
			simScenario_->Build(globalSimState, simParams_);
			globalSimState.ShrinkToFitAll();

			BuildLocalParticlesFromGlobalState(
				localSimState_,
				fluidParticleState_,
				boundaryParticleState_,
				globalSimState,
				*domainDecomp_,
				mpiContext_.Rank(),
				simParams_.kernelRadius);

			localSimState_.ShrinkToFitAll();
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to re-build initial sim state: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	return mpiContext_.AllTrue(localOk);
}

void DistSimulationMgr::RunStartupCheck() const
{
	if (!initialized_)
	{
		SPDLOG_ERROR("DistSimulationMgr was not initialized.");
		return;
	}

	const Sim::SimParams params = simParams_;

	if (mpiContext_.IsRoot() && domainDecomp_)
	{
		SPDLOG_INFO(
			"Distributed domain decomp initialized "
			"(method={}, globalBounds={})",
			domainDecomp_->Name(),
			FormatAABB(domainDecomp_->GlobalExtentAABB()));
	}

	// print one rank at a time to keep MPI startup logs readable
	for (int r = 0; r < mpiContext_.Size(); r++)
	{
		mpiContext_.Barrier();
		if (mpiContext_.Rank() == r)
		{
			SPDLOG_INFO(
				"[rank {}/{}] Distributed startup validated "
				"(backend={}, neighborListStorage={}, maxSteps={}, exportDir='{}')",
				mpiContext_.Rank(),
				mpiContext_.Size(),
				Base::ToCString(params.backendVersion),
				Base::ToEffectiveNeighborListStorageCString(
					params.backendVersion,
					params.neighborListMethod),
				params.maxSteps,
				params.exportDir);

			if (domainDecomp_)
			{
				SPDLOG_INFO(
					"[rank {}/{}] Domain "
					"(owned={}, halo={}, neighbors={})",
					mpiContext_.Rank(),
					mpiContext_.Size(),
					FormatAABB(domainDecomp_->OwnedExtentAABB(mpiContext_.Rank())),
					FormatAABB(domainDecomp_->HaloExtentAABB(mpiContext_.Rank())),
					FormatNeighborRanks(domainDecomp_->NeighborRanks(mpiContext_.Rank())));
			}

			SPDLOG_INFO(
				"[rank {}/{}] Local particles "
				"(fluid owned={}, inner={}, outer={}, halo={}, "
				"boundary owned={}, inner={}, outer={}, halo={})",
				mpiContext_.Rank(),
				mpiContext_.Size(),
				fluidParticleState_.particleLocalIdx_Owned.size(),
				fluidParticleState_.particleLocalIdx_Inner.size(),
				fluidParticleState_.particleLocalIdx_Outer.size(),
				fluidParticleState_.particleLocalIdx_Halo.size(),
				boundaryParticleState_.particleLocalIdx_Owned.size(),
				boundaryParticleState_.particleLocalIdx_Inner.size(),
				boundaryParticleState_.particleLocalIdx_Outer.size(),
				boundaryParticleState_.particleLocalIdx_Halo.size());

			spdlog::default_logger()->flush();
		}
	}
	mpiContext_.Barrier();

	if (mpiContext_.IsRoot())
	{
		SPDLOG_WARN("Distributed simulation stepping is not implemented yet; "
			"exiting after MPI startup validation.");
	}
}

bool DistSimulationMgr::LoadParamsFromProfileBroadcast(std::string& profile)
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

bool DistSimulationMgr::LoadParamsFromCliOverrides(
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

bool DistSimulationMgr::BuildDomainDecomp()
{
	bool localOk = true;

	try
	{
		if (!simScenario_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed to build distributed domain decomp (no scenario selected).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		else
		{
			const AABB globalBounds =
				ComputeGlobalDomainBounds(simScenario_->GetBoundaryBoxes(), simParams_);

			// kernel radius defines initial halo thickness for ownership queries
			const DomainDecompInput input
			{
				.globalBounds = globalBounds,
				.haloRadius = simParams_.kernelRadius,
				.rankCount = mpiContext_.Size()
			};

			domainDecomp_->Build(input);

			if (!domainDecomp_->IsValidForRankCount(mpiContext_.Size()))
			{
				SPDLOG_ERROR("[rank {}/{}] Invalid distributed domain decomp.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
			}
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to build distributed domain decomp: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	return mpiContext_.AllTrue(localOk);
}

bool DistSimulationMgr::TryReadProfileJson(
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
