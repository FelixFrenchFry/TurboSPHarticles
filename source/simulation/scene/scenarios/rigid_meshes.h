// rigid_meshes.h
#pragma once

// external
#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

// internal
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/sync/state_sim.h"
#include "source/simulation/scene/builder.h"
#include "source/simulation/scene/rigid_mesh_helpers.h"
#include "source/simulation/scene/scenario_sim.h"
#include "source/simulation/scene/scenario_json.h"
#include "source/simulation/scene/mesh_scenario_loader.h"



namespace Scene
{
	struct RigidMeshesConfig
	{
		// outer boundary container
		OBB container;

		// fluid block placed inside
		OBB fluid;

		// independently simulated sampled mesh bodies
		std::vector<RigidMeshEntryConfig> rigidBodies;
	};

	class RigidMeshesScene final : public ISimScenario
	{
	public:

		RigidMeshesScene(ScenarioMgr& scenarioMgr, RigidMeshesConfig cfg)
			: scenarioMgr_(scenarioMgr), cfg_(std::move(cfg)) {}

		const char* Name() const override { return "Rigid Meshes"; }

		void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const override
		{
			scenarioMgr_.BuildBoundaryContainer_OBB(cfg_.container, dst, params);

			scenarioMgr_.BuildFluidBlock_OBB(cfg_.fluid, dst, params);

			for (const RigidMeshEntryConfig& rigidMesh : cfg_.rigidBodies)
			{
				BuildRigidMesh(scenarioMgr_, rigidMesh, dst, params);
			}
		}

		std::vector<OBB> GetBoundaryBoxes() const override
		{
			return { cfg_.container };
		}

	private:

		ScenarioMgr& scenarioMgr_;
		RigidMeshesConfig cfg_;
	};

	inline RigidMeshesConfig LoadRigidMeshesConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		ValidateScenarioType(j, "RigidMeshes");
		RigidMeshesConfig cfg{};

		cfg.container = ScenarioMgr::ParseOBBFromJson(
			ReadScenarioObject(j, "container", "RigidMeshes"), scenarioMgr, "RigidMeshes.container");
		cfg.fluid = ScenarioMgr::ParseOBBFromJson(
			ReadScenarioObject(j, "fluid", "RigidMeshes"), scenarioMgr, "RigidMeshes.fluid");

		cfg.rigidBodies = LoadRigidMeshEntriesFromJson(j, "RigidMeshes");
		ValidateEmptyScenarioArray(j, "boundaryBodies", "RigidMeshes");

		// sanitize fluid lengths
		cfg.fluid.uLen = std::min(std::max(0.0f, cfg.fluid.uLen), cfg.container.uLen);
		cfg.fluid.vLen = std::min(std::max(0.0f, cfg.fluid.vLen), cfg.container.vLen);
		cfg.fluid.wLen = std::min(std::max(0.0f, cfg.fluid.wLen), cfg.container.wLen);

		return cfg;
	}

	inline RigidMeshesConfig LoadRigidMeshesConfigFromProfile(
		const std::string& profile, ScenarioMgr& scenarioMgr)
	{
		return LoadMeshScenarioConfigFromProfile<RigidMeshesConfig>(profile, "RigidMeshes",
			[&](const nlohmann::json& j)
			{
				RigidMeshesConfig cfg = LoadRigidMeshesConfigFromJson(j, scenarioMgr);
				ResolveRigidMeshModels(cfg.rigidBodies, "RigidMeshes");
				return cfg;
			});
	}

} // namespace Scene
