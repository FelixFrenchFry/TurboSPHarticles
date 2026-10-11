// boundary_meshes.h
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
#include "source/simulation/scene/boundary_mesh_helpers.h"
#include "source/simulation/scene/scenario_sim.h"
#include "source/simulation/scene/scenario_json.h"
#include "source/simulation/scene/mesh_scenario_loader.h"



namespace Scene
{
	struct BoundaryMeshesConfig
	{
		// outer boundary container
		OBB container;

		// fluid block placed inside
		OBB fluid;

		// static single-layer sampled mesh boundaries
		std::vector<BoundaryMeshEntryConfig> boundaryBodies;
	};

	class BoundaryMeshesScene final : public ISimScenario
	{
	public:

		BoundaryMeshesScene(ScenarioMgr& scenarioMgr, BoundaryMeshesConfig cfg)
			: scenarioMgr_(scenarioMgr), cfg_(std::move(cfg)) {}

		const char* Name() const override { return "Boundary Meshes"; }

		void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const override
		{
			scenarioMgr_.BuildBoundaryContainer_OBB(cfg_.container, dst, params);

			scenarioMgr_.BuildFluidBlock_OBB(cfg_.fluid, dst, params);

			for (const BoundaryMeshEntryConfig& boundaryMesh : cfg_.boundaryBodies)
			{
				BuildBoundaryMesh(scenarioMgr_, boundaryMesh, dst, params);
			}
		}

		std::vector<OBB> GetBoundaryBoxes() const override
		{
			return { cfg_.container };
		}

	private:

		ScenarioMgr& scenarioMgr_;
		BoundaryMeshesConfig cfg_;
	};

	inline BoundaryMeshesConfig LoadBoundaryMeshesConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		ValidateScenarioType(j, "BoundaryMeshes");
		BoundaryMeshesConfig cfg{};

		cfg.container = ScenarioMgr::ParseOBBFromJson(
			ReadScenarioObject(j, "container", "BoundaryMeshes"), scenarioMgr, "BoundaryMeshes.container");
		cfg.fluid = ScenarioMgr::ParseOBBFromJson(
			ReadScenarioObject(j, "fluid", "BoundaryMeshes"), scenarioMgr, "BoundaryMeshes.fluid");

		cfg.boundaryBodies = LoadBoundaryMeshEntriesFromJson(j, "BoundaryMeshes");
		ValidateEmptyScenarioArray(j, "rigidBodies", "BoundaryMeshes");

		// sanitize fluid lengths
		cfg.fluid.uLen = std::min(std::max(0.0f, cfg.fluid.uLen), cfg.container.uLen);
		cfg.fluid.vLen = std::min(std::max(0.0f, cfg.fluid.vLen), cfg.container.vLen);
		cfg.fluid.wLen = std::min(std::max(0.0f, cfg.fluid.wLen), cfg.container.wLen);

		return cfg;
	}

	inline BoundaryMeshesConfig LoadBoundaryMeshesConfigFromProfile(
		const std::string& profile, ScenarioMgr& scenarioMgr)
	{
		return LoadMeshScenarioConfigFromProfile<BoundaryMeshesConfig>(profile, "BoundaryMeshes",
			[&](const nlohmann::json& j)
			{
				BoundaryMeshesConfig cfg = LoadBoundaryMeshesConfigFromJson(j, scenarioMgr);
				ResolveBoundaryMeshModels(cfg.boundaryBodies, "BoundaryMeshes");
				return cfg;
			});
	}

} // namespace Scene
