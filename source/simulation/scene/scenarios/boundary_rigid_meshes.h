// boundary_rigid_meshes.h
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
#include "source/simulation/scene/rigid_mesh_helpers.h"
#include "source/simulation/scene/scenario_sim.h"
#include "source/simulation/scene/scenario_json.h"
#include "source/simulation/scene/mesh_scenario_loader.h"



namespace Scene
{
	struct BoundaryRigidMeshesConfig
	{
		// outer boundary container
		OBB container;

		// fluid block placed inside
		OBB fluid;

		// static single-layer sampled mesh boundaries
		std::vector<BoundaryMeshEntryConfig> boundaryBodies;

		// independently simulated sampled mesh bodies
		std::vector<RigidMeshEntryConfig> rigidBodies;
	};

	class BoundaryRigidMeshesScene final : public ISimScenario
	{
	public:

		BoundaryRigidMeshesScene(ScenarioMgr& scenarioMgr, BoundaryRigidMeshesConfig cfg)
			: scenarioMgr_(scenarioMgr), cfg_(std::move(cfg)) {}

		const char* Name() const override { return "Boundary Rigid Meshes"; }

		void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const override
		{
			scenarioMgr_.BuildBoundaryContainer_OBB(cfg_.container, dst, params);

			scenarioMgr_.BuildFluidBlock_OBB(cfg_.fluid, dst, params);

			for (const BoundaryMeshEntryConfig& boundaryMesh : cfg_.boundaryBodies)
			{
				BuildBoundaryMesh(scenarioMgr_, boundaryMesh, dst, params);
			}

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
		BoundaryRigidMeshesConfig cfg_;
	};

	inline BoundaryRigidMeshesConfig LoadBoundaryRigidMeshesConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		ValidateScenarioType(j, "BoundaryRigidMeshes");
		BoundaryRigidMeshesConfig cfg{};

		cfg.container = ScenarioMgr::ParseOBBFromJson(
			ReadScenarioObject(j, "container", "BoundaryRigidMeshes"), scenarioMgr, "BoundaryRigidMeshes.container");
		cfg.fluid = ScenarioMgr::ParseOBBFromJson(
			ReadScenarioObject(j, "fluid", "BoundaryRigidMeshes"), scenarioMgr, "BoundaryRigidMeshes.fluid");

		cfg.boundaryBodies = LoadBoundaryMeshEntriesFromJson(j, "BoundaryRigidMeshes");
		cfg.rigidBodies = LoadRigidMeshEntriesFromJson(j, "BoundaryRigidMeshes");

		// sanitize fluid lengths
		cfg.fluid.uLen = std::min(std::max(0.0f, cfg.fluid.uLen), cfg.container.uLen);
		cfg.fluid.vLen = std::min(std::max(0.0f, cfg.fluid.vLen), cfg.container.vLen);
		cfg.fluid.wLen = std::min(std::max(0.0f, cfg.fluid.wLen), cfg.container.wLen);

		return cfg;
	}

	inline BoundaryRigidMeshesConfig LoadBoundaryRigidMeshesConfigFromProfile(
		const std::string& profile, ScenarioMgr& scenarioMgr)
	{
		return LoadMeshScenarioConfigFromProfile<BoundaryRigidMeshesConfig>(profile, "BoundaryRigidMeshes",
			[&](const nlohmann::json& j)
			{
				BoundaryRigidMeshesConfig cfg = LoadBoundaryRigidMeshesConfigFromJson(j, scenarioMgr);
				ResolveBoundaryMeshModels(cfg.boundaryBodies, "BoundaryRigidMeshes");
				ResolveRigidMeshModels(cfg.rigidBodies, "BoundaryRigidMeshes");
				return cfg;
			});
	}

} // namespace Scene
