// rigid_mesh_group.h
#pragma once

// external
#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// internal
#include "source/base/helpers.h"
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/sync/state_sim.h"
#include "builder.h"
#include "boundary_mesh.h"
#include "rigid_mesh.h"
#include "scenario_sim.h"



namespace Scene
{
	struct RigidMeshGroupConfig
	{
		// outer boundary container
		OBB container;

		// fluid block placed inside
		OBB fluid;

		// static single-layer sampled mesh boundaries
		std::vector<BoundaryMeshEntryConfig> boundaryMeshes;

		// independently simulated sampled mesh bodies
		std::vector<RigidMeshEntryConfig> rigidMeshes;
	};

	class RigidMeshGroupScene final : public ISimScenario
	{
	public:

		RigidMeshGroupScene(ScenarioMgr& scenarioMgr, RigidMeshGroupConfig cfg)
			: scenarioMgr_(scenarioMgr), cfg_(std::move(cfg)) {}

		const char* Name() const override { return "Rigid Mesh Group"; }

		void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const override
		{
			scenarioMgr_.BuildBoundaryContainer_OBB(cfg_.container, dst, params);

			scenarioMgr_.BuildFluidBlock_OBB(cfg_.fluid, dst, params);

			for (const BoundaryMeshEntryConfig& boundaryMesh : cfg_.boundaryMeshes)
			{
				BuildBoundaryMesh(scenarioMgr_, boundaryMesh, dst, params);
			}

			for (const RigidMeshEntryConfig& rigidMesh : cfg_.rigidMeshes)
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
		RigidMeshGroupConfig cfg_;
	};

	inline RigidMeshGroupConfig LoadRigidMeshGroupConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		RigidMeshGroupConfig cfg{};

		if (auto it = j.find("type"); it != j.end() && it->is_string())
		{
			const std::string type = it->get<std::string>();
			if (type != "RigidMeshGroup")
			{
				SPDLOG_WARN(
					"RigidMeshGroup entry has type '{}' (expected 'RigidMeshGroup').",
					type);
			}
		}

		auto container = j.find("container");
		if (container == j.end())
		{
			throw std::runtime_error("RigidMeshGroup requires a 'container' object.");
		}
		cfg.container = ScenarioMgr::ParseOBBFromJson(*container, scenarioMgr);

		auto fluid = j.find("fluid");
		if (fluid == j.end())
		{
			throw std::runtime_error("RigidMeshGroup requires a 'fluid' object.");
		}
		cfg.fluid = ScenarioMgr::ParseOBBFromJson(*fluid, scenarioMgr);

		auto rigidBodies = j.find("rigidBodies");
		if (rigidBodies == j.end() || !rigidBodies->is_array() || rigidBodies->empty())
		{
			throw std::runtime_error(
				"RigidMeshGroup 'rigidBodies' must be a non-empty array.");
		}

		cfg.rigidMeshes.reserve(rigidBodies->size());
		for (const nlohmann::json& rigidMeshJson : *rigidBodies)
		{
			cfg.rigidMeshes.push_back(
				LoadRigidMeshEntryConfigFromJson(rigidMeshJson, "RigidMeshGroup"));
		}

		if (auto boundaryMeshes = j.find("boundaryMeshes"); boundaryMeshes != j.end())
		{
			if (!boundaryMeshes->is_array())
			{
				throw std::runtime_error(
					"RigidMeshGroup 'boundaryMeshes' must be an array.");
			}

			cfg.boundaryMeshes.reserve(boundaryMeshes->size());
			for (const nlohmann::json& boundaryMeshJson : *boundaryMeshes)
			{
				cfg.boundaryMeshes.push_back(
					LoadBoundaryMeshEntryConfigFromJson(boundaryMeshJson, "RigidMeshGroup"));
			}
		}

		// sanitize fluid lengths
		cfg.fluid.uLen = std::min(std::max(0.0f, cfg.fluid.uLen), cfg.container.uLen);
		cfg.fluid.vLen = std::min(std::max(0.0f, cfg.fluid.vLen), cfg.container.vLen);
		cfg.fluid.wLen = std::min(std::max(0.0f, cfg.fluid.wLen), cfg.container.wLen);

		return cfg;
	}

	inline RigidMeshGroupConfig LoadRigidMeshGroupConfigFromProfile(
		const std::string& profile, ScenarioMgr& scenarioMgr)
	{
		// try to load a RigidMeshGroup scenario from the config file of a given profile
		auto TryLoadFromScenarioFile = [&](const std::string& scenariosFile)
			-> std::optional<RigidMeshGroupConfig>
			{
				nlohmann::json root;
				if (!Base::LoadJsonObjectFromFile(scenariosFile, root))
				{
					return std::nullopt;
				}

				// expect root["scenarios"] object
				if (!root.contains("scenarios") || !root["scenarios"].is_object())
				{
					return std::nullopt;
				}

				const auto& scenarios = root["scenarios"];

				// expect scenarios["RigidMeshGroup"] object
				if (!scenarios.contains("RigidMeshGroup") ||
					!scenarios["RigidMeshGroup"].is_object())
				{
					return std::nullopt;
				}

				return LoadRigidMeshGroupConfigFromJson(
					scenarios["RigidMeshGroup"], scenarioMgr);
			};

		// load rigid and boundary model catalogs from assets and resolve models below /assets/models
		const std::filesystem::path assetDirectory =
			std::filesystem::path(TSP_CONFIG_DIR).parent_path() / "assets";
		const std::filesystem::path modelCatalogPath =
			assetDirectory / "models" / "rigid_models.json";
		const std::filesystem::path boundaryModelCatalogPath =
			assetDirectory / "models" / "boundary_models.json";

		auto LoadProfile = [&](const std::string& profileName)
			-> std::optional<RigidMeshGroupConfig>
			{
				const auto cfg = TryLoadFromScenarioFile(Base::ScenariosPathForProfile(profileName));
				if (!cfg) { return std::nullopt; }

				RigidMeshGroupConfig resolved = *cfg;
				for (RigidMeshEntryConfig& rigidMesh : resolved.rigidMeshes)
				{
					if (TryResolveRigidMeshModel(
						modelCatalogPath, rigidMesh, "RigidMeshGroup")) { continue; }

					throw std::runtime_error(
						"RigidMeshGroup rigid model '" + rigidMesh.modelName +
						"' is not defined in the rigid model catalog '" +
						modelCatalogPath.string() + "'.");
				}

				for (BoundaryMeshEntryConfig& boundaryMesh : resolved.boundaryMeshes)
				{
					if (TryResolveBoundaryMeshModel(
						boundaryModelCatalogPath, boundaryMesh, "RigidMeshGroup")) { continue; }

					throw std::runtime_error(
						"RigidMeshGroup boundary model '" + boundaryMesh.modelName +
						"' is not defined in the boundary model catalog '" +
						boundaryModelCatalogPath.string() + "'.");
				}
				return resolved;
			};

		// try chosen profile
		if (auto cfg = LoadProfile(profile)) { return *cfg; }
		SPDLOG_WARN(
			"Could not load RigidMeshGroup from '/{}'.",
			Base::ScenariosPathForProfile(profile));

		// fallback to default profile
		if (profile != "default")
		{
			if (auto cfg = LoadProfile("default")) { return *cfg; }
			SPDLOG_WARN(
				"Could not load RigidMeshGroup from '/{}'.",
				Base::ScenariosPathForProfile("default"));
		}

		throw std::runtime_error(
			"RigidMeshGroup is not defined in profile '" + profile + "' or the default profile.");
	}

} // namespace Scene
