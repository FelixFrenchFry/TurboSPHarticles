// boundary_mesh_helpers.h
#pragma once

// external
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// internal
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/sync/state_sim.h"
#include "geometry/obj_loading.h"
#include "geometry/surface_sampling.h"
#include "builder.h"
#include "scenario_json.h"



namespace Scene
{
	struct BoundaryMeshEntryConfig
	{
		// model asset nickname and resolved OBJ file
		std::string modelName;
		std::string modelFile;
		float modelTargetExtent; // target length of the largest local bounding box extent
		glm::mat3 modelSourceToCanonical{ 1.0f }; // local to canonical mesh orientation

		// static boundary mesh pose
		glm::vec3 boundaryMeshCenterOfPosition; // world position of the bounding box center
		float boundaryMeshYawDeg; // turn around world Y
		float boundaryMeshPitchDeg; // tilt around local Z
		float boundaryMeshRollDeg; // tilt around local X
	};

	// loads, scales, centers, and samples a boundary mesh into single-layer boundary particles
	inline void BuildBoundaryMesh(
		ScenarioMgr& scenarioMgr,
		const BoundaryMeshEntryConfig& cfg,
		Sync::UniversalSimState& dst,
		const Sim::SimParams& params)
	{
		const glm::mat3 scenarioOrientation =
			scenarioMgr.MakeOrientation_FromEulerAngles(
				cfg.boundaryMeshYawDeg,
				cfg.boundaryMeshPitchDeg,
				cfg.boundaryMeshRollDeg);
		const glm::mat3 boundaryMeshOrientation =
			scenarioOrientation * cfg.modelSourceToCanonical;

		const std::filesystem::path modelDirectory =
			std::filesystem::path(TSP_CONFIG_DIR).parent_path() /
			"assets" / "models" / "boundary_meshes";
		const ObjMeshLoader loader(modelDirectory);
		TriangleSurfaceMesh mesh = loader.Load(cfg.modelFile);

		// scale the model before centering it
		const glm::vec3 boundsCenter = ScaleToTargetExtent(mesh, cfg.modelTargetExtent);

		// shift the sampled model into its bounding box center frame
		for (glm::vec3& vertex : mesh.vertices)
		{
			vertex -= boundsCenter;
		}

		SPDLOG_INFO("Sampling boundary model '{}' surfaces...", cfg.modelName);
		SurfaceSampler sampler(mesh, params,
			params.surfaceSamplingMinDistanceB,
			params.surfaceSamplingMaxDistanceB,
			params.surfaceSamplingCoverageProbeSpacingB);
		SurfaceSamplingResult sampling = sampler.Sample();
		scenarioMgr.BuildBoundary_FromSurfaceSamples(
			std::move(sampling.acceptedSamples),
			cfg.boundaryMeshCenterOfPosition,
			boundaryMeshOrientation,
			dst,
			params);
	}

	// reads one boundary mesh entry (model, target extent, and pose)
	inline BoundaryMeshEntryConfig LoadBoundaryMeshEntryConfigFromJson(
		const nlohmann::json& j, const std::string& scenarioName)
	{
		BoundaryMeshEntryConfig cfg{};
		cfg.modelName = ReadScenarioString(j, "model", scenarioName);
		cfg.modelTargetExtent = ReadScenarioFloat(j, "modelTargetExtent", scenarioName);

		const auto& boundaryMesh = ReadScenarioObject(j, "boundaryMesh", scenarioName);
		const std::string pose = scenarioName + ".boundaryMesh";
		cfg.boundaryMeshCenterOfPosition = ReadScenarioVector<3>(boundaryMesh, "centerOfPosition", pose);
		cfg.boundaryMeshYawDeg = ReadScenarioFloat(boundaryMesh, "yawDeg", pose);
		cfg.boundaryMeshPitchDeg = ReadScenarioFloat(boundaryMesh, "pitchDeg", pose);
		cfg.boundaryMeshRollDeg = ReadScenarioFloat(boundaryMesh, "rollDeg", pose);

		return cfg;
	}

	// try to resolve a boundary model nickname to a file path and optional source-to-canonical orientation
	inline bool TryResolveBoundaryMeshModel(
		const std::filesystem::path& modelsFile,
		BoundaryMeshEntryConfig& cfg,
		const std::string& scenarioName)
	{
		std::ifstream in(modelsFile);
		if (!in) { return false; }

		nlohmann::json root;
		try
		{
			in >> root;
		}
		catch (const std::exception&)
		{
			return false;
		}
		if (!root.contains("models") || !root["models"].is_object())
		{
			return false;
		}

		const auto& models = root["models"];
		if (!models.contains(cfg.modelName) || !models[cfg.modelName].is_object())
		{
			return false;
		}

		const auto& model = models[cfg.modelName];
		if (!model.contains("file") || !model["file"].is_string())
		{
			return false;
		}
		cfg.modelFile = model["file"].get<std::string>();
		cfg.modelSourceToCanonical = glm::mat3{ 1.0f };

		if (auto it = model.find("sourceToCanonical"); it != model.end())
		{
			if (!it->is_object())
			{
				throw std::runtime_error(
					scenarioName + " boundary model '" + cfg.modelName +
					"' sourceToCanonical must be an object.");
			}

			auto ReadAxis = [&](const char* axisName) -> glm::vec3
				{
					if (!it->contains(axisName) ||
						!(*it)[axisName].is_array() ||
						(*it)[axisName].size() != 3)
					{
						throw std::runtime_error(
							scenarioName + " boundary model '" + cfg.modelName +
							"' sourceToCanonical '" + axisName +
							"' must contain 3 values.");
					}

					const auto& axis = (*it)[axisName];
					const glm::vec3 value
					{
						axis[0].get<float>(),
						axis[1].get<float>(),
						axis[2].get<float>(),
					};
					if (!std::isfinite(value.x) ||
						!std::isfinite(value.y) ||
						!std::isfinite(value.z))
					{
						throw std::runtime_error(
							scenarioName + " boundary model '" + cfg.modelName +
							"' sourceToCanonical values must be finite.");
					}
					return value;
				};

			cfg.modelSourceToCanonical = glm::mat3(
				ReadAxis("uDir"),
				ReadAxis("vDir"),
				ReadAxis("wDir"));
		}

		return true;
	}

	// reads independently placed mesh entries from a non-empty body array
	inline std::vector<BoundaryMeshEntryConfig> LoadBoundaryMeshEntriesFromJson(
		const nlohmann::json& j, const std::string& scenarioName)
	{
		const auto& boundaryBodies = ReadScenarioNonEmptyArray(j, "boundaryBodies", scenarioName);
		std::vector<BoundaryMeshEntryConfig> entries;
		entries.reserve(boundaryBodies.size());
		for (size_t i = 0; i < boundaryBodies.size(); i++)
		{
			entries.push_back(LoadBoundaryMeshEntryConfigFromJson(boundaryBodies[i],
				scenarioName + ".boundaryBodies[" + std::to_string(i) + "]"));
		}
		return entries;
	}

	// resolves mesh model nicknames using the existing asset catalog
	inline void ResolveBoundaryMeshModels(
		std::vector<BoundaryMeshEntryConfig>& boundaryBodies, const std::string& scenarioName)
	{
		const std::filesystem::path modelCatalogPath =
			std::filesystem::path(TSP_CONFIG_DIR).parent_path() /
			"assets" / "models" / "boundary_models.json";
		for (BoundaryMeshEntryConfig& boundaryMesh : boundaryBodies)
		{
			if (TryResolveBoundaryMeshModel(modelCatalogPath, boundaryMesh, scenarioName)) { continue; }

			throw std::runtime_error(
				scenarioName + " boundary model '" + boundaryMesh.modelName +
				"' is not defined in the boundary model catalog '" + modelCatalogPath.string() + "'.");
		}
	}

} // namespace Scene
