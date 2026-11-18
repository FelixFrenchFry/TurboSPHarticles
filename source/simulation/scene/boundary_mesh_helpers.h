// boundary_mesh_helpers.h
#pragma once

// external
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <glm/common.hpp>
#include <glm/mat3x3.hpp>
#include <glm/matrix.hpp>
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
	enum class BoundaryMeshAlignment : uint8_t
	{
		Min,
		Center,
		Max
	};

	struct BoundaryMeshEntryConfig
	{
		// model asset nickname and resolved OBJ file
		std::string modelName;
		std::string modelFile;
		float modelTargetExtent; // target length of the largest local bounding box extent
		glm::mat3 modelSourceToCanonical{ 1.0f }; // local to canonical mesh orientation
		std::array<BoundaryMeshAlignment, 3> modelContainerAlignment{}; // one alignment per container axis (u/v/w)

		// static boundary mesh pose
		bool useContainerAlignment = false;
		glm::vec3 boundaryMeshPosition{ 0.0f }; // world-space offset if aligned, absolute center otherwise
		float boundaryMeshYawDeg; // turn around world Y
		float boundaryMeshPitchDeg; // tilt around local Z
		float boundaryMeshRollDeg; // tilt around local X
	};

	// aligns the bounds of a nonempty mesh already scaled and centered in local space
	inline glm::vec3 ComputeBoundaryMeshCenter(
		const BoundaryMeshEntryConfig& cfg,
		const std::vector<glm::vec3>& meshLocalVertices,
		const glm::mat3& orientation,
		const OBB& container)
	{
		if (!cfg.useContainerAlignment) { return cfg.boundaryMeshPosition; }

		const glm::mat3 containerToWorld{ container.uHat, container.vHat, container.wHat };
		const float determinant = glm::determinant(containerToWorld);
		if (!std::isfinite(determinant) || determinant == 0.0f)
		{
			throw std::runtime_error("Boundary mesh alignment requires independent container axes.");
		}

		// measure the rotated mesh bounds in container coordinates
		const glm::mat3 meshToContainer = glm::inverse(containerToWorld) * orientation;
		glm::vec3 meshMin = meshToContainer * meshLocalVertices.front();
		glm::vec3 meshMax = meshMin;
		for (const glm::vec3& vertex : meshLocalVertices)
		{
			const glm::vec3 vertexInContainer = meshToContainer * vertex;
			meshMin = glm::min(meshMin, vertexInContainer);
			meshMax = glm::max(meshMax, vertexInContainer);
		}

		const glm::vec3 containerLengths{ container.uLen, container.vLen, container.wLen };
		glm::vec3 centerInContainer{ 0.0f };
		for (glm::length_t axis = 0; axis < 3; axis++)
		{
			switch (cfg.modelContainerAlignment[axis])
			{
			case BoundaryMeshAlignment::Min:
				centerInContainer[axis] = -meshMin[axis];
				break;
			case BoundaryMeshAlignment::Center:
				centerInContainer[axis] = 0.5f * (containerLengths[axis] - meshMin[axis] - meshMax[axis]);
				break;
			case BoundaryMeshAlignment::Max:
				centerInContainer[axis] = containerLengths[axis] - meshMax[axis];
				break;
			}
		}

		return container.origin + containerToWorld * centerInContainer + cfg.boundaryMeshPosition;
	}

	// loads, scales, centers, and samples a boundary mesh into single-layer boundary particles
	inline void BuildBoundaryMesh(
		ScenarioMgr& scenarioMgr,
		const BoundaryMeshEntryConfig& cfg,
		const OBB& container,
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

		SurfaceSamplingResult sampling;
		glm::vec3 boundaryMeshCenterOfPosition;

		// keep mesh storage scoped to sampling
		{
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

			boundaryMeshCenterOfPosition = ComputeBoundaryMeshCenter(
				cfg, mesh.vertices, boundaryMeshOrientation, container);

			SPDLOG_INFO("Sampling boundary model '{}' surfaces...", cfg.modelName);
			SurfaceSampler sampler(mesh, params,
				params.surfaceSamplingMinDistanceB,
				params.surfaceSamplingMaxDistanceB,
				params.surfaceSamplingCoverageProbeSpacingB);
			sampling = sampler.Sample();
		}

		scenarioMgr.BuildBoundary_FromSurfaceSamples(
			std::move(sampling.acceptedSamples),
			boundaryMeshCenterOfPosition,
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
		const bool hasAbsoluteCenter = boundaryMesh.contains("centerOfPosition");
		if (hasAbsoluteCenter && boundaryMesh.contains("position"))
		{
			throw std::runtime_error(pose + " must use either 'position' or 'centerOfPosition'.");
		}
		cfg.useContainerAlignment = !hasAbsoluteCenter;
		cfg.boundaryMeshPosition = ReadScenarioVector<3>(
			boundaryMesh, hasAbsoluteCenter ? "centerOfPosition" : "position", pose);
		cfg.boundaryMeshYawDeg = ReadScenarioFloat(boundaryMesh, "yawDeg", pose);
		cfg.boundaryMeshPitchDeg = ReadScenarioFloat(boundaryMesh, "pitchDeg", pose);
		cfg.boundaryMeshRollDeg = ReadScenarioFloat(boundaryMesh, "rollDeg", pose);

		return cfg;
	}

	// try to resolve a boundary model nickname to a file path, container alignment,
	// and optional source-to-canonical orientation
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

		if (cfg.useContainerAlignment)
		{
			const std::string context = scenarioName + " boundary model '" + cfg.modelName + "'";
			const auto& alignment = ReadScenarioObject(model, "containerAlignment", context);
			const std::string alignmentContext = context + ".containerAlignment";

			auto ReadAlignment = [&](const char* axisName) -> BoundaryMeshAlignment
				{
					const std::string value = ReadScenarioString(alignment, axisName, alignmentContext);
					if (value == "min") { return BoundaryMeshAlignment::Min; }
					if (value == "center") { return BoundaryMeshAlignment::Center; }
					if (value == "max") { return BoundaryMeshAlignment::Max; }
					throw std::runtime_error(
						alignmentContext + "." + axisName + " must be 'min', 'center', or 'max'.");
				};

			cfg.modelContainerAlignment = { ReadAlignment("u"), ReadAlignment("v"), ReadAlignment("w") };
		}

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
