// rigid_mesh_helpers.h
#pragma once

// external
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_inverse.hpp>
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
#include "geometry/rigid_body_properties.h"
#include "geometry/surface_sampling.h"
#include "builder.h"
#include "scenario_json.h"



namespace Scene
{
	struct RigidMeshEntryConfig
	{
		// model asset nickname and resolved OBJ file
		std::string modelName;
		std::string modelFile;
		float modelTargetExtent; // target length of the largest local bounding box extent
		glm::mat3 modelSourceToCanonical{ 1.0f }; // local to canonical body orientation
		glm::vec3 modelCenterOfMassOffset{ 0.0f }; // canonical offset relative to target extent

		// initial rigid body pose
		glm::vec3 rigidBodyCenterOfPosition; // world position of the volume centroid
		float rigidBodyYawDeg; // turn around world Y
		float rigidBodyPitchDeg; // tilt around local Z
		float rigidBodyRollDeg; // tilt around local X

		// rigid body material and motion
		float rigidDensity;
		glm::vec3 rigidInitVel;
		glm::vec3 rigidInitOmega;
	};

	// sets the initial linear and angular motion of a built rigid body
	inline void ApplyInitialRigidBodyState(
		Sync::RigidBodies& rigidBodies,
		const uint32_t bodyIdx,
		const RigidMeshEntryConfig& cfg)
	{
		auto& RB = rigidBodies;

		RB.xVel[bodyIdx] = cfg.rigidInitVel.x;
		RB.yVel[bodyIdx] = cfg.rigidInitVel.y;
		RB.zVel[bodyIdx] = cfg.rigidInitVel.z;

		RB.xOmega[bodyIdx] = cfg.rigidInitOmega.x;
		RB.yOmega[bodyIdx] = cfg.rigidInitOmega.y;
		RB.zOmega[bodyIdx] = cfg.rigidInitOmega.z;

		glm::mat3 orientation{ 1.0f };
		orientation[0] =
			{ RB.a00[bodyIdx], RB.a10[bodyIdx], RB.a20[bodyIdx] };
		orientation[1] =
			{ RB.a01[bodyIdx], RB.a11[bodyIdx], RB.a21[bodyIdx] };
		orientation[2] =
			{ RB.a02[bodyIdx], RB.a12[bodyIdx], RB.a22[bodyIdx] };

		glm::mat3 bodyInertiaInv{ 0.0f };
		bodyInertiaInv[0] =
			{ RB.iBodyInv00[bodyIdx], RB.iBodyInv10[bodyIdx], RB.iBodyInv20[bodyIdx] };
		bodyInertiaInv[1] =
			{ RB.iBodyInv01[bodyIdx], RB.iBodyInv11[bodyIdx], RB.iBodyInv21[bodyIdx] };
		bodyInertiaInv[2] =
			{ RB.iBodyInv02[bodyIdx], RB.iBodyInv12[bodyIdx], RB.iBodyInv22[bodyIdx] };

		const glm::vec3 omegaWorld = cfg.rigidInitOmega;
		const glm::vec3 omegaBody = glm::transpose(orientation) * omegaWorld;
		const glm::vec3 angMomBody = glm::inverse(bodyInertiaInv) * omegaBody;
		const glm::vec3 angMomWorld = orientation * angMomBody;

		RB.xAngMom[bodyIdx] = angMomWorld.x;
		RB.yAngMom[bodyIdx] = angMomWorld.y;
		RB.zAngMom[bodyIdx] = angMomWorld.z;
	}

	// loads, scales, centers, and samples a rigid mesh into a rigid body of single-layer rigid particles
	inline void BuildRigidMesh(
		ScenarioMgr& scenarioMgr,
		const RigidMeshEntryConfig& cfg,
		Sync::UniversalSimState& dst,
		const Sim::SimParams& params)
	{
		const glm::mat3 scenarioOrientation =
			scenarioMgr.MakeOrientation_FromEulerAngles(
				cfg.rigidBodyYawDeg,
				cfg.rigidBodyPitchDeg,
				cfg.rigidBodyRollDeg);
		const glm::mat3 rigidBodyOrientation =
			scenarioOrientation * cfg.modelSourceToCanonical;

		const std::filesystem::path modelDirectory =
			std::filesystem::path(TSP_CONFIG_DIR).parent_path() /
			"assets" / "models" / "rigid_meshes";
		const ObjMeshLoader loader(modelDirectory);
		TriangleSurfaceMesh mesh = loader.Load(cfg.modelFile);

		// scale the model before computing mass and inertia
		ScaleToTargetExtent(mesh, cfg.modelTargetExtent);

		RigidBodyPropertyComputer propertyComputer;
		const RigidBodyPhysicalProperties properties =
			propertyComputer.Compute(mesh, cfg.rigidDensity);

		// keep the mesh-derived centroidal inertia as the effective body inertia
		const glm::vec3 centerOfMassOffset =
			glm::transpose(cfg.modelSourceToCanonical) *
			(cfg.modelCenterOfMassOffset * cfg.modelTargetExtent);
		const glm::vec3 centerOfMass = properties.centerOfMass + centerOfMassOffset;
		const glm::vec3 rigidBodyCenterOfMass = cfg.rigidBodyCenterOfPosition +
			scenarioOrientation * (cfg.modelCenterOfMassOffset * cfg.modelTargetExtent);

		// shift the sampled model into its effective center-of-mass frame
		for (glm::vec3& vertex : mesh.vertices)
		{
			vertex -= centerOfMass;
		}

		SPDLOG_INFO("Sampling rigid model '{}' surfaces...", cfg.modelName);
		SurfaceSampler sampler(mesh, params,
			params.surfaceSamplingMinDistanceR,
			params.surfaceSamplingMaxDistanceR,
			params.surfaceSamplingCoverageProbeSpacingR);
		SurfaceSamplingResult sampling = sampler.Sample();
		const uint32_t bodyIdx = scenarioMgr.BuildRigidBody_FromSurfaceSamples(
			std::move(sampling.acceptedSamples),
			rigidBodyCenterOfMass,
			rigidBodyOrientation,
			properties.mass,
			properties.bodyInertiaInv,
			dst,
			params);
		if (bodyIdx == Sync::InvalidRigidBodyId)
		{
			throw std::runtime_error(
				"Rigid model '" + cfg.modelName + "' produced no rigid surface samples.");
		}

		ApplyInitialRigidBodyState(dst.rigidBodies, bodyIdx, cfg);
	}

	// reads one rigid mesh entry (model, target extent, pose, material, and motion)
	inline RigidMeshEntryConfig LoadRigidMeshEntryConfigFromJson(
		const nlohmann::json& j, const std::string& scenarioName)
	{
		RigidMeshEntryConfig cfg{};
		cfg.modelName = ReadScenarioString(j, "model", scenarioName);
		cfg.modelTargetExtent = ReadScenarioFloat(j, "modelTargetExtent", scenarioName);

		const auto& rigidBody = ReadScenarioObject(j, "rigidBody", scenarioName);
		const std::string pose = scenarioName + ".rigidBody";
		cfg.rigidBodyCenterOfPosition = ReadScenarioVector<3>(rigidBody, "centerOfPosition", pose);
		cfg.rigidBodyYawDeg = ReadScenarioFloat(rigidBody, "yawDeg", pose);
		cfg.rigidBodyPitchDeg = ReadScenarioFloat(rigidBody, "pitchDeg", pose);
		cfg.rigidBodyRollDeg = ReadScenarioFloat(rigidBody, "rollDeg", pose);
		cfg.rigidDensity = ReadScenarioFloat(j, "rigidDensity", scenarioName);
		cfg.rigidInitVel = ReadScenarioVector<3>(j, "rigidInitVel", scenarioName);
		cfg.rigidInitOmega = ReadScenarioVector<3>(j, "rigidInitOmega", scenarioName);

		return cfg;
	}

	// try to resolve a rigid model nickname to a file path, optional source-to-canonical orientation,
	// and optional mass properties
	inline bool TryResolveRigidMeshModel(
		const std::filesystem::path& modelsFile,
		RigidMeshEntryConfig& cfg,
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
		cfg.modelCenterOfMassOffset = glm::vec3{ 0.0f };

		if (auto it = model.find("sourceToCanonical"); it != model.end())
		{
			if (!it->is_object())
			{
				throw std::runtime_error(
					scenarioName + " rigid model '" + cfg.modelName +
					"' sourceToCanonical must be an object.");
			}

			auto ReadAxis = [&](const char* axisName) -> glm::vec3
				{
					if (!it->contains(axisName) ||
						!(*it)[axisName].is_array() ||
						(*it)[axisName].size() != 3)
					{
						throw std::runtime_error(
							scenarioName + " rigid model '" + cfg.modelName +
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
							scenarioName + " rigid model '" + cfg.modelName +
							"' sourceToCanonical values must be finite.");
					}
					return value;
				};

			cfg.modelSourceToCanonical = glm::mat3(
				ReadAxis("uDir"),
				ReadAxis("vDir"),
				ReadAxis("wDir"));
		}

		if (auto it = model.find("massProperties"); it != model.end())
		{
			if (!it->is_object())
			{
				throw std::runtime_error(
					scenarioName + " rigid model '" + cfg.modelName +
					"' massProperties must be an object.");
			}

			if (auto offset = it->find("centerOfMassOffset"); offset != it->end())
			{
				if (!offset->is_array() || offset->size() != 3 ||
					!(*offset)[0].is_number() ||
					!(*offset)[1].is_number() ||
					!(*offset)[2].is_number())
				{
					throw std::runtime_error(
						scenarioName + " rigid model '" + cfg.modelName +
						"' centerOfMassOffset must contain 3 values.");
				}

				cfg.modelCenterOfMassOffset =
				{
					(*offset)[0].get<float>(),
					(*offset)[1].get<float>(),
					(*offset)[2].get<float>(),
				};
				if (!std::isfinite(cfg.modelCenterOfMassOffset.x) ||
					!std::isfinite(cfg.modelCenterOfMassOffset.y) ||
					!std::isfinite(cfg.modelCenterOfMassOffset.z))
				{
					throw std::runtime_error(
						scenarioName + " rigid model '" + cfg.modelName +
						"' centerOfMassOffset values must be finite.");
				}
			}
		}

		return true;
	}

	// reads independently placed mesh entries from a non-empty body array
	inline std::vector<RigidMeshEntryConfig> LoadRigidMeshEntriesFromJson(
		const nlohmann::json& j, const std::string& scenarioName)
	{
		const auto& rigidBodies = ReadScenarioNonEmptyArray(j, "rigidBodies", scenarioName);
		std::vector<RigidMeshEntryConfig> entries;
		entries.reserve(rigidBodies.size());
		for (size_t i = 0; i < rigidBodies.size(); i++)
		{
			entries.push_back(LoadRigidMeshEntryConfigFromJson(rigidBodies[i],
				scenarioName + ".rigidBodies[" + std::to_string(i) + "]"));
		}
		return entries;
	}

	// resolves mesh model nicknames using the existing asset catalog
	inline void ResolveRigidMeshModels(
		std::vector<RigidMeshEntryConfig>& rigidBodies, const std::string& scenarioName)
	{
		const std::filesystem::path modelCatalogPath =
			std::filesystem::path(TSP_CONFIG_DIR).parent_path() /
			"assets" / "models" / "rigid_models.json";
		for (RigidMeshEntryConfig& rigidMesh : rigidBodies)
		{
			if (TryResolveRigidMeshModel(modelCatalogPath, rigidMesh, scenarioName)) { continue; }

			throw std::runtime_error(
				scenarioName + " rigid model '" + rigidMesh.modelName +
				"' is not defined in the rigid model catalog '" + modelCatalogPath.string() + "'.");
		}
	}

} // namespace Scene
