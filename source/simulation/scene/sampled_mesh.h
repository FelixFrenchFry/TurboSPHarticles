// sampled_mesh.h

#pragma once

// external
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
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
#include "geometry/obj_loading.h"
#include "geometry/rigid_body_properties.h"
#include "geometry/surface_sampling.h"
#include "builder.h"
#include "scenario_sim.h"



namespace Scene
{
	struct SampledMeshConfig
	{
		// outer boundary container
		OBB container;

		// fluid block placed inside
		OBB fluid;

		// model asset nickname and resolved OBJ file
		std::string modelName;
		std::string modelFile;
		float modelTargetExtent = 1.0f; // target length of the largest local bounding box extent
		glm::mat3 modelSourceToCanonical{ 1.0f }; // local to canonical body orientation

		// initial rigid body pose
		glm::vec3 rigidBodyCenterOfMass{ 0.0f };
		float rigidBodyYawDeg = 0.0f; // turn around world Y
		float rigidBodyPitchDeg = 0.0f; // tilt around local Z
		float rigidBodyRollDeg = 0.0f; // tilt around local X

		// rigid body material and motion
		float rigidDensity = 1'000.0f;
		glm::vec3 rigidInitVel{ 0.0f };
		glm::vec3 rigidInitOmega{ 0.0f };
	};

	class SampledMeshScene final : public ISimScenario
	{
	public: // ---------------------------------------------------------------------------------------

		SampledMeshScene(ScenarioMgr& scenarioMgr, SampledMeshConfig cfg)
			: scenarioMgr_(scenarioMgr), cfg_(std::move(cfg)) {}

		const char* Name() const override { return "Sampled Mesh"; }

		void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const override
		{
			scenarioMgr_.BuildBoundaryContainer_OBB(cfg_.container, dst, params);

			scenarioMgr_.BuildFluidBlock_OBB(cfg_.fluid, dst, params);

			if (!std::isfinite(cfg_.modelTargetExtent) || cfg_.modelTargetExtent <= 0.0f)
			{
				throw std::runtime_error("Rigid mesh model target extent must be > 0.");
			}

			const glm::mat3 scenarioOrientation =
				scenarioMgr_.MakeOrientation_FromEulerAngles(
					cfg_.rigidBodyYawDeg,
					cfg_.rigidBodyPitchDeg,
					cfg_.rigidBodyRollDeg);
			const glm::mat3 rigidBodyOrientation =
				scenarioOrientation * cfg_.modelSourceToCanonical;

			const std::filesystem::path modelDirectory =
				std::filesystem::path(TSP_CONFIG_DIR).parent_path() /
				"assets" / "models" / "meshes";
			const ObjMeshLoader loader(modelDirectory);
			TriangleSurfaceMesh mesh = loader.Load(cfg_.modelFile);

			// compute the raw model bounds
			glm::vec3 modelMin = mesh.vertices.front();
			glm::vec3 modelMax = mesh.vertices.front();
			for (const glm::vec3& vertex : mesh.vertices)
			{
				modelMin.x = std::min(modelMin.x, vertex.x);
				modelMin.y = std::min(modelMin.y, vertex.y);
				modelMin.z = std::min(modelMin.z, vertex.z);
				modelMax.x = std::max(modelMax.x, vertex.x);
				modelMax.y = std::max(modelMax.y, vertex.y);
				modelMax.z = std::max(modelMax.z, vertex.z);
			}

			const glm::vec3 modelExtent = modelMax - modelMin;
			const float rawModelExtent = std::max(
				modelExtent.x,
				std::max(modelExtent.y, modelExtent.z));
			if (!std::isfinite(rawModelExtent) || rawModelExtent <= 0.0f)
			{
				throw std::runtime_error("Rigid mesh model extent must be > 0.");
			}

			// derive a uniform scale from the largest model extent
			const float modelScale = cfg_.modelTargetExtent / rawModelExtent;
			if (!std::isfinite(modelScale) || modelScale <= 0.0f)
			{
				throw std::runtime_error("Rigid mesh model scale must be > 0.");
			}

			// scale the model before computing mass and inertia
			for (glm::vec3& vertex : mesh.vertices)
			{
				vertex *= modelScale;
			}

			RigidBodyPropertyComputer propertyComputer;
			const RigidBodyPhysicalProperties properties =
				propertyComputer.Compute(mesh, cfg_.rigidDensity);

			// shift the sampled model into its center-of-mass frame
			for (glm::vec3& vertex : mesh.vertices)
			{
				vertex -= properties.centerOfMass;
			}

			SPDLOG_INFO("Sampling model surfaces...");
			SurfaceSampler sampler(mesh, params);
			const SurfaceSamplingResult sampling = sampler.Sample();
			const uint32_t bodyIdx = scenarioMgr_.BuildRigidBody_FromSurfaceSamples(
				sampling.samples,
				cfg_.rigidBodyCenterOfMass,
				rigidBodyOrientation,
				properties.mass,
				properties.bodyInertiaInv,
				dst,
				params);
			if (bodyIdx == Sync::InvalidRigidBodyId) { return; }

			ApplyInitialRigidBodyState(dst.rigidBodies, bodyIdx);
		}

		std::vector<OBB> GetBoundaryBoxes() const override
		{
			return { cfg_.container };
		}

	private: // --------------------------------------------------------------------------------------

		void ApplyInitialRigidBodyState(
			Sync::RigidBodies& rigidBodies,
			const uint32_t bodyIdx) const
		{
			auto& RB = rigidBodies;

			RB.xVel[bodyIdx] = cfg_.rigidInitVel.x;
			RB.yVel[bodyIdx] = cfg_.rigidInitVel.y;
			RB.zVel[bodyIdx] = cfg_.rigidInitVel.z;

			RB.xOmega[bodyIdx] = cfg_.rigidInitOmega.x;
			RB.yOmega[bodyIdx] = cfg_.rigidInitOmega.y;
			RB.zOmega[bodyIdx] = cfg_.rigidInitOmega.z;

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

			const glm::vec3 omegaWorld = cfg_.rigidInitOmega;
			const glm::vec3 omegaBody = glm::transpose(orientation) * omegaWorld;
			const glm::vec3 angMomBody = glm::inverse(bodyInertiaInv) * omegaBody;
			const glm::vec3 angMomWorld = orientation * angMomBody;

			RB.xAngMom[bodyIdx] = angMomWorld.x;
			RB.yAngMom[bodyIdx] = angMomWorld.y;
			RB.zAngMom[bodyIdx] = angMomWorld.z;
		}

		ScenarioMgr& scenarioMgr_;
		SampledMeshConfig cfg_;
	};

	inline SampledMeshConfig MakeSampledMeshConfig_Default(ScenarioMgr& scenarioMgr)
	{
		constexpr glm::vec3 U{ 1, 0, 0 }; // forward
		constexpr glm::vec3 V{ 0, 1, 0 }; // up
		constexpr glm::vec3 W{ 0, 0, 1 }; // right

		SampledMeshConfig cfg{};

		cfg.container = scenarioMgr.MakeOBB_FromAxes
		(
			/* origin */		{ 0,0,0 },
			/* uDir, uLen */	U, 1.0f,
			/* vDir, vLen */	V, 1.0f,
			/* wDir, wLen */	W, 1.0f
		);

		cfg.fluid = scenarioMgr.MakeOBB_FromAxes
		(
			/* origin */		{ 0.0f,0.0f,0.0f },
			/* uDir, uLen */	U, 0.8f,
			/* vDir, vLen */	V, 0.2f,
			/* wDir, wLen */	W, 0.8f
		);

		cfg.modelName = "UnitCube";
		cfg.modelFile = "unit_cube.obj";
		cfg.modelTargetExtent = 0.16f;
		cfg.rigidBodyCenterOfMass = { 0.5f, 0.25f, 0.5f };
		cfg.rigidBodyYawDeg = 0.0f;
		cfg.rigidBodyPitchDeg = 0.0f;
		cfg.rigidBodyRollDeg = 0.0f;
		cfg.rigidDensity = 200.0f;
		cfg.rigidInitVel = { 0.0f, 0.0f, 0.0f };
		cfg.rigidInitOmega = { 0.0f, 0.0f, 0.0f };

		return cfg;
	}

	inline SampledMeshConfig LoadSampledMeshConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		// cache default config
		SampledMeshConfig cfg = MakeSampledMeshConfig_Default(scenarioMgr);

		if (auto it = j.find("type"); it != j.end() && it->is_string())
		{
			const std::string type = it->get<std::string>();
			if (type != "SampledMesh")
			{
				SPDLOG_WARN("SampledMesh entry has type '{}' (expected 'SampledMesh').", type);
			}
		}

		if (auto it = j.find("container"); it != j.end())
		{
			cfg.container = ScenarioMgr::ParseOBBFromJson(*it, scenarioMgr);
		}

		if (auto it = j.find("fluid"); it != j.end())
		{
			cfg.fluid = ScenarioMgr::ParseOBBFromJson(*it, scenarioMgr);
		}

		if (auto it = j.find("model"); it != j.end() && it->is_string())
		{
			cfg.modelName = it->get<std::string>();
		}

		if (auto it = j.find("modelTargetExtent"); it != j.end())
		{
			cfg.modelTargetExtent = it->get<float>();
		}
		if (j.find("modelScale") != j.end())
		{
			throw std::runtime_error(
				"SampledMesh configuration uses 'modelScale'; use 'modelTargetExtent' instead.");
		}

		if (auto it = j.find("rigidBody"); it != j.end() && it->is_object())
		{
			auto ReadVec3 = [](const nlohmann::json& value, glm::vec3& dst)
				{
					if (value.is_array() && value.size() == 3)
					{
						dst.x = value[0].get<float>();
						dst.y = value[1].get<float>();
						dst.z = value[2].get<float>();
					}
				};

			const auto& rigidBody = *it;
			if (auto value = rigidBody.find("centerOfMass"); value != rigidBody.end())
			{
				ReadVec3(*value, cfg.rigidBodyCenterOfMass);
			}
			if (rigidBody.contains("uDir") ||
				rigidBody.contains("vDir") ||
				rigidBody.contains("wDir"))
			{
				throw std::runtime_error(
					"SampledMesh rigidBody uses 'uDir/vDir/wDir'; use 'yawDeg/pitchDeg/rollDeg' instead.");
			}

			if (auto value = rigidBody.find("yawDeg"); value != rigidBody.end())
			{
				cfg.rigidBodyYawDeg = value->get<float>();
			}
			if (auto value = rigidBody.find("pitchDeg"); value != rigidBody.end())
			{
				cfg.rigidBodyPitchDeg = value->get<float>();
			}
			if (auto value = rigidBody.find("rollDeg"); value != rigidBody.end())
			{
				cfg.rigidBodyRollDeg = value->get<float>();
			}
		}

		if (auto it = j.find("rigidDensity"); it != j.end())
		{
			cfg.rigidDensity = it->get<float>();
		}

		if (auto it = j.find("rigidInitVel"); it != j.end() &&
			it->is_array() && it->size() == 3)
		{
			cfg.rigidInitVel.x = (*it)[0].get<float>();
			cfg.rigidInitVel.y = (*it)[1].get<float>();
			cfg.rigidInitVel.z = (*it)[2].get<float>();
		}

		if (auto it = j.find("rigidInitOmega"); it != j.end() &&
			it->is_array() && it->size() == 3)
		{
			cfg.rigidInitOmega.x = (*it)[0].get<float>();
			cfg.rigidInitOmega.y = (*it)[1].get<float>();
			cfg.rigidInitOmega.z = (*it)[2].get<float>();
		}

		// sanitize fluid lengths
		cfg.fluid.uLen = std::min(std::max(0.0f, cfg.fluid.uLen), cfg.container.uLen);
		cfg.fluid.vLen = std::min(std::max(0.0f, cfg.fluid.vLen), cfg.container.vLen);
		cfg.fluid.wLen = std::min(std::max(0.0f, cfg.fluid.wLen), cfg.container.wLen);

		return cfg;
	}

	inline SampledMeshConfig LoadSampledMeshConfigFromProfile(
		const std::string& profile, ScenarioMgr& scenarioMgr)
	{
		// try to load a SampledMesh scenario from scenarios.json file of a given profile
		auto TryLoadFromScenarioFile = [&](const std::string& scenariosFile)
			-> std::optional<SampledMeshConfig>
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

				// expect scenarios["SampledMesh"] object
				if (!scenarios.contains("SampledMesh") || !scenarios["SampledMesh"].is_object())
				{
					return std::nullopt;
				}

				return LoadSampledMeshConfigFromJson(scenarios["SampledMesh"], scenarioMgr);
			};

		// try to resolve a model nickname to a file path and optional source-to-canonical orientation
		auto TryResolveModel = [](const std::filesystem::path& modelsFile,
			SampledMeshConfig& cfg) -> bool
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
							"SampledMesh model '" + cfg.modelName +
							"' sourceToCanonical must be an object.");
					}

					auto ReadAxis = [&](const char* axisName) -> glm::vec3
						{
							if (!it->contains(axisName) ||
								!(*it)[axisName].is_array() ||
								(*it)[axisName].size() != 3)
							{
								throw std::runtime_error(
									"SampledMesh model '" + cfg.modelName +
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
									"SampledMesh model '" + cfg.modelName +
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
			};

		// load global model catalog from assets and resolve models below /assets/models
		const std::filesystem::path assetDirectory =
			std::filesystem::path(TSP_CONFIG_DIR).parent_path() / "assets";
		const std::filesystem::path modelDirectory =
			assetDirectory / "models" / "meshes";
		const std::filesystem::path modelCatalogPath =
			assetDirectory / "models" / "models.json";

		auto LoadProfile = [&](const std::string& profileName)
			-> std::optional<SampledMeshConfig>
			{
				const auto cfg = TryLoadFromScenarioFile(profileName + "/scenarios.json");
				if (!cfg) { return std::nullopt; }

				SampledMeshConfig resolved = *cfg;
				if (TryResolveModel(modelCatalogPath, resolved))
				{
					return resolved;
				}

				throw std::runtime_error(
					"SampledMesh model '" + resolved.modelName +
					"' is not defined in the global model catalog '" +
					modelCatalogPath.string() + "'.");
			};

		// try chosen profile
		if (auto cfg = LoadProfile(profile)) { return *cfg; }
		SPDLOG_WARN("Could not load SampledMesh from '/{}/scenarios.json'.", profile);

		// fallback to default profile
		if (profile != "default")
		{
			if (auto cfg = LoadProfile("default")) { return *cfg; }
			SPDLOG_WARN("Could not load SampledMesh from '/default/scenarios.json'.");
		}

		// final fallback to built-in defaults
		SPDLOG_WARN("Using built-in SampledMesh defaults.");
		SampledMeshConfig cfg = MakeSampledMeshConfig_Default(scenarioMgr);
		cfg.modelFile = "unit_cube.obj";
		return cfg;
	}

} // namespace Scene
