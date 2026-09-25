// rigid_mesh.h
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
	struct RigidMeshEntryConfig
	{
		// model asset nickname and resolved OBJ file
		std::string modelName;
		std::string modelFile;
		float modelTargetExtent = 1.0f; // target length of the largest local bounding box extent
		glm::mat3 modelSourceToCanonical{ 1.0f }; // local to canonical body orientation
		glm::vec3 modelCenterOfMassOffset{ 0.0f }; // canonical offset relative to target extent

		// initial rigid body pose
		glm::vec3 rigidBodyCenterOfPosition{ 0.0f }; // world position of the volume centroid
		float rigidBodyYawDeg = 0.0f; // turn around world Y
		float rigidBodyPitchDeg = 0.0f; // tilt around local Z
		float rigidBodyRollDeg = 0.0f; // tilt around local X

		// rigid body material and motion
		float rigidDensity = 1'000.0f;
		glm::vec3 rigidInitVel{ 0.0f };
		glm::vec3 rigidInitOmega{ 0.0f };
	};

	struct RigidMeshConfig
	{
		// outer boundary container
		OBB container;

		// fluid block placed inside
		OBB fluid;

		// independently simulated sampled mesh body
		RigidMeshEntryConfig rigidMesh;
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
		SurfaceSampler sampler(mesh, params);
		const SurfaceSamplingResult sampling = sampler.Sample();
		const uint32_t bodyIdx = scenarioMgr.BuildRigidBody_FromSurfaceSamples(
			sampling.samples,
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
		if (!j.is_object())
		{
			throw std::runtime_error(
				scenarioName + " rigid mesh entries must be objects.");
		}

		auto ReadVec3 = [](const nlohmann::json& value, glm::vec3& dst)
			{
				if (value.is_array() && value.size() == 3)
				{
					dst.x = value[0].get<float>();
					dst.y = value[1].get<float>();
					dst.z = value[2].get<float>();
				}
			};

		RigidMeshEntryConfig cfg{};
		auto model = j.find("model");
		if (model == j.end() || !model->is_string())
		{
			throw std::runtime_error(
				scenarioName + " rigid mesh requires a string 'model'.");
		}
		cfg.modelName = model->get<std::string>();

		if (auto value = j.find("modelTargetExtent"); value != j.end())
		{
			cfg.modelTargetExtent = value->get<float>();
		}
		if (j.find("modelScale") != j.end())
		{
			throw std::runtime_error(
				scenarioName + " configuration uses 'modelScale'; use 'modelTargetExtent' instead.");
		}

		auto rigidBody = j.find("rigidBody");
		if (rigidBody == j.end() || !rigidBody->is_object())
		{
			throw std::runtime_error(
				scenarioName + " rigid mesh requires a 'rigidBody' object.");
		}
		if (rigidBody->contains("centerOfMass"))
		{
			throw std::runtime_error(
				scenarioName + " rigidBody uses 'centerOfMass'; use 'centerOfPosition' instead.");
		}
		auto centerOfPosition = rigidBody->find("centerOfPosition");
		if (centerOfPosition == rigidBody->end() ||
			!centerOfPosition->is_array() || centerOfPosition->size() != 3 ||
			!(*centerOfPosition)[0].is_number() ||
			!(*centerOfPosition)[1].is_number() ||
			!(*centerOfPosition)[2].is_number())
		{
			throw std::runtime_error(
				scenarioName + " rigid mesh requires a three-value 'centerOfPosition'.");
		}
		ReadVec3(*centerOfPosition, cfg.rigidBodyCenterOfPosition);

		if (rigidBody->contains("uDir") ||
			rigidBody->contains("vDir") ||
			rigidBody->contains("wDir"))
		{
			throw std::runtime_error(
				scenarioName + " rigidBody uses 'uDir/vDir/wDir'; use 'yawDeg/pitchDeg/rollDeg' instead.");
		}

		if (auto value = rigidBody->find("yawDeg"); value != rigidBody->end())
		{
			cfg.rigidBodyYawDeg = value->get<float>();
		}
		if (auto value = rigidBody->find("pitchDeg"); value != rigidBody->end())
		{
			cfg.rigidBodyPitchDeg = value->get<float>();
		}
		if (auto value = rigidBody->find("rollDeg"); value != rigidBody->end())
		{
			cfg.rigidBodyRollDeg = value->get<float>();
		}

		if (auto value = j.find("rigidDensity"); value != j.end())
		{
			cfg.rigidDensity = value->get<float>();
		}

		if (auto value = j.find("rigidInitVel"); value != j.end())
		{
			ReadVec3(*value, cfg.rigidInitVel);
		}

		if (auto value = j.find("rigidInitOmega"); value != j.end())
		{
			ReadVec3(*value, cfg.rigidInitOmega);
		}

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

	class RigidMeshScene final : public ISimScenario
	{
	public:

		RigidMeshScene(ScenarioMgr& scenarioMgr, RigidMeshConfig cfg)
			: scenarioMgr_(scenarioMgr), cfg_(std::move(cfg)) {}

		const char* Name() const override { return "Rigid Mesh"; }

		void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const override
		{
			scenarioMgr_.BuildBoundaryContainer_OBB(cfg_.container, dst, params);

			scenarioMgr_.BuildFluidBlock_OBB(cfg_.fluid, dst, params);

			BuildRigidMesh(scenarioMgr_, cfg_.rigidMesh, dst, params);
		}

		std::vector<OBB> GetBoundaryBoxes() const override
		{
			return { cfg_.container };
		}

	private:

		ScenarioMgr& scenarioMgr_;
		RigidMeshConfig cfg_;
	};

	inline RigidMeshConfig LoadRigidMeshConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		RigidMeshConfig cfg{};

		if (auto it = j.find("type"); it != j.end() && it->is_string())
		{
			const std::string type = it->get<std::string>();
			if (type != "RigidMesh")
			{
				SPDLOG_WARN("RigidMesh entry has type '{}' (expected 'RigidMesh').", type);
			}
		}

		auto container = j.find("container");
		if (container == j.end())
		{
			throw std::runtime_error("RigidMesh requires a 'container' object.");
		}
		cfg.container = ScenarioMgr::ParseOBBFromJson(*container, scenarioMgr);

		auto fluid = j.find("fluid");
		if (fluid == j.end())
		{
			throw std::runtime_error("RigidMesh requires a 'fluid' object.");
		}
		cfg.fluid = ScenarioMgr::ParseOBBFromJson(*fluid, scenarioMgr);

		// the scenario object itself holds the rigid mesh entry keys
		cfg.rigidMesh = LoadRigidMeshEntryConfigFromJson(j, "RigidMesh");

		// sanitize fluid lengths
		cfg.fluid.uLen = std::min(std::max(0.0f, cfg.fluid.uLen), cfg.container.uLen);
		cfg.fluid.vLen = std::min(std::max(0.0f, cfg.fluid.vLen), cfg.container.vLen);
		cfg.fluid.wLen = std::min(std::max(0.0f, cfg.fluid.wLen), cfg.container.wLen);

		return cfg;
	}

	inline RigidMeshConfig LoadRigidMeshConfigFromProfile(
		const std::string& profile, ScenarioMgr& scenarioMgr)
	{
		// try to load a RigidMesh scenario from the config file of a given profile
		auto TryLoadFromScenarioFile = [&](const std::string& scenariosFile)
			-> std::optional<RigidMeshConfig>
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

				// expect scenarios["RigidMesh"] object
				if (!scenarios.contains("RigidMesh") || !scenarios["RigidMesh"].is_object())
				{
					return std::nullopt;
				}

				return LoadRigidMeshConfigFromJson(scenarios["RigidMesh"], scenarioMgr);
			};

		// load rigid model catalog from assets and resolve models below /assets/models
		const std::filesystem::path assetDirectory =
			std::filesystem::path(TSP_CONFIG_DIR).parent_path() / "assets";
		const std::filesystem::path modelCatalogPath =
			assetDirectory / "models" / "rigid_models.json";

		auto LoadProfile = [&](const std::string& profileName)
			-> std::optional<RigidMeshConfig>
			{
				const auto cfg = TryLoadFromScenarioFile(Base::ScenariosPathForProfile(profileName));
				if (!cfg) { return std::nullopt; }

				RigidMeshConfig resolved = *cfg;
				if (TryResolveRigidMeshModel(modelCatalogPath, resolved.rigidMesh, "RigidMesh"))
				{
					return resolved;
				}

				throw std::runtime_error(
					"RigidMesh model '" + resolved.rigidMesh.modelName +
					"' is not defined in the rigid model catalog '" +
					modelCatalogPath.string() + "'.");
			};

		// try chosen profile
		if (auto cfg = LoadProfile(profile)) { return *cfg; }
		SPDLOG_WARN(
			"Could not load RigidMesh from '/{}'.",
			Base::ScenariosPathForProfile(profile));

		// fallback to default profile
		if (profile != "default")
		{
			if (auto cfg = LoadProfile("default")) { return *cfg; }
			SPDLOG_WARN(
				"Could not load RigidMesh from '/{}'.",
				Base::ScenariosPathForProfile("default"));
		}

		throw std::runtime_error(
			"RigidMesh is not defined in profile '" + profile + "' or the default profile.");
	}

} // namespace Scene
