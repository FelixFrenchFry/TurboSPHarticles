// sampled_mesh_collection.h
#pragma once

// external
#include <algorithm>
#include <array>
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
	struct SampledMeshCollectionRigidBodyConfig
	{
		// model asset nickname and resolved OBJ file
		std::string modelName;
		std::string modelFile;
		float modelTargetExtent = 1.0f; // target length of the largest local bounding box extent
		glm::mat3 modelSourceToCanonical{ 1.0f }; // local to canonical body orientation
		glm::vec3 modelCenterOfMassOffset{ 0.0f }; // canonical offset relative to target extent

		// initial rigid body pose
		glm::vec3 rigidBodyCenterOfPosition{ 0.0f };
		float rigidBodyYawDeg = 0.0f; // turn around world Y
		float rigidBodyPitchDeg = 0.0f; // tilt around local Z
		float rigidBodyRollDeg = 0.0f; // tilt around local X

		// rigid body material and motion
		float rigidDensity = 1'000.0f;
		glm::vec3 rigidInitVel{ 0.0f };
		glm::vec3 rigidInitOmega{ 0.0f };
	};

	struct SampledMeshCollectionConfig
	{
		// outer boundary container
		OBB container;

		// fluid block placed inside
		OBB fluid;

		// independently simulated sampled mesh bodies
		std::vector<SampledMeshCollectionRigidBodyConfig> rigidBodies;
	};

	class SampledMeshCollectionScene final : public ISimScenario
	{
	public:

		SampledMeshCollectionScene(ScenarioMgr& scenarioMgr, SampledMeshCollectionConfig cfg)
			: scenarioMgr_(scenarioMgr), cfg_(std::move(cfg)) {}

		const char* Name() const override { return "Sampled Mesh Collection"; }

		void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const override
		{
			scenarioMgr_.BuildBoundaryContainer_OBB(cfg_.container, dst, params);

			scenarioMgr_.BuildFluidBlock_OBB(cfg_.fluid, dst, params);

			const std::filesystem::path modelDirectory =
				std::filesystem::path(TSP_CONFIG_DIR).parent_path() /
				"assets" / "models" / "rigid_meshes";
			const ObjMeshLoader loader(modelDirectory);

			for (const SampledMeshCollectionRigidBodyConfig& rigidBody : cfg_.rigidBodies)
			{
				if (!std::isfinite(rigidBody.modelTargetExtent) ||
					rigidBody.modelTargetExtent <= 0.0f)
				{
					throw std::runtime_error("Rigid mesh model target extent must be > 0.");
				}
				if (!std::isfinite(rigidBody.modelCenterOfMassOffset.x) ||
					!std::isfinite(rigidBody.modelCenterOfMassOffset.y) ||
					!std::isfinite(rigidBody.modelCenterOfMassOffset.z))
				{
					throw std::runtime_error("Rigid mesh center-of-mass offset must be finite.");
				}

				const glm::mat3 scenarioOrientation =
					scenarioMgr_.MakeOrientation_FromEulerAngles(
						rigidBody.rigidBodyYawDeg,
						rigidBody.rigidBodyPitchDeg,
						rigidBody.rigidBodyRollDeg);
				const glm::mat3 rigidBodyOrientation =
					scenarioOrientation * rigidBody.modelSourceToCanonical;

				TriangleSurfaceMesh mesh = loader.Load(rigidBody.modelFile);

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
				const float modelScale = rigidBody.modelTargetExtent / rawModelExtent;
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
					propertyComputer.Compute(mesh, rigidBody.rigidDensity);

				// keep the mesh-derived centroidal inertia as the effective body inertia
				// shift the sampled model into its effective center-of-mass frame
				const glm::vec3 centerOfMassOffset =
					glm::transpose(rigidBody.modelSourceToCanonical) *
					(rigidBody.modelCenterOfMassOffset * rigidBody.modelTargetExtent);
				const glm::vec3 centerOfMass = properties.centerOfMass + centerOfMassOffset;
				const glm::vec3 rigidBodyCenterOfMass = rigidBody.rigidBodyCenterOfPosition +
					scenarioOrientation *
					(rigidBody.modelCenterOfMassOffset * rigidBody.modelTargetExtent);
				for (glm::vec3& vertex : mesh.vertices)
				{
					vertex -= centerOfMass;
				}

				SPDLOG_INFO("Sampling model '{}' surfaces...", rigidBody.modelName);
				SurfaceSampler sampler(mesh, params);
				const SurfaceSamplingResult sampling = sampler.Sample();
				const uint32_t bodyIdx = scenarioMgr_.BuildRigidBody_FromSurfaceSamples(
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
						"SampledMeshCollection model '" + rigidBody.modelName +
						"' produced no rigid surface samples.");
				}

				ApplyInitialRigidBodyState(dst.rigidBodies, bodyIdx, rigidBody);
			}
		}

		std::vector<OBB> GetBoundaryBoxes() const override
		{
			return { cfg_.container };
		}

	private:

		static void ApplyInitialRigidBodyState(
			Sync::RigidBodies& rigidBodies,
			const uint32_t bodyIdx,
			const SampledMeshCollectionRigidBodyConfig& cfg)
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

		ScenarioMgr& scenarioMgr_;
		SampledMeshCollectionConfig cfg_;
	};

	inline SampledMeshCollectionRigidBodyConfig
		MakeSampledMeshCollectionRigidBodyConfig_Default()
	{
		SampledMeshCollectionRigidBodyConfig cfg{};

		cfg.modelName = "UnitCube";
		cfg.modelFile = "unit_cube.obj";
		cfg.modelTargetExtent = 0.16f;
		cfg.rigidBodyCenterOfPosition = { 0.5f, 0.25f, 0.5f };
		cfg.rigidBodyYawDeg = 0.0f;
		cfg.rigidBodyPitchDeg = 0.0f;
		cfg.rigidBodyRollDeg = 0.0f;
		cfg.rigidDensity = 200.0f;
		cfg.rigidInitVel = { 0.0f, 0.0f, 0.0f };
		cfg.rigidInitOmega = { 0.0f, 0.0f, 0.0f };

		return cfg;
	}

	inline SampledMeshCollectionConfig MakeSampledMeshCollectionConfig_Default(
		ScenarioMgr& scenarioMgr)
	{
		constexpr glm::vec3 U{ 1, 0, 0 }; // forward
		constexpr glm::vec3 V{ 0, 1, 0 }; // up
		constexpr glm::vec3 W{ 0, 0, 1 }; // right

		SampledMeshCollectionConfig cfg{};

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

		const std::array<glm::vec3, 4> centers
		{
			glm::vec3{ 0.3f, 0.25f, 0.3f },
			glm::vec3{ 0.7f, 0.25f, 0.3f },
			glm::vec3{ 0.3f, 0.25f, 0.7f },
			glm::vec3{ 0.7f, 0.25f, 0.7f }
		};
		cfg.rigidBodies.resize(centers.size());
		for (uint32_t bodyIdx = 0; bodyIdx != cfg.rigidBodies.size(); bodyIdx++)
		{
			cfg.rigidBodies[bodyIdx] = MakeSampledMeshCollectionRigidBodyConfig_Default();
			cfg.rigidBodies[bodyIdx].rigidBodyCenterOfPosition = centers[bodyIdx];
		}

		return cfg;
	}

	inline SampledMeshCollectionConfig LoadSampledMeshCollectionConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		// cache default config
		SampledMeshCollectionConfig cfg = MakeSampledMeshCollectionConfig_Default(scenarioMgr);

		if (auto it = j.find("type"); it != j.end() && it->is_string())
		{
			const std::string type = it->get<std::string>();
			if (type != "SampledMeshCollection")
			{
				SPDLOG_WARN(
					"SampledMeshCollection entry has type '{}' (expected 'SampledMeshCollection').",
					type);
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

		auto rigidBodies = j.find("rigidBodies");
		if (rigidBodies == j.end() || !rigidBodies->is_array() || rigidBodies->empty())
		{
			throw std::runtime_error(
				"SampledMeshCollection 'rigidBodies' must be a non-empty array.");
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

		cfg.rigidBodies.clear();
		cfg.rigidBodies.reserve(rigidBodies->size());
		for (uint32_t bodyIdx = 0; bodyIdx != rigidBodies->size(); bodyIdx++)
		{
			const nlohmann::json& rigidBodyJson = (*rigidBodies)[bodyIdx];
			if (!rigidBodyJson.is_object())
			{
				throw std::runtime_error(
					"SampledMeshCollection rigid body entries must be objects.");
			}

			SampledMeshCollectionRigidBodyConfig rigidBody =
				MakeSampledMeshCollectionRigidBodyConfig_Default();
			auto model = rigidBodyJson.find("model");
			if (model == rigidBodyJson.end() || !model->is_string())
			{
				throw std::runtime_error(
					"SampledMeshCollection rigid body requires a string 'model'.");
			}
			rigidBody.modelName = model->get<std::string>();

			if (auto value = rigidBodyJson.find("modelTargetExtent");
				value != rigidBodyJson.end())
			{
				rigidBody.modelTargetExtent = value->get<float>();
			}
			if (rigidBodyJson.find("modelScale") != rigidBodyJson.end())
			{
				throw std::runtime_error(
					"SampledMeshCollection configuration uses 'modelScale'; "
					"use 'modelTargetExtent' instead.");
			}

			auto rigidBodyState = rigidBodyJson.find("rigidBody");
			if (rigidBodyState == rigidBodyJson.end() || !rigidBodyState->is_object())
			{
				throw std::runtime_error(
					"SampledMeshCollection rigid body requires a 'rigidBody' object.");
			}
			if (rigidBodyState->contains("centerOfMass"))
			{
				throw std::runtime_error(
					"SampledMeshCollection rigidBody uses 'centerOfMass'; "
					"use 'centerOfPosition' instead.");
			}
			auto centerOfPosition = rigidBodyState->find("centerOfPosition");
			if (centerOfPosition == rigidBodyState->end() ||
				!centerOfPosition->is_array() || centerOfPosition->size() != 3 ||
				!(*centerOfPosition)[0].is_number() ||
				!(*centerOfPosition)[1].is_number() ||
				!(*centerOfPosition)[2].is_number())
			{
				throw std::runtime_error(
					"SampledMeshCollection rigid body requires a three-value 'centerOfPosition'.");
			}
			ReadVec3(*centerOfPosition, rigidBody.rigidBodyCenterOfPosition);

			if (rigidBodyState->contains("uDir") ||
				rigidBodyState->contains("vDir") ||
				rigidBodyState->contains("wDir"))
			{
				throw std::runtime_error(
					"SampledMeshCollection rigidBody uses 'uDir/vDir/wDir'; "
					"use 'yawDeg/pitchDeg/rollDeg' instead.");
			}

			if (auto value = rigidBodyState->find("yawDeg");
				value != rigidBodyState->end())
			{
				rigidBody.rigidBodyYawDeg = value->get<float>();
			}
			if (auto value = rigidBodyState->find("pitchDeg");
				value != rigidBodyState->end())
			{
				rigidBody.rigidBodyPitchDeg = value->get<float>();
			}
			if (auto value = rigidBodyState->find("rollDeg");
				value != rigidBodyState->end())
			{
				rigidBody.rigidBodyRollDeg = value->get<float>();
			}

			if (auto value = rigidBodyJson.find("rigidDensity");
				value != rigidBodyJson.end())
			{
				rigidBody.rigidDensity = value->get<float>();
			}

			if (auto value = rigidBodyJson.find("rigidInitVel");
				value != rigidBodyJson.end() && value->is_array() && value->size() == 3)
			{
				ReadVec3(*value, rigidBody.rigidInitVel);
			}

			if (auto value = rigidBodyJson.find("rigidInitOmega");
				value != rigidBodyJson.end() && value->is_array() && value->size() == 3)
			{
				ReadVec3(*value, rigidBody.rigidInitOmega);
			}

			cfg.rigidBodies.push_back(std::move(rigidBody));
		}

		// sanitize fluid lengths
		cfg.fluid.uLen = std::min(std::max(0.0f, cfg.fluid.uLen), cfg.container.uLen);
		cfg.fluid.vLen = std::min(std::max(0.0f, cfg.fluid.vLen), cfg.container.vLen);
		cfg.fluid.wLen = std::min(std::max(0.0f, cfg.fluid.wLen), cfg.container.wLen);

		return cfg;
	}

	inline SampledMeshCollectionConfig LoadSampledMeshCollectionConfigFromProfile(
		const std::string& profile, ScenarioMgr& scenarioMgr)
	{
		// try to load a SampledMeshCollection scenario from the config file of a given profile
		auto TryLoadFromScenarioFile = [&](const std::string& scenariosFile)
			-> std::optional<SampledMeshCollectionConfig>
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

				// expect scenarios["SampledMeshCollection"] object
				if (!scenarios.contains("SampledMeshCollection") ||
					!scenarios["SampledMeshCollection"].is_object())
				{
					return std::nullopt;
				}

				return LoadSampledMeshCollectionConfigFromJson(
					scenarios["SampledMeshCollection"], scenarioMgr);
			};

		// try to resolve a model nickname to a file path and optional source-to-canonical orientation
		auto TryResolveModel = [](const std::filesystem::path& modelsFile,
			SampledMeshCollectionRigidBodyConfig& cfg) -> bool
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
							"SampledMeshCollection model '" + cfg.modelName +
							"' sourceToCanonical must be an object.");
					}

					auto ReadAxis = [&](const char* axisName) -> glm::vec3
						{
							if (!it->contains(axisName) ||
								!(*it)[axisName].is_array() ||
								(*it)[axisName].size() != 3)
							{
								throw std::runtime_error(
									"SampledMeshCollection model '" + cfg.modelName +
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
									"SampledMeshCollection model '" + cfg.modelName +
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
							"SampledMeshCollection model '" + cfg.modelName +
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
								"SampledMeshCollection model '" + cfg.modelName +
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
								"SampledMeshCollection model '" + cfg.modelName +
								"' centerOfMassOffset values must be finite.");
						}
					}
				}

				return true;
			};

		// load global model catalog from assets and resolve models below /assets/models
		const std::filesystem::path assetDirectory =
			std::filesystem::path(TSP_CONFIG_DIR).parent_path() / "assets";
		const std::filesystem::path modelCatalogPath =
			assetDirectory / "models" / "rigid_models.json";

		auto LoadProfile = [&](const std::string& profileName)
			-> std::optional<SampledMeshCollectionConfig>
			{
				const auto cfg = TryLoadFromScenarioFile(Base::ScenariosPathForProfile(profileName));
				if (!cfg) { return std::nullopt; }

				SampledMeshCollectionConfig resolved = *cfg;
				for (SampledMeshCollectionRigidBodyConfig& rigidBody : resolved.rigidBodies)
				{
					if (TryResolveModel(modelCatalogPath, rigidBody)) { continue; }

					throw std::runtime_error(
						"SampledMeshCollection model '" + rigidBody.modelName +
						"' is not defined in the global model catalog '" +
						modelCatalogPath.string() + "'.");
				}
				return resolved;
			};

		// try chosen profile
		if (auto cfg = LoadProfile(profile)) { return *cfg; }
		SPDLOG_WARN(
			"Could not load SampledMeshCollection from '/{}'.",
			Base::ScenariosPathForProfile(profile));

		// fallback to default profile
		if (profile != "default")
		{
			if (auto cfg = LoadProfile("default")) { return *cfg; }
			SPDLOG_WARN(
				"Could not load SampledMeshCollection from '/{}'.",
				Base::ScenariosPathForProfile("default"));
		}

		// final fallback to built-in defaults
		SPDLOG_WARN("Using built-in SampledMeshCollection defaults.");
		return MakeSampledMeshCollectionConfig_Default(scenarioMgr);
	}

} // namespace Scene
