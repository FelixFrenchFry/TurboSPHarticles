// boundary_mesh.h
#pragma once

// external
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
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
#include "geometry/surface_sampling.h"
#include "builder.h"
#include "scenario_sim.h"



namespace Scene
{
	struct BoundaryMeshEntryConfig
	{
		// model asset nickname and resolved OBJ file
		std::string modelName;
		std::string modelFile;
		float modelTargetExtent = 1.0f; // target length of the largest local bounding box extent
		glm::mat3 modelSourceToCanonical{ 1.0f }; // local to canonical mesh orientation

		// static boundary mesh pose
		glm::vec3 boundaryMeshCenterOfPosition{ 0.0f }; // world position of the bounding box center
		float boundaryMeshYawDeg = 0.0f; // turn around world Y
		float boundaryMeshPitchDeg = 0.0f; // tilt around local Z
		float boundaryMeshRollDeg = 0.0f; // tilt around local X
	};

	struct BoundaryMeshConfig
	{
		// outer boundary container
		OBB container;

		// fluid block placed inside
		OBB fluid;

		// static single-layer sampled mesh boundary
		BoundaryMeshEntryConfig boundaryMesh;
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
		SurfaceSampler sampler(mesh, params);
		const SurfaceSamplingResult sampling = sampler.Sample();
		scenarioMgr.BuildBoundary_FromSurfaceSamples(
			sampling.samples,
			cfg.boundaryMeshCenterOfPosition,
			boundaryMeshOrientation,
			dst,
			params);
	}

	// reads one boundary mesh entry (model, target extent, and pose)
	inline BoundaryMeshEntryConfig LoadBoundaryMeshEntryConfigFromJson(
		const nlohmann::json& j, const std::string& scenarioName)
	{
		if (!j.is_object())
		{
			throw std::runtime_error(
				scenarioName + " boundary mesh entries must be objects.");
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

		BoundaryMeshEntryConfig cfg{};
		auto model = j.find("model");
		if (model == j.end() || !model->is_string())
		{
			throw std::runtime_error(
				scenarioName + " boundary mesh requires a string 'model'.");
		}
		cfg.modelName = model->get<std::string>();

		if (auto value = j.find("modelTargetExtent"); value != j.end())
		{
			cfg.modelTargetExtent = value->get<float>();
		}

		auto boundaryMesh = j.find("boundaryMesh");
		if (boundaryMesh == j.end() || !boundaryMesh->is_object())
		{
			throw std::runtime_error(
				scenarioName + " boundary mesh requires a 'boundaryMesh' object.");
		}
		auto centerOfPosition = boundaryMesh->find("centerOfPosition");
		if (centerOfPosition == boundaryMesh->end() ||
			!centerOfPosition->is_array() || centerOfPosition->size() != 3 ||
			!(*centerOfPosition)[0].is_number() ||
			!(*centerOfPosition)[1].is_number() ||
			!(*centerOfPosition)[2].is_number())
		{
			throw std::runtime_error(
				scenarioName + " boundary mesh requires a three-value 'centerOfPosition'.");
		}
		ReadVec3(*centerOfPosition, cfg.boundaryMeshCenterOfPosition);

		if (auto value = boundaryMesh->find("yawDeg"); value != boundaryMesh->end())
		{
			cfg.boundaryMeshYawDeg = value->get<float>();
		}
		if (auto value = boundaryMesh->find("pitchDeg"); value != boundaryMesh->end())
		{
			cfg.boundaryMeshPitchDeg = value->get<float>();
		}
		if (auto value = boundaryMesh->find("rollDeg"); value != boundaryMesh->end())
		{
			cfg.boundaryMeshRollDeg = value->get<float>();
		}

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

	class BoundaryMeshScene final : public ISimScenario
	{
	public:

		BoundaryMeshScene(ScenarioMgr& scenarioMgr, BoundaryMeshConfig cfg)
			: scenarioMgr_(scenarioMgr), cfg_(std::move(cfg)) {}

		const char* Name() const override { return "Boundary Mesh"; }

		void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const override
		{
			scenarioMgr_.BuildBoundaryContainer_OBB(cfg_.container, dst, params);

			scenarioMgr_.BuildFluidBlock_OBB(cfg_.fluid, dst, params);

			BuildBoundaryMesh(scenarioMgr_, cfg_.boundaryMesh, dst, params);
		}

		std::vector<OBB> GetBoundaryBoxes() const override
		{
			return { cfg_.container };
		}

	private:

		ScenarioMgr& scenarioMgr_;
		BoundaryMeshConfig cfg_;
	};

	inline BoundaryMeshConfig LoadBoundaryMeshConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		BoundaryMeshConfig cfg{};

		if (auto it = j.find("type"); it != j.end() && it->is_string())
		{
			const std::string type = it->get<std::string>();
			if (type != "BoundaryMesh")
			{
				SPDLOG_WARN("BoundaryMesh entry has type '{}' (expected 'BoundaryMesh').", type);
			}
		}

		auto container = j.find("container");
		if (container == j.end())
		{
			throw std::runtime_error("BoundaryMesh requires a 'container' object.");
		}
		cfg.container = ScenarioMgr::ParseOBBFromJson(*container, scenarioMgr);

		auto fluid = j.find("fluid");
		if (fluid == j.end())
		{
			throw std::runtime_error("BoundaryMesh requires a 'fluid' object.");
		}
		cfg.fluid = ScenarioMgr::ParseOBBFromJson(*fluid, scenarioMgr);

		// the scenario object itself holds the boundary mesh entry keys
		cfg.boundaryMesh = LoadBoundaryMeshEntryConfigFromJson(j, "BoundaryMesh");

		// sanitize fluid lengths
		cfg.fluid.uLen = std::min(std::max(0.0f, cfg.fluid.uLen), cfg.container.uLen);
		cfg.fluid.vLen = std::min(std::max(0.0f, cfg.fluid.vLen), cfg.container.vLen);
		cfg.fluid.wLen = std::min(std::max(0.0f, cfg.fluid.wLen), cfg.container.wLen);

		return cfg;
	}

	inline BoundaryMeshConfig LoadBoundaryMeshConfigFromProfile(
		const std::string& profile, ScenarioMgr& scenarioMgr)
	{
		// try to load a BoundaryMesh scenario from the config file of a given profile
		auto TryLoadFromScenarioFile = [&](const std::string& scenariosFile)
			-> std::optional<BoundaryMeshConfig>
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

				// expect scenarios["BoundaryMesh"] object
				if (!scenarios.contains("BoundaryMesh") || !scenarios["BoundaryMesh"].is_object())
				{
					return std::nullopt;
				}

				return LoadBoundaryMeshConfigFromJson(scenarios["BoundaryMesh"], scenarioMgr);
			};

		// load boundary model catalog from assets and resolve models below /assets/models
		const std::filesystem::path assetDirectory =
			std::filesystem::path(TSP_CONFIG_DIR).parent_path() / "assets";
		const std::filesystem::path modelCatalogPath =
			assetDirectory / "models" / "boundary_models.json";

		auto LoadProfile = [&](const std::string& profileName)
			-> std::optional<BoundaryMeshConfig>
			{
				const auto cfg = TryLoadFromScenarioFile(Base::ScenariosPathForProfile(profileName));
				if (!cfg) { return std::nullopt; }

				BoundaryMeshConfig resolved = *cfg;
				if (TryResolveBoundaryMeshModel(modelCatalogPath, resolved.boundaryMesh, "BoundaryMesh"))
				{
					return resolved;
				}

				throw std::runtime_error(
					"BoundaryMesh model '" + resolved.boundaryMesh.modelName +
					"' is not defined in the boundary model catalog '" +
					modelCatalogPath.string() + "'.");
			};

		// try chosen profile
		if (auto cfg = LoadProfile(profile)) { return *cfg; }
		SPDLOG_WARN(
			"Could not load BoundaryMesh from '/{}'.",
			Base::ScenariosPathForProfile(profile));

		// fallback to default profile
		if (profile != "default")
		{
			if (auto cfg = LoadProfile("default")) { return *cfg; }
			SPDLOG_WARN(
				"Could not load BoundaryMesh from '/{}'.",
				Base::ScenariosPathForProfile("default"));
		}

		throw std::runtime_error(
			"BoundaryMesh is not defined in profile '" + profile + "' or the default profile.");
	}

} // namespace Scene
