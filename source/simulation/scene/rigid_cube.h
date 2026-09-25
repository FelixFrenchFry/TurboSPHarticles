// rigid_cube.h
#pragma once

// external
#include <algorithm>
#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

// internal
#include "source/base/helpers.h"
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/sync/state_sim.h"
#include "builder.h"
#include "scenario_sim.h"
#include "scenario_json.h"



namespace Scene
{
	struct RigidCubeConfig
	{
		// outer boundary container
		OBB container;

		// fluid block placed inside
		OBB fluid;

		// rigid cube
		OBB rigidBody;

		float rigidDensity;
		glm::vec3 rigidInitVel;
		glm::vec3 rigidInitOmega;
	};

	class RigidCubeScene final : public ISimScenario
	{
	public:

		RigidCubeScene(ScenarioMgr& scenarioMgr, RigidCubeConfig cfg)
			: scenarioMgr_(scenarioMgr), cfg_(cfg) {}

		const char* Name() const override { return "Rigid Cube"; }

		void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const override
		{
			scenarioMgr_.BuildBoundaryContainer_OBB(cfg_.container, dst, params);

			scenarioMgr_.BuildFluidBlock_OBB(cfg_.fluid, dst, params);

			const uint32_t bodyIdx = scenarioMgr_.BuildRigidBody_OBB_Hollow(
				cfg_.rigidBody, dst, params, cfg_.rigidDensity);
			if (bodyIdx == Sync::InvalidRigidBodyId) { return; }

			ApplyInitialRigidBodyState(dst.rigidBodies, bodyIdx);
		}

		std::vector<OBB> GetBoundaryBoxes() const override
		{
			return { cfg_.container };
		}

	private:

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

			const float omegaBodyX =
				RB.a00[bodyIdx] * cfg_.rigidInitOmega.x +
				RB.a10[bodyIdx] * cfg_.rigidInitOmega.y +
				RB.a20[bodyIdx] * cfg_.rigidInitOmega.z;
			const float omegaBodyY =
				RB.a01[bodyIdx] * cfg_.rigidInitOmega.x +
				RB.a11[bodyIdx] * cfg_.rigidInitOmega.y +
				RB.a21[bodyIdx] * cfg_.rigidInitOmega.z;
			const float omegaBodyZ =
				RB.a02[bodyIdx] * cfg_.rigidInitOmega.x +
				RB.a12[bodyIdx] * cfg_.rigidInitOmega.y +
				RB.a22[bodyIdx] * cfg_.rigidInitOmega.z;

			const float iBodyX = RB.iBodyInv00[bodyIdx] > 0.0f
				? 1.0f / RB.iBodyInv00[bodyIdx] : 0.0f;
			const float iBodyY = RB.iBodyInv11[bodyIdx] > 0.0f
				? 1.0f / RB.iBodyInv11[bodyIdx] : 0.0f;
			const float iBodyZ = RB.iBodyInv22[bodyIdx] > 0.0f
				? 1.0f / RB.iBodyInv22[bodyIdx] : 0.0f;

			const float angMomBodyX = iBodyX * omegaBodyX;
			const float angMomBodyY = iBodyY * omegaBodyY;
			const float angMomBodyZ = iBodyZ * omegaBodyZ;

			RB.xAngMom[bodyIdx] =
				RB.a00[bodyIdx] * angMomBodyX +
				RB.a01[bodyIdx] * angMomBodyY +
				RB.a02[bodyIdx] * angMomBodyZ;
			RB.yAngMom[bodyIdx] =
				RB.a10[bodyIdx] * angMomBodyX +
				RB.a11[bodyIdx] * angMomBodyY +
				RB.a12[bodyIdx] * angMomBodyZ;
			RB.zAngMom[bodyIdx] =
				RB.a20[bodyIdx] * angMomBodyX +
				RB.a21[bodyIdx] * angMomBodyY +
				RB.a22[bodyIdx] * angMomBodyZ;
		}

		ScenarioMgr& scenarioMgr_;
		RigidCubeConfig cfg_;
	};

	inline RigidCubeConfig LoadRigidCubeConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		ValidateScenarioType(j, "RigidCube");
		RigidCubeConfig cfg{};

		cfg.container = ScenarioMgr::ParseOBBFromJson(
			ReadScenarioObject(j, "container", "RigidCube"), scenarioMgr, "RigidCube.container");
		cfg.fluid = ScenarioMgr::ParseOBBFromJson(
			ReadScenarioObject(j, "fluid", "RigidCube"), scenarioMgr, "RigidCube.fluid");
		cfg.rigidBody = ScenarioMgr::ParseOBBFromJson(
			ReadScenarioObject(j, "rigidBody", "RigidCube"), scenarioMgr, "RigidCube.rigidBody");
		cfg.rigidDensity = ReadScenarioFloat(j, "rigidDensity", "RigidCube");
		cfg.rigidInitVel = ReadScenarioVector<3>(j, "rigidInitVel", "RigidCube");
		cfg.rigidInitOmega = ReadScenarioVector<3>(j, "rigidInitOmega", "RigidCube");

		// sanitize values
		cfg.fluid.uLen = std::min(std::max(0.0f, cfg.fluid.uLen),
								  cfg.container.uLen);
		cfg.fluid.vLen = std::min(std::max(0.0f, cfg.fluid.vLen),
								  cfg.container.vLen);
		cfg.fluid.wLen = std::min(std::max(0.0f, cfg.fluid.wLen),
								  cfg.container.wLen);
		cfg.rigidBody.uLen = std::min(std::max(0.0f, cfg.rigidBody.uLen),
									  cfg.container.uLen);
		cfg.rigidBody.vLen = std::min(std::max(0.0f, cfg.rigidBody.vLen),
									  cfg.container.vLen);
		cfg.rigidBody.wLen = std::min(std::max(0.0f, cfg.rigidBody.wLen),
									  cfg.container.wLen);
		cfg.rigidDensity = std::max(0.0f, cfg.rigidDensity);

		return cfg;
	}

	inline RigidCubeConfig LoadRigidCubeConfigFromProfile(
		const std::string& profile, ScenarioMgr& scenarioMgr)
	{
		auto TryLoadFromScenarioFile = [&](const std::string& scenariosFile) -> std::optional<RigidCubeConfig>
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

				// expect scenarios["RigidCube"] object
				if (!scenarios.contains("RigidCube"))
				{
					return std::nullopt;
				}

				return LoadRigidCubeConfigFromJson(scenarios["RigidCube"], scenarioMgr);
			};

		// try chosen profile
		{
			const std::string file = Base::ScenariosPathForProfile(profile);
			if (auto cfg = TryLoadFromScenarioFile(file)) { return *cfg; }

			SPDLOG_WARN("Could not load RigidCube from '/{}'.", file);
		}

		// fallback to default profile if profile was not 'default' already
		if (profile != "default")
		{
			const std::string file = Base::ScenariosPathForProfile("default");
			if (auto cfg = TryLoadFromScenarioFile(file)) { return *cfg; }

			SPDLOG_WARN("Could not load RigidCube from '/{}'.", file);
		}

		throw std::runtime_error(
			"RigidCube is not defined in profile '" + profile + "' or the default profile.");
	}

} // namespace Scene
