// rigid_cube.h

#pragma once

// external
#include <algorithm>
#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>

// internal
#include "source/base/helpers.h"
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/sync/state_sim.h"
#include "builder.h"
#include "scenario_sim.h"



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
	public: // ---------------------------------------------------------------------------------------

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

	inline RigidCubeConfig MakeRigidCubeConfig_Default(ScenarioMgr& scenarioMgr)
	{
		constexpr glm::vec3 U{ 1, 0, 0 }; // forward
		constexpr glm::vec3 V{ 0, 1, 0 }; // up
		constexpr glm::vec3 W{ 0, 0, 1 }; // right

		RigidCubeConfig cfg{};

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
			/* uDir, uLen */	U, 0.45f,
			/* vDir, vLen */	V, 0.45f,
			/* wDir, wLen */	W, 0.45f
		);

		cfg.rigidBody = scenarioMgr.MakeOBB_FromAxes
		(
			/* origin */		{ 0.62f,0.12f,0.42f },
			/* uDir, uLen */	U, 0.16f,
			/* vDir, vLen */	V, 0.16f,
			/* wDir, wLen */	W, 0.16f
		);

		cfg.rigidDensity = 1'000.0f;
		cfg.rigidInitVel = { 0.0f, 0.0f, 0.0f };
		cfg.rigidInitOmega = { 0.0f, 0.0f, 0.0f };

		return cfg;
	}

	inline RigidCubeConfig LoadRigidCubeConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		// cache default config
		RigidCubeConfig cfg = MakeRigidCubeConfig_Default(scenarioMgr);

		if (auto it = j.find("container"); it != j.end())
		{
			cfg.container = ScenarioMgr::ParseOBBFromJson(*it, scenarioMgr);
		}

		if (auto it = j.find("fluid"); it != j.end())
		{
			cfg.fluid = ScenarioMgr::ParseOBBFromJson(*it, scenarioMgr);
		}

		if (auto it = j.find("rigidBody"); it != j.end())
		{
			cfg.rigidBody = ScenarioMgr::ParseOBBFromJson(*it, scenarioMgr);
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
				if (!scenarios.contains("RigidCube") || !scenarios["RigidCube"].is_object())
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

		// final fallback to built-in defaults
		SPDLOG_WARN("Using built-in RigidCube defaults.");
		return MakeRigidCubeConfig_Default(scenarioMgr);
	}

} // namespace Scene
