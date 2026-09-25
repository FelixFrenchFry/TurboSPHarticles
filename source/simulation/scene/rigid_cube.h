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
		RigidCubeConfig cfg{};

		auto container = j.find("container");
		if (container == j.end())
		{
			throw std::runtime_error("RigidCube requires a 'container' object.");
		}
		cfg.container = ScenarioMgr::ParseOBBFromJson(*container, scenarioMgr);

		auto fluid = j.find("fluid");
		if (fluid == j.end())
		{
			throw std::runtime_error("RigidCube requires a 'fluid' object.");
		}
		cfg.fluid = ScenarioMgr::ParseOBBFromJson(*fluid, scenarioMgr);

		auto rigidBody = j.find("rigidBody");
		if (rigidBody == j.end())
		{
			throw std::runtime_error("RigidCube requires a 'rigidBody' object.");
		}
		cfg.rigidBody = ScenarioMgr::ParseOBBFromJson(*rigidBody, scenarioMgr);

		auto rigidDensity = j.find("rigidDensity");
		if (rigidDensity == j.end())
		{
			throw std::runtime_error("RigidCube requires a 'rigidDensity' value.");
		}
		cfg.rigidDensity = rigidDensity->get<float>();

		auto rigidInitVel = j.find("rigidInitVel");
		if (rigidInitVel == j.end() ||
			!rigidInitVel->is_array() || rigidInitVel->size() != 3)
		{
			throw std::runtime_error("RigidCube requires a three-value 'rigidInitVel'.");
		}
		cfg.rigidInitVel.x = (*rigidInitVel)[0].get<float>();
		cfg.rigidInitVel.y = (*rigidInitVel)[1].get<float>();
		cfg.rigidInitVel.z = (*rigidInitVel)[2].get<float>();

		auto rigidInitOmega = j.find("rigidInitOmega");
		if (rigidInitOmega == j.end() ||
			!rigidInitOmega->is_array() || rigidInitOmega->size() != 3)
		{
			throw std::runtime_error("RigidCube requires a three-value 'rigidInitOmega'.");
		}
		cfg.rigidInitOmega.x = (*rigidInitOmega)[0].get<float>();
		cfg.rigidInitOmega.y = (*rigidInitOmega)[1].get<float>();
		cfg.rigidInitOmega.z = (*rigidInitOmega)[2].get<float>();

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

		throw std::runtime_error(
			"RigidCube is not defined in profile '" + profile + "' or the default profile.");
	}

} // namespace Scene
