// big_droplet.h

#pragma once

// external
#include <algorithm>
#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <tbb/parallel_for.h>

// internal
#include "source/base/helpers.h"
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/sync/state_sim.h"
#include "builder.h"
#include "scenario_sim.h"



namespace Scene
{
	struct BigDropletConfig
	{
		// outer boundary container
		OBB container;

		// fluid pool
		float poolDepth;

		// fluid droplet
		float dropletRadius;
		float dropletGap;
		glm::vec2 dropletOffset;
		glm::vec3 dropletInitVel;
	};

	class BigDropletScene : public ISimScenario
	{
	public: // ---------------------------------------------------------------------------------------

		BigDropletScene(ScenarioMgr& scenarioMgr, BigDropletConfig cfg)
			: scenarioMgr_(scenarioMgr), cfg_(cfg) {}

		const char* Name() const override { return "Big Droplet"; }

		void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const override
		{
			scenarioMgr_.BuildBoundaryContainer_OBB(cfg_.container, dst, params);

			// fluid pool
			if (cfg_.poolDepth > 0.0f)
			{
				OBB pool = cfg_.container;
				pool.vLen = cfg_.poolDepth;
				scenarioMgr_.BuildFluidBlock_OBB(pool, dst, params);
			}

			// fluid droplet
			if (cfg_.dropletRadius <= 0.0f) { return; }
			if (cfg_.poolDepth >= cfg_.container.vLen) { return; }

			// center in the container
			const float centerU = 0.5f * cfg_.container.uLen + cfg_.dropletOffset.x;
			const float centerW = 0.5f * cfg_.container.wLen + cfg_.dropletOffset.y;
			const float centerV = cfg_.poolDepth
								+ cfg_.dropletGap
								+ cfg_.dropletRadius;

			// build a cube OBB around the sphere, then filter positions
			OBB cube = cfg_.container;
			cube.origin = cfg_.container.origin
						+ cfg_.container.uHat * (centerU - cfg_.dropletRadius)
						+ cfg_.container.vHat * (centerV - cfg_.dropletRadius)
						+ cfg_.container.wHat * (centerW - cfg_.dropletRadius);
			cube.uLen = 2.0f * cfg_.dropletRadius;
			cube.vLen = 2.0f * cfg_.dropletRadius;
			cube.wLen = 2.0f * cfg_.dropletRadius;

			const uint32_t startIdx = dst.fluidParticles.N_;

			// effective radius to prevent particle centers on the sphere edge
			const float effRadius = std::max(0.0f, cfg_.dropletRadius - params.particleSizeF * 0.5f);
			const float effRadiusSqu = effRadius * effRadius;

			// predicate to filter out particles outside the sphere
			auto keep = [&](float u, float v, float w) -> bool
				{
					const float du = u - cfg_.dropletRadius;
					const float dv = v - cfg_.dropletRadius;
					const float dw = w - cfg_.dropletRadius;
					return (du * du + dv * dv + dw * dw) <= effRadiusSqu;
				};

			scenarioMgr_.BuildFluidBlock_OBB_Filtered(cube, dst, params, keep);

			const uint32_t endIdx = dst.fluidParticles.N_;

			// apply initial velocity to droplet particles (override 0.0 init)
			if (endIdx > startIdx)
			{
				auto& F = dst.fluidParticles;

				tbb::parallel_for(
					tbb::blocked_range<uint32_t>(startIdx, endIdx),
					[&](const tbb::blocked_range<uint32_t>& r)
					{
						for (uint32_t i = r.begin(); i < r.end(); i++)
						{
							F.xVel[i] = cfg_.dropletInitVel.x;
							F.yVel[i] = cfg_.dropletInitVel.y;
							F.zVel[i] = cfg_.dropletInitVel.z;

							F.xVelPred[i] = cfg_.dropletInitVel.x;
							F.yVelPred[i] = cfg_.dropletInitVel.y;
							F.zVelPred[i] = cfg_.dropletInitVel.z;
						}
					}
				);
			}
		}

		std::vector<OBB> GetBoundaryBoxes() const override
		{
			return { cfg_.container };
		}

	private: // --------------------------------------------------------------------------------------

		ScenarioMgr& scenarioMgr_;
		BigDropletConfig cfg_;
	};

	inline BigDropletConfig MakeBigDropletConfig_Default(ScenarioMgr& scenarioMgr)
	{
		constexpr glm::vec3 U{ 1, 0, 0 }; // forward
		constexpr glm::vec3 V{ 0, 1, 0 }; // up
		constexpr glm::vec3 W{ 0, 0, 1 }; // right

		BigDropletConfig cfg{};

		cfg.container = scenarioMgr.MakeOBB_FromAxes
		(
			/* origin */		{ 0,0,0 },
			/* uDir, uLen */	U, 1.0f,
			/* vDir, vLen */	V, 1.0f,
			/* wDir, wLen */	W, 1.0f
		);

		cfg.poolDepth = 0.2f;
		cfg.dropletRadius = 0.1f;
		cfg.dropletGap = 0.4f;
		cfg.dropletOffset = { 0.0f, 0.0f };
		cfg.dropletInitVel = { 0.0f, 0.0f, 0.0f };

		return cfg;
	}

	inline BigDropletConfig LoadBigDropletConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		// cache default config
		BigDropletConfig cfg = MakeBigDropletConfig_Default(scenarioMgr);

		if (auto it = j.find("container"); it != j.end())
		{
			cfg.container = ScenarioMgr::ParseOBBFromJson(*it, scenarioMgr);
		}

		if (auto it = j.find("poolDepth"); it != j.end())
		{
			cfg.poolDepth = it->get<float>();
		}

		if (auto it = j.find("dropletRadius"); it != j.end())
		{
			cfg.dropletRadius = it->get<float>();
		}

		if (auto it = j.find("dropletGap"); it != j.end())
		{
			cfg.dropletGap = it->get<float>();
		}

		if (auto it = j.find("dropletOffset"); it != j.end() &&
			it->is_array() && it->size() == 2)
		{
			cfg.dropletOffset.x = (*it)[0].get<float>();
			cfg.dropletOffset.y = (*it)[1].get<float>();
		}

		if (auto it = j.find("dropletInitVel"); it != j.end() &&
			it->is_array() && it->size() == 3)
		{
			cfg.dropletInitVel.x = (*it)[0].get<float>();
			cfg.dropletInitVel.y = (*it)[1].get<float>();
			cfg.dropletInitVel.z = (*it)[2].get<float>();
		}

		// sanitize values
		cfg.poolDepth = std::min(std::max(0.0f, cfg.poolDepth),
								 cfg.container.vLen);
		cfg.dropletGap = std::max(0.0f, cfg.dropletGap);
		// horizontal droplet radius limit
		const float maxRadius_UW = 0.5f * std::min(cfg.container.uLen, cfg.container.wLen);
		// vertical droplet radius limit
		const float maxRadius_V = 0.5f * std::max(0.0f, cfg.container.vLen - cfg.poolDepth - cfg.dropletGap);
		const float maxRadius = std::min(maxRadius_UW, maxRadius_V);
		cfg.dropletRadius = std::clamp(cfg.dropletRadius, 0.0f, maxRadius);
		const float maxGap = std::max(0.0f, cfg.container.vLen - cfg.poolDepth - 2.0f * cfg.dropletRadius);
		cfg.dropletGap = std::min(std::max(0.0f, cfg.dropletGap), maxGap);
		// droplet offset limits to remain fully inside container
		const float halfU = 0.5f * cfg.container.uLen;
		const float halfW = 0.5f * cfg.container.wLen;
		const float maxOffU = std::max(0.0f, halfU - cfg.dropletRadius);
		const float maxOffW = std::max(0.0f, halfW - cfg.dropletRadius);
		cfg.dropletOffset.x = std::clamp(cfg.dropletOffset.x, -maxOffU, +maxOffU);
		cfg.dropletOffset.y = std::clamp(cfg.dropletOffset.y, -maxOffW, +maxOffW);

		return cfg;
	}

	inline BigDropletConfig LoadBigDropletConfigFromProfile(
		const std::string& profile, ScenarioMgr& scenarioMgr)
	{
		auto TryLoadFromScenarioFile = [&](const std::string& scenariosFile) -> std::optional<BigDropletConfig>
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

				// expect scenarios["BigDroplet"] object
				if (!scenarios.contains("BigDroplet") || !scenarios["BigDroplet"].is_object())
				{
					return std::nullopt;
				}

				return LoadBigDropletConfigFromJson(scenarios["BigDroplet"], scenarioMgr);
			};

		// try chosen profile
		{
			const std::string file = profile + "/scenarios.json";
			if (auto cfg = TryLoadFromScenarioFile(file)) { return *cfg; }

			SPDLOG_WARN("Could not load BigDroplet from '/{}'.", file);
		}

		// fallback to default profile if profile was not 'default' already
		if (profile != "default")
		{
			const std::string file = "default/scenarios.json";
			if (auto cfg = TryLoadFromScenarioFile(file)) { return *cfg; }

			SPDLOG_WARN("Could not load BigDroplet from '/{}'.", file);
		}

		// final fallback to built-in defaults
		SPDLOG_WARN("Using built-in BigDroplet defaults.");
		return MakeBigDropletConfig_Default(scenarioMgr);
	}

} // namespace Scene
