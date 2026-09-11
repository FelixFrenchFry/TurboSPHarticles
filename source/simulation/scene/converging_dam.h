// converging_dam.h
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
	struct ConvergingDamConfig
	{
		// outer boundary container
		OBB container;

		// fluid definition
		float fluidHeight;
		float holeSize;
	};

	class ConvergingDamScene final : public ISimScenario
	{
	public:

		ConvergingDamScene(ScenarioMgr& scenarioMgr, ConvergingDamConfig cfg)
			: scenarioMgr_(scenarioMgr), cfg_(cfg) {}

		const char* Name() const override { return "Converging Dam"; }

		void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const override
		{
			scenarioMgr_.BuildBoundaryContainer_OBB(cfg_.container, dst, params);

			// fluid domain with specified height
			OBB fluid = cfg_.container;
			fluid.vLen = std::min(std::max(0.0f, cfg_.fluidHeight),
								  cfg_.container.vLen);

			if (cfg_.holeSize <= 0.0f) // just fill normally
			{
				scenarioMgr_.BuildFluidBlock_OBB(fluid, dst, params);
				return;
			}

			// compute centered square hole bounds in local u/w
			const float hole = std::min({cfg_.holeSize, fluid.uLen, fluid.wLen});

			const float holeMinU = 0.5f * (fluid.uLen - hole);
			const float holeMaxU = holeMinU + hole;

			const float holeMinW = 0.5f * (fluid.wLen - hole);
			const float holeMaxW = holeMinW + hole;

			// predicate to filter out particles inside the hole
			auto keep = [&](float u, float, float w) -> bool
				{
					const bool inHole = (u >= holeMinU && u <= holeMaxU) &&
										(w >= holeMinW && w <= holeMaxW);
					return !inHole;
				};

			scenarioMgr_.BuildFluidBlock_OBB_Filtered(fluid, dst, params, keep);
		}

		std::vector<OBB> GetBoundaryBoxes() const override
		{
			return { cfg_.container };
		}

	private:

		ScenarioMgr& scenarioMgr_;
		ConvergingDamConfig cfg_;
	};

	inline ConvergingDamConfig MakeConvergingDamConfig_Default(ScenarioMgr& scenarioMgr)
	{
		constexpr glm::vec3 U{ 1, 0, 0 }; // forward
		constexpr glm::vec3 V{ 0, 1, 0 }; // up
		constexpr glm::vec3 W{ 0, 0, 1 }; // right

		ConvergingDamConfig cfg{};

		cfg.container = scenarioMgr.MakeOBB_FromAxes
		(
			/* origin */		{ 0,0,0 },
			/* uDir, uLen */	U, 1.0f,
			/* vDir, vLen */	V, 1.0f,
			/* wDir, wLen */	W, 1.0f
		);

		cfg.fluidHeight = 0.5f;
		cfg.holeSize = 0.25f;

		return cfg;
	}

	inline ConvergingDamConfig LoadConvergingDamConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		// cache default config
		ConvergingDamConfig cfg = MakeConvergingDamConfig_Default(scenarioMgr);

		if (auto it = j.find("container"); it != j.end())
		{
			cfg.container = ScenarioMgr::ParseOBBFromJson(*it, scenarioMgr);
		}

		if (auto it = j.find("fluidHeight"); it != j.end())
		{
			cfg.fluidHeight = it->get<float>();
		}

		if (auto it = j.find("holeSize"); it != j.end())
		{
			cfg.holeSize = it->get<float>();
		}

		// sanitize values
		cfg.fluidHeight = std::min(std::max(0.0f, cfg.fluidHeight),
								   cfg.container.vLen);
		const float maxHoleSize = std::min(cfg.container.uLen, cfg.container.wLen);
		cfg.holeSize = std::min(std::max(0.0f, cfg.holeSize),
								maxHoleSize);

		return cfg;
	}

	inline ConvergingDamConfig LoadConvergingDamConfigFromProfile(
		const std::string& profile, ScenarioMgr& scenarioMgr)
	{
		auto TryLoadFromScenarioFile = [&](const std::string& scenariosFile) -> std::optional<ConvergingDamConfig>
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

				// expect scenarios["ConvergingDam"] object
				if (!scenarios.contains("ConvergingDam") || !scenarios["ConvergingDam"].is_object())
				{
					return std::nullopt;
				}

				return LoadConvergingDamConfigFromJson(scenarios["ConvergingDam"], scenarioMgr);
			};

		// try chosen profile
		{
			const std::string file = Base::ScenariosPathForProfile(profile);
			if (auto cfg = TryLoadFromScenarioFile(file)) { return *cfg; }

			SPDLOG_WARN("Could not load ConvergingDam from '/{}'.", file);
		}

		// fallback to default profile if profile was not 'default' already
		if (profile != "default")
		{
			const std::string file = Base::ScenariosPathForProfile("default");
			if (auto cfg = TryLoadFromScenarioFile(file)) { return *cfg; }

			SPDLOG_WARN("Could not load ConvergingDam from '/{}'.", file);
		}

		// final fallback to built-in defaults
		SPDLOG_WARN("Using built-in ConvergingDam defaults.");
		return MakeConvergingDamConfig_Default(scenarioMgr);
	}

} // namespace Scene
