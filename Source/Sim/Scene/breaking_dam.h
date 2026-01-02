// breaking_dam.h
//
// TODO: comment

#pragma once

// external
#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>

// internal
#include "Source/Base/helpers.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Sync/state_sim.h"
#include "builder.h"
#include "scenario_sim.h"



namespace Scene
{
	struct BreakingDamConfig
	{
		// outer boundary container
		OBB container;

		// fluid block placed inside
		OBB fluid;
	};

	class BreakingDamScene final : public ISimScenario
	{
	public: // -----------------------------------------------------------------

		BreakingDamScene(ScenarioMgr& scenarioMgr, BreakingDamConfig cfg)
			: scenarioMgr_(scenarioMgr), cfg_(cfg) {}

		const char* Name() const override { return "Breaking Dam"; }

		void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const override
		{
			scenarioMgr_.BuildBoundaryContainer_OBB(cfg_.container, dst, params);

			scenarioMgr_.BuildFluidBlock_OBB(cfg_.fluid, dst, params);
		}

		std::vector<OBB> GetBoundaryBoxes() const override
		{
			return { cfg_.container };
		}

	private: // ----------------------------------------------------------------

		ScenarioMgr& scenarioMgr_;
		BreakingDamConfig cfg_;
	};

	inline BreakingDamConfig MakeBreakingDamConfig_Default(ScenarioMgr& scenarioMgr)
	{
		constexpr glm::vec3 U{ 1, 0, 0 }; // forward
		constexpr glm::vec3 V{ 0, 1, 0 }; // up
		constexpr glm::vec3 W{ 0, 0, 1 }; // right

		BreakingDamConfig cfg{};

		cfg.container = scenarioMgr.MakeOBB_FromAxes
		(
			/* origin */		{0,0,0},
			/* uDir, uLen */	U, 1.0f,
			/* vDir, vLen */	V, 1.0f,
			/* wDir, wLen */	W, 1.0f
		);

		cfg.fluid = scenarioMgr.MakeOBB_FromAxes
		(
			/* origin */		{0.0,0.0,0.0},
			/* uDir, uLen */	U, 0.5f,
			/* vDir, vLen */	V, 0.5f,
			/* wDir, wLen */	W, 0.5f
		);

		return cfg;
	}

	inline BreakingDamConfig LoadBreakingDamConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		// cache default config
		BreakingDamConfig cfg = MakeBreakingDamConfig_Default(scenarioMgr);

		if (auto it = j.find("container"); it != j.end())
		{
			cfg.container = ScenarioMgr::ParseOBBFromJson(*it, scenarioMgr);
		}

		if (auto it = j.find("fluid"); it != j.end())
		{
			cfg.fluid = ScenarioMgr::ParseOBBFromJson(*it, scenarioMgr);
		}

		// sanitize values
		cfg.fluid.uLen = std::min(std::max(0.0f, cfg.fluid.uLen), 
								  cfg.container.uLen);
		cfg.fluid.vLen = std::min(std::max(0.0f, cfg.fluid.vLen), 
								  cfg.container.vLen);
		cfg.fluid.wLen = std::min(std::max(0.0f, cfg.fluid.wLen), 
								  cfg.container.wLen);

		return cfg;
	}

	inline BreakingDamConfig LoadBreakingDamConfigFromProfile(
		const std::string& profile, ScenarioMgr& scenarioMgr)
	{
		auto TryLoadFromScenarioFile = [&](const std::string& scenariosFile) -> std::optional<BreakingDamConfig>
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

				// expect scenarios["BreakingDam"] object
				if (!scenarios.contains("BreakingDam") || !scenarios["BreakingDam"].is_object())
				{
					return std::nullopt;
				}

				const auto& jBD = scenarios["BreakingDam"];

				// sanity check of "type" field
				if (auto it = jBD.find("type"); it != jBD.end() && it->is_string())
				{
					const std::string t = it->get<std::string>();
					if (t != "BreakingDam")
					{
						SPDLOG_WARN("BreakingDam entry has type='{}' (expected 'BreakingDam')", t);
					}
				}

				return LoadBreakingDamConfigFromJson(jBD, scenarioMgr);
			};

		// try chosen profile
		{
			const std::string file = profile + "/scenarios.json";
			if (auto cfg = TryLoadFromScenarioFile(file)) { return *cfg; }

			SPDLOG_WARN("Could not load BreakingDam from '/{}'.", file);
		}

		// fallback to default profile if profile was not 'Default' already
		if (profile != "Default")
		{
			const std::string file = "Default/scenarios.json";
			if (auto cfg = TryLoadFromScenarioFile(file)) { return *cfg; }

			SPDLOG_WARN("Could not load BreakingDam from '/{}'.", file);
		}

		// final fallback to built-in defaults
		SPDLOG_WARN("Using built-in BreakingDam defaults.");
		return MakeBreakingDamConfig_Default(scenarioMgr);
	}

} // namespace Scene
