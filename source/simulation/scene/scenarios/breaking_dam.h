// breaking_dam.h
#pragma once

// external
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

// internal
#include "source/base/helpers.h"
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/sync/state_sim.h"
#include "source/simulation/scene/builder.h"
#include "source/simulation/scene/scenario_sim.h"
#include "source/simulation/scene/scenario_json.h"



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
	public:

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

	private:

		ScenarioMgr& scenarioMgr_;
		BreakingDamConfig cfg_;
	};

	inline BreakingDamConfig LoadBreakingDamConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		ValidateScenarioType(j, "BreakingDam");
		BreakingDamConfig cfg{};

		cfg.container = ScenarioMgr::ParseOBBFromJson(
			ReadScenarioObject(j, "container", "BreakingDam"), scenarioMgr, "BreakingDam.container");
		cfg.fluid = ScenarioMgr::ParseOBBFromJson(
			ReadScenarioObject(j, "fluid", "BreakingDam"), scenarioMgr, "BreakingDam.fluid");

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
				if (!scenarios.contains("BreakingDam"))
				{
					return std::nullopt;
				}

				const auto& jBD = scenarios["BreakingDam"];

				return LoadBreakingDamConfigFromJson(jBD, scenarioMgr);
			};

		// try chosen profile
		{
			const std::string file = Base::ScenariosPathForProfile(profile);
			if (auto cfg = TryLoadFromScenarioFile(file)) { return *cfg; }

			SPDLOG_WARN("Could not load BreakingDam from '/{}'.", file);
		}

		// fallback to default profile if profile was not 'default' already
		if (profile != "default")
		{
			const std::string file = Base::ScenariosPathForProfile("default");
			if (auto cfg = TryLoadFromScenarioFile(file)) { return *cfg; }

			SPDLOG_WARN("Could not load BreakingDam from '/{}'.", file);
		}

		throw std::runtime_error(
			"BreakingDam is not defined in profile '" + profile + "' or the default profile.");
	}

} // namespace Scene
