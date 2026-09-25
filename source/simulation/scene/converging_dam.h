// converging_dam.h
#pragma once

// external
#include <algorithm>
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

	inline ConvergingDamConfig LoadConvergingDamConfigFromJson(
		const nlohmann::json& j, ScenarioMgr& scenarioMgr)
	{
		ValidateScenarioType(j, "ConvergingDam");
		ConvergingDamConfig cfg{};

		cfg.container = ScenarioMgr::ParseOBBFromJson(
			ReadScenarioObject(j, "container", "ConvergingDam"), scenarioMgr, "ConvergingDam.container");
		cfg.fluidHeight = ReadScenarioFloat(j, "fluidHeight", "ConvergingDam");
		cfg.holeSize = ReadScenarioFloat(j, "holeSize", "ConvergingDam");

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
				if (!scenarios.contains("ConvergingDam"))
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

		throw std::runtime_error(
			"ConvergingDam is not defined in profile '" + profile + "' or the default profile.");
	}

} // namespace Scene
