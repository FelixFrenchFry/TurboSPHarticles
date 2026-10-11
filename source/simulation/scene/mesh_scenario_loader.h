// mesh_scenario_loader.h
#pragma once

// external
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

// internal
#include "source/base/helpers.h"



namespace Scene
{
	// loads a mesh scenario from the chosen profile, with the default profile as fallback
	template <typename Config, typename LoadConfig>
	inline Config LoadMeshScenarioConfigFromProfile(
		const std::string& profile, const std::string& scenarioName, LoadConfig loadConfig)
	{
		auto TryLoadFromScenarioFile = [&](const std::string& scenariosFile)
			-> std::optional<Config>
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
				if (!scenarios.contains(scenarioName))
				{
					return std::nullopt;
				}

				return loadConfig(scenarios[scenarioName]);
			};

		// try chosen profile
		if (auto cfg = TryLoadFromScenarioFile(Base::ScenariosPathForProfile(profile))) { return *cfg; }
		SPDLOG_WARN("Could not load {} from '/{}'.", scenarioName, Base::ScenariosPathForProfile(profile));

		// fallback to default profile
		if (profile != "default")
		{
			if (auto cfg = TryLoadFromScenarioFile(Base::ScenariosPathForProfile("default"))) { return *cfg; }
			SPDLOG_WARN("Could not load {} from '/{}'.", scenarioName, Base::ScenariosPathForProfile("default"));
		}

		throw std::runtime_error(
			scenarioName + " is not defined in profile '" + profile + "' or the default profile.");
	}

} // namespace Scene
