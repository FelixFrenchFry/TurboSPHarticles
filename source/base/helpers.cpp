// helpers.cpp

// external
#include <exception>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>

// internal
#include "helpers.h"



namespace Base
{
	std::string ParamsPathForProfile(const std::string& profile)
	{
		return profile == "default" ? "params_def.json" : profile + "/params.json";
	}

	std::string ScenariosPathForProfile(const std::string& profile)
	{
		return profile == "default" ? "scenarios_def.json" : profile + "/scenarios.json";
	}

	bool LoadJsonObjectFromFile(
		const std::string& fileName,
		nlohmann::json& out)
	{
		const std::filesystem::path path = std::filesystem::path(TSP_CONFIG_DIR) / fileName;
		std::ifstream in(path);
		if (!in)
		{
			SPDLOG_ERROR("Failed to open JSON file '{}'.",
				path.string());
			return false;
		}

		try
		{
			in >> out;
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR("Failed to parse JSON file '{}': {}",
				path.string(), ex.what());
			return false;
		}

		return true;
	}

} // namespace Base
