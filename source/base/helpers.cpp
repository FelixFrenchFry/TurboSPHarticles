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
