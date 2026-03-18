// metrics_logger.cpp

// external
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <spdlog/spdlog.h>
#include <sstream>
#include <system_error>

// internal
#include "metrics_logger.h"



namespace Export
{
	namespace
	{
		bool LoadJsonObjectFromFile_Silent(
			const std::string& relPath,
			nlohmann::json& out)
		{
			namespace fs = std::filesystem;

			const fs::path base{ TSP_CONFIG_DIR };
			const fs::path full = base / relPath;

			std::ifstream in(full);
			if (!in) { return false; }

			try
			{
				in >> out;
			}
			catch (const std::exception&)
			{
				return false;
			}

			return out.is_object();
		}
	}

	bool LoggingMgr::BeginRun(
		const std::string& configProfile,
		const std::string& paramsFile,
		const std::string& scenariosFile,
		const Sim::SimParams& simParams)
	{
		EndRun();

		if (!CreateNextRunDirectory())
		{
			SPDLOG_ERROR("Failed to initialize run logging directory");
			return false;
		}

		if (!WriteRunMeta(configProfile, paramsFile, scenariosFile, simParams))
		{
			SPDLOG_ERROR("Failed to write run metadata for '{}'",
				runDir_.string());
			Disable();
			return false;
		}

		if (!OpenMemoryMetricsCsv())
		{
			SPDLOG_ERROR("Failed to open memory metrics CSV for '{}'",
				runDir_.string());
			Disable();
			return false;
		}

		WriteMemoryMetricsHeader();
		if (!memoryMetricsCsv_.good())
		{
			SPDLOG_ERROR("Failed to write memory metrics CSV header for '{}'",
				runDir_.string());
			Disable();
			return false;
		}

		active_ = true;

		SPDLOG_INFO("Writing run logs to '{}'", runDir_.string());

		return true;
	}

	void LoggingMgr::EndRun()
	{
		if (memoryMetricsCsv_.is_open())
		{
			memoryMetricsCsv_.flush();
			memoryMetricsCsv_.close();
		}

		active_ = false;
		runDir_.clear();
	}

	void LoggingMgr::AppendMemoryRecord(
		const Mem::MemRecordStep& record)
	{
		if (!active_ || !memoryMetricsCsv_.is_open()) { return; }

		uint64_t totalUsed = 0;
		uint64_t totalAlloc = 0;
		uint64_t totalOccupied = 0;

		memoryMetricsCsv_ << record.simStep << ','
						  << std::setprecision(17) << record.simTime;

		for (uint8_t i = 0; i < Mem::GetIdx(Mem::MemBucket::Count); i++)
		{
			const Mem::MemUsage& usage = record.mem[i];

			totalUsed += usage.usedBytes;
			totalAlloc += usage.allocBytes;
			totalOccupied += usage.occupiedBytes;

			memoryMetricsCsv_ << ','
							  << usage.usedBytes << ','
							  << usage.allocBytes << ','
							  << usage.occupiedBytes;
		}

		memoryMetricsCsv_ << ','
						  << totalUsed << ','
						  << totalAlloc << ','
						  << totalOccupied << '\n';
		memoryMetricsCsv_.flush();

		if (!memoryMetricsCsv_.good())
		{
			SPDLOG_ERROR("Failed while appending memory metrics to '{}'",
				(runDir_ / "memory_metrics.csv").string());
			Disable();
		}
	}

	bool LoggingMgr::CreateNextRunDirectory()
	{
		namespace fs = std::filesystem;

		std::error_code ec;
		fs::create_directories(loggingOptions_.baseDir, ec);
		if (ec)
		{
			SPDLOG_ERROR("Failed to create Logs base directory '{}': {}",
				loggingOptions_.baseDir.string(), ec.message());
			return false;
		}

		uint64_t nextIdx = 1;
		for (const auto& entry : fs::directory_iterator(loggingOptions_.baseDir, ec))
		{
			if (ec)
			{
				SPDLOG_ERROR("Failed to iterate Logs directory '{}': {}",
					loggingOptions_.baseDir.string(), ec.message());
				return false;
			}
			if (!entry.is_directory()) { continue; }

			const std::string name = entry.path().filename().string();
			if (name.empty()) { continue; }
			if (!std::all_of(name.begin(), name.end(),
				[](const char ch)
				{
					return std::isdigit(static_cast<unsigned char>(ch)) != 0;
				}))
			{
				continue;
			}

			try
			{
				const uint64_t idx = std::stoull(name);
				nextIdx = std::max(nextIdx, idx + 1);
			}
			catch (const std::exception&)
			{
				// ignore malformed directory names
			}
		}

		for (uint64_t idx = nextIdx; ; idx++)
		{
			const fs::path candidate = loggingOptions_.baseDir / std::to_string(idx);
			ec.clear();

			if (fs::create_directory(candidate, ec))
			{
				runDir_ = candidate;
				return true;
			}

			if (ec)
			{
				SPDLOG_ERROR("Failed to create run directory '{}': {}",
					candidate.string(), ec.message());
				return false;
			}
		}
	}

	bool LoggingMgr::OpenMemoryMetricsCsv()
	{
		const std::filesystem::path csvPath = runDir_ / "memory_metrics.csv";

		memoryMetricsCsv_.open(csvPath, std::ios::out | std::ios::trunc);
		if (!memoryMetricsCsv_.is_open())
		{
			return false;
		}

		memoryMetricsCsv_.imbue(std::locale::classic());

		return true;
	}

	bool LoggingMgr::WriteRunMeta(
		const std::string& configProfile,
		const std::string& paramsFile,
		const std::string& scenariosFile,
		const Sim::SimParams& simParams) const
	{
		nlohmann::json paramsJson = nullptr;
		nlohmann::json scenariosJson = nullptr;
		std::string resolvedParamsFile{};
		std::string resolvedScenariosFile{};

		const bool hasParamsJson = TryLoadConfigJson(
			paramsFile, paramsJson, resolvedParamsFile);
		const bool hasScenariosJson = TryLoadConfigJson(
			scenariosFile, scenariosJson, resolvedScenariosFile);

		nlohmann::json meta{};
		meta["configProfile"] = configProfile;
		meta["createdAt"] = MakeTimestampString();
		meta["runDir"] = runDir_.string();

		meta["paramsConfig"] =
		{
			{ "requestedFile", paramsFile },
			{ "resolvedFile", resolvedParamsFile },
			{ "found", hasParamsJson },
			{ "json", paramsJson }
		};

		meta["scenariosConfig"] =
		{
			{ "requestedFile", scenariosFile },
			{ "resolvedFile", resolvedScenariosFile },
			{ "found", hasScenariosJson },
			{ "json", scenariosJson }
		};

		meta["simSetup"] =
		{
			{ "backendVersion", static_cast<int>(simParams.backendVersion) },
			{ "scenarioType", static_cast<int>(simParams.scenarioType) },
			{ "neighborSearchMethod", static_cast<int>(simParams.neighborSearchMethod) },
			{ "neighborListStorage", static_cast<int>(simParams.neighborListStorage) },
			{ "exportEnabled", simParams.exportEnabled },
			{ "exportDir", simParams.exportDir },
			{ "exportInterval", simParams.exportInterval },
			{ "expctdAvgNbrCountHardMax_FF", simParams.expctdAvgNbrCountHardMax_FF },
			{ "expctdAvgNbrCountHardMax_FB", simParams.expctdAvgNbrCountHardMax_FB },
			{ "gridCellSizeRatio", simParams.gridCellSizeRatio },
			{ "maxThreadsSim", simParams.maxThreadsSim }
		};

		const std::filesystem::path metaPath = runDir_ / "run_meta.json";
		std::ofstream out(metaPath, std::ios::out | std::ios::trunc);
		if (!out.is_open())
		{
			return false;
		}

		out.imbue(std::locale::classic());
		out << std::setw(4) << meta << '\n';

		return out.good();
	}

	void LoggingMgr::WriteMemoryMetricsHeader()
	{
		if (!memoryMetricsCsv_.is_open()) { return; }

		memoryMetricsCsv_ << "simStep,simTime";

		for (const auto& def : Mem::Definitions)
		{
			const std::string key = SanitizeCsvKey(def.name);

			memoryMetricsCsv_ << ','
							  << key << "_usedBytes"
							  << ','
							  << key << "_allocBytes"
							  << ','
							  << key << "_occupiedBytes";
		}

		memoryMetricsCsv_ << ','
						  << "total_usedBytes"
						  << ','
						  << "total_allocBytes"
						  << ','
						  << "total_occupiedBytes"
						  << '\n';
	}

	void LoggingMgr::Disable()
	{
		if (memoryMetricsCsv_.is_open())
		{
			memoryMetricsCsv_.close();
		}

		active_ = false;
	}

	bool LoggingMgr::TryLoadConfigJson(
		const std::string& requestedRelPath,
		nlohmann::json& out,
		std::string& resolvedRelPath)
	{
		out = nullptr;
		resolvedRelPath.clear();

		if (LoadJsonObjectFromFile_Silent(requestedRelPath, out))
		{
			resolvedRelPath = requestedRelPath;
			return true;
		}

		const std::filesystem::path requestedPath{ requestedRelPath };
		const std::string fileName = requestedPath.filename().string();
		const std::string parentName = requestedPath.parent_path().filename().string();

		if (parentName != "Default" && !fileName.empty())
		{
			const std::string fallbackRelPath = std::string("Default/") + fileName;

			if (LoadJsonObjectFromFile_Silent(fallbackRelPath, out))
			{
				resolvedRelPath = fallbackRelPath;
				return true;
			}
		}

		return false;
	}

	std::string LoggingMgr::MakeTimestampString()
	{
		const auto now = std::chrono::system_clock::now();
		const std::time_t t = std::chrono::system_clock::to_time_t(now);

		std::tm tm{};
	#if defined(_WIN32)
		localtime_s(&tm, &t);
	#else
		localtime_r(&t, &tm);
	#endif

		std::ostringstream oss;
		oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
		return oss.str();
	}

	std::string LoggingMgr::SanitizeCsvKey(
		std::string_view src)
	{
		std::string out{};
		out.reserve(src.size());

		bool prevUnderscore = false;

		for (const char ch : src)
		{
			if (std::isalnum(static_cast<unsigned char>(ch)))
			{
				out.push_back(static_cast<char>(
					std::tolower(static_cast<unsigned char>(ch))));
				prevUnderscore = false;
			}
			else if (!prevUnderscore)
			{
				out.push_back('_');
				prevUnderscore = true;
			}
		}

		while (!out.empty() && out.front() == '_')
		{
			out.erase(out.begin());
		}
		while (!out.empty() && out.back() == '_')
		{
			out.pop_back();
		}

		return out;
	}

} // namespace Export
