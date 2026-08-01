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
#include "Source/Sim/Core/build_config.h"



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

	std::string MakeCompactRunDirString(
		const std::filesystem::path& runDir,
		const std::filesystem::path& baseDir)
	{
		namespace fs = std::filesystem;

		const fs::path base = baseDir.lexically_normal();
		const fs::path logsName = base.filename();
		const fs::path runName = runDir.filename();

		if (!logsName.empty() && !runName.empty())
		{
			return (fs::path("...") / logsName / runName).generic_string();
		}

		return runDir.lexically_normal().generic_string();
	}

} // namespace anonymous

namespace Export
{
	bool LoggingMgr::BeginRun(
		const std::string& configProfile,
		const std::string& paramsFile,
		const std::string& scenariosFile,
		const Sim::SimParams& simParams,
		const bool logStatus)
	{
		EndRun();

		if (!CreateNextRunDirectory())
		{
			return false;
		}

		if (!WriteRunMeta(configProfile, paramsFile, scenariosFile, simParams))
		{
			SPDLOG_ERROR("Failed to write run metadata for '{}'.",
				runDir_.string());
			Disable();
			return false;
		}

		if (!OpenMemoryMetricsCsv())
		{
			SPDLOG_ERROR("Failed to open memory metrics CSV for '{}'.",
				runDir_.string());
			Disable();
			return false;
		}

		if (!OpenBenchmarkMetricsCsv())
		{
			SPDLOG_ERROR("Failed to open benchmark metrics CSV for '{}'.",
				runDir_.string());
			Disable();
			return false;
		}

		WriteMemoryMetricsHeader();
		if (!memoryMetricsCsv_.good())
		{
			SPDLOG_ERROR("Failed to write memory metrics CSV header for '{}'.",
				runDir_.string());
			Disable();
			return false;
		}

		WriteBenchmarkMetricsHeader();
		if (!benchmarkMetricsCsv_.good())
		{
			SPDLOG_ERROR("Failed to write benchmark metrics CSV header for '{}'.",
				runDir_.string());
			Disable();
			return false;
		}

		active_ = true;

		if (logStatus)
		{
			SPDLOG_INFO("Writing run logs to '{}'",
				MakeCompactRunDirString(runDir_, loggingOptions_.baseDir));
		}

		return true;
	}

	void LoggingMgr::EndRun()
	{
		if (memoryMetricsCsv_.is_open())
		{
			memoryMetricsCsv_.flush();
			memoryMetricsCsv_.close();
		}
		if (benchmarkMetricsCsv_.is_open())
		{
			benchmarkMetricsCsv_.flush();
			benchmarkMetricsCsv_.close();
		}

		active_ = false;
		runDir_.clear();
	}

	void LoggingMgr::AppendBenchmarkRecord(
		const Bench::BenchRecordStep& record)
	{
		if (!active_ || !benchmarkMetricsCsv_.is_open()) { return; }

		const double constructionMs =
			static_cast<double>(record.time[Bench::GetIdx(Bench::BenchOp::Neighbors_Construction)]) * 1e-6;
		const double queryMs =
			static_cast<double>(record.time[Bench::GetIdx(Bench::BenchOp::Neighbors_Query)]) * 1e-6;
		const double totalMs =
			static_cast<double>(record.time[Bench::GetIdx(Bench::BenchOp::Total)]) * 1e-6;
		const double rigidStateUpdateMs =
			static_cast<double>(record.time[Bench::GetIdx(Bench::BenchOp::Rigid_StateUpdate)]) * 1e-6;
		const double rigidForceAccumMs =
			static_cast<double>(record.time[Bench::GetIdx(Bench::BenchOp::Rigid_ForceAccumulation)]) * 1e-6;

		benchmarkMetricsCsv_ << record.simStep << ','
							 << std::setprecision(17) << record.simTime << ','
							 << std::setprecision(17) << constructionMs << ','
							 << std::setprecision(17) << queryMs << ','
							 << std::setprecision(17) << totalMs << ','
							 << std::setprecision(17) << rigidStateUpdateMs << ','
							 << std::setprecision(17) << rigidForceAccumMs;

		benchmarkMetricsCsv_ << '\n';
		benchmarkMetricsCsv_.flush();

		if (!benchmarkMetricsCsv_.good())
		{
			SPDLOG_ERROR("Failed while appending benchmark metrics to '{}'.",
				(runDir_ / "benchmark_metrics.csv").string());
			Disable();
		}
	}

	void LoggingMgr::AppendMemoryRecord(
		const Mem::MemRecordStep& record)
	{
		if (!active_ || !memoryMetricsCsv_.is_open()) { return; }

		memoryMetricsCsv_ << record.simStep << ','
						  << std::setprecision(17) << record.simTime;

		for (uint8_t s = 0; s < Mem::GetIdx(Mem::MemSpace::Count); s++)
		{
			uint64_t totalUsed = 0;
			uint64_t totalAlloc = 0;

			for (uint8_t i = 0; i < Mem::GetIdx(Mem::MemBucket::Count); i++)
			{
				const Mem::MemUsage& usage = record.mem[s][i];

				totalUsed += usage.N_usedBytes;
				totalAlloc += usage.N_allocBytes;

				memoryMetricsCsv_ << ','
								  << usage.N_usedBytes << ','
								  << usage.N_allocBytes;
			}

			memoryMetricsCsv_ << ','
							  << totalUsed << ','
							  << totalAlloc;
		}

		memoryMetricsCsv_ << ','
						  << record.process.N_peakResidentBytes
						  << '\n';
		memoryMetricsCsv_.flush();

		if (!memoryMetricsCsv_.good())
		{
			SPDLOG_ERROR("Failed while appending memory metrics to '{}'.",
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
			SPDLOG_ERROR("Failed to create Logs base directory '{}': {}.",
				loggingOptions_.baseDir.string(), ec.message());
			return false;
		}

		uint64_t nextIdx = 1;
		for (const auto& entry : fs::directory_iterator(loggingOptions_.baseDir, ec))
		{
			if (ec)
			{
				SPDLOG_ERROR("Failed to iterate Logs directory '{}': {}.",
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
				SPDLOG_ERROR("Failed to create run directory '{}': {}.",
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

	bool LoggingMgr::OpenBenchmarkMetricsCsv()
	{
		const std::filesystem::path csvPath = runDir_ / "benchmark_metrics.csv";

		benchmarkMetricsCsv_.open(csvPath, std::ios::out | std::ios::trunc);
		if (!benchmarkMetricsCsv_.is_open())
		{
			return false;
		}

		benchmarkMetricsCsv_.imbue(std::locale::classic());

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

		meta["buildConfig"] =
		{
			{ "gitCommitHash", Sim::BuildConfig::gitCommitHash },
			{ "buildType", Sim::BuildConfig::buildType },
			{ "compilerId", Sim::BuildConfig::cxxCompilerId },
			{ "compilerVersion", Sim::BuildConfig::cxxCompilerVersion },
			{ "cpuSimd",
				{
					{ "arch", Sim::BuildConfig::cpuSimdArch },
					{ "width", Sim::BuildConfig::cpuSimdWidth }
				}
			},
			{ "cpuIsaCompilerFlags", Sim::BuildConfig::cpuIsaCompilerFlags }
		};

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
			{ "backendVersionName", Base::ToCString(simParams.backendVersion) },
			{ "scenarioType", static_cast<int>(simParams.scenarioType) },
			{ "scenarioTypeName", Base::ToCString(simParams.scenarioType) },
			{ "neighborSearchMethod", static_cast<int>(simParams.neighborSearchMethod) },
			{ "neighborSearchMethodName", Base::ToCString(simParams.neighborSearchMethod) },
			{ "neighborListStorage", static_cast<int>(simParams.neighborListMethod) },
			{ "neighborListStorageName", Base::ToCString(simParams.neighborListMethod) },
			{ "neighborListStorageEffectiveName",
				Base::ToEffectiveNeighborListStorageCString(
					simParams.backendVersion,
					simParams.neighborListMethod) },
			{ "domainDecompMethod", static_cast<int>(simParams.domainDecompMethod) },
			{ "domainDecompMethodName", Base::ToCString(simParams.domainDecompMethod) },
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

		for (const auto& spaceDef : Mem::SpaceDefinitions)
		{
			const std::string spaceKey = SanitizeCsvKey(spaceDef.key);

			for (const auto& def : Mem::Definitions)
			{
				const std::string key = spaceKey + "_" + SanitizeCsvKey(def.name);

				memoryMetricsCsv_ << ','
								  << key << "_usedBytes"
								  << ','
								  << key << "_allocBytes";
			}

			memoryMetricsCsv_ << ','
							  << spaceKey << "_total_usedBytes"
							  << ','
							  << spaceKey << "_total_allocBytes";
		}

		memoryMetricsCsv_ << ",process_peakResidentBytes\n";
	}

	void LoggingMgr::WriteBenchmarkMetricsHeader()
	{
		if (!benchmarkMetricsCsv_.is_open()) { return; }

		benchmarkMetricsCsv_
			<< "simStep,simTime,construction_ms,query_ms,total_ms,"
			<< "rigid_state_update_ms,rigid_force_accum_ms\n";
	}

	void LoggingMgr::Disable()
	{
		if (memoryMetricsCsv_.is_open())
		{
			memoryMetricsCsv_.close();
		}
		if (benchmarkMetricsCsv_.is_open())
		{
			benchmarkMetricsCsv_.close();
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
