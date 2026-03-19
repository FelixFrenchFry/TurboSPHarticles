// metrics_logger.h
//
// TODO: comment

#pragma once

// external
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <utility>

// internal
#include "Source/Eval/Bench/operations.h"
#include "Source/Eval/Mem/metrics_mem.h"
#include "Source/Sim/Core/parameters_sim.h"



namespace Export
{
	struct LoggingOptions
	{
	#ifdef TSP_LOGS_DIR
		std::filesystem::path baseDir = TSP_LOGS_DIR;
	#else
		std::filesystem::path baseDir = "Logs";
	#endif
	};

	class LoggingMgr
	{
	public: // -------------------------------------------------------------------------------------

		explicit LoggingMgr(LoggingOptions opt = {})
			: loggingOptions_{ std::move(opt) } {}

		bool BeginRun(const std::string& configProfile,
					  const std::string& paramsFile,
					  const std::string& scenariosFile,
					  const Sim::SimParams& simParams);

		void EndRun();

		void AppendBenchmarkRecord(const Bench::BenchRecordStep& record);

		void AppendMemoryRecord(const Mem::MemRecordStep& record);

		bool IsActive() const { return active_; }

		const std::filesystem::path& GetRunDir() const { return runDir_; }

	private: // ------------------------------------------------------------------------------------

		LoggingOptions loggingOptions_;

		std::filesystem::path runDir_{};

		std::ofstream memoryMetricsCsv_{};
		std::ofstream benchmarkMetricsCsv_{};

		bool active_ = false;

		bool CreateNextRunDirectory();
		bool OpenBenchmarkMetricsCsv();
		bool OpenMemoryMetricsCsv();
		bool WriteRunMeta(const std::string& configProfile,
						  const std::string& paramsFile,
						  const std::string& scenariosFile,
						  const Sim::SimParams& simParams) const;
		void WriteBenchmarkMetricsHeader();
		void WriteMemoryMetricsHeader();
		void Disable();

		static bool TryLoadConfigJson(const std::string& requestedRelPath,
									  nlohmann::json& out,
									  std::string& resolvedRelPath);
		static std::string MakeTimestampString();
		static std::string SanitizeCsvKey(std::string_view src);
	};

} // namespace Export
