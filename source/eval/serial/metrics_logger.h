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
#include "source/eval/bench/operations.h"
#include "source/eval/memory/metrics_mem.h"
#include "source/simulation/core/parameters_sim.h"



namespace Export
{
	struct LoggingOptions
	{
	#ifdef TSP_LOGS_DIR
		std::filesystem::path baseDir = TSP_LOGS_DIR;
	#else
		std::filesystem::path baseDir = "logs";
	#endif

	#ifdef TSP_EXPORT_DIR
		std::filesystem::path exportBaseDir = TSP_EXPORT_DIR;
	#else
		std::filesystem::path exportBaseDir = "exports";
	#endif

		// disabled on non-root ranks, which write no export snapshots
		bool writeRunMetaToExportDir = true;

		// a zero rank count marks a non-distributed run
		int mpiRank = 0;
		int N_mpiRanks = 0;
	};

	class LoggingMgr
	{
	public: // -------------------------------------------------------------------------------------

		explicit LoggingMgr(LoggingOptions opt = {})
			: loggingOptions_{ std::move(opt) } {}

		bool BeginRun(const std::string& configProfile,
					  const std::string& paramsFile,
					  const std::string& scenariosFile,
					  const Sim::SimParams& simParams,
					  const nlohmann::json& cliOverrides,
					  bool logStatus = true);

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
						  const Sim::SimParams& simParams,
						  const nlohmann::json& cliOverrides) const;
		void WriteRunMetaToExportDir(const nlohmann::json& meta,
									 const std::string& exportDir) const;
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
