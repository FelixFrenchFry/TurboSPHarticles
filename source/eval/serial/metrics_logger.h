// metrics_logger.h
#pragma once

// external
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <utility>

// internal
#include "source/eval/bench/operations.h"
#include "source/eval/memory/categories.h"
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

		// distributed runs copy metadata explicitly after reserving their export directory
		bool writeRunMetaToExportDir = true;

		// rank directory below the run directory, empty for non-distributed runs
		std::filesystem::path rankSubDir{};

		// a zero rank count marks a non-distributed run
		int mpiRank = 0;
		int N_mpiRanks = 0;
	};

	class LoggingMgr
	{
	public:

		explicit LoggingMgr(LoggingOptions opt = {})
			: loggingOptions_{ std::move(opt) } {}

		// claims the next free run directory below the logs base directory
		// (returns zero on failure, valid run indices start at one)
		uint64_t ReserveNextRunIdx() const;

		// a zero run index reserves one, distributed runs pass a broadcast index
		bool BeginRun(const std::string& configProfile,
					  const std::string& paramsFile,
					  const std::string& scenariosFile,
					  const Sim::SimParams& simParams,
					  const nlohmann::json& cliOverrides,
					  bool logStatus = true,
					  uint64_t runIdx = 0);

		void EndRun();

		void AppendBenchmarkRecord(const Bench::BenchRecordStep& record);

		void AppendMemoryRecord(const Mem::MemRecordStep& record);

		bool IsActive() const { return active_; }

		const std::filesystem::path& GetRunDir() const { return runDir_; }

		void CopyRunMetaToExportDir(const std::string& exportDir) const;

	private:

		LoggingOptions loggingOptions_;

		std::filesystem::path runDir_{};

		std::ofstream memoryMetricsCsv_{};
		std::ofstream benchmarkMetricsCsv_{};

		bool active_ = false;
		bool runMetaWritten_ = false;

		bool CreateRunDirectory(uint64_t runIdx);
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
