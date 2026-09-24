// dist_diagnostics.cpp

// external
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// internal
#include "dist_diagnostics.h"
#include "dist_helpers.h"
#include "mpi_context.h"
#include "source/base/enum_hub.h"
#include "source/base/helpers.h"
#include "source/simulation/core/diagnostics.h"
#include "source/simulation/core/parameters_sim.h"

using namespace Dist;

namespace
{
	constexpr int DIST_DIAGNOSTICS_RANK_TABLE_LIMIT = 16;
	constexpr int N_CLI_SECTION_SEPARATOR_CHARS = 80;
	constexpr const char* CLI_SECTION_SEPARATOR =
		"--------------------------------------------------------------------------------";

	using steady_clock = std::chrono::steady_clock;
	struct ProgressState
	{
		bool initialized = false;
		bool overwriteProgress = true;
		uint64_t startStep = 0;
		uint64_t lastStep = 0;
		steady_clock::time_point startSteady{};
		steady_clock::time_point pauseStart{};
		bool running = false;
		int N_lastLines = 0;
	};

	struct DistHardwareReport
	{
		int rank = 0;
		char host[MPI_MAX_PROCESSOR_NAME] = {};
		uint32_t N_totalThreads = 0;
		uint32_t N_simThreads = 0;
		uint32_t N_visThreads = 0;
	};

	struct DistRankProgressReport
	{
		DistRankStepReport stepReport;
		DistRankBenchReport benchReport;
	};

	ProgressState& GetProgressState()
	{
		static ProgressState state{};
		return state;
	}

	std::string FormatRamUsage(
		const uint64_t N_residentBytes,
		const int residentRank,
		const uint64_t N_peakResidentBytes,
		const int peakRank)
	{
		char buffer[96];
		std::snprintf(buffer, sizeof(buffer),
			" | RSS/HWM %.2f/%.2f GB (RSS rank %d, HWM rank %d)",
			Mem::ConvertFromBytes(Mem::ByteUnit::GB, N_residentBytes),
			Mem::ConvertFromBytes(Mem::ByteUnit::GB, N_peakResidentBytes),
			residentRank,
			peakRank);
		return buffer;
	}

	void FormatHms(double seconds, int& h, int& m, int& s)
	{
		seconds = std::max(seconds, 0.0);
		const int N_totalSeconds = static_cast<int>(seconds + 0.5);
		h = N_totalSeconds / 3600;
		m = (N_totalSeconds % 3600) / 60;
		s = N_totalSeconds % 60;
	}

	std::string FormatProgressLine(
		const std::vector<DistRankStepReport>& reports,
		const uint64_t startStep,
		const double elapsedS,
		const uint64_t N_residentBytes,
		const int residentRank,
		const uint64_t N_peakResidentBytes,
		const int peakRank)
	{
		if (reports.empty()) { return {}; }

		const uint64_t step = reports.front().step;
		const int64_t maxSteps = reports.front().maxSteps;
		const bool hasTotal = maxSteps > 0;
		const uint64_t maxStepsU = hasTotal ? static_cast<uint64_t>(maxSteps) : 0;

		const double elapsed = std::max(elapsedS, 1e-14);

		const double stepsPerSec = static_cast<double>(step - startStep) / elapsed;
		double percent = 0.0;
		if (hasTotal)
		{
			const double progress = std::min(1.0, std::max(0.0,
				static_cast<double>(step) / static_cast<double>(maxStepsU)));
			percent = progress * 100.0;
		}

		int elH, elM, elS;
		FormatHms(elapsed, elH, elM, elS);
		const std::string ramUsageLabel = FormatRamUsage(
			N_residentBytes, residentRank, N_peakResidentBytes, peakRank);

		char progressLine[256];
		if (hasTotal)
		{
			std::snprintf(progressLine, sizeof(progressLine),
				"%02d:%02d:%02d | %" PRIu64 "/%" PRId64 " steps (%.1f %%) | %.3f steps/s%s",
				elH, elM, elS,
				step, maxSteps, percent, stepsPerSec, ramUsageLabel.c_str());
		}
		else
		{
			std::snprintf(progressLine, sizeof(progressLine),
				"%02d:%02d:%02d | %" PRIu64 " steps | %.3f steps/s%s",
				elH, elM, elS, step, stepsPerSec, ramUsageLabel.c_str());
		}

		return progressLine;
	}

	void PrintCliSeparator()
	{
		std::printf("%s\n", CLI_SECTION_SEPARATOR);
	}

	void PrintCliSectionHeader(const char* title)
	{
		int N_chars = std::printf("--- [ %s ] ", title);
		N_chars = std::max(0, N_chars);

		for (int i = N_chars; i < N_CLI_SECTION_SEPARATOR_CHARS; i++)
		{
			std::putchar('-');
		}

		std::putchar('\n');
	}

	void PrintLine(int& N_lines)
	{
		if (GetProgressState().overwriteProgress)
		{
			std::printf("\x1b[2K");
		}
		std::printf("\n");
		N_lines++;
	}

	template <typename... Args>
	void PrintLine(int& N_lines, const char* fmt, Args... args)
	{
		if (GetProgressState().overwriteProgress)
		{
			std::printf("\x1b[2K");
		}
		std::printf(fmt, args...);
		std::printf("\n");
		N_lines++;
	}

	void FormatRankLabel(char* buffer, const size_t bufferSize, const int rank)
	{
		std::snprintf(buffer, bufferSize, "%d", rank);
	}

	template <size_t N>
	void FillProcessorName(char (&host)[N])
	{
		int hostLen = 0;
		if (MPI_Get_processor_name(host, &hostLen) != MPI_SUCCESS ||
			hostLen <= 0)
		{
			std::snprintf(host, N, "%s", "unknown");
			return;
		}

		const int lastIdx = std::min(hostLen, static_cast<int>(N) - 1);
		host[lastIdx] = '\0';
	}

	std::vector<std::string> WrapCliField(
		const char* text,
		const size_t maxWidth,
		const char* breakDelimiter = nullptr)
	{
		std::vector<std::string> lines{};
		std::string remaining = text;
		const std::string delimiter = breakDelimiter ? breakDelimiter : "";

		while (remaining.size() > maxWidth)
		{
			const size_t separatorPos = delimiter.empty()
				? std::string::npos
				: remaining.rfind(delimiter, maxWidth - 1);
			size_t lineLength = separatorPos == std::string::npos
				? maxWidth
				: separatorPos + delimiter.size();
			if (separatorPos != std::string::npos &&
				!delimiter.empty() && delimiter.back() == ' ')
			{
				lineLength--;
			}

			lines.emplace_back(remaining.substr(0, lineLength));
			remaining.erase(0, lineLength);
			while (!delimiter.empty() && delimiter.back() == ' ' &&
				!remaining.empty() && remaining.front() == ' ')
			{
				remaining.erase(0, 1);
			}
		}

		lines.emplace_back(remaining);
		return lines;
	}

	const char* MemSpaceLabel(const Base::BackendType backendType)
	{
		return backendType == Base::BackendType::GPU ? "VRAM" : "RAM";
	}

	DistHardwareReport BuildLocalHardwareReport(
		const MpiContext& mpiContext,
		const uint32_t N_totalThreads,
		const uint32_t N_simThreads,
		const uint32_t N_visThreads)
	{
		DistHardwareReport report{};
		report.rank = mpiContext.Rank();
		report.N_totalThreads = N_totalThreads;
		report.N_simThreads = N_simThreads;
		report.N_visThreads = N_visThreads;

		FillProcessorName(report.host);

		return report;
	}

	void GatherStartupRankReports(
		const MpiContext& mpiContext,
		const DistStartupRankReport& localReport,
		std::vector<DistStartupRankReport>& reports)
	{
		std::vector<MPI_Request> requests{};

		if (mpiContext.IsRoot())
		{
			reports.assign(static_cast<uint64_t>(mpiContext.Size()), DistStartupRankReport{});
			reports[static_cast<uint64_t>(mpiContext.Rank())] = localReport;
			requests.reserve(static_cast<uint64_t>(std::max(0, mpiContext.Size() - 1)));

			for (int rank = 0; rank < mpiContext.Size(); rank++)
			{
				if (rank == mpiContext.RootRank()) { continue; }

				requests.emplace_back(MPI_REQUEST_NULL);
				mpiContext.IRecvBytes(
					&reports[static_cast<uint64_t>(rank)],
					static_cast<int>(sizeof(DistStartupRankReport)),
					rank,
					DIST_DIAGNOSTICS_TAG_STARTUP_REPORT,
					requests.back());
			}
		}
		else
		{
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.ISendBytes(
				&localReport,
				static_cast<int>(sizeof(DistStartupRankReport)),
				mpiContext.RootRank(),
				DIST_DIAGNOSTICS_TAG_STARTUP_REPORT,
				requests.back());
		}

		mpiContext.WaitAll(requests, "startup diagnostic reports");
	}

	void GatherHardwareReports(
		const MpiContext& mpiContext,
		const DistHardwareReport& localReport,
		std::vector<DistHardwareReport>& reports)
	{
		std::vector<MPI_Request> requests{};

		if (mpiContext.IsRoot())
		{
			reports.assign(static_cast<uint64_t>(mpiContext.Size()), DistHardwareReport{});
			reports[static_cast<uint64_t>(mpiContext.Rank())] = localReport;
			requests.reserve(static_cast<uint64_t>(std::max(0, mpiContext.Size() - 1)));

			for (int rank = 0; rank < mpiContext.Size(); rank++)
			{
				if (rank == mpiContext.RootRank()) { continue; }

				requests.emplace_back(MPI_REQUEST_NULL);
				mpiContext.IRecvBytes(
					&reports[static_cast<uint64_t>(rank)],
					static_cast<int>(sizeof(DistHardwareReport)),
					rank,
					DIST_DIAGNOSTICS_TAG_HARDWARE_REPORT,
					requests.back());
			}
		}
		else
		{
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.ISendBytes(
				&localReport,
				static_cast<int>(sizeof(DistHardwareReport)),
				mpiContext.RootRank(),
				DIST_DIAGNOSTICS_TAG_HARDWARE_REPORT,
				requests.back());
		}

		mpiContext.WaitAll(requests, "hardware diagnostic reports");
	}

	void PrintHardwareReports(const std::vector<DistHardwareReport>& reports)
	{
		std::printf("\n");
		PrintCliSectionHeader("distributed hardware specifications");
		PrintCliSeparator();

		std::printf("%-2s | %-19s | %-2s | %14s | %14s | %14s\n",
			"n", "host name", "r", "avail threads",
			"sim budget", "vis budget");
		std::printf("%s\n", CLI_SECTION_SEPARATOR);

		std::vector<std::string> nodeHosts{};
		nodeHosts.reserve(reports.size());

		for (uint64_t reportIdx = 0; reportIdx < reports.size(); reportIdx++)
		{
			const DistHardwareReport& report = reports[reportIdx];
			const std::string host = report.host;
			const auto nodeIt = std::find(nodeHosts.begin(), nodeHosts.end(), host);
			int node = 0;
			if (nodeIt == nodeHosts.end())
			{
				node = static_cast<int>(nodeHosts.size());
				nodeHosts.emplace_back(host);
			}
			else
			{
				node = static_cast<int>(nodeIt - nodeHosts.begin());
			}
			const bool firstRankOnNode = reportIdx == 0 ||
				std::strcmp(reports[reportIdx - 1].host, report.host) != 0;

			char nodeLabel[32];
			char rankLabel[32];
			FormatRankLabel(nodeLabel, sizeof(nodeLabel), node);
			FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

			const std::vector<std::string> hostLines = firstRankOnNode
				? WrapCliField(report.host, 19)
				: std::vector<std::string>{};

			std::printf("%-2s | %-19.19s | %-2s | %14u | %14u | %14u\n",
				firstRankOnNode ? nodeLabel : "",
				hostLines.empty() ? "" : hostLines.front().c_str(),
				rankLabel,
				report.N_totalThreads,
				report.N_simThreads,
				report.N_visThreads);

			for (uint64_t hostLineIdx = 1;
				hostLineIdx < hostLines.size(); hostLineIdx++)
			{
				std::printf("%-2s | %-19s | %-2s | %14s | %14s | %14s\n",
					"", hostLines[hostLineIdx].c_str(), "",
					"", "", "");
			}
		}

		std::printf("\n");
	}

	void GatherRankStepReports(
		const MpiContext& mpiContext,
		const DistRankStepReport& localReport,
		std::vector<DistRankStepReport>& reports)
	{
		std::vector<MPI_Request> requests{};

		if (mpiContext.IsRoot())
		{
			reports.assign(static_cast<uint64_t>(mpiContext.Size()), DistRankStepReport{});
			reports[static_cast<uint64_t>(mpiContext.Rank())] = localReport;
			requests.reserve(static_cast<uint64_t>(std::max(0, mpiContext.Size() - 1)));

			for (int rank = 0; rank < mpiContext.Size(); rank++)
			{
				if (rank == mpiContext.RootRank()) { continue; }

				requests.emplace_back(MPI_REQUEST_NULL);
				mpiContext.IRecvBytes(
					&reports[static_cast<uint64_t>(rank)],
					static_cast<int>(sizeof(DistRankStepReport)),
					rank,
					DIST_DIAGNOSTICS_TAG_STEP_REPORT,
					requests.back());
			}
		}
		else
		{
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.ISendBytes(
				&localReport,
				static_cast<int>(sizeof(DistRankStepReport)),
				mpiContext.RootRank(),
				DIST_DIAGNOSTICS_TAG_STEP_REPORT,
				requests.back());
		}

		mpiContext.WaitAll(requests, "step diagnostic reports");
	}

	void GatherRankProgressReports(
		const MpiContext& mpiContext,
		const DistRankProgressReport& localReport,
		std::vector<DistRankProgressReport>& reports)
	{
		std::vector<MPI_Request> requests{};

		if (mpiContext.IsRoot())
		{
			reports.assign(static_cast<uint64_t>(mpiContext.Size()), DistRankProgressReport{});
			reports[static_cast<uint64_t>(mpiContext.Rank())] = localReport;
			requests.reserve(static_cast<uint64_t>(std::max(0, mpiContext.Size() - 1)));

			for (int rank = 0; rank < mpiContext.Size(); rank++)
			{
				if (rank == mpiContext.RootRank()) { continue; }

				requests.emplace_back(MPI_REQUEST_NULL);
				mpiContext.IRecvBytes(
					&reports[static_cast<uint64_t>(rank)],
					static_cast<int>(sizeof(DistRankProgressReport)),
					rank,
					DIST_DIAGNOSTICS_TAG_PROGRESS_REPORT,
					requests.back());
			}
		}
		else
		{
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.ISendBytes(
				&localReport,
				static_cast<int>(sizeof(DistRankProgressReport)),
				mpiContext.RootRank(),
				DIST_DIAGNOSTICS_TAG_PROGRESS_REPORT,
				requests.back());
		}

		mpiContext.WaitAll(requests, "progress diagnostic reports");
	}

	uint64_t SumReportField(
		const std::vector<DistRankStepReport>& reports,
		uint32_t DistRankStepReport::* field)
	{
		uint64_t sum = 0;
		for (const DistRankStepReport& report : reports)
		{
			sum += report.*field;
		}
		return sum;
	}

	uint64_t SumReportField(
		const std::vector<DistRankStepReport>& reports,
		uint64_t DistRankStepReport::* field)
	{
		uint64_t sum = 0;
		for (const DistRankStepReport& report : reports)
		{
			sum += report.*field;
		}
		return sum;
	}

	int RankWithMax(
		const std::vector<DistRankStepReport>& reports,
		uint64_t DistRankStepReport::* field)
	{
		if (reports.empty()) { return -1; }

		const auto it = std::max_element(
			reports.begin(), reports.end(),
			[field](const DistRankStepReport& a, const DistRankStepReport& b)
			{
				return a.*field < b.*field;
			});
		return it != reports.end() ? it->rank : -1;
	}

	uint64_t MaxReportField(
		const std::vector<DistRankStepReport>& reports,
		uint64_t DistRankStepReport::* field)
	{
		uint64_t maxValue = 0;
		for (const DistRankStepReport& report : reports)
		{
			maxValue = std::max(maxValue, report.*field);
		}
		return maxValue;
	}

	uint64_t MinReportField(
		const std::vector<DistRankStepReport>& reports,
		uint64_t DistRankStepReport::* field)
	{
		if (reports.empty()) { return 0; }

		uint64_t minValue = reports.front().*field;
		for (const DistRankStepReport& report : reports)
		{
			minValue = std::min(minValue, report.*field);
		}
		return minValue;
	}

	double AvgReportField(
		const std::vector<DistRankStepReport>& reports,
		uint64_t DistRankStepReport::* field)
	{
		if (reports.empty()) { return 0.0; }

		const uint64_t sum = SumReportField(reports, field);
		return static_cast<double>(sum) / static_cast<double>(reports.size());
	}

	uint64_t SaturatingSubtract(const uint64_t a, const uint64_t b)
	{
		return a > b ? a - b : 0;
	}

	bool IsTotalOp(const Bench::BenchOp op)
	{
		return op == Bench::BenchOp::Total;
	}

	bool IsDistBenchOp(const Bench::BenchOp op)
	{
		return op == Bench::BenchOp::Dist_ParticleRedistribution ||
			op == Bench::BenchOp::Dist_PlanRebuild ||
			op == Bench::BenchOp::Dist_HaloFieldExchange ||
			op == Bench::BenchOp::Dist_LoadBalancing ||
			op == Bench::BenchOp::Dist_Misc;
	}

	int FindTotalOpIdx(const Bench::BenchOpList& benchOps)
	{
		for (uint32_t i = 0; i < benchOps.size(); i++)
		{
			if (IsTotalOp(benchOps[i]))
			{
				return static_cast<int>(i);
			}
		}

		return -1;
	}

	uint64_t BenchOpTimeNs(
		const DistRankBenchReport& report,
		const Bench::BenchOp op,
		const bool useAvg)
	{
		const uint8_t i = Bench::GetIdx(op);
		return useAvg ? report.avgNs[i] : report.latestNs[i];
	}

	std::string BenchOpName(const Bench::BenchOp op)
	{
		if (IsTotalOp(op)) { return "total"; }

		const auto& def = Bench::Definitions[Bench::GetIdx(op)];
		return std::string(def.name);
	}

	struct DistBenchOpSummary
	{
		uint64_t latestNs = 0;
		uint64_t avgNs = 0;
	};

	DistBenchOpSummary SummarizeBenchOp(
		const std::vector<DistRankBenchReport>& reports,
		const Bench::BenchOp op)
	{
		DistBenchOpSummary summary{};

		for (const DistRankBenchReport& report : reports)
		{
			const uint64_t latestNs = BenchOpTimeNs(report, op, false);
			const uint64_t avgNs = BenchOpTimeNs(report, op, true);

			if (latestNs > summary.latestNs)
			{
				summary.latestNs = latestNs;
			}
			summary.avgNs = std::max(summary.avgNs, avgNs);
		}

		return summary;
	}

	const DistRankBenchReport* FindReportWithMaxBenchOpTime(
		const std::vector<DistRankBenchReport>& reports,
		const Bench::BenchOp op,
		const bool useAvg)
	{
		const DistRankBenchReport* bestReport = nullptr;
		uint64_t bestNs = 0;

		for (const DistRankBenchReport& report : reports)
		{
			const uint64_t timeNs = BenchOpTimeNs(report, op, useAvg);
			if (bestReport == nullptr || timeNs > bestNs)
			{
				bestReport = &report;
				bestNs = timeNs;
			}
		}

		return bestReport;
	}

	struct DistBenchTableSummary
	{
		std::vector<DistBenchOpSummary> ops{};
		DistBenchOpSummary other{};
		int totalIdx = -1;
		int latestCriticalRank = -1;
		int avgCriticalRank = -1;
	};

	DistBenchTableSummary BuildCriticalRankBenchSummaries(
		const std::vector<DistRankBenchReport>& reports,
		const Bench::BenchOpList& benchOps)
	{
		DistBenchTableSummary tableSummary{};
		tableSummary.ops.reserve(benchOps.size());

		tableSummary.totalIdx = FindTotalOpIdx(benchOps);

		if (tableSummary.totalIdx < 0)
		{
			for (const Bench::BenchOp op : benchOps)
			{
				tableSummary.ops.push_back(SummarizeBenchOp(reports, op));
			}
			return tableSummary;
		}

		const Bench::BenchOp totalOp =
			benchOps[static_cast<std::size_t>(tableSummary.totalIdx)];
		const DistRankBenchReport* latestTotalReport =
			FindReportWithMaxBenchOpTime(reports, totalOp, false);
		const DistRankBenchReport* avgTotalReport =
			FindReportWithMaxBenchOpTime(reports, totalOp, true);
		tableSummary.latestCriticalRank =
			latestTotalReport != nullptr ? latestTotalReport->rank : -1;
		tableSummary.avgCriticalRank =
			avgTotalReport != nullptr ? avgTotalReport->rank : -1;

		uint64_t accountedLatestNs = 0;
		uint64_t accountedAvgNs = 0;

		for (uint32_t i = 0; i < benchOps.size(); i++)
		{
			const Bench::BenchOp op = benchOps[i];
			DistBenchOpSummary opSummary{};

			if (latestTotalReport != nullptr)
			{
				opSummary.latestNs =
					BenchOpTimeNs(*latestTotalReport, op, false);
			}
			if (avgTotalReport != nullptr)
			{
				opSummary.avgNs =
					BenchOpTimeNs(*avgTotalReport, op, true);
			}

			if (static_cast<int>(i) != tableSummary.totalIdx)
			{
				accountedLatestNs += opSummary.latestNs;
				accountedAvgNs += opSummary.avgNs;
			}

			tableSummary.ops.push_back(opSummary);
		}

		const DistBenchOpSummary& totalSummary =
			tableSummary.ops[static_cast<std::size_t>(tableSummary.totalIdx)];
		tableSummary.other.latestNs =
			SaturatingSubtract(totalSummary.latestNs, accountedLatestNs);
		tableSummary.other.avgNs =
			SaturatingSubtract(totalSummary.avgNs, accountedAvgNs);

		return tableSummary;
	}

	struct DistRankTimingSummary
	{
		int rank = 0;

		uint64_t redistributionAvgNs = 0;
		uint64_t planRebuildAvgNs = 0;
		uint64_t haloFieldExchangeAvgNs = 0;
		uint64_t loadBalancingAvgNs = 0;
		uint64_t miscAvgNs = 0;
	};

	std::vector<DistRankTimingSummary> BuildRankTimingSummaries(
		const std::vector<DistRankBenchReport>& reports,
		const Bench::BenchOpList& benchOps)
	{
		std::vector<DistRankTimingSummary> summaries{};
		if (reports.empty() || benchOps.empty()) { return summaries; }

		if (FindTotalOpIdx(benchOps) < 0) { return summaries; }

		summaries.reserve(reports.size());

		for (const DistRankBenchReport& report : reports)
		{
			DistRankTimingSummary summary{};
			summary.rank = report.rank;

			summary.redistributionAvgNs =
				BenchOpTimeNs(report, Bench::BenchOp::Dist_ParticleRedistribution, true);
			summary.planRebuildAvgNs =
				BenchOpTimeNs(report, Bench::BenchOp::Dist_PlanRebuild, true);
			summary.haloFieldExchangeAvgNs =
				BenchOpTimeNs(report, Bench::BenchOp::Dist_HaloFieldExchange, true);
			summary.loadBalancingAvgNs =
				BenchOpTimeNs(report, Bench::BenchOp::Dist_LoadBalancing, true);
			summary.miscAvgNs =
				BenchOpTimeNs(report, Bench::BenchOp::Dist_Misc, true);

			summaries.push_back(summary);
		}

		return summaries;
	}

	void FormatPercentLabel(char* buffer, const size_t bufferSize, const double pct)
	{
		std::snprintf(buffer, bufferSize, "%.1f %%", std::max(0.0, pct));
	}

	void PrintParticleLayoutTable(
		int& N_lines,
		const std::vector<DistRankStepReport>& reports)
	{
		PrintLine(N_lines,
			"%-2s | %10s | %10s | %10s | %10s | %10s | %10s",
			"r", "inner F", "outer F", "halo F",
			"inner B", "outer B", "halo B");
		PrintLine(N_lines, "%s", CLI_SECTION_SEPARATOR);

		for (const DistRankStepReport& report : reports)
		{
			char rankLabel[32];
			FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

			PrintLine(N_lines,
				"%-2s | %10u | %10u | %10u | %10u | %10u | %10u",
				rankLabel,
				report.N_innerF,
				report.N_outerF,
				report.N_haloF,
				report.N_innerB,
				report.N_outerB,
				report.N_haloB);
		}

		PrintLine(N_lines);
		PrintLine(N_lines,
			"%-2s | %10s | %10s | %10s",
			"r", "inner R", "outer R", "halo R");
		PrintLine(N_lines, "%s", CLI_SECTION_SEPARATOR);

		for (const DistRankStepReport& report : reports)
		{
			char rankLabel[32];
			FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

			PrintLine(N_lines,
				"%-2s | %10u | %10u | %10u",
				rankLabel,
				report.N_innerR,
				report.N_outerR,
				report.N_haloR);
		}
	}

	void PrintParticleExchangeTable(
		int& N_lines,
		const std::vector<DistRankStepReport>& reports)
	{
		PrintLine(N_lines,
			"%-2s | %17s | %16s | %17s | %16s",
			"r", "migr F send/recv", "migr B send/recv",
			"halo F send/recv", "halo B send/recv");
		PrintLine(N_lines, "%s", CLI_SECTION_SEPARATOR);

		for (const DistRankStepReport& report : reports)
		{
			char rankLabel[32];
			FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

			PrintLine(N_lines,
				"%-2s | %7" PRIu32 " / %-7" PRIu32 " | %6" PRIu32 " / %-7" PRIu32
				" | %7" PRIu64 " / %-7" PRIu64 " | %6" PRIu64 " / %-7" PRIu64,
				rankLabel,
				report.N_sentMigratedF,
				report.N_recvMigratedF,
				report.N_sentMigratedB,
				report.N_recvMigratedB,
				report.N_sentHaloF,
				report.N_recvHaloF,
				report.N_sentHaloB,
				report.N_recvHaloB);
		}

		PrintLine(N_lines);
		PrintLine(N_lines,
			"%-2s | %17s | %17s",
			"r", "migr R send/recv", "halo R send/recv");
		PrintLine(N_lines, "%s", CLI_SECTION_SEPARATOR);

		for (const DistRankStepReport& report : reports)
		{
			char rankLabel[32];
			FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

			PrintLine(N_lines,
				"%-2s | %7" PRIu32 " / %-7" PRIu32 " | %7" PRIu64 " / %-7" PRIu64,
				rankLabel,
				report.N_sentMigratedR,
				report.N_recvMigratedR,
				report.N_sentHaloR,
				report.N_recvHaloR);
		}
	}

	void PrintBenchmarkSummaryTable(
		int& N_lines,
		const std::vector<DistRankBenchReport>& reports,
		const Bench::BenchOpList& benchOps,
		const Bench::TimeUnit unit)
	{
		if (reports.empty() || benchOps.empty()) { return; }

		const DistBenchTableSummary tableSummary =
			BuildCriticalRankBenchSummaries(reports, benchOps);

		const uint64_t totalLatestNs = tableSummary.totalIdx >= 0
			? tableSummary.ops[static_cast<std::size_t>(
				tableSummary.totalIdx)].latestNs
			: 0;
		const uint64_t totalAvgNs = tableSummary.totalIdx >= 0
			? tableSummary.ops[static_cast<std::size_t>(
				tableSummary.totalIdx)].avgNs
			: 0;

		const char* unitStr = Bench::UnitLabel(unit);

		char latestLabel[32];
		char avgLabel[32];
		std::snprintf(latestLabel, sizeof(latestLabel), "latest [%s]", unitStr);
		std::snprintf(avgLabel, sizeof(avgLabel), "avg [%s]", unitStr);

		PrintLine(N_lines,
			"%-33.33s | %11s  %8s | %10s  %8s",
			"benchmark summary", latestLabel, "% total", avgLabel, "% total");
		PrintLine(N_lines, "%s", CLI_SECTION_SEPARATOR);

		auto PrintBenchRow =
			[&](const char* name,
				const DistBenchOpSummary& summary,
				const bool isTotal)
		{
			const double latestTime =
				Bench::ConvertFromNs(unit, summary.latestNs);
			const double avgTime =
				Bench::ConvertFromNs(unit, summary.avgNs);

			char pctLatestLabel[16];
			char pctAvgLabel[16];
			if (isTotal)
			{
				std::snprintf(pctLatestLabel, sizeof(pctLatestLabel), "%s", "---");
				std::snprintf(pctAvgLabel, sizeof(pctAvgLabel), "%s", "---");
			}
			else
			{
				const double pctLatest = totalLatestNs > 0
					? 100.0 * static_cast<double>(summary.latestNs) /
						static_cast<double>(totalLatestNs)
					: 0.0;
				const double pctAvg = totalAvgNs > 0
					? 100.0 * static_cast<double>(summary.avgNs) /
						static_cast<double>(totalAvgNs)
					: 0.0;

				FormatPercentLabel(pctLatestLabel, sizeof(pctLatestLabel), pctLatest);
				FormatPercentLabel(pctAvgLabel, sizeof(pctAvgLabel), pctAvg);
			}

			PrintLine(N_lines,
				"%-33.33s | %11.2f  %8s | %10.2f  %8s",
				name, latestTime, pctLatestLabel,
				avgTime, pctAvgLabel);
		};

		bool distSeparatorPrinted = false;
		for (uint32_t i = 0; i < benchOps.size(); i++)
		{
			if (static_cast<int>(i) == tableSummary.totalIdx) { continue; }
			if (!distSeparatorPrinted && IsDistBenchOp(benchOps[i]))
			{
				// the residual of all operations not explicitly measured
				if (tableSummary.totalIdx >= 0)
				{
					PrintBenchRow("other", tableSummary.other, false);
				}
				PrintLine(N_lines, "%s", CLI_SECTION_SEPARATOR);
				distSeparatorPrinted = true;
			}

			const std::string name = BenchOpName(benchOps[i]);
			PrintBenchRow(name.c_str(), tableSummary.ops[i], false);
		}

		if (tableSummary.totalIdx >= 0)
		{
			PrintLine(N_lines, "%s", CLI_SECTION_SEPARATOR);
			PrintBenchRow(
				"total",
				tableSummary.ops[static_cast<std::size_t>(
					tableSummary.totalIdx)],
				true);
			PrintLine(N_lines,
				"%-33.33s | %11d  %8s | %10d  %8s",
				"critical rank for latest/avg",
				tableSummary.latestCriticalRank, "---",
				tableSummary.avgCriticalRank, "---");
		}
	}

	void PrintRankTimingTable(
		int& N_lines,
		const std::vector<DistRankBenchReport>& reports,
		const Bench::BenchOpList& benchOps,
		const Bench::TimeUnit unit)
	{
		const std::vector<DistRankTimingSummary> summaries =
			BuildRankTimingSummaries(reports, benchOps);
		if (summaries.empty()) { return; }

		char unitLabel[32];
		std::snprintf(unitLabel, sizeof(unitLabel), "avg [%s]", Bench::UnitLabel(unit));

		PrintLine(N_lines,
			"%-2s | %8s | %8s | %8s | %8s | %8s | %9s",
			"r", "redist", "plan", "halo", "load", "misc", "");
		PrintLine(N_lines, "%s", CLI_SECTION_SEPARATOR);

		for (const DistRankTimingSummary& summary : summaries)
		{
			char rankLabel[32];
			FormatRankLabel(rankLabel, sizeof(rankLabel), summary.rank);

			PrintLine(N_lines,
				"%-2s | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %9s",
				rankLabel,
				Bench::ConvertFromNs(
					unit,
					summary.redistributionAvgNs),
				Bench::ConvertFromNs(unit, summary.planRebuildAvgNs),
				Bench::ConvertFromNs(unit, summary.haloFieldExchangeAvgNs),
				Bench::ConvertFromNs(unit, summary.loadBalancingAvgNs),
				Bench::ConvertFromNs(unit, summary.miscAvgNs),
				unitLabel);
		}
	}

	std::string BuildMemoryTable(
		const std::vector<DistRankStepReport>& reports,
		const Mem::ByteUnit unit,
		const Base::BackendType backendType)
	{
		std::string table{};
		char memoryUnitLabel[32];
		std::snprintf(memoryUnitLabel, sizeof(memoryUnitLabel), "%s [%s]",
			MemSpaceLabel(backendType), Mem::UnitLabel(unit));

		char line[256];
		std::snprintf(line, sizeof(line),
			"%-2s | %8s | %8s | %8s | %8s | %8s | %8s | %9s",
			"r", "backend", "metadata", "exchange", "misc", "total", "HWM", "");
		table += line;
		table += '\n';
		table += CLI_SECTION_SEPARATOR;
		table += '\n';

		for (const DistRankStepReport& report : reports)
		{
			char rankLabel[32];
			FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

			const uint64_t N_maxDistMiscAndStateMemAllocBytes =
				report.N_maxDistLocalStateMemAllocBytes +
				report.N_maxDistMiscMemAllocBytes;

			std::snprintf(line, sizeof(line),
				"%-2s | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %9s",
				rankLabel,
				Mem::ConvertFromBytes(unit, report.N_maxBackendMemAllocBytes),
				Mem::ConvertFromBytes(unit, report.N_maxDistParticleMetadataMemAllocBytes),
				Mem::ConvertFromBytes(unit, report.N_maxDistExchangeBuffersMemAllocBytes),
				Mem::ConvertFromBytes(unit, N_maxDistMiscAndStateMemAllocBytes),
				Mem::ConvertFromBytes(unit, report.N_maxTotalTrackedMemAllocBytes),
				Mem::ConvertFromBytes(unit, report.N_processPeakResidentBytes),
				memoryUnitLabel);
			table += line;
			table += '\n';
		}

		return table;
	}

	void PrintMemoryTable(
		const std::vector<DistRankStepReport>& reports,
		const Mem::ByteUnit unit,
		const Base::BackendType backendType)
	{
		const std::string table = BuildMemoryTable(reports, unit, backendType);
		std::printf("%s", table.c_str());
	}

	void PrintAggregateTable(
		int& N_lines,
		const std::vector<DistRankStepReport>& reports,
		const Sim::SimParams& params)
	{
		const uint64_t totalOwnedFluid =
			SumReportField(reports, &DistRankStepReport::N_ownedF);
		const uint64_t totalHaloFluid =
			SumReportField(reports, &DistRankStepReport::N_haloF);
		const uint64_t totalOwnedBoundary =
			SumReportField(reports, &DistRankStepReport::N_ownedB);
		const uint64_t totalHaloBoundary =
			SumReportField(reports, &DistRankStepReport::N_haloB);
		const uint64_t totalOwnedRigid =
			SumReportField(reports, &DistRankStepReport::N_ownedR);
		const uint64_t totalHaloRigid =
			SumReportField(reports, &DistRankStepReport::N_haloR);
		const uint64_t totalMigrationSent =
			SumReportField(reports, &DistRankStepReport::N_sentMigratedF);
		const uint64_t totalMigrationReceived =
			SumReportField(reports, &DistRankStepReport::N_recvMigratedF);
		const uint64_t totalHaloFluidSent =
			SumReportField(reports, &DistRankStepReport::N_sentHaloF);
		const uint64_t totalHaloFluidReceived =
			SumReportField(reports, &DistRankStepReport::N_recvHaloF);
		const uint64_t totalBoundaryMigrationSent =
			SumReportField(reports, &DistRankStepReport::N_sentMigratedB);
		const uint64_t totalBoundaryMigrationReceived =
			SumReportField(reports, &DistRankStepReport::N_recvMigratedB);
		const uint64_t totalHaloBoundarySent =
			SumReportField(reports, &DistRankStepReport::N_sentHaloB);
		const uint64_t totalHaloBoundaryReceived =
			SumReportField(reports, &DistRankStepReport::N_recvHaloB);
		const uint64_t totalRigidMigrationSent =
			SumReportField(reports, &DistRankStepReport::N_sentMigratedR);
		const uint64_t totalRigidMigrationReceived =
			SumReportField(reports, &DistRankStepReport::N_recvMigratedR);
		const uint64_t totalHaloRigidSent =
			SumReportField(reports, &DistRankStepReport::N_sentHaloR);
		const uint64_t totalHaloRigidReceived =
			SumReportField(reports, &DistRankStepReport::N_recvHaloR);

		const double latestMin =
			Bench::ConvertFromNs(params.timeUnit,
				MinReportField(reports, &DistRankStepReport::totalStepLatestNs));
		const double latestAvg =
			Bench::ConvertFromNs(params.timeUnit,
				static_cast<uint64_t>(AvgReportField(reports, &DistRankStepReport::totalStepLatestNs)));
		const double latestMax =
			Bench::ConvertFromNs(params.timeUnit,
				MaxReportField(reports, &DistRankStepReport::totalStepLatestNs));
		const int slowestRank = RankWithMax(reports, &DistRankStepReport::totalStepLatestNs);

		const char* timeUnitStr = Bench::UnitLabel(params.timeUnit);
		PrintLine(N_lines,
			"particles | owned F %" PRIu64 " | halo F %" PRIu64
			" | owned B %" PRIu64 " | halo B %" PRIu64
			" | owned R %" PRIu64 " | halo R %" PRIu64,
			totalOwnedFluid, totalHaloFluid,
			totalOwnedBoundary, totalHaloBoundary,
			totalOwnedRigid, totalHaloRigid);
		PrintLine(N_lines,
			"traffic   | migrated F %" PRIu64 " / %" PRIu64
			" | halo F sent/recv %" PRIu64 " / %" PRIu64
			" | migrated B %" PRIu64 " / %" PRIu64
			" | halo B sent/recv %" PRIu64 " / %" PRIu64
			" | migrated R %" PRIu64 " / %" PRIu64
			" | halo R sent/recv %" PRIu64 " / %" PRIu64,
			totalMigrationSent, totalMigrationReceived,
			totalHaloFluidSent, totalHaloFluidReceived,
			totalBoundaryMigrationSent, totalBoundaryMigrationReceived,
			totalHaloBoundarySent, totalHaloBoundaryReceived,
			totalRigidMigrationSent, totalRigidMigrationReceived,
			totalHaloRigidSent, totalHaloRigidReceived);
		PrintLine(N_lines,
			"timing    | min/avg/max %.3f / %.3f / %.3f %s | slowest rank %d",
			latestMin, latestAvg, latestMax, timeUnitStr, slowestRank);
	}

	void PrintDashboard(
		const std::vector<DistRankStepReport>& reports,
		const std::vector<DistRankBenchReport>& benchReports,
		const Bench::BenchOpList& benchOps,
		const Sim::SimParams& params)
	{
		if (reports.empty()) { return; }

		ProgressState& state = GetProgressState();
		state.overwriteProgress = Sim::DiagnosticsMgr::ShouldOverwriteProgress(
			params.progressReportMode);
		const auto now = steady_clock::now();
		const uint64_t step = reports.front().step;
		if (!state.initialized || step < state.lastStep)
		{
			state.initialized = true;
			state.startSteady = now;
			state.pauseStart = now;
			state.N_lastLines = 0;
			state.startStep = step;
			state.lastStep = step;
		}

		if (step <= state.lastStep) { return; }
		state.lastStep = step;

		if (state.overwriteProgress && state.N_lastLines > 0)
		{
			std::printf("\x1b[%dA\r", state.N_lastLines);
		}

		// cumulative running time, excluding startup and pauses
		const auto end = state.running ? now : state.pauseStart;
		double elapsedS = std::chrono::duration<double>(
			end - state.startSteady).count();
		elapsedS = std::max(elapsedS, 1e-14);

		int N_lines = 0;
		if (static_cast<int>(reports.size()) <= DIST_DIAGNOSTICS_RANK_TABLE_LIMIT)
		{
			PrintParticleLayoutTable(N_lines, reports);
			PrintLine(N_lines);
			PrintParticleExchangeTable(N_lines, reports);
		}
		else
		{
			PrintAggregateTable(N_lines, reports, params);
		}

		PrintLine(N_lines);
		PrintBenchmarkSummaryTable(N_lines, benchReports, benchOps, params.timeUnit);
		if (static_cast<int>(benchReports.size()) <= DIST_DIAGNOSTICS_RANK_TABLE_LIMIT)
		{
			PrintLine(N_lines);
			PrintRankTimingTable(N_lines, benchReports, benchOps, params.timeUnit);
		}
		PrintLine(N_lines);
		const uint64_t processResidentMax = MaxReportField(
			reports, &DistRankStepReport::N_processResidentBytes);
		const int processResidentMaxRank = RankWithMax(
			reports, &DistRankStepReport::N_processResidentBytes);
		const uint64_t processPeakResidentMax = MaxReportField(
			reports, &DistRankStepReport::N_processPeakResidentBytes);
		const int processPeakResidentMaxRank = RankWithMax(
			reports, &DistRankStepReport::N_processPeakResidentBytes);
		const std::string progressLine = FormatProgressLine(
			reports, state.startStep, elapsedS,
			processResidentMax, processResidentMaxRank,
			processPeakResidentMax, processPeakResidentMaxRank);
		PrintLine(N_lines, "%s", progressLine.c_str());
		std::fflush(stdout);
		state.N_lastLines = N_lines;
	}

} // namespace anonymous

namespace Dist
{
	void DistDiagnosticsMgr::PrintBuildConfig(const MpiContext& mpiContext)
	{
		if (!mpiContext.IsRoot()) { return; }

		Sim::DiagnosticsMgr::PrintBuildConfig();
		std::fflush(stdout);
	}

	void DistDiagnosticsMgr::PrintHardwareSpecs(
		const MpiContext& mpiContext,
		const uint32_t N_totalThreads,
		const uint32_t N_simThreads,
		const uint32_t N_visThreads)
	{
		const DistHardwareReport localReport =
			BuildLocalHardwareReport(
				mpiContext,
				N_totalThreads,
				N_simThreads,
				N_visThreads);

		std::vector<DistHardwareReport> reports{};
		GatherHardwareReports(mpiContext, localReport, reports);

		if (!mpiContext.IsRoot()) { return; }

		PrintHardwareReports(reports);
		std::fflush(stdout);
	}

	void DistDiagnosticsMgr::PrintSimParams(
		const MpiContext& mpiContext,
		const Sim::SimParams& params)
	{
		if (!mpiContext.IsRoot()) { return; }

		Sim::DiagnosticsMgr::PrintSimParams(params);
		std::fflush(stdout);
	}

	void DistDiagnosticsMgr::PrintDistSimSetup(
		const MpiContext& mpiContext,
		const DistSimSetupReport& setup,
		const DistStartupRankReport& localReport)
	{
		DistStartupRankReport localReportWithHost = localReport;
		FillProcessorName(localReportWithHost.host);

		std::vector<DistStartupRankReport> reports{};
		GatherStartupRankReports(mpiContext, localReportWithHost, reports);

		if (!mpiContext.IsRoot()) { return; }

		std::printf("\n");
		PrintCliSectionHeader("distributed simulation setup");
		PrintCliSeparator();

		std::printf("%-30s = %s\n", "sim backend", setup.backendType);
		std::printf("%-30s = %s\n", "sim scenario", setup.scenarioType);
		std::printf("%-30s = %s\n", "neighbor search method",
			setup.neighborSearchMethod);
		std::printf("%-30s = %s\n", "neighbor list storage",
			setup.neighborListStorage);

		std::printf("%-30s = %s\n", "domain decomp method", setup.domainDecompMethod);
		std::printf("%-30s = %s\n", "distributed halo mode", setup.distHaloMode);
		std::printf("%-30s = %s\n", "load balance metric", setup.loadBalanceMetric);
		std::printf("%-30s = %s\n", "load balance trigger", setup.loadBalanceTrigger);
		std::printf("%-30s = %s\n", "load balance weight model",
			setup.loadBalanceWeightModel);
		std::printf("%-30s = %s\n", "mpi thread support",
			MpiContext::ThreadSupportName(setup.mpiThreadSupport));
		std::printf("%-30s = [ %" PRIu64 " / %" PRIu64 " / %" PRIu64 " ]\n",
			"global particles [F / B / R]",
			setup.N_globalF,
			setup.N_globalB,
			setup.N_globalR);
		std::printf("%-30s = %u\n", "global rigid bodies",
			setup.N_globalRB);
		std::printf("\n");

		std::printf("%-2s | %-19s | %-2s | %-9s | %10s | %10s | %10s\n",
			"n", "host name", "r", "nbrs",
			"owned F", "owned B", "owned R");
		std::printf("%s\n", CLI_SECTION_SEPARATOR);

		std::vector<std::string> nodeHosts{};
		nodeHosts.reserve(reports.size());

		for (uint64_t reportIdx = 0; reportIdx < reports.size(); reportIdx++)
		{
			const DistStartupRankReport& report = reports[reportIdx];
			const std::string host = report.host;
			const auto nodeIt = std::find(nodeHosts.begin(), nodeHosts.end(), host);
			int node = 0;
			if (nodeIt == nodeHosts.end())
			{
				node = static_cast<int>(nodeHosts.size());
				nodeHosts.emplace_back(host);
			}
			else
			{
				node = static_cast<int>(nodeIt - nodeHosts.begin());
			}
			const bool firstRankOnNode = reportIdx == 0 ||
				std::strcmp(reports[reportIdx - 1].host, report.host) != 0;

			char nodeLabel[32];
			char rankLabel[32];
			const std::vector<std::string> hostLines = firstRankOnNode
				? WrapCliField(report.host, 19)
				: std::vector<std::string>{};
			const std::vector<std::string> neighborLines =
				WrapCliField(report.neighborRanks, 9, ", ");
			const uint64_t N_wrappedLines = std::max(
				hostLines.size(), neighborLines.size());
			FormatRankLabel(nodeLabel, sizeof(nodeLabel), node);
			FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

			std::printf("%-2s | %-19.19s | %-2s | %-9s | %10u | %10u | %10u\n",
				firstRankOnNode ? nodeLabel : "",
				hostLines.empty() ? "" : hostLines.front().c_str(),
				rankLabel,
				neighborLines.front().c_str(),
				report.N_ownedF,
				report.N_ownedB,
				report.N_ownedR);

			for (uint64_t wrappedLineIdx = 1;
				wrappedLineIdx < N_wrappedLines; wrappedLineIdx++)
			{
				const char* hostLine = wrappedLineIdx < hostLines.size()
					? hostLines[wrappedLineIdx].c_str()
					: "";
				const char* neighborLine = wrappedLineIdx < neighborLines.size()
					? neighborLines[wrappedLineIdx].c_str()
					: "";

				std::printf("%-2s | %-19s | %-2s | %-9s | %10s | %10s | %10s\n",
					"", hostLine, "", neighborLine,
					"", "", "");
			}
		}

		std::printf("\n");
		std::printf("%-2s | %10s | %10s | %10s\n", "r", "halo F", "halo B", "halo R");
		std::printf("%s\n", CLI_SECTION_SEPARATOR);

		for (const DistStartupRankReport& report : reports)
		{
			char rankLabel[32];
			FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

			std::printf("%-2s | %10u | %10u | %10u\n",
				rankLabel,
				report.N_haloF,
				report.N_haloB,
				report.N_haloR);
		}

		std::printf("\n");
		std::fflush(stdout);
	}

	void DistDiagnosticsMgr::ResetProgress(const uint64_t step)
	{
		ProgressState& state = GetProgressState();
		state = ProgressState{};
		state.initialized = true;
		state.startStep = step;
		state.lastStep = step;
		state.startSteady = steady_clock::now();
		state.pauseStart = state.startSteady;
	}

	void DistDiagnosticsMgr::SetProgressRunning(const bool running)
	{
		ProgressState& state = GetProgressState();
		if (state.running == running) { return; }

		const auto now = steady_clock::now();
		if (running)
		{
			state.startSteady += now - state.pauseStart;
		}
		else
		{
			state.pauseStart = now;
		}
		state.running = running;
	}

	void DistDiagnosticsMgr::PrintFinalProgressPeakRam(
		const MpiContext& mpiContext,
		const DistRankStepReport& localStepReport)
	{
		std::vector<DistRankStepReport> reports{};
		GatherRankStepReports(mpiContext, localStepReport, reports);
		if (!mpiContext.IsRoot() || reports.empty()) { return; }

		ProgressState& state = GetProgressState();
		if (state.N_lastLines == 0) { return; }

		const uint64_t processResidentMax = MaxReportField(
			reports, &DistRankStepReport::N_processResidentBytes);
		const int processResidentMaxRank = RankWithMax(
			reports, &DistRankStepReport::N_processResidentBytes);
		const uint64_t processPeakResidentMax = MaxReportField(
			reports, &DistRankStepReport::N_processPeakResidentBytes);
		const int processPeakResidentMaxRank = RankWithMax(
			reports, &DistRankStepReport::N_processPeakResidentBytes);

		const auto end = state.running ? steady_clock::now() : state.pauseStart;
		const double elapsedS = std::chrono::duration<double>(
			end - state.startSteady).count();
		const std::string progressLine = FormatProgressLine(
			reports, state.startStep, elapsedS,
			processResidentMax, processResidentMaxRank,
			processPeakResidentMax, processPeakResidentMaxRank);
		SPDLOG_INFO_MPI("Final progress report:", mpiContext);
		std::printf("\n%s\n\n", progressLine.c_str());
		std::fflush(stdout);
	}

	void DistDiagnosticsMgr::PrintLatestPerformanceSummaryAndProgress(
		const MpiContext& mpiContext,
		const Sim::SimParams& params,
		const Bench::BenchOpList& benchOps,
		const DistRankStepReport& localStepReport,
		const DistRankBenchReport& localBenchReport)
	{
		const DistRankProgressReport localReport{
			.stepReport = localStepReport,
			.benchReport = localBenchReport
		};
		std::vector<DistRankProgressReport> progressReports{};
		GatherRankProgressReports(mpiContext, localReport, progressReports);
		if (!mpiContext.IsRoot()) { return; }

		std::vector<DistRankStepReport> reports{};
		std::vector<DistRankBenchReport> benchReports{};
		reports.reserve(progressReports.size());
		benchReports.reserve(progressReports.size());
		for (const DistRankProgressReport& report : progressReports)
		{
			reports.push_back(report.stepReport);
			benchReports.push_back(report.benchReport);
		}

		PrintDashboard(
			reports, benchReports, benchOps, params);
	}

	void DistDiagnosticsMgr::PrintMemorySummary(
		const MpiContext& mpiContext,
		const Sim::SimParams& params,
		const DistRankStepReport& localStepReport)
	{
		std::vector<DistRankStepReport> reports{};
		GatherRankStepReports(mpiContext, localStepReport, reports);

		if (!mpiContext.IsRoot()) { return; }

		PrintMemoryTable(reports, params.byteUnit, params.backendType);
		std::printf("\n");
		std::fflush(stdout);
	}

} // namespace Dist
