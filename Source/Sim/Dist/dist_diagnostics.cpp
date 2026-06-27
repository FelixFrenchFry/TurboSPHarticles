// dist_diagnostics.cpp

// external
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <deque>
#include <string>
#include <utility>
#include <vector>

// internal
#include "dist_diagnostics.h"
#include "dist_sync_diagnostics.h"
#include "mpi_context.h"
#include "Source/Base/enum_hub.h"
#include "Source/Base/helpers.h"
#include "Source/Sim/Core/diagnostics.h"
#include "Source/Sim/Core/parameters_sim.h"



namespace Dist
{
	namespace
	{
		constexpr int DIST_DIAGNOSTICS_TAG_HARDWARE_REPORT = 8100;
		constexpr int DIST_DIAGNOSTICS_TAG_STEP_REPORT = 8101;
		constexpr int DIST_DIAGNOSTICS_TAG_STARTUP_REPORT = 8102;
		constexpr int DIST_DIAGNOSTICS_TAG_BENCH_REPORT = 8103;
		constexpr int DIST_DIAGNOSTICS_TAG_SYNC_TIMING_REPORT = 8104;
		constexpr int DIST_DIAGNOSTICS_RANK_TABLE_LIMIT = 16;
		constexpr int N_CLI_SECTION_SEPARATOR_CHARS = 80;
		constexpr const char* CLI_SECTION_SEPARATOR =
			"--------------------------------------------------------------------------------";

		using steady_clock = std::chrono::steady_clock;
		using Sample = std::pair<uint32_t, steady_clock::time_point>;

		struct ProgressState
		{
			bool initialized = false;
			uint32_t lastStep = 0;
			steady_clock::time_point startSteady{};
			std::deque<Sample> history{};
			int lastNumLines = 0;
		};

		struct DistHardwareReport
		{
			int rank = 0;
			char host[MPI_MAX_PROCESSOR_NAME] = {};
			uint32_t N_totalThreads = 0;
			uint32_t N_simThreads = 0;
			uint32_t N_visThreads = 0;
		};

		ProgressState& GetProgressState()
		{
			static ProgressState state{};
			return state;
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

		void PrintLine(int& lineCount)
		{
			std::printf("\x1b[2K\n");
			lineCount++;
		}

		template <typename... Args>
		void PrintLine(int& lineCount, const char* fmt, Args... args)
		{
			std::printf("\x1b[2K");
			std::printf(fmt, args...);
			std::printf("\n");
			lineCount++;
		}

		void FormatHms(double seconds, int& h, int& m, int& s)
		{
			seconds = std::max(seconds, 0.0);
			const int total = static_cast<int>(seconds + 0.5);
			h = total / 3600;
			m = (total % 3600) / 60;
			s = total % 60;
		}

		void FormatRankLabel(char* buffer, const size_t bufferSize, const int rank)
		{
			std::snprintf(buffer, bufferSize, "%d", rank);
		}

		const char* MemSpaceLabel(const Base::BackendVersion backendVersion)
		{
			return backendVersion == Base::BackendVersion::GPU01 ? "VRAM" : "RAM";
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

			int hostLen = 0;
			if (MPI_Get_processor_name(report.host, &hostLen) != MPI_SUCCESS ||
				hostLen <= 0)
			{
				std::snprintf(report.host, sizeof(report.host), "%s", "unknown");
			}
			else
			{
				const int lastIdx = std::min(hostLen, MPI_MAX_PROCESSOR_NAME - 1);
				report.host[lastIdx] = '\0';
			}

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

			mpiContext.WaitAll(requests);
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

			mpiContext.WaitAll(requests);
		}

		void PrintHardwareReports(const std::vector<DistHardwareReport>& reports)
		{
			std::printf("\n");
			PrintCliSectionHeader("distributed hardware specifications");
			PrintCliSeparator();

			std::printf("%-2s | %-30s | %13s | %10s | %10s\n",
				"r", "host name", "total threads",
				"sim budget", "vis budget");
			std::printf("%s\n", CLI_SECTION_SEPARATOR);

			for (const DistHardwareReport& report : reports)
			{
				char rankLabel[32];
				FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

				std::printf("%-2s | %-30s | %13u | %10u | %10u\n",
					rankLabel,
					report.host,
					report.N_totalThreads,
					report.N_simThreads,
					report.N_visThreads);
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

			mpiContext.WaitAll(requests);
		}

		void GatherRankBenchReports(
			const MpiContext& mpiContext,
			const DistRankBenchReport& localReport,
			std::vector<DistRankBenchReport>& reports)
		{
			std::vector<MPI_Request> requests{};

			if (mpiContext.IsRoot())
			{
				reports.assign(static_cast<uint64_t>(mpiContext.Size()), DistRankBenchReport{});
				reports[static_cast<uint64_t>(mpiContext.Rank())] = localReport;
				requests.reserve(static_cast<uint64_t>(std::max(0, mpiContext.Size() - 1)));

				for (int rank = 0; rank < mpiContext.Size(); rank++)
				{
					if (rank == mpiContext.RootRank()) { continue; }

					requests.emplace_back(MPI_REQUEST_NULL);
					mpiContext.IRecvBytes(
						&reports[static_cast<uint64_t>(rank)],
						static_cast<int>(sizeof(DistRankBenchReport)),
						rank,
						DIST_DIAGNOSTICS_TAG_BENCH_REPORT,
						requests.back());
				}
			}
			else
			{
				requests.emplace_back(MPI_REQUEST_NULL);
				mpiContext.ISendBytes(
					&localReport,
					static_cast<int>(sizeof(DistRankBenchReport)),
					mpiContext.RootRank(),
					DIST_DIAGNOSTICS_TAG_BENCH_REPORT,
					requests.back());
			}

			mpiContext.WaitAll(requests);
		}

		void GatherDistSyncTimingReports(
			const MpiContext& mpiContext,
			const DistSyncTimingReport& localReport,
			std::vector<DistSyncTimingReport>& reports)
		{
			std::vector<MPI_Request> requests{};

			if (mpiContext.IsRoot())
			{
				reports.assign(static_cast<uint64_t>(mpiContext.Size()), DistSyncTimingReport{});
				reports[static_cast<uint64_t>(mpiContext.Rank())] = localReport;
				requests.reserve(static_cast<uint64_t>(std::max(0, mpiContext.Size() - 1)));

				for (int rank = 0; rank < mpiContext.Size(); rank++)
				{
					if (rank == mpiContext.RootRank()) { continue; }

					requests.emplace_back(MPI_REQUEST_NULL);
					mpiContext.IRecvBytes(
						&reports[static_cast<uint64_t>(rank)],
						static_cast<int>(sizeof(DistSyncTimingReport)),
						rank,
						DIST_DIAGNOSTICS_TAG_SYNC_TIMING_REPORT,
						requests.back());
				}
			}
			else
			{
				requests.emplace_back(MPI_REQUEST_NULL);
				mpiContext.ISendBytes(
					&localReport,
					static_cast<int>(sizeof(DistSyncTimingReport)),
					mpiContext.RootRank(),
					DIST_DIAGNOSTICS_TAG_SYNC_TIMING_REPORT,
					requests.back());
			}

			mpiContext.WaitAll(requests);
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

		uint64_t MinOrZero(const uint64_t a, const uint64_t b)
		{
			if (a == 0) { return b; }
			if (b == 0) { return a; }

			return std::min(a, b);
		}

		uint64_t SaturatingSubtract(const uint64_t a, const uint64_t b)
		{
			return a > b ? a - b : 0;
		}

		bool IsSingleBenchOpGroup(
			const Bench::BenchOpGroupDef& group,
			const Bench::BenchOp op)
		{
			return group.ops.size() == 1 && group.ops[0] == op;
		}

		bool IsTotalSimGroup(const Bench::BenchOpGroupDef& group)
		{
			return IsSingleBenchOpGroup(group, Bench::BenchOp::Total_Sim);
		}

		bool IsDistBenchOp(const Bench::BenchOp op)
		{
			return op == Bench::BenchOp::Dist_ParticleMigration ||
				op == Bench::BenchOp::Dist_ParticleClass ||
				op == Bench::BenchOp::Dist_HaloExchange ||
				op == Bench::BenchOp::Dist_StateSync;
		}

		bool IsDistBenchGroup(const Bench::BenchOpGroupDef& group)
		{
			for (const Bench::BenchOp op : group.ops)
			{
				if (IsDistBenchOp(op)) { return true; }
			}

			return false;
		}

		int FindTotalSimGroupIdx(
			const std::vector<Bench::BenchOpGroupDef>& opGroups)
		{
			for (uint32_t i = 0; i < opGroups.size(); i++)
			{
				if (IsTotalSimGroup(opGroups[i]))
				{
					return static_cast<int>(i);
				}
			}

			return -1;
		}

		uint64_t BenchGroupTimeNs(
			const DistRankBenchReport& report,
			const Bench::BenchOpGroupDef& group,
			const bool useAvg)
		{
			uint64_t groupTimeNs = 0;
			for (const Bench::BenchOp op : group.ops)
			{
				const uint8_t i = Bench::GetIdx(op);
				const uint64_t opTimeNs = useAvg ? report.avgNs[i] : report.latestNs[i];
				groupTimeNs = MinOrZero(groupTimeNs, opTimeNs);
			}

			return groupTimeNs;
		}

		std::string BenchGroupAlias(const Bench::BenchOpGroupDef& group)
		{
			if (IsTotalSimGroup(group)) { return "total"; }
			if (group.ops.empty()) { return "<empty>"; }

			std::string alias{};
			for (std::size_t k = 0; k < group.ops.size(); k++)
			{
				const Bench::BenchOp op = group.ops[k];
				const auto& def = Bench::Definitions[Bench::GetIdx(op)];

				if (k > 0) { alias += " + "; }
				alias += std::string(def.alias);
			}

			return alias;
		}

		struct DistBenchGroupSummary
		{
			uint64_t latestNs = 0;
			uint64_t avgNs = 0;
		};

		DistBenchGroupSummary SummarizeBenchGroup(
			const std::vector<DistRankBenchReport>& reports,
			const Bench::BenchOpGroupDef& group)
		{
			DistBenchGroupSummary summary{};

			for (const DistRankBenchReport& report : reports)
			{
				const uint64_t latestNs = BenchGroupTimeNs(report, group, false);
				const uint64_t avgNs = BenchGroupTimeNs(report, group, true);

				if (latestNs > summary.latestNs)
				{
					summary.latestNs = latestNs;
				}
				summary.avgNs = std::max(summary.avgNs, avgNs);
			}

			return summary;
		}

		const DistRankBenchReport* FindReportWithMaxBenchGroupTime(
			const std::vector<DistRankBenchReport>& reports,
			const Bench::BenchOpGroupDef& group,
			const bool useAvg)
		{
			const DistRankBenchReport* bestReport = nullptr;
			uint64_t bestNs = 0;

			for (const DistRankBenchReport& report : reports)
			{
				const uint64_t timeNs = BenchGroupTimeNs(report, group, useAvg);
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
			std::vector<DistBenchGroupSummary> groups{};
			DistBenchGroupSummary other{};
			int totalSimIdx = -1;
			int latestCriticalRank = -1;
			int avgCriticalRank = -1;
		};

		DistBenchTableSummary BuildCriticalRankBenchSummaries(
			const std::vector<DistRankBenchReport>& reports,
			const std::vector<Bench::BenchOpGroupDef>& opGroups)
		{
			DistBenchTableSummary tableSummary{};
			tableSummary.groups.reserve(opGroups.size());

			tableSummary.totalSimIdx = FindTotalSimGroupIdx(opGroups);

			if (tableSummary.totalSimIdx < 0)
			{
				for (const Bench::BenchOpGroupDef& group : opGroups)
				{
					tableSummary.groups.push_back(SummarizeBenchGroup(reports, group));
				}
				return tableSummary;
			}

			const Bench::BenchOpGroupDef& totalGroup =
				opGroups[static_cast<uint32_t>(tableSummary.totalSimIdx)];
			const DistRankBenchReport* latestTotalReport =
				FindReportWithMaxBenchGroupTime(reports, totalGroup, false);
			const DistRankBenchReport* avgTotalReport =
				FindReportWithMaxBenchGroupTime(reports, totalGroup, true);
			tableSummary.latestCriticalRank =
				latestTotalReport != nullptr ? latestTotalReport->rank : -1;
			tableSummary.avgCriticalRank =
				avgTotalReport != nullptr ? avgTotalReport->rank : -1;

			uint64_t accountedLatestNs = 0;
			uint64_t accountedAvgNs = 0;

			for (uint32_t i = 0; i < opGroups.size(); i++)
			{
				const Bench::BenchOpGroupDef& group = opGroups[i];
				DistBenchGroupSummary groupSummary{};

				if (latestTotalReport != nullptr)
				{
					groupSummary.latestNs =
						BenchGroupTimeNs(*latestTotalReport, group, false);
				}
				if (avgTotalReport != nullptr)
				{
					groupSummary.avgNs =
						BenchGroupTimeNs(*avgTotalReport, group, true);
				}

				if (static_cast<int>(i) != tableSummary.totalSimIdx)
				{
					accountedLatestNs += groupSummary.latestNs;
					accountedAvgNs += groupSummary.avgNs;
				}

				tableSummary.groups.push_back(groupSummary);
			}

			const DistBenchGroupSummary& totalSummary =
				tableSummary.groups[static_cast<uint32_t>(tableSummary.totalSimIdx)];
			tableSummary.other.latestNs =
				SaturatingSubtract(totalSummary.latestNs, accountedLatestNs);
			tableSummary.other.avgNs =
				SaturatingSubtract(totalSummary.avgNs, accountedAvgNs);

			return tableSummary;
		}

		struct DistRankTimingSummary
		{
			int rank = 0;

			uint64_t backendAvgNs = 0;
			uint64_t distMigrAvgNs = 0;
			uint64_t distClassAvgNs = 0;
			uint64_t distHaloAvgNs = 0;
			uint64_t distSyncAvgNs = 0;
			uint64_t otherAvgNs = 0;
		};

		std::vector<DistRankTimingSummary> BuildRankTimingSummaries(
			const std::vector<DistRankBenchReport>& reports,
			const std::vector<Bench::BenchOpGroupDef>& opGroups)
		{
			std::vector<DistRankTimingSummary> summaries{};
			if (reports.empty() || opGroups.empty()) { return summaries; }

			const int totalSimIdx = FindTotalSimGroupIdx(opGroups);
			if (totalSimIdx < 0) { return summaries; }

			summaries.reserve(reports.size());

			for (const DistRankBenchReport& report : reports)
			{
				DistRankTimingSummary summary{};
				summary.rank = report.rank;

				uint64_t totalAvgNs = 0;

				for (uint32_t i = 0; i < opGroups.size(); i++)
				{
					const Bench::BenchOpGroupDef& group = opGroups[i];
					const uint64_t avgNs = BenchGroupTimeNs(report, group, true);

					if (static_cast<int>(i) == totalSimIdx)
					{
						totalAvgNs = avgNs;
						continue;
					}

					if (IsSingleBenchOpGroup(group,
						Bench::BenchOp::Dist_ParticleMigration))
					{
						summary.distMigrAvgNs += avgNs;
					}
					else if (IsSingleBenchOpGroup(group,
						Bench::BenchOp::Dist_ParticleClass))
					{
						summary.distClassAvgNs += avgNs;
					}
					else if (IsSingleBenchOpGroup(group,
						Bench::BenchOp::Dist_HaloExchange))
					{
						summary.distHaloAvgNs += avgNs;
					}
					else if (IsSingleBenchOpGroup(group,
						Bench::BenchOp::Dist_StateSync))
					{
						summary.distSyncAvgNs += avgNs;
					}
					else if (!IsDistBenchGroup(group))
					{
						summary.backendAvgNs += avgNs;
					}
				}

				const uint64_t displayedAvgNs =
					summary.backendAvgNs +
					summary.distMigrAvgNs +
					summary.distClassAvgNs +
					summary.distHaloAvgNs +
					summary.distSyncAvgNs;
				summary.otherAvgNs =
					SaturatingSubtract(totalAvgNs, displayedAvgNs);

				summaries.push_back(summary);
			}

			return summaries;
		}

		void FormatPercentLabel(char* buffer, const size_t bufferSize, const double pct)
		{
			std::snprintf(buffer, bufferSize, "%.1f %%", std::max(0.0, pct));
		}

		void PrintParticleLayoutTable(
			int& lineCount,
			const std::vector<DistRankStepReport>& reports)
		{
			PrintLine(lineCount,
				"%-2s | %10s | %10s | %10s | %10s | %10s | %10s",
				"r", "inner F", "outer F", "halo F",
				"inner B", "outer B", "halo B");
			PrintLine(lineCount, "%s", CLI_SECTION_SEPARATOR);

			for (const DistRankStepReport& report : reports)
			{
				char rankLabel[32];
				FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

				PrintLine(lineCount,
					"%-2s | %10u | %10u | %10u | %10u | %10u | %10u",
					rankLabel,
					report.innerFluidParticleCount,
					report.outerFluidParticleCount,
					report.haloFluidParticleCount,
					report.innerBoundaryParticleCount,
					report.outerBoundaryParticleCount,
					report.haloBoundaryParticleCount);
			}
		}

		void PrintParticleExchangeTable(
			int& lineCount,
			const std::vector<DistRankStepReport>& reports)
		{
			PrintLine(lineCount,
				"%-2s | %17s | %17s | %16s | %16s",
				"r", "migr F send/recv", "halo F send/recv",
				"migr B send/recv", "halo B send/recv");
			PrintLine(lineCount, "%s", CLI_SECTION_SEPARATOR);

			for (const DistRankStepReport& report : reports)
			{
				char rankLabel[32];
				FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

				PrintLine(lineCount,
					"%-2s | %7u / %-7u | %7u / %-7u | %6u / %-7u | %6u / %-7u",
					rankLabel,
					report.migratedFluidParticleSendCount,
					report.migratedFluidParticleRecvCount,
					report.haloFluidParticleSendCount,
					report.haloFluidParticleRecvCount,
					report.migratedBoundaryParticleSendCount,
					report.migratedBoundaryParticleRecvCount,
					report.haloBoundaryParticleSendCount,
					report.haloBoundaryParticleRecvCount);
			}
		}

		void PrintDistSyncTimingTable(
			int& lineCount,
			const std::vector<DistSyncTimingReport>& reports,
			const Bench::TimeUnit unit)
		{
			const std::vector<std::string> lines =
				BuildDistSyncTimingLines(reports, unit);

			for (const std::string& line : lines)
			{
				PrintLine(lineCount, "%s", line.c_str());
			}
		}

		void PrintBenchmarkSummaryTable(
			int& lineCount,
			const std::vector<DistRankBenchReport>& reports,
			const std::vector<Bench::BenchOpGroupDef>& opGroups,
			const Bench::TimeUnit unit)
		{
			if (reports.empty() || opGroups.empty()) { return; }

			const DistBenchTableSummary tableSummary =
				BuildCriticalRankBenchSummaries(reports, opGroups);

			const uint64_t totalLatestNs = tableSummary.totalSimIdx >= 0
				? tableSummary.groups[static_cast<uint64_t>(
					tableSummary.totalSimIdx)].latestNs
				: 0;
			const uint64_t totalAvgNs = tableSummary.totalSimIdx >= 0
				? tableSummary.groups[static_cast<uint64_t>(
					tableSummary.totalSimIdx)].avgNs
				: 0;

			const char* unitStr = Bench::UnitLabel(unit);

			char latestLabel[32];
			char avgLabel[32];
			std::snprintf(latestLabel, sizeof(latestLabel), "latest [%s]", unitStr);
			std::snprintf(avgLabel, sizeof(avgLabel), "avg [%s]", unitStr);

			PrintLine(lineCount,
				"%-33.33s | %11s  %8s | %10s  %8s",
				"benchmark summary", latestLabel, "% total", avgLabel, "% total");
			PrintLine(lineCount, "%s", CLI_SECTION_SEPARATOR);

			auto PrintBenchRow =
				[&](const char* alias,
					const DistBenchGroupSummary& summary,
					const bool isTotalSim)
			{
				const double latestTime =
					Bench::ConvertFromNs(unit, summary.latestNs);
				const double avgTime =
					Bench::ConvertFromNs(unit, summary.avgNs);

				char pctLatestLabel[16];
				char pctAvgLabel[16];
				if (isTotalSim)
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

				PrintLine(lineCount,
					"%-33.33s | %11.2f  %8s | %10.2f  %8s",
					alias, latestTime, pctLatestLabel,
					avgTime, pctAvgLabel);
			};

			for (uint32_t i = 0; i < opGroups.size(); i++)
			{
				if (static_cast<int>(i) == tableSummary.totalSimIdx) { continue; }
				const std::string alias = BenchGroupAlias(opGroups[i]);
				PrintBenchRow(alias.c_str(), tableSummary.groups[i], false);
			}

			if (tableSummary.totalSimIdx >= 0)
			{
				PrintBenchRow("other", tableSummary.other, false);
				PrintLine(lineCount, "%s", CLI_SECTION_SEPARATOR);
				PrintBenchRow(
					"total",
					tableSummary.groups[static_cast<uint32_t>(
						tableSummary.totalSimIdx)],
					true);
				PrintLine(lineCount,
					"%-33.33s | %11d  %8s | %10d  %8s",
					"critical rank for latest/avg",
					tableSummary.latestCriticalRank, "---",
					tableSummary.avgCriticalRank, "---");
			}
		}

		void PrintRankTimingTable(
			int& lineCount,
			const std::vector<DistRankBenchReport>& reports,
			const std::vector<Bench::BenchOpGroupDef>& opGroups,
			const Bench::TimeUnit unit)
		{
			const std::vector<DistRankTimingSummary> summaries =
				BuildRankTimingSummaries(reports, opGroups);
			if (summaries.empty()) { return; }

			char unitLabel[8];
			std::snprintf(unitLabel, sizeof(unitLabel), "[%s]", Bench::UnitLabel(unit));

			PrintLine(lineCount,
				"%-2s | %10s | %9s | %9s | %9s | %9s | %9s %4s",
				"r", "backend", "distMigr", "distClass",
				"distHalo", "distSync", "other", "avg");
			PrintLine(lineCount, "%s", CLI_SECTION_SEPARATOR);

			for (const DistRankTimingSummary& summary : summaries)
			{
				char rankLabel[32];
				FormatRankLabel(rankLabel, sizeof(rankLabel), summary.rank);

				PrintLine(lineCount,
					"%-2s | %10.2f | %9.2f | %9.2f | %9.2f | %9.2f | %9.2f %4s",
					rankLabel,
					Bench::ConvertFromNs(unit, summary.backendAvgNs),
					Bench::ConvertFromNs(unit, summary.distMigrAvgNs),
					Bench::ConvertFromNs(unit, summary.distClassAvgNs),
					Bench::ConvertFromNs(unit, summary.distHaloAvgNs),
					Bench::ConvertFromNs(unit, summary.distSyncAvgNs),
					Bench::ConvertFromNs(unit, summary.otherAvgNs),
					unitLabel);
			}
		}

		std::string BuildMemoryTable(
			const std::vector<DistRankStepReport>& reports,
			const Mem::ByteUnit unit,
			const Base::BackendVersion backendVersion)
		{
			std::string table{};
			char sectionTitle[64];
			std::snprintf(sectionTitle, sizeof(sectionTitle),
				"distributed memory usage [%s in %s]",
				MemSpaceLabel(backendVersion), Mem::UnitLabel(unit));

			char headerLine[128];
			int N_chars = std::snprintf(headerLine, sizeof(headerLine), "--- [ %s ] ",
				sectionTitle);
			N_chars = std::max(0, N_chars);

			table += headerLine;
			for (int i = N_chars; i < N_CLI_SECTION_SEPARATOR_CHARS; i++)
			{
				table += '-';
			}
			table += '\n';
			table += CLI_SECTION_SEPARATOR;
			table += '\n';

			char line[256];
			std::snprintf(line, sizeof(line),
				"%-2s | %8s | %8s | %8s | %8s | %8s | %8s | %9s",
				"r", "backend", "global", "local",
				"meta", "exch", "misc", "total");
			table += line;
			table += '\n';
			table += CLI_SECTION_SEPARATOR;
			table += '\n';

			for (const DistRankStepReport& report : reports)
			{
				char rankLabel[32];
				FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

				std::snprintf(line, sizeof(line),
					"%-2s | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %9.2f",
					rankLabel,
					Mem::ConvertFromBytes(unit, report.maxBackendMemAllocBytes),
					Mem::ConvertFromBytes(unit, report.maxDistGlobalStateMemAllocBytes),
					Mem::ConvertFromBytes(unit, report.maxDistLocalStateMemAllocBytes),
					Mem::ConvertFromBytes(unit, report.maxDistParticleMetadataMemAllocBytes),
					Mem::ConvertFromBytes(unit, report.maxDistExchangeBuffersMemAllocBytes),
					Mem::ConvertFromBytes(unit, report.maxDistMiscMemAllocBytes),
					Mem::ConvertFromBytes(unit, report.maxTotalTrackedMemAllocBytes));
				table += line;
				table += '\n';
			}

			return table;
		}

		void PrintMemoryTable(
			const std::vector<DistRankStepReport>& reports,
			const Mem::ByteUnit unit,
			const Base::BackendVersion backendVersion)
		{
			const std::string table = BuildMemoryTable(reports, unit, backendVersion);
			std::printf("%s", table.c_str());
		}

		void PrintAggregateTable(
			int& lineCount,
			const std::vector<DistRankStepReport>& reports,
			const Sim::SimParams& params)
		{
			const uint64_t totalOwnedFluid =
				SumReportField(reports, &DistRankStepReport::ownedFluidParticleCount);
			const uint64_t totalHaloFluid =
				SumReportField(reports, &DistRankStepReport::haloFluidParticleCount);
			const uint64_t totalOwnedBoundary =
				SumReportField(reports, &DistRankStepReport::ownedBoundaryParticleCount);
			const uint64_t totalHaloBoundary =
				SumReportField(reports, &DistRankStepReport::haloBoundaryParticleCount);
			const uint64_t totalMigrationSent =
				SumReportField(reports, &DistRankStepReport::migratedFluidParticleSendCount);
			const uint64_t totalMigrationReceived =
				SumReportField(reports, &DistRankStepReport::migratedFluidParticleRecvCount);
			const uint64_t totalHaloFluidSent =
				SumReportField(reports, &DistRankStepReport::haloFluidParticleSendCount);
			const uint64_t totalHaloFluidReceived =
				SumReportField(reports, &DistRankStepReport::haloFluidParticleRecvCount);
			const uint64_t totalBoundaryMigrationSent =
				SumReportField(reports, &DistRankStepReport::migratedBoundaryParticleSendCount);
			const uint64_t totalBoundaryMigrationReceived =
				SumReportField(reports, &DistRankStepReport::migratedBoundaryParticleRecvCount);
			const uint64_t totalHaloBoundarySent =
				SumReportField(reports, &DistRankStepReport::haloBoundaryParticleSendCount);
			const uint64_t totalHaloBoundaryReceived =
				SumReportField(reports, &DistRankStepReport::haloBoundaryParticleRecvCount);

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
			PrintLine(lineCount,
				"particles | owned F %" PRIu64 " | halo F %" PRIu64
				" | owned B %" PRIu64 " | halo B %" PRIu64,
				totalOwnedFluid, totalHaloFluid, totalOwnedBoundary, totalHaloBoundary);
			PrintLine(lineCount,
				"traffic   | migrated F %" PRIu64 " / %" PRIu64
				" | halo F sent/recv %" PRIu64 " / %" PRIu64
				" | migrated B %" PRIu64 " / %" PRIu64
				" | halo B sent/recv %" PRIu64 " / %" PRIu64,
				totalMigrationSent, totalMigrationReceived,
				totalHaloFluidSent, totalHaloFluidReceived,
				totalBoundaryMigrationSent, totalBoundaryMigrationReceived,
				totalHaloBoundarySent, totalHaloBoundaryReceived);
			PrintLine(lineCount,
				"timing    | min/avg/max %.3f / %.3f / %.3f %s | slowest rank %d",
				latestMin, latestAvg, latestMax, timeUnitStr, slowestRank);
		}

		void PrintDashboard(
			const std::vector<DistRankStepReport>& reports,
			const std::vector<DistRankBenchReport>& benchReports,
			const std::vector<DistSyncTimingReport>& syncTimingReports,
			const std::vector<Bench::BenchOpGroupDef>& benchOpGroups,
			const Sim::SimParams& params)
		{
			if (reports.empty()) { return; }

			ProgressState& state = GetProgressState();
			const auto now = steady_clock::now();
			if (!state.initialized)
			{
				state.initialized = true;
				state.startSteady = now;
				state.history.clear();
				state.lastNumLines = 0;
				state.lastStep = 0;
			}

			const uint32_t step = reports.front().step;
			if (step <= state.lastStep) { return; }
			state.lastStep = step;

			state.history.emplace_back(step, now);
			constexpr uint32_t MAX_HISTORY = 10;
			if (state.history.size() > MAX_HISTORY) { state.history.pop_front(); }

			double elapsedS =
				std::chrono::duration<double>(now - state.startSteady).count();
			elapsedS = std::max(elapsedS, 1e-14);

			double stepsPerSec = 0.0;
			if (state.history.size() >= 2)
			{
				const auto& oldest = state.history.front();
				const auto& newest = state.history.back();
				const double dt =
					std::chrono::duration<double>(newest.second - oldest.second).count();
				const uint32_t ds = newest.first - oldest.first;

				if (dt > 1e-14 && ds > 0)
				{
					stepsPerSec = static_cast<double>(ds) / dt;
				}
			}
			if (stepsPerSec <= 0.0)
			{
				stepsPerSec = static_cast<double>(step) / elapsedS;
			}

			const int64_t maxSteps = reports.front().maxSteps;
			const bool hasTotal = maxSteps > 0;
			double percent = 0.0;
			if (hasTotal)
			{
				const double progress = std::min(1.0, std::max(0.0,
					static_cast<double>(step) / static_cast<double>(maxSteps)));
				percent = progress * 100.0;
			}

			if (state.lastNumLines > 0)
			{
				std::printf("\x1b[%dA\r", state.lastNumLines);
			}

			int lineCount = 0;
			if (static_cast<int>(reports.size()) <= DIST_DIAGNOSTICS_RANK_TABLE_LIMIT)
			{
				PrintParticleLayoutTable(lineCount, reports);
				PrintLine(lineCount);
				PrintParticleExchangeTable(lineCount, reports);
				PrintLine(lineCount);
				PrintDistSyncTimingTable(
					lineCount, syncTimingReports, params.timeUnit);
			}
			else
			{
				PrintAggregateTable(lineCount, reports, params);
			}

			PrintLine(lineCount);
			PrintBenchmarkSummaryTable(lineCount, benchReports, benchOpGroups, params.timeUnit);
			if (static_cast<int>(benchReports.size()) <= DIST_DIAGNOSTICS_RANK_TABLE_LIMIT)
			{
				PrintLine(lineCount);
				PrintRankTimingTable(lineCount, benchReports, benchOpGroups, params.timeUnit);
			}
			PrintLine(lineCount);
			int elH, elM, elS;
			FormatHms(elapsedS, elH, elM, elS);
			if (hasTotal)
			{
				PrintLine(lineCount,
					"%02d:%02d:%02d elapsed | %u/%lld steps (%.1f %%) | %.3f steps/s (recently)",
					elH, elM, elS,
					step, static_cast<long long>(maxSteps), percent, stepsPerSec);
			}
			else
			{
				PrintLine(lineCount,
					"%02d:%02d:%02d elapsed | %u steps | %.3f steps/s (recently)",
					elH, elM, elS, step, stepsPerSec);
			}

			std::fflush(stdout);
			state.lastNumLines = lineCount;
		}
	}

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
		std::vector<DistStartupRankReport> reports{};
		GatherStartupRankReports(mpiContext, localReport, reports);

		if (!mpiContext.IsRoot()) { return; }

		std::printf("\n");
		PrintCliSectionHeader("distributed simulation setup");
		PrintCliSeparator();

		std::printf("%-30s = %s\n", "sim backend", setup.backendVersion);
		std::printf("%-30s = %s\n", "sim scenario", setup.scenarioType);
		std::printf("%-30s = %s\n", "neighbor search method",
			setup.neighborSearchMethod);
		std::printf("%-30s = %s\n", "neighbor list storage",
			setup.neighborListStorage);

		std::printf("%-30s = %s\n", "decomp method", setup.domainDecompMethod);
		std::printf("%-30s = %s\n", "mpi thread support",
			MpiContext::ThreadSupportName(setup.mpiThreadSupport));
		std::printf("%-30s = [ %u / %u ]\n", "global particles [F / B]",
			setup.globalFluidParticleCount,
			setup.globalBoundaryParticleCount);
		std::printf("\n");

		std::printf("%-2s | %-35s | %17s | %17s\n",
			"r", "neighbor ranks", "initially owned F", "initially owned B");
		std::printf("%s\n", CLI_SECTION_SEPARATOR);

		for (const DistStartupRankReport& report : reports)
		{
			char rankLabel[32];
			FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

			std::printf("%-2s | %-35s | %17u | %17u\n",
				rankLabel,
				report.neighborRanks,
				report.ownedFluidParticleCount,
				report.ownedBoundaryParticleCount);
		}

		std::printf("\n");
		std::fflush(stdout);
	}

	void DistDiagnosticsMgr::ResetProgress()
	{
		ProgressState& state = GetProgressState();
		state = ProgressState{};
		state.initialized = true;
		state.startSteady = steady_clock::now();
	}

	void DistDiagnosticsMgr::PrintLatestPerformanceSummaryAndProgress(
		const MpiContext& mpiContext,
		const Sim::SimParams& params,
		const std::vector<Bench::BenchOpGroupDef>& benchOpGroups,
		const DistRankStepReport& localStepReport,
		const DistRankBenchReport& localBenchReport,
		const DistSyncTimingReport& localDistSyncTimingReport)
	{
		std::vector<DistRankStepReport> reports{};
		GatherRankStepReports(mpiContext, localStepReport, reports);

		std::vector<DistRankBenchReport> benchReports{};
		GatherRankBenchReports(mpiContext, localBenchReport, benchReports);

		std::vector<DistSyncTimingReport> syncTimingReports{};
		GatherDistSyncTimingReports(
			mpiContext, localDistSyncTimingReport, syncTimingReports);

		if (!mpiContext.IsRoot()) { return; }

		PrintDashboard(
			reports, benchReports, syncTimingReports, benchOpGroups, params);
	}

	void DistDiagnosticsMgr::PrintMemorySummary(
		const MpiContext& mpiContext,
		const Sim::SimParams& params,
		const DistRankStepReport& localStepReport)
	{
		std::vector<DistRankStepReport> reports{};
		GatherRankStepReports(mpiContext, localStepReport, reports);

		if (!mpiContext.IsRoot()) { return; }

		PrintMemoryTable(reports, params.byteUnit, params.backendVersion);
		std::printf("\n");
		std::fflush(stdout);
	}

} // namespace Dist
