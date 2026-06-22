// particle_migration_diagnostics.cpp

// external
#include <cstddef>
#include <cstdint>
#include <cstdio>

// internal
#include "particle_migration_diagnostics.h"



namespace Dist
{
	namespace
	{
		constexpr const char* CLI_SECTION_SEPARATOR =
			"--------------------------------------------------------------------------------";

		uint64_t SumAverageNs(const ParticleMigrationTimingReport& report)
		{
			uint64_t totalNs = 0;
			for (const uint64_t ns : report.totalNs)
			{
				totalNs += ns;
			}
			return report.sampleCount > 0 ? totalNs / report.sampleCount : 0;
		}

		uint64_t AverageNs(
			const ParticleMigrationTimingReport& report,
			const ParticleMigrationTimingOp op)
		{
			if (report.sampleCount == 0) { return 0; }
			return report.totalNs[GetIdx(op)] / report.sampleCount;
		}

		void FormatRankLabel(
			char* buffer,
			const size_t bufferSize,
			const int rank)
		{
			std::snprintf(buffer, bufferSize, "%d", rank);
		}
	}

	ParticleMigrationTimingScope::ParticleMigrationTimingScope(
		ParticleMigrationTimingReport* report,
		const ParticleMigrationTimingOp op)
		: report_(report)
		, op_(op)
		, t0_(Clock::now())
		, active_(report != nullptr && op != ParticleMigrationTimingOp::Count)
	{
		// nothing else to do
	}

	ParticleMigrationTimingScope::~ParticleMigrationTimingScope()
	{
		if (!active_) { return; }

		const auto t1 = Clock::now();
		const uint64_t ns = static_cast<uint64_t>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0_).count());

		report_->totalNs[GetIdx(op_)] += ns;
	}

	void BeginParticleMigrationTimingReport(
		ParticleMigrationTimingReport& report,
		const int rank)
	{
		report.rank = rank;
		report.sampleCount++;
	}

	std::vector<std::string> BuildParticleMigrationTimingLines(
		const std::vector<ParticleMigrationTimingReport>& reports,
		const Bench::TimeUnit unit)
	{
		std::vector<std::string> lines{};
		if (reports.empty()) { return lines; }

		lines.reserve(reports.size() + 2);

		char header[128];
		std::snprintf(
			header, sizeof(header),
			"%-2s | %9s | %9s | %9s | %9s | %9s | %9s %4s",
			"r", "migrPack", "migrCntX", "migrPayX",
			"migrAggr", "migrRebld", "migrTotal", "");
		lines.emplace_back(header);
		lines.emplace_back(CLI_SECTION_SEPARATOR);

		char unitLabel[8];
		std::snprintf(unitLabel, sizeof(unitLabel), "[%s]", Bench::UnitLabel(unit));

		for (const ParticleMigrationTimingReport& report : reports)
		{
			char rankLabel[32];
			FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

			char line[128];
			std::snprintf(
				line, sizeof(line),
				"%-2s | %9.2f | %9.2f | %9.2f | %9.2f | %9.2f | %9.2f %4s",
				rankLabel,
				Bench::ConvertFromNs(
					unit, AverageNs(report, ParticleMigrationTimingOp::Pack)),
				Bench::ConvertFromNs(
					unit, AverageNs(report, ParticleMigrationTimingOp::CountExchange)),
				Bench::ConvertFromNs(
					unit, AverageNs(report, ParticleMigrationTimingOp::PayloadExchange)),
				Bench::ConvertFromNs(
					unit, AverageNs(report, ParticleMigrationTimingOp::Aggregate)),
				Bench::ConvertFromNs(
					unit, AverageNs(report, ParticleMigrationTimingOp::Rebuild)),
				Bench::ConvertFromNs(unit, SumAverageNs(report)),
				unitLabel);
			lines.emplace_back(line);
		}

		return lines;
	}

} // namespace Dist
