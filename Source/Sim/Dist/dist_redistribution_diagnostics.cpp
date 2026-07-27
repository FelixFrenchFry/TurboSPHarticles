// dist_redistribution_diagnostics.cpp

// external
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>

// internal
#include "dist_redistribution_diagnostics.h"



using namespace Dist;

namespace
{
	constexpr int MIN_TIMING_COLUMN_WIDTH = 7;

	struct ParticleRedistributionTimingColumn
	{
		std::string_view title;
		ParticleRedistributionTimingOp op;
	};

	constexpr std::array<
		ParticleRedistributionTimingColumn,
		GetIdx(ParticleRedistributionTimingOp::Count) - 1>
		TIMING_COLUMNS
	{{
		{ "routeF", ParticleRedistributionTimingOp::RouteFluid },
		{ "routeB", ParticleRedistributionTimingOp::RouteBoundary },
		{ "pack", ParticleRedistributionTimingOp::Pack },
		{ "exch", ParticleRedistributionTimingOp::Exchange },
		{ "prepF", ParticleRedistributionTimingOp::FluidPrep },
		{ "keptF", ParticleRedistributionTimingOp::KeptFluid },
		{ "migrF", ParticleRedistributionTimingOp::MigratedFluid },
		{ "haloF", ParticleRedistributionTimingOp::HaloFluid },
		{ "prepB", ParticleRedistributionTimingOp::BoundaryPrep },
		{ "keptB", ParticleRedistributionTimingOp::KeptBoundary },
		{ "haloB", ParticleRedistributionTimingOp::HaloBoundary },
		{ "backend", ParticleRedistributionTimingOp::Backend },
		{ "wPack", ParticleRedistributionTimingOp::SyncPack },
		{ "wCounts", ParticleRedistributionTimingOp::SyncCounts },
		{ "wRebuild", ParticleRedistributionTimingOp::SyncRebuild },
		{ "wBackend", ParticleRedistributionTimingOp::SyncBackend }
	}};

	int TimingColumnWidth(const std::string_view title)
	{
		return title.size() > MIN_TIMING_COLUMN_WIDTH
			? static_cast<int>(title.size())
			: MIN_TIMING_COLUMN_WIDTH;
	}

	uint64_t AverageNs(
		const ParticleRedistributionTimingReport& report,
		const ParticleRedistributionTimingOp op)
	{
		if (report.sampleCount == 0) { return 0; }

		uint64_t totalNs = report.totalNs[GetIdx(op)];
		if (op == ParticleRedistributionTimingOp::RouteFluid)
		{
			totalNs += report.totalNs[
				GetIdx(ParticleRedistributionTimingOp::RoutePrep)];
		}

		return totalNs / report.sampleCount;
	}

	void FormatRankLabel(
		char* buffer,
		const size_t bufferSize,
		const int rank)
	{
		std::snprintf(buffer, bufferSize, "%d", rank);
	}

} // namespace anonymous

namespace Dist
{
	ParticleRedistributionTimingScope::ParticleRedistributionTimingScope(
		ParticleRedistributionTimingReport* report,
		const ParticleRedistributionTimingOp op)
		: report_(report)
		, op_(op)
		, t0_(Clock::now())
		, active_(
			report != nullptr &&
			op != ParticleRedistributionTimingOp::Count)
	{
		// nothing else to do
	}

	ParticleRedistributionTimingScope::~ParticleRedistributionTimingScope()
	{
		if (!active_) { return; }

		const auto t1 = Clock::now();
		const uint64_t ns = static_cast<uint64_t>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0_).count());

		report_->totalNs[GetIdx(op_)] += ns;
	}

	void BeginParticleRedistributionTimingReport(
		ParticleRedistributionTimingReport& report,
		const int rank)
	{
		report.rank = rank;
		report.sampleCount++;
	}

	std::vector<std::string> BuildParticleRedistributionTimingLines(
		const std::vector<ParticleRedistributionTimingReport>& reports,
		const Bench::TimeUnit unit)
	{
		std::vector<std::string> lines{};
		if (reports.empty()) { return lines; }

		lines.reserve(reports.size() + 2);

		// build the compact header using the minimum width required by each title
		std::string header = "r ";
		char cell[64];
		for (const ParticleRedistributionTimingColumn& column : TIMING_COLUMNS)
		{
			std::snprintf(
				cell, sizeof(cell),
				" | %*s", TimingColumnWidth(column.title), column.title.data());
			header += cell;
		}
		lines.emplace_back(header);
		lines.emplace_back(lines.front().size(), '-');

		char unitLabel[8];
		std::snprintf(unitLabel, sizeof(unitLabel), "[%s]", Bench::UnitLabel(unit));

		for (const ParticleRedistributionTimingReport& report : reports)
		{
			char rankLabel[32];
			FormatRankLabel(rankLabel, sizeof(rankLabel), report.rank);

			char rankCell[32];
			std::snprintf(
				rankCell, sizeof(rankCell),
				"%-2s", rankLabel);
			std::string line = rankCell;

			// append one average using the same width as its header
			for (const ParticleRedistributionTimingColumn& column : TIMING_COLUMNS)
			{
				std::snprintf(
					cell, sizeof(cell),
					" | %*.2f", TimingColumnWidth(column.title),
					Bench::ConvertFromNs(
						unit, AverageNs(report, column.op)));
				line += cell;
			}

			std::snprintf(cell, sizeof(cell), " %4s", unitLabel);
			line += cell;
			lines.emplace_back(line);
		}

		return lines;
	}

} // namespace Dist
