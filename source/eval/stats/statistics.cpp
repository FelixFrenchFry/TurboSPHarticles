// statistics.cpp

// external
#include <algorithm>
#include <limits>

// internal
#include "statistics.h"



using namespace Stats;

namespace
{
	double ComputeAverage(const double sum, const uint64_t N)
	{
		if (N == 0) { return 0.0; }

		return sum / static_cast<double>(N);
	}

	double ComputeP99(const DistributionPartial& partial)
	{
		if (partial.N == 0) { return 0.0; }

		const uint64_t i99 = (99ull * (partial.N - 1)) / 100ull;
		uint64_t N_cumulative = 0;
		for (uint64_t value = 0; value < partial.histogram.size(); value++)
		{
			N_cumulative += partial.histogram[value];
			if (N_cumulative > i99)
			{
				return static_cast<double>(value);
			}
		}

		return 0.0;
	}

} // namespace anonymous

void DistributionPartial::Clear()
{
	N = 0;
	sum = 0.0;
	min = 0.0f;
	max = 0.0f;
	histogram.clear();
}

void StatisticsPartial::Clear()
{
	N_F = 0;
	N_B = 0;
	N_R = 0;

	nbrCountFF.Clear();
	nbrCountFB.Clear();
	nbrCountFR.Clear();
	nbrCountRF.Clear();
	nbrCountRB.Clear();
	nbrCountRR.Clear();
	velMag.Clear();

	N_solverIters = 0;

	densRelErrorRawSum = 0.0;
	densRelErrorPosSum = 0.0;
	densRelErrorAbsSum = 0.0;
}

StatisticsTracker::StatisticsTracker(const Sim::SimParams& simParams)
	: N_maxRecords_(simParams.maxStatRecs)
	, N_records_(0u)
{
	recLatest_ = {};
	std::ranges::fill(recLatest_.stats, 0.0);
	ClearAllTime();
}

void StatisticsTracker::BeginStep()
{
	recLatest_ = {};
	std::ranges::fill(recLatest_.stats, 0.0);
}

void StatisticsTracker::EndStep(uint64_t simStep, double simTime)
{
	// fill latest record
	recLatest_.simStep = simStep;
	recLatest_.simTime = simTime;

	// push latest record to history and check excess
	recHistory_.push_back(recLatest_);
	if (recHistory_.size() > N_maxRecords_)
	{
		recHistory_.pop_front();
	}

	if (N_records_ == 0) // first entry
	{
		statsSum_allTime_ = recLatest_.stats;
		statsMin_allTime_ = recLatest_.stats;
		statsMax_allTime_ = recLatest_.stats;
	}
	else
	{
		for (uint8_t i = 0; i < GetIdx(StatMetric::Count); i++)
		{
			// update all-time sum
			statsSum_allTime_[i] += recLatest_.stats[i];

			// update all-time min
			statsMin_allTime_[i] = std::min(recLatest_.stats[i],
										    statsMin_allTime_[i]);

			// update all-time max
			statsMax_allTime_[i] = std::max(recLatest_.stats[i],
										    statsMax_allTime_[i]);
		}
	}

	N_records_ += 1;
}

void StatisticsTracker::PackStatSnapshot(Sync::StatSnapshot& snap) const
{
	// metadata
	snap.simStep = recLatest_.simStep;
	snap.simTime = recLatest_.simTime;

	// fill in the latest, average, min, max stat data
	for (uint8_t i = 0; i < GetIdx(StatMetric::Count); i++)
	{
		snap.latest[i] = GetStat_Latest(static_cast<StatMetric>(i));
		snap.avg[i] = GetStat_Avg(static_cast<StatMetric>(i));
		snap.min[i] = GetStat_Min(static_cast<StatMetric>(i));
		snap.max[i] = GetStat_Max(static_cast<StatMetric>(i));
	}
}

void StatisticsTracker::FinalizeStatisticsRecord(
	StatRecordStep& record,
	const StatisticsPartial& partial,
	const Sim::SimParams& simParams)
{
	record = {};

	Set(record, StatMetric::NumF, static_cast<double>(partial.N_F));
	Set(record, StatMetric::NumB, static_cast<double>(partial.N_B));
	Set(record, StatMetric::NumR, static_cast<double>(partial.N_R));

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Particles)])
	{
		const auto setDistributionStats = [&](
			const DistributionPartial& distribution,
			const StatMetric avgMetric,
			const StatMetric minMetric,
			const StatMetric maxMetric,
			const StatMetric p99Metric)
			{
				Set(record, avgMetric, ComputeAverage(distribution.sum, distribution.N));
				Set(record, minMetric, distribution.min);
				Set(record, maxMetric, distribution.max);
				Set(record, p99Metric, ComputeP99(distribution));
			};

		setDistributionStats(
			partial.nbrCountFF,
			StatMetric::NumNbrFF_Avg,
			StatMetric::NumNbrFF_Min,
			StatMetric::NumNbrFF_Max,
			StatMetric::NumNbrFF_P99);
		setDistributionStats(
			partial.nbrCountFB,
			StatMetric::NumNbrFB_Avg,
			StatMetric::NumNbrFB_Min,
			StatMetric::NumNbrFB_Max,
			StatMetric::NumNbrFB_P99);
		setDistributionStats(
			partial.nbrCountFR,
			StatMetric::NumNbrFR_Avg,
			StatMetric::NumNbrFR_Min,
			StatMetric::NumNbrFR_Max,
			StatMetric::NumNbrFR_P99);
		setDistributionStats(
			partial.nbrCountRF,
			StatMetric::NumNbrRF_Avg,
			StatMetric::NumNbrRF_Min,
			StatMetric::NumNbrRF_Max,
			StatMetric::NumNbrRF_P99);
		setDistributionStats(
			partial.nbrCountRB,
			StatMetric::NumNbrRB_Avg,
			StatMetric::NumNbrRB_Min,
			StatMetric::NumNbrRB_Max,
			StatMetric::NumNbrRB_P99);
		setDistributionStats(
			partial.nbrCountRR,
			StatMetric::NumNbrRR_Avg,
			StatMetric::NumNbrRR_Min,
			StatMetric::NumNbrRR_Max,
			StatMetric::NumNbrRR_P99);
	}

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Density)])
	{
		Set(record, StatMetric::DensRelError_Raw_Avg,
			ComputeAverage(partial.densRelErrorRawSum, partial.N_F));
		Set(record, StatMetric::DensRelError_Pos_Avg,
			ComputeAverage(partial.densRelErrorPosSum, partial.N_F));
		Set(record, StatMetric::DensRelError_Abs_Avg,
			ComputeAverage(partial.densRelErrorAbsSum, partial.N_F));
	}

	Set(record, StatMetric::NumSolverIters, static_cast<double>(partial.N_solverIters));

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Velocity)])
	{
		Set(record, StatMetric::CFL_Number,
			partial.velMag.max * simParams.timeStep / simParams.particleSizeF);
		Set(record, StatMetric::VelMag_Avg,
			ComputeAverage(partial.velMag.sum, partial.velMag.N));
		Set(record, StatMetric::VelMag_Min, partial.velMag.min);
		Set(record, StatMetric::VelMag_Max, partial.velMag.max);
	}
}

double StatisticsTracker::GetStat_Latest(StatMetric metric) const
{
	if (N_records_ == 0) { return 0.0; }

	return recLatest_.stats[GetIdx(metric)];
}

double StatisticsTracker::GetStat_Avg(StatMetric metric) const
{
	if (N_records_ == 0) { return 0.0; }

	return statsSum_allTime_[GetIdx(metric)] / static_cast<double>(N_records_);
}

double StatisticsTracker::GetStat_Min(StatMetric metric) const
{
	if (N_records_ == 0) { return 0.0; }

	return statsMin_allTime_[GetIdx(metric)];
}

double StatisticsTracker::GetStat_Max(StatMetric metric) const
{
	if (N_records_ == 0) { return 0.0; }

	return statsMax_allTime_[GetIdx(metric)];
}

void StatisticsTracker::ClearAllTime()
{
	std::ranges::fill(statsSum_allTime_, 0.0);
	std::ranges::fill(statsMin_allTime_, std::numeric_limits<double>::max());
	std::ranges::fill(statsMax_allTime_, 0.0);
}
