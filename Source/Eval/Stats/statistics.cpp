// statistics.cpp

// external
#include <algorithm>
#include <limits>

// internal
#include "statistics.h"



using namespace Stats;

StatisticsMgr::StatisticsMgr(const Sim::SimParams& simParams)
	: maxRecCount_(simParams.maxStatRecCount)
	, curRecCount_(0)
{
	recLatest_ = {};
	recLatest_.stats.fill(0.0f);
	ClearAllTime();
}

void StatisticsMgr::BeginStep()
{
	recLatest_ = {};
	recLatest_.stats.fill(0.0f);
}

void StatisticsMgr::EndStep(int64_t simStep, double simTime)
{
	// fill latest record
	recLatest_.simStep = simStep;
	recLatest_.simTime = simTime;

	// push latest record to history and check excess
	recHistory_.push_back(recLatest_);
	if (recHistory_.size() > maxRecCount_)
	{
		recHistory_.pop_front();
	}

	if (curRecCount_ == 0) // first entry
	{
		statsSum_allTime_ = recLatest_.stats;
		statsMin_allTime_ = recLatest_.stats;
		statsMax_allTime_ = recLatest_.stats;
	}
	else
	{
		for (uint16_t i = 0; i < GetIdx(StatMetric::Count); i++)
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

	curRecCount_ += 1;
}

void StatisticsMgr::PackStatSnapshot(Sync::StatSnapshot& snap) const
{
	// metadata
	snap.simStep = recLatest_.simStep;
	snap.simTime = recLatest_.simTime;

	// fill in the latest, average, min, max stat data
	for (uint16_t i = 0; i < GetIdx(StatMetric::Count); i++)
	{
		snap.latest[i] = GetStat_Latest(static_cast<StatMetric>(i));
		snap.avg[i] = GetStat_Avg(static_cast<StatMetric>(i));
		snap.min[i] = GetStat_Min(static_cast<StatMetric>(i));
		snap.max[i] = GetStat_Max(static_cast<StatMetric>(i));
	}
}

float StatisticsMgr::GetStat_Latest(StatMetric metric) const
{
	if (curRecCount_ == 0) { return 0.0f; }

	return recLatest_.stats[GetIdx(metric)];
}

float StatisticsMgr::GetStat_Avg(StatMetric metric) const
{
	if (curRecCount_ == 0) { return 0.0f; }

	return statsSum_allTime_[GetIdx(metric)] / static_cast<float>(curRecCount_);
}

float StatisticsMgr::GetStat_Min(StatMetric metric) const
{
	if (curRecCount_ == 0) { return 0.0f; }

	return statsMin_allTime_[GetIdx(metric)];
}

float StatisticsMgr::GetStat_Max(StatMetric metric) const
{
	if (curRecCount_ == 0) { return 0.0f; }

	return statsMax_allTime_[GetIdx(metric)];
}

void StatisticsMgr::ClearAllTime()
{
	statsSum_allTime_.fill(0.0f);
	statsMin_allTime_.fill(std::numeric_limits<float>::max());
	statsMax_allTime_.fill(0.0f);
}
