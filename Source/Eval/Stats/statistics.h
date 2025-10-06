// statistics.h
//
// TODO: comment

#pragma once

// external
#include <array>
#include <cstdint>
#include <deque>

// internal
#include "Source/Sim/Core/parameters_sim.h"
#include "metrics_stat.h"



namespace Stats
{
	class StatisticsMgr
	{
	public: // ==============================

		StatisticsMgr(const Sim::SimParams& simParams);

		// gives back a reference to the writable current record
		StatRecordStep& CurrentRecord() { return recLatest_; }

		// resets the latest stat data before pulling in a new record
		void BeginStep();

		// pushes the record to history and aggregates step data
		void EndStep(int64_t simStep, float simTime);

		// stat queries
		float GetStat_Latest(StatMetric metric) const;
		float GetStat_Average(StatMetric metric) const;
		float GetStat_Min(StatMetric metric) const;
		float GetStat_Max(StatMetric metric) const;

	private: // ==============================

		// max and current number of steps for which stats are kept
		int64_t maxRecCount_;
		int64_t curRecCount_;

		StatRecordStep recLatest_;

		// rolling buffer for the stats for the last maxCount_ steps
		std::deque<StatRecordStep> recHistory_;

		// all-time aggregates
		std::array<float, GetIdx(StatMetric::Count)> statsSum_allTime_;
		std::array<float, GetIdx(StatMetric::Count)> statsMin_allTime_;
		std::array<float, GetIdx(StatMetric::Count)> statsMax_allTime_;
	};

} // namespace Stats
