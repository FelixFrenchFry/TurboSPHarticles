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
#include "Source/Sim/Sync/snapshots.h"
#include "metrics.h"



namespace Stats
{
	class StatisticsMgr
	{
	public: // ==============================

		StatisticsMgr(const Sim::SimParams& simParams);

		// resets the latest stat data before pulling in a new record
		void BeginStep();

		// pushes the record to history and aggregates step's stat data
		void EndStep(int64_t simStep, double simTime);

		void PackStatSnapshot(Sync::StatSnapshot& snap) const;

		// per-metric queries
		float GetStat_Latest(StatMetric metric) const;
		float GetStat_Avg(StatMetric metric) const;
		float GetStat_Min(StatMetric metric) const;
		float GetStat_Max(StatMetric metric) const;

		// helpers
		StatRecordStep& GetCurrentRecord() { return recLatest_; }
		const StatRecordStep& GetCurrentRecord() const { return recLatest_; }
		const std::deque<StatRecordStep>& GetHistory() const { return recHistory_; }

	private: // ==============================

		// max and current number of steps for which stats are kept
		int64_t maxRecCount_;
		int64_t curRecCount_;

		StatRecordStep recLatest_;

		// rolling buffer for the stats for the last maxCount_ steps
		std::deque<StatRecordStep> recHistory_;

		// all-time per-metric aggregates
		std::array<float, GetIdx(StatMetric::Count)> statsSum_allTime_{};
		std::array<float, GetIdx(StatMetric::Count)> statsMin_allTime_{};
		std::array<float, GetIdx(StatMetric::Count)> statsMax_allTime_{};

		// helpers
		void ClearAllTime();
	};

} // namespace Stats
