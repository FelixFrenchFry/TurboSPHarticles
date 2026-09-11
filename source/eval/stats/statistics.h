// statistics.h
#pragma once

// external
#include <array>
#include <cstdint>
#include <deque>
#include <vector>

// internal
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/sync/snapshots.h"
#include "metrics.h"



namespace Stats
{
	// rank-local partial stat data for a single metric
	struct DistributionPartial
	{
		uint64_t N = 0;
		double sum = 0.0;
		float min = 0.0f;
		float max = 0.0f;
		std::vector<uint32_t> histogram;

		void Clear();
	};

	// rank-local partial stat data used to gather distributed statistics
	struct StatisticsPartial
	{
		uint64_t N_F = 0;
		uint64_t N_B = 0;
		uint64_t N_R = 0;

		DistributionPartial nbrCountFF;
		DistributionPartial nbrCountFB;
		DistributionPartial nbrCountFR;
		DistributionPartial nbrCountRF;
		DistributionPartial nbrCountRB;
		DistributionPartial nbrCountRR;
		DistributionPartial velMag;

		uint32_t N_solverIters = 0;
		double densRelErrorRawSum = 0.0;
		double densRelErrorPosSum = 0.0;
		double densRelErrorAbsSum = 0.0;

		void Clear();
	};

	class StatisticsTracker
	{
	public:

		StatisticsTracker(const Sim::SimParams& simParams);

		// resets the latest stat data before pulling in a new record
		void BeginStep();

		// pushes the record to history and aggregates step's stat data
		void EndStep(uint64_t simStep, double simTime);

		void PackStatSnapshot(Sync::StatSnapshot& snap) const;

		// finalizes the regular stat record from partial stat data
		static void FinalizeStatisticsRecord(
			StatRecordStep& record,
			const StatisticsPartial& partial,
			const Sim::SimParams& simParams);

		// per-metric queries
		float GetStat_Latest(StatMetric metric) const;
		float GetStat_Avg(StatMetric metric) const;
		float GetStat_Min(StatMetric metric) const;
		float GetStat_Max(StatMetric metric) const;

		// helpers
		StatRecordStep& GetCurrentRecord() { return recLatest_; }
		const StatRecordStep& GetCurrentRecord() const { return recLatest_; }
		const std::deque<StatRecordStep>& GetHistory() const { return recHistory_; }

	private:

		// max and current number of steps for which stat data is kept
		uint32_t N_maxRecords_;
		uint32_t N_records_;

		StatRecordStep recLatest_;

		// rolling buffer for the stat data for the last N_maxRecords_ steps
		std::deque<StatRecordStep> recHistory_;

		// all-time per-metric aggregates
		std::array<float, GetIdx(StatMetric::Count)> statsSum_allTime_{};
		std::array<float, GetIdx(StatMetric::Count)> statsMin_allTime_{};
		std::array<float, GetIdx(StatMetric::Count)> statsMax_allTime_{};

		// helpers
		void ClearAllTime();
	};

} // namespace Stats
