// dist_statistics.h

#pragma once

// internal
#include "source/eval/stats/statistics.h"
#include "source/simulation/core/parameters_sim.h"



namespace Dist
{
	class MpiContext;

	// reduces rank-local partial stats data and fills the regular (global) stat record
	void FinalizeDistributedStatisticsRecord(
		Stats::StatRecordStep& record,
		Stats::StatisticsPartial& partial,
		const Sim::SimParams& simParams,
		const MpiContext& mpiContext);

} // namespace Dist
