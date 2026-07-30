// dist_statistics.cpp

// external
#include <algorithm>
#include <cstdint>
#include <limits>

// internal
#include "dist_statistics.h"
#include "mpi_context.h"



using namespace Dist;

namespace
{
	// reduces one neighbor-count distribution across ranks
	void ReduceNbrCountPartial(
		Stats::DistributionPartial& distribution,
		const MpiContext& mpiContext)
	{
		const uint64_t N_local = distribution.N;
		distribution.N = mpiContext.AllReduceSumUint64(N_local);
		distribution.sum = mpiContext.AllReduceSumDouble(distribution.sum);

		const uint32_t maxLocal = N_local > 0 ? static_cast<uint32_t>(distribution.max) : 0;
		const uint32_t maxGlobal = mpiContext.AllReduceMaxUint32(maxLocal);

		if (distribution.N == 0)
		{
			distribution.min = 0.0f;
			distribution.max = 0.0f;
			distribution.histogram.clear();
			return;
		}

		distribution.histogram.resize(static_cast<uint64_t>(maxGlobal) + 1, 0);
		mpiContext.AllReduceSumUint32s(distribution.histogram);

		const auto minIt = std::find_if(
			distribution.histogram.begin(),
			distribution.histogram.end(),
			[](const uint32_t N_nbrs) { return N_nbrs > 0; });

		distribution.min = static_cast<float>(std::distance(distribution.histogram.begin(), minIt));
		distribution.max = static_cast<float>(maxGlobal);
	}

	// reduces one scalar distribution across ranks
	void ReduceDistributionPartial(
		Stats::DistributionPartial& distribution,
		const MpiContext& mpiContext)
	{
		const uint64_t N_local = distribution.N;
		const float minLocal = N_local > 0 ? distribution.min : std::numeric_limits<float>::max();

		distribution.N = mpiContext.AllReduceSumUint64(N_local);
		distribution.sum = mpiContext.AllReduceSumDouble(distribution.sum);
		distribution.min = mpiContext.AllReduceMinFloat(minLocal);
		distribution.max = mpiContext.AllReduceMaxFloat(distribution.max);

		if (distribution.N == 0)
		{
			distribution.min = 0.0f;
			distribution.max = 0.0f;
		}
	}

} // namespace anonymous

void Dist::FinalizeDistributedStatisticsRecord(
	Stats::StatRecordStep& record,
	Stats::StatisticsPartial& partial,
	const Sim::SimParams& simParams,
	const MpiContext& mpiContext)
{
	using namespace Stats;

	// --- [ particle counts ] -----------------------------------------------------------------------

	partial.N_F = mpiContext.AllReduceSumUint64(partial.N_F);
	partial.N_B = mpiContext.AllReduceSumUint64(partial.N_B);
	partial.N_R = mpiContext.AllReduceSumUint64(partial.N_R);

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Particles)])
	{
		// neighbor count distributions
		ReduceNbrCountPartial(partial.nbrCountFF, mpiContext);
		ReduceNbrCountPartial(partial.nbrCountFB, mpiContext);
		ReduceNbrCountPartial(partial.nbrCountFR, mpiContext);
		ReduceNbrCountPartial(partial.nbrCountRF, mpiContext);
		ReduceNbrCountPartial(partial.nbrCountRB, mpiContext);
		ReduceNbrCountPartial(partial.nbrCountRR, mpiContext);
	}

	// --- [ density stats ] -------------------------------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Density)])
	{
		partial.densRelErrorRawSum = mpiContext.AllReduceSumDouble(partial.densRelErrorRawSum);
		partial.densRelErrorPosSum = mpiContext.AllReduceSumDouble(partial.densRelErrorPosSum);
		partial.densRelErrorAbsSum = mpiContext.AllReduceSumDouble(partial.densRelErrorAbsSum);
	}

	// --- [ solver stats ] --------------------------------------------------------------------------

	partial.N_solverIters = mpiContext.AllReduceMaxUint32(partial.N_solverIters);

	// --- [ velocity stats ] ------------------------------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Velocity)])
	{
		ReduceDistributionPartial(partial.velMag, mpiContext);
	}

	// finalize regular stat record
	StatisticsTracker::FinalizeStatisticsRecord(record, partial, simParams);
}
