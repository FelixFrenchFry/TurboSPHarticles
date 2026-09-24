// dist_statistics.cpp

// external
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

// internal
#include "dist_statistics.h"
#include "mpi_context.h"



using namespace Dist;

namespace
{
	// reduces the histogram after scalar statistics have been combined across ranks
	void ReduceNbrCountHistogram(
		Stats::DistributionPartial& distribution,
		const uint32_t maxGlobal,
		const MpiContext& mpiContext)
	{
		if (distribution.N == 0)
		{
			distribution.min = 0.0f;
			distribution.max = 0.0f;
			distribution.histogram.clear();
			return;
		}

		distribution.histogram.resize(static_cast<uint64_t>(maxGlobal) + 1, 0);
		mpiContext.AllReduceSumUint64s(distribution.histogram);

		const auto minIt = std::find_if(
			distribution.histogram.begin(),
			distribution.histogram.end(),
			[](const uint64_t N_nbrs) { return N_nbrs > 0; });

		distribution.min = static_cast<float>(std::distance(distribution.histogram.begin(), minIt));
		distribution.max = static_cast<float>(maxGlobal);
	}

} // namespace anonymous

void Dist::FinalizeDistributedStatisticsRecord(
	Stats::StatRecordStep& record,
	Stats::StatisticsPartial& partial,
	const Sim::SimParams& simParams,
	const MpiContext& mpiContext)
{
	using namespace Stats;

	const bool collectParticles = simParams.statGroupsEnabled[GetIdx(StatGroup::Particles)];
	const bool collectDensity = simParams.statGroupsEnabled[GetIdx(StatGroup::Density)];
	const bool collectVelocity = simParams.statGroupsEnabled[GetIdx(StatGroup::Velocity)];
	const std::array neighborDistributions{
		&partial.nbrCountFF, &partial.nbrCountFB, &partial.nbrCountFR,
		&partial.nbrCountRF, &partial.nbrCountRB, &partial.nbrCountRR
	};
	constexpr uint32_t N_neighborDistributions = static_cast<uint32_t>(neighborDistributions.size());

	// counts: F/B/R, neighbor distributions, velocity
	// sums: neighbor distributions, raw/positive/absolute density errors, velocity
	// maxima: solver iterations, neighbor distributions
	std::array<uint64_t, 3 + N_neighborDistributions + 1> counts{
		partial.N_F, partial.N_B, partial.N_R
	};
	std::array<double, N_neighborDistributions + 3 + 1> sums{};
	std::array<uint32_t, 1 + N_neighborDistributions> maxima{ partial.N_solverIters };
	if (collectParticles)
	{
		for (uint32_t idx = 0; idx != N_neighborDistributions; idx++)
		{
			const DistributionPartial& distribution = *neighborDistributions[idx];
			counts[3 + idx] = distribution.N;
			sums[idx] = distribution.sum;
			maxima[1 + idx] = distribution.N > 0 ? static_cast<uint32_t>(distribution.max) : 0;
		}
	}
	if (collectDensity)
	{
		sums[N_neighborDistributions] = partial.densRelErrorRawSum;
		sums[N_neighborDistributions + 1] = partial.densRelErrorPosSum;
		sums[N_neighborDistributions + 2] = partial.densRelErrorAbsSum;
	}
	if (collectVelocity)
	{
		counts.back() = partial.velMag.N;
		sums.back() = partial.velMag.sum;
	}

	mpiContext.AllReduceSumUint64s(counts);
	if (collectParticles || collectDensity || collectVelocity)
	{
		mpiContext.AllReduceSumDoubles(sums);
	}
	mpiContext.AllReduceMaxUint32s(maxima);

	partial.N_F = counts[0];
	partial.N_B = counts[1];
	partial.N_R = counts[2];
	partial.N_solverIters = maxima[0];
	if (collectParticles)
	{
		for (uint32_t idx = 0; idx != N_neighborDistributions; idx++)
		{
			DistributionPartial& distribution = *neighborDistributions[idx];
			distribution.N = counts[3 + idx];
			distribution.sum = sums[idx];
			ReduceNbrCountHistogram(distribution, maxima[1 + idx], mpiContext);
		}
	}
	if (collectDensity)
	{
		partial.densRelErrorRawSum = sums[N_neighborDistributions];
		partial.densRelErrorPosSum = sums[N_neighborDistributions + 1];
		partial.densRelErrorAbsSum = sums[N_neighborDistributions + 2];
	}
	if (collectVelocity)
	{
		const float minLocal = partial.velMag.N > 0
			? partial.velMag.min : std::numeric_limits<float>::max();
		partial.velMag.N = counts.back();
		partial.velMag.sum = sums.back();
		partial.velMag.min = mpiContext.AllReduceMinFloat(minLocal);
		partial.velMag.max = mpiContext.AllReduceMaxFloat(partial.velMag.max);
		if (partial.velMag.N == 0)
		{
			partial.velMag.min = 0.0f;
			partial.velMag.max = 0.0f;
		}
	}

	// finalize regular stat record
	StatisticsTracker::FinalizeStatisticsRecord(record, partial, simParams);
}
