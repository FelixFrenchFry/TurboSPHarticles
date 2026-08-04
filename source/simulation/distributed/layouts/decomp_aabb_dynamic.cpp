// decomp_aabb_dynamic.cpp

// external
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// internal
#include "decomp_aabb_dynamic.h"



using namespace Dist;

namespace
{
	void AccumulateOwnedParticleCounts(
		const DomainDecompParticlePositionView& particles,
		const AABB& globalBounds,
		const uint32_t profileBinCount,
		std::vector<uint32_t>& profile,
		const char* particleType)
	{
		const float splitMin = globalBounds.min.x;
		const float splitMax = globalBounds.max.x;
		const float splitLength = splitMax - splitMin;
		if (!std::isfinite(splitLength) || splitLength <= 0.0f)
		{
			throw std::runtime_error(
				"Cannot build a dynamic AABB repartition profile from an invalid global extent.");
		}

		for (const uint32_t localIdx : particles.ownedLocalIdx)
		{
			if (localIdx >= particles.x.size())
			{
				throw std::runtime_error(
					std::string("Cannot build a dynamic AABB repartition profile with an invalid ") +
					particleType + " particle index.");
			}

			const float splitCoordinate = particles.x[localIdx];
			if (!std::isfinite(splitCoordinate))
			{
				throw std::runtime_error(
					std::string("Cannot build a dynamic AABB repartition profile from an invalid ") +
					particleType + " particle position.");
			}

			// particles that left the global domain count towards the nearest bin
			const double normalizedCoordinate = std::clamp(
				static_cast<double>(splitCoordinate - splitMin) /
				static_cast<double>(splitLength),
				0.0, 1.0);
			const uint32_t bin = std::min(
				static_cast<uint32_t>(normalizedCoordinate * profileBinCount),
				profileBinCount - 1);
			uint32_t& binCount = profile[bin];
			if (binCount == std::numeric_limits<uint32_t>::max())
			{
				throw std::runtime_error(
					"Cannot build a dynamic AABB repartition profile with an overflowing bin.");
			}
			binCount += 1;
		}
	}

} // namespace anonymous

void Decomp_AABB_Dynamic::Build(const DomainDecompInput& input)
{
	if (!input.globalBounds.IsValid())
	{
		throw std::runtime_error("Cannot build Decomp_AABB_Dynamic with invalid global domain bounds.");
	}
	if (input.haloRadius < 0.0f)
	{
		throw std::runtime_error("Cannot build Decomp_AABB_Dynamic with negative halo radius.");
	}
	if (input.N_ranks <= 0)
	{
		throw std::runtime_error("Cannot build Decomp_AABB_Dynamic for non-positive rank count.");
	}

	// initially built with one uniform AABB slab per rank, split along the global X axis
	slabLayout_.Build(
		input.globalBounds,
		input.haloRadius,
		input.N_ranks,
		SPLIT_AXIS,
		MakeUniformSplitPositions(input.globalBounds, input.N_ranks, SPLIT_AXIS));
}

const AABB& Decomp_AABB_Dynamic::GlobalExtentAABB() const
{
	return slabLayout_.GlobalExtentAABB();
}

int Decomp_AABB_Dynamic::OwnerRank(const glm::vec3& position) const
{
	return slabLayout_.OwnerRank(position);
}

bool Decomp_AABB_Dynamic::IsInnerOwnedParticle(
	const int rank,
	const glm::vec3& position,
	const float kernelRadius) const
{
	return slabLayout_.IsInnerOwnedParticle(rank, position, kernelRadius);
}

const std::vector<int>& Decomp_AABB_Dynamic::NeighborRanks(const int rank) const
{
	return slabLayout_.NeighborRanks(rank);
}

AABB Decomp_AABB_Dynamic::OwnedExtentAABB(const int rank) const
{
	return slabLayout_.OwnedExtentAABB(rank);
}

AABB Decomp_AABB_Dynamic::HaloExtentAABB(const int rank) const
{
	return slabLayout_.HaloExtentAABB(rank);
}

bool Decomp_AABB_Dynamic::IsValidForRankCount(const int N_ranks) const
{
	return slabLayout_.IsValidForRankCount(N_ranks);
}

bool Decomp_AABB_Dynamic::SupportsRepartition(
	const Base::LoadBalanceWeightModel weightModel) const
{
	return weightModel == Base::LoadBalanceWeightModel::OwnedParticleCount;
}

void Decomp_AABB_Dynamic::BuildRepartitionProfile(
	const DomainDecompRepartitionParticleInput& input,
	DomainDecompRepartitionProfile& profile) const
{
	if (input.weightModel != Base::LoadBalanceWeightModel::OwnedParticleCount)
	{
		throw std::runtime_error(
			"Decomp_AABB_Dynamic does not support the requested repartition weight model.");
	}

	const int N_ranks = slabLayout_.RankCount();
	if (input.rank < 0 || input.rank >= N_ranks)
	{
		throw std::runtime_error(
			"Cannot build a dynamic AABB repartition profile for an invalid rank.");
	}

	// one fixed-resolution X histogram is kept over the global domain
	profile.ownedFluidByUnit.assign(PROFILE_BIN_COUNT, 0);
	profile.ownedBoundaryByUnit.assign(PROFILE_BIN_COUNT, 0);

	const AABB globalBounds = slabLayout_.GlobalExtentAABB();
	AccumulateOwnedParticleCounts(
		input.fluid,
		globalBounds,
		PROFILE_BIN_COUNT,
		profile.ownedFluidByUnit,
		"fluid");
	AccumulateOwnedParticleCounts(
		input.boundary,
		globalBounds,
		PROFILE_BIN_COUNT,
		profile.ownedBoundaryByUnit,
		"boundary");
}

void Decomp_AABB_Dynamic::Repartition(const DomainDecompRepartitionInput& input)
{
	if (input.weightModel != Base::LoadBalanceWeightModel::OwnedParticleCount)
	{
		throw std::runtime_error(
			"Decomp_AABB_Dynamic does not support the requested repartition weight model.");
	}

	// validate rank loads against the current slab layout
	const int N_ranks = static_cast<int>(input.loadByRank.size());
	if (!slabLayout_.IsValidForRankCount(N_ranks))
	{
		throw std::runtime_error(
			"Cannot repartition Decomp_AABB_Dynamic with a rank load count that "
			"does not match the current layout.");
	}

	// validate rank loads supplied to the trigger
	for (const double load : input.loadByRank)
	{
		if (!std::isfinite(load) || load < 0.0)
		{
			throw std::runtime_error(
				"Cannot repartition Decomp_AABB_Dynamic with invalid rank loads.");
		}
	}

	if (input.ownedFluidByUnit.size() != PROFILE_BIN_COUNT ||
		input.ownedBoundaryByUnit.size() != PROFILE_BIN_COUNT)
	{
		throw std::runtime_error(
			"Cannot repartition Decomp_AABB_Dynamic with an invalid particle weight profile.");
	}

	// sum type-specific profile bins into one global owned-particle profile
	std::vector<uint64_t> weightByBin(PROFILE_BIN_COUNT, 0);
	uint64_t totalWeight = 0;
	for (uint32_t bin = 0; bin < PROFILE_BIN_COUNT; bin++)
	{
		// TODO: restore a boundary-particle weight when boundary work is modeled?
		weightByBin[bin] = static_cast<uint64_t>(input.ownedFluidByUnit[bin]);
		totalWeight += weightByBin[bin];
	}
	if (totalWeight == 0) { return; }

	// recover current walls from rank-owned extents
	std::vector<float> oldSplitPositions(static_cast<uint64_t>(N_ranks) + 1);
	oldSplitPositions.front() = slabLayout_.OwnedExtentAABB(0).min[SPLIT_AXIS];
	for (int rank = 0; rank < N_ranks; rank++)
	{
		oldSplitPositions[rank + 1] =
			slabLayout_.OwnedExtentAABB(rank).max[SPLIT_AXIS];
	}

	// preserve global walls while deriving new interior walls
	std::vector<float> newSplitPositions(static_cast<uint64_t>(N_ranks) + 1);
	newSplitPositions.front() = oldSplitPositions.front();
	newSplitPositions.back() = oldSplitPositions.back();

	// place interior walls at equal cumulative particle-count targets
	const uint64_t rankCount = static_cast<uint64_t>(N_ranks);
	const float globalMin = oldSplitPositions.front();
	const float globalMax = oldSplitPositions.back();
	const double binLength = static_cast<double>(globalMax - globalMin) /
		static_cast<double>(PROFILE_BIN_COUNT);
	uint32_t weightBin = 0;
	uint64_t cumulativeWeight = 0;
	for (int splitIdx = 1; splitIdx < N_ranks; splitIdx++)
	{
		const uint64_t targetWeight =
			(totalWeight / rankCount) * static_cast<uint64_t>(splitIdx) +
			(totalWeight % rankCount) * static_cast<uint64_t>(splitIdx) / rankCount;

		// find the histogram bin containing this cumulative target
		while (weightBin < PROFILE_BIN_COUNT &&
			cumulativeWeight + weightByBin[weightBin] < targetWeight)
		{
			cumulativeWeight += weightByBin[weightBin];
			weightBin += 1;
		}
		if (weightBin >= PROFILE_BIN_COUNT)
		{
			throw std::runtime_error(
				"Cannot derive dynamic AABB split positions from the supplied particle weights.");
		}

		const uint64_t targetWithinBin = targetWeight - cumulativeWeight;
		const double binWeight = static_cast<double>(weightByBin[weightBin]);
		const double binFraction = binWeight > 0.0
			? static_cast<double>(targetWithinBin) / binWeight
			: 0.0;
		double newWall = static_cast<double>(globalMin) +
			(static_cast<double>(weightBin) + std::clamp(binFraction, 0.0, 1.0)) *
			binLength;

		// keep walls strictly increasing when several targets share a position
		const float previousWall = newSplitPositions[splitIdx - 1];
		float newWallFloat = static_cast<float>(newWall);
		if (!(newWallFloat > previousWall))
		{
			newWallFloat = std::nextafter(previousWall, oldSplitPositions.back());
		}
		if (!(newWallFloat < oldSplitPositions.back()))
		{
			throw std::runtime_error(
				"Cannot derive strictly increasing dynamic AABB split positions.");
		}
		newSplitPositions[splitIdx] = newWallFloat;
	}

	// rebuild owned and halo extents from new walls
	slabLayout_.Rebuild(newSplitPositions);
}
