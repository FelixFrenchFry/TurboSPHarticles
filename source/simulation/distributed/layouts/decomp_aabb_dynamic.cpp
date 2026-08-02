// decomp_aabb_dynamic.cpp

// external
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// internal
#include "decomp_aabb_dynamic.h"



using namespace Dist;

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

void Decomp_AABB_Dynamic::Repartition(const DomainDecompRepartitionInput& input)
{
	// validate rank loads against the current slab layout
	const int N_ranks = static_cast<int>(input.loadByRank.size());
	if (!slabLayout_.IsValidForRankCount(N_ranks))
	{
		throw std::runtime_error(
			"Cannot repartition Decomp_AABB_Dynamic with a rank load count that "
			"does not match the current layout.");
	}

	// validate and sum rank loads
	double totalLoad = 0.0;
	for (const double load : input.loadByRank)
	{
		if (!std::isfinite(load) || load < 0.0)
		{
			throw std::runtime_error(
				"Cannot repartition Decomp_AABB_Dynamic with invalid rank loads.");
		}
		totalLoad += load;
	}
	if (!std::isfinite(totalLoad))
	{
		throw std::runtime_error(
			"Cannot repartition Decomp_AABB_Dynamic with an invalid total rank load.");
	}
	if (totalLoad == 0.0) { return; }

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

	// place interior walls at equal cumulative-load targets
	int loadRank = 0;
	double cumulativeLoad = 0.0;
	for (int splitIdx = 1; splitIdx < N_ranks; splitIdx++)
	{
		const double targetLoad = totalLoad *
			static_cast<double>(splitIdx) / static_cast<double>(N_ranks);

		// advance to the slab containing this target
		while (loadRank < N_ranks &&
			(input.loadByRank[loadRank] == 0.0 ||
			 cumulativeLoad + input.loadByRank[loadRank] < targetLoad))
		{
			cumulativeLoad += input.loadByRank[loadRank];
			loadRank += 1;
		}
		if (loadRank >= N_ranks)
		{
			throw std::runtime_error(
				"Cannot derive dynamic AABB split positions from the supplied rank loads.");
		}

		// interpolate wall position within the containing slab
		const double rankLoad = input.loadByRank[loadRank];
		const double loadFraction = std::clamp(
			(targetLoad - cumulativeLoad) / rankLoad,
			0.0,
			1.0);
		const double splitMin = oldSplitPositions[loadRank];
		const double splitMax = oldSplitPositions[loadRank + 1];
		newSplitPositions[splitIdx] = static_cast<float>(
				splitMin + loadFraction * (splitMax - splitMin));
	}

	// rebuild owned and halo extents from new walls
	slabLayout_.Rebuild(newSplitPositions);
}
