// decomp_aabb_dynamic.cpp

// external
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
	const std::vector<float> splitPositions(
		input.partitionPositions.begin(),
		input.partitionPositions.end());

	slabLayout_.Rebuild(splitPositions);
}
