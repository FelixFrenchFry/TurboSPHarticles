// decomp_aabb_uniform.cpp

// external
#include <cstdint>
#include <stdexcept>
#include <vector>

// internal
#include "decomp_aabb_uniform.h"



using namespace Dist;

namespace
{
	std::vector<float> MakeUniformSplitPositions(
		const DomainDecompInput& input,
		const int splitAxis)
	{
		const float splitMin = input.globalBounds.min[splitAxis];
		const float splitMax = input.globalBounds.max[splitAxis];
		const float splitLen = splitMax - splitMin;
		const float slabLen = splitLen / static_cast<float>(input.N_ranks);

		std::vector<float> splitPositions{};
		splitPositions.resize(static_cast<uint64_t>(input.N_ranks) + 1);

		for (int rank = 0; rank < input.N_ranks; rank++)
		{
			splitPositions[rank] = splitMin + static_cast<float>(rank) * slabLen;
		}
		splitPositions[input.N_ranks] = splitMax;

		return splitPositions;
	}

} // namespace anonymous

void Decomp_AABB_Uniform::Build(const DomainDecompInput& input)
{
	if (!input.globalBounds.IsValid())
	{
		throw std::runtime_error("Cannot build Decomp_AABB_Uniform with invalid global domain bounds.");
	}
	if (input.haloRadius < 0.0f)
	{
		throw std::runtime_error("Cannot build Decomp_AABB_Uniform with negative halo radius.");
	}
	if (input.N_ranks <= 0)
	{
		throw std::runtime_error("Cannot build Decomp_AABB_Uniform for non-positive rank count.");
	}

	// one uniform AABB slab per rank, split along the global X axis
	slabLayout_.Build(
		input.globalBounds,
		input.haloRadius,
		input.N_ranks,
		SPLIT_AXIS,
		MakeUniformSplitPositions(input, SPLIT_AXIS));
}

const AABB& Decomp_AABB_Uniform::GlobalExtentAABB() const
{
	return slabLayout_.GlobalExtentAABB();
}

int Decomp_AABB_Uniform::OwnerRank(const glm::vec3& position) const
{
	return slabLayout_.OwnerRank(position);
}

bool Decomp_AABB_Uniform::IsInnerOwnedParticle(
	const int rank,
	const glm::vec3& position,
	const float kernelRadius) const
{
	return slabLayout_.IsInnerOwnedParticle(rank, position, kernelRadius);
}

const std::vector<int>& Decomp_AABB_Uniform::NeighborRanks(const int rank) const
{
	return slabLayout_.NeighborRanks(rank);
}

AABB Decomp_AABB_Uniform::OwnedExtentAABB(const int rank) const
{
	return slabLayout_.OwnedExtentAABB(rank);
}

AABB Decomp_AABB_Uniform::HaloExtentAABB(const int rank) const
{
	return slabLayout_.HaloExtentAABB(rank);
}

bool Decomp_AABB_Uniform::IsValidForRankCount(const int N_ranks) const
{
	return slabLayout_.IsValidForRankCount(N_ranks);
}
