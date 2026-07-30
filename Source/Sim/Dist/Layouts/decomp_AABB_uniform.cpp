// decomp_AABB_uniform.cpp

// external
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

// internal
#include "decomp_AABB_uniform.h"



using namespace Dist;

namespace
{
	bool AABBIntersects(const AABB& a, const AABB& b)
	{
		return a.min.x <= b.max.x && a.max.x >= b.min.x
			&& a.min.y <= b.max.y && a.max.y >= b.min.y
			&& a.min.z <= b.max.z && a.max.z >= b.min.z;
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
	std::vector<AABBRankSubdomain> rankSubdomains = BuildSlabSubdomains(input, SPLIT_AXIS);

	globalBounds_ = input.globalBounds;
	rankSubdomains_ = std::move(rankSubdomains);
	haloRadius_ = input.haloRadius;

	// scales a split axis offset to a slab index
	slabLenInv_ = static_cast<float>(input.N_ranks) /
		(input.globalBounds.max[SPLIT_AXIS] - input.globalBounds.min[SPLIT_AXIS]);

	// slab lower bounds packed for the owner lookup
	slabMinSplitPos_.resize(static_cast<uint64_t>(input.N_ranks));
	for (int rank = 0; rank < input.N_ranks; rank++)
	{
		slabMinSplitPos_[rank] = rankSubdomains_[rank].ownedBounds.min[SPLIT_AXIS];
	}
}

const AABB& Decomp_AABB_Uniform::GlobalExtentAABB() const
{
	return globalBounds_;
}

int Decomp_AABB_Uniform::OwnerRank(const glm::vec3& position) const
{
	const int N_ranks = static_cast<int>(slabMinSplitPos_.size());
	if (N_ranks <= 0 || !globalBounds_.Contains(position)) { return -1; }

	// uniform slabs tile the split axis, so the offset maps straight to a slab index
	const float splitPos = position[SPLIT_AXIS];
	const float slabIdx = std::floor((splitPos - globalBounds_.min[SPLIT_AXIS]) * slabLenInv_);
	int ownerRank = std::clamp(static_cast<int>(slabIdx), 0, N_ranks - 1);

	// settle on the last slab the position reaches, so shared faces go to the higher rank
	while (ownerRank + 1 < N_ranks && splitPos >= slabMinSplitPos_[ownerRank + 1])
	{
		ownerRank++;
	}
	while (ownerRank > 0 && splitPos < slabMinSplitPos_[ownerRank])
	{
		ownerRank--;
	}

	return ownerRank;
}

bool Decomp_AABB_Uniform::IsInnerOwnedParticle(
	const int rank,
	const glm::vec3& position,
	const float kernelRadius) const
{
	// stored halo bounds already expand each subdomain by the build radius,
	// so they only answer this query for the radius used to build the decomp
	if (std::max(0.0f, kernelRadius) != haloRadius_)
	{
		throw std::runtime_error(
			"Decomp_AABB_Uniform inner particle query radius does not match the build halo radius.");
	}

	if (OwnerRank(position) != rank) { return false; }

	const AABBRankSubdomain& localSubdomain = Subdomain(rank);

	for (const int neighborRank : localSubdomain.neighborRanks)
	{
		// outer owned particles are needed by at least one neighbor rank
		if (Subdomain(neighborRank).haloBounds.Contains(position))
		{
			return false;
		}
	}

	return true;
}

const std::vector<int>& Decomp_AABB_Uniform::NeighborRanks(const int rank) const
{
	return Subdomain(rank).neighborRanks;
}

AABB Decomp_AABB_Uniform::OwnedExtentAABB(const int rank) const
{
	return Subdomain(rank).ownedBounds;
}

AABB Decomp_AABB_Uniform::HaloExtentAABB(const int rank) const
{
	return Subdomain(rank).haloBounds;
}

bool Decomp_AABB_Uniform::IsValidForRankCount(const int N_ranks) const
{
	if (N_ranks <= 0) { return false; }
	if (static_cast<int>(rankSubdomains_.size()) != N_ranks) { return false; }
	if (!globalBounds_.IsValid()) { return false; }

	for (int rank = 0; rank < N_ranks; rank++)
	{
		const AABBRankSubdomain& subdomain = rankSubdomains_[rank];

		if (subdomain.rank != rank) { return false; }
		if (!subdomain.ownedBounds.IsValid()) { return false; }
		if (!subdomain.haloBounds.IsValid()) { return false; }
		if (!globalBounds_.Contains(subdomain.ownedBounds.min)) { return false; }
		if (!globalBounds_.Contains(subdomain.ownedBounds.max)) { return false; }

		for (const int neighborRank : subdomain.neighborRanks)
		{
			if (neighborRank < 0 || neighborRank >= N_ranks) { return false; }
			if (neighborRank == rank) { return false; }

			const std::vector<int>& neighborRanks = rankSubdomains_[neighborRank].neighborRanks;
			const auto it = std::find(neighborRanks.begin(), neighborRanks.end(), rank);
			if (it == neighborRanks.end()) { return false; }
		}
	}

	return true;
}

const Decomp_AABB_Uniform::AABBRankSubdomain& Decomp_AABB_Uniform::Subdomain(
	const int rank) const
{
	if (rank < 0 || rank >= static_cast<int>(rankSubdomains_.size()))
	{
		throw std::runtime_error("Invalid Decomp_AABB_Uniform rank.");
	}

	return rankSubdomains_[rank];
}

std::vector<Decomp_AABB_Uniform::AABBRankSubdomain>
Decomp_AABB_Uniform::BuildSlabSubdomains(
	const DomainDecompInput& input,
	const int splitAxis) const
{
	std::vector<AABBRankSubdomain> subdomains{};
	subdomains.reserve(static_cast<uint64_t>(input.N_ranks));

	const float splitMin = input.globalBounds.min[splitAxis];
	const float splitMax = input.globalBounds.max[splitAxis];
	const float splitLen = splitMax - splitMin;
	const float slabLen = splitLen / static_cast<float>(input.N_ranks);

	for (int rank = 0; rank < input.N_ranks; rank++)
	{
		AABB ownedBounds = input.globalBounds;

		ownedBounds.min[splitAxis] =
			splitMin + static_cast<float>(rank) * slabLen;
		ownedBounds.max[splitAxis] = (rank == input.N_ranks - 1)
			? splitMax
			: splitMin + static_cast<float>(rank + 1) * slabLen;

		AABBRankSubdomain subdomain{};
		subdomain.rank = rank;
		subdomain.ownedBounds = ownedBounds;
		subdomain.haloBounds = ownedBounds.Expanded(input.haloRadius);

		subdomains.push_back(std::move(subdomain));
	}

	for (int rank = 0; rank < input.N_ranks; rank++)
	{
		for (int otherRank = 0; otherRank < input.N_ranks; otherRank++)
		{
			if (otherRank == rank) { continue; }

			if (AABBIntersects(
				subdomains[rank].haloBounds,
				subdomains[otherRank].ownedBounds))
			{
				subdomains[rank].neighborRanks.push_back(otherRank);
			}
		}
	}

	return subdomains;
}
