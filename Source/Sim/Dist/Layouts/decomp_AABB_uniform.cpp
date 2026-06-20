// decomp_AABB_uniform.cpp

// external
#include <algorithm>
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
}

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
	if (input.rankCount <= 0)
	{
		throw std::runtime_error("Cannot build Decomp_AABB_Uniform for non-positive rank count.");
	}

	// one uniform AABB slab per rank, always split along the global X axis
	constexpr int splitAxis = 0;
	std::vector<AABBRankSubdomain> rankSubdomains =
		BuildSlabSubdomains(input, splitAxis);

	globalBounds_ = input.globalBounds;
	rankSubdomains_ = std::move(rankSubdomains);
}

const AABB& Decomp_AABB_Uniform::GlobalExtentAABB() const
{
	return globalBounds_;
}

int Decomp_AABB_Uniform::OwnerRank(const glm::vec3& position) const
{
	if (!globalBounds_.Contains(position)) { return -1; }

	int ownerRank = -1;
	for (const AABBRankSubdomain& subdomain : rankSubdomains_)
	{
		if (subdomain.ownedBounds.Contains(position))
		{
			// deterministic tie-break on shared AABB faces
			ownerRank = subdomain.rank;
		}
	}

	return ownerRank;
}

bool Decomp_AABB_Uniform::IsInnerOwnedParticle(
	const int rank,
	const glm::vec3& position,
	const float kernelRadius) const
{
	if (OwnerRank(position) != rank) { return false; }

	const AABBRankSubdomain& localSubdomain = Subdomain(rank);
	const float haloRadius = std::max(0.0f, kernelRadius);

	for (const int neighborRank : localSubdomain.neighborRanks)
	{
		// outer owned particles are needed by at least one neighbor rank
		const AABB neighborReach =
			Subdomain(neighborRank).ownedBounds.Expanded(haloRadius);
		if (neighborReach.Contains(position))
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

bool Decomp_AABB_Uniform::IsValidForRankCount(const int rankCount) const
{
	if (rankCount <= 0) { return false; }
	if (static_cast<int>(rankSubdomains_.size()) != rankCount) { return false; }
	if (!globalBounds_.IsValid()) { return false; }

	for (int rank = 0; rank < rankCount; rank++)
	{
		const AABBRankSubdomain& subdomain = rankSubdomains_[rank];

		if (subdomain.rank != rank) { return false; }
		if (!subdomain.ownedBounds.IsValid()) { return false; }
		if (!subdomain.haloBounds.IsValid()) { return false; }
		if (!globalBounds_.Contains(subdomain.ownedBounds.min)) { return false; }
		if (!globalBounds_.Contains(subdomain.ownedBounds.max)) { return false; }

		for (const int neighborRank : subdomain.neighborRanks)
		{
			if (neighborRank < 0 || neighborRank >= rankCount) { return false; }
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
		throw std::out_of_range("Invalid Decomp_AABB_Uniform rank.");
	}

	return rankSubdomains_[rank];
}

std::vector<Decomp_AABB_Uniform::AABBRankSubdomain>
Decomp_AABB_Uniform::BuildSlabSubdomains(
	const DomainDecompInput& input,
	const int splitAxis) const
{
	std::vector<AABBRankSubdomain> subdomains{};
	subdomains.reserve(static_cast<uint64_t>(input.rankCount));

	const float splitMin = input.globalBounds.min[splitAxis];
	const float splitMax = input.globalBounds.max[splitAxis];
	const float splitLen = splitMax - splitMin;
	const float slabLen = splitLen / static_cast<float>(input.rankCount);

	for (int rank = 0; rank < input.rankCount; rank++)
	{
		AABB ownedBounds = input.globalBounds;

		ownedBounds.min[splitAxis] =
			splitMin + static_cast<float>(rank) * slabLen;
		ownedBounds.max[splitAxis] = (rank == input.rankCount - 1)
			? splitMax
			: splitMin + static_cast<float>(rank + 1) * slabLen;

		AABBRankSubdomain subdomain{};
		subdomain.rank = rank;
		subdomain.ownedBounds = ownedBounds;
		subdomain.haloBounds = ownedBounds.Expanded(input.haloRadius);

		subdomains.push_back(std::move(subdomain));
	}

	for (int rank = 0; rank < input.rankCount; rank++)
	{
		for (int otherRank = 0; otherRank < input.rankCount; otherRank++)
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
