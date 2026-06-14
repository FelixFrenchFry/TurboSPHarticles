// uniform_AABB_decomp.cpp

// external
#include <algorithm>
#include <stdexcept>
#include <utility>

// internal
#include "uniform_AABB_decomp.h"



using namespace Dist;

const char* UniformAABBDecomp::Name() const
{
	return "UniformAABBDecomp";
}

void UniformAABBDecomp::Build(const DomainDecompInput& input)
{
	if (!input.globalBounds.IsValid())
	{
		throw std::runtime_error("Cannot decompose invalid global domain bounds.");
	}
	if (input.haloRadius < 0.0f)
	{
		throw std::runtime_error("Cannot decompose domain with negative halo radius.");
	}
	if (input.rankCount <= 0)
	{
		throw std::runtime_error("Cannot decompose domain for non-positive rank count.");
	}

	// one uniform slab per rank
	const int splitAxis = ChooseSplitAxis(input.globalBounds);
	std::vector<AABBRankSubdomain> rankSubdomains =
		BuildSlabSubdomains(input, splitAxis);

	globalBounds_ = input.globalBounds;
	rankSubdomains_ = std::move(rankSubdomains);
}

const AABB& UniformAABBDecomp::GlobalExtentAABB() const
{
	return globalBounds_;
}

int UniformAABBDecomp::OwnerRank(const glm::vec3& position) const
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

bool UniformAABBDecomp::IsInnerOwnedParticle(
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

const std::vector<int>& UniformAABBDecomp::NeighborRanks(const int rank) const
{
	return Subdomain(rank).neighborRanks;
}

AABB UniformAABBDecomp::OwnedExtentAABB(const int rank) const
{
	return Subdomain(rank).ownedBounds;
}

AABB UniformAABBDecomp::HaloExtentAABB(const int rank) const
{
	return Subdomain(rank).haloBounds;
}

bool UniformAABBDecomp::IsValidForRankCount(const int rankCount) const
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

const UniformAABBDecomp::AABBRankSubdomain& UniformAABBDecomp::Subdomain(
	const int rank) const
{
	if (rank < 0 || rank >= static_cast<int>(rankSubdomains_.size()))
	{
		throw std::out_of_range("Invalid uniform AABB decomp rank.");
	}

	return rankSubdomains_[rank];
}

int UniformAABBDecomp::ChooseSplitAxis(const AABB& globalBounds) const
{
	const glm::vec3 extents = globalBounds.Extents();

	// split the longest axis
	int axis = 0;
	if (extents.y > extents[axis]) { axis = 1; }
	if (extents.z > extents[axis]) { axis = 2; }

	return axis;
}

std::vector<UniformAABBDecomp::AABBRankSubdomain>
UniformAABBDecomp::BuildSlabSubdomains(
	const DomainDecompInput& input,
	const int splitAxis) const
{
	std::vector<AABBRankSubdomain> subdomains{};
	subdomains.reserve(static_cast<std::size_t>(input.rankCount));

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

		// each rank only touches previous / next ranks
		if (rank > 0)
		{
			subdomain.neighborRanks.push_back(rank - 1);
		}
		if (rank + 1 < input.rankCount)
		{
			subdomain.neighborRanks.push_back(rank + 1);
		}

		subdomains.push_back(std::move(subdomain));
	}

	return subdomains;
}
