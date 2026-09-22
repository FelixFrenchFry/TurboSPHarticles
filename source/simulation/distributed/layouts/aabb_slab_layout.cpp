// aabb_slab_layout.cpp

// external
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

// internal
#include "aabb_slab_layout.h"



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

std::vector<float> Dist::MakeUniformSlabSplitPositions(
	const AABB& globalBounds,
	const int N_ranks,
	const int splitAxis)
{
	if (!globalBounds.IsValid())
	{
		throw std::runtime_error(
			"Cannot make uniform split positions with invalid global domain bounds.");
	}
	if (N_ranks <= 0)
	{
		throw std::runtime_error(
			"Cannot make uniform split positions for non-positive rank count.");
	}
	if (splitAxis < 0 || splitAxis > 2)
	{
		throw std::runtime_error("Cannot make uniform split positions with invalid split axis.");
	}

	const float splitMin = globalBounds.min[splitAxis];
	const float splitMax = globalBounds.max[splitAxis];
	const float splitLen = splitMax - splitMin;
	const float slabLen = splitLen / static_cast<float>(N_ranks);

	std::vector<float> splitPositions{};
	splitPositions.resize(static_cast<uint64_t>(N_ranks) + 1);

	for (int rank = 0; rank < N_ranks; rank++)
	{
		splitPositions[rank] = splitMin + static_cast<float>(rank) * slabLen;
	}
	splitPositions[N_ranks] = splitMax;

	return splitPositions;
}

std::vector<float> Dist::MakeBalancedSlabSplitPositions(
	const std::vector<uint64_t>& weightByBin,
	const AABB& globalBounds,
	const int N_ranks,
	const int splitAxis)
{
	if (weightByBin.empty())
	{
		throw std::runtime_error(
			"Cannot make balanced split positions from an empty weight profile.");
	}

	// axis bounds and rank count are validated by the uniform positions this starts from
	std::vector<float> splitPositions =
		MakeUniformSlabSplitPositions(globalBounds, N_ranks, splitAxis);

	uint64_t totalWeight = 0;
	for (const uint64_t binWeight : weightByBin) { totalWeight += binWeight; }

	// an empty profile carries no balancing information, so the slabs stay uniform
	if (totalWeight == 0) { return splitPositions; }

	// place interior walls at equal cumulative weight targets
	const uint64_t N_bins = weightByBin.size();
	const uint64_t rankCount = static_cast<uint64_t>(N_ranks);
	const float splitMin = splitPositions.front();
	const float splitMax = splitPositions.back();
	const double binLength = static_cast<double>(splitMax - splitMin) /
		static_cast<double>(N_bins);
	uint64_t weightBin = 0;
	uint64_t cumulativeWeight = 0;
	for (int splitIdx = 1; splitIdx < N_ranks; splitIdx++)
	{
		const uint64_t targetWeight =
			(totalWeight / rankCount) * static_cast<uint64_t>(splitIdx) +
			(totalWeight % rankCount) * static_cast<uint64_t>(splitIdx) / rankCount;

		// find the histogram bin containing this cumulative target
		while (weightBin < N_bins && cumulativeWeight + weightByBin[weightBin] < targetWeight)
		{
			cumulativeWeight += weightByBin[weightBin];
			weightBin += 1;
		}
		if (weightBin >= N_bins)
		{
			throw std::runtime_error(
				"Cannot derive balanced split positions from the supplied weights.");
		}

		const uint64_t targetWithinBin = targetWeight - cumulativeWeight;
		const double binWeight = static_cast<double>(weightByBin[weightBin]);
		const double binFraction = binWeight > 0.0
			? static_cast<double>(targetWithinBin) / binWeight
			: 0.0;
		const double newWall = static_cast<double>(splitMin) +
			(static_cast<double>(weightBin) + std::clamp(binFraction, 0.0, 1.0)) * binLength;

		// keep walls strictly increasing when several targets share a position
		const float previousWall = splitPositions[splitIdx - 1];
		float newWallFloat = static_cast<float>(newWall);
		if (!(newWallFloat > previousWall))
		{
			newWallFloat = std::nextafter(previousWall, splitMax);
		}
		if (!(newWallFloat < splitMax))
		{
			throw std::runtime_error(
				"Cannot derive strictly increasing balanced split positions.");
		}
		splitPositions[splitIdx] = newWallFloat;
	}

	return splitPositions;
}

void AABBSlabLayout::Build(
	const AABB& globalBounds,
	const float haloRadius,
	const int N_ranks,
	const int splitAxis,
	const std::vector<float>& splitPositions)
{
	// validate inputs before building the layout
	if (!globalBounds.IsValid())
	{
		throw std::runtime_error("Cannot build AABBSlabLayout with invalid global domain bounds.");
	}
	if (haloRadius < 0.0f)
	{
		throw std::runtime_error("Cannot build AABBSlabLayout with negative halo radius.");
	}
	if (N_ranks <= 0)
	{
		throw std::runtime_error("Cannot build AABBSlabLayout for non-positive rank count.");
	}
	if (splitAxis < 0 || splitAxis > 2)
	{
		throw std::runtime_error("Cannot build AABBSlabLayout with invalid split axis.");
	}

	const uint64_t N_splitPositions = static_cast<uint64_t>(N_ranks) + 1;
	if (splitPositions.size() != N_splitPositions)
	{
		throw std::runtime_error(
			"Cannot build AABBSlabLayout with an invalid number of split positions.");
	}

	if (splitPositions.front() != globalBounds.min[splitAxis] ||
		splitPositions.back() != globalBounds.max[splitAxis])
	{
		throw std::runtime_error(
			"Cannot build AABBSlabLayout with split positions outside the global bounds.");
	}

	for (int splitIdx = 0; splitIdx < N_ranks; splitIdx++)
	{
		if (!std::isfinite(splitPositions[splitIdx]) ||
			!std::isfinite(splitPositions[splitIdx + 1]) ||
			splitPositions[splitIdx] >= splitPositions[splitIdx + 1])
		{
			throw std::runtime_error(
				"Cannot build AABBSlabLayout with non-increasing split positions.");
		}
	}

	// build the complete rank-local slab state from the supplied walls
	std::vector<AABBRankSubdomain> rankSubdomains = BuildSlabSubdomains(
		globalBounds,
		haloRadius,
		N_ranks,
		splitAxis,
		splitPositions);

	// commit only after validation and subdomain construction succeed
	globalBounds_ = globalBounds;
	splitAxis_ = splitAxis;
	rankSubdomains_ = std::move(rankSubdomains);
	splitPositions_ = splitPositions;
	haloRadius_ = haloRadius;
}

void AABBSlabLayout::Rebuild(const std::vector<float>& splitPositions)
{
	if (rankSubdomains_.empty())
	{
		throw std::runtime_error("Cannot rebuild AABBSlabLayout before it was built.");
	}

	// only the walls change, so everything else is re-derived from the stored build state
	Build(
		globalBounds_,
		haloRadius_,
		static_cast<int>(rankSubdomains_.size()),
		splitAxis_,
		splitPositions);
}

const AABB& AABBSlabLayout::GlobalExtentAABB() const
{
	return globalBounds_;
}

int AABBSlabLayout::OwnerRank(const glm::vec3& position) const
{
	const int N_ranks = static_cast<int>(splitPositions_.size()) - 1;
	if (N_ranks <= 0) { return -1; }

	// only the split axis decides ownership, so it is the one coordinate that has to be usable
	const float splitPosition = position[splitAxis_];
	if (!std::isfinite(splitPosition)) { return -1; }

	// split positions tile the split axis, so shared faces go to the higher rank
	const auto it = std::upper_bound(
		splitPositions_.begin(),
		splitPositions_.end(),
		splitPosition);
	const int ownerRank = static_cast<int>(it - splitPositions_.begin()) - 1;

	// particles that left the global domain stay owned by the nearest slab
	return std::clamp(ownerRank, 0, N_ranks - 1);
}

bool AABBSlabLayout::IsInnerOwnedParticle(
	const int rank,
	const glm::vec3& position,
	const float haloRadius) const
{
	// stored halo bounds already expand each subdomain by the build halo radius,
	// so they only answer this query for the radius used to build the decomp
	if (std::max(0.0f, haloRadius) != haloRadius_)
	{
		throw std::runtime_error(
			"AABBSlabLayout inner particle query radius does not match the build halo radius.");
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

const std::vector<int>& AABBSlabLayout::NeighborRanks(const int rank) const
{
	return Subdomain(rank).neighborRanks;
}

AABB AABBSlabLayout::OwnedExtentAABB(const int rank) const
{
	return Subdomain(rank).ownedBounds;
}

AABB AABBSlabLayout::HaloExtentAABB(const int rank) const
{
	return Subdomain(rank).haloBounds;
}

bool AABBSlabLayout::IsValidForRankCount(const int N_ranks) const
{
	// validate the stored layout state for the given rank count
	if (N_ranks <= 0) { return false; }
	if (static_cast<int>(rankSubdomains_.size()) != N_ranks) { return false; }
	if (!globalBounds_.IsValid()) { return false; }
	if (splitAxis_ < 0 || splitAxis_ > 2) { return false; }

	// validate the stored wall vector before checking rank-local topology
	const uint64_t N_splitPositions = static_cast<uint64_t>(N_ranks) + 1;
	if (splitPositions_.size() != N_splitPositions) { return false; }
	if (splitPositions_.front() != globalBounds_.min[splitAxis_]) { return false; }
	if (splitPositions_.back() != globalBounds_.max[splitAxis_]) { return false; }

	for (int splitIdx = 0; splitIdx < N_ranks; splitIdx++)
	{
		if (!std::isfinite(splitPositions_[splitIdx]) ||
			!std::isfinite(splitPositions_[splitIdx + 1]) ||
			splitPositions_[splitIdx] >= splitPositions_[splitIdx + 1])
		{
			return false;
		}
	}

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

const AABBSlabLayout::AABBRankSubdomain& AABBSlabLayout::Subdomain(const int rank) const
{
	if (rank < 0 || rank >= static_cast<int>(rankSubdomains_.size()))
	{
		throw std::runtime_error("Invalid AABBSlabLayout rank.");
	}

	return rankSubdomains_[rank];
}

std::vector<AABBSlabLayout::AABBRankSubdomain>
AABBSlabLayout::BuildSlabSubdomains(
	const AABB& globalBounds,
	const float haloRadius,
	const int N_ranks,
	const int splitAxis,
	const std::vector<float>& splitPositions) const
{
	std::vector<AABBRankSubdomain> subdomains{};
	subdomains.reserve(static_cast<uint64_t>(N_ranks));

	// build owned and halo bounds from adjacent split positions
	for (int rank = 0; rank < N_ranks; rank++)
	{
		AABB ownedBounds = globalBounds;

		ownedBounds.min[splitAxis] = splitPositions[rank];
		ownedBounds.max[splitAxis] = splitPositions[rank + 1];

		AABBRankSubdomain subdomain{};
		subdomain.rank = rank;
		subdomain.ownedBounds = ownedBounds;
		subdomain.haloBounds = ownedBounds.ExtendedAtGlobalBounds(globalBounds).Expanded(haloRadius);

		subdomains.push_back(std::move(subdomain));
	}

	// connect ranks whose owned bounds overlap the halo on this rank
	for (int rank = 0; rank < N_ranks; rank++)
	{
		for (int otherRank = 0; otherRank < N_ranks; otherRank++)
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
