// aabb_column_layout.cpp

// external
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

// internal
#include "aabb_column_layout.h"



using namespace Dist;

namespace
{
	bool AABBIntersects(const AABB& a, const AABB& b)
	{
		return a.min.x <= b.max.x && a.max.x >= b.min.x
			&& a.min.y <= b.max.y && a.max.y >= b.min.y
			&& a.min.z <= b.max.z && a.max.z >= b.min.z;
	}

	// index of the section containing the coordinate within one split position row
	int SectionAtSplitPositions(
		const std::vector<float>& splitPositions,
		const uint64_t rowBegin,
		const int N_sections,
		const float coordinate)
	{
		const auto rowFirst = splitPositions.begin() + static_cast<int64_t>(rowBegin);
		const auto rowLast = rowFirst + N_sections + 1;

		// shared faces go to the higher section
		const auto it = std::upper_bound(rowFirst, rowLast, coordinate);
		const int section = static_cast<int>(it - rowFirst) - 1;

		// positions outside the global domain fall into the nearest section
		return std::clamp(section, 0, N_sections - 1);
	}

	// checks that one split position row tiles the axis with increasing finite walls
	bool IsValidSplitPositionRow(
		const std::vector<float>& splitPositions,
		const uint64_t rowBegin,
		const int N_sections,
		const float minPosition,
		const float maxPosition)
	{
		if (splitPositions[rowBegin] != minPosition) { return false; }
		if (splitPositions[rowBegin + static_cast<uint64_t>(N_sections)] != maxPosition)
		{
			return false;
		}

		for (int splitIdx = 0; splitIdx < N_sections; splitIdx++)
		{
			const float lowerWall = splitPositions[rowBegin + static_cast<uint64_t>(splitIdx)];
			const float upperWall = splitPositions[rowBegin + static_cast<uint64_t>(splitIdx) + 1];
			if (!std::isfinite(lowerWall) ||
				!std::isfinite(upperWall) ||
				lowerWall >= upperWall)
			{
				return false;
			}
		}

		return true;
	}

	// checks that every split position row of the layout is usable
	bool AreValidSplitPositions(
		const AABB& globalBounds,
		const AABBColumnCounts& columnCounts,
		const std::vector<float>& xSplitPositions,
		const std::vector<float>& zSplitPositions)
	{
		if (!IsValidSplitPositionRow(
			xSplitPositions, 0, columnCounts.N_sectionsX,
			globalBounds.min.x, globalBounds.max.x))
		{
			return false;
		}

		for (int sectionX = 0; sectionX < columnCounts.N_sectionsX; sectionX++)
		{
			if (!IsValidSplitPositionRow(
				zSplitPositions,
				AABBColumnLayout::ZSplitRowBegin(columnCounts, sectionX),
				columnCounts.N_sectionsZ,
				globalBounds.min.z, globalBounds.max.z))
			{
				return false;
			}
		}

		return true;
	}

	// expected flattened split position counts of a column grid
	bool HasExpectedSplitPositionCounts(
		const AABBColumnCounts& columnCounts,
		const std::vector<float>& xSplitPositions,
		const std::vector<float>& zSplitPositions)
	{
		const uint64_t N_sectionsX = static_cast<uint64_t>(columnCounts.N_sectionsX);
		const uint64_t N_sectionsZ = static_cast<uint64_t>(columnCounts.N_sectionsZ);

		return xSplitPositions.size() == N_sectionsX + 1
			&& zSplitPositions.size() == N_sectionsX * (N_sectionsZ + 1);
	}

} // namespace anonymous

AABBColumnCounts Dist::MakeBalancedColumnCounts(const AABB& globalBounds, const int N_ranks)
{
	if (!globalBounds.IsValid())
	{
		throw std::runtime_error(
			"Cannot make balanced column counts with invalid global domain bounds.");
	}
	if (N_ranks <= 0)
	{
		throw std::runtime_error(
			"Cannot make balanced column counts for non-positive rank count.");
	}

	// interface area of the uniform column grid the counts would produce
	// (global Y extent is a factor of both terms, so only the footprint decides)
	const glm::vec3 extents = globalBounds.Extents();
	AABBColumnCounts bestColumnCounts{ .N_sectionsX = N_ranks, .N_sectionsZ = 1 };
	double bestInterfaceArea = std::numeric_limits<double>::max();

	// counted down, so equal areas keep splitting the X axis first like the slab decomps
	for (int N_sectionsX = N_ranks; N_sectionsX >= 1; N_sectionsX--)
	{
		if (N_ranks % N_sectionsX != 0) { continue; }

		const int N_sectionsZ = N_ranks / N_sectionsX;
		const double interfaceArea =
			static_cast<double>(N_sectionsX - 1) * extents.y * extents.z +
			static_cast<double>(N_sectionsZ - 1) * extents.x * extents.y;

		// ties keep the first counts found, so every rank derives the same column grid
		if (interfaceArea < bestInterfaceArea)
		{
			bestInterfaceArea = interfaceArea;
			bestColumnCounts = AABBColumnCounts{
				.N_sectionsX = N_sectionsX,
				.N_sectionsZ = N_sectionsZ };
		}
	}

	return bestColumnCounts;
}

std::vector<float> Dist::MakeUniformColumnSplitPositions(
	const float minPosition,
	const float maxPosition,
	const int N_sections)
{
	if (N_sections <= 0)
	{
		throw std::runtime_error(
			"Cannot make uniform column split positions for non-positive section count.");
	}
	if (!std::isfinite(minPosition) || !std::isfinite(maxPosition) || minPosition >= maxPosition)
	{
		throw std::runtime_error(
			"Cannot make uniform column split positions with invalid axis bounds.");
	}

	const float sectionLength = (maxPosition - minPosition) / static_cast<float>(N_sections);

	std::vector<float> splitPositions{};
	splitPositions.resize(static_cast<uint64_t>(N_sections) + 1);

	for (int sectionIdx = 0; sectionIdx < N_sections; sectionIdx++)
	{
		splitPositions[sectionIdx] = minPosition + static_cast<float>(sectionIdx) * sectionLength;
	}
	splitPositions[N_sections] = maxPosition;

	return splitPositions;
}

std::vector<float> Dist::MakeBalancedColumnSplitPositions(
	const std::vector<uint64_t>& weightByBin,
	const float minPosition,
	const float maxPosition,
	const int N_sections,
	const float minSectionExtent)
{
	if (weightByBin.empty())
	{
		throw std::runtime_error(
			"Cannot make balanced column split positions from an empty weight profile.");
	}
	if (!(minSectionExtent > 0.0f))
	{
		throw std::runtime_error(
			"Cannot make balanced column split positions with a non-positive minimum section extent.");
	}

	// axis bounds and section count are validated by the uniform positions this starts from
	std::vector<float> splitPositions =
		MakeUniformColumnSplitPositions(minPosition, maxPosition, N_sections);

	uint64_t totalWeight = 0;
	for (const uint64_t binWeight : weightByBin) { totalWeight += binWeight; }

	// an empty profile carries no balancing information, so the sections stay uniform
	if (totalWeight == 0) { return splitPositions; }

	// place interior walls at equal cumulative weight targets
	const uint64_t N_bins = weightByBin.size();
	const uint64_t sectionCount = static_cast<uint64_t>(N_sections);
	const double binLength = static_cast<double>(maxPosition - minPosition) /
		static_cast<double>(N_bins);
	uint64_t weightBin = 0;
	uint64_t cumulativeWeight = 0;
	for (int splitIdx = 1; splitIdx < N_sections; splitIdx++)
	{
		const uint64_t targetWeight =
			(totalWeight / sectionCount) * static_cast<uint64_t>(splitIdx) +
			(totalWeight % sectionCount) * static_cast<uint64_t>(splitIdx) / sectionCount;

		// find the histogram bin containing this cumulative target
		while (weightBin < N_bins && cumulativeWeight + weightByBin[weightBin] < targetWeight)
		{
			cumulativeWeight += weightByBin[weightBin];
			weightBin += 1;
		}
		if (weightBin >= N_bins)
		{
			throw std::runtime_error(
				"Cannot derive balanced column split positions from the supplied weights.");
		}

		const uint64_t targetWithinBin = targetWeight - cumulativeWeight;
		const double binWeight = static_cast<double>(weightByBin[weightBin]);
		const double binFraction = binWeight > 0.0
			? static_cast<double>(targetWithinBin) / binWeight
			: 0.0;
		const double newWall = static_cast<double>(minPosition) +
			(static_cast<double>(weightBin) + std::clamp(binFraction, 0.0, 1.0)) * binLength;

		// keep every section at least the minimum extent wide, leaving room for the remaining ones
		const float lowerLimit = splitPositions[splitIdx - 1] + minSectionExtent;
		const float upperLimit = maxPosition -
			static_cast<float>(N_sections - splitIdx) * minSectionExtent;
		if (!(lowerLimit <= upperLimit))
		{
			throw std::runtime_error(
				"Cannot fit the requested sections into the axis at the minimum section extent.");
		}
		splitPositions[splitIdx] = std::clamp(
			static_cast<float>(newWall), lowerLimit, upperLimit);
	}

	return splitPositions;
}

uint64_t AABBColumnLayout::ZSplitRowBegin(
	const AABBColumnCounts& columnCounts,
	const int sectionX)
{
	return static_cast<uint64_t>(sectionX) * (static_cast<uint64_t>(columnCounts.N_sectionsZ) + 1);
}

int AABBColumnLayout::ColumnRank(
	const AABBColumnCounts& columnCounts,
	const int sectionX,
	const int sectionZ)
{
	return sectionX * columnCounts.N_sectionsZ + sectionZ;
}

void AABBColumnLayout::Build(
	const AABB& globalBounds,
	const float haloRadius,
	const AABBColumnCounts& columnCounts,
	const std::vector<float>& xSplitPositions,
	const std::vector<float>& zSplitPositions)
{
	// validate inputs before building the layout
	if (!globalBounds.IsValid())
	{
		throw std::runtime_error("Cannot build AABBColumnLayout with invalid global domain bounds.");
	}
	if (haloRadius < 0.0f)
	{
		throw std::runtime_error("Cannot build AABBColumnLayout with negative halo radius.");
	}
	if (columnCounts.N_sectionsX <= 0 || columnCounts.N_sectionsZ <= 0)
	{
		throw std::runtime_error("Cannot build AABBColumnLayout with non-positive section counts.");
	}

	if (!HasExpectedSplitPositionCounts(columnCounts, xSplitPositions, zSplitPositions))
	{
		throw std::runtime_error(
			"Cannot build AABBColumnLayout with an invalid number of split positions.");
	}
	if (!AreValidSplitPositions(globalBounds, columnCounts, xSplitPositions, zSplitPositions))
	{
		throw std::runtime_error(
			"Cannot build AABBColumnLayout with non-increasing split positions "
			"or split positions outside the global bounds.");
	}

	// build the complete rank-local column state from the supplied walls
	std::vector<AABBRankSubdomain> rankSubdomains = BuildColumnSubdomains(
		globalBounds,
		haloRadius,
		columnCounts,
		xSplitPositions,
		zSplitPositions);

	// commit only after validation and subdomain construction succeed
	globalBounds_ = globalBounds;
	columnCounts_ = columnCounts;
	rankSubdomains_ = std::move(rankSubdomains);
	xSplitPositions_ = xSplitPositions;
	zSplitPositions_ = zSplitPositions;
	haloRadius_ = haloRadius;
}

void AABBColumnLayout::Rebuild(
	const std::vector<float>& xSplitPositions,
	const std::vector<float>& zSplitPositions)
{
	if (rankSubdomains_.empty())
	{
		throw std::runtime_error("Cannot rebuild AABBColumnLayout before it was built.");
	}

	// only the walls change, so everything else is re-derived from the stored build state
	Build(
		globalBounds_,
		haloRadius_,
		columnCounts_,
		xSplitPositions,
		zSplitPositions);
}

const AABB& AABBColumnLayout::GlobalExtentAABB() const
{
	return globalBounds_;
}

int AABBColumnLayout::OwnerRank(const glm::vec3& position) const
{
	if (rankSubdomains_.empty()) { return -1; }

	// only the X and Z axes decide ownership, so they are the coordinates that have to be usable
	if (!std::isfinite(position.x) || !std::isfinite(position.z)) { return -1; }

	// the Z walls depend on the X section, while the Y axis is never split
	const int sectionX = SectionAtSplitPositions(
		xSplitPositions_, 0, columnCounts_.N_sectionsX, position.x);
	const int sectionZ = SectionAtSplitPositions(
		zSplitPositions_,
		ZSplitRowBegin(columnCounts_, sectionX),
		columnCounts_.N_sectionsZ,
		position.z);

	return ColumnRank(columnCounts_, sectionX, sectionZ);
}

bool AABBColumnLayout::IsInnerOwnedParticle(
	const int rank,
	const glm::vec3& position,
	const float haloRadius) const
{
	// stored halo bounds already expand each subdomain by the build halo radius,
	// so they only answer this query for the radius used to build the decomp
	if (std::max(0.0f, haloRadius) != haloRadius_)
	{
		throw std::runtime_error(
			"AABBColumnLayout inner particle query radius does not match the build halo radius.");
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

const std::vector<int>& AABBColumnLayout::NeighborRanks(const int rank) const
{
	return Subdomain(rank).neighborRanks;
}

AABB AABBColumnLayout::OwnedExtentAABB(const int rank) const
{
	return Subdomain(rank).ownedBounds;
}

AABB AABBColumnLayout::HaloExtentAABB(const int rank) const
{
	return Subdomain(rank).haloBounds;
}

bool AABBColumnLayout::IsValidForRankCount(const int N_ranks) const
{
	// validate the stored layout state for the given rank count
	if (N_ranks <= 0) { return false; }
	if (static_cast<int>(rankSubdomains_.size()) != N_ranks) { return false; }
	if (!globalBounds_.IsValid()) { return false; }
	if (columnCounts_.N_sectionsX <= 0 || columnCounts_.N_sectionsZ <= 0) { return false; }
	if (columnCounts_.N_sectionsX * columnCounts_.N_sectionsZ != N_ranks) { return false; }

	// validate the stored wall rows before checking rank-local topology
	if (!HasExpectedSplitPositionCounts(columnCounts_, xSplitPositions_, zSplitPositions_))
	{
		return false;
	}
	if (!AreValidSplitPositions(
		globalBounds_, columnCounts_, xSplitPositions_, zSplitPositions_))
	{
		return false;
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

const AABBColumnLayout::AABBRankSubdomain& AABBColumnLayout::Subdomain(const int rank) const
{
	if (rank < 0 || rank >= static_cast<int>(rankSubdomains_.size()))
	{
		throw std::runtime_error("Invalid AABBColumnLayout rank.");
	}

	return rankSubdomains_[rank];
}

std::vector<AABBColumnLayout::AABBRankSubdomain>
AABBColumnLayout::BuildColumnSubdomains(
	const AABB& globalBounds,
	const float haloRadius,
	const AABBColumnCounts& columnCounts,
	const std::vector<float>& xSplitPositions,
	const std::vector<float>& zSplitPositions) const
{
	const int N_ranks = columnCounts.N_sectionsX * columnCounts.N_sectionsZ;

	std::vector<AABBRankSubdomain> subdomains{};
	subdomains.reserve(static_cast<uint64_t>(N_ranks));

	// build owned and halo bounds from adjacent split positions, in increasing rank order,
	// where the owned bounds keep the global Y extent because that axis is never split
	for (int sectionX = 0; sectionX < columnCounts.N_sectionsX; sectionX++)
	{
		const uint64_t zSplitRowBegin = ZSplitRowBegin(columnCounts, sectionX);

		for (int sectionZ = 0; sectionZ < columnCounts.N_sectionsZ; sectionZ++)
		{
			AABB ownedBounds = globalBounds;

			ownedBounds.min.x = xSplitPositions[sectionX];
			ownedBounds.max.x = xSplitPositions[sectionX + 1];
			ownedBounds.min.z =
				zSplitPositions[zSplitRowBegin + static_cast<uint64_t>(sectionZ)];
			ownedBounds.max.z =
				zSplitPositions[zSplitRowBegin + static_cast<uint64_t>(sectionZ) + 1];

			AABBRankSubdomain subdomain{};
			subdomain.rank = ColumnRank(columnCounts, sectionX, sectionZ);
			subdomain.ownedBounds = ownedBounds;
			subdomain.haloBounds = ownedBounds.ExtendedAtGlobalBounds(globalBounds).Expanded(haloRadius);

			subdomains.push_back(std::move(subdomain));
		}
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
