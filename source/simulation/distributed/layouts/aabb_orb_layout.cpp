// aabb_orb_layout.cpp

// external
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

// internal
#include "aabb_orb_layout.h"



using namespace Dist;

namespace
{
	bool AABBIntersects(const AABB& a, const AABB& b)
	{
		return a.min.x <= b.max.x && a.max.x >= b.min.x
			&& a.min.y <= b.max.y && a.max.y >= b.min.y
			&& a.min.z <= b.max.z && a.max.z >= b.min.z;
	}

	// append each parent before its children
	int AppendOrbNodes(std::vector<AABBOrbNode>& nodes, const int rankBegin, const int rankEnd)
	{
		const int nodeIdx = static_cast<int>(nodes.size());
		nodes.push_back(AABBOrbNode{ .rankBegin = rankBegin, .rankEnd = rankEnd });
		if (rankEnd - rankBegin == 1) { return nodeIdx; }

		// the lower child holds the smaller half of an odd rank range
		const int rankSplit = rankBegin + (rankEnd - rankBegin) / 2;
		const int lowerNode = AppendOrbNodes(nodes, rankBegin, rankSplit);
		const int upperNode = AppendOrbNodes(nodes, rankSplit, rankEnd);
		nodes[nodeIdx].lowerNode = lowerNode;
		nodes[nodeIdx].upperNode = upperNode;

		return nodeIdx;
	}

	// derive the same tree topology on every rank
	std::vector<AABBOrbNode> MakeOrbNodeTopology(const int N_ranks)
	{
		std::vector<AABBOrbNode> nodes{};
		nodes.reserve(2 * static_cast<uint64_t>(N_ranks) - 1);
		AppendOrbNodes(nodes, 0, N_ranks);

		return nodes;
	}

	// checks that the nodes form the tree shape of their root rank range
	bool HasExpectedOrbTopology(const std::vector<AABBOrbNode>& nodes)
	{
		if (nodes.empty()) { return false; }
		if (nodes.front().rankBegin != 0 || nodes.front().rankEnd <= 0) { return false; }

		const int N_ranks = nodes.front().rankEnd;
		if (nodes.size() != 2 * static_cast<uint64_t>(N_ranks) - 1) { return false; }

		const std::vector<AABBOrbNode> expectedNodes = MakeOrbNodeTopology(N_ranks);
		for (uint64_t nodeIdx = 0; nodeIdx != nodes.size(); nodeIdx++)
		{
			const AABBOrbNode& node = nodes[nodeIdx];
			const AABBOrbNode& expectedNode = expectedNodes[nodeIdx];
			if (node.rankBegin != expectedNode.rankBegin ||
				node.rankEnd != expectedNode.rankEnd ||
				node.lowerNode != expectedNode.lowerNode ||
				node.upperNode != expectedNode.upperNode)
			{
				return false;
			}
		}

		return true;
	}

	// checks that every split lies strictly within the bounds of its node
	bool AreValidOrbSplits(const AABB& globalBounds, const std::vector<AABBOrbNode>& nodes)
	{
		std::vector<AABB> nodeBounds(nodes.size());
		nodeBounds.front() = globalBounds;

		for (uint64_t nodeIdx = 0; nodeIdx != nodes.size(); nodeIdx++)
		{
			const AABBOrbNode& node = nodes[nodeIdx];
			if (node.lowerNode < 0) { continue; }

			if (node.splitAxis < 0 || node.splitAxis > 2) { return false; }
			const AABB& bounds = nodeBounds[nodeIdx];
			if (!std::isfinite(node.splitPosition) ||
				node.splitPosition <= bounds.min[node.splitAxis] ||
				node.splitPosition >= bounds.max[node.splitAxis])
			{
				return false;
			}

			nodeBounds[node.lowerNode] = bounds;
			nodeBounds[node.lowerNode].max[node.splitAxis] = node.splitPosition;
			nodeBounds[node.upperNode] = bounds;
			nodeBounds[node.upperNode].min[node.splitAxis] = node.splitPosition;
		}

		return true;
	}

	// bounds of every tree node, derived from the validated splits
	std::vector<AABB> MakeNodeExtentAABBs(
		const AABB& globalBounds,
		const std::vector<AABBOrbNode>& nodes)
	{
		std::vector<AABB> nodeBounds(nodes.size());
		nodeBounds.front() = globalBounds;

		for (uint64_t nodeIdx = 0; nodeIdx != nodes.size(); nodeIdx++)
		{
			const AABBOrbNode& node = nodes[nodeIdx];
			if (node.lowerNode < 0) { continue; }

			nodeBounds[node.lowerNode] = nodeBounds[nodeIdx];
			nodeBounds[node.lowerNode].max[node.splitAxis] = node.splitPosition;
			nodeBounds[node.upperNode] = nodeBounds[nodeIdx];
			nodeBounds[node.upperNode].min[node.splitAxis] = node.splitPosition;
		}

		return nodeBounds;
	}

} // namespace anonymous

std::vector<AABBOrbNode> Dist::MakeUniformOrbNodes(const AABB& globalBounds, const int N_ranks)
{
	if (!globalBounds.IsValid())
	{
		throw std::runtime_error(
			"Cannot make uniform bisection nodes with invalid global domain bounds.");
	}
	if (N_ranks <= 0)
	{
		throw std::runtime_error(
			"Cannot make uniform bisection nodes for non-positive rank count.");
	}

	std::vector<AABBOrbNode> nodes = MakeOrbNodeTopology(N_ranks);
	std::vector<AABB> nodeBounds(nodes.size());
	nodeBounds.front() = globalBounds;

	for (uint64_t nodeIdx = 0; nodeIdx != nodes.size(); nodeIdx++)
	{
		AABBOrbNode& node = nodes[nodeIdx];
		if (node.lowerNode < 0) { continue; }

		// prefer the lower axis index on ties
		const AABB bounds = nodeBounds[nodeIdx];
		const glm::vec3 extents = bounds.Extents();
		int splitAxis = 0;
		for (int axis = 1; axis < 3; axis++)
		{
			if (extents[axis] > extents[splitAxis]) { splitAxis = axis; }
		}

		const int N_lowerRanks = nodes[node.lowerNode].rankEnd - node.rankBegin;
		const int N_nodeRanks = node.rankEnd - node.rankBegin;
		node.splitAxis = splitAxis;
		node.splitPosition = bounds.min[splitAxis] + extents[splitAxis] *
			static_cast<float>(N_lowerRanks) / static_cast<float>(N_nodeRanks);

		nodeBounds[node.lowerNode] = bounds;
		nodeBounds[node.lowerNode].max[splitAxis] = node.splitPosition;
		nodeBounds[node.upperNode] = bounds;
		nodeBounds[node.upperNode].min[splitAxis] = node.splitPosition;
	}

	return nodes;
}

float Dist::MakeBalancedOrbSplitPosition(
	const std::vector<uint64_t>& weightByBin,
	const float axisMinPosition,
	const float axisMaxPosition,
	const float lowerPosition,
	const float upperPosition,
	const int N_lowerRanks,
	const int N_ranks,
	const float minSectionExtent)
{
	if (weightByBin.empty())
	{
		throw std::runtime_error(
			"Cannot make a balanced bisection split position from an empty weight profile.");
	}
	if (!(minSectionExtent > 0.0f))
	{
		throw std::runtime_error(
			"Cannot make a balanced bisection split position with a non-positive minimum section extent.");
	}
	if (!std::isfinite(axisMinPosition) || !std::isfinite(axisMaxPosition) ||
		axisMinPosition >= axisMaxPosition ||
		!std::isfinite(lowerPosition) || !std::isfinite(upperPosition) ||
		lowerPosition >= upperPosition)
	{
		throw std::runtime_error(
			"Cannot make a balanced bisection split position with invalid axis bounds.");
	}
	if (N_lowerRanks <= 0 || N_lowerRanks >= N_ranks)
	{
		throw std::runtime_error(
			"Cannot make a balanced bisection split position for an invalid rank split.");
	}

	// keep both sides at least the minimum extent wide
	const float lowerLimit = lowerPosition + minSectionExtent;
	const float upperLimit = upperPosition - minSectionExtent;
	if (!(lowerLimit <= upperLimit))
	{
		throw std::runtime_error(
			"Cannot fit the requested split into the axis at the minimum section extent.");
	}

	uint64_t totalWeight = 0;
	for (const uint64_t binWeight : weightByBin) { totalWeight += binWeight; }

	// fall back to a uniform split when the profile has no weight to distribute
	if (totalWeight == 0)
	{
		const float uniformSplit = lowerPosition + (upperPosition - lowerPosition) *
			static_cast<float>(N_lowerRanks) / static_cast<float>(N_ranks);
		return std::clamp(uniformSplit, lowerLimit, upperLimit);
	}

	// place the split at the cumulative weight target of the lower ranks
	const uint64_t N_bins = weightByBin.size();
	const uint64_t rankCount = static_cast<uint64_t>(N_ranks);
	const uint64_t lowerRankCount = static_cast<uint64_t>(N_lowerRanks);
	const uint64_t targetWeight =
		(totalWeight / rankCount) * lowerRankCount +
		(totalWeight % rankCount) * lowerRankCount / rankCount;
	const double binLength = static_cast<double>(axisMaxPosition - axisMinPosition) /
		static_cast<double>(N_bins);

	// find the histogram bin containing this cumulative target
	uint64_t weightBin = 0;
	uint64_t cumulativeWeight = 0;
	while (weightBin < N_bins && cumulativeWeight + weightByBin[weightBin] < targetWeight)
	{
		cumulativeWeight += weightByBin[weightBin];
		weightBin += 1;
	}
	if (weightBin >= N_bins)
	{
		throw std::runtime_error(
			"Cannot derive a balanced bisection split position from the supplied weights.");
	}

	const uint64_t targetWithinBin = targetWeight - cumulativeWeight;
	const double binWeight = static_cast<double>(weightByBin[weightBin]);
	const double binFraction = binWeight > 0.0
		? static_cast<double>(targetWithinBin) / binWeight
		: 0.0;
	const double newSplit = static_cast<double>(axisMinPosition) +
		(static_cast<double>(weightBin) + std::clamp(binFraction, 0.0, 1.0)) * binLength;

	return std::clamp(static_cast<float>(newSplit), lowerLimit, upperLimit);
}

void AABBOrbLayout::Build(
	const AABB& globalBounds,
	const float haloRadius,
	const std::vector<AABBOrbNode>& nodes)
{
	// validate inputs before building the layout
	if (!globalBounds.IsValid())
	{
		throw std::runtime_error("Cannot build AABBOrbLayout with invalid global domain bounds.");
	}
	if (haloRadius < 0.0f)
	{
		throw std::runtime_error("Cannot build AABBOrbLayout with negative halo radius.");
	}

	if (!HasExpectedOrbTopology(nodes))
	{
		throw std::runtime_error(
			"Cannot build AABBOrbLayout with nodes that do not form the bisection tree of their ranks.");
	}
	if (!AreValidOrbSplits(globalBounds, nodes))
	{
		throw std::runtime_error(
			"Cannot build AABBOrbLayout with split positions outside their node bounds.");
	}

	// build the complete rank-local box state from the supplied splits
	std::vector<AABBRankSubdomain> rankSubdomains = BuildOrbSubdomains(
		globalBounds,
		haloRadius,
		nodes);

	// commit only after validation and subdomain construction succeed
	globalBounds_ = globalBounds;
	rankSubdomains_ = std::move(rankSubdomains);
	nodes_ = nodes;
	haloRadius_ = haloRadius;
}

void AABBOrbLayout::Rebuild(const std::vector<AABBOrbNode>& nodes)
{
	if (rankSubdomains_.empty())
	{
		throw std::runtime_error("Cannot rebuild AABBOrbLayout before it was built.");
	}
	if (nodes.empty() || nodes.front().rankEnd != RankCount())
	{
		throw std::runtime_error(
			"Cannot rebuild AABBOrbLayout with nodes for a different rank count.");
	}

	Build(
		globalBounds_,
		haloRadius_,
		nodes);
}

const AABB& AABBOrbLayout::GlobalExtentAABB() const
{
	return globalBounds_;
}

int AABBOrbLayout::OwnerRank(const glm::vec3& position) const
{
	if (rankSubdomains_.empty()) { return -1; }

	// shared faces belong to the upper child
	int nodeIdx = 0;
	while (nodes_[nodeIdx].lowerNode >= 0)
	{
		const AABBOrbNode& node = nodes_[nodeIdx];

		const float splitCoordinate = position[node.splitAxis];
		if (!std::isfinite(splitCoordinate)) { return -1; }

		nodeIdx = splitCoordinate < node.splitPosition ? node.lowerNode : node.upperNode;
	}

	// positions outside the global domain end in the nearest leaf
	return nodes_[nodeIdx].rankBegin;
}

bool AABBOrbLayout::IsInnerOwnedParticle(
	const int rank,
	const glm::vec3& position,
	const float haloRadius) const
{
	// inner particle queries require the radius used to build the halo bounds
	if (std::max(0.0f, haloRadius) != haloRadius_)
	{
		throw std::runtime_error(
			"AABBOrbLayout inner particle query radius does not match the build halo radius.");
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

const std::vector<int>& AABBOrbLayout::NeighborRanks(const int rank) const
{
	return Subdomain(rank).neighborRanks;
}

AABB AABBOrbLayout::OwnedExtentAABB(const int rank) const
{
	return Subdomain(rank).ownedBounds;
}

AABB AABBOrbLayout::HaloExtentAABB(const int rank) const
{
	return Subdomain(rank).haloBounds;
}

bool AABBOrbLayout::IsValidForRankCount(const int N_ranks) const
{
	// validate the stored layout state for the given rank count
	if (N_ranks <= 0) { return false; }
	if (static_cast<int>(rankSubdomains_.size()) != N_ranks) { return false; }
	if (!globalBounds_.IsValid()) { return false; }

	// validate the stored tree before checking rank-local topology
	if (!HasExpectedOrbTopology(nodes_)) { return false; }
	if (nodes_.front().rankEnd != N_ranks) { return false; }
	if (!AreValidOrbSplits(globalBounds_, nodes_)) { return false; }

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

const AABBOrbLayout::AABBRankSubdomain& AABBOrbLayout::Subdomain(const int rank) const
{
	if (rank < 0 || rank >= static_cast<int>(rankSubdomains_.size()))
	{
		throw std::runtime_error("Invalid AABBOrbLayout rank.");
	}

	return rankSubdomains_[rank];
}

std::vector<AABBOrbLayout::AABBRankSubdomain>
AABBOrbLayout::BuildOrbSubdomains(
	const AABB& globalBounds,
	const float haloRadius,
	const std::vector<AABBOrbNode>& nodes) const
{
	const int N_ranks = nodes.front().rankEnd;
	const std::vector<AABB> nodeBounds = MakeNodeExtentAABBs(globalBounds, nodes);

	std::vector<AABBRankSubdomain> subdomains{};
	subdomains.resize(static_cast<uint64_t>(N_ranks));

	// each leaf supplies one rank's owned bounds
	for (uint64_t nodeIdx = 0; nodeIdx != nodes.size(); nodeIdx++)
	{
		const AABBOrbNode& node = nodes[nodeIdx];
		if (node.lowerNode >= 0) { continue; }

		AABBRankSubdomain& subdomain = subdomains[static_cast<uint64_t>(node.rankBegin)];
		subdomain.rank = node.rankBegin;
		subdomain.ownedBounds = nodeBounds[nodeIdx];
		subdomain.haloBounds = nodeBounds[nodeIdx].ExtendedAtGlobalBounds(globalBounds).Expanded(haloRadius);
	}

	// halo overlap can round differently in each direction
	for (int rank = 0; rank < N_ranks; rank++)
	{
		AABBRankSubdomain& subdomain = subdomains[rank];
		for (int otherRank = rank + 1; otherRank < N_ranks; otherRank++)
		{
			AABBRankSubdomain& otherSubdomain = subdomains[otherRank];
			if (AABBIntersects(subdomain.haloBounds, otherSubdomain.ownedBounds) ||
				AABBIntersects(otherSubdomain.haloBounds, subdomain.ownedBounds))
			{
				subdomain.neighborRanks.push_back(otherRank);
				otherSubdomain.neighborRanks.push_back(rank);
			}
		}
	}

	return subdomains;
}
