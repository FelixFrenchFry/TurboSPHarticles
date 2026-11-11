// aabb_orb_layout.h
#pragma once

// external
#include <cstdint>
#include <vector>

// internal
#include "source/simulation/distributed/domain_geometry.h"



namespace Dist
{
	// internal nodes split a rank range along one axis
	struct AABBOrbNode
	{
		int rankBegin = 0; // first rank of the half-open rank range below this node
		int rankEnd = 1;
		int lowerNode = -1; // child node indices, -1 at leaves
		int upperNode = -1;
		int splitAxis = 0;
		float splitPosition = 0.0f;
	};

	// split the longest axis in proportion to each child's rank count
	std::vector<AABBOrbNode> MakeUniformOrbNodes(const AABB& globalBounds, int N_ranks);

	// split weight by rank count and honor the minimum width
	float MakeBalancedOrbSplitPosition(
		const std::vector<uint64_t>& weightByBin,
		float axisMinPosition,
		float axisMaxPosition,
		float lowerPosition,
		float upperPosition,
		int N_lowerRanks,
		int N_ranks,
		float minSectionExtent);

	// AABB geometry from a bisection tree whose shape depends on rank count
	class AABBOrbLayout final
	{
	public:

		void Build(
			const AABB& globalBounds,
			float haloRadius,
			const std::vector<AABBOrbNode>& nodes);

		// reuse the stored global bounds and halo radius
		void Rebuild(const std::vector<AABBOrbNode>& nodes);

		const AABB& GlobalExtentAABB() const;

		int OwnerRank(const glm::vec3& position) const;

		bool IsInnerOwnedParticle(
			int rank,
			const glm::vec3& position,
			float haloRadius) const;

		const std::vector<int>& NeighborRanks(int rank) const;
		AABB OwnedExtentAABB(int rank) const;
		AABB HaloExtentAABB(int rank) const;
		int RankCount() const { return static_cast<int>(rankSubdomains_.size()); }
		const std::vector<AABBOrbNode>& Nodes() const { return nodes_; }
		float HaloRadius() const { return haloRadius_; }

		bool IsValidForRankCount(int N_ranks) const;

	private:

		struct AABBRankSubdomain
		{
			int rank = 0;
			AABB ownedBounds; // domain bounds owned by this rank
			AABB haloBounds; // particle halo, unbounded past outer global faces
			std::vector<int> neighborRanks; // ranks with halo overlap in either direction
		};

		const AABBRankSubdomain& Subdomain(int rank) const;

		std::vector<AABBRankSubdomain> BuildOrbSubdomains(
			const AABB& globalBounds,
			float haloRadius,
			const std::vector<AABBOrbNode>& nodes) const;

		AABB globalBounds_;
		std::vector<AABBRankSubdomain> rankSubdomains_;
		std::vector<AABBOrbNode> nodes_;
		float haloRadius_ = 0.0f;
	};

} // namespace Dist
