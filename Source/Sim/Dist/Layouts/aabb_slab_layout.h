// aabb_slab_layout.h

#pragma once

// external
#include <vector>

// internal
#include "Source/Sim/Dist/domain_geometry.h"



namespace Dist
{
	// reusable AABB slab geometry built from ordered split positions
	class AABBSlabLayout final
	{
	public:

		void Build(
			const AABB& globalBounds,
			float haloRadius,
			int N_ranks,
			int splitAxis,
			const std::vector<float>& splitPositions);

		const AABB& GlobalExtentAABB() const;

		int OwnerRank(const glm::vec3& position) const;

		bool IsInnerOwnedParticle(
			int rank,
			const glm::vec3& position,
			float kernelRadius) const;

		const std::vector<int>& NeighborRanks(int rank) const;
		AABB OwnedExtentAABB(int rank) const;
		AABB HaloExtentAABB(int rank) const;

		bool IsValidForRankCount(int N_ranks) const;

	private:

		struct AABBRankSubdomain
		{
			int rank = 0;
			AABB ownedBounds; // domain bounds owned by this rank
			AABB haloBounds; // owned bounds expanded by halo radius
			std::vector<int> neighborRanks; // ranks whose owned bounds overlap this halo
		};

		const AABBRankSubdomain& Subdomain(int rank) const;

		std::vector<AABBRankSubdomain> BuildSlabSubdomains(
			const AABB& globalBounds,
			float haloRadius,
			int N_ranks,
			int splitAxis,
			const std::vector<float>& splitPositions) const;

		int splitAxis_ = 0;
		AABB globalBounds_;
		std::vector<AABBRankSubdomain> rankSubdomains_;
		std::vector<float> splitPositions_;
		float haloRadius_ = 0.0f;
	};

} // namespace Dist
