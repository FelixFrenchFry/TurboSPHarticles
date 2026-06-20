// decomp_AABB_uniform.h

#pragma once

// external
#include <vector>

// internal
#include "Source/Sim/Dist/domain_decomp.h"



namespace Dist
{
	// uniform AABB slab decomp, split along the global X axis
	class Decomp_AABB_Uniform final : public IDomainDecomp
	{
	public: // ---------------------------------------------------------------------------------------

		const char* Name() const override { return "Decomp_AABB_Uniform"; }

		void Build(const DomainDecompInput& input) override;
		const AABB& GlobalExtentAABB() const override;

		int OwnerRank(const glm::vec3& position) const override;
		bool IsInnerOwnedParticle(
			int rank,
			const glm::vec3& position,
			float kernelRadius) const override;

		const std::vector<int>& NeighborRanks(int rank) const override;
		AABB OwnedExtentAABB(int rank) const override;
		AABB HaloExtentAABB(int rank) const override;

		bool IsValidForRankCount(int rankCount) const override;

	private: // --------------------------------------------------------------------------------------

		struct AABBRankSubdomain
		{
			int rank = 0;
			AABB ownedBounds; // domain bounds owned by this rank
			AABB haloBounds; // owned bounds expanded by halo radius
			std::vector<int> neighborRanks; // ranks whose owned bounds overlap this halo
		};

		const AABBRankSubdomain& Subdomain(int rank) const;

		std::vector<AABBRankSubdomain> BuildSlabSubdomains(
			const DomainDecompInput& input,
			int splitAxis) const;

		AABB globalBounds_;
		std::vector<AABBRankSubdomain> rankSubdomains_;
	};

} // namespace Dist
