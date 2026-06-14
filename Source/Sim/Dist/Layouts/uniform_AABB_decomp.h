// uniform_AABB_decomp.h

#pragma once

// external
#include <vector>

// internal
#include "Source/Sim/Dist/domain_decomp.h"



namespace Dist
{
	class UniformAABBDecomp final : public IDomainDecomp
	{
	public: // ---------------------------------------------------------------------------------------

		const char* Name() const override;
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
			AABB ownedBounds;
			AABB haloBounds;
			std::vector<int> neighborRanks;
		};

		const AABBRankSubdomain& Subdomain(int rank) const;

		int ChooseSplitAxis(const AABB& globalBounds) const;
		std::vector<AABBRankSubdomain> BuildSlabSubdomains(
			const DomainDecompInput& input,
			int splitAxis) const;

		AABB globalBounds_;
		std::vector<AABBRankSubdomain> rankSubdomains_;
	};

} // namespace Dist
