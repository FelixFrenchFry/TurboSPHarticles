// decomp_aabb_uniform.h

#pragma once

// external
#include <vector>

// internal
#include "Source/Sim/Dist/domain_decomp.h"
#include "aabb_slab_layout.h"



namespace Dist
{
	// uniform AABB slab decomp, split along the global X axis
	class Decomp_AABB_Uniform final : public IDomainDecomp
	{
	public:

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

		bool IsValidForRankCount(int N_ranks) const override;
		bool SupportsRepartition() const override { return false; }
		void Repartition(const DomainDecompRepartitionInput& input) override;

	private:

		static constexpr int SPLIT_AXIS = 0;

		AABBSlabLayout slabLayout_;
	};

} // namespace Dist
