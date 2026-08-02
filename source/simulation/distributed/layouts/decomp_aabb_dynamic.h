// decomp_aabb_dynamic.h

#pragma once

// external
#include <vector>

// internal
#include "source/simulation/distributed/domain_decomp.h"
#include "aabb_slab_layout.h"



namespace Dist
{
	// dynamic AABB slab decomp, currently seeded with uniform walls along global X
	class Decomp_AABB_Dynamic final : public IDomainDecomp
	{
	public:

		const char* Name() const override { return "Decomp_AABB_Dynamic"; }

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
		bool SupportsRepartition(Base::LoadBalanceWeightModel weightModel) const override;
		void BuildRepartitionProfile(
			const DomainDecompRepartitionParticleInput& input,
			DomainDecompRepartitionProfile& profile) const override;
		void Repartition(const DomainDecompRepartitionInput& input) override;

	private:

		static constexpr int SPLIT_AXIS = 0;
		static constexpr uint32_t PROFILE_BIN_COUNT = 256;

		AABBSlabLayout slabLayout_;
	};

} // namespace Dist
