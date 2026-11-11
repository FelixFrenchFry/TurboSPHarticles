// decomp_aabb_orb.h
#pragma once

// external
#include <vector>

// internal
#include "source/simulation/distributed/domain_decomp.h"
#include "aabb_orb_layout.h"



namespace Dist
{
	// dynamically sized AABB boxes, split by orthogonal recursive bisection
	class Decomp_AABB_Orb final : public IDomainDecomp
	{
	public:

		const char* Name() const override { return "Decomp_AABB_Orb"; }

		void Build(const DomainDecompInput& input) override;
		const AABB& GlobalExtentAABB() const override;

		int OwnerRank(const glm::vec3& position) const override;
		bool IsInnerOwnedParticle(
			int rank,
			const glm::vec3& position,
			float haloRadius) const override;

		const std::vector<int>& NeighborRanks(int rank) const override;
		AABB OwnedExtentAABB(int rank) const override;
		AABB HaloExtentAABB(int rank) const override;

		bool IsValidForRankCount(int N_ranks) const override;
		void BuildRepartitionProfile(
			const DomainDecompRepartitionParticleInput& input,
			DomainDecompRepartitionProfile& profile) const override;
		void Repartition(
			const DomainDecompRepartitionProfile& profile,
			std::span<const double> loadByRank,
			Base::LoadBalanceWeightModel weightModel) override;

	private:

		// fixed-resolution X/Y/Z histogram over the global domain
		static constexpr uint32_t PROFILE_BIN_COUNT_PER_AXIS = 64;
		static constexpr uint32_t PROFILE_BIN_COUNT =
			PROFILE_BIN_COUNT_PER_AXIS * PROFILE_BIN_COUNT_PER_AXIS * PROFILE_BIN_COUNT_PER_AXIS;

		// occupied-span ratio required to switch between viable split axes
		static constexpr float SPLIT_AXIS_SWITCH_RATIO = 1.25f;

		AABBOrbLayout orbLayout_;
	};

} // namespace Dist
