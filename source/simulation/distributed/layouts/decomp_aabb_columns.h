// decomp_aabb_columns.h
#pragma once

// external
#include <vector>

// internal
#include "source/simulation/distributed/domain_decomp.h"
#include "aabb_column_layout.h"



namespace Dist
{
	// dynamically sized AABB columns, split along the global X and Z axes
	class Decomp_AABB_Columns final : public IDomainDecomp
	{
	public:

		const char* Name() const override { return "Decomp_AABB_Columns"; }

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
		void Repartition(const DomainDecompRepartitionInput& input) override;

	private:

		// one fixed-resolution X/Z histogram is kept over the global domain,
		// so section rows can be balanced from the marginals of the bins they cover
		// TODO: find better solution?
		static constexpr uint32_t PROFILE_BIN_COUNT_PER_AXIS = 128;
		static constexpr uint32_t PROFILE_BIN_COUNT =
			PROFILE_BIN_COUNT_PER_AXIS * PROFILE_BIN_COUNT_PER_AXIS;

		AABBColumnLayout columnLayout_;
	};

} // namespace Dist
