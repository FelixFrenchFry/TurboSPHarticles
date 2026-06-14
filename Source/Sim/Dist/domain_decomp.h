// domain_decomp.h

#pragma once

// external
#include <vector>

// internal
#include "domain_geometry.h"



namespace Dist
{
	struct DomainDecompInput
	{
		AABB globalBounds;
		float haloRadius = 0.0f;
		int rankCount = 1;
	};

	class IDomainDecomp
	{
	public: // ---------------------------------------------------------------------------------------

		virtual ~IDomainDecomp() = default;

		virtual const char* Name() const = 0;
		virtual void Build(const DomainDecompInput& input) = 0;
		virtual const AABB& GlobalExtentAABB() const = 0;

		virtual int OwnerRank(const glm::vec3& position) const = 0;
		// inner = owned by rank and not needed as halo data by a neighboring rank
		virtual bool IsInnerOwnedParticle(
			int rank,
			const glm::vec3& position,
			float kernelRadius) const = 0;

		virtual const std::vector<int>& NeighborRanks(int rank) const = 0;
		virtual AABB OwnedExtentAABB(int rank) const = 0;
		virtual AABB HaloExtentAABB(int rank) const = 0;

		virtual bool IsValidForRankCount(int rankCount) const = 0;
	};

} // namespace Dist
