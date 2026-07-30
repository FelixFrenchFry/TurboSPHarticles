// domain_decomp.h

#pragma once

// external
#include <vector>

// internal
#include "domain_geometry.h"



namespace Dist
{
	// input used to build a domain decomp
	struct DomainDecompInput
	{
		AABB globalBounds;
		float haloRadius = 0.0f;
		int N_ranks = 1;
	};

	// interface for domain decomp strategies
	class IDomainDecomp
	{
	public:

		virtual ~IDomainDecomp() = default;

		virtual const char* Name() const = 0;
		virtual void Build(const DomainDecompInput& input) = 0;
		virtual const AABB& GlobalExtentAABB() const = 0;

		// owner query used for migration and local classification
		virtual int OwnerRank(const glm::vec3& position) const = 0;

		virtual bool IsInnerOwnedParticle(
			int rank,
			const glm::vec3& position,
			float kernelRadius) const = 0;

		virtual const std::vector<int>& NeighborRanks(int rank) const = 0;

		// rank-local owned / halo extents used for exchange and diagnostics
		virtual AABB OwnedExtentAABB(int rank) const = 0;
		virtual AABB HaloExtentAABB(int rank) const = 0;

		virtual bool IsValidForRankCount(int N_ranks) const = 0;
	};

} // namespace Dist
