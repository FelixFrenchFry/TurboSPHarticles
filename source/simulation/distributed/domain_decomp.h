// domain_decomp.h

#pragma once

// external
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

// internal
#include "source/base/enum_hub.h"
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

	// rank loads supplied by the load balancing coordinator
	struct DomainDecompRepartitionInput
	{
		std::span<const double> loadByRank;
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

		// reports whether the strategy can apply a new partition
		virtual bool SupportsRepartition() const { return false; }

		// strategies that do not support repartitioning must not silently ignore a request
		virtual void Repartition(const DomainDecompRepartitionInput&)
		{
			throw std::runtime_error("This domain decomp strategy does not support repartitioning.");
		}
	};

	// domain decomp factory
	std::unique_ptr<IDomainDecomp> MakeDomainDecomp(Base::DomainDecompMethod method);

} // namespace Dist
