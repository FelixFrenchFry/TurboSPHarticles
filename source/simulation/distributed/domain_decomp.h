// domain_decomp.h
#pragma once

// external
#include <cstdint>
#include <memory>
#include <span>
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
		float haloRadius = 0.0f; // complete stored halo extent
		int N_ranks = 1;
	};

	// positions of the owned particles used to build repartition profiles
	struct DomainDecompParticlePositionView
	{
		std::span<const float> x;
		std::span<const float> y;
		std::span<const float> z;
	};

	struct DomainDecompRepartitionParticleInput
	{
		Base::LoadBalanceWeightModel weightModel =
			Base::LoadBalanceWeightModel::OwnedParticleCount;
		int rank = 0;
		DomainDecompParticlePositionView fluid;
		DomainDecompParticlePositionView boundary;
		DomainDecompParticlePositionView rigid;
	};

	// local strategy-defined spatial weights split by owned particle type
	struct DomainDecompRepartitionProfile
	{
		std::vector<uint64_t> ownedFluidByUnit;
		std::vector<uint64_t> ownedBoundaryByUnit;
		std::vector<uint64_t> ownedRigidByUnit;
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
			float haloRadius) const = 0;

		virtual const std::vector<int>& NeighborRanks(int rank) const = 0;

		// bounded rank extents for diagnostics, particle halos extend past outer global faces
		virtual AABB OwnedExtentAABB(int rank) const = 0;
		virtual AABB HaloExtentAABB(int rank) const = 0;

		virtual bool IsValidForRankCount(int N_ranks) const = 0;

		// builds the local strategy-defined spatial weight profile
		virtual void BuildRepartitionProfile(
			const DomainDecompRepartitionParticleInput& input,
			DomainDecompRepartitionProfile& profile) const = 0;

		// re-derives the rank subdomains from the globally reduced weight profile
		virtual void Repartition(
			const DomainDecompRepartitionProfile& profile,
			std::span<const double> loadByRank,
			Base::LoadBalanceWeightModel weightModel) = 0;
	};

	// domain decomp factory
	std::unique_ptr<IDomainDecomp> MakeDomainDecomp(Base::DomainDecompMethod method);

} // namespace Dist
