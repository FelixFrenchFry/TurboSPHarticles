// decomp_aabb_slabs.cpp

// external
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// internal
#include "decomp_aabb_slabs.h"



using namespace Dist;

namespace
{
	void AccumulateOwnedParticleCounts(
		const DomainDecompParticlePositionView& particles,
		const AABB& globalBounds,
		const int splitAxis,
		const uint32_t profileBinCount,
		std::vector<uint64_t>& profile,
		const char* particleType)
	{
		const std::span<const float> splitCoordinates = splitAxis == 0
			? particles.x
			: splitAxis == 1
			? particles.y
			: particles.z;
		const float splitMin = globalBounds.min[splitAxis];
		const float splitMax = globalBounds.max[splitAxis];
		const float splitLength = splitMax - splitMin;
		if (!std::isfinite(splitLength) || splitLength <= 0.0f)
		{
			throw std::runtime_error(
				"Cannot build a slab AABB repartition profile from an invalid global extent.");
		}

		for (const uint32_t localIdx : particles.ownedLocalIdx)
		{
			if (localIdx >= splitCoordinates.size())
			{
				throw std::runtime_error(
					std::string("Cannot build a slab AABB repartition profile with an invalid ") +
					particleType + " particle index.");
			}

			const float splitCoordinate = splitCoordinates[localIdx];
			if (!std::isfinite(splitCoordinate))
			{
				throw std::runtime_error(
					std::string("Cannot build a slab AABB repartition profile from an invalid ") +
					particleType + " particle position.");
			}

			// particles that left the global domain count towards the nearest bin
			const double normalizedCoordinate = std::clamp(
				static_cast<double>(splitCoordinate - splitMin) /
				static_cast<double>(splitLength),
				0.0, 1.0);
			const uint32_t bin = std::min(
				static_cast<uint32_t>(normalizedCoordinate * profileBinCount),
				profileBinCount - 1);
			uint64_t& binCount = profile[bin];
			if (binCount == std::numeric_limits<uint64_t>::max())
			{
				throw std::runtime_error(
					"Cannot build a slab AABB repartition profile with an overflowing bin.");
			}
			binCount += 1;
		}
	}

} // namespace anonymous

void Decomp_AABB_Slabs::Build(const DomainDecompInput& input)
{
	if (!input.globalBounds.IsValid())
	{
		throw std::runtime_error(
			"Cannot build Decomp_AABB_Slabs with invalid global domain bounds.");
	}
	if (input.haloRadius <= 0.0f)
	{
		throw std::runtime_error(
			"Cannot build Decomp_AABB_Slabs with a non-positive halo radius.");
	}
	if (input.N_ranks <= 0)
	{
		throw std::runtime_error(
			"Cannot build Decomp_AABB_Slabs for non-positive rank count.");
	}

	// initially built uniformly sized AABB slabs (one per rank)
	slabLayout_.Build(
		input.globalBounds,
		input.haloRadius,
		input.N_ranks,
		SPLIT_AXIS,
		MakeUniformSlabSplitPositions(
			input.globalBounds, input.N_ranks, SPLIT_AXIS));
}

void Decomp_AABB_Slabs::Restore(
	const DomainDecompInput& input,
	const std::span<const AABB> ownedBounds)
{
	if (input.N_ranks <= 0 || ownedBounds.size() != static_cast<size_t>(input.N_ranks))
	{
		throw std::runtime_error("Invalid checkpoint slab count.");
	}

	Build(input);
	std::vector<float> splitPositions{ ownedBounds.front().min[SPLIT_AXIS] };
	for (const AABB& bounds : ownedBounds)
	{
		splitPositions.push_back(bounds.max[SPLIT_AXIS]);
	}

	slabLayout_.Rebuild(splitPositions);
	for (int rank = 0; rank < input.N_ranks; rank++)
	{
		const AABB restored = OwnedExtentAABB(rank);
		if (restored.min != ownedBounds[rank].min || restored.max != ownedBounds[rank].max)
		{
			throw std::runtime_error("Checkpoint bounds do not form a valid slab decomposition.");
		}
	}
}

const AABB& Decomp_AABB_Slabs::GlobalExtentAABB() const
{
	return slabLayout_.GlobalExtentAABB();
}

int Decomp_AABB_Slabs::OwnerRank(const glm::vec3& position) const
{
	return slabLayout_.OwnerRank(position);
}

bool Decomp_AABB_Slabs::IsInnerOwnedParticle(
	const int rank,
	const glm::vec3& position,
	const float haloRadius) const
{
	return slabLayout_.IsInnerOwnedParticle(rank, position, haloRadius);
}

const std::vector<int>& Decomp_AABB_Slabs::NeighborRanks(const int rank) const
{
	return slabLayout_.NeighborRanks(rank);
}

AABB Decomp_AABB_Slabs::OwnedExtentAABB(const int rank) const
{
	return slabLayout_.OwnedExtentAABB(rank);
}

AABB Decomp_AABB_Slabs::HaloExtentAABB(const int rank) const
{
	return slabLayout_.HaloExtentAABB(rank);
}

bool Decomp_AABB_Slabs::IsValidForRankCount(const int N_ranks) const
{
	return slabLayout_.IsValidForRankCount(N_ranks);
}

void Decomp_AABB_Slabs::BuildRepartitionProfile(
	const DomainDecompRepartitionParticleInput& input,
	DomainDecompRepartitionProfile& profile) const
{
	if (input.weightModel != Base::LoadBalanceWeightModel::OwnedParticleCount)
	{
		throw std::runtime_error(
			"Decomp_AABB_Slabs does not support the requested repartition weight model.");
	}

	const int N_ranks = slabLayout_.RankCount();
	if (input.rank < 0 || input.rank >= N_ranks)
	{
		throw std::runtime_error(
			"Cannot build a slab AABB repartition profile for an invalid rank.");
	}

	// one fixed-resolution histogram is kept over the global domain
	profile.ownedFluidByUnit.assign(PROFILE_BIN_COUNT, 0);
	profile.ownedBoundaryByUnit.assign(PROFILE_BIN_COUNT, 0);

	const AABB globalBounds = slabLayout_.GlobalExtentAABB();
	AccumulateOwnedParticleCounts(
		input.fluid,
		globalBounds,
		SPLIT_AXIS,
		PROFILE_BIN_COUNT,
		profile.ownedFluidByUnit,
		"fluid");
	AccumulateOwnedParticleCounts(
		input.boundary,
		globalBounds,
		SPLIT_AXIS,
		PROFILE_BIN_COUNT,
		profile.ownedBoundaryByUnit,
		"boundary");
}

void Decomp_AABB_Slabs::Repartition(const DomainDecompRepartitionInput& input)
{
	if (input.weightModel != Base::LoadBalanceWeightModel::OwnedParticleCount)
	{
		throw std::runtime_error(
			"Decomp_AABB_Slabs does not support the requested repartition weight model.");
	}

	// validate rank loads against the current slab layout
	const int N_ranks = static_cast<int>(input.loadByRank.size());
	if (!slabLayout_.IsValidForRankCount(N_ranks))
	{
		throw std::runtime_error(
			"Cannot repartition Decomp_AABB_Slabs with a rank load count that "
			"does not match the current layout.");
	}

	// validate rank loads supplied to the trigger
	for (const double load : input.loadByRank)
	{
		if (!std::isfinite(load) || load < 0.0)
		{
			throw std::runtime_error(
				"Cannot repartition Decomp_AABB_Slabs with invalid rank loads.");
		}
	}

	if (input.ownedFluidByUnit.size() != PROFILE_BIN_COUNT ||
		input.ownedBoundaryByUnit.size() != PROFILE_BIN_COUNT)
	{
		throw std::runtime_error(
			"Cannot repartition Decomp_AABB_Slabs with an invalid particle weight profile.");
	}

	const AABB globalBounds = slabLayout_.GlobalExtentAABB();

	// TODO: restore a boundary-particle weight when boundary work is modeled?
	const std::span<const uint64_t> weightByBin = input.ownedFluidByUnit;

	// an empty profile carries no balancing information, so the current layout is kept
	uint64_t totalWeight = 0;
	for (const uint64_t binWeight : weightByBin) { totalWeight += binWeight; }
	if (totalWeight == 0) { return; }

	// place the global walls from the profile, which is already binned along the split axis
	const std::vector<uint64_t> axisWeightByBin(weightByBin.begin(), weightByBin.end());
	const std::vector<float> splitPositions = MakeBalancedSlabSplitPositions(
		axisWeightByBin, globalBounds, N_ranks, SPLIT_AXIS);

	// rebuild owned and halo extents from new walls
	slabLayout_.Rebuild(splitPositions);
}
