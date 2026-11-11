// decomp_aabb_orb.cpp

// external
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// internal
#include "decomp_aabb_orb.h"



using namespace Dist;

namespace
{
	// half-open bin ranges for one node
	struct ProfileBinRegion
	{
		std::array<uint32_t, 3> beginBins{};
		std::array<uint32_t, 3> endBins{};
	};

	uint64_t ProfileBinIdx(
		const uint32_t binX,
		const uint32_t binY,
		const uint32_t binZ,
		const uint32_t binCountPerAxis)
	{
		return (static_cast<uint64_t>(binX) * binCountPerAxis + binY) * binCountPerAxis + binZ;
	}

	uint32_t ProfileBinAtPosition(
		const float position,
		const float axisMin,
		const float axisExtent,
		const uint32_t binCountPerAxis)
	{
		// positions outside the global domain fall into the nearest bin
		const double normalizedCoordinate = std::clamp(
			static_cast<double>(position - axisMin) / static_cast<double>(axisExtent),
			0.0, 1.0);

		return std::min(
			static_cast<uint32_t>(normalizedCoordinate * binCountPerAxis),
			binCountPerAxis - 1);
	}

	// half-open bin range for one axis of a node
	void SetProfileBinRange(
		const float minPosition,
		const float maxPosition,
		const float axisMin,
		const float axisMax,
		const uint32_t binCountPerAxis,
		uint32_t& beginBin,
		uint32_t& endBin)
	{
		const float axisExtent = axisMax - axisMin;
		beginBin = ProfileBinAtPosition(minPosition, axisMin, axisExtent, binCountPerAxis);

		// include the last bin at the global wall
		endBin = maxPosition >= axisMax
			? binCountPerAxis
			: ProfileBinAtPosition(maxPosition, axisMin, axisExtent, binCountPerAxis);

		// a node thinner than one bin still covers the bin it starts in
		endBin = std::max(endBin, beginBin + 1);
	}

	// sum the region's weights into one histogram per axis
	void AccumulateAxisWeights(
		const std::span<const uint64_t> weightByBin,
		const uint32_t binCountPerAxis,
		const ProfileBinRegion& region,
		std::array<std::vector<uint64_t>, 3>& axisWeightByBin)
	{
		for (std::vector<uint64_t>& weights : axisWeightByBin) { weights.assign(binCountPerAxis, 0); }

		for (uint32_t binX = region.beginBins[0]; binX != region.endBins[0]; binX++)
		{
			for (uint32_t binY = region.beginBins[1]; binY != region.endBins[1]; binY++)
			{
				for (uint32_t binZ = region.beginBins[2]; binZ != region.endBins[2]; binZ++)
				{
					const uint64_t binWeight =
						weightByBin[ProfileBinIdx(binX, binY, binZ, binCountPerAxis)];

					axisWeightByBin[0][binX] += binWeight;
					axisWeightByBin[1][binY] += binWeight;
					axisWeightByBin[2][binZ] += binWeight;
				}
			}
		}
	}

	// span from the first occupied bin to the last
	float WeightedAxisExtent(const std::vector<uint64_t>& axisWeightByBin, const float binLength)
	{
		const auto first = std::find_if(axisWeightByBin.begin(), axisWeightByBin.end(),
			[](const uint64_t binWeight) { return binWeight > 0; });
		if (first == axisWeightByBin.end()) { return 0.0f; }

		const auto last = std::find_if(axisWeightByBin.rbegin(), axisWeightByBin.rend(),
			[](const uint64_t binWeight) { return binWeight > 0; });
		const int64_t N_weightedBins = (axisWeightByBin.rend() - last) - (first - axisWeightByBin.begin());

		return static_cast<float>(N_weightedBins) * binLength;
	}

	void AccumulateOwnedParticleCounts(
		const DomainDecompParticlePositionView& particles,
		const AABB& globalBounds,
		const uint32_t binCountPerAxis,
		std::vector<uint64_t>& profile,
		const char* particleType)
	{
		const glm::vec3 extents = globalBounds.Extents();
		for (int axis = 0; axis < 3; axis++)
		{
			if (!std::isfinite(extents[axis]) || extents[axis] <= 0.0f)
			{
				throw std::runtime_error(
					"Cannot build a bisection AABB repartition profile from an invalid global extent.");
			}
		}

		if (particles.x.size() != particles.y.size() || particles.x.size() != particles.z.size())
		{
			throw std::runtime_error(
				std::string("Cannot build a bisection AABB repartition profile with inconsistent ") +
				particleType + " particle positions.");
		}

		for (uint64_t localIdx = 0; localIdx != particles.x.size(); localIdx++)
		{
			const float xCoordinate = particles.x[localIdx];
			const float yCoordinate = particles.y[localIdx];
			const float zCoordinate = particles.z[localIdx];
			if (!std::isfinite(xCoordinate) || !std::isfinite(yCoordinate) || !std::isfinite(zCoordinate))
			{
				throw std::runtime_error(
					std::string("Cannot build a bisection AABB repartition profile from an invalid ") +
					particleType + " particle position.");
			}

			const uint32_t binX = ProfileBinAtPosition(
				xCoordinate, globalBounds.min.x, extents.x, binCountPerAxis);
			const uint32_t binY = ProfileBinAtPosition(
				yCoordinate, globalBounds.min.y, extents.y, binCountPerAxis);
			const uint32_t binZ = ProfileBinAtPosition(
				zCoordinate, globalBounds.min.z, extents.z, binCountPerAxis);

			uint64_t& binCount = profile[ProfileBinIdx(binX, binY, binZ, binCountPerAxis)];
			if (binCount == std::numeric_limits<uint64_t>::max())
			{
				throw std::runtime_error(
					"Cannot build a bisection AABB repartition profile with an overflowing bin.");
			}
			binCount += 1;
		}
	}

} // namespace anonymous

void Decomp_AABB_Orb::Build(const DomainDecompInput& input)
{
	if (!input.globalBounds.IsValid())
	{
		throw std::runtime_error(
			"Cannot build Decomp_AABB_Orb with invalid global domain bounds.");
	}
	if (input.haloRadius <= 0.0f)
	{
		throw std::runtime_error(
			"Cannot build Decomp_AABB_Orb with a non-positive halo radius.");
	}
	if (input.N_ranks <= 0)
	{
		throw std::runtime_error(
			"Cannot build Decomp_AABB_Orb for non-positive rank count.");
	}

	// derive the same equal-volume partition on every rank
	AABBOrbLayout orbLayout{};
	orbLayout.Build(
		input.globalBounds,
		input.haloRadius,
		MakeUniformOrbNodes(input.globalBounds, input.N_ranks));

	// split boxes thinner than the halo radius reach past their direct neighbors
	const glm::vec3 globalExtents = input.globalBounds.Extents();
	for (int rank = 0; rank < input.N_ranks; rank++)
	{
		const glm::vec3 ownedExtents = orbLayout.OwnedExtentAABB(rank).Extents();
		for (int axis = 0; axis < 3; axis++)
		{
			if (ownedExtents[axis] < globalExtents[axis] && ownedExtents[axis] < input.haloRadius)
			{
				throw std::runtime_error(
					"Cannot build Decomp_AABB_Orb with boxes thinner than the halo radius.");
			}
		}
	}

	orbLayout_ = std::move(orbLayout);
}

const AABB& Decomp_AABB_Orb::GlobalExtentAABB() const
{
	return orbLayout_.GlobalExtentAABB();
}

int Decomp_AABB_Orb::OwnerRank(const glm::vec3& position) const
{
	return orbLayout_.OwnerRank(position);
}

bool Decomp_AABB_Orb::IsInnerOwnedParticle(
	const int rank,
	const glm::vec3& position,
	const float haloRadius) const
{
	return orbLayout_.IsInnerOwnedParticle(rank, position, haloRadius);
}

const std::vector<int>& Decomp_AABB_Orb::NeighborRanks(const int rank) const
{
	return orbLayout_.NeighborRanks(rank);
}

AABB Decomp_AABB_Orb::OwnedExtentAABB(const int rank) const
{
	return orbLayout_.OwnedExtentAABB(rank);
}

AABB Decomp_AABB_Orb::HaloExtentAABB(const int rank) const
{
	return orbLayout_.HaloExtentAABB(rank);
}

bool Decomp_AABB_Orb::IsValidForRankCount(const int N_ranks) const
{
	return orbLayout_.IsValidForRankCount(N_ranks);
}

void Decomp_AABB_Orb::BuildRepartitionProfile(
	const DomainDecompRepartitionParticleInput& input,
	DomainDecompRepartitionProfile& profile) const
{
	if (input.weightModel != Base::LoadBalanceWeightModel::OwnedParticleCount)
	{
		throw std::runtime_error(
			"Decomp_AABB_Orb does not support the requested repartition weight model.");
	}

	const int N_ranks = orbLayout_.RankCount();
	if (input.rank < 0 || input.rank >= N_ranks)
	{
		throw std::runtime_error(
			"Cannot build a bisection AABB repartition profile for an invalid rank.");
	}

	// one fixed-resolution histogram is kept over the global domain
	profile.ownedFluidByUnit.assign(PROFILE_BIN_COUNT, 0);
	profile.ownedBoundaryByUnit.assign(PROFILE_BIN_COUNT, 0);
	profile.ownedRigidByUnit.assign(PROFILE_BIN_COUNT, 0);

	const AABB globalBounds = orbLayout_.GlobalExtentAABB();
	AccumulateOwnedParticleCounts(
		input.fluid,
		globalBounds,
		PROFILE_BIN_COUNT_PER_AXIS,
		profile.ownedFluidByUnit,
		"fluid");
	AccumulateOwnedParticleCounts(
		input.boundary,
		globalBounds,
		PROFILE_BIN_COUNT_PER_AXIS,
		profile.ownedBoundaryByUnit,
		"boundary");
	AccumulateOwnedParticleCounts(
		input.rigid,
		globalBounds,
		PROFILE_BIN_COUNT_PER_AXIS,
		profile.ownedRigidByUnit,
		"rigid");
}

void Decomp_AABB_Orb::Repartition(
	const DomainDecompRepartitionProfile& profile,
	const std::span<const double> loadByRank,
	const Base::LoadBalanceWeightModel weightModel)
{
	if (weightModel != Base::LoadBalanceWeightModel::OwnedParticleCount)
	{
		throw std::runtime_error(
			"Decomp_AABB_Orb does not support the requested repartition weight model.");
	}

	// validate rank loads against the current box layout
	const int N_ranks = static_cast<int>(loadByRank.size());
	if (!orbLayout_.IsValidForRankCount(N_ranks))
	{
		throw std::runtime_error(
			"Cannot repartition Decomp_AABB_Orb with a rank load count that "
			"does not match the current layout.");
	}

	// validate rank loads supplied to the trigger
	for (const double load : loadByRank)
	{
		if (!std::isfinite(load) || load < 0.0)
		{
			throw std::runtime_error(
				"Cannot repartition Decomp_AABB_Orb with invalid rank loads.");
		}
	}

	if (profile.ownedFluidByUnit.size() != PROFILE_BIN_COUNT ||
		profile.ownedBoundaryByUnit.size() != PROFILE_BIN_COUNT ||
		profile.ownedRigidByUnit.size() != PROFILE_BIN_COUNT)
	{
		throw std::runtime_error(
			"Cannot repartition Decomp_AABB_Orb with an invalid particle weight profile.");
	}

	const AABB globalBounds = orbLayout_.GlobalExtentAABB();
	const glm::vec3 extents = globalBounds.Extents();
	const float minSectionExtent = orbLayout_.HaloRadius();

	// TODO: restore a boundary-particle weight when boundary work is modeled?
	const std::span<const uint64_t> weightByBin = profile.ownedFluidByUnit;

	// keep the current layout when the profile has no fluid weight to balance
	uint64_t totalWeight = 0;
	for (const uint64_t binWeight : weightByBin) { totalWeight += binWeight; }
	if (totalWeight == 0) { return; }

	// reuse the tree topology
	std::vector<AABBOrbNode> nodes = orbLayout_.Nodes();
	std::vector<AABB> nodeBounds(nodes.size());
	nodeBounds.front() = globalBounds;
	std::array<std::vector<uint64_t>, 3> axisWeightByBin{};

	// place the splits top-down
	for (uint64_t nodeIdx = 0; nodeIdx != nodes.size(); nodeIdx++)
	{
		AABBOrbNode& node = nodes[nodeIdx];
		if (node.lowerNode < 0) { continue; }

		const AABB bounds = nodeBounds[nodeIdx];
		ProfileBinRegion nodeRegion{};
		for (int axis = 0; axis < 3; axis++)
		{
			SetProfileBinRange(
				bounds.min[axis], bounds.max[axis],
				globalBounds.min[axis], globalBounds.max[axis],
				PROFILE_BIN_COUNT_PER_AXIS,
				nodeRegion.beginBins[axis], nodeRegion.endBins[axis]);
		}
		AccumulateAxisWeights(weightByBin, PROFILE_BIN_COUNT_PER_AXIS, nodeRegion, axisWeightByBin);

		// axes too short for two sections at the minimum extent cannot be split
		std::array<float, 3> weightedExtents{};
		std::array<bool, 3> splittableAxes{};
		int widestAxis = -1;
		for (int axis = 0; axis < 3; axis++)
		{
			weightedExtents[axis] = WeightedAxisExtent(
				axisWeightByBin[axis],
				extents[axis] / static_cast<float>(PROFILE_BIN_COUNT_PER_AXIS));
			const float lowerLimit = bounds.min[axis] + minSectionExtent;
			const float upperLimit = bounds.max[axis] - minSectionExtent;
			splittableAxes[axis] = lowerLimit <= upperLimit;

			if (splittableAxes[axis] &&
				(widestAxis < 0 || weightedExtents[axis] > weightedExtents[widestAxis]))
			{
				widestAxis = axis;
			}
		}
		// keep the current layout if a candidate node has no room for its split
		if (widestAxis < 0) { return; }

		// prefer the current axis to limit particle migration
		int splitAxis = node.splitAxis;
		if (!splittableAxes[splitAxis] ||
			weightedExtents[widestAxis] > SPLIT_AXIS_SWITCH_RATIO * weightedExtents[splitAxis])
		{
			splitAxis = widestAxis;
		}

		node.splitAxis = splitAxis;
		node.splitPosition = MakeBalancedOrbSplitPosition(
			axisWeightByBin[splitAxis],
			globalBounds.min[splitAxis], globalBounds.max[splitAxis],
			bounds.min[splitAxis], bounds.max[splitAxis],
			nodes[node.lowerNode].rankEnd - node.rankBegin,
			node.rankEnd - node.rankBegin,
			minSectionExtent);

		// the minimum extent can round away when added to a node boundary
		if (!(node.splitPosition > bounds.min[splitAxis] &&
			node.splitPosition < bounds.max[splitAxis]))
		{
			return;
		}

		nodeBounds[node.lowerNode] = bounds;
		nodeBounds[node.lowerNode].max[splitAxis] = node.splitPosition;
		nodeBounds[node.upperNode] = bounds;
		nodeBounds[node.upperNode].min[splitAxis] = node.splitPosition;
	}

	// rebuild owned and halo extents from new splits
	orbLayout_.Rebuild(nodes);
}
