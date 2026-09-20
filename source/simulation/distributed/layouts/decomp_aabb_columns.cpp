// decomp_aabb_columns.cpp

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
#include "decomp_aabb_columns.h"



using namespace Dist;

namespace
{
	// half-open profile bin ranges of the region a split position row is balanced within
	struct ProfileBinRegion
	{
		uint32_t beginX = 0;
		uint32_t endX = 0;
		uint32_t beginZ = 0;
		uint32_t endZ = 0;
	};

	uint64_t ProfileBinIdx(
		const uint32_t binX,
		const uint32_t binZ,
		const uint32_t binCountPerAxis)
	{
		return static_cast<uint64_t>(binX) * binCountPerAxis + binZ;
	}

	// profile bin containing the coordinate, matching how owned particles are binned
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

	// half-open profile bin range covered by one section of a split position row
	void SetProfileBinRange(
		const std::vector<float>& splitPositions,
		const int sectionIdx,
		const int N_sections,
		const float axisMin,
		const float axisExtent,
		const uint32_t binCountPerAxis,
		uint32_t& beginBin,
		uint32_t& endBin)
	{
		beginBin = ProfileBinAtPosition(
			splitPositions[static_cast<uint64_t>(sectionIdx)],
			axisMin, axisExtent, binCountPerAxis);

		// the last section owns every remaining bin, including the one holding the global wall
		endBin = sectionIdx + 1 == N_sections
			? binCountPerAxis
			: ProfileBinAtPosition(
				splitPositions[static_cast<uint64_t>(sectionIdx) + 1],
				axisMin, axisExtent, binCountPerAxis);

		// a section thinner than one bin still covers the bin it starts in
		endBin = std::max(endBin, beginBin + 1);
	}

	// sums the profile bins of one region into the per-bin weights of the balanced axis
	void AccumulateAxisWeights(
		const std::span<const uint64_t> weightByBin,
		const uint32_t binCountPerAxis,
		const ProfileBinRegion& region,
		const bool balanceAxisX,
		std::vector<uint64_t>& axisWeightByBin)
	{
		axisWeightByBin.assign(binCountPerAxis, 0);

		for (uint32_t binX = region.beginX; binX != region.endX; binX++)
		{
			for (uint32_t binZ = region.beginZ; binZ != region.endZ; binZ++)
			{
				const uint32_t axisBin = balanceAxisX ? binX : binZ;

				axisWeightByBin[axisBin] +=
					weightByBin[ProfileBinIdx(binX, binZ, binCountPerAxis)];
			}
		}
	}

	void AccumulateOwnedParticleCounts(
		const DomainDecompParticlePositionView& particles,
		const AABB& globalBounds,
		const uint32_t binCountPerAxis,
		std::vector<uint64_t>& profile,
		const char* particleType)
	{
		// only the split axes are binned, because the Y axis is never split
		const glm::vec3 extents = globalBounds.Extents();
		if (!std::isfinite(extents.x) || extents.x <= 0.0f ||
			!std::isfinite(extents.z) || extents.z <= 0.0f)
		{
			throw std::runtime_error(
				"Cannot build a column AABB repartition profile from an invalid global extent.");
		}

		for (const uint32_t localIdx : particles.ownedLocalIdx)
		{
			if (localIdx >= particles.x.size() || localIdx >= particles.z.size())
			{
				throw std::runtime_error(
					std::string("Cannot build a column AABB repartition profile with an invalid ") +
					particleType + " particle index.");
			}

			const float xCoordinate = particles.x[localIdx];
			const float zCoordinate = particles.z[localIdx];
			if (!std::isfinite(xCoordinate) || !std::isfinite(zCoordinate))
			{
				throw std::runtime_error(
					std::string("Cannot build a column AABB repartition profile from an invalid ") +
					particleType + " particle position.");
			}

			// particles that left the global domain count towards the nearest bin
			const uint32_t binX = ProfileBinAtPosition(
				xCoordinate, globalBounds.min.x, extents.x, binCountPerAxis);
			const uint32_t binZ = ProfileBinAtPosition(
				zCoordinate, globalBounds.min.z, extents.z, binCountPerAxis);

			uint64_t& binCount = profile[ProfileBinIdx(binX, binZ, binCountPerAxis)];
			if (binCount == std::numeric_limits<uint64_t>::max())
			{
				throw std::runtime_error(
					"Cannot build a column AABB repartition profile with an overflowing bin.");
			}
			binCount += 1;
		}
	}

	// repeat one uniform split position row for every section row on that axis
	std::vector<float> MakeUniformSplitPositionRows(
		const float minPosition,
		const float maxPosition,
		const int N_sections,
		const int N_rows)
	{
		const std::vector<float> splitRow =
			MakeUniformColumnSplitPositions(minPosition, maxPosition, N_sections);

		std::vector<float> splitPositions{};
		splitPositions.reserve(static_cast<uint64_t>(N_rows) * splitRow.size());
		for (int rowIdx = 0; rowIdx < N_rows; rowIdx++)
		{
			splitPositions.insert(splitPositions.end(), splitRow.begin(), splitRow.end());
		}

		return splitPositions;
	}

} // namespace anonymous

void Decomp_AABB_Columns::Build(const DomainDecompInput& input)
{
	if (!input.globalBounds.IsValid())
	{
		throw std::runtime_error(
			"Cannot build Decomp_AABB_Columns with invalid global domain bounds.");
	}
	if (input.haloRadius <= 0.0f)
	{
		throw std::runtime_error(
			"Cannot build Decomp_AABB_Columns with a non-positive halo radius.");
	}
	if (input.N_ranks <= 0)
	{
		throw std::runtime_error(
			"Cannot build Decomp_AABB_Columns for non-positive rank count.");
	}

	// column grid derived identically on every rank
	const AABBColumnCounts columnCounts =
		MakeBalancedColumnCounts(input.globalBounds, input.N_ranks);

	// split sections thinner than the halo radius reach past their direct neighbors
	const glm::vec3 extents = input.globalBounds.Extents();
	if ((columnCounts.N_sectionsX > 1 &&
		extents.x < static_cast<float>(columnCounts.N_sectionsX) * input.haloRadius) ||
		(columnCounts.N_sectionsZ > 1 &&
		extents.z < static_cast<float>(columnCounts.N_sectionsZ) * input.haloRadius))
	{
		throw std::runtime_error(
			"Cannot build Decomp_AABB_Columns with sections thinner than the halo radius.");
	}

	// initially built uniformly sized AABB columns (one per rank)
	columnLayout_.Build(
		input.globalBounds,
		input.haloRadius,
		columnCounts,
		MakeUniformColumnSplitPositions(
			input.globalBounds.min.x, input.globalBounds.max.x,
			columnCounts.N_sectionsX),
		MakeUniformSplitPositionRows(
			input.globalBounds.min.z, input.globalBounds.max.z,
			columnCounts.N_sectionsZ, columnCounts.N_sectionsX));
}

const AABB& Decomp_AABB_Columns::GlobalExtentAABB() const
{
	return columnLayout_.GlobalExtentAABB();
}

int Decomp_AABB_Columns::OwnerRank(const glm::vec3& position) const
{
	return columnLayout_.OwnerRank(position);
}

bool Decomp_AABB_Columns::IsInnerOwnedParticle(
	const int rank,
	const glm::vec3& position,
	const float haloRadius) const
{
	return columnLayout_.IsInnerOwnedParticle(rank, position, haloRadius);
}

const std::vector<int>& Decomp_AABB_Columns::NeighborRanks(const int rank) const
{
	return columnLayout_.NeighborRanks(rank);
}

AABB Decomp_AABB_Columns::OwnedExtentAABB(const int rank) const
{
	return columnLayout_.OwnedExtentAABB(rank);
}

AABB Decomp_AABB_Columns::HaloExtentAABB(const int rank) const
{
	return columnLayout_.HaloExtentAABB(rank);
}

bool Decomp_AABB_Columns::IsValidForRankCount(const int N_ranks) const
{
	return columnLayout_.IsValidForRankCount(N_ranks);
}

void Decomp_AABB_Columns::BuildRepartitionProfile(
	const DomainDecompRepartitionParticleInput& input,
	DomainDecompRepartitionProfile& profile) const
{
	if (input.weightModel != Base::LoadBalanceWeightModel::OwnedParticleCount)
	{
		throw std::runtime_error(
			"Decomp_AABB_Columns does not support the requested repartition weight model.");
	}

	const int N_ranks = columnLayout_.RankCount();
	if (input.rank < 0 || input.rank >= N_ranks)
	{
		throw std::runtime_error(
			"Cannot build a column AABB repartition profile for an invalid rank.");
	}

	// one fixed-resolution histogram is kept over the global domain
	profile.ownedFluidByUnit.assign(PROFILE_BIN_COUNT, 0);
	profile.ownedBoundaryByUnit.assign(PROFILE_BIN_COUNT, 0);

	const AABB globalBounds = columnLayout_.GlobalExtentAABB();
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
}

void Decomp_AABB_Columns::Repartition(const DomainDecompRepartitionInput& input)
{
	if (input.weightModel != Base::LoadBalanceWeightModel::OwnedParticleCount)
	{
		throw std::runtime_error(
			"Decomp_AABB_Columns does not support the requested repartition weight model.");
	}

	// validate rank loads against the current column layout
	const int N_ranks = static_cast<int>(input.loadByRank.size());
	if (!columnLayout_.IsValidForRankCount(N_ranks))
	{
		throw std::runtime_error(
			"Cannot repartition Decomp_AABB_Columns with a rank load count that "
			"does not match the current layout.");
	}

	// validate rank loads supplied to the trigger
	for (const double load : input.loadByRank)
	{
		if (!std::isfinite(load) || load < 0.0)
		{
			throw std::runtime_error(
				"Cannot repartition Decomp_AABB_Columns with invalid rank loads.");
		}
	}

	if (input.ownedFluidByUnit.size() != PROFILE_BIN_COUNT ||
		input.ownedBoundaryByUnit.size() != PROFILE_BIN_COUNT)
	{
		throw std::runtime_error(
			"Cannot repartition Decomp_AABB_Columns with an invalid particle weight profile.");
	}

	const AABB globalBounds = columnLayout_.GlobalExtentAABB();
	const glm::vec3 extents = globalBounds.Extents();
	const AABBColumnCounts columnCounts = columnLayout_.ColumnCounts();
	const float minSectionExtent = columnLayout_.HaloRadius();

	// TODO: restore a boundary-particle weight when boundary work is modeled?
	const std::span<const uint64_t> weightByBin = input.ownedFluidByUnit;

	// an empty profile carries no balancing information, so the current layout is kept
	uint64_t totalWeight = 0;
	for (const uint64_t binWeight : weightByBin) { totalWeight += binWeight; }
	if (totalWeight == 0) { return; }

	const ProfileBinRegion globalRegion
	{
		.beginX = 0, .endX = PROFILE_BIN_COUNT_PER_AXIS,
		.beginZ = 0, .endZ = PROFILE_BIN_COUNT_PER_AXIS
	};
	std::vector<uint64_t> axisWeightByBin{};

	// place the global X walls from the profile marginalized onto X
	AccumulateAxisWeights(
		weightByBin, PROFILE_BIN_COUNT_PER_AXIS, globalRegion, true, axisWeightByBin);
	const std::vector<float> xSplitPositions = MakeBalancedColumnSplitPositions(
		axisWeightByBin,
		globalBounds.min.x, globalBounds.max.x,
		columnCounts.N_sectionsX, minSectionExtent);

	std::vector<float> zSplitPositions{};
	zSplitPositions.reserve(
		static_cast<uint64_t>(columnCounts.N_sectionsX) *
		(static_cast<uint64_t>(columnCounts.N_sectionsZ) + 1));

	// place the Z walls within every new X section
	for (int sectionX = 0; sectionX < columnCounts.N_sectionsX; sectionX++)
	{
		ProfileBinRegion sectionRegion = globalRegion;
		SetProfileBinRange(
			xSplitPositions, sectionX, columnCounts.N_sectionsX,
			globalBounds.min.x, extents.x, PROFILE_BIN_COUNT_PER_AXIS,
			sectionRegion.beginX, sectionRegion.endX);

		AccumulateAxisWeights(
			weightByBin, PROFILE_BIN_COUNT_PER_AXIS, sectionRegion, false, axisWeightByBin);
		const std::vector<float> zSplitRow = MakeBalancedColumnSplitPositions(
			axisWeightByBin,
			globalBounds.min.z, globalBounds.max.z,
			columnCounts.N_sectionsZ, minSectionExtent);
		zSplitPositions.insert(zSplitPositions.end(), zSplitRow.begin(), zSplitRow.end());
	}

	// rebuild owned and halo extents from new walls
	columnLayout_.Rebuild(xSplitPositions, zSplitPositions);
}
