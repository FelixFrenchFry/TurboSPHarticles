// aabb_column_layout.h
#pragma once

// external
#include <cstdint>
#include <vector>

// internal
#include "source/simulation/distributed/domain_geometry.h"



namespace Dist
{
	// section counts of a column grid, which never splits the Y axis
	struct AABBColumnCounts
	{
		int N_sectionsX = 1;
		int N_sectionsZ = 1;
	};

	// pick the section counts whose uniform column grid has the smallest total interface area
	AABBColumnCounts MakeBalancedColumnCounts(const AABB& globalBounds, int N_ranks);

	// create uniformly spaced split positions for one section row
	std::vector<float> MakeUniformColumnSplitPositions(
		float minPosition,
		float maxPosition,
		int N_sections);

	// create split positions placing equal cumulative weight into every section,
	// keeping each of them at least the minimum extent wide
	std::vector<float> MakeBalancedColumnSplitPositions(
		const std::vector<uint64_t>& weightByBin,
		float minPosition,
		float maxPosition,
		int N_sections,
		float minSectionExtent);

	// reusable AABB column geometry built from nested split positions, where the Z walls are
	// stored per X section and the Y axis is left whole, so every rank owns a full-height column
	class AABBColumnLayout final
	{
	public:

		void Build(
			const AABB& globalBounds,
			float haloRadius,
			const AABBColumnCounts& columnCounts,
			const std::vector<float>& xSplitPositions,
			const std::vector<float>& zSplitPositions);

		// re-derives the layout from new walls, keeping the state the last build was based on
		void Rebuild(
			const std::vector<float>& xSplitPositions,
			const std::vector<float>& zSplitPositions);

		const AABB& GlobalExtentAABB() const;

		int OwnerRank(const glm::vec3& position) const;

		bool IsInnerOwnedParticle(
			int rank,
			const glm::vec3& position,
			float haloRadius) const;

		const std::vector<int>& NeighborRanks(int rank) const;
		AABB OwnedExtentAABB(int rank) const;
		AABB HaloExtentAABB(int rank) const;
		int RankCount() const { return static_cast<int>(rankSubdomains_.size()); }
		const AABBColumnCounts& ColumnCounts() const { return columnCounts_; }
		float HaloRadius() const { return haloRadius_; }

		bool IsValidForRankCount(int N_ranks) const;

		// flattened nested split position addressing, shared with the repartition that fills them
		static uint64_t ZSplitRowBegin(const AABBColumnCounts& columnCounts, int sectionX);
		static int ColumnRank(
			const AABBColumnCounts& columnCounts,
			int sectionX,
			int sectionZ);

	private:

		struct AABBRankSubdomain
		{
			int rank = 0;
			AABB ownedBounds; // domain bounds owned by this rank
			AABB haloBounds; // owned bounds expanded by halo radius
			std::vector<int> neighborRanks; // ranks whose owned bounds overlap this halo
		};

		const AABBRankSubdomain& Subdomain(int rank) const;

		std::vector<AABBRankSubdomain> BuildColumnSubdomains(
			const AABB& globalBounds,
			float haloRadius,
			const AABBColumnCounts& columnCounts,
			const std::vector<float>& xSplitPositions,
			const std::vector<float>& zSplitPositions) const;

		AABBColumnCounts columnCounts_{};
		AABB globalBounds_;
		std::vector<AABBRankSubdomain> rankSubdomains_;
		std::vector<float> xSplitPositions_;
		std::vector<float> zSplitPositions_;
		float haloRadius_ = 0.0f;
	};

} // namespace Dist
