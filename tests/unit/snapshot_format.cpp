// snapshot_format.cpp

// external
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

// internal
#include "source/eval/serial/snapshot_format.h"



namespace
{
	// empty snapshots contain only the header, with no space reserved for absent arrays
	TEST(SnapshotFormatTests, EmptyPopulationsContainOnlyHeader)
	{
		const Export::ExportSnapshotBinInfo info;
		const auto offsets = Export::ComputeExportSnapshotOffsets(info);
		const auto boundary = Export::ComputeStaticBoundaryOffsets(info);

		EXPECT_EQ(offsets.posF, 64u);
		EXPECT_EQ(offsets.vF, 64u);
		EXPECT_EQ(offsets.rankF, 64u);
		EXPECT_EQ(offsets.posR, 64u);
		EXPECT_EQ(offsets.volR, 64u);
		EXPECT_EQ(offsets.bodyIdR, 64u);
		EXPECT_EQ(offsets.totalBytes, 64u);
		EXPECT_EQ(boundary.posB, 64u);
		EXPECT_EQ(boundary.volB, 64u);
		EXPECT_EQ(boundary.isMeshB, 64u);
		EXPECT_EQ(boundary.totalBytes, 64u);
	}

	// packs two fluid particles, two rank IDs, and three rigid particles in wire order
	TEST(SnapshotFormatTests, ComputesMixedPopulationOffsets)
	{
		Export::ExportSnapshotBinInfo info;
		info.N_F = 2;
		info.N_rankF = 2;
		info.N_R = 3;

		const auto offsets = Export::ComputeExportSnapshotOffsets(info);

		EXPECT_EQ(offsets.posF, 64u);
		EXPECT_EQ(offsets.vF, 88u);
		EXPECT_EQ(offsets.rankF, 96u);
		EXPECT_EQ(offsets.posR, 100u);
		EXPECT_EQ(offsets.volR, 136u);
		EXPECT_EQ(offsets.bodyIdR, 148u);
		EXPECT_EQ(offsets.totalBytes, 160u);
	}

	// omitting rank IDs moves the rigid fields directly after the fluid fields
	TEST(SnapshotFormatTests, OmitsAbsentRankStorage)
	{
		Export::ExportSnapshotBinInfo info;
		info.N_F = 2;
		info.N_R = 3;

		const auto offsets = Export::ComputeExportSnapshotOffsets(info);

		EXPECT_EQ(offsets.posR, 96u);
		EXPECT_EQ(offsets.volR, 132u);
		EXPECT_EQ(offsets.bodyIdR, 144u);
		EXPECT_EQ(offsets.totalBytes, 156u);
	}

	// packs boundary positions, volumes, and one-byte mesh flags without gaps
	TEST(SnapshotFormatTests, ComputesStaticBoundaryOffsets)
	{
		Export::ExportSnapshotBinInfo info;
		info.N_B = 5;

		const auto offsets = Export::ComputeStaticBoundaryOffsets(info);

		EXPECT_EQ(offsets.posB, 64u);
		EXPECT_EQ(offsets.volB, 124u);
		EXPECT_EQ(offsets.isMeshB, 144u);
		EXPECT_EQ(offsets.totalBytes, 149u);
	}

	// retains offsets above 4 GiB without allocating particle arrays
	TEST(SnapshotFormatTests, KeepsOffsetsWiderThan32Bits)
	{
		Export::ExportSnapshotBinInfo info;
		info.N_F = 268'435'456;

		const auto offsets = Export::ComputeExportSnapshotOffsets(info);

		EXPECT_EQ(offsets.vF, 3'221'225'536ull);
		EXPECT_EQ(offsets.posR, 4'294'967'360ull);
		EXPECT_EQ(offsets.totalBytes, 4'294'967'360ull);
	}

	// rejects counts whose individual fields overflow the byte-size representation
	TEST(SnapshotFormatTests, RejectsFieldSizeOverflow)
	{
		Export::ExportSnapshotBinInfo info;
		info.N_F = std::numeric_limits<uint64_t>::max();
		info.N_B = std::numeric_limits<uint64_t>::max();

		EXPECT_THROW(Export::ComputeExportSnapshotOffsets(info), std::runtime_error);
		EXPECT_THROW(Export::ComputeStaticBoundaryOffsets(info), std::runtime_error);
	}

	// rejects a combined snapshot size that overflows even though each field fits
	TEST(SnapshotFormatTests, RejectsCombinedSizeOverflow)
	{
		Export::ExportSnapshotBinInfo info;
		info.N_F = std::numeric_limits<uint64_t>::max() / 16;

		EXPECT_THROW(Export::ComputeExportSnapshotOffsets(info), std::runtime_error);
	}

	// computes per-population totals and rank offsets for uneven and empty partitions
	TEST(SnapshotFormatTests, ComputesDistributedPopulationLayout)
	{
		const std::vector<uint64_t> rankCounts{ 2, 1, 0, 0, 3, 1, 4, 0, 2 };
		const std::array<uint64_t, 3> expectedTotals{ 6, 4, 3 };
		const std::array<uint64_t, 3> expectedMaxima{ 4, 3, 2 };
		const std::array<std::array<uint64_t, 3>, 3> expectedBegins = {{
			{ 0, 0, 0 }, { 2, 1, 0 }, { 2, 4, 1 }
		}};

		for (int rank = 0; rank < 3; rank++)
		{
			SCOPED_TRACE(rank);
			const auto layout = Export::ComputeDistributedExportLayout(rankCounts, rank);

			EXPECT_EQ(layout.N_total, expectedTotals);
			EXPECT_EQ(layout.N_max, expectedMaxima);
			EXPECT_EQ(layout.beginIdx, expectedBegins[rank]);
		}
	}

	// rejects missing populations and ranks outside the supplied count table
	TEST(SnapshotFormatTests, RejectsMalformedDistributedTables)
	{
		EXPECT_THROW(Export::ComputeDistributedExportLayout({}, 0), std::runtime_error);
		EXPECT_THROW(Export::ComputeDistributedExportLayout({ 1, 2 }, 0), std::runtime_error);
		EXPECT_THROW(Export::ComputeDistributedExportLayout({ 1, 2, 3 }, -1), std::runtime_error);
		EXPECT_THROW(Export::ComputeDistributedExportLayout({ 1, 2, 3 }, 1), std::runtime_error);
	}

	// accepts the largest rank ID representable by the 16-bit snapshot field
	TEST(SnapshotFormatTests, AcceptsMaximumRankId)
	{
		const std::vector<uint64_t> rankCounts(3u * 65'536u, 0);

		EXPECT_NO_THROW(Export::ComputeDistributedExportLayout(rankCounts, 65'535));
	}

	// rejects rank tables that exceed the 16-bit rank-ID range
	TEST(SnapshotFormatTests, RejectsTooManyRanks)
	{
		const std::vector<uint64_t> rankCounts(3u * 65'537u, 0);

		EXPECT_THROW(Export::ComputeDistributedExportLayout(rankCounts, 0), std::runtime_error);
	}

	// rejects rank-local counts above the signed 32-bit particle limit
	TEST(SnapshotFormatTests, RejectsOversizedRankCounts)
	{
		EXPECT_THROW(Export::ComputeDistributedExportLayout({ 2'147'483'648ull, 0, 0 }, 0),
			std::length_error);
	}
}
