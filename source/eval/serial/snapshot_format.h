// snapshot_format.h
#pragma once

// external
#include <array>
#include <cstdint>
#include <string>
#include <vector>



namespace Export
{
	// header-only metadata for rebuilding XDMF time series
	struct ExportSnapshotBinInfo
	{
		std::string fileName;

		uint64_t step = 0;
		double time = 0.0;

		uint64_t N_F = 0;
		uint64_t N_B = 0;
		uint64_t N_R = 0;
		uint64_t N_rankF = 0;
	};

	#pragma pack(push, 1)
	struct BinHeaderSnap
	{
		char magic[8];
		uint32_t endianTag;
		uint32_t headerBytes;
		uint64_t step;
		double time;
		uint64_t N_F;
		uint64_t N_B;
		uint64_t N_R;
		uint64_t N_rankF;
	};
	#pragma pack(pop)

	static_assert(sizeof(BinHeaderSnap) == 64);

	struct ExportSnapshotOffsets
	{
		uint64_t posF = 0;
		uint64_t vF = 0;
		uint64_t rankF = 0;

		uint64_t posR = 0;
		uint64_t volR = 0;
		uint64_t bodyIdR = 0;

		// total size implied by the arrays above
		uint64_t totalBytes = 0;
	};

	struct StaticBoundaryOffsets
	{
		uint64_t posB = 0;
		uint64_t volB = 0;

		uint64_t totalBytes = 0;
	};

	// global counts, rank offsets, and maximum local counts in F/B/R order
	struct DistributedExportLayout
	{
		std::array<uint64_t, 3> N_total{};
		std::array<uint64_t, 3> beginIdx{};
		std::array<uint64_t, 3> N_max{};
	};

	ExportSnapshotOffsets ComputeExportSnapshotOffsets(const ExportSnapshotBinInfo& info);
	StaticBoundaryOffsets ComputeStaticBoundaryOffsets(const ExportSnapshotBinInfo& info);
	BinHeaderSnap MakeExportSnapshotHeader(const ExportSnapshotBinInfo& info);
	DistributedExportLayout ComputeDistributedExportLayout(const std::vector<uint64_t>& rankCounts, int rank);

} // namespace Export
