// serialization.h
#pragma once

// external
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>

// internal
#include "source/simulation/sync/snapshots.h"
#include "source/simulation/sync/state_sim.h"

namespace Export
{
	struct SaveOptions
	{
	#ifdef TSP_EXPORT_DIR
		std::filesystem::path baseDir = TSP_EXPORT_DIR;
	#else
		std::filesystem::path baseDir = "exports";
	#endif

		// write to temp file and rename on success
		std::string tempSuffix = ".tmp";
		bool atomicWrite = true;
	};

	// header-only metadata for rebuilding XDMF time series
	struct ExportSnapshotBinInfo
	{
		std::string fileName;

		uint64_t step = 0;
		double time = 0.0;

		uint32_t NF = 0;
		uint32_t NB = 0;
		uint32_t NR = 0;
		uint32_t NrankF = 0;
	};

	class SerializationMgr
	{
	public:

		explicit SerializationMgr(SaveOptions opt = {})
			: saveOptions_(std::move(opt)) {}

		// lossless serialization of the full simulation state into a file
		// (relative paths are resolved below SaveOptions::baseDir)
		void SaveUniversalSimState(const Sync::UniversalSimState& state,
			const std::string& path) const;

		// lossless de-serialization of the full simulation state from a file
		// (relative paths are resolved below SaveOptions::baseDir)
		void LoadUniversalSimState(Sync::UniversalSimState& state,
			const std::string& path) const;

		// serialization of a subset of the simulation state and metadata into separate files
		void SaveExportSnapshot(const Sync::ExportSnapshot& snap, const std::string& dir) const;
		void SaveExportSnapshotXdmfViews(const Sync::ExportSnapshot& snap, const std::string& dir) const;

		// the next export rebuilds everything memorized about the export directory
		void ResetExportDirState()
		{
			std::scoped_lock lock(exportDirStateMutex_);
			staticBoundaryDir_.clear();
			exportSnapshotInfos_.clear();
		}

	private:

		SaveOptions saveOptions_;

		// the export runs on the simulation thread, while a reset comes from the GUI thread
		mutable std::mutex exportDirStateMutex_;

		// the export directory the static boundary file was last written to
		mutable std::filesystem::path staticBoundaryDir_;

		// parsed .bin headers of the export directory, keyed by file name
		mutable std::filesystem::path exportSnapshotInfosDir_;
		mutable std::unordered_map<std::string, ExportSnapshotBinInfo> exportSnapshotInfos_;

		// helpers
		void SaveStaticBoundaryBin(
			const Sync::ExportSnapshot& snap,
			const std::filesystem::path& dirPath) const;
	};

} // namespace Export
