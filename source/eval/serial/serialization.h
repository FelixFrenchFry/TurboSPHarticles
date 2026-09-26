// serialization.h
#pragma once

// external
#include <cstdint>
#include <filesystem>
#include <functional>
#include <glm/vec3.hpp>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

// internal
#include "snapshot_format.h"
#include "state_format.h"
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

	struct ExportSnapshotFilePaths
	{
		std::filesystem::path path;
		std::filesystem::path writePath;
	};

	// position spans are valid only for the duration of the scan callback
	struct SimStatePositionView
	{
		std::span<const float> x;
		std::span<const float> y;
		std::span<const float> z;
	};

	class SerializationMgr
	{
	public:

		explicit SerializationMgr(SaveOptions opt = {})
			: saveOptions_(std::move(opt)) {}

		const std::filesystem::path& BaseDir() const { return saveOptions_.baseDir; }

		// lossless serialization of the full simulation state into a file
		// (relative paths are resolved below SaveOptions::baseDir)
		void SaveUniversalSimState(const Sync::UniversalSimState& state,
			const std::string& path) const;

		// lossless de-serialization of the full simulation state from a file
		// (relative paths are resolved below SaveOptions::baseDir)
		void LoadUniversalSimState(Sync::UniversalSimState& state,
			const std::string& path) const;

		// scans one disjoint file partition in bounded world-position blocks for each particle type
		BinHeaderSim VisitSimStatePositions(
			const std::string& path,
			int partIdx,
			int N_parts,
			const std::function<void(Base::ParticleType, const SimStatePositionView&)>& visit) const;

		// reads selected particles in bounded blocks and retains all body state
		// selectRigid overrides select for rigid particles, e.g. to include halos
		// returned header counts describe the global file
		BinHeaderSim LoadSelectedSimState(
			Sync::UniversalSimState& state,
			const std::string& path,
			const std::function<bool(const glm::vec3&)>& select,
			const std::function<bool(const glm::vec3&)>& selectRigid = {}) const;

		// serialization of a subset of the simulation state and metadata into separate files
		void SaveExportSnapshot(
			const Sync::ExportSnapshot& snap,
			const std::string& dir) const;
		void SaveExportSnapshotXdmfViews(const Sync::ExportSnapshot& snap, const std::string& dir) const;

		// reserve a fresh run directory below the configured export directory
		std::string CreateExportDirectory(const std::string& dir) const;

		// distributed export metadata, written by the root rank
		// publish views after the shared particle files are complete
		ExportSnapshotFilePaths GetDistributedExportFilePaths(
			const std::string& dir,
			uint64_t step,
			bool staticBoundary) const;
		// validate the completed file size after all writers have closed it
		void PublishDistributedExportFile(
			const ExportSnapshotFilePaths& paths, uint64_t N_expectedBytes) const;
		void SaveDistributedExportSnapshotXdmfViews(
			uint64_t step,
			double time,
			const std::vector<uint64_t>& rankCounts,
			const std::string& dir) const;

		// the next export rebuilds everything memorized about the export directory
		void ResetExportDirState()
		{
			std::scoped_lock lock(exportDirStateMutex_);
			staticBoundaryDir_.clear();
			exportSnapshotInfos_.clear();
			distributedExportSnapshotInfos_.clear();
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

		// shared snapshot metadata, keyed by simulation step
		mutable std::filesystem::path distributedExportDir_;
		mutable std::map<uint64_t, ExportSnapshotBinInfo> distributedExportSnapshotInfos_;

		// helpers
		void SaveStaticBoundaryBin(
			const Sync::ExportSnapshot& snap,
			const std::filesystem::path& dirPath) const;
	};

} // namespace Export
