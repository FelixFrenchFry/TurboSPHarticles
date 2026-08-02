// serialization.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>
#include <filesystem>
#include <string>

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

	class SerializationMgr
	{
	public: // -------------------------------------------------------------------------------------

		explicit SerializationMgr(SaveOptions opt = {})
			: saveOptions_(std::move(opt)) {}

		// lossless serialization of the full simulation state into a file
		void SaveUniversalSimState(const Sync::UniversalSimState& state,
			const std::string& path) const;

		// lossless de-serialization of the full simulation state from a file
		void LoadUniversalSimState(Sync::UniversalSimState& state,
			const std::string& path) const;

		// serialization of a subset of the simulation state and metadata into separate files
		void SaveExportSnapshot(const Sync::ExportSnapshot& snap, const std::string& dir) const;
		void SaveExportSnapshotXdmfViews(const Sync::ExportSnapshot& snap, const std::string& dir) const;

	private: // ------------------------------------------------------------------------------------

		SaveOptions saveOptions_;
	};

} // namespace Export
