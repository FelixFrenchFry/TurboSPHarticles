// serialization.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>
#include <string>

// internal
#include "Source/Sim/Sync/snapshots.h"
#include "Source/Sim/Sync/state_sim.h"



namespace Export
{
	struct SaveOptions
	{
		// compression
		int compressionLevel		= 3;    // [0,...,9], 0 = none, 9 = max
		bool compress				= true; // zlib compression (non-contiguous chunks)
		bool shuffleFilter			= true; // byte shuffle for better compression ratio on FP data

		// chunk target size
		int64_t targetChunkBytes	= 1u << 22; // 4 MiB (2^22 bytes)

		// safer saves
		// (write to temp file and rename on success)
		std::string tempSuffix		= ".tmp";
		bool atomicWrite			= true;
	};

	class SerializationMgr
	{
	public: // -----------------------------------------------------------------

		explicit SerializationMgr(const SaveOptions& opt = {})
			: saveOptions_(opt){}

		// lossless, versioned serialization of the full simulation state into a file
		void SaveUniversalSimState(const std::string& path,
								   const Sync::UniversalSimState& state);

		// lossless, versioned de-serialization of the full simulation state from a file
		void LoadUniversalSimState(const std::string& path,
								   Sync::UniversalSimState& state);

		// serialization of a subset of the simulation state and metadata into separate files
		void SaveExportSnapshot(const Sync::ExportSnapshot& snap);
		void SaveExportSnapXdmfInfo(const Sync::ExportSnapshot& snap);

	private: // ----------------------------------------------------------------

		SaveOptions saveOptions_;
	};

} // namespace Export
