// serialization.cpp

// external
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// internal
#include "serialization.h"



namespace Export
{
	// binary format helpers
	static constexpr uint32_t kEndianLE = 0x01020304u;

	static bool IsLittleEndian()
	{
		const uint32_t x = 0x01020304u;
		return *reinterpret_cast<const uint8_t*>(&x) == 0x04u;
	}

	#pragma pack(push, 1)
	struct BinHeaderSimCommon
	{
		char magic[8];        // "TSPSIM\1"
		uint32_t endianTag;   // kEndianLE
		uint32_t headerBytes; // sizeof(BinHeaderSim)
	};

	struct BinHeaderSimV1
	{
		char magic[8];        // "TSPSIM\1"
		uint32_t endianTag;   // kEndianLE
		uint32_t headerBytes; // sizeof(BinHeaderSimV1)

		uint32_t schemaVersion;
		uint32_t step;
		double time;

		uint32_t NF;
		uint32_t NB;
	};

	struct BinHeaderSim
	{
		char magic[8];        // "TSPSIM\1"
		uint32_t endianTag;   // kEndianLE
		uint32_t headerBytes; // sizeof(BinHeaderSim)

		uint64_t schemaVersion;
		uint32_t step;
		double time;

		uint32_t NF;
		uint32_t NB;
	};

	struct BinHeaderSnap
	{
		char magic[8];        // "TSPSNAP\2"
		uint32_t endianTag;   // kEndianLE
		uint32_t headerBytes; // sizeof(BinHeaderSnap)

		uint32_t step;
		double time;

		uint32_t NF;
		uint32_t NB;
	};
	#pragma pack(pop)

	static void EnsureStreamOk(const std::ios& s, const char* what, const std::string& path)
	{
		if (!s.good())
		{
			SPDLOG_ERROR("I/O error while {}: {}", what, path);
			throw std::runtime_error("I/O error");
		}
	}

	static void WriteExact(std::ofstream& ofs, const void* data, size_t bytes)
	{
		ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
	}

	static void ReadExact(std::ifstream& ifs, void* data, size_t bytes)
	{
		ifs.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
	}

	template <class T>
	static void WritePod(std::ofstream& ofs, const T& v)
	{
		static_assert(std::is_trivially_copyable_v<T>, "WritePod requires trivially copyable type");
		WriteExact(ofs, &v, sizeof(T));
	}

	template <class T>
	static void ReadPod(std::ifstream& ifs, T& v)
	{
		static_assert(std::is_trivially_copyable_v<T>, "ReadPod requires trivially copyable type");
		ReadExact(ifs, &v, sizeof(T));
	}

	template <class T>
	static void WriteVec(std::ofstream& ofs, const std::vector<T>& v)
	{
		static_assert(std::is_trivially_copyable_v<T>, "WriteVec requires trivially copyable element type");
		if (!v.empty())
		{
			WriteExact(ofs, v.data(), v.size() * sizeof(T));
		}
	}

	template <class T>
	static void ReadVec(std::ifstream& ifs, std::vector<T>& v)
	{
		static_assert(std::is_trivially_copyable_v<T>, "ReadVec requires trivially copyable element type");
		if (!v.empty())
		{
			ReadExact(ifs, v.data(), v.size() * sizeof(T));
		}
	}

	// --- [ small utilities ] ---------------------------------------------------------------------

	static void AtomicReplace(const std::filesystem::path& tmp,
		const std::filesystem::path& dst)
	{
		std::error_code ec;

		// try to rename directly
		std::filesystem::rename(tmp, dst, ec);
		if (!ec) { return; }

		// remove existing then rename as fallback
		std::filesystem::remove(dst, ec);
		std::filesystem::rename(tmp, dst);
	}

	static std::string FormatStepPadded(uint32_t step, uint32_t width = 9)
	{
		std::ostringstream oss;
		oss << std::setfill('0') << std::setw(width) << step;
		return oss.str();
	}

	static void CheckFluidConsistent(const Sync::FluidParticles& F)
	{
		[[maybe_unused]] const auto N = static_cast<size_t>(F.N_);

		assert(F.xPos.size() == N && F.yPos.size() == N && F.zPos.size() == N);

		assert(F.xVel.size() == N && F.yVel.size() == N && F.zVel.size() == N);
		assert(F.xVelPred.size() == N && F.yVelPred.size() == N && F.zVelPred.size() == N);

		assert(F.xAccPress.size() == N && F.yAccPress.size() == N && F.zAccPress.size() == N);

		assert(F.sourceTerm.size() == N);
		assert(F.diagElem.size() == N);

		assert(F.density.size() == N);
		assert(F.pressure.size() == N);
	}

	static void CheckBoundaryConsistent(const Sync::BoundaryParticles& B)
	{
		[[maybe_unused]] const auto N = static_cast<size_t>(B.N_);

		assert(B.xPos.size() == N && B.yPos.size() == N && B.zPos.size() == N);
		assert(B.volB.size() == N && B.psiB.size() == N);
	}

	// --- [ UniversalSimState <-> .bin ] ----------------------------------------------------------

	void SerializationMgr::SaveUniversalSimState(
		const Sync::UniversalSimState& state,
		const std::string& path) const
	{
		// file path handling
		const std::filesystem::path dstPath{ path };
		const std::filesystem::path tmpPath = saveOptions_.atomicWrite
			? std::filesystem::path(dstPath.string() + saveOptions_.tempSuffix)
			: dstPath;

		// sanity check before writing
		CheckFluidConsistent(state.fluidParticles);
		CheckBoundaryConsistent(state.boundaryParticles);

		// open scope for writer
		{
			std::ofstream ofs(tmpPath, std::ios::binary | std::ios::out | std::ios::trunc);
			if (!ofs)
			{
				SPDLOG_ERROR("Failed to open file for write: {}", tmpPath.string());
				return;
			}

			// header
			BinHeaderSim hdr{};
			std::memcpy(hdr.magic, "TSPSIM\1", 7);
			hdr.magic[7] = '\0';
			hdr.endianTag = kEndianLE;
			hdr.headerBytes = static_cast<uint32_t>(sizeof(BinHeaderSim));

			hdr.schemaVersion = state.schemaVersion;
			hdr.step = state.step;
			hdr.time = state.time;

			hdr.NF = state.fluidParticles.N_;
			hdr.NB = state.boundaryParticles.N_;

			WritePod(ofs, hdr);

			// fluid data (fixed order)
			{
				const auto& F = state.fluidParticles;

				WriteVec(ofs, F.xPos);
				WriteVec(ofs, F.yPos);
				WriteVec(ofs, F.zPos);

				WriteVec(ofs, F.xVel);
				WriteVec(ofs, F.yVel);
				WriteVec(ofs, F.zVel);

				WriteVec(ofs, F.xVelPred);
				WriteVec(ofs, F.yVelPred);
				WriteVec(ofs, F.zVelPred);

				WriteVec(ofs, F.xAccPress);
				WriteVec(ofs, F.yAccPress);
				WriteVec(ofs, F.zAccPress);

				WriteVec(ofs, F.sourceTerm);
				WriteVec(ofs, F.diagElem);
				WriteVec(ofs, F.pressure);
				WriteVec(ofs, F.density);
			}

			// boundary data (fixed order)
			{
				const auto& B = state.boundaryParticles;

				WriteVec(ofs, B.xPos);
				WriteVec(ofs, B.yPos);
				WriteVec(ofs, B.zPos);

				WriteVec(ofs, B.volB);
				WriteVec(ofs, B.psiB);
			}

			EnsureStreamOk(ofs, "writing", tmpPath.string());
		}

		if (saveOptions_.atomicWrite)
		{
			AtomicReplace(tmpPath, dstPath);
		}
	}

	void SerializationMgr::LoadUniversalSimState(
		Sync::UniversalSimState& state,
		const std::string& path) const
	{
		std::ifstream ifs(path, std::ios::binary | std::ios::in);
		if (!ifs)
		{
			SPDLOG_ERROR("Failed to open file for read: {}", path);
			return;
		}

		// common header prefix
		BinHeaderSimCommon hdrCommon{};
		ReadPod(ifs, hdrCommon);

		// basic validation
		if (std::memcmp(hdrCommon.magic, "TSPSIM\1", 7) != 0)
		{
			SPDLOG_ERROR("Invalid sim state file magic: {}", path);
			return;
		}
		if (!IsLittleEndian() || hdrCommon.endianTag != kEndianLE)
		{
			SPDLOG_ERROR("Endianness mismatch (file is LE, host must be LE): {}", path);
			return;
		}
		// parse the remaining header bytes
		uint64_t schemaVersion = 0;
		uint32_t step = 0;
		double time = 0.0;
		uint32_t NF = 0;
		uint32_t NB = 0;

		if (hdrCommon.headerBytes == sizeof(BinHeaderSim))
		{
			BinHeaderSim hdr{};
			std::memcpy(&hdr, &hdrCommon, sizeof(BinHeaderSimCommon));
			ReadExact(ifs,
				reinterpret_cast<char*>(&hdr) + sizeof(BinHeaderSimCommon),
				sizeof(BinHeaderSim) - sizeof(BinHeaderSimCommon));

			schemaVersion = hdr.schemaVersion;
			step = hdr.step;
			time = hdr.time;
			NF = hdr.NF;
			NB = hdr.NB;
		}
		else if (hdrCommon.headerBytes == sizeof(BinHeaderSimV1))
		{
			BinHeaderSimV1 hdr{};
			std::memcpy(&hdr, &hdrCommon, sizeof(BinHeaderSimCommon));
			ReadExact(ifs,
				reinterpret_cast<char*>(&hdr) + sizeof(BinHeaderSimCommon),
				sizeof(BinHeaderSimV1) - sizeof(BinHeaderSimCommon));

			schemaVersion = hdr.schemaVersion;
			step = hdr.step;
			time = hdr.time;
			NF = hdr.NF;
			NB = hdr.NB;
		}
		else
		{
			SPDLOG_ERROR("Unexpected sim header size ({}): {}",
				hdrCommon.headerBytes, path);
			return;
		}

		// metadata
		state.schemaVersion = schemaVersion;
		state.step = step;
		state.time = time;

		// resize storage
		state.ResizeFluid(NF);
		state.ResizeBoundary(NB);

		// fluid data (fixed order)
		{
			auto& F = state.fluidParticles;

			ReadVec(ifs, F.xPos);
			ReadVec(ifs, F.yPos);
			ReadVec(ifs, F.zPos);

			ReadVec(ifs, F.xVel);
			ReadVec(ifs, F.yVel);
			ReadVec(ifs, F.zVel);

			ReadVec(ifs, F.xVelPred);
			ReadVec(ifs, F.yVelPred);
			ReadVec(ifs, F.zVelPred);

			ReadVec(ifs, F.xAccPress);
			ReadVec(ifs, F.yAccPress);
			ReadVec(ifs, F.zAccPress);

			ReadVec(ifs, F.sourceTerm);
			ReadVec(ifs, F.diagElem);
			ReadVec(ifs, F.pressure);
			ReadVec(ifs, F.density);

			// sanity check
			F.N_ = NF;
		}

		// boundary data (fixed order)
		{
			auto& B = state.boundaryParticles;

			ReadVec(ifs, B.xPos);
			ReadVec(ifs, B.yPos);
			ReadVec(ifs, B.zPos);

			if (schemaVersion >= 2)
			{
				ReadVec(ifs, B.volB);
				ReadVec(ifs, B.psiB);
			}
			else
			{
				B.volB.clear();
				B.psiB.clear();
			}

			// sanity check
			B.N_ = NB;
		}

		EnsureStreamOk(ifs, "reading", path);
	}

	// --- [ ExportSnapshot (.bin) + XDMF ] --------------------------------------------------------

	void SerializationMgr::SaveExportSnapshot(
		const Sync::ExportSnapshot& snap,
		const std::string& dir) const
	{
		const std::string baseName = "pos_vel_step_" + FormatStepPadded(snap.simStep) + ".bin";

		// ensure export directory exists
		std::error_code ec;
		std::filesystem::create_directories(saveOptions_.baseDir / dir, ec);
		if (ec)
		{
			SPDLOG_ERROR("Failed to create export directory '{}': {}",
				saveOptions_.baseDir.string(), ec.message());
			return;
		}

		// make sure data for every particle is present
		const uint32_t NF_size = snap.N_F;
		const uint32_t NB_size = snap.N_B;
		assert(snap.xPosF.size() == NF_size);
		assert(snap.yPosF.size() == NF_size);
		assert(snap.zPosF.size() == NF_size);
		assert(snap.velMagF.size() == NF_size);
		assert(snap.xPosB.size() == NB_size);
		assert(snap.yPosB.size() == NB_size);
		assert(snap.zPosB.size() == NB_size);
		assert(snap.volB.size() == NB_size);

		// file path handling
		const std::filesystem::path dstPath = saveOptions_.baseDir / dir / baseName;
		const std::filesystem::path tmpPath = saveOptions_.atomicWrite
			? std::filesystem::path(dstPath.string() + saveOptions_.tempSuffix)
			: dstPath;

		// open scope for writer
		{
			std::ofstream ofs(tmpPath, std::ios::binary | std::ios::out | std::ios::trunc);
			if (!ofs)
			{
				SPDLOG_ERROR("Failed to open snapshot file for write: {}",
					tmpPath.string());
				return;
			}

			// arrays are written in the same order as referenced by the XDMF file:
			// xPosF, yPosF, zPosF, velMagF, xPosB, yPosB, zPosB, volB

			BinHeaderSnap hdr{};
			std::memcpy(hdr.magic, "TSPSNAP\2", 8);
			hdr.endianTag = kEndianLE;
			hdr.headerBytes = static_cast<uint32_t>(sizeof(BinHeaderSnap));
			hdr.step = snap.simStep;
			hdr.time = snap.simTime;
			hdr.NF = NF_size;
			hdr.NB = NB_size;

			WritePod(ofs, hdr);

			// arrays (fixed order)
			WriteVec(ofs, snap.xPosF);
			WriteVec(ofs, snap.yPosF);
			WriteVec(ofs, snap.zPosF);
			WriteVec(ofs, snap.velMagF);

			WriteVec(ofs, snap.xPosB);
			WriteVec(ofs, snap.yPosB);
			WriteVec(ofs, snap.zPosB);
			WriteVec(ofs, snap.volB);

			EnsureStreamOk(ofs, "writing", tmpPath.string());
		}

		if (saveOptions_.atomicWrite)
		{
			AtomicReplace(tmpPath, dstPath);
		}
	}

	void SerializationMgr::SaveExportSnapXdmfInfo(
		const Sync::ExportSnapshot& snap,
		const std::string& dir) const
	{
		const std::string baseName_bin = "pos_vel_step_" + FormatStepPadded(snap.simStep) + ".bin";
		const std::string baseName_xdmf = "pos_vel_step_" + FormatStepPadded(snap.simStep) + ".xdmf";

		// ensure export directory exists
		std::error_code ec;
		std::filesystem::create_directories(saveOptions_.baseDir / dir, ec);
		if (ec)
		{
			SPDLOG_ERROR("Failed to create export directory '{}': {}",
				saveOptions_.baseDir.string(), ec.message());
			return;
		}

		const std::filesystem::path xdmfPath = saveOptions_.baseDir / dir / baseName_xdmf;

		// open XDMF file
		std::ofstream ofs(xdmfPath, std::ios::out | std::ios::trunc);
		if (!ofs)
		{
			SPDLOG_ERROR("Failed to open XDMF file for write: {}",
				xdmfPath.string());
			return;
		}

		// make sure data for every particle is present
		const uint32_t NF_size = snap.N_F;
		const uint32_t NB_size = snap.N_B;
		assert(snap.xPosF.size() == NF_size);
		assert(snap.yPosF.size() == NF_size);
		assert(snap.zPosF.size() == NF_size);
		assert(snap.velMagF.size() == NF_size);
		assert(snap.xPosB.size() == NB_size);
		assert(snap.yPosB.size() == NB_size);
		assert(snap.zPosB.size() == NB_size);
		assert(snap.volB.size() == NB_size);

		// byte offsets (Seek) into the .bin file
		//
		// layout:
		// [BinHeaderSnap]
		// xPosF[NF] (float32)
		// yPosF[NF] (float32)
		// zPosF[NF] (float32)
		// velMagF[NF] (float32)
		// xPosB[NB] (float32)
		// yPosB[NB] (float32)
		// zPosB[NB] (float32)
		// volB[NB] (float32)
		const uint64_t headerBytes = sizeof(BinHeaderSnap);
		const uint64_t elemBytes = 4; // float32

		const uint64_t off_xF = headerBytes;
		const uint64_t off_yF = off_xF + uint64_t(NF_size) * elemBytes;
		const uint64_t off_zF = off_yF + uint64_t(NF_size) * elemBytes;
		const uint64_t off_vF = off_zF + uint64_t(NF_size) * elemBytes;
		const uint64_t off_xB = off_vF + uint64_t(NF_size) * elemBytes;
		const uint64_t off_yB = off_xB + uint64_t(NB_size) * elemBytes;
		const uint64_t off_zB = off_yB + uint64_t(NB_size) * elemBytes;
		const uint64_t off_volB = off_zB + uint64_t(NB_size) * elemBytes;

		// XDMF header
		ofs << R"(<?xml version="1.0" ?>)" << "\n";
		ofs << R"(<Xdmf Version="3.0">)" << "\n";
		ofs << R"(  <Domain>)" << "\n";

		// one uniform grid for exported fluid particles
		ofs << "    <Grid Name=\"Step_" << snap.simStep
			<< "_Fluid\" GridType=\"Uniform\">" << "\n";

		// time metadata
		ofs << "      <Time Value=\""
			<< std::setprecision(16) << snap.simTime
			<< "\" />\n";

		// topology of point cloud with NF vertices
		ofs << "      <Topology TopologyType=\"Polyvertex\" "
			<< "Dimensions=\"" << NF_size << "\"/>\n";

		// geometry of XYZ coordinates
		ofs << "      <Geometry GeometryType=\"XYZ\">\n";
		ofs << "        <DataItem ItemType=\"Function\" "
			<< "Function=\"JOIN($0, $1, $2)\" "
			<< "Dimensions=\"" << NF_size << " 3\" "
			<< "NumberType=\"Float\" Precision=\"4\">\n";

		// x pos (Binary)
		ofs << "          <DataItem Dimensions=\"" << NF_size
			<< "\" NumberType=\"Float\" Precision=\"4\" "
			<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
			<< off_xF << "\">\n";
		ofs << "            " << baseName_bin << "\n";
		ofs << "          </DataItem>\n";

		// y pos (Binary)
		ofs << "          <DataItem Dimensions=\"" << NF_size
			<< "\" NumberType=\"Float\" Precision=\"4\" "
			<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
			<< off_yF << "\">\n";
		ofs << "            " << baseName_bin << "\n";
		ofs << "          </DataItem>\n";

		// z pos (Binary)
		ofs << "          <DataItem Dimensions=\"" << NF_size
			<< "\" NumberType=\"Float\" Precision=\"4\" "
			<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
			<< off_zF << "\">\n";
		ofs << "            " << baseName_bin << "\n";
		ofs << "          </DataItem>\n";

		ofs << "        </DataItem>\n";
		ofs << "      </Geometry>\n";

		// velocity magnitude (Binary)
		ofs << "      <Attribute Name=\"velMagF\" "
			<< "AttributeType=\"Scalar\" Center=\"Node\">\n";
		ofs << "        <DataItem Dimensions=\"" << NF_size
			<< "\" NumberType=\"Float\" Precision=\"4\" "
			<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
			<< off_vF << "\">\n";
		ofs << "          " << baseName_bin << "\n";
		ofs << "        </DataItem>\n";
		ofs << "      </Attribute>\n";

		ofs << "    </Grid>\n";

		if (NB_size > 0)
		{
			// one uniform grid for exported boundary particles
			ofs << "    <Grid Name=\"Step_" << snap.simStep
				<< "_Boundary\" GridType=\"Uniform\">" << "\n";

			// time metadata
			ofs << "      <Time Value=\""
				<< std::setprecision(16) << snap.simTime
				<< "\" />\n";

			// topology of point cloud with NB vertices
			ofs << "      <Topology TopologyType=\"Polyvertex\" "
				<< "Dimensions=\"" << NB_size << "\"/>\n";

			// geometry of XYZ coordinates
			ofs << "      <Geometry GeometryType=\"XYZ\">\n";
			ofs << "        <DataItem ItemType=\"Function\" "
				<< "Function=\"JOIN($0, $1, $2)\" "
				<< "Dimensions=\"" << NB_size << " 3\" "
				<< "NumberType=\"Float\" Precision=\"4\">\n";

			// x pos (Binary)
			ofs << "          <DataItem Dimensions=\"" << NB_size
				<< "\" NumberType=\"Float\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_xB << "\">\n";
			ofs << "            " << baseName_bin << "\n";
			ofs << "          </DataItem>\n";

			// y pos (Binary)
			ofs << "          <DataItem Dimensions=\"" << NB_size
				<< "\" NumberType=\"Float\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_yB << "\">\n";
			ofs << "            " << baseName_bin << "\n";
			ofs << "          </DataItem>\n";

			// z pos (Binary)
			ofs << "          <DataItem Dimensions=\"" << NB_size
				<< "\" NumberType=\"Float\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_zB << "\">\n";
			ofs << "            " << baseName_bin << "\n";
			ofs << "          </DataItem>\n";

			ofs << "        </DataItem>\n";
			ofs << "      </Geometry>\n";

			// represented boundary volume (Binary)
			ofs << "      <Attribute Name=\"volB\" "
				<< "AttributeType=\"Scalar\" Center=\"Node\">\n";
			ofs << "        <DataItem Dimensions=\"" << NB_size
				<< "\" NumberType=\"Float\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_volB << "\">\n";
			ofs << "          " << baseName_bin << "\n";
			ofs << "        </DataItem>\n";
			ofs << "      </Attribute>\n";

			ofs << "    </Grid>\n";
		}

		// close everything
		ofs << "  </Domain>\n";
		ofs << "</Xdmf>\n";
	}

} // namespace Export
