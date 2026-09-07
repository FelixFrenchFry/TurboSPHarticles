// serialization.cpp

// external
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

// internal
#include "serialization.h"



namespace Export
{
	// binary format helpers
	static constexpr uint32_t kEndianLE = 0x01020304u;
	static constexpr char kSimStateMagic[8] =
		{ 'T', 'S', 'P', 'S', 'I', 'M', '\0', '\0' };
	static constexpr char kExportSnapshotMagic[8] =
		{ 'T', 'S', 'P', 'S', 'N', 'A', 'P', '\0' };

	static constexpr char kStaticBoundaryBinFileName[] =
		"static_boundary.bin";

	static std::filesystem::path ResolveStatePath(
		const std::filesystem::path& baseDir,
		const std::string& path)
	{
		const std::filesystem::path requestedPath{ path };
		return requestedPath.is_absolute()
			? requestedPath
			: baseDir / requestedPath;
	}

	static bool IsLittleEndian()
	{
		const uint32_t x = 0x01020304u;
		return *reinterpret_cast<const uint8_t*>(&x) == 0x04u;
	}

	#pragma pack(push, 1)
	struct BinHeaderSim
	{
		char magic[8];        // file magic
		uint32_t endianTag;   // kEndianLE
		uint32_t headerBytes; // sizeof(file header)
		uint64_t step;
		double time;

		uint32_t NF;
		uint32_t NB;
		uint32_t NR;
		uint32_t NRB;
	};

	struct BinHeaderSnap
	{
		char magic[8];        // file magic
		uint32_t endianTag;   // kEndianLE
		uint32_t headerBytes; // sizeof(file header)
		uint64_t step;
		double time;

		uint32_t NF;
		uint32_t NB;
		uint32_t NR;
		uint32_t NrankF;      // NF in distributed runs, 0 when the ranks are not written
	};
	#pragma pack(pop)

	static void EnsureStreamOk(const std::ios& s, const char* what, const std::string& path)
	{
		if (!s.good())
		{
			throw std::runtime_error(
				std::string("I/O error while ") + what + " '" + path + "'.");
		}
	}

	static void WriteExact(std::ofstream& ofs, const void* data, size_t N_bytes)
	{
		ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(N_bytes));
	}

	static void ReadExact(std::ifstream& ifs, void* data, size_t N_bytes)
	{
		ifs.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(N_bytes));
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

	static bool TryAtomicReplace(
		const std::filesystem::path& tmp,
		const std::filesystem::path& dst,
		std::string& error)
	{
	#ifdef _WIN32
		if (MoveFileExW(
			tmp.native().c_str(),
			dst.native().c_str(),
			MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
		{
			return true;
		}

		error = "Win32 error " + std::to_string(GetLastError());
		return false;
	#else
		std::error_code ec;
		std::filesystem::rename(tmp, dst, ec);
		if (!ec) { return true; }

		error = ec.message();
		return false;
	#endif
	}

	static void AtomicReplace(
		const std::filesystem::path& tmp,
		const std::filesystem::path& dst)
	{
		std::string error;
		if (!TryAtomicReplace(tmp, dst, error))
		{
			throw std::runtime_error("Atomic replace failed: " + error + ".");
		}
	}

	static std::string FormatStepPadded(uint64_t step, uint32_t width = 9)
	{
		std::ostringstream oss;
		oss << std::setfill('0') << std::setw(width) << step;
		return oss.str();
	}

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

	struct ExportSnapshotOffsets
	{
		uint64_t xF = 0;
		uint64_t yF = 0;
		uint64_t zF = 0;
		uint64_t vF = 0;
		uint64_t idF = 0;
		uint64_t rankF = 0;

		uint64_t xPosR = 0;
		uint64_t yPosR = 0;
		uint64_t zPosR = 0;
		uint64_t volR = 0;
		uint64_t idR = 0;
		uint64_t bodyIdR = 0;

		// total size implied by the arrays above, used to recognize files of the current layout
		uint64_t totalBytes = 0;
	};

	// the static boundary never moves, so it lives in one file only
	struct StaticBoundaryOffsets
	{
		uint64_t xPosB = 0;
		uint64_t yPosB = 0;
		uint64_t zPosB = 0;
		uint64_t volB = 0;
		uint64_t idB = 0;

		uint64_t totalBytes = 0;
	};

	static ExportSnapshotOffsets ComputeExportSnapshotOffsets(
		const ExportSnapshotBinInfo& info)
	{
		constexpr uint64_t N_elementBytes = 4; // float32 / uint32
		constexpr uint64_t N_rankBytes = 2; // uint16

		// offsets must match SaveExportSnapshot write order
		// (an absent rank array has no entries, so the rigid arrays move up on their own)
		ExportSnapshotOffsets off{};
		off.xF = sizeof(BinHeaderSnap);
		off.yF = off.xF + static_cast<uint64_t>(info.NF) * N_elementBytes;
		off.zF = off.yF + static_cast<uint64_t>(info.NF) * N_elementBytes;
		off.vF = off.zF + static_cast<uint64_t>(info.NF) * N_elementBytes;
		off.idF = off.vF + static_cast<uint64_t>(info.NF) * N_elementBytes;
		off.rankF = off.idF + static_cast<uint64_t>(info.NF) * N_elementBytes;

		off.xPosR = off.rankF + static_cast<uint64_t>(info.NrankF) * N_rankBytes;
		off.yPosR = off.xPosR + static_cast<uint64_t>(info.NR) * N_elementBytes;
		off.zPosR = off.yPosR + static_cast<uint64_t>(info.NR) * N_elementBytes;
		off.volR = off.zPosR + static_cast<uint64_t>(info.NR) * N_elementBytes;
		off.idR = off.volR + static_cast<uint64_t>(info.NR) * N_elementBytes;
		off.bodyIdR = off.idR + static_cast<uint64_t>(info.NR) * N_elementBytes;

		off.totalBytes = off.bodyIdR + static_cast<uint64_t>(info.NR) * N_elementBytes;

		return off;
	}

	static StaticBoundaryOffsets ComputeStaticBoundaryOffsets(uint32_t NB)
	{
		constexpr uint64_t N_elementBytes = 4; // float32 / uint32

		// offsets must match SaveStaticBoundaryBin write order
		StaticBoundaryOffsets off{};
		off.xPosB = sizeof(BinHeaderSnap);
		off.yPosB = off.xPosB + static_cast<uint64_t>(NB) * N_elementBytes;
		off.zPosB = off.yPosB + static_cast<uint64_t>(NB) * N_elementBytes;
		off.volB = off.zPosB + static_cast<uint64_t>(NB) * N_elementBytes;
		off.idB = off.volB + static_cast<uint64_t>(NB) * N_elementBytes;

		off.totalBytes = off.idB + static_cast<uint64_t>(NB) * N_elementBytes;

		return off;
	}

	static bool IsExportSnapshotBinFileName(const std::string& fileName)
	{
		constexpr char prefix[] = "particle_data_step_";
		constexpr char suffix[] = ".bin";

		return fileName.rfind(prefix, 0) == 0
			&& fileName.size() >= sizeof(prefix) + sizeof(suffix) - 2
			&& fileName.compare(fileName.size() - (sizeof(suffix) - 1),
				sizeof(suffix) - 1, suffix) == 0;
	}

	static bool TryReadExportSnapshotBinInfo(
		const std::filesystem::path& path,
		ExportSnapshotBinInfo& info)
	{
		std::ifstream ifs(path, std::ios::binary | std::ios::in);
		if (!ifs) { return false; }

		BinHeaderSnap hdr{};
		ReadPod(ifs, hdr);
		if (!ifs) { return false; }

		if (std::memcmp(hdr.magic, kExportSnapshotMagic, sizeof(hdr.magic)) != 0)
		{
			return false;
		}
		if (!IsLittleEndian() || hdr.endianTag != kEndianLE)
		{
			return false;
		}
		if (hdr.headerBytes != sizeof(BinHeaderSnap))
		{
			return false;
		}

		info.fileName = path.filename().string();
		info.step = hdr.step;
		info.time = hdr.time;
		info.NF = hdr.NF;
		info.NB = hdr.NB;
		info.NR = hdr.NR;
		info.NrankF = hdr.NrankF;

		// there is only one layout, so a file belongs to it matches the size implied by the array set
		// (this rejects files left over from older layouts and truncated writes)
		std::error_code ec;
		const uint64_t N_fileBytes = std::filesystem::file_size(path, ec);
		if (ec || N_fileBytes != ComputeExportSnapshotOffsets(info).totalBytes)
		{
			return false;
		}

		return true;
	}

	static bool TryReadStaticBoundaryBinInfo(
		const std::filesystem::path& path,
		uint32_t& NB)
	{
		std::ifstream ifs(path, std::ios::binary | std::ios::in);
		if (!ifs) { return false; }

		BinHeaderSnap hdr{};
		ReadPod(ifs, hdr);
		if (!ifs) { return false; }

		if (std::memcmp(hdr.magic, kExportSnapshotMagic, sizeof(hdr.magic)) != 0)
		{
			return false;
		}
		if (!IsLittleEndian() || hdr.endianTag != kEndianLE)
		{
			return false;
		}
		if (hdr.headerBytes != sizeof(BinHeaderSnap))
		{
			return false;
		}

		// the boundary file carries no fluid or rigid arrays
		if (hdr.NF != 0 || hdr.NR != 0)
		{
			return false;
		}

		std::error_code ec;
		const uint64_t N_fileBytes = std::filesystem::file_size(path, ec);
		if (ec || N_fileBytes != ComputeStaticBoundaryOffsets(hdr.NB).totalBytes)
		{
			return false;
		}

		NB = hdr.NB;
		return true;
	}

	static std::vector<ExportSnapshotBinInfo> CollectExportSnapshotBinInfos(
		const std::filesystem::path& dirPath)
	{
		// all valid export bins in this directory become one time series
		std::vector<ExportSnapshotBinInfo> infos;
		uint32_t N_skipped = 0;

		std::error_code ec;
		for (const auto& entry : std::filesystem::directory_iterator(dirPath, ec))
		{
			if (ec) { break; }

			std::error_code fileEc;
			if (!entry.is_regular_file(fileEc) || fileEc) { continue; }

			const std::string fileName = entry.path().filename().string();
			if (!IsExportSnapshotBinFileName(fileName)) { continue; }

			ExportSnapshotBinInfo info{};
			if (TryReadExportSnapshotBinInfo(entry.path(), info))
			{
				infos.push_back(std::move(info));
			}
			else
			{
				N_skipped++;
			}
		}

		if (N_skipped > 0)
		{
			// most likely snapshots from an older layout or truncated writes
			spdlog::warn("Skipped {} export snapshot(s) in '{}' that do not match the current layout.",
				N_skipped, dirPath.string());
		}

		std::sort(infos.begin(), infos.end(),
			[](const ExportSnapshotBinInfo& a, const ExportSnapshotBinInfo& b)
			{
				if (a.time != b.time) { return a.time < b.time; }
				return a.step < b.step;
			});

		return infos;
	}

	static void WriteBinaryDataItem(
		std::ofstream& ofs,
		uint32_t N_values,
		const char* numberType,
		uint32_t precisionBytes,
		uint64_t offset,
		const std::string& fileName,
		const char* indent)
	{
		ofs << indent << "<DataItem Dimensions=\"" << N_values
			<< "\" NumberType=\"" << numberType
			<< "\" Precision=\"" << precisionBytes << "\" "
			<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
			<< offset << "\">\n";
		ofs << indent << "  " << fileName << "\n";
		ofs << indent << "</DataItem>\n";
	}

	static void WriteXyzGeometry(
		std::ofstream& ofs,
		const char* name,
		uint32_t N_points,
		uint64_t offX,
		uint64_t offY,
		uint64_t offZ,
		const std::string& fileName)
	{
		ofs << "        <Geometry Name=\"" << name << "\" GeometryType=\"XYZ\">\n";
		ofs << "          <DataItem ItemType=\"Function\" "
			<< "Function=\"JOIN($0, $1, $2)\" "
			<< "Dimensions=\"" << N_points << " 3\" "
			<< "NumberType=\"Float\" Precision=\"4\">\n";
		WriteBinaryDataItem(ofs, N_points, "Float", 4, offX, fileName, "            ");
		WriteBinaryDataItem(ofs, N_points, "Float", 4, offY, fileName, "            ");
		WriteBinaryDataItem(ofs, N_points, "Float", 4, offZ, fileName, "            ");
		ofs << "          </DataItem>\n";
		ofs << "        </Geometry>\n";
	}

	static void WriteScalarAttribute(
		std::ofstream& ofs,
		const char* name,
		uint32_t N_values,
		const char* numberType,
		uint32_t precisionBytes,
		uint64_t offset,
		const std::string& fileName)
	{
		ofs << "        <Attribute Name=\"" << name << "\" "
			<< "AttributeType=\"Scalar\" Center=\"Node\">\n";
		WriteBinaryDataItem(ofs, N_values, numberType, precisionBytes, offset, fileName, "          ");
		ofs << "        </Attribute>\n";
	}

	static void WriteFluidSurfaceGrid(
		std::ofstream& ofs,
		const ExportSnapshotBinInfo& info)
	{
		const ExportSnapshotOffsets off = ComputeExportSnapshotOffsets(info);

		ofs << "      <Grid Name=\"FluidSurface\" GridType=\"Uniform\">\n";
		ofs << "        <Time Value=\""
			<< std::setprecision(16) << info.time
			<< "\" />\n";
		ofs << "        <Topology TopologyType=\"Polyvertex\" "
			<< "Dimensions=\"" << info.NF << "\"/>\n";
		WriteXyzGeometry(ofs, "posF", info.NF, off.xF, off.yF, off.zF, info.fileName);
		WriteScalarAttribute(ofs, "velMagF", info.NF, "Float", 4, off.vF, info.fileName);
		WriteScalarAttribute(ofs, "idF", info.NF, "UInt", 4, off.idF, info.fileName);

		// non-distributed runs own every particle, so the ranks carry no information there
		if (info.NrankF > 0)
		{
			WriteScalarAttribute(ofs, "rankF", info.NF, "UInt", 2, off.rankF, info.fileName);
		}
		ofs << "      </Grid>\n";
	}

	// no <Time> element, so the reader reports no time steps and stays visible for the whole animation
	static void WriteStaticBoundaryGrid(
		std::ofstream& ofs,
		uint32_t NB,
		const std::string& fileName)
	{
		const StaticBoundaryOffsets off = ComputeStaticBoundaryOffsets(NB);

		ofs << "      <Grid Name=\"StaticBoundary\" GridType=\"Uniform\">\n";
		ofs << "        <Topology TopologyType=\"Polyvertex\" "
			<< "Dimensions=\"" << NB << "\"/>\n";
		WriteXyzGeometry(ofs, "posB", NB,
			off.xPosB, off.yPosB, off.zPosB, fileName);
		WriteScalarAttribute(ofs, "volB", NB,
			"Float", 4, off.volB, fileName);
		WriteScalarAttribute(ofs, "idB", NB,
			"UInt", 4, off.idB, fileName);
		ofs << "      </Grid>\n";
	}

	static void WriteRigidParticleGrid(
		std::ofstream& ofs,
		const ExportSnapshotBinInfo& info)
	{
		const ExportSnapshotOffsets off = ComputeExportSnapshotOffsets(info);

		ofs << "      <Grid Name=\"RigidParticles\" GridType=\"Uniform\">\n";
		ofs << "        <Time Value=\""
			<< std::setprecision(16) << info.time
			<< "\" />\n";
		ofs << "        <Topology TopologyType=\"Polyvertex\" "
			<< "Dimensions=\"" << info.NR << "\"/>\n";
		WriteXyzGeometry(ofs, "posR", info.NR,
			off.xPosR, off.yPosR, off.zPosR, info.fileName);
		WriteScalarAttribute(ofs, "volR", info.NR,
			"Float", 4, off.volR, info.fileName);
		WriteScalarAttribute(ofs, "idR", info.NR,
			"UInt", 4, off.idR, info.fileName);
		WriteScalarAttribute(ofs, "bodyIdR", info.NR,
			"UInt", 4, off.bodyIdR, info.fileName);
		ofs << "      </Grid>\n";
	}

	using XdmfGridWriter = void(*)(std::ofstream&, const ExportSnapshotBinInfo&);

	// writes the XDMF envelope, writeBody fills the <Domain>
	template <class WriteBodyFn>
	static void WriteXdmfFile(
		const std::filesystem::path& path,
		const SaveOptions& saveOptions,
		WriteBodyFn writeBody)
	{
		// rewrite the complete file through a temp file when requested
		const std::string tempSuffix = saveOptions.tempSuffix.empty()
			? ".tmp"
			: saveOptions.tempSuffix;
		std::filesystem::path writePath = path;
		if (saveOptions.atomicWrite)
		{
			writePath += tempSuffix;
		}

		std::ofstream ofs(writePath, std::ios::out | std::ios::trunc);
		if (!ofs)
		{
			throw std::runtime_error(
				"Failed to open XDMF file for write '" + writePath.string() + "'.");
		}

		const auto removeTempFile = [&]()
		{
			if (!saveOptions.atomicWrite) { return; }

			std::error_code removeEc;
			std::filesystem::remove(writePath, removeEc);
		};

		ofs << R"(<?xml version="1.0" ?>)" << "\n";
		ofs << R"(<Xdmf Version="3.0">)" << "\n";
		ofs << R"(  <Domain>)" << "\n";

		writeBody(ofs);

		ofs << "  </Domain>\n";
		ofs << "</Xdmf>\n";
		if (!ofs.good())
		{
			ofs.close();
			removeTempFile();
			throw std::runtime_error(
				"I/O error while writing XDMF file '" + writePath.string() + "'.");
		}

		ofs.close();
		if (!ofs)
		{
			removeTempFile();
			throw std::runtime_error(
				"I/O error while closing XDMF file '" + writePath.string() + "'.");
		}

		if (saveOptions.atomicWrite)
		{
			std::string error;
			if (!TryAtomicReplace(writePath, path, error))
			{
				removeTempFile();
				throw std::runtime_error("Failed to replace XDMF file '" +
					path.string() + "': " + error + ".");
			}
		}
	}

	static void WriteXdmfCollectionFile(
		const std::filesystem::path& path,
		const char* name,
		const std::vector<ExportSnapshotBinInfo>& infos,
		XdmfGridWriter writeGrid,
		const SaveOptions& saveOptions)
	{
		WriteXdmfFile(path, saveOptions, [&](std::ofstream& ofs)
		{
			ofs << "    <Grid Name=\"" << name
				<< "\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";

			for (const ExportSnapshotBinInfo& info : infos)
			{
				writeGrid(ofs, info);
			}

			ofs << "    </Grid>\n";
		});
	}

	static void WriteXdmfStaticBoundaryFile(
		const std::filesystem::path& path,
		uint32_t NB,
		const std::string& fileName,
		const SaveOptions& saveOptions)
	{
		WriteXdmfFile(path, saveOptions, [&](std::ofstream& ofs)
		{
			WriteStaticBoundaryGrid(ofs, NB, fileName);
		});
	}

	static bool HasRigidParticles(const std::vector<ExportSnapshotBinInfo>& infos)
	{
		return std::any_of(infos.begin(), infos.end(),
			[](const ExportSnapshotBinInfo& info)
			{
				return info.NR > 0;
			});
	}

	static void CheckFluidConsistent(const Sync::FluidParticles& F)
	{
		[[maybe_unused]] const auto N = static_cast<size_t>(F.N_);

		assert(F.id.size() == N);
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

		assert(B.id.size() == N);
		assert(B.xPos.size() == N && B.yPos.size() == N && B.zPos.size() == N);
		assert(B.volB.size() == N && B.psiB.size() == N);
	}

	static void CheckRigidConsistent(const Sync::RigidParticles& R)
	{
		[[maybe_unused]] const auto N = static_cast<size_t>(R.N_);

		assert(R.id.size() == N && R.bodyId.size() == N);
		assert(R.xRelPos.size() == N && R.yRelPos.size() == N && R.zRelPos.size() == N);
		assert(R.volR.size() == N && R.psiR.size() == N);
	}

	static void CheckRigidBodiesConsistent(const Sync::RigidBodies& RB)
	{
		[[maybe_unused]] const auto N = static_cast<size_t>(RB.N_);

		assert(RB.id.size() == N);
		assert(RB.rigidParticleBegin.size() == N && RB.rigidParticleCount.size() == N);
		assert(RB.xCM.size() == N && RB.yCM.size() == N && RB.zCM.size() == N);
		assert(RB.xVel.size() == N && RB.yVel.size() == N && RB.zVel.size() == N);
		assert(RB.xOmega.size() == N && RB.yOmega.size() == N && RB.zOmega.size() == N);
		assert(RB.xAngMom.size() == N && RB.yAngMom.size() == N && RB.zAngMom.size() == N);
		assert(RB.a00.size() == N && RB.a01.size() == N && RB.a02.size() == N);
		assert(RB.a10.size() == N && RB.a11.size() == N && RB.a12.size() == N);
		assert(RB.a20.size() == N && RB.a21.size() == N && RB.a22.size() == N);
		assert(RB.mass.size() == N && RB.massInv.size() == N);
		assert(RB.iBodyInv00.size() == N && RB.iBodyInv01.size() == N && RB.iBodyInv02.size() == N);
		assert(RB.iBodyInv10.size() == N && RB.iBodyInv11.size() == N && RB.iBodyInv12.size() == N);
		assert(RB.iBodyInv20.size() == N && RB.iBodyInv21.size() == N && RB.iBodyInv22.size() == N);
	}

	// --- [ UniversalSimState <-> .bin ] ----------------------------------------------------------

	void SerializationMgr::SaveUniversalSimState(
		const Sync::UniversalSimState& state,
		const std::string& path) const
	{
		// file path handling
		// (full sim state is a fixed-order SoA dump for exact restart)
		const std::filesystem::path dstPath =
			ResolveStatePath(saveOptions_.baseDir, path);

		const std::filesystem::path parentPath = dstPath.parent_path();
		if (!parentPath.empty())
		{
			std::error_code ec;
			std::filesystem::create_directories(parentPath, ec);
			if (ec)
			{
				throw std::runtime_error(
					"Failed to create simulation state directory '" +
					parentPath.string() + "': " + ec.message() + ".");
			}
		}

		const std::filesystem::path tmpPath = saveOptions_.atomicWrite
			? std::filesystem::path(dstPath.string() + saveOptions_.tempSuffix)
			: dstPath;

		// sanity check before writing
		CheckFluidConsistent(state.particlesF);
		CheckBoundaryConsistent(state.particlesB);
		CheckRigidConsistent(state.particlesR);
		CheckRigidBodiesConsistent(state.rigidBodies);
		state.ValidateRigidAttachments();

		// open scope for writer
		{
			std::ofstream ofs(tmpPath, std::ios::binary | std::ios::out | std::ios::trunc);
			if (!ofs)
			{
				throw std::runtime_error(
					"Failed to open file for write '" + tmpPath.string() + "'.");
			}

			// header
			BinHeaderSim hdr{};
			std::memcpy(hdr.magic, kSimStateMagic, sizeof(hdr.magic));
			hdr.endianTag = kEndianLE;
			hdr.headerBytes = static_cast<uint32_t>(sizeof(BinHeaderSim));

			hdr.step = state.step;
			hdr.time = state.time;

			hdr.NF = state.particlesF.N_;
			hdr.NB = state.particlesB.N_;
			hdr.NR = state.particlesR.N_;
			hdr.NRB = state.rigidBodies.N_;

			WritePod(ofs, hdr);

			// fluid particle data (fixed order)
			{
				const auto& F = state.particlesF;

				WriteVec(ofs, F.id);

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

			// boundary particle data (fixed order)
			{
				const auto& B = state.particlesB;

				WriteVec(ofs, B.id);

				WriteVec(ofs, B.xPos);
				WriteVec(ofs, B.yPos);
				WriteVec(ofs, B.zPos);

				WriteVec(ofs, B.volB);
				WriteVec(ofs, B.psiB);
			}

			// rigid particle data (fixed order)
			{
				const auto& R = state.particlesR;

				WriteVec(ofs, R.id);
				WriteVec(ofs, R.bodyId);

				WriteVec(ofs, R.xRelPos);
				WriteVec(ofs, R.yRelPos);
				WriteVec(ofs, R.zRelPos);

				WriteVec(ofs, R.volR);
				WriteVec(ofs, R.psiR);
			}

			// rigid body data (fixed order)
			{
				const auto& RB = state.rigidBodies;

				WriteVec(ofs, RB.id);
				WriteVec(ofs, RB.rigidParticleBegin);
				WriteVec(ofs, RB.rigidParticleCount);

				WriteVec(ofs, RB.xCM);
				WriteVec(ofs, RB.yCM);
				WriteVec(ofs, RB.zCM);

				WriteVec(ofs, RB.xVel);
				WriteVec(ofs, RB.yVel);
				WriteVec(ofs, RB.zVel);

				WriteVec(ofs, RB.xOmega);
				WriteVec(ofs, RB.yOmega);
				WriteVec(ofs, RB.zOmega);

				WriteVec(ofs, RB.xAngMom);
				WriteVec(ofs, RB.yAngMom);
				WriteVec(ofs, RB.zAngMom);

				WriteVec(ofs, RB.a00);
				WriteVec(ofs, RB.a01);
				WriteVec(ofs, RB.a02);
				WriteVec(ofs, RB.a10);
				WriteVec(ofs, RB.a11);
				WriteVec(ofs, RB.a12);
				WriteVec(ofs, RB.a20);
				WriteVec(ofs, RB.a21);
				WriteVec(ofs, RB.a22);

				WriteVec(ofs, RB.mass);
				WriteVec(ofs, RB.massInv);

				WriteVec(ofs, RB.iBodyInv00);
				WriteVec(ofs, RB.iBodyInv01);
				WriteVec(ofs, RB.iBodyInv02);
				WriteVec(ofs, RB.iBodyInv10);
				WriteVec(ofs, RB.iBodyInv11);
				WriteVec(ofs, RB.iBodyInv12);
				WriteVec(ofs, RB.iBodyInv20);
				WriteVec(ofs, RB.iBodyInv21);
				WriteVec(ofs, RB.iBodyInv22);
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
		const std::filesystem::path srcPath =
			ResolveStatePath(saveOptions_.baseDir, path);

		std::ifstream ifs(srcPath, std::ios::binary | std::ios::in);
		if (!ifs)
		{
			throw std::runtime_error(
				"Failed to open file for read '" + srcPath.string() + "'.");
		}

		// header
		BinHeaderSim hdr{};
		ReadPod(ifs, hdr);
		if (!ifs)
		{
			throw std::runtime_error("Failed to read simulation state header from '" +
				srcPath.string() + "'.");
		}

		if (std::memcmp(hdr.magic, kSimStateMagic, sizeof(hdr.magic)) != 0)
		{
			throw std::runtime_error(
				"Invalid simulation state file magic in '" + srcPath.string() + "'.");
		}

		if (!IsLittleEndian() || hdr.endianTag != kEndianLE)
		{
			throw std::runtime_error("Endianness mismatch in '" + srcPath.string() +
				"': File and host must be little-endian.");
		}
		if (hdr.headerBytes != sizeof(BinHeaderSim))
		{
			throw std::runtime_error("Unexpected simulation state header size in '" +
				srcPath.string() + "'.");
		}

		// metadata
		state.step = hdr.step;
		state.time = hdr.time;

		// resize storage
		state.ResizeFluid(hdr.NF);
		state.ResizeBoundary(hdr.NB);
		state.ResizeRigid(hdr.NR);
		state.ResizeRigidBodies(hdr.NRB);

		// fluid particle data (read order must match SaveUniversalSimState)
		{
			auto& F = state.particlesF;

			ReadVec(ifs, F.id);
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
			F.N_ = hdr.NF;
		}

		// boundary particle data (fixed order)
		{
			auto& B = state.particlesB;

			ReadVec(ifs, B.id);

			ReadVec(ifs, B.xPos);
			ReadVec(ifs, B.yPos);
			ReadVec(ifs, B.zPos);

			ReadVec(ifs, B.volB);
			ReadVec(ifs, B.psiB);
		}

		// rigid particle data (fixed order)
		{
			auto& R = state.particlesR;

			ReadVec(ifs, R.id);
			ReadVec(ifs, R.bodyId);

			ReadVec(ifs, R.xRelPos);
			ReadVec(ifs, R.yRelPos);
			ReadVec(ifs, R.zRelPos);

			ReadVec(ifs, R.volR);
			ReadVec(ifs, R.psiR);
		}

		// rigid body data (fixed order)
		{
			auto& RB = state.rigidBodies;

			ReadVec(ifs, RB.id);
			ReadVec(ifs, RB.rigidParticleBegin);
			ReadVec(ifs, RB.rigidParticleCount);

			ReadVec(ifs, RB.xCM);
			ReadVec(ifs, RB.yCM);
			ReadVec(ifs, RB.zCM);

			ReadVec(ifs, RB.xVel);
			ReadVec(ifs, RB.yVel);
			ReadVec(ifs, RB.zVel);

			ReadVec(ifs, RB.xOmega);
			ReadVec(ifs, RB.yOmega);
			ReadVec(ifs, RB.zOmega);

			ReadVec(ifs, RB.xAngMom);
			ReadVec(ifs, RB.yAngMom);
			ReadVec(ifs, RB.zAngMom);

			ReadVec(ifs, RB.a00);
			ReadVec(ifs, RB.a01);
			ReadVec(ifs, RB.a02);
			ReadVec(ifs, RB.a10);
			ReadVec(ifs, RB.a11);
			ReadVec(ifs, RB.a12);
			ReadVec(ifs, RB.a20);
			ReadVec(ifs, RB.a21);
			ReadVec(ifs, RB.a22);

			ReadVec(ifs, RB.mass);
			ReadVec(ifs, RB.massInv);

			ReadVec(ifs, RB.iBodyInv00);
			ReadVec(ifs, RB.iBodyInv01);
			ReadVec(ifs, RB.iBodyInv02);
			ReadVec(ifs, RB.iBodyInv10);
			ReadVec(ifs, RB.iBodyInv11);
			ReadVec(ifs, RB.iBodyInv12);
			ReadVec(ifs, RB.iBodyInv20);
			ReadVec(ifs, RB.iBodyInv21);
			ReadVec(ifs, RB.iBodyInv22);
		}

		EnsureStreamOk(ifs, "reading", srcPath.string());
		state.ValidateRigidAttachments();
	}

	// --- [ ExportSnapshot (.bin) + XDMF ] --------------------------------------------------------

	// the static boundary is identical in every step, so all steps share one file
	void SerializationMgr::SaveStaticBoundaryBin(
		const Sync::ExportSnapshot& snap,
		const std::filesystem::path& dirPath) const
	{
		if (staticBoundaryWritten_) { return; }

		const std::filesystem::path dstPath = dirPath / kStaticBoundaryBinFileName;

		std::error_code removeEc;
		if (snap.N_B == 0)
		{
			std::filesystem::remove(dstPath, removeEc);
			staticBoundaryWritten_ = true;
			return;
		}

		const std::filesystem::path tmpPath = saveOptions_.atomicWrite
			? std::filesystem::path(dstPath.string() + saveOptions_.tempSuffix)
			: dstPath;

		// open scope for writer
		{
			std::ofstream ofs(tmpPath, std::ios::binary | std::ios::out | std::ios::trunc);
			if (!ofs)
			{
				throw std::runtime_error("Failed to open static boundary file for write '" +
					tmpPath.string() + "'.");
			}

			// arrays are written in the same order as referenced by the XDMF file:
			// xPosB, yPosB, zPosB, volB, idB

			// NF and NR stay zero, this file holds boundary particles only
			BinHeaderSnap hdr{};
			std::memcpy(hdr.magic, kExportSnapshotMagic, sizeof(hdr.magic));
			hdr.endianTag = kEndianLE;
			hdr.headerBytes = static_cast<uint32_t>(sizeof(BinHeaderSnap));
			hdr.step = snap.simStep;
			hdr.time = snap.simTime;
			hdr.NF = 0;
			hdr.NB = snap.N_B;
			hdr.NR = 0;
			hdr.NrankF = 0;

			WritePod(ofs, hdr);

			// arrays (fixed order)
			WriteVec(ofs, snap.xPosB);
			WriteVec(ofs, snap.yPosB);
			WriteVec(ofs, snap.zPosB);
			WriteVec(ofs, snap.volB);
			WriteVec(ofs, snap.idB);

			EnsureStreamOk(ofs, "writing", tmpPath.string());
		}

		if (saveOptions_.atomicWrite)
		{
			AtomicReplace(tmpPath, dstPath);
		}

		staticBoundaryWritten_ = true;
	}

	void SerializationMgr::SaveExportSnapshot(
		const Sync::ExportSnapshot& snap,
		const std::string& dir) const
	{
		// binary payload owns the data, XDMF only references offsets into it
		const std::string binFileName =
			"particle_data_step_" + FormatStepPadded(snap.simStep) + ".bin";

		// ensure export directory exists
		std::error_code ec;
		std::filesystem::create_directories(saveOptions_.baseDir / dir, ec);
		if (ec)
		{
			throw std::runtime_error("Failed to create export directory '" +
				(saveOptions_.baseDir / dir).string() + "': " + ec.message() + ".");
		}

		// make sure data for every particle is present
		const uint32_t NF_size = snap.N_F;
		const uint32_t NB_size = snap.N_B;
		const uint32_t NR_size = snap.N_R;
		assert(snap.idF.size() == NF_size);
		assert(snap.xPosF.size() == NF_size);
		assert(snap.yPosF.size() == NF_size);
		assert(snap.zPosF.size() == NF_size);
		assert(snap.velMagF.size() == NF_size);
		assert(snap.rankF.empty() || snap.rankF.size() == NF_size);
		assert(snap.idB.size() == NB_size);
		assert(snap.xPosB.size() == NB_size);
		assert(snap.yPosB.size() == NB_size);
		assert(snap.zPosB.size() == NB_size);
		assert(snap.volB.size() == NB_size);
		assert(snap.idR.size() == NR_size);
		assert(snap.bodyIdR.size() == NR_size);
		assert(snap.xPosR.size() == NR_size);
		assert(snap.yPosR.size() == NR_size);
		assert(snap.zPosR.size() == NR_size);
		assert(snap.volR.size() == NR_size);

		SaveStaticBoundaryBin(snap, saveOptions_.baseDir / dir);

		// file path handling
		const std::filesystem::path dstPath = saveOptions_.baseDir / dir / binFileName;
		const std::filesystem::path tmpPath = saveOptions_.atomicWrite
			? std::filesystem::path(dstPath.string() + saveOptions_.tempSuffix)
			: dstPath;

		// open scope for writer
		{
			std::ofstream ofs(tmpPath, std::ios::binary | std::ios::out | std::ios::trunc);
			if (!ofs)
			{
				throw std::runtime_error("Failed to open snapshot file for write '" +
					tmpPath.string() + "'.");
			}

			// arrays are written in the same order as referenced by the XDMF file:
			// xPosF, yPosF, zPosF, velMagF, idF, rankF (only in distributed runs)
			// xPosR, yPosR, zPosR, volR, idR, bodyIdR
			// (boundary arrays live in the shared static boundary file)

			// NB is informational here, the boundary arrays are not part of this file
			BinHeaderSnap hdr{};
			std::memcpy(hdr.magic, kExportSnapshotMagic, sizeof(hdr.magic));
			hdr.endianTag = kEndianLE;
			hdr.headerBytes = static_cast<uint32_t>(sizeof(BinHeaderSnap));
			hdr.step = snap.simStep;
			hdr.time = snap.simTime;
			hdr.NF = NF_size;
			hdr.NB = NB_size;
			hdr.NR = NR_size;
			hdr.NrankF = static_cast<uint32_t>(snap.rankF.size());

			WritePod(ofs, hdr);

			// arrays (fixed order)
			WriteVec(ofs, snap.xPosF);
			WriteVec(ofs, snap.yPosF);
			WriteVec(ofs, snap.zPosF);
			WriteVec(ofs, snap.velMagF);
			WriteVec(ofs, snap.idF);
			WriteVec(ofs, snap.rankF);

			WriteVec(ofs, snap.xPosR);
			WriteVec(ofs, snap.yPosR);
			WriteVec(ofs, snap.zPosR);
			WriteVec(ofs, snap.volR);
			WriteVec(ofs, snap.idR);
			WriteVec(ofs, snap.bodyIdR);

			EnsureStreamOk(ofs, "writing", tmpPath.string());
		}

		if (saveOptions_.atomicWrite)
		{
			AtomicReplace(tmpPath, dstPath);
		}
	}

	void SerializationMgr::SaveExportSnapshotXdmfViews(
		const Sync::ExportSnapshot& snap,
		const std::string& dir) const
	{
		const std::string fluidXdmfFileName = "fluid_surface.xdmf";
		const std::string staticBoundaryXdmfFileName = "static_boundary.xdmf";
		const std::string rigidParticleXdmfFileName = "rigid_particles.xdmf";
		const std::filesystem::path dirPath = saveOptions_.baseDir / dir;

		// ensure export directory exists
		// (stable XDMF files are rebuilt as temporal collections)
		std::error_code ec;
		std::filesystem::create_directories(dirPath, ec);
		if (ec)
		{
			throw std::runtime_error("Failed to create export directory '" +
				dirPath.string() + "': " + ec.message() + ".");
		}

		const std::filesystem::path fluidXdmfPath =
			dirPath / fluidXdmfFileName;
		const std::filesystem::path staticBoundaryXdmfPath =
			dirPath / staticBoundaryXdmfFileName;
		const std::filesystem::path rigidParticleXdmfPath =
			dirPath / rigidParticleXdmfFileName;

		// make sure data for every particle is present
		[[maybe_unused]] const uint32_t NF_size = snap.N_F;
		[[maybe_unused]] const uint32_t NB_size = snap.N_B;
		[[maybe_unused]] const uint32_t NR_size = snap.N_R;
		assert(snap.idF.size() == NF_size);
		assert(snap.xPosF.size() == NF_size);
		assert(snap.yPosF.size() == NF_size);
		assert(snap.zPosF.size() == NF_size);
		assert(snap.velMagF.size() == NF_size);
		assert(snap.rankF.empty() || snap.rankF.size() == NF_size);
		assert(snap.idB.size() == NB_size);
		assert(snap.xPosB.size() == NB_size);
		assert(snap.yPosB.size() == NB_size);
		assert(snap.zPosB.size() == NB_size);
		assert(snap.volB.size() == NB_size);
		assert(snap.idR.size() == NR_size);
		assert(snap.bodyIdR.size() == NR_size);
		assert(snap.xPosR.size() == NR_size);
		assert(snap.yPosR.size() == NR_size);
		assert(snap.zPosR.size() == NR_size);
		assert(snap.volR.size() == NR_size);

		const std::vector<ExportSnapshotBinInfo> infos =
			CollectExportSnapshotBinInfos(dirPath);
		if (infos.empty())
		{
			throw std::runtime_error("No export snapshot binary files found in '" +
				dirPath.string() + "'.");
		}

		WriteXdmfCollectionFile(
			fluidXdmfPath, "FluidSurface", infos, WriteFluidSurfaceGrid, saveOptions_);

		// the boundary file itself decides whether the view exists
		std::error_code removeEc;
		uint32_t NB_boundary = 0;
		if (TryReadStaticBoundaryBinInfo(dirPath / kStaticBoundaryBinFileName, NB_boundary))
		{
			WriteXdmfStaticBoundaryFile(staticBoundaryXdmfPath, NB_boundary,
				kStaticBoundaryBinFileName, saveOptions_);
		}
		else
		{
			std::filesystem::remove(staticBoundaryXdmfPath, removeEc);
		}

		if (HasRigidParticles(infos))
		{
			WriteXdmfCollectionFile(
				rigidParticleXdmfPath, "RigidParticles", infos,
				WriteRigidParticleGrid, saveOptions_);
		}
		else
		{
			std::filesystem::remove(rigidParticleXdmfPath, removeEc);
		}
	}

} // namespace Export
