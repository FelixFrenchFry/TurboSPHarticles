// serialization.cpp

// external
#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
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
		{ 'T', 'S', 'P', 'S', 'I', 'M', '2', '\0' };
	static constexpr char kExportSnapshotMagic[8] =
		{ 'T', 'S', 'P', 'S', 'N', 'P', '4', '\0' };

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

	static_assert(sizeof(BinHeaderSnap) == 64);

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

	// write interleaved XYZ positions in bounded blocks
	static void WriteXyz(std::ofstream& ofs,
		const std::vector<float>& x,
		const std::vector<float>& y,
		const std::vector<float>& z)
	{
		assert(x.size() == y.size() && x.size() == z.size());
		constexpr size_t N_block = 16384;
		std::vector<float> xyz(3 * std::min(x.size(), N_block));
		for (size_t begin = 0; begin < x.size();)
		{
			const size_t N_points = std::min(x.size() - begin, N_block);
			for (size_t i = 0; i < N_points; i++)
			{
				xyz[3 * i] = x[begin + i];
				xyz[3 * i + 1] = y[begin + i];
				xyz[3 * i + 2] = z[begin + i];
			}
			WriteExact(ofs, xyz.data(), 3 * N_points * sizeof(float));
			begin += N_points;
		}
	}

	static uint64_t CheckedByteProduct(
		const uint64_t count, const uint64_t width, const char* context)
	{
		if (width != 0 && count > std::numeric_limits<uint64_t>::max() / width)
		{
			throw std::runtime_error(std::string(context) + " byte size overflow.");
		}
		return count * width;
	}

	static uint64_t CheckedByteSum(
		const uint64_t a, const uint64_t b, const char* context)
	{
		if (b > std::numeric_limits<uint64_t>::max() - a)
		{
			throw std::runtime_error(std::string(context) + " byte size overflow.");
		}
		return a + b;
	}

	BinHeaderSim MakeSimStateHeader(
		const uint64_t step,
		const double time,
		const uint64_t N_F,
		const uint64_t N_B,
		const uint64_t N_R,
		const uint64_t N_RB)
	{
		static_assert(sizeof(float) == 4 && sizeof(double) == 8);
		static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
		if (!IsLittleEndian())
		{
			throw std::runtime_error("Simulation state requires little-endian data.");
		}

		BinHeaderSim header{};
		std::memcpy(header.magic, kSimStateMagic, sizeof(header.magic));
		header.endianTag = kEndianLE;
		header.headerBytes = sizeof(BinHeaderSim);
		header.step = step;
		header.time = time;
		header.N_F = N_F;
		header.N_B = N_B;
		header.N_R = N_R;
		header.N_RB = N_RB;

		return header;
	}

	SimStateOffsets ComputeSimStateOffsets(const BinHeaderSim& header)
	{
		uint64_t offset = sizeof(BinHeaderSim);
		const auto AppendPopulation = [&](const uint64_t count, const uint64_t particleBytes)
		{
			const uint64_t begin = offset;
			offset = CheckedByteSum(offset,
				CheckedByteProduct(count, particleBytes, "simulation state population"), "simulation state");
			return begin;
		};

		SimStateOffsets off{};
		off.fluid = AppendPopulation(header.N_F, sizeof(uint64_t) + 16ull * sizeof(float));
		off.boundary = AppendPopulation(header.N_B, sizeof(uint64_t) + 5ull * sizeof(float));
		off.rigid = AppendPopulation(header.N_R, sizeof(uint64_t) + sizeof(uint32_t) + 5ull * sizeof(float));
		off.rigidBodies = AppendPopulation(header.N_RB, 2ull * sizeof(uint32_t) + 32ull * sizeof(float));
		off.totalBytes = offset;

		return off;
	}

	static BinHeaderSim ReadSimStateHeader(std::ifstream& ifs, const std::filesystem::path& path)
	{
		BinHeaderSim header{};

		ReadPod(ifs, header);
		if (!ifs || std::memcmp(header.magic, kSimStateMagic, sizeof(header.magic)) != 0 ||
			!IsLittleEndian() || header.endianTag != kEndianLE || header.headerBytes != sizeof(BinHeaderSim))
		{
			throw std::runtime_error("Invalid simulation state header in '" + path.string() + "'.");
		}
		if (!std::isfinite(header.time) || header.time < 0 || header.step == std::numeric_limits<uint64_t>::max())
		{
			throw std::runtime_error("Invalid simulation state step/time.");
		}
		if (header.N_RB > std::numeric_limits<uint32_t>::max())
		{
			throw std::length_error("Serialized rigid body count exceeds uint32_t range.");
		}
		const auto off = ComputeSimStateOffsets(header);
		if (off.totalBytes > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()))
		{
			throw std::length_error("Simulation state exceeds stream offset range.");
		}
		if (std::filesystem::file_size(path) != off.totalBytes)
		{
			throw std::runtime_error("Simulation state file length does not match its header in '" + path.string() + "'.");
		}

		return header;
	}

	static void ValidateParticleIds(
		const std::span<const uint64_t> ids,
		const char* const context)
	{
		if (std::find(ids.begin(), ids.end(), Sim::INVALID_PARTICLE_ID) != ids.end())
		{
			throw std::runtime_error(std::string(context) + " contains the invalid particle ID sentinel.");
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

	ExportSnapshotOffsets ComputeExportSnapshotOffsets(
		const ExportSnapshotBinInfo& info)
	{
		uint64_t offset = sizeof(BinHeaderSnap);
		const auto AppendField = [&](const uint64_t count, const uint64_t elementBytes)
		{
			const uint64_t start = offset;
			offset = CheckedByteSum(offset,
				CheckedByteProduct(count, elementBytes, "snapshot field"),
				"snapshot offsets");
			return start;
		};

		// offsets must match SaveExportSnapshot write order
		// (an absent rank array has no entries, so the rigid arrays move up on their own)
		ExportSnapshotOffsets off{};
		off.posF = AppendField(info.N_F, 3 * sizeof(float));
		off.vF = AppendField(info.N_F, sizeof(float));
		off.rankF = AppendField(info.N_rankF, sizeof(uint16_t));
		off.posR = AppendField(info.N_R, 3 * sizeof(float));
		off.volR = AppendField(info.N_R, sizeof(float));
		off.bodyIdR = AppendField(info.N_R, sizeof(uint32_t));
		off.totalBytes = offset;

		return off;
	}

	StaticBoundaryOffsets ComputeStaticBoundaryOffsets(
		const ExportSnapshotBinInfo& info)
	{
		uint64_t offset = sizeof(BinHeaderSnap);
		const auto AppendField = [&](const uint64_t count, const uint64_t elementBytes)
		{
			const uint64_t start = offset;
			offset = CheckedByteSum(offset,
				CheckedByteProduct(count, elementBytes, "static boundary field"),
				"static boundary offsets");
			return start;
		};

		// offsets must match SaveStaticBoundaryBin write order
		StaticBoundaryOffsets off{};
		off.posB = AppendField(info.N_B, 3 * sizeof(float));
		off.volB = AppendField(info.N_B, sizeof(float));
		off.totalBytes = offset;

		return off;
	}

	BinHeaderSnap MakeExportSnapshotHeader(const ExportSnapshotBinInfo& info)
	{
		static_assert(sizeof(float) == 4 && sizeof(double) == 8);
		static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
		if (!IsLittleEndian()) { throw std::runtime_error("Snapshot output requires little-endian data."); }

		BinHeaderSnap hdr{};
		std::memcpy(hdr.magic, kExportSnapshotMagic, sizeof(hdr.magic));
		hdr.endianTag = kEndianLE;
		hdr.headerBytes = sizeof(BinHeaderSnap);
		hdr.step = info.step;
		hdr.time = info.time;
		hdr.N_F = info.N_F;
		hdr.N_B = info.N_B;
		hdr.N_R = info.N_R;
		hdr.N_rankF = info.N_rankF;

		return hdr;
	}

	DistributedExportLayout ComputeDistributedExportLayout(
		const std::vector<uint64_t>& rankCounts,
		const int rank)
	{
		const size_t N_ranks = rankCounts.size() / 3;
		if (rankCounts.size() % 3 != 0 || N_ranks == 0 ||
			N_ranks > uint64_t{1} + std::numeric_limits<uint16_t>::max() ||
			rank < 0 || static_cast<size_t>(rank) >= N_ranks)
		{
			throw std::runtime_error("Invalid distributed snapshot count table.");
		}

		DistributedExportLayout layout{};
		for (size_t rankIdx = 0; rankIdx < N_ranks; rankIdx++)
		{
			for (size_t populationIdx = 0; populationIdx < 3; populationIdx++)
			{
				const uint64_t N_particles = rankCounts[3 * rankIdx + populationIdx];
				Sim::CheckedLocalParticleCount(N_particles, "rank-local export count");
				if (rankIdx == static_cast<size_t>(rank))
				{
					layout.beginIdx[populationIdx] = layout.N_total[populationIdx];
				}

				// local counts and the rank limit keep the total within uint64_t
				layout.N_total[populationIdx] += N_particles;
				layout.N_max[populationIdx] = std::max(layout.N_max[populationIdx], N_particles);
			}
		}

		return layout;
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
		if (!ifs || std::memcmp(hdr.magic, kExportSnapshotMagic, sizeof(hdr.magic)) != 0 ||
			!IsLittleEndian() ||
			hdr.endianTag != kEndianLE ||
			hdr.headerBytes != sizeof(BinHeaderSnap))
		{
			return false;
		}

		info.fileName = path.filename().string();
		info.step = hdr.step;
		info.time = hdr.time;
		info.N_F = hdr.N_F;
		info.N_B = hdr.N_B;
		info.N_R = hdr.N_R;
		info.N_rankF = hdr.N_rankF;
		if (info.N_rankF != 0 && info.N_rankF != info.N_F) { return false; }

		// validate the complete payload before using its offsets
		std::error_code ec;
		const uint64_t N_fileBytes = std::filesystem::file_size(path, ec);
		if (ec) { return false; }
		try
		{
			if (N_fileBytes != ComputeExportSnapshotOffsets(info).totalBytes) { return false; }
		}
		catch (const std::runtime_error&) { return false; }

		return true;
	}

	static bool TryReadStaticBoundaryBinInfo(
		const std::filesystem::path& path,
		ExportSnapshotBinInfo& info)
	{
		std::ifstream ifs(path, std::ios::binary | std::ios::in);
		if (!ifs) { return false; }
		BinHeaderSnap hdr{};
		ReadPod(ifs, hdr);
		if (!ifs || std::memcmp(hdr.magic, kExportSnapshotMagic, sizeof(hdr.magic)) != 0 ||
			!IsLittleEndian() ||
			hdr.endianTag != kEndianLE ||
			hdr.headerBytes != sizeof(BinHeaderSnap))
		{
			return false;
		}

		info.fileName = path.filename().string();
		info.step = hdr.step;
		info.time = hdr.time;
		info.N_F = hdr.N_F;
		info.N_B = hdr.N_B;
		info.N_R = hdr.N_R;
		info.N_rankF = hdr.N_rankF;
		// the boundary file carries no fluid or rigid arrays
		if (info.N_F != 0 || info.N_R != 0 || info.N_rankF != 0) { return false; }

		std::error_code ec;
		const uint64_t N_fileBytes = std::filesystem::file_size(path, ec);
		if (ec) { return false; }
		try
		{
			if (N_fileBytes != ComputeStaticBoundaryOffsets(info).totalBytes) { return false; }
		}
		catch (const std::runtime_error&) { return false; }

		return true;
	}

	static std::vector<ExportSnapshotBinInfo> CollectExportSnapshotBinInfos(
		const std::filesystem::path& dirPath,
		std::unordered_map<std::string, ExportSnapshotBinInfo>& infoCache)
	{
		// all valid export bins in this directory become one time series
		std::vector<ExportSnapshotBinInfo> infos;
		uint32_t N_skipped = 0;

		try
		{
			for (const auto& entry : std::filesystem::directory_iterator(dirPath))
			{
				std::error_code fileEc;
				if (!entry.is_regular_file(fileEc) || fileEc) { continue; }

				const std::string fileName = entry.path().filename().string();
				if (!IsExportSnapshotBinFileName(fileName)) { continue; }

				// the header of an already parsed file never changes, only new files are read
				if (const auto it = infoCache.find(fileName); it != infoCache.end())
				{
					infos.push_back(it->second);
					continue;
				}

				ExportSnapshotBinInfo info{};
				if (TryReadExportSnapshotBinInfo(entry.path(), info))
				{
					infoCache.emplace(fileName, info);
					infos.push_back(std::move(info));
				}
				else
				{
					N_skipped++;
				}
			}
		}
		catch (const std::filesystem::filesystem_error& ex)
		{
			spdlog::warn("Failed to iterate export snapshots in '{}': {}.",
				dirPath.string(), ex.what());
			throw;
		}

		if (N_skipped > 0)
		{
			// most likely snapshots from an unsupported layout or truncated writes
			spdlog::warn("Skipped {} export snapshot(s) in '{}' that do not match a supported layout.",
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
		const std::string& dimensions,
		const char* numberType,
		uint32_t precisionBytes,
		uint64_t offset,
		const std::string& fileName,
		const char* indent)
	{
		ofs << indent << "<DataItem Dimensions=\"" << dimensions
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
		uint64_t N_points,
		uint64_t offset,
		const std::string& fileName)
	{
		ofs << "        <Geometry Name=\"" << name << "\" GeometryType=\"XYZ\">\n";
		WriteBinaryDataItem(ofs, std::to_string(N_points) + " 3",
			"Float", 4, offset, fileName, "          ");
		ofs << "        </Geometry>\n";
	}

	static void WriteScalarAttribute(
		std::ofstream& ofs,
		const char* name,
		uint64_t N_values,
		const char* numberType,
		uint32_t precisionBytes,
		uint64_t offset,
		const std::string& fileName)
	{
		ofs << "        <Attribute Name=\"" << name << "\" "
			<< "AttributeType=\"Scalar\" Center=\"Node\">\n";
		WriteBinaryDataItem(ofs, std::to_string(N_values),
			numberType, precisionBytes, offset, fileName, "          ");
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
			<< "Dimensions=\"" << info.N_F << "\"/>\n";
		WriteXyzGeometry(ofs, "posF", info.N_F, off.posF, info.fileName);
		WriteScalarAttribute(ofs, "velMagF", info.N_F, "Float", 4, off.vF, info.fileName);

		// non-distributed runs own every particle, so the ranks carry no information there
		if (info.N_rankF > 0)
		{
			WriteScalarAttribute(ofs, "rankF", info.N_F, "UShort", 2, off.rankF, info.fileName);
		}
		ofs << "      </Grid>\n";
	}

	// no <Time> element, so the reader reports no time steps and stays visible for the whole animation
	static void WriteStaticBoundaryGrid(
		std::ofstream& ofs,
		const ExportSnapshotBinInfo& info,
		const std::string& fileName)
	{
		const StaticBoundaryOffsets off = ComputeStaticBoundaryOffsets(info);

		ofs << "      <Grid Name=\"StaticBoundary\" GridType=\"Uniform\">\n";
		ofs << "        <Topology TopologyType=\"Polyvertex\" "
			<< "Dimensions=\"" << info.N_B << "\"/>\n";
		WriteXyzGeometry(ofs, "posB", info.N_B,
			off.posB, fileName);
		WriteScalarAttribute(ofs, "volB", info.N_B,
			"Float", 4, off.volB, fileName);
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
			<< "Dimensions=\"" << info.N_R << "\"/>\n";
		WriteXyzGeometry(ofs, "posR", info.N_R,
			off.posR, info.fileName);
		WriteScalarAttribute(ofs, "volR", info.N_R,
			"Float", 4, off.volR, info.fileName);
		WriteScalarAttribute(ofs, "bodyIdR", info.N_R,
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
		const ExportSnapshotBinInfo& info,
		const std::string& fileName,
		const SaveOptions& saveOptions)
	{
		WriteXdmfFile(path, saveOptions, [&](std::ofstream& ofs)
		{
			WriteStaticBoundaryGrid(ofs, info, fileName);
		});
	}

	static bool HasRigidParticles(const std::vector<ExportSnapshotBinInfo>& infos)
	{
		return std::any_of(infos.begin(), infos.end(),
			[](const ExportSnapshotBinInfo& info)
			{
				return info.N_R > 0;
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
		assert(RB.rigidParticleCount.size() == N);
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
		ValidateParticleIds(state.particlesF.id, "Fluid state");
		ValidateParticleIds(state.particlesB.id, "Boundary state");
		ValidateParticleIds(state.particlesR.id, "Rigid state");

		// open scope for writer
		{
			std::ofstream ofs(tmpPath, std::ios::binary | std::ios::out | std::ios::trunc);
			if (!ofs)
			{
				throw std::runtime_error(
					"Failed to open file for write '" + tmpPath.string() + "'.");
			}

			// header
			const auto hdr = MakeSimStateHeader(state.step, state.time,
				state.particlesF.N_, state.particlesB.N_,
				state.particlesR.N_, state.rigidBodies.N_);
			WritePod(ofs, hdr);

			const auto WriteField = [&](const auto& field) { WriteVec(ofs, field); };
			VisitFluidStateFields(WriteField, state.particlesF);
			VisitBoundaryStateFields(WriteField, state.particlesB);
			VisitRigidStateFields(WriteField, state.particlesR);
			VisitRigidBodyStateFields(WriteField, state.rigidBodies);

			ofs.close();
			EnsureStreamOk(ofs, "closing simulation state", tmpPath.string());
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

		const BinHeaderSim hdr = ReadSimStateHeader(ifs, srcPath);
		const uint32_t N_localF = Sim::CheckedLocalParticleCount(hdr.N_F, "serialized fluid state");
		const uint32_t N_localB = Sim::CheckedLocalParticleCount(hdr.N_B, "serialized boundary state");
		const uint32_t N_localR = Sim::CheckedLocalParticleCount(hdr.N_R, "serialized rigid state");
		const uint32_t N_localRB = static_cast<uint32_t>(hdr.N_RB);

		state.step = hdr.step;
		state.time = hdr.time;
		state.ResizeFluid(N_localF);
		state.ResizeBoundary(N_localB);
		state.ResizeRigid(N_localR);
		state.ResizeRigidBodies(N_localRB);

		const auto ReadField = [&](auto& field) { ReadVec(ifs, field); };
		VisitFluidStateFields(ReadField, state.particlesF);
		VisitBoundaryStateFields(ReadField, state.particlesB);
		VisitRigidStateFields(ReadField, state.particlesR);
		VisitRigidBodyStateFields(ReadField, state.rigidBodies);

		EnsureStreamOk(ifs, "reading", srcPath.string());
		ValidateParticleIds(state.particlesF.id, "Fluid state");
		ValidateParticleIds(state.particlesB.id, "Boundary state");
		ValidateParticleIds(state.particlesR.id, "Rigid state");
		state.ValidateRigidAttachments();
	}

	static constexpr uint32_t N_simStateChunk = 262144;

	static void ReadSimStateRange(
		std::ifstream& ifs,
		const uint64_t offset,
		void* data,
		const uint64_t N_bytes)
	{
		ifs.seekg(static_cast<std::streamoff>(offset));
		ifs.read(static_cast<char*>(data), static_cast<std::streamsize>(N_bytes));
		if (!ifs)
		{
			throw std::runtime_error("Failed to read simulation state range.");
		}
	}

	// both fluid and boundary arrays start with IDs followed by X/Y/Z positions
	static void ReadSimStatePositions(
		std::ifstream& ifs,
		const uint64_t N_total,
		const uint64_t populationOffset,
		const uint64_t beginIdx,
		const uint32_t N_particles,
		std::array<std::vector<float>, 3>& positions)
	{
		uint64_t offset = populationOffset + N_total * sizeof(uint64_t);
		for (auto& component : positions)
		{
			component.resize(N_particles);
			ReadSimStateRange(ifs, offset + beginIdx * sizeof(float),
				component.data(), N_particles * sizeof(float));
			offset += N_total * sizeof(float);
		}
		for (uint32_t i = 0; i < N_particles; i++)
		{
			if (!std::isfinite(positions[0][i]) || !std::isfinite(positions[1][i]) ||
				!std::isfinite(positions[2][i]))
			{
				throw std::runtime_error("Non-finite particle position in simulation state.");
			}
		}
	}

	static glm::vec3 RigidWorldPosition(
		const Sync::RigidBodies& bodies,
		const uint32_t bodyIdx,
		const float x,
		const float y,
		const float z)
	{
		if (bodyIdx >= bodies.N_)
		{
			throw std::runtime_error("Invalid rigid body index in simulation state.");
		}
		const glm::vec3 position
		{
			bodies.xCM[bodyIdx] + std::fmaf(bodies.a00[bodyIdx], x,
				std::fmaf(bodies.a01[bodyIdx], y, bodies.a02[bodyIdx] * z)),
			bodies.yCM[bodyIdx] + std::fmaf(bodies.a10[bodyIdx], x,
				std::fmaf(bodies.a11[bodyIdx], y, bodies.a12[bodyIdx] * z)),
			bodies.zCM[bodyIdx] + std::fmaf(bodies.a20[bodyIdx], x,
				std::fmaf(bodies.a21[bodyIdx], y, bodies.a22[bodyIdx] * z))
		};
		if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z))
		{
			throw std::runtime_error("Non-finite rigid world position in simulation state.");
		}
		return position;
	}

	BinHeaderSim SerializationMgr::VisitSimStatePositions(
		const std::string& path,
		const int partIdx,
		const int N_parts,
		const std::function<void(Base::ParticleType, const SimStatePositionView&)>& visit) const
	{
		if (N_parts <= 0 || partIdx < 0 || partIdx >= N_parts)
		{
			throw std::runtime_error("Invalid simulation state scan partition.");
		}
		const auto srcPath = ResolveStatePath(saveOptions_.baseDir, path);
		std::ifstream ifs(srcPath, std::ios::binary);
		if (!ifs)
		{
			throw std::runtime_error("Failed to open simulation state '" + srcPath.string() + "'.");
		}
		const auto header = ReadSimStateHeader(ifs, srcPath);
		const auto off = ComputeSimStateOffsets(header);
		std::array<std::vector<float>, 3> positions;
		std::vector<uint32_t> bodyIds;
		Sync::RigidBodies bodies;
		if (header.N_R > 0)
		{
			bodies.resizeAll(Sim::CheckedLocalParticleCount(header.N_RB, "serialized rigid bodies"));
			ifs.seekg(static_cast<std::streamoff>(off.rigidBodies));
			VisitRigidBodyStateFields([&](auto& field) { ReadVec(ifs, field); }, bodies);
		}
		const auto ScanPopulation = [&](const uint64_t N_total, const uint64_t offset, const Base::ParticleType type)
		{
			// divide without multiplying the global count by the partition index
			const uint64_t N_perPart = N_total / N_parts;
			const uint64_t N_remainder = N_total % N_parts;
			const uint64_t beginIdx = N_perPart * partIdx + std::min<uint64_t>(partIdx, N_remainder);
			const uint64_t endIdx = beginIdx + N_perPart + (static_cast<uint64_t>(partIdx) < N_remainder ? 1 : 0);
			for (uint64_t begin = beginIdx; begin < endIdx; begin += N_simStateChunk)
			{
				const uint32_t N_particles = static_cast<uint32_t>(std::min<uint64_t>(N_simStateChunk, endIdx - begin));
				if (type == Base::ParticleType::Rigid)
				{
					// rigid arrays store body indices between IDs and relative positions
					ReadSimStatePositions(
						ifs, N_total, offset + N_total * sizeof(uint32_t), begin, N_particles, positions);
					bodyIds.resize(N_particles);
					ReadSimStateRange(ifs, offset + N_total * sizeof(uint64_t) + begin * sizeof(uint32_t),
						bodyIds.data(), static_cast<uint64_t>(N_particles) * sizeof(uint32_t));
					for (uint32_t i = 0; i != N_particles; i++)
					{
						const glm::vec3 position = RigidWorldPosition(
							bodies, bodyIds[i], positions[0][i], positions[1][i], positions[2][i]);
						positions[0][i] = position.x;
						positions[1][i] = position.y;
						positions[2][i] = position.z;
					}
				}
				else
				{
					ReadSimStatePositions(ifs, N_total, offset, begin, N_particles, positions);
				}
				visit(type, { positions[0], positions[1], positions[2] });
			}
		};
		ScanPopulation(header.N_F, off.fluid, Base::ParticleType::Fluid);
		ScanPopulation(header.N_B, off.boundary, Base::ParticleType::Boundary);
		ScanPopulation(header.N_R, off.rigid, Base::ParticleType::Rigid);
		return header;
	}

	// resolve ownership from positions before reading any other particle fields
	template <class Particles, class Visit>
	static void ReadSelectedParticles(
		std::ifstream& ifs,
		Particles& particles,
		const uint64_t N_total,
		const uint64_t populationOffset,
		const std::function<bool(const glm::vec3&)>& select,
		Visit visitFields)
	{
		std::array<std::vector<float>, 3> positions;
		std::vector<uint32_t> selectedLocalIdxs;
		for (uint64_t beginIdx = 0; beginIdx < N_total; beginIdx += N_simStateChunk)
		{
			const uint32_t N_particles = static_cast<uint32_t>(std::min<uint64_t>(N_simStateChunk, N_total - beginIdx));
			ReadSimStatePositions(ifs, N_total, populationOffset, beginIdx, N_particles, positions);
			selectedLocalIdxs.clear();
			for (uint32_t i = 0; i < N_particles; i++)
			{
				if (select({ positions[0][i], positions[1][i], positions[2][i] })) { selectedLocalIdxs.push_back(i); }
			}
			if (selectedLocalIdxs.empty()) { continue; }

			const uint32_t N_previous = particles.N_;
			const uint32_t N_local = Sim::CheckedLocalParticleCount(
				static_cast<uint64_t>(N_previous) + selectedLocalIdxs.size(), "selected simulation state");
			uint64_t offset = populationOffset;
			const auto ReadSelectedField = [&](auto& field)
			{
				const uint64_t fieldOffset = offset;
				const uint64_t valueBytes = sizeof(field[0]);
				offset += N_total * valueBytes;
				field.resize(N_local);

				// reuse positions already read for ownership selection
				if constexpr (std::is_same_v<typename std::decay_t<decltype(field)>::value_type, float>)
				{
					const int axis = &field == &particles.xPos ? 0
						: &field == &particles.yPos ? 1
						: &field == &particles.zPos ? 2 : -1;
					if (axis >= 0)
					{
						for (size_t i = 0; i < selectedLocalIdxs.size(); i++)
						{
							field[N_previous + i] = positions[axis][selectedLocalIdxs[i]];
						}
						return;
					}
				}

				// consecutive selected indices form one contiguous file read
				for (size_t first = 0; first < selectedLocalIdxs.size();)
				{
					size_t end = first + 1;
					while (end < selectedLocalIdxs.size() && selectedLocalIdxs[end] == selectedLocalIdxs[end - 1] + 1)
					{
						end++;
					}
					ReadSimStateRange(ifs, fieldOffset + (beginIdx + selectedLocalIdxs[first]) * valueBytes,
						field.data() + N_previous + first, (end - first) * valueBytes);
					first = end;
				}
			};
			visitFields(ReadSelectedField, particles);
			ValidateParticleIds(std::span<const uint64_t>(particles.id).subspan(N_previous), "Selected state block");
			particles.N_ = N_local;
		}
	}

	static void ReadSelectedRigidParticles(
		std::ifstream& ifs,
		Sync::RigidParticles& particles,
		const Sync::RigidBodies& bodies,
		const uint64_t N_total,
		const uint64_t populationOffset,
		const std::function<bool(const glm::vec3&)>& select)
	{
		Sync::RigidParticles chunk;
		std::vector<uint32_t> selectedLocalIdxs;
		std::vector<uint64_t> bodyParticleCounts(bodies.N_, 0);
		for (uint64_t beginIdx = 0; beginIdx < N_total; beginIdx += N_simStateChunk)
		{
			chunk.resizeAll(static_cast<uint32_t>(std::min<uint64_t>(N_simStateChunk, N_total - beginIdx)));
			uint64_t offset = populationOffset;
			const auto ReadField = [&](auto& field)
			{
				const uint64_t valueBytes = sizeof(field[0]);
				ReadSimStateRange(ifs, offset + beginIdx * valueBytes, field.data(), chunk.N_ * valueBytes);
				offset += N_total * valueBytes;
			};
			VisitRigidStateFields(ReadField, chunk);
			ValidateParticleIds(chunk.id, "Rigid state block");
			selectedLocalIdxs.clear();
			for (uint32_t i = 0; i < chunk.N_; i++)
			{
				const glm::vec3 position = RigidWorldPosition(bodies, chunk.bodyId[i],
					chunk.xRelPos[i], chunk.yRelPos[i], chunk.zRelPos[i]);
				bodyParticleCounts[chunk.bodyId[i]]++;
				if (select(position)) { selectedLocalIdxs.push_back(i); }
			}
			const uint32_t N_local = Sim::CheckedLocalParticleCount(
				static_cast<uint64_t>(particles.N_) + selectedLocalIdxs.size(), "selected rigid state");
			const auto AppendSelectedField = [&](auto& destination, const auto& source)
			{
				for (const uint32_t localIdx : selectedLocalIdxs) { destination.push_back(source[localIdx]); }
			};
			VisitRigidStateFields(AppendSelectedField, particles, chunk);
			particles.N_ = N_local;
		}
		for (uint32_t bodyIdx = 0; bodyIdx < bodies.N_; bodyIdx++)
		{
			if (bodyParticleCounts[bodyIdx] != bodies.rigidParticleCount[bodyIdx])
			{
				throw std::runtime_error("Rigid particle memberships do not match body counts.");
			}
		}
	}

	BinHeaderSim SerializationMgr::LoadSelectedSimState(
		Sync::UniversalSimState& state,
		const std::string& path,
		const std::function<bool(const glm::vec3&)>& select,
		const std::function<bool(const glm::vec3&)>& selectRigid) const
	{
		const auto srcPath = ResolveStatePath(saveOptions_.baseDir, path);
		std::ifstream ifs(srcPath, std::ios::binary);
		if (!ifs)
		{
			throw std::runtime_error("Failed to open simulation state '" + srcPath.string() + "'.");
		}

		const auto header = ReadSimStateHeader(ifs, srcPath);
		const auto off = ComputeSimStateOffsets(header);
		state = {};
		state.step = header.step;
		state.time = header.time;

		// fluid and boundary state follows the caller's ownership selection
		ReadSelectedParticles(ifs, state.particlesF, header.N_F, off.fluid, select,
			[](auto visit, auto&... states) { VisitFluidStateFields(visit, states...); });
		ReadSelectedParticles(ifs, state.particlesB, header.N_B, off.boundary, select,
			[](auto visit, auto&... states) { VisitBoundaryStateFields(visit, states...); });

		// body transforms resolve rigid ownership in world space
		state.ResizeRigidBodies(static_cast<uint32_t>(header.N_RB));
		ifs.seekg(static_cast<std::streamoff>(off.rigidBodies));
		const auto ReadField = [&](auto& field) { ReadVec(ifs, field); };
		VisitRigidBodyStateFields(ReadField, state.rigidBodies);
		EnsureStreamOk(ifs, "reading", srcPath.string());
		ReadSelectedRigidParticles(ifs, state.particlesR, state.rigidBodies, header.N_R, off.rigid,
			selectRigid ? selectRigid : select);

		return header;
	}

	// --- [ ExportSnapshot (.bin) + XDMF ] --------------------------------------------------------

	// the static boundary is identical in every step, so all steps share one file
	void SerializationMgr::SaveStaticBoundaryBin(
		const Sync::ExportSnapshot& snap,
		const std::filesystem::path& dirPath) const
	{
		std::scoped_lock lock(exportDirStateMutex_);

		const std::filesystem::path dstPath = dirPath / kStaticBoundaryBinFileName;

		// a boundary-less scenario writes no file, so its absence is the finished state
		std::error_code existsEc;
		const bool dirHoldsThisBoundary =
			std::filesystem::exists(dstPath, existsEc) == (snap.N_B > 0);
		if (staticBoundaryDir_ == dirPath && dirHoldsThisBoundary) { return; }

		std::error_code removeEc;
		if (snap.N_B == 0)
		{
			std::filesystem::remove(dstPath, removeEc);
			staticBoundaryDir_ = dirPath;
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
			// posB (XYZ triples), volB

			// N_F and N_R stay zero, this file holds boundary particles only
			ExportSnapshotBinInfo info{};
			info.step = snap.simStep;
			info.time = snap.simTime;
			info.N_B = snap.N_B;
			const BinHeaderSnap hdr = MakeExportSnapshotHeader(info);

			WritePod(ofs, hdr);

			// arrays (fixed order)
			WriteXyz(ofs, snap.xPosB, snap.yPosB, snap.zPosB);
			WriteVec(ofs, snap.volB);

			EnsureStreamOk(ofs, "writing", tmpPath.string());
			ofs.close();
			EnsureStreamOk(ofs, "closing", tmpPath.string());
		}

		if (saveOptions_.atomicWrite)
		{
			AtomicReplace(tmpPath, dstPath);
		}

		staticBoundaryDir_ = dirPath;
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
		const uint32_t N_F_size = snap.N_F;
		const uint32_t N_B_size = snap.N_B;
		const uint32_t N_R_size = snap.N_R;
		assert(snap.xPosF.size() == N_F_size);
		assert(snap.yPosF.size() == N_F_size);
		assert(snap.zPosF.size() == N_F_size);
		assert(snap.velMagF.size() == N_F_size);
		assert(snap.rankF.empty() || snap.rankF.size() == N_F_size);
		assert(snap.xPosB.size() == N_B_size);
		assert(snap.yPosB.size() == N_B_size);
		assert(snap.zPosB.size() == N_B_size);
		assert(snap.volB.size() == N_B_size);
		assert(snap.bodyIdR.size() == N_R_size);
		assert(snap.xPosR.size() == N_R_size);
		assert(snap.yPosR.size() == N_R_size);
		assert(snap.zPosR.size() == N_R_size);
		assert(snap.volR.size() == N_R_size);

		SaveStaticBoundaryBin(snap, saveOptions_.baseDir / dir);

		// since a re-run overwrites the snapshots of the previous one
		{
			std::scoped_lock lock(exportDirStateMutex_);
			exportSnapshotInfos_.erase(binFileName);
		}

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
			// posF (XYZ triples), velMagF, rankF (only in distributed runs)
			// posR (XYZ triples), volR, bodyIdR
			// (boundary arrays live in the shared static boundary file)

			// N_B is informational here, the boundary arrays are not part of this file
			ExportSnapshotBinInfo info{};
			info.step = snap.simStep;
			info.time = snap.simTime;
			info.N_F = N_F_size;
			info.N_B = N_B_size;
			info.N_R = N_R_size;
			info.N_rankF = snap.rankF.size();
			const BinHeaderSnap hdr = MakeExportSnapshotHeader(info);

			WritePod(ofs, hdr);

			// arrays (fixed order)
			WriteXyz(ofs, snap.xPosF, snap.yPosF, snap.zPosF);
			WriteVec(ofs, snap.velMagF);
			WriteVec(ofs, snap.rankF);

			WriteXyz(ofs, snap.xPosR, snap.yPosR, snap.zPosR);
			WriteVec(ofs, snap.volR);
			WriteVec(ofs, snap.bodyIdR);

			EnsureStreamOk(ofs, "writing", tmpPath.string());
			ofs.close();
			EnsureStreamOk(ofs, "closing", tmpPath.string());
		}

		if (saveOptions_.atomicWrite)
		{
			AtomicReplace(tmpPath, dstPath);
		}
	}

	std::string SerializationMgr::CreateDistributedExportDirectory(
		const std::string& dir) const
	{
		const std::filesystem::path parent = std::filesystem::absolute(saveOptions_.baseDir / dir);
		std::filesystem::create_directories(parent);

		uint64_t nextIdx = 0;
		for (const auto& entry : std::filesystem::directory_iterator(parent))
		{
			const std::string name = entry.path().filename().string();
			if (!name.starts_with("run_") || name.size() == 4 ||
				!std::all_of(name.begin() + 4, name.end(), [](const char ch)
				{
					return ch >= '0' && ch <= '9';
				}))
			{
				continue;
			}
			uint64_t idx = 0;
			const auto result = std::from_chars(name.data() + 4, name.data() + name.size(), idx);
			if (result.ec != std::errc{} || idx == std::numeric_limits<uint64_t>::max())
			{
				throw std::runtime_error("Distributed export run index exceeds available uint64_t range.");
			}

			// include occupied file and symlink names when choosing the next index
			nextIdx = std::max(nextIdx, idx + 1);
		}

		for (uint64_t idx = nextIdx; ; idx++)
		{
			std::ostringstream name;
			name << "run_" << std::setw(3) << std::setfill('0') << idx;
			const std::filesystem::path path = parent / name.str();
			std::error_code ec;
			// retry if another job reserved the same directory
			if (std::filesystem::create_directory(path, ec)) { return path.string(); }
			if (ec && ec != std::errc::file_exists)
			{
				throw std::filesystem::filesystem_error("Failed to reserve distributed export directory", path, ec);
			}
			if (idx == std::numeric_limits<uint64_t>::max())
			{
				throw std::runtime_error("Distributed export run index exceeds available uint64_t range.");
			}
		}
	}

	ExportSnapshotFilePaths SerializationMgr::GetDistributedExportFilePaths(
		const std::string& dir,
		const uint64_t step,
		const bool staticBoundary) const
	{
		if (saveOptions_.tempSuffix.empty())
		{
			throw std::runtime_error("Distributed snapshots require a nonempty temporary suffix.");
		}

		ExportSnapshotFilePaths paths;
		paths.path = saveOptions_.baseDir / dir / (staticBoundary
			? std::string(kStaticBoundaryBinFileName) : "particle_data_step_" + FormatStepPadded(step) + ".bin");
		paths.writePath = paths.path;
		paths.writePath += saveOptions_.tempSuffix;
		if (std::filesystem::exists(paths.path) || std::filesystem::exists(paths.writePath))
		{
			throw std::runtime_error("Distributed snapshot destination already exists: " + paths.path.string());
		}

		return paths;
	}

	void SerializationMgr::PublishDistributedExportFile(
		const ExportSnapshotFilePaths& paths,
		const uint64_t N_expectedBytes) const
	{
		if (!std::filesystem::is_regular_file(paths.writePath) || std::filesystem::exists(paths.path))
		{
			throw std::runtime_error("Cannot publish distributed snapshot: " + paths.path.string());
		}
		if (std::filesystem::file_size(paths.writePath) != N_expectedBytes)
		{
			throw std::runtime_error("Unexpected distributed snapshot file size: " + paths.writePath.string());
		}
		AtomicReplace(paths.writePath, paths.path);
	}

	void SerializationMgr::SaveDistributedExportSnapshotXdmfViews(
		const uint64_t step,
		const double time,
		const std::vector<uint64_t>& rankCounts,
		const std::string& dir) const
	{
		const DistributedExportLayout layout = ComputeDistributedExportLayout(rankCounts, 0);
		const std::filesystem::path dirPath = saveOptions_.baseDir / dir;
		const std::string binFileName = "particle_data_step_" + FormatStepPadded(step) + ".bin";
		std::scoped_lock lock(exportDirStateMutex_);
		if (distributedExportDir_ != dirPath)
		{
			distributedExportSnapshotInfos_.clear();
			distributedExportDir_ = dirPath;
		}
		if (!distributedExportSnapshotInfos_.empty() &&
			step <= distributedExportSnapshotInfos_.rbegin()->first)
		{
			throw std::runtime_error("Distributed export steps must increase within a snapshot directory.");
		}

		// validate the new shared files before adding the frame
		ExportSnapshotBinInfo info{};
		ExportSnapshotBinInfo boundaryInfo{};
		const bool firstFrame = distributedExportSnapshotInfos_.empty();
		if (!TryReadExportSnapshotBinInfo(dirPath / binFileName, info) ||
			info.step != step || info.time != time || info.N_rankF != info.N_F ||
			info.N_F != layout.N_total[0] || info.N_B != layout.N_total[1] || info.N_R != layout.N_total[2])
		{
			throw std::runtime_error("Missing or invalid distributed snapshot '" + (dirPath / binFileName).string() + "'.");
		}
		if (firstFrame && info.N_B > 0 &&
			(!TryReadStaticBoundaryBinInfo(dirPath / kStaticBoundaryBinFileName, boundaryInfo) ||
				boundaryInfo.N_B != info.N_B || boundaryInfo.step != step || boundaryInfo.time != time))
		{
			throw std::runtime_error("Missing or invalid distributed static boundary.");
		}

		// replace each XDMF file through a temporary file
		SaveOptions options = saveOptions_;
		options.atomicWrite = true;
		distributedExportSnapshotInfos_.emplace(step, info);
		try
		{
			const auto WriteCollection = [&](
				const char* fileName,
				const char* name,
				XdmfGridWriter writeGrid)
			{
				WriteXdmfFile(dirPath / fileName, options, [&](std::ofstream& ofs)
				{
					ofs << "    <Grid Name=\"" << name
						<< "\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
					for (const auto& frame : distributedExportSnapshotInfos_)
					{
						writeGrid(ofs, frame.second);
					}
					ofs << "    </Grid>\n";
				});
			};
			if (firstFrame && info.N_B > 0)
			{
				WriteXdmfStaticBoundaryFile(
					dirPath / "static_boundary.xdmf", boundaryInfo, boundaryInfo.fileName, options);
			}
			const bool hasRigid = std::any_of(
				distributedExportSnapshotInfos_.begin(),
				distributedExportSnapshotInfos_.end(),
				[](const auto& frame)
				{
					return frame.second.N_R > 0;
				});
			if (hasRigid) { WriteCollection("rigid_particles.xdmf", "RigidParticles", WriteRigidParticleGrid); }
			WriteCollection("fluid_surface.xdmf", "FluidSurface", WriteFluidSurfaceGrid);
		}
		catch (...)
		{
			distributedExportSnapshotInfos_.erase(step);
			throw;
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
		[[maybe_unused]] const uint32_t N_F_size = snap.N_F;
		[[maybe_unused]] const uint32_t N_B_size = snap.N_B;
		[[maybe_unused]] const uint32_t N_R_size = snap.N_R;
		assert(snap.xPosF.size() == N_F_size);
		assert(snap.yPosF.size() == N_F_size);
		assert(snap.zPosF.size() == N_F_size);
		assert(snap.velMagF.size() == N_F_size);
		assert(snap.rankF.empty() || snap.rankF.size() == N_F_size);
		assert(snap.xPosB.size() == N_B_size);
		assert(snap.yPosB.size() == N_B_size);
		assert(snap.zPosB.size() == N_B_size);
		assert(snap.volB.size() == N_B_size);
		assert(snap.bodyIdR.size() == N_R_size);
		assert(snap.xPosR.size() == N_R_size);
		assert(snap.yPosR.size() == N_R_size);
		assert(snap.zPosR.size() == N_R_size);
		assert(snap.volR.size() == N_R_size);

		std::vector<ExportSnapshotBinInfo> infos;
		{
			std::scoped_lock lock(exportDirStateMutex_);

			// snapshots of another directory say nothing about this one
			if (exportSnapshotInfosDir_ != dirPath)
			{
				exportSnapshotInfos_.clear();
				exportSnapshotInfosDir_ = dirPath;
			}

			infos = CollectExportSnapshotBinInfos(dirPath, exportSnapshotInfos_);
		}
		if (infos.empty())
		{
			throw std::runtime_error("No export snapshot binary files found in '" +
				dirPath.string() + "'.");
		}

		WriteXdmfCollectionFile(
			fluidXdmfPath, "FluidSurface", infos, WriteFluidSurfaceGrid, saveOptions_);

		// the boundary file itself decides whether the view exists
		std::error_code removeEc;
		ExportSnapshotBinInfo boundaryInfo{};
		if (TryReadStaticBoundaryBinInfo(
			dirPath / kStaticBoundaryBinFileName, boundaryInfo))
		{
			WriteXdmfStaticBoundaryFile(staticBoundaryXdmfPath, boundaryInfo,
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
