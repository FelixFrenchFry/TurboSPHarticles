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
	static constexpr char kSimStateMagic[8] =
		{ 'T', 'S', 'P', 'S', 'I', 'M', '\3', '\0' };
	static constexpr char kExportSnapshotMagic[8] =
		{ 'T', 'S', 'P', 'S', 'N', 'A', 'P', '\3' };

	static bool IsLittleEndian()
	{
		const uint32_t x = 0x01020304u;
		return *reinterpret_cast<const uint8_t*>(&x) == 0x04u;
	}

	#pragma pack(push, 1)
	struct BinHeaderSim
	{
		char magic[8];        // sim-state file magic
		uint32_t endianTag;   // kEndianLE
		uint32_t headerBytes; // sizeof(BinHeaderSim)

		uint32_t step;
		double time;

		uint32_t NF;
		uint32_t NB;
		uint32_t NR;
		uint32_t NS;
	};

	struct BinHeaderSnap
	{
		char magic[8];        // export-snapshot file magic
		uint32_t endianTag;   // kEndianLE
		uint32_t headerBytes; // sizeof(BinHeaderSnap)

		uint32_t step;
		double time;

		uint32_t NF;
		uint32_t NB;
		uint32_t NB_static;
		uint32_t NB_rigid;
	};
	#pragma pack(pop)

	struct BoundaryExportViews
	{
		uint32_t NB_static = 0;
		uint32_t NB_rigid = 0;

		std::vector<uint32_t> idB_static;
		std::vector<float> xPosB_static;
		std::vector<float> yPosB_static;
		std::vector<float> zPosB_static;
		std::vector<float> volB_static;

		std::vector<uint32_t> idB_rigid;
		std::vector<uint32_t> bodyIdB_rigid;
		std::vector<float> xPosB_rigid;
		std::vector<float> yPosB_rigid;
		std::vector<float> zPosB_rigid;
		std::vector<float> volB_rigid;
	};

	static BoundaryExportViews BuildBoundaryExportViews(const Sync::ExportSnapshot& snap)
	{
		assert(snap.bodyIdB.empty() || snap.bodyIdB.size() == snap.N_B);

		BoundaryExportViews views{};
		views.idB_static.reserve(snap.N_B);
		views.xPosB_static.reserve(snap.N_B);
		views.yPosB_static.reserve(snap.N_B);
		views.zPosB_static.reserve(snap.N_B);
		views.volB_static.reserve(snap.N_B);
		views.idB_rigid.reserve(snap.N_B);
		views.bodyIdB_rigid.reserve(snap.N_B);
		views.xPosB_rigid.reserve(snap.N_B);
		views.yPosB_rigid.reserve(snap.N_B);
		views.zPosB_rigid.reserve(snap.N_B);
		views.volB_rigid.reserve(snap.N_B);

		const bool hasBodyIdB = snap.bodyIdB.size() == snap.N_B;
		for (uint32_t i = 0; i < snap.N_B; i++)
		{
			const bool isRigidB = hasBodyIdB && snap.bodyIdB[i] != Sync::InvalidRigidBodyId;

			if (isRigidB)
			{
				views.idB_rigid.push_back(snap.idB[i]);
				views.bodyIdB_rigid.push_back(snap.bodyIdB[i]);
				views.xPosB_rigid.push_back(snap.xPosB[i]);
				views.yPosB_rigid.push_back(snap.yPosB[i]);
				views.zPosB_rigid.push_back(snap.zPosB[i]);
				views.volB_rigid.push_back(snap.volB[i]);
			}
			else
			{
				views.idB_static.push_back(snap.idB[i]);
				views.xPosB_static.push_back(snap.xPosB[i]);
				views.yPosB_static.push_back(snap.yPosB[i]);
				views.zPosB_static.push_back(snap.zPosB[i]);
				views.volB_static.push_back(snap.volB[i]);
			}
		}

		views.NB_static = static_cast<uint32_t>(views.idB_static.size());
		views.NB_rigid = static_cast<uint32_t>(views.idB_rigid.size());
		return views;
	}

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
		assert(B.bodyId.size() == N && B.rigidSampleIdx.size() == N);
		assert(B.xPos.size() == N && B.yPos.size() == N && B.zPos.size() == N);
		assert(B.xVel.size() == N && B.yVel.size() == N && B.zVel.size() == N);
		assert(B.volB.size() == N && B.psiB.size() == N);
	}

	static void CheckRigidBodiesConsistent(const Sync::RigidBodies& R)
	{
		[[maybe_unused]] const auto N = static_cast<size_t>(R.N_);
		[[maybe_unused]] const auto NS = static_cast<size_t>(R.sampleN_);

		assert(R.id.size() == N);
		assert(R.sampleBegin.size() == N && R.sampleCount.size() == N);
		assert(R.xCM.size() == N && R.yCM.size() == N && R.zCM.size() == N);
		assert(R.xVel.size() == N && R.yVel.size() == N && R.zVel.size() == N);
		assert(R.xOmega.size() == N && R.yOmega.size() == N && R.zOmega.size() == N);
		assert(R.xAngMom.size() == N && R.yAngMom.size() == N && R.zAngMom.size() == N);
		assert(R.xForce.size() == N && R.yForce.size() == N && R.zForce.size() == N);
		assert(R.xTorque.size() == N && R.yTorque.size() == N && R.zTorque.size() == N);
		assert(R.a00.size() == N && R.a01.size() == N && R.a02.size() == N);
		assert(R.a10.size() == N && R.a11.size() == N && R.a12.size() == N);
		assert(R.a20.size() == N && R.a21.size() == N && R.a22.size() == N);
		assert(R.mass.size() == N && R.massInv.size() == N);
		assert(R.iBodyInv00.size() == N && R.iBodyInv01.size() == N && R.iBodyInv02.size() == N);
		assert(R.iBodyInv10.size() == N && R.iBodyInv11.size() == N && R.iBodyInv12.size() == N);
		assert(R.iBodyInv20.size() == N && R.iBodyInv21.size() == N && R.iBodyInv22.size() == N);
		assert(R.xLocalB.size() == NS && R.yLocalB.size() == NS && R.zLocalB.size() == NS);

		for (uint32_t bodyIdx = 0; bodyIdx < R.N_; bodyIdx++)
		{
			const uint32_t sampleBegin = R.sampleBegin[bodyIdx];
			const uint32_t sampleCount = R.sampleCount[bodyIdx];
			assert(sampleBegin <= R.sampleN_);
			assert(sampleCount <= R.sampleN_ - sampleBegin);
		}
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
		CheckRigidBodiesConsistent(state.rigidBodies);

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
			std::memcpy(hdr.magic, kSimStateMagic, sizeof(hdr.magic));
			hdr.endianTag = kEndianLE;
			hdr.headerBytes = static_cast<uint32_t>(sizeof(BinHeaderSim));

			hdr.step = state.step;
			hdr.time = state.time;

			hdr.NF = state.fluidParticles.N_;
			hdr.NB = state.boundaryParticles.N_;
			hdr.NR = state.rigidBodies.N_;
			hdr.NS = state.rigidBodies.sampleN_;

			WritePod(ofs, hdr);

			// fluid data (fixed order)
			{
				const auto& F = state.fluidParticles;

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

			// boundary data (fixed order)
			{
				const auto& B = state.boundaryParticles;

				WriteVec(ofs, B.id);
				WriteVec(ofs, B.bodyId);
				WriteVec(ofs, B.rigidSampleIdx);
				WriteVec(ofs, B.xPos);
				WriteVec(ofs, B.yPos);
				WriteVec(ofs, B.zPos);

				WriteVec(ofs, B.xVel);
				WriteVec(ofs, B.yVel);
				WriteVec(ofs, B.zVel);

				WriteVec(ofs, B.volB);
				WriteVec(ofs, B.psiB);
			}

			// rigid body data (fixed order)
			{
				const auto& R = state.rigidBodies;

				WriteVec(ofs, R.id);
				WriteVec(ofs, R.sampleBegin);
				WriteVec(ofs, R.sampleCount);

				WriteVec(ofs, R.xCM);
				WriteVec(ofs, R.yCM);
				WriteVec(ofs, R.zCM);

				WriteVec(ofs, R.xVel);
				WriteVec(ofs, R.yVel);
				WriteVec(ofs, R.zVel);

				WriteVec(ofs, R.xOmega);
				WriteVec(ofs, R.yOmega);
				WriteVec(ofs, R.zOmega);

				WriteVec(ofs, R.xAngMom);
				WriteVec(ofs, R.yAngMom);
				WriteVec(ofs, R.zAngMom);

				WriteVec(ofs, R.a00);
				WriteVec(ofs, R.a01);
				WriteVec(ofs, R.a02);
				WriteVec(ofs, R.a10);
				WriteVec(ofs, R.a11);
				WriteVec(ofs, R.a12);
				WriteVec(ofs, R.a20);
				WriteVec(ofs, R.a21);
				WriteVec(ofs, R.a22);

				WriteVec(ofs, R.iBodyInv00);
				WriteVec(ofs, R.iBodyInv01);
				WriteVec(ofs, R.iBodyInv02);
				WriteVec(ofs, R.iBodyInv10);
				WriteVec(ofs, R.iBodyInv11);
				WriteVec(ofs, R.iBodyInv12);
				WriteVec(ofs, R.iBodyInv20);
				WriteVec(ofs, R.iBodyInv21);
				WriteVec(ofs, R.iBodyInv22);

				WriteVec(ofs, R.mass);
				WriteVec(ofs, R.massInv);
				WriteVec(ofs, R.xForce);
				WriteVec(ofs, R.yForce);
				WriteVec(ofs, R.zForce);
				WriteVec(ofs, R.xTorque);
				WriteVec(ofs, R.yTorque);
				WriteVec(ofs, R.zTorque);

				WriteVec(ofs, R.xLocalB);
				WriteVec(ofs, R.yLocalB);
				WriteVec(ofs, R.zLocalB);
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

		// header
		BinHeaderSim hdr{};
		ReadPod(ifs, hdr);

		// basic validation
		if (std::memcmp(hdr.magic, kSimStateMagic, sizeof(hdr.magic)) != 0)
		{
			SPDLOG_ERROR("Invalid sim state file magic: {}", path);
			return;
		}
		if (!IsLittleEndian() || hdr.endianTag != kEndianLE)
		{
			SPDLOG_ERROR("Endianness mismatch (file is LE, host must be LE): {}", path);
			return;
		}
		if (hdr.headerBytes != sizeof(BinHeaderSim))
		{
			SPDLOG_ERROR("Unexpected sim header size ({}): {}",
				hdr.headerBytes, path);
			return;
		}

		// metadata
		state.step = hdr.step;
		state.time = hdr.time;

		// resize storage
		state.ResizeFluid(hdr.NF);
		state.ResizeBoundary(hdr.NB);
		state.ResizeRigidBodies(hdr.NR);
		state.rigidBodies.resizeSamples(hdr.NS);

		// fluid data (fixed order)
		{
			auto& F = state.fluidParticles;

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

		// boundary data (fixed order)
		{
			auto& B = state.boundaryParticles;

			ReadVec(ifs, B.id);
			ReadVec(ifs, B.bodyId);
			ReadVec(ifs, B.rigidSampleIdx);
			ReadVec(ifs, B.xPos);
			ReadVec(ifs, B.yPos);
			ReadVec(ifs, B.zPos);

			ReadVec(ifs, B.xVel);
			ReadVec(ifs, B.yVel);
			ReadVec(ifs, B.zVel);

			ReadVec(ifs, B.volB);
			ReadVec(ifs, B.psiB);

			// sanity check
			B.N_ = hdr.NB;
		}

		// rigid body data (fixed order)
		{
			auto& R = state.rigidBodies;

			ReadVec(ifs, R.id);
			ReadVec(ifs, R.sampleBegin);
			ReadVec(ifs, R.sampleCount);

			ReadVec(ifs, R.xCM);
			ReadVec(ifs, R.yCM);
			ReadVec(ifs, R.zCM);

			ReadVec(ifs, R.xVel);
			ReadVec(ifs, R.yVel);
			ReadVec(ifs, R.zVel);

			ReadVec(ifs, R.xOmega);
			ReadVec(ifs, R.yOmega);
			ReadVec(ifs, R.zOmega);

			ReadVec(ifs, R.xAngMom);
			ReadVec(ifs, R.yAngMom);
			ReadVec(ifs, R.zAngMom);

			ReadVec(ifs, R.a00);
			ReadVec(ifs, R.a01);
			ReadVec(ifs, R.a02);
			ReadVec(ifs, R.a10);
			ReadVec(ifs, R.a11);
			ReadVec(ifs, R.a12);
			ReadVec(ifs, R.a20);
			ReadVec(ifs, R.a21);
			ReadVec(ifs, R.a22);

			ReadVec(ifs, R.iBodyInv00);
			ReadVec(ifs, R.iBodyInv01);
			ReadVec(ifs, R.iBodyInv02);
			ReadVec(ifs, R.iBodyInv10);
			ReadVec(ifs, R.iBodyInv11);
			ReadVec(ifs, R.iBodyInv12);
			ReadVec(ifs, R.iBodyInv20);
			ReadVec(ifs, R.iBodyInv21);
			ReadVec(ifs, R.iBodyInv22);

			ReadVec(ifs, R.mass);
			ReadVec(ifs, R.massInv);
			ReadVec(ifs, R.xForce);
			ReadVec(ifs, R.yForce);
			ReadVec(ifs, R.zForce);
			ReadVec(ifs, R.xTorque);
			ReadVec(ifs, R.yTorque);
			ReadVec(ifs, R.zTorque);

			ReadVec(ifs, R.xLocalB);
			ReadVec(ifs, R.yLocalB);
			ReadVec(ifs, R.zLocalB);

			R.N_ = hdr.NR;
			R.sampleN_ = hdr.NS;
		}

		EnsureStreamOk(ifs, "reading", path);
	}

	// --- [ ExportSnapshot (.bin) + XDMF ] --------------------------------------------------------

	void SerializationMgr::SaveExportSnapshot(
		const Sync::ExportSnapshot& snap,
		const std::string& dir) const
	{
		const std::string baseName =
			"particle_data_step_" + FormatStepPadded(snap.simStep) + ".bin";

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
		assert(snap.idF.size() == NF_size);
		assert(snap.xPosF.size() == NF_size);
		assert(snap.yPosF.size() == NF_size);
		assert(snap.zPosF.size() == NF_size);
		assert(snap.velMagF.size() == NF_size);
		assert(snap.idB.size() == NB_size);
		assert(snap.bodyIdB.empty() || snap.bodyIdB.size() == NB_size);
		assert(snap.xPosB.size() == NB_size);
		assert(snap.yPosB.size() == NB_size);
		assert(snap.zPosB.size() == NB_size);
		assert(snap.volB.size() == NB_size);

		const BoundaryExportViews boundaryViews = BuildBoundaryExportViews(snap);
		const uint32_t NB_static = boundaryViews.NB_static;
		const uint32_t NB_rigid = boundaryViews.NB_rigid;

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
			// xPosF, yPosF, zPosF, velMagF, idF
			// xPosB_static, yPosB_static, zPosB_static, volB_static, idB_static
			// xPosB_rigid, yPosB_rigid, zPosB_rigid, volB_rigid, idB_rigid, bodyIdB_rigid

			BinHeaderSnap hdr{};
			std::memcpy(hdr.magic, kExportSnapshotMagic, sizeof(hdr.magic));
			hdr.endianTag = kEndianLE;
			hdr.headerBytes = static_cast<uint32_t>(sizeof(BinHeaderSnap));
			hdr.step = snap.simStep;
			hdr.time = snap.simTime;
			hdr.NF = NF_size;
			hdr.NB = NB_size;
			hdr.NB_static = NB_static;
			hdr.NB_rigid = NB_rigid;

			WritePod(ofs, hdr);

			// arrays (fixed order)
			WriteVec(ofs, snap.xPosF);
			WriteVec(ofs, snap.yPosF);
			WriteVec(ofs, snap.zPosF);
			WriteVec(ofs, snap.velMagF);
			WriteVec(ofs, snap.idF);

			WriteVec(ofs, boundaryViews.xPosB_static);
			WriteVec(ofs, boundaryViews.yPosB_static);
			WriteVec(ofs, boundaryViews.zPosB_static);
			WriteVec(ofs, boundaryViews.volB_static);
			WriteVec(ofs, boundaryViews.idB_static);

			WriteVec(ofs, boundaryViews.xPosB_rigid);
			WriteVec(ofs, boundaryViews.yPosB_rigid);
			WriteVec(ofs, boundaryViews.zPosB_rigid);
			WriteVec(ofs, boundaryViews.volB_rigid);
			WriteVec(ofs, boundaryViews.idB_rigid);
			WriteVec(ofs, boundaryViews.bodyIdB_rigid);

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
		const std::string step = FormatStepPadded(snap.simStep);
		const std::string baseName_bin = "particle_data_step_" + step + ".bin";
		const std::string fluidName_xdmf = "fluid_surface_step_" + step + ".xdmf";
		const std::string staleBoundaryName_xdmf = "boundary_step_" + step + ".xdmf";
		const std::string staticBoundaryName_xdmf = "static_boundary_step_" + step + ".xdmf";
		const std::string rigidBoundaryName_xdmf = "rigid_boundary_step_" + step + ".xdmf";

		// ensure export directory exists
		std::error_code ec;
		std::filesystem::create_directories(saveOptions_.baseDir / dir, ec);
		if (ec)
		{
			SPDLOG_ERROR("Failed to create export directory '{}': {}",
				saveOptions_.baseDir.string(), ec.message());
			return;
		}

		// remove stale boundary views for this step
		std::error_code removeEc;
		std::filesystem::remove(
			saveOptions_.baseDir / dir / staleBoundaryName_xdmf,
			removeEc);
		std::filesystem::remove(
			saveOptions_.baseDir / dir / staticBoundaryName_xdmf,
			removeEc);
		std::filesystem::remove(
			saveOptions_.baseDir / dir / rigidBoundaryName_xdmf,
			removeEc);

		const std::filesystem::path fluidXdmfPath =
			saveOptions_.baseDir / dir / fluidName_xdmf;

		// open fluid XDMF view
		std::ofstream ofs(fluidXdmfPath, std::ios::out | std::ios::trunc);
		if (!ofs)
		{
			SPDLOG_ERROR("Failed to open XDMF file for write: {}",
				fluidXdmfPath.string());
			return;
		}

		// make sure data for every particle is present
		const uint32_t NF_size = snap.N_F;
		const uint32_t NB_size = snap.N_B;
		assert(snap.idF.size() == NF_size);
		assert(snap.xPosF.size() == NF_size);
		assert(snap.yPosF.size() == NF_size);
		assert(snap.zPosF.size() == NF_size);
		assert(snap.velMagF.size() == NF_size);
		assert(snap.idB.size() == NB_size);
		assert(snap.bodyIdB.empty() || snap.bodyIdB.size() == NB_size);
		assert(snap.xPosB.size() == NB_size);
		assert(snap.yPosB.size() == NB_size);
		assert(snap.zPosB.size() == NB_size);
		assert(snap.volB.size() == NB_size);

		const BoundaryExportViews boundaryViews = BuildBoundaryExportViews(snap);
		const uint32_t NB_static = boundaryViews.NB_static;
		const uint32_t NB_rigid = boundaryViews.NB_rigid;

		// byte offsets (Seek) into the .bin file
		//
		// layout:
		// [BinHeaderSnap]
		// xPosF[NF] (float32)
		// yPosF[NF] (float32)
		// zPosF[NF] (float32)
		// velMagF[NF] (float32)
		// idF[NF] (uint32)
		// xPosB_static[NB_static] (float32)
		// yPosB_static[NB_static] (float32)
		// zPosB_static[NB_static] (float32)
		// volB_static[NB_static] (float32)
		// idB_static[NB_static] (uint32)
		// xPosB_rigid[NB_rigid] (float32)
		// yPosB_rigid[NB_rigid] (float32)
		// zPosB_rigid[NB_rigid] (float32)
		// volB_rigid[NB_rigid] (float32)
		// idB_rigid[NB_rigid] (uint32)
		// bodyIdB_rigid[NB_rigid] (uint32)
		const uint64_t headerBytes = sizeof(BinHeaderSnap);
		const uint64_t elemBytes = 4; // float32 / uint32

		const uint64_t off_xF = headerBytes;
		const uint64_t off_yF = off_xF + uint64_t(NF_size) * elemBytes;
		const uint64_t off_zF = off_yF + uint64_t(NF_size) * elemBytes;
		const uint64_t off_vF = off_zF + uint64_t(NF_size) * elemBytes;
		const uint64_t off_idF = off_vF + uint64_t(NF_size) * elemBytes;
		const uint64_t off_xPosB_static = off_idF + uint64_t(NF_size) * elemBytes;
		const uint64_t off_yPosB_static = off_xPosB_static + uint64_t(NB_static) * elemBytes;
		const uint64_t off_zPosB_static = off_yPosB_static + uint64_t(NB_static) * elemBytes;
		const uint64_t off_volB_static = off_zPosB_static + uint64_t(NB_static) * elemBytes;
		const uint64_t off_idB_static = off_volB_static + uint64_t(NB_static) * elemBytes;
		const uint64_t off_xPosB_rigid = off_idB_static + uint64_t(NB_static) * elemBytes;
		const uint64_t off_yPosB_rigid = off_xPosB_rigid + uint64_t(NB_rigid) * elemBytes;
		const uint64_t off_zPosB_rigid = off_yPosB_rigid + uint64_t(NB_rigid) * elemBytes;
		const uint64_t off_volB_rigid = off_zPosB_rigid + uint64_t(NB_rigid) * elemBytes;
		const uint64_t off_idB_rigid = off_volB_rigid + uint64_t(NB_rigid) * elemBytes;
		const uint64_t off_bodyIdB_rigid = off_idB_rigid + uint64_t(NB_rigid) * elemBytes;

		// XDMF header
		ofs << R"(<?xml version="1.0" ?>)" << "\n";
		ofs << R"(<Xdmf Version="3.0">)" << "\n";
		ofs << R"(  <Domain>)" << "\n";

		// one uniform grid for exported fluid particles
		ofs << "    <Grid Name=\"FluidSurface\" GridType=\"Uniform\">" << "\n";

		// time metadata
		ofs << "      <Time Value=\""
			<< std::setprecision(16) << snap.simTime
			<< "\" />\n";

		// topology of point cloud with NF vertices
		ofs << "      <Topology TopologyType=\"Polyvertex\" "
			<< "Dimensions=\"" << NF_size << "\"/>\n";

		// geometry of XYZ coordinates
		ofs << "      <Geometry Name=\"posF\" GeometryType=\"XYZ\">\n";
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

		// exported fluid particle id (Binary)
		ofs << "      <Attribute Name=\"idF\" "
			<< "AttributeType=\"Scalar\" Center=\"Node\">\n";
		ofs << "        <DataItem Dimensions=\"" << NF_size
			<< "\" NumberType=\"UInt\" Precision=\"4\" "
			<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
			<< off_idF << "\">\n";
		ofs << "          " << baseName_bin << "\n";
		ofs << "        </DataItem>\n";
		ofs << "      </Attribute>\n";

		ofs << "    </Grid>\n";
		ofs << "  </Domain>\n";
		ofs << "</Xdmf>\n";
		EnsureStreamOk(ofs, "writing", fluidXdmfPath.string());
		ofs.close();

		if (NB_static > 0)
		{
			const std::filesystem::path staticBoundaryXdmfPath =
				saveOptions_.baseDir / dir / staticBoundaryName_xdmf;

			ofs.clear();
			ofs.open(staticBoundaryXdmfPath, std::ios::out | std::ios::trunc);
			if (!ofs)
			{
				SPDLOG_ERROR("Failed to open XDMF file for write: {}",
					staticBoundaryXdmfPath.string());
				return;
			}

			// XDMF header
			ofs << R"(<?xml version="1.0" ?>)" << "\n";
			ofs << R"(<Xdmf Version="3.0">)" << "\n";
			ofs << R"(  <Domain>)" << "\n";

			// one uniform grid for exported static boundary particles
			ofs << "    <Grid Name=\"StaticBoundary\" GridType=\"Uniform\">" << "\n";

			// time metadata
			ofs << "      <Time Value=\""
				<< std::setprecision(16) << snap.simTime
				<< "\" />\n";

			// topology of point cloud with NB_static vertices
			ofs << "      <Topology TopologyType=\"Polyvertex\" "
				<< "Dimensions=\"" << NB_static << "\"/>\n";

			// geometry of XYZ coordinates
			ofs << "      <Geometry Name=\"posStaticB\" GeometryType=\"XYZ\">\n";
			ofs << "        <DataItem ItemType=\"Function\" "
				<< "Function=\"JOIN($0, $1, $2)\" "
				<< "Dimensions=\"" << NB_static << " 3\" "
				<< "NumberType=\"Float\" Precision=\"4\">\n";

			// x pos (Binary)
			ofs << "          <DataItem Dimensions=\"" << NB_static
				<< "\" NumberType=\"Float\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_xPosB_static << "\">\n";
			ofs << "            " << baseName_bin << "\n";
			ofs << "          </DataItem>\n";

			// y pos (Binary)
			ofs << "          <DataItem Dimensions=\"" << NB_static
				<< "\" NumberType=\"Float\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_yPosB_static << "\">\n";
			ofs << "            " << baseName_bin << "\n";
			ofs << "          </DataItem>\n";

			// z pos (Binary)
			ofs << "          <DataItem Dimensions=\"" << NB_static
				<< "\" NumberType=\"Float\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_zPosB_static << "\">\n";
			ofs << "            " << baseName_bin << "\n";
			ofs << "          </DataItem>\n";

			ofs << "        </DataItem>\n";
			ofs << "      </Geometry>\n";

			// represented boundary volume (Binary)
			ofs << "      <Attribute Name=\"volStaticB\" "
				<< "AttributeType=\"Scalar\" Center=\"Node\">\n";
			ofs << "        <DataItem Dimensions=\"" << NB_static
				<< "\" NumberType=\"Float\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_volB_static << "\">\n";
			ofs << "          " << baseName_bin << "\n";
			ofs << "        </DataItem>\n";
			ofs << "      </Attribute>\n";

			// exported boundary particle id (Binary)
			ofs << "      <Attribute Name=\"idStaticB\" "
				<< "AttributeType=\"Scalar\" Center=\"Node\">\n";
			ofs << "        <DataItem Dimensions=\"" << NB_static
				<< "\" NumberType=\"UInt\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_idB_static << "\">\n";
			ofs << "          " << baseName_bin << "\n";
			ofs << "        </DataItem>\n";
			ofs << "      </Attribute>\n";

			ofs << "    </Grid>\n";
			ofs << "  </Domain>\n";
			ofs << "</Xdmf>\n";
			EnsureStreamOk(ofs, "writing", staticBoundaryXdmfPath.string());
			ofs.close();
		}

		if (NB_rigid > 0)
		{
			const std::filesystem::path rigidBoundaryXdmfPath =
				saveOptions_.baseDir / dir / rigidBoundaryName_xdmf;

			ofs.clear();
			ofs.open(rigidBoundaryXdmfPath, std::ios::out | std::ios::trunc);
			if (!ofs)
			{
				SPDLOG_ERROR("Failed to open XDMF file for write: {}",
					rigidBoundaryXdmfPath.string());
				return;
			}

			// XDMF header
			ofs << R"(<?xml version="1.0" ?>)" << "\n";
			ofs << R"(<Xdmf Version="3.0">)" << "\n";
			ofs << R"(  <Domain>)" << "\n";

			// one uniform grid for exported rigid boundary particles
			ofs << "    <Grid Name=\"RigidBoundary\" GridType=\"Uniform\">" << "\n";

			// time metadata
			ofs << "      <Time Value=\""
				<< std::setprecision(16) << snap.simTime
				<< "\" />\n";

			// topology of point cloud with NB_rigid vertices
			ofs << "      <Topology TopologyType=\"Polyvertex\" "
				<< "Dimensions=\"" << NB_rigid << "\"/>\n";

			// geometry of XYZ coordinates
			ofs << "      <Geometry Name=\"posRigidB\" GeometryType=\"XYZ\">\n";
			ofs << "        <DataItem ItemType=\"Function\" "
				<< "Function=\"JOIN($0, $1, $2)\" "
				<< "Dimensions=\"" << NB_rigid << " 3\" "
				<< "NumberType=\"Float\" Precision=\"4\">\n";

			// x pos (Binary)
			ofs << "          <DataItem Dimensions=\"" << NB_rigid
				<< "\" NumberType=\"Float\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_xPosB_rigid << "\">\n";
			ofs << "            " << baseName_bin << "\n";
			ofs << "          </DataItem>\n";

			// y pos (Binary)
			ofs << "          <DataItem Dimensions=\"" << NB_rigid
				<< "\" NumberType=\"Float\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_yPosB_rigid << "\">\n";
			ofs << "            " << baseName_bin << "\n";
			ofs << "          </DataItem>\n";

			// z pos (Binary)
			ofs << "          <DataItem Dimensions=\"" << NB_rigid
				<< "\" NumberType=\"Float\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_zPosB_rigid << "\">\n";
			ofs << "            " << baseName_bin << "\n";
			ofs << "          </DataItem>\n";

			ofs << "        </DataItem>\n";
			ofs << "      </Geometry>\n";

			// represented boundary volume (Binary)
			ofs << "      <Attribute Name=\"volRigidB\" "
				<< "AttributeType=\"Scalar\" Center=\"Node\">\n";
			ofs << "        <DataItem Dimensions=\"" << NB_rigid
				<< "\" NumberType=\"Float\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_volB_rigid << "\">\n";
			ofs << "          " << baseName_bin << "\n";
			ofs << "        </DataItem>\n";
			ofs << "      </Attribute>\n";

			// exported boundary particle id (Binary)
			ofs << "      <Attribute Name=\"idRigidB\" "
				<< "AttributeType=\"Scalar\" Center=\"Node\">\n";
			ofs << "        <DataItem Dimensions=\"" << NB_rigid
				<< "\" NumberType=\"UInt\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_idB_rigid << "\">\n";
			ofs << "          " << baseName_bin << "\n";
			ofs << "        </DataItem>\n";
			ofs << "      </Attribute>\n";

			// rigid body id (Binary)
			ofs << "      <Attribute Name=\"bodyIdRigidB\" "
				<< "AttributeType=\"Scalar\" Center=\"Node\">\n";
			ofs << "        <DataItem Dimensions=\"" << NB_rigid
				<< "\" NumberType=\"UInt\" Precision=\"4\" "
				<< "Format=\"Binary\" Endian=\"Little\" Seek=\""
				<< off_bodyIdB_rigid << "\">\n";
			ofs << "          " << baseName_bin << "\n";
			ofs << "        </DataItem>\n";
			ofs << "      </Attribute>\n";

			ofs << "    </Grid>\n";
			ofs << "  </Domain>\n";
			ofs << "</Xdmf>\n";
			EnsureStreamOk(ofs, "writing", rigidBoundaryXdmfPath.string());
		}
	}

} // namespace Export
