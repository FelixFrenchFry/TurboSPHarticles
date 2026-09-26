// dist_snapshots.cpp

// external
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// internal
#include "dist_snapshots.h"
#include "mpi_context.h"
#include "source/eval/serial/serialization.h"



bool Dist::SaveDistributedExportSnapshot(
	const Sync::ExportSnapshot& exportSnap,
	const Export::SerializationMgr& serializationMgr,
	const std::string& dir,
	const bool includeB,
	const MpiContext& mpiContext,
	std::vector<uint64_t>& rankCounts)
{
	try
	{
		bool writeOk = true;
		try
		{
			if (mpiContext.Size() > int{std::numeric_limits<uint16_t>::max()} + 1 ||
				!std::isfinite(exportSnap.simTime))
			{
				throw std::runtime_error("Invalid distributed snapshot rank count or time.");
			}

			const auto ValidateArrays = [](
				const uint64_t N_particles,
				const auto& x,
				const auto& y,
				const auto& z,
				const auto& values)
			{
				Sim::CheckedLocalParticleCount(N_particles, "rank-local export count");
				if (x.size() != N_particles || y.size() != N_particles ||
					z.size() != N_particles || values.size() != N_particles)
				{
					throw std::runtime_error("Inconsistent local snapshot array sizes.");
				}
			};
			ValidateArrays(exportSnap.N_F, exportSnap.xPosF, exportSnap.yPosF, exportSnap.zPosF, exportSnap.velMagF);
			ValidateArrays(exportSnap.N_B, exportSnap.xPosB, exportSnap.yPosB, exportSnap.zPosB, exportSnap.volB);
			ValidateArrays(exportSnap.N_R, exportSnap.xPosR, exportSnap.yPosR, exportSnap.zPosR, exportSnap.volR);
			if (exportSnap.rankF.size() != exportSnap.N_F || exportSnap.bodyIdR.size() != exportSnap.N_R ||
				exportSnap.isMeshB.size() != exportSnap.N_B || (!includeB && exportSnap.N_B != 0))
			{
				throw std::runtime_error("Inconsistent local snapshot rank, body or boundary counts.");
			}

			rankCounts.assign(size_t{3} * mpiContext.Size(), 0);
			rankCounts[3 * mpiContext.Rank()] = exportSnap.N_F;
			rankCounts[3 * mpiContext.Rank() + 1] = exportSnap.N_B;
			rankCounts[3 * mpiContext.Rank() + 2] = exportSnap.N_R;
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to prepare distributed snapshot: {}",
				mpiContext.Rank(), mpiContext.Size(), ex.what());
			writeOk = false;
		}
		if (!mpiContext.AllTrue(writeOk, "Snapshot array validation")) { return false; }

		// all ranks must write the same frame and participate in the same file operations
		uint64_t step = exportSnap.simStep;
		uint64_t timeBits = std::bit_cast<uint64_t>(exportSnap.simTime);
		int boundary = includeB ? 1 : 0;
		mpiContext.BroadcastUint64(step, mpiContext.RootRank());
		mpiContext.BroadcastUint64(timeBits, mpiContext.RootRank());
		mpiContext.BroadcastInt(boundary, mpiContext.RootRank());
		if (!mpiContext.AllTrue(
			step == exportSnap.simStep && timeBits == std::bit_cast<uint64_t>(exportSnap.simTime) &&
			boundary == (includeB ? 1 : 0), "Snapshot frame agreement"))
		{
			return false;
		}
		mpiContext.AllReduceSumUint64s(rankCounts);

		Export::DistributedExportLayout layout{};
		Export::ExportSnapshotOffsets off{};
		Export::StaticBoundaryOffsets boundaryOff{};
		Export::BinHeaderSnap header{};
		Export::BinHeaderSnap boundaryHeader{};
		std::array<Export::ExportSnapshotFilePaths, 2> paths;
		std::array<std::string, 2> writePaths;
		std::vector<float> xyz;
		constexpr uint64_t N_xyzChunk = (64ull * 1024ull * 1024ull) / (3 * sizeof(float));
		bool writeBoundary = false;
		try
		{
			layout = Export::ComputeDistributedExportLayout(rankCounts, mpiContext.Rank());
			Export::ExportSnapshotBinInfo info{};
			info.step = exportSnap.simStep;
			info.time = exportSnap.simTime;
			info.N_F = info.N_rankF = layout.N_total[0];
			info.N_B = layout.N_total[1];
			info.N_R = layout.N_total[2];
			header = Export::MakeExportSnapshotHeader(info);
			off = Export::ComputeExportSnapshotOffsets(info);
			info.N_F = info.N_R = info.N_rankF = 0;
			boundaryHeader = Export::MakeExportSnapshotHeader(info);
			boundaryOff = Export::ComputeStaticBoundaryOffsets(info);
			const auto maxOffset = static_cast<uint64_t>(std::numeric_limits<MPI_Offset>::max());
			if (off.totalBytes > maxOffset || boundaryOff.totalBytes > maxOffset)
			{
				throw std::runtime_error("Shared snapshot exceeds MPI_Offset range.");
			}

			const uint64_t N_localMax = std::max({ exportSnap.N_F, exportSnap.N_B, exportSnap.N_R });
			xyz.resize(3 * std::min(N_localMax, N_xyzChunk));
			writeBoundary = includeB && layout.N_total[1] > 0;
			if (mpiContext.IsRoot())
			{
				paths[0] = serializationMgr.GetDistributedExportFilePaths(dir, exportSnap.simStep, false);
				writePaths[0] = paths[0].writePath.string();
				if (writeBoundary)
				{
					paths[1] = serializationMgr.GetDistributedExportFilePaths(dir, exportSnap.simStep, true);
					writePaths[1] = paths[1].writePath.string();
				}
			}
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to prepare shared snapshot files: {}",
				mpiContext.Rank(), mpiContext.Size(), ex.what());
			writeOk = false;
		}
		if (!mpiContext.AllTrue(writeOk, "Snapshot file preparation")) { return false; }
		mpiContext.BroadcastString(writePaths[0], mpiContext.RootRank());
		if (writeBoundary) { mpiContext.BroadcastString(writePaths[1], mpiContext.RootRank()); }

		// offset each field to this rank's particle range
		off.posF += layout.beginIdx[0] * 12;
		off.vF += layout.beginIdx[0] * 4;
		off.rankF += layout.beginIdx[0] * 2;
		off.posR += layout.beginIdx[2] * 12;
		off.volR += layout.beginIdx[2] * 4;
		off.bodyIdR += layout.beginIdx[2] * 4;
		boundaryOff.posB += layout.beginIdx[1] * 12;
		boundaryOff.volB += layout.beginIdx[1] * 4;
		boundaryOff.isMeshB += layout.beginIdx[1];
		const char dummy = 0;

		// empty ranks participate in every chunk write
		const auto WriteXyz = [&](
			MPI_File file,
			const auto& x,
			const auto& y,
			const auto& z,
			const uint64_t offset,
			const uint64_t N_max)
		{
			for (uint64_t beginIdx = 0; beginIdx < N_max; beginIdx += N_xyzChunk)
			{
				const uint64_t N_particles = beginIdx < x.size() ? std::min(N_xyzChunk, x.size() - beginIdx) : 0;
				for (uint64_t i = 0; i < N_particles; i++)
				{
					xyz[3 * i] = x[beginIdx + i];
					xyz[3 * i + 1] = y[beginIdx + i];
					xyz[3 * i + 2] = z[beginIdx + i];
				}
				mpiContext.WriteFileBytesAtAll(
					file,
					N_particles ? offset + beginIdx * 12 : 0,
					N_particles ? static_cast<const void*>(xyz.data()) : &dummy,
					static_cast<int>(N_particles * 12),
					"Snapshot XYZ write");
			}
		};
		const auto WriteValues = [&](
			MPI_File file,
			const auto& values,
			const uint64_t offset,
			const uint64_t N_max)
		{
			const uint64_t valueBytes = sizeof(values[0]);
			const uint64_t N_bytes = values.size() * valueBytes;
			for (uint64_t byteOffset = 0; byteOffset < N_max * valueBytes; byteOffset += MPI_CONTEXT_BYTE_CHUNK_SIZE)
			{
				const uint64_t N_chunkBytes = byteOffset < N_bytes
					? std::min(MPI_CONTEXT_BYTE_CHUNK_SIZE, N_bytes - byteOffset) : 0;
				mpiContext.WriteFileBytesAtAll(
					file,
					N_chunkBytes ? offset + byteOffset : 0,
					N_chunkBytes ? reinterpret_cast<const char*>(values.data()) + byteOffset : &dummy,
					static_cast<int>(N_chunkBytes),
					"Snapshot scalar write");
			}
		};
		const auto WriteFile = [&](const bool staticBoundary)
		{
			const size_t fileIdx = staticBoundary ? 1 : 0;
			const uint64_t N_bytes = staticBoundary ? boundaryOff.totalBytes : off.totalBytes;
			MPI_File file = mpiContext.OpenFileForWrite(writePaths[fileIdx], N_bytes);
			mpiContext.WriteFileBytesAtAll(
				file,
				0,
				staticBoundary ? &boundaryHeader : &header,
				mpiContext.IsRoot() ? static_cast<int>(sizeof(Export::BinHeaderSnap)) : 0,
				"Snapshot header write");
			if (staticBoundary)
			{
				WriteXyz(file, exportSnap.xPosB, exportSnap.yPosB, exportSnap.zPosB, boundaryOff.posB, layout.N_max[1]);
				WriteValues(file, exportSnap.volB, boundaryOff.volB, layout.N_max[1]);
				WriteValues(file, exportSnap.isMeshB, boundaryOff.isMeshB, layout.N_max[1]);
			}
			else
			{
				WriteXyz(file, exportSnap.xPosF, exportSnap.yPosF, exportSnap.zPosF, off.posF, layout.N_max[0]);
				WriteValues(file, exportSnap.velMagF, off.vF, layout.N_max[0]);
				WriteValues(file, exportSnap.rankF, off.rankF, layout.N_max[0]);
				WriteXyz(file, exportSnap.xPosR, exportSnap.yPosR, exportSnap.zPosR, off.posR, layout.N_max[2]);
				WriteValues(file, exportSnap.volR, off.volR, layout.N_max[2]);
				WriteValues(file, exportSnap.bodyIdR, off.bodyIdR, layout.N_max[2]);
			}
			mpiContext.CloseFile(file);
			// every rank must close the file before root checks its size and publishes it
			mpiContext.Barrier();
			if (mpiContext.IsRoot())
			{
				try
				{
					serializationMgr.PublishDistributedExportFile(paths[fileIdx], N_bytes);
				}
				catch (const std::exception& ex)
				{
					SPDLOG_ERROR("Failed to publish shared snapshot: {}", ex.what());
					writeOk = false;
				}
			}
			return mpiContext.AllTrue(writeOk, "Snapshot file publication");
		};
		if (writeBoundary && !WriteFile(true)) { return false; }
		return WriteFile(false);
	}
	catch (...)
	{
		mpiContext.Abort("Unexpected exception during collective snapshot output");
	}
}

bool Dist::PackOwnedExportSnapshot(
	Sync::ExportSnapshot& exportSnap,
	const Sync::FluidConstView& particlesF,
	const Sync::BoundaryConstView& particlesB,
	const Sync::RigidConstView& particlesR,
	const Sync::RigidBodies& rigidBodies,
	const DistParticleState& particleStateF,
	const DistParticleState& particleStateB,
	const DistParticleState& particleStateR,
	std::vector<uint32_t>& exportFluidParticleLocalIdxs,
	const int rank,
	const int N_ranks,
	const bool includeB,
	const std::optional<uint64_t> meshIdBeginB,
	const uint64_t simStep,
	const double simTime)
{
	if (rank < 0 || rank >= N_ranks || rank > std::numeric_limits<uint16_t>::max()) { return false; }

	// compact the sorted selection to owned fluid particles
	size_t N_selected = 0;
	uint32_t prevLocalIdx = 0;
	bool first = true;

	for (const uint32_t localIdx : exportFluidParticleLocalIdxs)
	{
		if ((!first && localIdx <= prevLocalIdx) ||
			localIdx >= particlesF.N_ ||
			localIdx >= particlesF.id.size()) { return false; }
		first = false;
		prevLocalIdx = localIdx;
		if (localIdx >= particleStateF.N_owned) { continue; }
		if (particlesF.id[localIdx] == Sim::INVALID_PARTICLE_ID) { return false; }
		exportFluidParticleLocalIdxs[N_selected++] = localIdx;
	}

	exportFluidParticleLocalIdxs.resize(N_selected);
	exportSnap.simStep = simStep;
	exportSnap.simTime = simTime;
	exportSnap.N_F = Sim::CheckedLocalParticleCount(N_selected, "rank-local fluid export");
	exportSnap.xPosF.resize(exportSnap.N_F);
	exportSnap.yPosF.resize(exportSnap.N_F);
	exportSnap.zPosF.resize(exportSnap.N_F);
	exportSnap.velMagF.resize(exportSnap.N_F);
	exportSnap.rankF.resize(exportSnap.N_F);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, exportSnap.N_F),
		[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t i = range.begin(); i != range.end(); i++)
			{
				const uint32_t localIdx = exportFluidParticleLocalIdxs[i];
				exportSnap.xPosF[i] = particlesF.xPos[localIdx];
				exportSnap.yPosF[i] = particlesF.yPos[localIdx];
				exportSnap.zPosF[i] = particlesF.zPos[localIdx];

				const float xVel = particlesF.xVel[localIdx];
				const float yVel = particlesF.yVel[localIdx];
				const float zVel = particlesF.zVel[localIdx];

				exportSnap.velMagF[i] = std::sqrt(xVel * xVel + yVel * yVel + zVel * zVel);
				exportSnap.rankF[i] = static_cast<uint16_t>(rank);
			}
		});

	exportSnap.N_B = includeB
		? Sim::CheckedLocalParticleCount(particleStateB.N_owned, "rank-local boundary export")
		: 0;
	exportSnap.xPosB.resize(exportSnap.N_B);
	exportSnap.yPosB.resize(exportSnap.N_B);
	exportSnap.zPosB.resize(exportSnap.N_B);
	exportSnap.volB.resize(exportSnap.N_B);
	exportSnap.isMeshB.resize(exportSnap.N_B);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, exportSnap.N_B),
		[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t i = range.begin(); i != range.end(); i++)
			{
				// owned particles occupy the compact local prefix
				const uint32_t localIdx = i;
				if (localIdx >= particlesB.N_ ||
					localIdx >= particlesB.id.size() ||
					particlesB.id[localIdx] == Sim::INVALID_PARTICLE_ID)
				{
					throw std::runtime_error("Invalid owned boundary particle during snapshot packing.");
				}
				exportSnap.xPosB[i] = particlesB.xPos[localIdx];
				exportSnap.yPosB[i] = particlesB.yPos[localIdx];
				exportSnap.zPosB[i] = particlesB.zPos[localIdx];
				exportSnap.volB[i] = particlesB.volB[localIdx];
				exportSnap.isMeshB[i] = meshIdBeginB && particlesB.id[localIdx] >= *meshIdBeginB;
			}
		});

	// output the stored owners even if body integration crossed a split this step
	const Sync::RigidConstView& R = particlesR;
	const Sync::RigidBodies& RB = rigidBodies;
	exportSnap.N_R = Sim::CheckedLocalParticleCount(
		particleStateR.N_owned, "rank-local rigid export");
	exportSnap.bodyIdR.resize(exportSnap.N_R);
	exportSnap.xPosR.resize(exportSnap.N_R);
	exportSnap.yPosR.resize(exportSnap.N_R);
	exportSnap.zPosR.resize(exportSnap.N_R);
	exportSnap.volR.resize(exportSnap.N_R);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, exportSnap.N_R),
		[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t i = range.begin(); i != range.end(); i++)
			{
				// owned particles occupy the compact local prefix
				const uint32_t localIdx = i;
				if (localIdx >= R.N_ ||
					localIdx >= R.id.size() ||
					R.id[localIdx] == Sim::INVALID_PARTICLE_ID)
				{
					throw std::runtime_error("Invalid owned rigid particle during snapshot packing.");
				}
				const uint32_t bodyIdx = R.bodyId[localIdx];
				if (bodyIdx >= RB.N_) { throw std::runtime_error("Invalid rigid export body index."); }

				const float lx = R.xRelPos[localIdx];
				const float ly = R.yRelPos[localIdx];
				const float lz = R.zRelPos[localIdx];

				exportSnap.bodyIdR[i] = bodyIdx;
				exportSnap.volR[i] = R.volR[localIdx];
				exportSnap.xPosR[i] = RB.xCM[bodyIdx] + std::fmaf(
					RB.a00[bodyIdx], lx, std::fmaf(RB.a01[bodyIdx], ly, RB.a02[bodyIdx] * lz));
				exportSnap.yPosR[i] = RB.yCM[bodyIdx] + std::fmaf(
					RB.a10[bodyIdx], lx, std::fmaf(RB.a11[bodyIdx], ly, RB.a12[bodyIdx] * lz));
				exportSnap.zPosR[i] = RB.zCM[bodyIdx] + std::fmaf(
					RB.a20[bodyIdx], lx, std::fmaf(RB.a21[bodyIdx], ly, RB.a22[bodyIdx] * lz));
			}
		});

	return true;
}
