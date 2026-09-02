// dist_snapshots.cpp

// external
#include <algorithm>
#include <cmath>
#include <limits>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <tbb/parallel_sort.h>
#include <utility>

// internal
#include "dist_snapshots.h"
#include "mpi_context.h"



using namespace Dist;

namespace
{
	struct RankDistExportSnapshotGather
	{
		int rank = -1;

		int N_recvF = 0;
		int N_recvB = 0;

		// packet slots reserved for this rank behind the locally packed packets
		uint64_t recvFluidBeginIdx = 0;
		uint64_t recvBoundaryBeginIdx = 0;
	};

	template <typename T>
	int CheckedPacketCount(const std::vector<T>& packets)
	{
		if (packets.size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
		{
			throw std::runtime_error("Distributed snapshot packet count exceeds MPI int count range.");
		}

		return static_cast<int>(packets.size());
	}

	template <typename T>
	uint64_t CheckedByteSize(const uint64_t N_packets)
	{
		const uint64_t N_max = std::numeric_limits<uint64_t>::max() / sizeof(T);

		if (N_packets > N_max)
		{
			throw std::runtime_error(
				"Distributed snapshot payload byte size exceeds uint64_t range.");
		}

		return N_packets * sizeof(T);
	}

	template <typename T>
	uint64_t CheckedByteSize(const std::vector<T>& packets)
	{
		return CheckedByteSize<T>(packets.size());
	}

	uint64_t ByteChunkRequestCount(const uint64_t N_bytes)
	{
		return N_bytes > 0
			? 1 + (N_bytes - 1) / MPI_CONTEXT_BYTE_CHUNK_SIZE
			: 0;
	}

	template <typename T>
	void* DataOrNull(std::vector<T>& packets)
	{
		return packets.empty() ? nullptr : packets.data();
	}

	// receive destination inside the shared gather storage,
	// only addressed when that rank actually sends packets
	template <typename T>
	void* DataAtOrNull(
		std::vector<T>& packets,
		const uint64_t beginIdx,
		const uint64_t N_packets)
	{
		return (N_packets == 0) ? nullptr : packets.data() + beginIdx;
	}

	DistExportFluidParticlePacket PackExportFluidParticle(
		const Sync::FluidConstView& particlesF,
		const uint32_t localIdx,
		const int rank)
	{
		const float xVel = particlesF.xVel[localIdx];
		const float yVel = particlesF.yVel[localIdx];
		const float zVel = particlesF.zVel[localIdx];

		DistExportFluidParticlePacket packet{};
		packet.globalIdx = particlesF.id[localIdx];
		packet.id = particlesF.id[localIdx];
		packet.rank = static_cast<uint32_t>(rank);

		packet.xPos = particlesF.xPos[localIdx];
		packet.yPos = particlesF.yPos[localIdx];
		packet.zPos = particlesF.zPos[localIdx];
		packet.velMag = std::sqrt(xVel * xVel + yVel * yVel + zVel * zVel);

		return packet;
	}

	DistExportBoundaryParticlePacket PackExportBoundaryParticle(
		const Sync::BoundaryConstView& particlesB,
		const uint32_t localIdx)
	{
		DistExportBoundaryParticlePacket packet{};
		packet.globalIdx = particlesB.id[localIdx];
		packet.id = particlesB.id[localIdx];

		packet.xPos = particlesB.xPos[localIdx];
		packet.yPos = particlesB.yPos[localIdx];
		packet.zPos = particlesB.zPos[localIdx];
		packet.volB = particlesB.volB[localIdx];

		return packet;
	}

} // namespace anonymous

void DistExportSnapshotBuffer::Clear()
{
	particlesF.clear();
	particlesB.clear();
}

bool Dist::PackOwnedExportSnapshotBuffer(
	DistExportSnapshotBuffer& gatherBuffer,
	const Sync::FluidConstView& particlesF,
	const Sync::BoundaryConstView& particlesB,
	const DistParticleState& particleStateF,
	const DistParticleState& particleStateB,
	const std::vector<uint32_t>& exportFluidParticleLocalIdxs,
	const int rank)
{
	if (rank < 0) { return false; }

	gatherBuffer.Clear();
	gatherBuffer.particlesF.reserve(exportFluidParticleLocalIdxs.size());
	gatherBuffer.particlesB.reserve(particleStateB.particleLocalIdx_Owned.size());

	// fluid particles
	std::vector<uint8_t> localIdxUsed(particlesF.N_, 0);
	for (const uint32_t localIdx : exportFluidParticleLocalIdxs)
	{
		if (localIdx >= particlesF.N_ ||
			localIdx >= particlesF.id.size()) { return false; }
		if (!particleStateF.HasMetadataAtLocalIdx(localIdx)) { return false; }
		if (localIdxUsed[localIdx] != 0) { return false; }
		localIdxUsed[localIdx] = 1;

		const DistParticleClass particleClass = particleStateF.particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer) { continue; }

		if (particlesF.id[localIdx] == NO_GLOBAL_PARTICLE_IDX)
		{
			return false;
		}

		gatherBuffer.particlesF.push_back(
			PackExportFluidParticle(particlesF, localIdx, rank));
	}

	// boundary particles
	for (const uint32_t localIdx : particleStateB.particleLocalIdx_Owned)
	{
		if (localIdx >= particlesB.N_ ||
			localIdx >= particlesB.id.size()) { return false; }
		if (!particleStateB.HasMetadataAtLocalIdx(localIdx)) { return false; }

		const DistParticleClass particleClass = particleStateB.particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer) { return false; }

		if (particlesB.id[localIdx] == NO_GLOBAL_PARTICLE_IDX)
		{
			return false;
		}

		gatherBuffer.particlesB.push_back(
			PackExportBoundaryParticle(particlesB, localIdx));
	}

	return true;
}

bool Dist::RebuildExportSnapshotFromGatheredParticles(
	Sync::ExportSnapshot& exportSnap,
	DistExportSnapshotBuffer& gatherBuffer,
	const uint64_t simStep,
	const double simTime)
{
	if (gatherBuffer.particlesF.size() >
		static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
	{
		return false;
	}
	if (gatherBuffer.particlesB.size() >
		static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
	{
		return false;
	}

	// gathered packets are ordered by global index in place
	std::vector<DistExportFluidParticlePacket>& sortedParticlesF = gatherBuffer.particlesF;
	std::vector<DistExportBoundaryParticlePacket>& sortedParticlesB = gatherBuffer.particlesB;
	tbb::parallel_sort(
		sortedParticlesF.begin(),
		sortedParticlesF.end(),
		[](const DistExportFluidParticlePacket& a, const DistExportFluidParticlePacket& b)
		{
			return a.globalIdx < b.globalIdx;
		});
	tbb::parallel_sort(
		sortedParticlesB.begin(),
		sortedParticlesB.end(),
		[](const DistExportBoundaryParticlePacket& a, const DistExportBoundaryParticlePacket& b)
		{
			return a.globalIdx < b.globalIdx;
		});

	exportSnap.simStep = simStep;
	exportSnap.simTime = simTime;
	exportSnap.N_F = static_cast<uint32_t>(sortedParticlesF.size());
	exportSnap.N_B = static_cast<uint32_t>(sortedParticlesB.size());
	exportSnap.N_R = 0;

	exportSnap.idF.resize(exportSnap.N_F);
	exportSnap.xPosF.resize(exportSnap.N_F);
	exportSnap.yPosF.resize(exportSnap.N_F);
	exportSnap.zPosF.resize(exportSnap.N_F);
	exportSnap.velMagF.resize(exportSnap.N_F);
	exportSnap.rankF.resize(exportSnap.N_F);
	exportSnap.idB.resize(exportSnap.N_B);
	exportSnap.xPosB.resize(exportSnap.N_B);
	exportSnap.yPosB.resize(exportSnap.N_B);
	exportSnap.zPosB.resize(exportSnap.N_B);
	exportSnap.volB.resize(exportSnap.N_B);
	exportSnap.idR.clear();
	exportSnap.bodyIdR.clear();
	exportSnap.xPosR.clear();
	exportSnap.yPosR.clear();
	exportSnap.zPosR.clear();
	exportSnap.volR.clear();

	uint32_t prevGlobalIdx = NO_GLOBAL_PARTICLE_IDX;
	for (uint32_t i = 0; i < exportSnap.N_F; i++)
	{
		const DistExportFluidParticlePacket& packet = sortedParticlesF[i];
		if (packet.globalIdx == NO_GLOBAL_PARTICLE_IDX) { return false; }
		if (packet.globalIdx == prevGlobalIdx) { return false; }

		exportSnap.idF[i] = packet.id;
		exportSnap.xPosF[i] = packet.xPos;
		exportSnap.yPosF[i] = packet.yPos;
		exportSnap.zPosF[i] = packet.zPos;
		exportSnap.velMagF[i] = packet.velMag;
		exportSnap.rankF[i] = packet.rank;

		prevGlobalIdx = packet.globalIdx;
	}

	prevGlobalIdx = NO_GLOBAL_PARTICLE_IDX;
	for (uint32_t i = 0; i < exportSnap.N_B; i++)
	{
		const DistExportBoundaryParticlePacket& packet = sortedParticlesB[i];
		if (packet.globalIdx == NO_GLOBAL_PARTICLE_IDX) { return false; }
		if (packet.globalIdx == prevGlobalIdx) { return false; }

		exportSnap.idB[i] = packet.id;
		exportSnap.xPosB[i] = packet.xPos;
		exportSnap.yPosB[i] = packet.yPos;
		exportSnap.zPosB[i] = packet.zPos;
		exportSnap.volB[i] = packet.volB;

		prevGlobalIdx = packet.globalIdx;
	}

	return true;
}

void Dist::PackReplicatedRigidExportSnapshot(
	Sync::ExportSnapshot& exportSnap,
	const Sync::RigidConstView& particlesR,
	const Sync::RigidBodies& rigidBodies)
{
	const Sync::RigidConstView& R = particlesR;
	const Sync::RigidBodies& RB = rigidBodies;

	exportSnap.N_R = R.N_;
	exportSnap.idR.assign(R.id.begin(), R.id.end());
	exportSnap.bodyIdR.assign(R.bodyId.begin(), R.bodyId.end());
	exportSnap.xPosR.resize(R.N_);
	exportSnap.yPosR.resize(R.N_);
	exportSnap.zPosR.resize(R.N_);
	exportSnap.volR.assign(R.volR.begin(), R.volR.end());

	for (uint32_t rigidParticleIdx = 0; rigidParticleIdx < R.N_; rigidParticleIdx++)
	{
		const uint32_t bodyIdx = R.bodyId[rigidParticleIdx];

		const float lx = R.xRelPos[rigidParticleIdx];
		const float ly = R.yRelPos[rigidParticleIdx];
		const float lz = R.zRelPos[rigidParticleIdx];

		exportSnap.xPosR[rigidParticleIdx] =
			RB.xCM[bodyIdx] + std::fmaf(
				RB.a00[bodyIdx], lx,
				std::fmaf(RB.a01[bodyIdx], ly, RB.a02[bodyIdx] * lz));
		exportSnap.yPosR[rigidParticleIdx] =
			RB.yCM[bodyIdx] + std::fmaf(
				RB.a10[bodyIdx], lx,
				std::fmaf(RB.a11[bodyIdx], ly, RB.a12[bodyIdx] * lz));
		exportSnap.zPosR[rigidParticleIdx] =
			RB.zCM[bodyIdx] + std::fmaf(
				RB.a20[bodyIdx], lx,
				std::fmaf(RB.a21[bodyIdx], ly, RB.a22[bodyIdx] * lz));
	}
}

bool Dist::GatherExportSnapshotToRoot(
	Sync::ExportSnapshot& exportSnap,
	DistExportSnapshotBuffer& gatherBuffer,
	const Sync::FluidConstView& particlesF,
	const Sync::BoundaryConstView& particlesB,
	const Sync::RigidConstView& particlesR,
	const Sync::RigidBodies& rigidBodies,
	const DistParticleState& particleStateF,
	const DistParticleState& particleStateB,
	const std::vector<uint32_t>& exportFluidParticleLocalIdxs,
	const MpiContext& mpiContext,
	const uint64_t simStep,
	const double simTime,
	DistSnapshotGatherStats* stats)
{
	if (stats) { *stats = {}; }

	bool localOk = true;
	int N_sendF = 0;
	int N_sendB = 0;
	std::vector<RankDistExportSnapshotGather> rankGathers{};
	std::vector<MPI_Request> requests{};
	try
	{
		// locally packed packets occupy the front of the gather storage on every rank
		localOk = PackOwnedExportSnapshotBuffer(
			gatherBuffer,
			particlesF,
			particlesB,
			particleStateF,
			particleStateB,
			exportFluidParticleLocalIdxs,
			mpiContext.Rank());

		if (localOk)
		{
			N_sendF = CheckedPacketCount(gatherBuffer.particlesF);
			N_sendB = CheckedPacketCount(gatherBuffer.particlesB);
		}

		if (mpiContext.IsRoot())
		{
			rankGathers.reserve(
				static_cast<uint64_t>(std::max(0, mpiContext.Size() - 1)));
			for (int rank = 0; rank < mpiContext.Size(); rank++)
			{
				if (rank == mpiContext.RootRank()) { continue; }

				RankDistExportSnapshotGather rankGather{};
				rankGather.rank = rank;
				rankGathers.push_back(std::move(rankGather));
			}
		}

		const uint64_t N_countRequests = mpiContext.IsRoot()
			? static_cast<uint64_t>(rankGathers.size()) * 2
			: 2;
		requests.reserve(N_countRequests);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to prepare distributed snapshot gather storage: {}",
			mpiContext.Rank(), mpiContext.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext.AllTrue(localOk)) { return false; }

	// gather packet counts before gathering raw particle payloads
	if (mpiContext.IsRoot())
	{
		for (RankDistExportSnapshotGather& rankGather : rankGathers)
		{
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvInt(
				rankGather.N_recvF,
				rankGather.rank,
				DIST_SNAPSHOT_TAG_EXPORT_FLUID_COUNT,
				requests.back());
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvInt(
				rankGather.N_recvB,
				rankGather.rank,
				DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_COUNT,
				requests.back());
		}
	}
	else
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			N_sendF,
			mpiContext.RootRank(),
			DIST_SNAPSHOT_TAG_EXPORT_FLUID_COUNT,
			requests.back());
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			N_sendB,
			mpiContext.RootRank(),
			DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_COUNT,
			requests.back());
	}
	mpiContext.WaitAll(requests, "distributed snapshot counts");
	requests.clear();

	try
	{
		uint64_t N_payloadRequests = 0;
		if (mpiContext.IsRoot())
		{
			// reserve one packet slot range per rank behind the locally packed packets
			uint64_t N_gatheredF = gatherBuffer.particlesF.size();
			uint64_t N_gatheredB = gatherBuffer.particlesB.size();

			for (RankDistExportSnapshotGather& rankGather : rankGathers)
			{
				if (rankGather.N_recvF < 0 || rankGather.N_recvB < 0)
				{
					localOk = false;
					break;
				}

				rankGather.recvFluidBeginIdx = N_gatheredF;
				rankGather.recvBoundaryBeginIdx = N_gatheredB;
				N_gatheredF += static_cast<uint64_t>(rankGather.N_recvF);
				N_gatheredB += static_cast<uint64_t>(rankGather.N_recvB);
				N_payloadRequests += ByteChunkRequestCount(
					CheckedByteSize<DistExportFluidParticlePacket>(
						static_cast<uint64_t>(rankGather.N_recvF)));
				N_payloadRequests += ByteChunkRequestCount(
					CheckedByteSize<DistExportBoundaryParticlePacket>(
						static_cast<uint64_t>(rankGather.N_recvB)));
			}

			if (localOk)
			{
				gatherBuffer.particlesF.resize(N_gatheredF);
				gatherBuffer.particlesB.resize(N_gatheredB);
			}
		}
		else
		{
			N_payloadRequests += ByteChunkRequestCount(
				CheckedByteSize(gatherBuffer.particlesF));
			N_payloadRequests += ByteChunkRequestCount(
				CheckedByteSize(gatherBuffer.particlesB));
		}
		requests.reserve(N_payloadRequests);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to prepare distributed snapshot gather storage: {}",
			mpiContext.Rank(), mpiContext.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext.AllTrue(localOk)) { return false; }

	// gather payloads after the root gather storage holds a slot range for every rank
	if (mpiContext.IsRoot())
	{
		for (const RankDistExportSnapshotGather& rankGather : rankGathers)
		{
			const uint64_t N_recvF = static_cast<uint64_t>(rankGather.N_recvF);
			const uint64_t N_recvB = static_cast<uint64_t>(rankGather.N_recvB);

			mpiContext.IRecvBytesChunked(
				DataAtOrNull(
					gatherBuffer.particlesF, rankGather.recvFluidBeginIdx, N_recvF),
				CheckedByteSize<DistExportFluidParticlePacket>(N_recvF),
				rankGather.rank,
				DIST_SNAPSHOT_TAG_EXPORT_FLUID_PAYLOAD,
				requests);
			mpiContext.IRecvBytesChunked(
				DataAtOrNull(
					gatherBuffer.particlesB, rankGather.recvBoundaryBeginIdx, N_recvB),
				CheckedByteSize<DistExportBoundaryParticlePacket>(N_recvB),
				rankGather.rank,
				DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_PAYLOAD,
				requests);
		}
	}
	else
	{
		mpiContext.ISendBytesChunked(
			DataOrNull(gatherBuffer.particlesF),
			CheckedByteSize(gatherBuffer.particlesF),
			mpiContext.RootRank(),
			DIST_SNAPSHOT_TAG_EXPORT_FLUID_PAYLOAD,
			requests);
		mpiContext.ISendBytesChunked(
			DataOrNull(gatherBuffer.particlesB),
			CheckedByteSize(gatherBuffer.particlesB),
			mpiContext.RootRank(),
			DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_PAYLOAD,
			requests);
	}
	mpiContext.WaitAll(requests, "distributed snapshot payloads");

	if (mpiContext.IsRoot())
	{
		try
		{
			localOk = RebuildExportSnapshotFromGatheredParticles(
				exportSnap,
				gatherBuffer,
				simStep,
				simTime);
			if (localOk)
			{
				PackReplicatedRigidExportSnapshot(
					exportSnap, particlesR, rigidBodies);
			}

			if (localOk && stats)
			{
				stats->N_gatheredF =
					static_cast<uint32_t>(gatherBuffer.particlesF.size());
				stats->N_gatheredB =
					static_cast<uint32_t>(gatherBuffer.particlesB.size());
				stats->N_gatheredR = particlesR.N_;
			}
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to rebuild distributed export snapshot: {}",
				mpiContext.Rank(), mpiContext.Size(), ex.what());
			localOk = false;
		}
	}

	return mpiContext.AllTrue(localOk);
}
