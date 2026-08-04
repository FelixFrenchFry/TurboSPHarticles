// dist_snapshots.cpp

// external
#include <algorithm>
#include <cmath>
#include <limits>
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
	int CheckedByteCount(const uint64_t N_packets)
	{
		const uint64_t N_max =
			static_cast<uint64_t>(std::numeric_limits<int>::max()) / sizeof(T);

		if (N_packets > N_max)
		{
			throw std::runtime_error("Distributed snapshot payload exceeds MPI byte count range.");
		}

		return static_cast<int>(N_packets * sizeof(T));
	}

	template <typename T>
	int CheckedByteCount(const std::vector<T>& packets)
	{
		return CheckedByteCount<T>(packets.size());
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
		const Sync::FluidParticles& fluidParticles,
		const DistParticleState& particleState,
		const uint32_t localIdx,
		const int rank)
	{
		const float xVel = fluidParticles.xVel[localIdx];
		const float yVel = fluidParticles.yVel[localIdx];
		const float zVel = fluidParticles.zVel[localIdx];

		DistExportFluidParticlePacket packet{};
		packet.globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
		packet.id = fluidParticles.id[localIdx];
		packet.rank = static_cast<uint32_t>(rank);

		packet.xPos = fluidParticles.xPos[localIdx];
		packet.yPos = fluidParticles.yPos[localIdx];
		packet.zPos = fluidParticles.zPos[localIdx];
		packet.velMag = std::sqrt(xVel * xVel + yVel * yVel + zVel * zVel);

		return packet;
	}

	DistExportBoundaryParticlePacket PackExportBoundaryParticle(
		const Sync::BoundaryParticles& boundaryParticles,
		const DistParticleState& particleState,
		const uint32_t localIdx)
	{
		DistExportBoundaryParticlePacket packet{};
		packet.globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
		packet.id = boundaryParticles.id[localIdx];

		packet.xPos = boundaryParticles.xPos[localIdx];
		packet.yPos = boundaryParticles.yPos[localIdx];
		packet.zPos = boundaryParticles.zPos[localIdx];
		packet.volB = boundaryParticles.volB[localIdx];

		return packet;
	}

} // namespace anonymous

void DistExportSnapshotBuffer::Clear()
{
	fluidParticles.clear();
	boundaryParticles.clear();
}

bool Dist::PackOwnedExportSnapshotBuffer(
	DistExportSnapshotBuffer& gatherBuffer,
	const Sync::UniversalSimState& localState,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	const std::vector<uint32_t>& exportFluidParticleLocalIdxs,
	const int rank)
{
	if (rank < 0) { return false; }

	gatherBuffer.Clear();
	gatherBuffer.fluidParticles.reserve(exportFluidParticleLocalIdxs.size());
	gatherBuffer.boundaryParticles.reserve(boundaryParticleState.particleLocalIdx_Owned.size());

	// fluid particles
	std::vector<uint8_t> localIdxUsed(localState.fluidParticles.N_, 0);
	for (const uint32_t localIdx : exportFluidParticleLocalIdxs)
	{
		if (localIdx >= localState.fluidParticles.N_) { return false; }
		if (!fluidParticleState.HasMetadataAtLocalIdx(localIdx)) { return false; }
		if (localIdxUsed[localIdx] != 0) { return false; }
		localIdxUsed[localIdx] = 1;

		const DistParticleClass particleClass = fluidParticleState.particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer) { continue; }

		if (fluidParticleState.particleGlobalIdx_FromLocalIdx[localIdx] ==
			NO_GLOBAL_PARTICLE_IDX)
		{
			return false;
		}

		gatherBuffer.fluidParticles.push_back(
			PackExportFluidParticle(localState.fluidParticles, fluidParticleState, localIdx, rank));
	}

	// boundary particles
	for (const uint32_t localIdx : boundaryParticleState.particleLocalIdx_Owned)
	{
		if (localIdx >= localState.boundaryParticles.N_) { return false; }
		if (!boundaryParticleState.HasMetadataAtLocalIdx(localIdx)) { return false; }

		const DistParticleClass particleClass = boundaryParticleState.particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer) { return false; }

		if (boundaryParticleState.particleGlobalIdx_FromLocalIdx[localIdx] ==
			NO_GLOBAL_PARTICLE_IDX)
		{
			return false;
		}

		gatherBuffer.boundaryParticles.push_back(
			PackExportBoundaryParticle(localState.boundaryParticles, boundaryParticleState, localIdx));
	}

	return true;
}

bool Dist::RebuildExportSnapshotFromGatheredParticles(
	Sync::ExportSnapshot& exportSnap,
	DistExportSnapshotBuffer& gatherBuffer,
	const uint64_t simStep,
	const double simTime)
{
	if (gatherBuffer.fluidParticles.size() >
		static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
	{
		return false;
	}
	if (gatherBuffer.boundaryParticles.size() >
		static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
	{
		return false;
	}

	// gathered packets are ordered by global index in place
	std::vector<DistExportFluidParticlePacket>& sortedFluidParticles = gatherBuffer.fluidParticles;
	std::vector<DistExportBoundaryParticlePacket>& sortedBoundaryParticles = gatherBuffer.boundaryParticles;
	tbb::parallel_sort(
		sortedFluidParticles.begin(),
		sortedFluidParticles.end(),
		[](const DistExportFluidParticlePacket& a, const DistExportFluidParticlePacket& b)
		{
			return a.globalIdx < b.globalIdx;
		});
	tbb::parallel_sort(
		sortedBoundaryParticles.begin(),
		sortedBoundaryParticles.end(),
		[](const DistExportBoundaryParticlePacket& a, const DistExportBoundaryParticlePacket& b)
		{
			return a.globalIdx < b.globalIdx;
		});

	exportSnap.simStep = simStep;
	exportSnap.simTime = simTime;
	exportSnap.N_F = static_cast<uint32_t>(sortedFluidParticles.size());
	exportSnap.N_B = static_cast<uint32_t>(sortedBoundaryParticles.size());
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
		const DistExportFluidParticlePacket& packet = sortedFluidParticles[i];
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
		const DistExportBoundaryParticlePacket& packet = sortedBoundaryParticles[i];
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
	const Sync::UniversalSimState& localState)
{
	const auto& R = localState.rigidParticles;
	const auto& RB = localState.rigidBodies;

	exportSnap.N_R = R.N_;
	exportSnap.idR = R.id;
	exportSnap.bodyIdR = R.bodyId;
	exportSnap.xPosR.resize(R.N_);
	exportSnap.yPosR.resize(R.N_);
	exportSnap.zPosR.resize(R.N_);
	exportSnap.volR = R.volR;

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
	const Sync::UniversalSimState& localState,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	const std::vector<uint32_t>& exportFluidParticleLocalIdxs,
	const MpiContext& mpiContext,
	const uint64_t simStep,
	const double simTime,
	DistSnapshotGatherStats* stats)
{
	if (stats) { *stats = {}; }

	// locally packed packets occupy the front of the gather storage on every rank
	bool localOk = PackOwnedExportSnapshotBuffer(
		gatherBuffer,
		localState,
		fluidParticleState,
		boundaryParticleState,
		exportFluidParticleLocalIdxs,
		mpiContext.Rank());

	int N_sendF = 0;
	int N_sendB = 0;
	if (localOk)
	{
		N_sendF = CheckedPacketCount(gatherBuffer.fluidParticles);
		N_sendB = CheckedPacketCount(gatherBuffer.boundaryParticles);
	}

	if (!mpiContext.AllTrue(localOk)) { return false; }

	std::vector<RankDistExportSnapshotGather> rankGathers{};
	rankGathers.reserve(static_cast<uint64_t>(std::max(0, mpiContext.Size() - 1)));

	if (mpiContext.IsRoot())
	{
		for (int rank = 0; rank < mpiContext.Size(); rank++)
		{
			if (rank == mpiContext.RootRank()) { continue; }

			RankDistExportSnapshotGather rankGather{};
			rankGather.rank = rank;
			rankGathers.push_back(std::move(rankGather));
		}
	}

	std::vector<MPI_Request> requests{};
	requests.reserve(rankGathers.size() * 2);

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
	mpiContext.WaitAll(requests);
	requests.clear();

	if (mpiContext.IsRoot())
	{
		// reserve one packet slot range per rank behind the locally packed packets
		uint64_t N_gatheredF = gatherBuffer.fluidParticles.size();
		uint64_t N_gatheredB = gatherBuffer.boundaryParticles.size();

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
		}

		if (localOk)
		{
			gatherBuffer.fluidParticles.resize(N_gatheredF);
			gatherBuffer.boundaryParticles.resize(N_gatheredB);
		}
	}
	if (!mpiContext.AllTrue(localOk)) { return false; }

	requests.reserve(rankGathers.size() * 2);

	// gather payloads after the root gather storage holds a slot range for every rank
	if (mpiContext.IsRoot())
	{
		for (const RankDistExportSnapshotGather& rankGather : rankGathers)
		{
			const uint64_t N_recvF = static_cast<uint64_t>(rankGather.N_recvF);
			const uint64_t N_recvB = static_cast<uint64_t>(rankGather.N_recvB);

			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvBytes(
				DataAtOrNull(
					gatherBuffer.fluidParticles, rankGather.recvFluidBeginIdx, N_recvF),
				CheckedByteCount<DistExportFluidParticlePacket>(N_recvF),
				rankGather.rank,
				DIST_SNAPSHOT_TAG_EXPORT_FLUID_PAYLOAD,
				requests.back());
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvBytes(
				DataAtOrNull(
					gatherBuffer.boundaryParticles, rankGather.recvBoundaryBeginIdx, N_recvB),
				CheckedByteCount<DistExportBoundaryParticlePacket>(N_recvB),
				rankGather.rank,
				DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_PAYLOAD,
				requests.back());
		}
	}
	else
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendBytes(
			DataOrNull(gatherBuffer.fluidParticles),
			CheckedByteCount(gatherBuffer.fluidParticles),
			mpiContext.RootRank(),
			DIST_SNAPSHOT_TAG_EXPORT_FLUID_PAYLOAD,
			requests.back());
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendBytes(
			DataOrNull(gatherBuffer.boundaryParticles),
			CheckedByteCount(gatherBuffer.boundaryParticles),
			mpiContext.RootRank(),
			DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_PAYLOAD,
			requests.back());
	}
	mpiContext.WaitAll(requests);

	if (mpiContext.IsRoot())
	{
		localOk = RebuildExportSnapshotFromGatheredParticles(
			exportSnap,
			gatherBuffer,
			simStep,
			simTime);
		if (localOk)
		{
			PackReplicatedRigidExportSnapshot(exportSnap, localState);
		}

		if (localOk && stats)
		{
			stats->N_gatheredF =
				static_cast<uint32_t>(gatherBuffer.fluidParticles.size());
			stats->N_gatheredB =
				static_cast<uint32_t>(gatherBuffer.boundaryParticles.size());
			stats->N_gatheredR = localState.rigidParticles.N_;
		}
	}

	return mpiContext.AllTrue(localOk);
}
