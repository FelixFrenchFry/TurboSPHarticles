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

bool Dist::ValidateDistExportSnapshotRootCapacity(
	const uint64_t N_gatheredF,
	const uint64_t N_gatheredB,
	const uint64_t N_replicatedR,
	uint64_t* const rootStorageBytes,
	std::string* const failMsg)
{
	if (rootStorageBytes) { *rootStorageBytes = 0; }
	if (failMsg) { failMsg->clear(); }
	try
	{
		Sim::CheckedLocalParticleCount(N_gatheredF, "fluid export root output");
		Sim::CheckedLocalParticleCount(N_gatheredB, "boundary export root output");
		Sim::CheckedLocalParticleCount(N_replicatedR, "rigid export root output");

		constexpr uint64_t FluidSnapshotParticleBytes =
			sizeof(uint64_t) + 4ull * sizeof(float) + sizeof(uint16_t);
		constexpr uint64_t BoundarySnapshotParticleBytes =
			sizeof(uint64_t) + 4ull * sizeof(float);
		constexpr uint64_t RigidSnapshotParticleBytes =
			sizeof(uint64_t) + sizeof(uint32_t) + 4ull * sizeof(float);

		uint64_t N_bytes = CheckedByteSize<DistExportFluidParticlePacket>(N_gatheredF);
		N_bytes = Sim::CheckedParticleAdd(
			N_bytes,
			CheckedByteSize<DistExportBoundaryParticlePacket>(N_gatheredB),
			"distributed export gather storage");
		N_bytes = Sim::CheckedParticleAdd(
			N_bytes,
			Sim::CheckedParticleMultiply(
				N_gatheredF, FluidSnapshotParticleBytes, "fluid export snapshot storage"),
			"distributed export root storage");
		N_bytes = Sim::CheckedParticleAdd(
			N_bytes,
			Sim::CheckedParticleMultiply(
				N_gatheredB, BoundarySnapshotParticleBytes, "boundary export snapshot storage"),
			"distributed export root storage");
		N_bytes = Sim::CheckedParticleAdd(
			N_bytes,
			Sim::CheckedParticleMultiply(
				N_replicatedR, RigidSnapshotParticleBytes, "rigid export snapshot storage"),
			"distributed export root storage");

		if (rootStorageBytes) { *rootStorageBytes = N_bytes; }
		return true;
	}
	catch (const std::exception& ex)
	{
		if (failMsg) { *failMsg = ex.what(); }
		return false;
	}
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

		if (particlesF.id[localIdx] == Sim::INVALID_PARTICLE_ID)
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

		if (particlesB.id[localIdx] == Sim::INVALID_PARTICLE_ID)
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
	if (gatherBuffer.particlesF.size() > Sim::MaxLocalParticleCount)
	{
		return false;
	}
	if (gatherBuffer.particlesB.size() > Sim::MaxLocalParticleCount)
	{
		return false;
	}

	// gathered packets are ordered by particle ID in place
	std::vector<DistExportFluidParticlePacket>& sortedParticlesF = gatherBuffer.particlesF;
	std::vector<DistExportBoundaryParticlePacket>& sortedParticlesB = gatherBuffer.particlesB;
	tbb::parallel_sort(
		sortedParticlesF.begin(),
		sortedParticlesF.end(),
		[](const DistExportFluidParticlePacket& a, const DistExportFluidParticlePacket& b)
		{
			return a.id < b.id;
		});
	tbb::parallel_sort(
		sortedParticlesB.begin(),
		sortedParticlesB.end(),
		[](const DistExportBoundaryParticlePacket& a, const DistExportBoundaryParticlePacket& b)
		{
			return a.id < b.id;
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

	uint64_t prevParticleId = Sim::INVALID_PARTICLE_ID;
	for (uint32_t i = 0; i < exportSnap.N_F; i++)
	{
		const DistExportFluidParticlePacket& packet = sortedParticlesF[i];
		if (packet.id == Sim::INVALID_PARTICLE_ID) { return false; }
		if (packet.id == prevParticleId) { return false; }

		exportSnap.idF[i] = packet.id;
		exportSnap.xPosF[i] = packet.xPos;
		exportSnap.yPosF[i] = packet.yPos;
		exportSnap.zPosF[i] = packet.zPos;
		exportSnap.velMagF[i] = packet.velMag;
		exportSnap.rankF[i] = static_cast<uint16_t>(packet.rank);

		prevParticleId = packet.id;
	}

	prevParticleId = Sim::INVALID_PARTICLE_ID;
	for (uint32_t i = 0; i < exportSnap.N_B; i++)
	{
		const DistExportBoundaryParticlePacket& packet = sortedParticlesB[i];
		if (packet.id == Sim::INVALID_PARTICLE_ID) { return false; }
		if (packet.id == prevParticleId) { return false; }

		exportSnap.idB[i] = packet.id;
		exportSnap.xPosB[i] = packet.xPos;
		exportSnap.yPosB[i] = packet.yPos;
		exportSnap.zPosB[i] = packet.zPos;
		exportSnap.volB[i] = packet.volB;

		prevParticleId = packet.id;
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
	const uint32_t N_R = Sim::CheckedLocalParticleCount(R.N_, "rigid export root output");

	exportSnap.N_R = N_R;
	exportSnap.idR.assign(R.id.begin(), R.id.end());
	exportSnap.bodyIdR.assign(R.bodyId.begin(), R.bodyId.end());
	exportSnap.xPosR.resize(N_R);
	exportSnap.yPosR.resize(N_R);
	exportSnap.zPosR.resize(N_R);
	exportSnap.volR.assign(R.volR.begin(), R.volR.end());

	for (uint32_t rigidParticleIdx = 0; rigidParticleIdx < N_R; rigidParticleIdx++)
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

	uint64_t N_gatheredF = gatherBuffer.particlesF.size();
	uint64_t N_gatheredB = gatherBuffer.particlesB.size();
	uint32_t N_materializedF = 0;
	uint32_t N_materializedB = 0;
	uint32_t N_materializedR = 0;
	uint64_t N_payloadRequests = 0;
	try
	{
		if (mpiContext.IsRoot())
		{
			for (RankDistExportSnapshotGather& rankGather : rankGathers)
			{
				if (rankGather.N_recvF < 0 || rankGather.N_recvB < 0)
				{
					SPDLOG_ERROR(
						"[rank {}/{}] Distributed snapshot gather received a negative packet count "
						"from rank {}.",
						mpiContext.Rank(), mpiContext.Size(), rankGather.rank);
					localOk = false;
					break;
				}

				rankGather.recvFluidBeginIdx = N_gatheredF;
				rankGather.recvBoundaryBeginIdx = N_gatheredB;
				const uint32_t N_rankF = static_cast<uint32_t>(rankGather.N_recvF);
				const uint32_t N_rankB = static_cast<uint32_t>(rankGather.N_recvB);
				N_gatheredF = Sim::CheckedParticleAdd(
					N_gatheredF, N_rankF, "distributed export fluid count");
				N_gatheredB = Sim::CheckedParticleAdd(
					N_gatheredB, N_rankB, "distributed export boundary count");
				N_payloadRequests = Sim::CheckedParticleAdd(
					N_payloadRequests,
					ByteChunkRequestCount(
						CheckedByteSize<DistExportFluidParticlePacket>(N_rankF)),
					"distributed export payload request count");
				N_payloadRequests = Sim::CheckedParticleAdd(
					N_payloadRequests,
					ByteChunkRequestCount(
						CheckedByteSize<DistExportBoundaryParticlePacket>(N_rankB)),
					"distributed export payload request count");
			}

			if (localOk)
			{
				std::string capacityFailure{};
				localOk = ValidateDistExportSnapshotRootCapacity(
					N_gatheredF,
					N_gatheredB,
					particlesR.N_,
					nullptr,
					&capacityFailure);
				if (!localOk)
				{
					SPDLOG_ERROR(
						"[rank {}/{}] Cannot materialize requested distributed export: {}",
						mpiContext.Rank(), mpiContext.Size(), capacityFailure);
				}
				else
				{
					N_materializedF = Sim::CheckedLocalParticleCount(
						N_gatheredF, "fluid export root output");
					N_materializedB = Sim::CheckedLocalParticleCount(
						N_gatheredB, "boundary export root output");
					N_materializedR = Sim::CheckedLocalParticleCount(
						particlesR.N_, "rigid export root output");
				}
			}
		}
		else
		{
			N_payloadRequests = Sim::CheckedParticleAdd(
				ByteChunkRequestCount(CheckedByteSize(gatherBuffer.particlesF)),
				ByteChunkRequestCount(CheckedByteSize(gatherBuffer.particlesB)),
				"distributed export payload request count");
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Invalid distributed snapshot gather count or byte size: {}",
			mpiContext.Rank(), mpiContext.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext.AllTrue(localOk)) { return false; }

	try
	{
		if (mpiContext.IsRoot())
		{
			exportSnap.idF.reserve(N_materializedF);
			exportSnap.xPosF.reserve(N_materializedF);
			exportSnap.yPosF.reserve(N_materializedF);
			exportSnap.zPosF.reserve(N_materializedF);
			exportSnap.velMagF.reserve(N_materializedF);
			exportSnap.rankF.reserve(N_materializedF);
			exportSnap.idB.reserve(N_materializedB);
			exportSnap.xPosB.reserve(N_materializedB);
			exportSnap.yPosB.reserve(N_materializedB);
			exportSnap.zPosB.reserve(N_materializedB);
			exportSnap.volB.reserve(N_materializedB);
			exportSnap.idR.reserve(N_materializedR);
			exportSnap.bodyIdR.reserve(N_materializedR);
			exportSnap.xPosR.reserve(N_materializedR);
			exportSnap.yPosR.reserve(N_materializedR);
			exportSnap.zPosR.reserve(N_materializedR);
			exportSnap.volR.reserve(N_materializedR);
			gatherBuffer.particlesF.resize(N_materializedF);
			gatherBuffer.particlesB.resize(N_materializedB);
		}
		requests.reserve(N_payloadRequests);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to allocate distributed snapshot storage "
			"below the root-output capacity limit: {}",
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
				stats->N_gatheredF = exportSnap.N_F;
				stats->N_gatheredB = exportSnap.N_B;
				stats->N_gatheredR = exportSnap.N_R;
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
