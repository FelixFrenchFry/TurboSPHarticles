// dist_snapshots.cpp

// external
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

// internal
#include "dist_snapshots.h"
#include "mpi_context.h"



using namespace Dist;

namespace
{
	constexpr int DIST_SNAPSHOT_TAG_EXPORT_FLUID_COUNT = 7101;
	constexpr int DIST_SNAPSHOT_TAG_EXPORT_FLUID_PAYLOAD = 7102;
	constexpr int DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_COUNT = 7103;
	constexpr int DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_PAYLOAD = 7104;

	struct RankDistExportSnapshotGather
	{
		int rank = -1;

		DistExportSnapshotBuffer recvBuffer;

		int recvFluidCount = 0;
		int recvBoundaryCount = 0;
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
	int CheckedByteCount(const std::vector<T>& packets)
	{
		const uint64_t packetCount = packets.size();
		const uint64_t maxCount =
			static_cast<uint64_t>(std::numeric_limits<int>::max()) / sizeof(T);

		if (packetCount > maxCount)
		{
			throw std::runtime_error("Distributed snapshot payload exceeds MPI byte count range.");
		}

		return static_cast<int>(packetCount * sizeof(T));
	}

	template <typename T>
	const void* DataOrNull(const std::vector<T>& packets)
	{
		return packets.empty() ? nullptr : packets.data();
	}

	template <typename T>
	void* DataOrNull(std::vector<T>& packets)
	{
		return packets.empty() ? nullptr : packets.data();
	}

	bool IsLocalIdxInRange(
		const int localIdx,
		const uint32_t particleCount)
	{
		return localIdx >= 0 &&
			static_cast<uint32_t>(localIdx) < particleCount;
	}

	bool HasParticleMetadataAtLocalIdx(
		const DistParticleState& particleState,
		const int localIdx)
	{
		return localIdx >= 0 &&
			static_cast<uint64_t>(localIdx) < particleState.particleClasses.size() &&
			static_cast<uint64_t>(localIdx) < particleState.particleGlobalIdx_FromLocalIdx.size();
	}

	bool IsOwnedParticleClass(const DistParticleClass particleClass)
	{
		return particleClass == DistParticleClass::Inner ||
			particleClass == DistParticleClass::Outer;
	}

	DistExportFluidParticlePacket PackExportFluidParticle(
		const Sync::FluidParticles& fluidParticles,
		const DistParticleState& particleState,
		const int localIdx)
	{
		const float xVel = fluidParticles.xVel[localIdx];
		const float yVel = fluidParticles.yVel[localIdx];
		const float zVel = fluidParticles.zVel[localIdx];

		DistExportFluidParticlePacket packet{};
		packet.globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];

		packet.xPos = fluidParticles.xPos[localIdx];
		packet.yPos = fluidParticles.yPos[localIdx];
		packet.zPos = fluidParticles.zPos[localIdx];
		packet.velMag = std::sqrt(xVel * xVel + yVel * yVel + zVel * zVel);

		return packet;
	}

	DistExportBoundaryParticlePacket PackExportBoundaryParticle(
		const Sync::BoundaryParticles& boundaryParticles,
		const DistParticleState& particleState,
		const int localIdx)
	{
		DistExportBoundaryParticlePacket packet{};
		packet.globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];

		packet.xPos = boundaryParticles.xPos[localIdx];
		packet.yPos = boundaryParticles.yPos[localIdx];
		packet.zPos = boundaryParticles.zPos[localIdx];
		packet.volB = boundaryParticles.volB[localIdx];

		return packet;
	}

	void AppendDistExportSnapshotBuffer(
		DistExportSnapshotBuffer& dst,
		const DistExportSnapshotBuffer& src)
	{
		dst.fluidParticles.insert(
			dst.fluidParticles.end(),
			src.fluidParticles.begin(),
			src.fluidParticles.end());
		dst.boundaryParticles.insert(
			dst.boundaryParticles.end(),
			src.boundaryParticles.begin(),
			src.boundaryParticles.end());
	}
}

void DistExportSnapshotBuffer::Clear()
{
	fluidParticles.clear();
	boundaryParticles.clear();
}

bool Dist::PackOwnedExportSnapshotBuffer(
	DistExportSnapshotBuffer& gatherBuffer,
	const Sync::UniversalSimState& localState,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState)
{
	gatherBuffer.Clear();
	gatherBuffer.fluidParticles.reserve(fluidParticleState.particleLocalIdx_Owned.size());
	gatherBuffer.boundaryParticles.reserve(boundaryParticleState.particleLocalIdx_Owned.size());

	// fluid particles
	for (const int localIdx : fluidParticleState.particleLocalIdx_Owned)
	{
		if (!IsLocalIdxInRange(localIdx, localState.fluidParticles.N_)) { return false; }
		if (!HasParticleMetadataAtLocalIdx(fluidParticleState, localIdx)) { return false; }

		const DistParticleClass particleClass = fluidParticleState.particleClasses[localIdx];
		if (!IsOwnedParticleClass(particleClass)) { return false; }

		if (fluidParticleState.particleGlobalIdx_FromLocalIdx[localIdx] < 0) { return false; }

		gatherBuffer.fluidParticles.push_back(
			PackExportFluidParticle(localState.fluidParticles, fluidParticleState, localIdx));
	}

	// boundary particles
	for (const int localIdx : boundaryParticleState.particleLocalIdx_Owned)
	{
		if (!IsLocalIdxInRange(localIdx, localState.boundaryParticles.N_)) { return false; }
		if (!HasParticleMetadataAtLocalIdx(boundaryParticleState, localIdx)) { return false; }

		const DistParticleClass particleClass = boundaryParticleState.particleClasses[localIdx];
		if (!IsOwnedParticleClass(particleClass)) { return false; }

		if (boundaryParticleState.particleGlobalIdx_FromLocalIdx[localIdx] < 0) { return false; }

		gatherBuffer.boundaryParticles.push_back(
			PackExportBoundaryParticle(localState.boundaryParticles, boundaryParticleState, localIdx));
	}

	return true;
}

bool Dist::RebuildExportSnapshotFromGatheredParticles(
	Sync::ExportSnapshot& exportSnap,
	const DistExportSnapshotBuffer& gatherBuffer,
	const uint32_t simStep,
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

	std::vector<DistExportFluidParticlePacket> sortedFluidParticles = gatherBuffer.fluidParticles;
	std::vector<DistExportBoundaryParticlePacket> sortedBoundaryParticles = gatherBuffer.boundaryParticles;
	std::sort(
		sortedFluidParticles.begin(),
		sortedFluidParticles.end(),
		[](const DistExportFluidParticlePacket& a, const DistExportFluidParticlePacket& b)
		{
			return a.globalIdx < b.globalIdx;
		});
	std::sort(
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

	exportSnap.xPosF.resize(exportSnap.N_F);
	exportSnap.yPosF.resize(exportSnap.N_F);
	exportSnap.zPosF.resize(exportSnap.N_F);
	exportSnap.velMagF.resize(exportSnap.N_F);
	exportSnap.xPosB.resize(exportSnap.N_B);
	exportSnap.yPosB.resize(exportSnap.N_B);
	exportSnap.zPosB.resize(exportSnap.N_B);
	exportSnap.volB.resize(exportSnap.N_B);

	int prevGlobalIdx = -1;
	for (uint32_t i = 0; i < exportSnap.N_F; i++)
	{
		const DistExportFluidParticlePacket& packet = sortedFluidParticles[i];
		if (packet.globalIdx < 0) { return false; }
		if (packet.globalIdx == prevGlobalIdx) { return false; }

		exportSnap.xPosF[i] = packet.xPos;
		exportSnap.yPosF[i] = packet.yPos;
		exportSnap.zPosF[i] = packet.zPos;
		exportSnap.velMagF[i] = packet.velMag;

		prevGlobalIdx = packet.globalIdx;
	}

	prevGlobalIdx = -1;
	for (uint32_t i = 0; i < exportSnap.N_B; i++)
	{
		const DistExportBoundaryParticlePacket& packet = sortedBoundaryParticles[i];
		if (packet.globalIdx < 0) { return false; }
		if (packet.globalIdx == prevGlobalIdx) { return false; }

		exportSnap.xPosB[i] = packet.xPos;
		exportSnap.yPosB[i] = packet.yPos;
		exportSnap.zPosB[i] = packet.zPos;
		exportSnap.volB[i] = packet.volB;

		prevGlobalIdx = packet.globalIdx;
	}

	return true;
}

bool Dist::GatherExportSnapshotToRoot(
	Sync::ExportSnapshot& exportSnap,
	const Sync::UniversalSimState& localState,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	const MpiContext& mpiContext,
	const uint32_t simStep,
	const double simTime,
	DistSnapshotGatherStats* stats)
{
	if (stats) { *stats = {}; }

	DistExportSnapshotBuffer sendBuffer{};
	bool localOk = PackOwnedExportSnapshotBuffer(
		sendBuffer,
		localState,
		fluidParticleState,
		boundaryParticleState);

	int sendFluidCount = 0;
	int sendBoundaryCount = 0;
	if (localOk)
	{
		sendFluidCount = CheckedPacketCount(sendBuffer.fluidParticles);
		sendBoundaryCount = CheckedPacketCount(sendBuffer.boundaryParticles);
	}

	if (!mpiContext.AllTrue(localOk)) { return false; }

	DistExportSnapshotBuffer gatheredBuffer{};
	std::vector<RankDistExportSnapshotGather> rankGathers{};
	rankGathers.reserve(static_cast<uint64_t>(std::max(0, mpiContext.Size() - 1)));

	if (mpiContext.IsRoot())
	{
		gatheredBuffer = sendBuffer;

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
				rankGather.recvFluidCount,
				rankGather.rank,
				DIST_SNAPSHOT_TAG_EXPORT_FLUID_COUNT,
				requests.back());
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvInt(
				rankGather.recvBoundaryCount,
				rankGather.rank,
				DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_COUNT,
				requests.back());
		}
	}
	else
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			sendFluidCount,
			mpiContext.RootRank(),
			DIST_SNAPSHOT_TAG_EXPORT_FLUID_COUNT,
			requests.back());
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			sendBoundaryCount,
			mpiContext.RootRank(),
			DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_COUNT,
			requests.back());
	}
	mpiContext.WaitAll(requests);
	requests.clear();

	if (mpiContext.IsRoot())
	{
		for (RankDistExportSnapshotGather& rankGather : rankGathers)
		{
			if (rankGather.recvFluidCount < 0 || rankGather.recvBoundaryCount < 0)
			{
				localOk = false;
				break;
			}

			rankGather.recvBuffer.fluidParticles.resize(
				static_cast<uint64_t>(rankGather.recvFluidCount));
			rankGather.recvBuffer.boundaryParticles.resize(
				static_cast<uint64_t>(rankGather.recvBoundaryCount));
		}
	}
	if (!mpiContext.AllTrue(localOk)) { return false; }

	requests.reserve(rankGathers.size() * 2);

	// gather payloads after all root receive buffers are sized correctly
	if (mpiContext.IsRoot())
	{
		for (RankDistExportSnapshotGather& rankGather : rankGathers)
		{
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvBytes(
				DataOrNull(rankGather.recvBuffer.fluidParticles),
				CheckedByteCount(rankGather.recvBuffer.fluidParticles),
				rankGather.rank,
				DIST_SNAPSHOT_TAG_EXPORT_FLUID_PAYLOAD,
				requests.back());
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvBytes(
				DataOrNull(rankGather.recvBuffer.boundaryParticles),
				CheckedByteCount(rankGather.recvBuffer.boundaryParticles),
				rankGather.rank,
				DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_PAYLOAD,
				requests.back());
		}
	}
	else
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendBytes(
			DataOrNull(sendBuffer.fluidParticles),
			CheckedByteCount(sendBuffer.fluidParticles),
			mpiContext.RootRank(),
			DIST_SNAPSHOT_TAG_EXPORT_FLUID_PAYLOAD,
			requests.back());
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendBytes(
			DataOrNull(sendBuffer.boundaryParticles),
			CheckedByteCount(sendBuffer.boundaryParticles),
			mpiContext.RootRank(),
			DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_PAYLOAD,
			requests.back());
	}
	mpiContext.WaitAll(requests);

	if (mpiContext.IsRoot())
	{
		for (const RankDistExportSnapshotGather& rankGather : rankGathers)
		{
			AppendDistExportSnapshotBuffer(gatheredBuffer, rankGather.recvBuffer);
		}

		localOk = RebuildExportSnapshotFromGatheredParticles(
			exportSnap,
			gatheredBuffer,
			simStep,
			simTime);

		if (localOk && stats)
		{
			stats->gatheredFluidParticleCount =
				static_cast<uint32_t>(gatheredBuffer.fluidParticles.size());
			stats->gatheredBoundaryParticleCount =
				static_cast<uint32_t>(gatheredBuffer.boundaryParticles.size());
		}
	}

	return mpiContext.AllTrue(localOk);
}
