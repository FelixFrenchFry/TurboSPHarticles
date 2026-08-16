// owned_particle_gather.cpp

// external
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

// internal
#include "source/base/parallel_checks.h"
#include "owned_particle_gather.h"
#include "mpi_context.h"



using namespace Dist;

namespace
{
	struct RankOwnedParticleGather
	{
		int rank = -1;

		OwnedParticleGatherBuffer recvBuffer;

		int N_recvF = 0;
		int N_recvB = 0;
	};

	template <typename T>
	int CheckedPacketCount(const std::vector<T>& packets)
	{
		if (packets.size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
		{
			throw std::runtime_error(
				"Owned particle gather packet count exceeds MPI int count range.");
		}

		return static_cast<int>(packets.size());
	}

	template <typename T>
	int CheckedByteCount(const std::vector<T>& packets)
	{
		const uint64_t N_packets = packets.size();
		const uint64_t N_max =
			static_cast<uint64_t>(std::numeric_limits<int>::max()) / sizeof(T);

		if (N_packets > N_max)
		{
			throw std::runtime_error(
				"Owned particle gather payload exceeds MPI byte count range.");
		}

		return static_cast<int>(N_packets * sizeof(T));
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

	bool IsGlobalIdxInRange(
		const uint32_t globalIdx,
		const uint32_t N_particles)
	{
		return globalIdx < N_particles;
	}

	OwnedFluidParticlePacket PackFluidParticle(
		const Sync::FluidParticles& fluidParticles,
		const DistParticleState& particleState,
		const uint32_t localIdx)
	{
		OwnedFluidParticlePacket packet{};
		packet.globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
		packet.id = fluidParticles.id[localIdx];

		packet.xPos = fluidParticles.xPos[localIdx];
		packet.yPos = fluidParticles.yPos[localIdx];
		packet.zPos = fluidParticles.zPos[localIdx];

		packet.xVel = fluidParticles.xVel[localIdx];
		packet.yVel = fluidParticles.yVel[localIdx];
		packet.zVel = fluidParticles.zVel[localIdx];

		packet.xVelPred = fluidParticles.xVelPred[localIdx];
		packet.yVelPred = fluidParticles.yVelPred[localIdx];
		packet.zVelPred = fluidParticles.zVelPred[localIdx];

		packet.xAccPress = fluidParticles.xAccPress[localIdx];
		packet.yAccPress = fluidParticles.yAccPress[localIdx];
		packet.zAccPress = fluidParticles.zAccPress[localIdx];

		packet.sourceTerm = fluidParticles.sourceTerm[localIdx];
		packet.diagElem = fluidParticles.diagElem[localIdx];
		packet.pressure = fluidParticles.pressure[localIdx];
		packet.density = fluidParticles.density[localIdx];

		return packet;
	}

	OwnedBoundaryParticlePacket PackBoundaryParticle(
		const Sync::BoundaryParticles& boundaryParticles,
		const DistParticleState& particleState,
		const uint32_t localIdx)
	{
		OwnedBoundaryParticlePacket packet{};
		packet.globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
		packet.id = boundaryParticles.id[localIdx];

		packet.xPos = boundaryParticles.xPos[localIdx];
		packet.yPos = boundaryParticles.yPos[localIdx];
		packet.zPos = boundaryParticles.zPos[localIdx];
		packet.volB = boundaryParticles.volB[localIdx];
		packet.psiB = boundaryParticles.psiB[localIdx];

		return packet;
	}

	void CopyFluidPacketToGlobal(
		const OwnedFluidParticlePacket& packet,
		Sync::FluidParticles& dst)
	{
		const uint32_t globalIdx = packet.globalIdx;

		dst.id[globalIdx] = packet.id;
		dst.xPos[globalIdx] = packet.xPos;
		dst.yPos[globalIdx] = packet.yPos;
		dst.zPos[globalIdx] = packet.zPos;

		dst.xVel[globalIdx] = packet.xVel;
		dst.yVel[globalIdx] = packet.yVel;
		dst.zVel[globalIdx] = packet.zVel;

		dst.xVelPred[globalIdx] = packet.xVelPred;
		dst.yVelPred[globalIdx] = packet.yVelPred;
		dst.zVelPred[globalIdx] = packet.zVelPred;

		dst.xAccPress[globalIdx] = packet.xAccPress;
		dst.yAccPress[globalIdx] = packet.yAccPress;
		dst.zAccPress[globalIdx] = packet.zAccPress;

		dst.sourceTerm[globalIdx] = packet.sourceTerm;
		dst.diagElem[globalIdx] = packet.diagElem;
		dst.pressure[globalIdx] = packet.pressure;
		dst.density[globalIdx] = packet.density;
	}

	void CopyBoundaryPacketToGlobal(
		const OwnedBoundaryParticlePacket& packet,
		Sync::BoundaryParticles& dst)
	{
		const uint32_t globalIdx = packet.globalIdx;

		dst.id[globalIdx] = packet.id;
		dst.xPos[globalIdx] = packet.xPos;
		dst.yPos[globalIdx] = packet.yPos;
		dst.zPos[globalIdx] = packet.zPos;
		dst.volB[globalIdx] = packet.volB;
		dst.psiB[globalIdx] = packet.psiB;
	}

	bool MarkGatheredParticle(
		std::vector<uint8_t>& gatheredParticles,
		const uint32_t globalIdx)
	{
		if (!IsGlobalIdxInRange(globalIdx, static_cast<uint32_t>(gatheredParticles.size())))
		{
			return false;
		}

		if (gatheredParticles[globalIdx] != 0) { return false; }
		gatheredParticles[globalIdx] = 1;
		return true;
	}

	bool AllParticlesGathered(const std::vector<uint8_t>& gatheredParticles)
	{
		return Check::CountViolations(
			0,
			static_cast<uint32_t>(gatheredParticles.size()),
			[&](const uint32_t globalIdx)
			{
				return gatheredParticles[globalIdx] == 0;
			}) == 0;
	}

	void AppendOwnedParticleGatherBuffer(
		OwnedParticleGatherBuffer& dst,
		const OwnedParticleGatherBuffer& src)
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

} // namespace anonymous

void OwnedParticleGatherBuffer::Clear()
{
	fluidParticles.clear();
	boundaryParticles.clear();
}

bool Dist::PackOwnedParticleGatherBuffer(
	OwnedParticleGatherBuffer& gatherBuffer,
	const Sync::UniversalSimState& localState,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState)
{
	gatherBuffer.Clear();
	gatherBuffer.fluidParticles.reserve(fluidParticleState.particleLocalIdx_Owned.size());
	gatherBuffer.boundaryParticles.reserve(boundaryParticleState.particleLocalIdx_Owned.size());

	for (const uint32_t localIdx : fluidParticleState.particleLocalIdx_Owned)
	{
		if (localIdx >= localState.fluidParticles.N_) { return false; }
		if (!fluidParticleState.HasMetadataAtLocalIdx(localIdx)) { return false; }

		const DistParticleClass particleClass = fluidParticleState.particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer) { return false; }

		if (fluidParticleState.particleGlobalIdx_FromLocalIdx[localIdx] ==
			NO_GLOBAL_PARTICLE_IDX)
		{
			return false;
		}

		gatherBuffer.fluidParticles.push_back(
			PackFluidParticle(localState.fluidParticles, fluidParticleState, localIdx));
	}

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
			PackBoundaryParticle(localState.boundaryParticles, boundaryParticleState, localIdx));
	}

	return true;
}

bool Dist::RebuildGlobalOwnedStateFromGatheredParticles(
	Sync::UniversalSimState& globalOwnedState,
	const OwnedParticleGatherBuffer& gatherBuffer,
	const uint32_t N_globalF,
	const uint32_t N_globalB)
{
	globalOwnedState = {};
	globalOwnedState.ResizeFluid(N_globalF);
	globalOwnedState.ResizeBoundary(N_globalB);

	std::vector<uint8_t> gatheredFluidParticles(N_globalF, 0);
	std::vector<uint8_t> gatheredBoundaryParticles(N_globalB, 0);

	for (const OwnedFluidParticlePacket& packet : gatherBuffer.fluidParticles)
	{
		if (!MarkGatheredParticle(gatheredFluidParticles, packet.globalIdx)) { return false; }
		CopyFluidPacketToGlobal(packet, globalOwnedState.fluidParticles);
	}

	for (const OwnedBoundaryParticlePacket& packet : gatherBuffer.boundaryParticles)
	{
		if (!MarkGatheredParticle(gatheredBoundaryParticles, packet.globalIdx)) { return false; }
		CopyBoundaryPacketToGlobal(packet, globalOwnedState.boundaryParticles);
	}

	if (!AllParticlesGathered(gatheredFluidParticles)) { return false; }
	if (!AllParticlesGathered(gatheredBoundaryParticles)) { return false; }

	return true;
}

bool Dist::GatherOwnedParticlesToRoot(
	Sync::UniversalSimState& globalOwnedState,
	const Sync::UniversalSimState& localState,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	const MpiContext& mpiContext,
	const uint32_t N_globalF,
	const uint32_t N_globalB,
	OwnedParticleGatherStats* stats)
{
	if (stats) { *stats = {}; }

	OwnedParticleGatherBuffer sendBuffer{};
	bool localOk = PackOwnedParticleGatherBuffer(
		sendBuffer,
		localState,
		fluidParticleState,
		boundaryParticleState);

	int N_sendF = 0;
	int N_sendB = 0;
	if (localOk)
	{
		N_sendF = CheckedPacketCount(sendBuffer.fluidParticles);
		N_sendB = CheckedPacketCount(sendBuffer.boundaryParticles);
	}

	if (!mpiContext.AllTrue(localOk)) { return false; }

	OwnedParticleGatherBuffer gatheredBuffer{};
	std::vector<RankOwnedParticleGather> rankGathers{};
	rankGathers.reserve(static_cast<uint64_t>(std::max(0, mpiContext.Size() - 1)));

	if (mpiContext.IsRoot())
	{
		gatheredBuffer = sendBuffer;

		for (int rank = 0; rank < mpiContext.Size(); rank++)
		{
			if (rank == mpiContext.RootRank()) { continue; }

			RankOwnedParticleGather rankGather{};
			rankGather.rank = rank;
			rankGathers.push_back(std::move(rankGather));
		}
	}

	std::vector<MPI_Request> requests{};
	requests.reserve(rankGathers.size() * 2);

	// gather packet counts before gathering raw particle payloads
	if (mpiContext.IsRoot())
	{
		for (RankOwnedParticleGather& rankGather : rankGathers)
		{
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvInt(
				rankGather.N_recvF,
				rankGather.rank,
				OWNED_PARTICLE_GATHER_TAG_FLUID_COUNT,
				requests.back());
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvInt(
				rankGather.N_recvB,
				rankGather.rank,
				OWNED_PARTICLE_GATHER_TAG_BOUNDARY_COUNT,
				requests.back());
		}
	}
	else
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			N_sendF,
			mpiContext.RootRank(),
			OWNED_PARTICLE_GATHER_TAG_FLUID_COUNT,
			requests.back());
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			N_sendB,
			mpiContext.RootRank(),
			OWNED_PARTICLE_GATHER_TAG_BOUNDARY_COUNT,
			requests.back());
	}
	mpiContext.WaitAll(requests);
	requests.clear();

	if (mpiContext.IsRoot())
	{
		for (RankOwnedParticleGather& rankGather : rankGathers)
		{
			if (rankGather.N_recvF < 0 || rankGather.N_recvB < 0)
			{
				localOk = false;
				break;
			}

			rankGather.recvBuffer.fluidParticles.resize(
				static_cast<uint64_t>(rankGather.N_recvF));
			rankGather.recvBuffer.boundaryParticles.resize(
				static_cast<uint64_t>(rankGather.N_recvB));
		}
	}
	if (!mpiContext.AllTrue(localOk)) { return false; }

	requests.reserve(rankGathers.size() * 2);

	// gather payloads after all root receive buffers are sized correctly
	if (mpiContext.IsRoot())
	{
		for (RankOwnedParticleGather& rankGather : rankGathers)
		{
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvBytes(
				DataOrNull(rankGather.recvBuffer.fluidParticles),
				CheckedByteCount(rankGather.recvBuffer.fluidParticles),
				rankGather.rank,
				OWNED_PARTICLE_GATHER_TAG_FLUID_PAYLOAD,
				requests.back());
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvBytes(
				DataOrNull(rankGather.recvBuffer.boundaryParticles),
				CheckedByteCount(rankGather.recvBuffer.boundaryParticles),
				rankGather.rank,
				OWNED_PARTICLE_GATHER_TAG_BOUNDARY_PAYLOAD,
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
			OWNED_PARTICLE_GATHER_TAG_FLUID_PAYLOAD,
			requests.back());
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendBytes(
			DataOrNull(sendBuffer.boundaryParticles),
			CheckedByteCount(sendBuffer.boundaryParticles),
			mpiContext.RootRank(),
			OWNED_PARTICLE_GATHER_TAG_BOUNDARY_PAYLOAD,
			requests.back());
	}
	mpiContext.WaitAll(requests);

	if (mpiContext.IsRoot())
	{
		for (const RankOwnedParticleGather& rankGather : rankGathers)
		{
			AppendOwnedParticleGatherBuffer(gatheredBuffer, rankGather.recvBuffer);
		}

		localOk = RebuildGlobalOwnedStateFromGatheredParticles(
			globalOwnedState,
			gatheredBuffer,
			N_globalF,
			N_globalB);

		if (localOk)
		{
			globalOwnedState.step = localState.step;
			globalOwnedState.time = localState.time;

			if (stats)
			{
				stats->N_gatheredF =
					static_cast<uint32_t>(gatheredBuffer.fluidParticles.size());
				stats->N_gatheredB =
					static_cast<uint32_t>(gatheredBuffer.boundaryParticles.size());
			}
		}
	}

	return mpiContext.AllTrue(localOk);
}
