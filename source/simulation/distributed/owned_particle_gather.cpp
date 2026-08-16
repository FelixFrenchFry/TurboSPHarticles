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
		const Sync::FluidParticles& particlesF,
		const DistParticleState& particleState,
		const uint32_t localIdx)
	{
		OwnedFluidParticlePacket packet{};
		packet.globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
		packet.id = particlesF.id[localIdx];

		packet.xPos = particlesF.xPos[localIdx];
		packet.yPos = particlesF.yPos[localIdx];
		packet.zPos = particlesF.zPos[localIdx];

		packet.xVel = particlesF.xVel[localIdx];
		packet.yVel = particlesF.yVel[localIdx];
		packet.zVel = particlesF.zVel[localIdx];

		packet.xVelPred = particlesF.xVelPred[localIdx];
		packet.yVelPred = particlesF.yVelPred[localIdx];
		packet.zVelPred = particlesF.zVelPred[localIdx];

		packet.xAccPress = particlesF.xAccPress[localIdx];
		packet.yAccPress = particlesF.yAccPress[localIdx];
		packet.zAccPress = particlesF.zAccPress[localIdx];

		packet.sourceTerm = particlesF.sourceTerm[localIdx];
		packet.diagElem = particlesF.diagElem[localIdx];
		packet.pressure = particlesF.pressure[localIdx];
		packet.density = particlesF.density[localIdx];

		return packet;
	}

	OwnedBoundaryParticlePacket PackBoundaryParticle(
		const Sync::BoundaryParticles& particlesB,
		const DistParticleState& particleState,
		const uint32_t localIdx)
	{
		OwnedBoundaryParticlePacket packet{};
		packet.globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
		packet.id = particlesB.id[localIdx];

		packet.xPos = particlesB.xPos[localIdx];
		packet.yPos = particlesB.yPos[localIdx];
		packet.zPos = particlesB.zPos[localIdx];
		packet.volB = particlesB.volB[localIdx];
		packet.psiB = particlesB.psiB[localIdx];

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
		dst.particlesF.insert(
			dst.particlesF.end(),
			src.particlesF.begin(),
			src.particlesF.end());
		dst.particlesB.insert(
			dst.particlesB.end(),
			src.particlesB.begin(),
			src.particlesB.end());
	}

} // namespace anonymous

void OwnedParticleGatherBuffer::Clear()
{
	particlesF.clear();
	particlesB.clear();
}

bool Dist::PackOwnedParticleGatherBuffer(
	OwnedParticleGatherBuffer& gatherBuffer,
	const Sync::UniversalSimState& localState,
	const DistParticleState& particleStateF,
	const DistParticleState& particleStateB)
{
	gatherBuffer.Clear();
	gatherBuffer.particlesF.reserve(particleStateF.particleLocalIdx_Owned.size());
	gatherBuffer.particlesB.reserve(particleStateB.particleLocalIdx_Owned.size());

	for (const uint32_t localIdx : particleStateF.particleLocalIdx_Owned)
	{
		if (localIdx >= localState.particlesF.N_) { return false; }
		if (!particleStateF.HasMetadataAtLocalIdx(localIdx)) { return false; }

		const DistParticleClass particleClass = particleStateF.particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer) { return false; }

		if (particleStateF.particleGlobalIdx_FromLocalIdx[localIdx] ==
			NO_GLOBAL_PARTICLE_IDX)
		{
			return false;
		}

		gatherBuffer.particlesF.push_back(
			PackFluidParticle(localState.particlesF, particleStateF, localIdx));
	}

	for (const uint32_t localIdx : particleStateB.particleLocalIdx_Owned)
	{
		if (localIdx >= localState.particlesB.N_) { return false; }
		if (!particleStateB.HasMetadataAtLocalIdx(localIdx)) { return false; }

		const DistParticleClass particleClass = particleStateB.particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer) { return false; }

		if (particleStateB.particleGlobalIdx_FromLocalIdx[localIdx] ==
			NO_GLOBAL_PARTICLE_IDX)
		{
			return false;
		}

		gatherBuffer.particlesB.push_back(
			PackBoundaryParticle(localState.particlesB, particleStateB, localIdx));
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

	std::vector<uint8_t> gatheredParticlesF(N_globalF, 0);
	std::vector<uint8_t> gatheredParticlesB(N_globalB, 0);

	for (const OwnedFluidParticlePacket& packet : gatherBuffer.particlesF)
	{
		if (!MarkGatheredParticle(gatheredParticlesF, packet.globalIdx)) { return false; }
		CopyFluidPacketToGlobal(packet, globalOwnedState.particlesF);
	}

	for (const OwnedBoundaryParticlePacket& packet : gatherBuffer.particlesB)
	{
		if (!MarkGatheredParticle(gatheredParticlesB, packet.globalIdx)) { return false; }
		CopyBoundaryPacketToGlobal(packet, globalOwnedState.particlesB);
	}

	if (!AllParticlesGathered(gatheredParticlesF)) { return false; }
	if (!AllParticlesGathered(gatheredParticlesB)) { return false; }

	return true;
}

bool Dist::GatherOwnedParticlesToRoot(
	Sync::UniversalSimState& globalOwnedState,
	const Sync::UniversalSimState& localState,
	const DistParticleState& particleStateF,
	const DistParticleState& particleStateB,
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
		particleStateF,
		particleStateB);

	int N_sendF = 0;
	int N_sendB = 0;
	if (localOk)
	{
		N_sendF = CheckedPacketCount(sendBuffer.particlesF);
		N_sendB = CheckedPacketCount(sendBuffer.particlesB);
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

			rankGather.recvBuffer.particlesF.resize(
				static_cast<uint64_t>(rankGather.N_recvF));
			rankGather.recvBuffer.particlesB.resize(
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
				DataOrNull(rankGather.recvBuffer.particlesF),
				CheckedByteCount(rankGather.recvBuffer.particlesF),
				rankGather.rank,
				OWNED_PARTICLE_GATHER_TAG_FLUID_PAYLOAD,
				requests.back());
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvBytes(
				DataOrNull(rankGather.recvBuffer.particlesB),
				CheckedByteCount(rankGather.recvBuffer.particlesB),
				rankGather.rank,
				OWNED_PARTICLE_GATHER_TAG_BOUNDARY_PAYLOAD,
				requests.back());
		}
	}
	else
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendBytes(
			DataOrNull(sendBuffer.particlesF),
			CheckedByteCount(sendBuffer.particlesF),
			mpiContext.RootRank(),
			OWNED_PARTICLE_GATHER_TAG_FLUID_PAYLOAD,
			requests.back());
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendBytes(
			DataOrNull(sendBuffer.particlesB),
			CheckedByteCount(sendBuffer.particlesB),
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
					static_cast<uint32_t>(gatheredBuffer.particlesF.size());
				stats->N_gatheredB =
					static_cast<uint32_t>(gatheredBuffer.particlesB.size());
			}
		}
	}

	return mpiContext.AllTrue(localOk);
}
