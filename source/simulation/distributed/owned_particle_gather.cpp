// owned_particle_gather.cpp

// external
#include <limits>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// internal
#include "source/base/parallel_checks.h"
#include "owned_particle_gather.h"
#include "mpi_context.h"



using namespace Dist;

namespace
{
	// particles packed per block, matching the distributed redistribution gather granularity
	inline constexpr uint32_t OWNED_PARTICLE_GATHER_BLOCK_SIZE = 2048;

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
		const uint32_t localIdx)
	{
		OwnedFluidParticlePacket packet{};

		packet.globalIdx = particlesF.id[localIdx];
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
		const uint32_t localIdx)
	{
		OwnedBoundaryParticlePacket packet{};

		packet.globalIdx = particlesB.id[localIdx];
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

	void InitializeGlobalOwnedState(
		Sync::UniversalSimState& globalOwnedState,
		const Sync::UniversalSimState& replicatedState,
		const uint32_t N_globalF, const uint32_t N_globalB,
		std::vector<uint8_t>& gatheredParticlesF,
		std::vector<uint8_t>& gatheredParticlesB)
	{
		globalOwnedState = {};
		globalOwnedState.ResizeFluid(N_globalF);
		globalOwnedState.ResizeBoundary(N_globalB);
		globalOwnedState.particlesR = replicatedState.particlesR;
		globalOwnedState.rigidBodies = replicatedState.rigidBodies;
		globalOwnedState.step = replicatedState.step;
		globalOwnedState.time = replicatedState.time;

		gatheredParticlesF.assign(N_globalF, 0);
		gatheredParticlesB.assign(N_globalB, 0);
	}

	bool CopyOwnedParticleGatherBufferToGlobal(
		Sync::UniversalSimState& globalOwnedState,
		const OwnedParticleGatherBuffer& gatherBuffer,
		std::vector<uint8_t>& gatheredParticlesF,
		std::vector<uint8_t>& gatheredParticlesB)
	{
		bool localOk = true;

		// TODO: parallelize with TBB?
		for (const OwnedFluidParticlePacket& packet : gatherBuffer.particlesF)
		{
			if (!MarkGatheredParticle(gatheredParticlesF, packet.globalIdx))
			{
				localOk = false;
				continue;
			}
			CopyFluidPacketToGlobal(packet, globalOwnedState.particlesF);
		}

		// TODO: parallelize with TBB?
		for (const OwnedBoundaryParticlePacket& packet : gatherBuffer.particlesB)
		{
			if (!MarkGatheredParticle(gatheredParticlesB, packet.globalIdx))
			{
				localOk = false;
				continue;
			}
			CopyBoundaryPacketToGlobal(packet, globalOwnedState.particlesB);
		}

		return localOk;
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
	const uint32_t N_localF = localState.particlesF.N_;
	const uint32_t N_localB = localState.particlesB.N_;

	// check that all fluid particle fields have the same size
	const auto fluidSizeMatches = [N_localF](const auto& values)
	{
		return values.size() == N_localF;
	};
	if (!fluidSizeMatches(localState.particlesF.id) ||
		!fluidSizeMatches(localState.particlesF.xPos) ||
		!fluidSizeMatches(localState.particlesF.yPos) ||
		!fluidSizeMatches(localState.particlesF.zPos) ||
		!fluidSizeMatches(localState.particlesF.xVel) ||
		!fluidSizeMatches(localState.particlesF.yVel) ||
		!fluidSizeMatches(localState.particlesF.zVel) ||
		!fluidSizeMatches(localState.particlesF.xVelPred) ||
		!fluidSizeMatches(localState.particlesF.yVelPred) ||
		!fluidSizeMatches(localState.particlesF.zVelPred) ||
		!fluidSizeMatches(localState.particlesF.xAccPress) ||
		!fluidSizeMatches(localState.particlesF.yAccPress) ||
		!fluidSizeMatches(localState.particlesF.zAccPress) ||
		!fluidSizeMatches(localState.particlesF.sourceTerm) ||
		!fluidSizeMatches(localState.particlesF.diagElem) ||
		!fluidSizeMatches(localState.particlesF.pressure) ||
		!fluidSizeMatches(localState.particlesF.density))
	{
		return false;
	}

	// check that all boundary particle fields have the same size
	const auto boundarySizeMatches = [N_localB](const auto& values)
	{
		return values.size() == N_localB;
	};
	if (!boundarySizeMatches(localState.particlesB.id) ||
		!boundarySizeMatches(localState.particlesB.xPos) ||
		!boundarySizeMatches(localState.particlesB.yPos) ||
		!boundarySizeMatches(localState.particlesB.zPos) ||
		!boundarySizeMatches(localState.particlesB.volB) ||
		!boundarySizeMatches(localState.particlesB.psiB))
	{
		return false;
	}

	// check metadata storage and owned counts once before the parallel passes
	if (particleStateF.N_owned > N_localF ||
		particleStateF.particleClasses.size() != N_localF ||
		particleStateF.particleLocalIdx_Owned.size() != particleStateF.N_owned ||
		particleStateB.N_owned > N_localB ||
		particleStateB.particleClasses.size() != N_localB ||
		particleStateB.particleLocalIdx_Owned.size() != particleStateB.N_owned)
	{
		return false;
	}

	const uint32_t N_ownedF = particleStateF.N_owned;
	const uint32_t N_ownedB = particleStateB.N_owned;
	gatherBuffer.particlesF.resize(N_ownedF);
	gatherBuffer.particlesB.resize(N_ownedB);

	if (N_ownedF != 0)
	{
		// pack particle fields sequentially within each parallel block
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_ownedF, OWNED_PARTICLE_GATHER_BLOCK_SIZE),
			[&](const tbb::blocked_range<uint32_t>& blockRange)
		{
			for (uint32_t ownedIdx = blockRange.begin();
				ownedIdx != blockRange.end(); ownedIdx++)
			{
				const uint32_t localIdx =
					particleStateF.particleLocalIdx_Owned[ownedIdx];

				gatherBuffer.particlesF[ownedIdx] =
					PackFluidParticle(localState.particlesF, localIdx);
			}
		});
	}

	if (N_ownedB != 0)
	{
		// pack particle fields sequentially within each parallel block
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_ownedB, OWNED_PARTICLE_GATHER_BLOCK_SIZE),
			[&](const tbb::blocked_range<uint32_t>& blockRange)
		{
			for (uint32_t ownedIdx = blockRange.begin();
				ownedIdx != blockRange.end(); ownedIdx++)
			{
				const uint32_t localIdx =
					particleStateB.particleLocalIdx_Owned[ownedIdx];

				gatherBuffer.particlesB[ownedIdx] =
					PackBoundaryParticle(localState.particlesB, localIdx);
			}
		});
	}

	return true;
}

bool Dist::GatherOwnedParticleBufferToRoot(
	Sync::UniversalSimState& globalOwnedState,
	const OwnedParticleGatherBuffer& sendBuffer,
	const Sync::UniversalSimState& localState,
	const MpiContext& mpiContext,
	const uint32_t N_globalF, const uint32_t N_globalB)
{
	int N_sendF = 0;
	int N_sendB = 0;
	N_sendF = CheckedPacketCount(sendBuffer.particlesF);
	N_sendB = CheckedPacketCount(sendBuffer.particlesB);

	bool localOk = true;

	std::vector<MPI_Request> requests{};
	requests.reserve(static_cast<uint64_t>(mpiContext.Size()) * 2);
	std::vector<int> N_recvF{};
	std::vector<int> N_recvB{};
	if (mpiContext.IsRoot())
	{
		N_recvF.resize(static_cast<uint64_t>(mpiContext.Size()), 0);
		N_recvB.resize(static_cast<uint64_t>(mpiContext.Size()), 0);
	}

	// gather packet counts before gathering raw particle payloads
	if (mpiContext.IsRoot())
	{
		for (int rank = 0; rank < mpiContext.Size(); rank++)
		{
			if (rank == mpiContext.RootRank()) { continue; }

			// receive fluid count from rank
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvInt(
				N_recvF[rank],
				rank,
				OWNED_PARTICLE_GATHER_TAG_FLUID_COUNT,
				requests.back());

			// receive boundary count from rank
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvInt(
				N_recvB[rank],
				rank,
				OWNED_PARTICLE_GATHER_TAG_BOUNDARY_COUNT,
				requests.back());
		}
	}
	else
	{
		// send fluid count to root
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			N_sendF,
			mpiContext.RootRank(),
			OWNED_PARTICLE_GATHER_TAG_FLUID_COUNT,
			requests.back());

		// send boundary count to root
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
		for (int rank = 0; rank < mpiContext.Size(); rank++)
		{
			if (rank == mpiContext.RootRank()) { continue; }

			if (N_recvF[rank] < 0 || N_recvB[rank] < 0)
			{
				localOk = false;
				break;
			}
		}
	}
	if (!mpiContext.AllTrue(localOk)) { return false; }

	std::vector<uint8_t> gatheredParticlesF{};
	std::vector<uint8_t> gatheredParticlesB{};
	if (mpiContext.IsRoot())
	{
		InitializeGlobalOwnedState(
			globalOwnedState,
			localState,
			N_globalF,
			N_globalB,
			gatheredParticlesF,
			gatheredParticlesB);

		localOk = CopyOwnedParticleGatherBufferToGlobal(
			globalOwnedState,
			sendBuffer,
			gatheredParticlesF,
			gatheredParticlesB);
	}
	if (!mpiContext.AllTrue(localOk)) { return false; }

	OwnedParticleGatherBuffer recvBuffer{};

	// gather ranks sequentially to limit root memory usage
	if (mpiContext.IsRoot())
	{
		for (int rank = 0; rank < mpiContext.Size(); rank++)
		{
			if (rank == mpiContext.RootRank()) { continue; }

			recvBuffer.particlesF.resize(static_cast<uint64_t>(N_recvF[rank]));
			recvBuffer.particlesB.resize(static_cast<uint64_t>(N_recvB[rank]));
			requests.clear();
			requests.reserve(2);

			// receive fluid payload from rank
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvBytes(
				DataOrNull(recvBuffer.particlesF),
				CheckedByteCount(recvBuffer.particlesF),
				rank,
				OWNED_PARTICLE_GATHER_TAG_FLUID_PAYLOAD,
				requests.back());

			// receive boundary payload from rank
			requests.emplace_back(MPI_REQUEST_NULL);
			mpiContext.IRecvBytes(
				DataOrNull(recvBuffer.particlesB),
				CheckedByteCount(recvBuffer.particlesB),
				rank,
				OWNED_PARTICLE_GATHER_TAG_BOUNDARY_PAYLOAD,
				requests.back());
			mpiContext.WaitAll(requests);

			localOk = CopyOwnedParticleGatherBufferToGlobal(
				globalOwnedState,
				recvBuffer,
				gatheredParticlesF,
				gatheredParticlesB) && localOk;
			recvBuffer.Clear();
		}
	}
	else
	{
		// send fluid paylod to root
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendBytes(
			DataOrNull(sendBuffer.particlesF),
			CheckedByteCount(sendBuffer.particlesF),
			mpiContext.RootRank(),
			OWNED_PARTICLE_GATHER_TAG_FLUID_PAYLOAD,
			requests.back());

		// send boundary payload to root
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendBytes(
			DataOrNull(sendBuffer.particlesB),
			CheckedByteCount(sendBuffer.particlesB),
			mpiContext.RootRank(),
			OWNED_PARTICLE_GATHER_TAG_BOUNDARY_PAYLOAD,
			requests.back());

		mpiContext.WaitAll(requests);
	}

	// check that all particles are now present at the root
	if (mpiContext.IsRoot())
	{
		localOk = AllParticlesGathered(gatheredParticlesF) && localOk;
		localOk = AllParticlesGathered(gatheredParticlesB) && localOk;
	}

	return mpiContext.AllTrue(localOk);
}
