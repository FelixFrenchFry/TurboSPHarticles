// particle_migration.cpp

// external
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

// internal
#include "particle_migration.h"
#include "mpi_context.h"



using namespace Dist;

namespace
{
	constexpr int PARTICLE_MIGRATION_TAG_FLUID_COUNT = 5101;
	constexpr int PARTICLE_MIGRATION_TAG_FLUID_PAYLOAD = 5102;

	struct RankParticleMigration
	{
		int rank = -1;

		ParticleMigrationBuffer sendBuffer;
		ParticleMigrationBuffer recvBuffer;

		int sendFluidCount = 0;
		int recvFluidCount = 0;
	};

	template <typename T>
	int CheckedPacketCount(const std::vector<T>& packets)
	{
		if (packets.size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
		{
			throw std::runtime_error("Particle migration packet count exceeds MPI int count range.");
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
			throw std::runtime_error("Particle migration payload exceeds MPI byte count range.");
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

	glm::vec3 FluidPositionAtLocalIdx(
		const Sync::FluidParticles& fluidParticles,
		const int localIdx)
	{
		return
		{
			fluidParticles.xPos[localIdx],
			fluidParticles.yPos[localIdx],
			fluidParticles.zPos[localIdx]
		};
	}

	glm::vec3 FluidPositionFromPacket(const MigratingFluidParticlePacket& packet)
	{
		return
		{
			packet.xPos,
			packet.yPos,
			packet.zPos
		};
	}

	bool IsLocalIdxInRange(
		const int localIdx,
		const uint32_t particleCount)
	{
		return localIdx >= 0 &&
			static_cast<uint32_t>(localIdx) < particleCount;
	}

	bool IsGlobalIdxInRange(
		const int globalIdx,
		const uint32_t particleCount)
	{
		return globalIdx >= 0 &&
			static_cast<uint32_t>(globalIdx) < particleCount;
	}

	bool HasParticleMetadataAtLocalIdx(
		const DistParticleState& particleState,
		const int localIdx)
	{
		return localIdx >= 0 &&
			static_cast<uint64_t>(localIdx) < particleState.particleClasses.size() &&
			static_cast<uint64_t>(localIdx) < particleState.particleGlobalIdx_FromLocalIdx.size();
	}

	MigratingFluidParticlePacket PackFluidParticle(
		const Sync::FluidParticles& fluidParticles,
		const DistParticleState& particleState,
		const int localIdx)
	{
		MigratingFluidParticlePacket packet{};
		packet.globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];

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

	void CopyFluidParticle_LocalToLocal(
		const Sync::FluidParticles& src,
		Sync::FluidParticles& dst,
		const int srcLocalIdx,
		const int dstLocalIdx)
	{
		dst.xPos[dstLocalIdx] = src.xPos[srcLocalIdx];
		dst.yPos[dstLocalIdx] = src.yPos[srcLocalIdx];
		dst.zPos[dstLocalIdx] = src.zPos[srcLocalIdx];

		dst.xVel[dstLocalIdx] = src.xVel[srcLocalIdx];
		dst.yVel[dstLocalIdx] = src.yVel[srcLocalIdx];
		dst.zVel[dstLocalIdx] = src.zVel[srcLocalIdx];

		dst.xVelPred[dstLocalIdx] = src.xVelPred[srcLocalIdx];
		dst.yVelPred[dstLocalIdx] = src.yVelPred[srcLocalIdx];
		dst.zVelPred[dstLocalIdx] = src.zVelPred[srcLocalIdx];

		dst.xAccPress[dstLocalIdx] = src.xAccPress[srcLocalIdx];
		dst.yAccPress[dstLocalIdx] = src.yAccPress[srcLocalIdx];
		dst.zAccPress[dstLocalIdx] = src.zAccPress[srcLocalIdx];

		dst.sourceTerm[dstLocalIdx] = src.sourceTerm[srcLocalIdx];
		dst.diagElem[dstLocalIdx] = src.diagElem[srcLocalIdx];
		dst.pressure[dstLocalIdx] = src.pressure[srcLocalIdx];
		dst.density[dstLocalIdx] = src.density[srcLocalIdx];
	}

	void CopyFluidPacketToLocal(
		const MigratingFluidParticlePacket& packet,
		Sync::FluidParticles& dst,
		const int localIdx)
	{
		dst.xPos[localIdx] = packet.xPos;
		dst.yPos[localIdx] = packet.yPos;
		dst.zPos[localIdx] = packet.zPos;

		dst.xVel[localIdx] = packet.xVel;
		dst.yVel[localIdx] = packet.yVel;
		dst.zVel[localIdx] = packet.zVel;

		dst.xVelPred[localIdx] = packet.xVelPred;
		dst.yVelPred[localIdx] = packet.yVelPred;
		dst.zVelPred[localIdx] = packet.zVelPred;

		dst.xAccPress[localIdx] = packet.xAccPress;
		dst.yAccPress[localIdx] = packet.yAccPress;
		dst.zAccPress[localIdx] = packet.zAccPress;

		dst.sourceTerm[localIdx] = packet.sourceTerm;
		dst.diagElem[localIdx] = packet.diagElem;
		dst.pressure[localIdx] = packet.pressure;
		dst.density[localIdx] = packet.density;
	}

	bool AddParticleMapEntry(
		DistParticleState& particleState,
		const int globalIdx,
		const int localIdx,
		const uint32_t globalParticleCount)
	{
		if (!IsGlobalIdxInRange(globalIdx, globalParticleCount)) { return false; }
		if (particleState.particleLocalIdx_FromGlobalIdx[globalIdx] != -1) { return false; }

		particleState.particleGlobalIdx_FromLocalIdx[localIdx] = globalIdx;
		particleState.particleLocalIdx_FromGlobalIdx[globalIdx] = localIdx;
		return true;
	}

	bool CountKeptOwnedFluidParticles(
		uint32_t& keptParticleCount,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const IDomainDecomp& domainDecomp,
		const int rank)
	{
		keptParticleCount = 0;

		for (const int localIdx : fluidParticleState.particleLocalIdx_Owned)
		{
			if (!IsLocalIdxInRange(localIdx, localState.fluidParticles.N_)) { return false; }
			if (!HasParticleMetadataAtLocalIdx(fluidParticleState, localIdx)) { return false; }

			const int ownerRank =
				domainDecomp.OwnerRank(FluidPositionAtLocalIdx(localState.fluidParticles, localIdx));
			if (ownerRank < 0) { return false; }
			if (ownerRank == rank) { keptParticleCount++; }
		}

		return true;
	}
}

void ParticleMigrationBuffer::Clear()
{
	fluidParticles.clear();
}

bool Dist::PackMigratingFluidParticlesForRank(
	ParticleMigrationBuffer& sendBuffer,
	const Sync::UniversalSimState& localState,
	const DistParticleState& fluidParticleState,
	const IDomainDecomp& domainDecomp,
	const int dstRank)
{
	sendBuffer.Clear();
	if (dstRank < 0) { return false; }

	for (const int localIdx : fluidParticleState.particleLocalIdx_Owned)
	{
		if (!IsLocalIdxInRange(localIdx, localState.fluidParticles.N_)) { return false; }
		if (!HasParticleMetadataAtLocalIdx(fluidParticleState, localIdx)) { return false; }

		const DistParticleClass particleClass = fluidParticleState.particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer) { return false; }

		const int ownerRank =
			domainDecomp.OwnerRank(FluidPositionAtLocalIdx(localState.fluidParticles, localIdx));
		if (ownerRank < 0) { return false; }
		if (ownerRank != dstRank) { continue; }

		sendBuffer.fluidParticles.push_back(
			PackFluidParticle(localState.fluidParticles, fluidParticleState, localIdx));
	}

	return true;
}

bool Dist::RebuildLocalStateWithMigratedFluidParticles(
	Sync::UniversalSimState& localState,
	DistParticleState& fluidParticleState,
	const ParticleMigrationBuffer& recvBuffer,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float kernelRadius,
	const uint32_t globalFluidParticleCount)
{
	uint32_t keptParticleCount = 0;
	if (!CountKeptOwnedFluidParticles(
		keptParticleCount,
		localState,
		fluidParticleState,
		domainDecomp,
		rank)) { return false; }

	if (recvBuffer.fluidParticles.size() >
		static_cast<uint64_t>(std::numeric_limits<uint32_t>::max() - keptParticleCount))
	{
		return false;
	}

	const uint32_t receivedParticleCount =
		static_cast<uint32_t>(recvBuffer.fluidParticles.size());
	const uint32_t localParticleCount = keptParticleCount + receivedParticleCount;

	Sync::FluidParticles rebuiltFluidParticles{};
	DistParticleState rebuiltParticleState{};

	rebuiltFluidParticles.resizeAll(localParticleCount);
	rebuiltParticleState.Clear();
	rebuiltParticleState.particleGlobalIdx_FromLocalIdx.assign(localParticleCount, -1);
	rebuiltParticleState.particleLocalIdx_FromGlobalIdx.assign(globalFluidParticleCount, -1);

	int dstLocalIdx = 0;
	for (const int srcLocalIdx : fluidParticleState.particleLocalIdx_Owned)
	{
		if (!IsLocalIdxInRange(srcLocalIdx, localState.fluidParticles.N_)) { return false; }
		if (!HasParticleMetadataAtLocalIdx(fluidParticleState, srcLocalIdx)) { return false; }

		const int ownerRank =
			domainDecomp.OwnerRank(FluidPositionAtLocalIdx(localState.fluidParticles, srcLocalIdx));
		if (ownerRank < 0) { return false; }
		if (ownerRank != rank) { continue; }

		const int globalIdx = fluidParticleState.particleGlobalIdx_FromLocalIdx[srcLocalIdx];
		if (!AddParticleMapEntry(
			rebuiltParticleState,
			globalIdx,
			dstLocalIdx,
			globalFluidParticleCount)) { return false; }

		CopyFluidParticle_LocalToLocal(
			localState.fluidParticles,
			rebuiltFluidParticles,
			srcLocalIdx,
			dstLocalIdx);

		dstLocalIdx++;
	}

	for (const MigratingFluidParticlePacket& packet : recvBuffer.fluidParticles)
	{
		if (domainDecomp.OwnerRank(FluidPositionFromPacket(packet)) != rank) { return false; }
		if (!AddParticleMapEntry(
			rebuiltParticleState,
			packet.globalIdx,
			dstLocalIdx,
			globalFluidParticleCount)) { return false; }

		CopyFluidPacketToLocal(packet, rebuiltFluidParticles, dstLocalIdx);
		dstLocalIdx++;
	}

	if (dstLocalIdx != static_cast<int>(localParticleCount)) { return false; }

	localState.fluidParticles = std::move(rebuiltFluidParticles);
	fluidParticleState = std::move(rebuiltParticleState);

	ClassifyOwnedFluidParticlesIntoInnerOuter(
		fluidParticleState,
		localState.fluidParticles,
		domainDecomp,
		rank,
		kernelRadius);

	return fluidParticleState.IsValidLocalLayout(localParticleCount);
}

bool Dist::MigrateFluidParticlesToOwnerRanks(
	Sync::UniversalSimState& localState,
	DistParticleState& fluidParticleState,
	const IDomainDecomp& domainDecomp,
	const MpiContext& mpiContext,
	const float kernelRadius,
	const uint32_t globalFluidParticleCount,
	ParticleMigrationStats* stats)
{
	ParticleMigrationStats localStats{};

	std::vector<RankParticleMigration> migrations{};
	migrations.reserve(static_cast<uint64_t>(std::max(0, mpiContext.Size() - 1)));

	bool localOk = true;
	for (int rank = 0; rank < mpiContext.Size(); rank++)
	{
		if (rank == mpiContext.Rank()) { continue; }

		RankParticleMigration migration{};
		migration.rank = rank;
		if (!PackMigratingFluidParticlesForRank(
			migration.sendBuffer,
			localState,
			fluidParticleState,
			domainDecomp,
			rank))
		{
			localOk = false;
		}

		if (localOk)
		{
			migration.sendFluidCount =
				CheckedPacketCount(migration.sendBuffer.fluidParticles);
			localStats.sentFluidParticleCount +=
				static_cast<uint32_t>(migration.sendBuffer.fluidParticles.size());
		}

		migrations.push_back(std::move(migration));
	}

	if (!mpiContext.AllTrue(localOk)) { return false; }

	std::vector<MPI_Request> requests{};
	requests.reserve(migrations.size() * 2);

	// exchange migration counts before exchanging raw particle payloads
	for (RankParticleMigration& migration : migrations)
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.IRecvInt(
			migration.recvFluidCount,
			migration.rank,
			PARTICLE_MIGRATION_TAG_FLUID_COUNT,
			requests.back());
	}
	for (RankParticleMigration& migration : migrations)
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			migration.sendFluidCount,
			migration.rank,
			PARTICLE_MIGRATION_TAG_FLUID_COUNT,
			requests.back());
	}
	mpiContext.WaitAll(requests);
	requests.clear();

	for (RankParticleMigration& migration : migrations)
	{
		if (migration.recvFluidCount < 0) { localOk = false; }
		if (!localOk) { continue; }

		migration.recvBuffer.fluidParticles.resize(
			static_cast<uint64_t>(migration.recvFluidCount));
	}
	if (!mpiContext.AllTrue(localOk)) { return false; }

	// exchange payloads after all receive buffers are sized correctly
	for (RankParticleMigration& migration : migrations)
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.IRecvBytes(
			DataOrNull(migration.recvBuffer.fluidParticles),
			CheckedByteCount(migration.recvBuffer.fluidParticles),
			migration.rank,
			PARTICLE_MIGRATION_TAG_FLUID_PAYLOAD,
			requests.back());
	}
	for (RankParticleMigration& migration : migrations)
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendBytes(
			DataOrNull(migration.sendBuffer.fluidParticles),
			CheckedByteCount(migration.sendBuffer.fluidParticles),
			migration.rank,
			PARTICLE_MIGRATION_TAG_FLUID_PAYLOAD,
			requests.back());
	}
	mpiContext.WaitAll(requests);

	ParticleMigrationBuffer recvBuffer{};
	for (const RankParticleMigration& migration : migrations)
	{
		recvBuffer.fluidParticles.insert(
			recvBuffer.fluidParticles.end(),
			migration.recvBuffer.fluidParticles.begin(),
			migration.recvBuffer.fluidParticles.end());
	}

	localStats.receivedFluidParticleCount =
		static_cast<uint32_t>(recvBuffer.fluidParticles.size());

	localOk = RebuildLocalStateWithMigratedFluidParticles(
		localState,
		fluidParticleState,
		recvBuffer,
		domainDecomp,
		mpiContext.Rank(),
		kernelRadius,
		globalFluidParticleCount);

	const bool globalOk = mpiContext.AllTrue(localOk);
	if (globalOk && stats)
	{
		*stats = localStats;
	}

	return globalOk;
}
