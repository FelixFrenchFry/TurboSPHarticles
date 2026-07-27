// particle_redistribution.cpp

// external
#include <limits>
#include <stdexcept>
#include <utility>

// internal
#include "particle_redistribution.h"
#include "dist_redistribution_diagnostics.h"
#include "mpi_context.h"



using namespace Dist;

namespace
{
	// packed particle count lanes exchanged once per redistribution
	constexpr int PARTICLE_REDISTRIBUTION_LANE_COUNT = 3;
	constexpr int PARTICLE_REDISTRIBUTION_LANE_OWNED_FLUID = 0;
	constexpr int PARTICLE_REDISTRIBUTION_LANE_HALO_FLUID = 1;
	constexpr int PARTICLE_REDISTRIBUTION_LANE_HALO_BOUNDARY = 2;

	// disjoint MPI tag ranges for the three particle payloads
	constexpr int PARTICLE_REDISTRIBUTION_TAG_OWNED_FLUID = 5310;
	constexpr int PARTICLE_REDISTRIBUTION_TAG_HALO_FLUID = 5410;
	constexpr int PARTICLE_REDISTRIBUTION_TAG_HALO_BOUNDARY = 5510;

	// byte-sized payload helpers for chunked MPI transfers
	template <typename T>
	uint64_t CheckedByteSize(const std::vector<T>& values)
	{
		const uint64_t valueCount = values.size();
		const uint64_t maxCount = std::numeric_limits<uint64_t>::max() / sizeof(T);

		if (valueCount > maxCount)
		{
			throw std::runtime_error("Particle redistribution payload byte size exceeds uint64_t range.");
		}

		return valueCount * sizeof(T);
	}

	template <typename T>
	void QueueIRecvVector(
		const MpiContext& mpiContext,
		std::vector<T>& values,
		const int peerRank,
		const int tag,
		std::vector<MPI_Request>& requests)
	{
		mpiContext.IRecvBytesChunked(
			values.empty() ? nullptr : values.data(),
			CheckedByteSize(values),
			peerRank,
			tag,
			requests);
	}

	template <typename T>
	void QueueISendVector(
		const MpiContext& mpiContext,
		const std::vector<T>& values,
		const int peerRank,
		const int tag,
		std::vector<MPI_Request>& requests)
	{
		mpiContext.ISendBytesChunked(
			values.empty() ? nullptr : values.data(),
			CheckedByteSize(values),
			peerRank,
			tag,
			requests);
	}

	// memory accounting for manager-owned particle storage
	Mem::MemUsage GetFluidParticleMemoryUsage(
		const Sync::FluidParticles& particles)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.id));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.xPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.yPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.zPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.xVel));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.yVel));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.zVel));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.xVelPred));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.yVelPred));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.zVelPred));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.xAccPress));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.yAccPress));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.zAccPress));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.sourceTerm));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.diagElem));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.pressure));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.density));

		return usage;
	}

	Mem::MemUsage GetBoundaryParticleMemoryUsage(
		const Sync::BoundaryParticles& particles)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.id));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.xPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.yPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.zPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.volB));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particles.psiB));

		return usage;
	}

	Mem::MemUsage GetParticleStateMemoryUsage(
		const DistParticleState& particleState)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleClasses));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_Owned));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_Inner));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_Outer));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_Halo));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleGlobalIdx_FromLocalIdx));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_FromGlobalIdx));

		return usage;
	}

} // namespace anonymous

void ParticleRedistributionMgr::FluidExchangeBuffer::Clear()
{
	globalIdx_.clear();
	particles_.resizeAll(0);
}

void ParticleRedistributionMgr::FluidExchangeBuffer::Resize(
	const uint32_t N)
{
	globalIdx_.resize(N);
	particles_.resizeAll(N);
}

void ParticleRedistributionMgr::FluidExchangeBuffer::SetFromParticleView(
	const Sync::FluidConstView& src,
	const DistParticleState& particleState,
	const uint64_t bufferIdx,
	const int localIdx)
{
	globalIdx_[bufferIdx] = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
	particles_.id[bufferIdx] = src.id[localIdx];
	particles_.xPos[bufferIdx] = src.xPos[localIdx];
	particles_.yPos[bufferIdx] = src.yPos[localIdx];
	particles_.zPos[bufferIdx] = src.zPos[localIdx];
	particles_.xVel[bufferIdx] = src.xVel[localIdx];
	particles_.yVel[bufferIdx] = src.yVel[localIdx];
	particles_.zVel[bufferIdx] = src.zVel[localIdx];
	particles_.xVelPred[bufferIdx] = src.xVelPred[localIdx];
	particles_.yVelPred[bufferIdx] = src.yVelPred[localIdx];
	particles_.zVelPred[bufferIdx] = src.zVelPred[localIdx];
	particles_.xAccPress[bufferIdx] = src.xAccPress[localIdx];
	particles_.yAccPress[bufferIdx] = src.yAccPress[localIdx];
	particles_.zAccPress[bufferIdx] = src.zAccPress[localIdx];
	particles_.sourceTerm[bufferIdx] = src.sourceTerm[localIdx];
	particles_.diagElem[bufferIdx] = src.diagElem[localIdx];
	particles_.pressure[bufferIdx] = src.pressure[localIdx];
	particles_.density[bufferIdx] = src.density[localIdx];
}

void ParticleRedistributionMgr::FluidExchangeBuffer::CopyToLocalParticle(
	Sync::FluidParticles& dst,
	const uint64_t bufferIdx,
	const int localIdx) const
{
	dst.id[localIdx] = particles_.id[bufferIdx];
	dst.xPos[localIdx] = particles_.xPos[bufferIdx];
	dst.yPos[localIdx] = particles_.yPos[bufferIdx];
	dst.zPos[localIdx] = particles_.zPos[bufferIdx];
	dst.xVel[localIdx] = particles_.xVel[bufferIdx];
	dst.yVel[localIdx] = particles_.yVel[bufferIdx];
	dst.zVel[localIdx] = particles_.zVel[bufferIdx];
	dst.xVelPred[localIdx] = particles_.xVelPred[bufferIdx];
	dst.yVelPred[localIdx] = particles_.yVelPred[bufferIdx];
	dst.zVelPred[localIdx] = particles_.zVelPred[bufferIdx];
	dst.xAccPress[localIdx] = particles_.xAccPress[bufferIdx];
	dst.yAccPress[localIdx] = particles_.yAccPress[bufferIdx];
	dst.zAccPress[localIdx] = particles_.zAccPress[bufferIdx];
	dst.sourceTerm[localIdx] = particles_.sourceTerm[bufferIdx];
	dst.diagElem[localIdx] = particles_.diagElem[bufferIdx];
	dst.pressure[localIdx] = particles_.pressure[bufferIdx];
	dst.density[localIdx] = particles_.density[bufferIdx];
}

void ParticleRedistributionMgr::FluidExchangeBuffer::QueueIRecvPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests)
{
	if (IsEmpty()) { return; }

	QueueIRecvVector(mpiContext, globalIdx_, peerRank, tagBase + 1, requests);
	QueueIRecvVector(mpiContext, particles_.xPos, peerRank, tagBase + 2, requests);
	QueueIRecvVector(mpiContext, particles_.yPos, peerRank, tagBase + 3, requests);
	QueueIRecvVector(mpiContext, particles_.zPos, peerRank, tagBase + 4, requests);
	QueueIRecvVector(mpiContext, particles_.xVel, peerRank, tagBase + 5, requests);
	QueueIRecvVector(mpiContext, particles_.yVel, peerRank, tagBase + 6, requests);
	QueueIRecvVector(mpiContext, particles_.zVel, peerRank, tagBase + 7, requests);
	QueueIRecvVector(mpiContext, particles_.xVelPred, peerRank, tagBase + 8, requests);
	QueueIRecvVector(mpiContext, particles_.yVelPred, peerRank, tagBase + 9, requests);
	QueueIRecvVector(mpiContext, particles_.zVelPred, peerRank, tagBase + 10, requests);
	QueueIRecvVector(mpiContext, particles_.xAccPress, peerRank, tagBase + 11, requests);
	QueueIRecvVector(mpiContext, particles_.yAccPress, peerRank, tagBase + 12, requests);
	QueueIRecvVector(mpiContext, particles_.zAccPress, peerRank, tagBase + 13, requests);
	QueueIRecvVector(mpiContext, particles_.sourceTerm, peerRank, tagBase + 14, requests);
	QueueIRecvVector(mpiContext, particles_.diagElem, peerRank, tagBase + 15, requests);
	QueueIRecvVector(mpiContext, particles_.pressure, peerRank, tagBase + 16, requests);
	QueueIRecvVector(mpiContext, particles_.density, peerRank, tagBase + 17, requests);
	QueueIRecvVector(mpiContext, particles_.id, peerRank, tagBase + 18, requests);
}

void ParticleRedistributionMgr::FluidExchangeBuffer::QueueISendPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests) const
{
	if (IsEmpty()) { return; }

	QueueISendVector(mpiContext, globalIdx_, peerRank, tagBase + 1, requests);
	QueueISendVector(mpiContext, particles_.xPos, peerRank, tagBase + 2, requests);
	QueueISendVector(mpiContext, particles_.yPos, peerRank, tagBase + 3, requests);
	QueueISendVector(mpiContext, particles_.zPos, peerRank, tagBase + 4, requests);
	QueueISendVector(mpiContext, particles_.xVel, peerRank, tagBase + 5, requests);
	QueueISendVector(mpiContext, particles_.yVel, peerRank, tagBase + 6, requests);
	QueueISendVector(mpiContext, particles_.zVel, peerRank, tagBase + 7, requests);
	QueueISendVector(mpiContext, particles_.xVelPred, peerRank, tagBase + 8, requests);
	QueueISendVector(mpiContext, particles_.yVelPred, peerRank, tagBase + 9, requests);
	QueueISendVector(mpiContext, particles_.zVelPred, peerRank, tagBase + 10, requests);
	QueueISendVector(mpiContext, particles_.xAccPress, peerRank, tagBase + 11, requests);
	QueueISendVector(mpiContext, particles_.yAccPress, peerRank, tagBase + 12, requests);
	QueueISendVector(mpiContext, particles_.zAccPress, peerRank, tagBase + 13, requests);
	QueueISendVector(mpiContext, particles_.sourceTerm, peerRank, tagBase + 14, requests);
	QueueISendVector(mpiContext, particles_.diagElem, peerRank, tagBase + 15, requests);
	QueueISendVector(mpiContext, particles_.pressure, peerRank, tagBase + 16, requests);
	QueueISendVector(mpiContext, particles_.density, peerRank, tagBase + 17, requests);
	QueueISendVector(mpiContext, particles_.id, peerRank, tagBase + 18, requests);
}

int ParticleRedistributionMgr::FluidExchangeBuffer::CheckedCount() const
{
	if (Size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
	{
		throw std::runtime_error("Fluid particle redistribution count exceeds MPI int range.");
	}

	return static_cast<int>(Size());
}

int ParticleRedistributionMgr::FluidExchangeBuffer::GlobalIdxAt(
	const uint64_t bufferIdx) const
{
	return globalIdx_[bufferIdx];
}

Mem::MemUsage ParticleRedistributionMgr::FluidExchangeBuffer::GetMemoryUsage() const
{
	Mem::MemUsage usage = Mem::MemUsageOf_Vector(globalIdx_);

	Mem::Add(usage, GetFluidParticleMemoryUsage(particles_));

	return usage;
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::Clear()
{
	globalIdx_.clear();
	particles_.resizeAll(0);
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::Resize(
	const uint32_t N)
{
	globalIdx_.resize(N);
	particles_.resizeAll(N);
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::SetFromParticleView(
	const Sync::BoundaryConstView& src,
	const DistParticleState& particleState,
	const uint64_t bufferIdx,
	const int localIdx)
{
	globalIdx_[bufferIdx] = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
	particles_.id[bufferIdx] = src.id[localIdx];
	particles_.xPos[bufferIdx] = src.xPos[localIdx];
	particles_.yPos[bufferIdx] = src.yPos[localIdx];
	particles_.zPos[bufferIdx] = src.zPos[localIdx];
	particles_.volB[bufferIdx] = src.volB[localIdx];
	particles_.psiB[bufferIdx] = src.psiB[localIdx];
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::CopyToLocalParticle(
	Sync::BoundaryParticles& dst,
	const uint64_t bufferIdx,
	const int localIdx) const
{
	dst.id[localIdx] = particles_.id[bufferIdx];
	dst.xPos[localIdx] = particles_.xPos[bufferIdx];
	dst.yPos[localIdx] = particles_.yPos[bufferIdx];
	dst.zPos[localIdx] = particles_.zPos[bufferIdx];
	dst.volB[localIdx] = particles_.volB[bufferIdx];
	dst.psiB[localIdx] = particles_.psiB[bufferIdx];
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::QueueIRecvPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests)
{
	if (IsEmpty()) { return; }

	QueueIRecvVector(mpiContext, globalIdx_, peerRank, tagBase + 1, requests);
	QueueIRecvVector(mpiContext, particles_.xPos, peerRank, tagBase + 2, requests);
	QueueIRecvVector(mpiContext, particles_.yPos, peerRank, tagBase + 3, requests);
	QueueIRecvVector(mpiContext, particles_.zPos, peerRank, tagBase + 4, requests);
	QueueIRecvVector(mpiContext, particles_.volB, peerRank, tagBase + 5, requests);
	QueueIRecvVector(mpiContext, particles_.psiB, peerRank, tagBase + 6, requests);
	QueueIRecvVector(mpiContext, particles_.id, peerRank, tagBase + 7, requests);
}

void ParticleRedistributionMgr::BoundaryExchangeBuffer::QueueISendPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests) const
{
	if (IsEmpty()) { return; }

	QueueISendVector(mpiContext, globalIdx_, peerRank, tagBase + 1, requests);
	QueueISendVector(mpiContext, particles_.xPos, peerRank, tagBase + 2, requests);
	QueueISendVector(mpiContext, particles_.yPos, peerRank, tagBase + 3, requests);
	QueueISendVector(mpiContext, particles_.zPos, peerRank, tagBase + 4, requests);
	QueueISendVector(mpiContext, particles_.volB, peerRank, tagBase + 5, requests);
	QueueISendVector(mpiContext, particles_.psiB, peerRank, tagBase + 6, requests);
	QueueISendVector(mpiContext, particles_.id, peerRank, tagBase + 7, requests);
}

int ParticleRedistributionMgr::BoundaryExchangeBuffer::CheckedCount() const
{
	if (Size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
	{
		throw std::runtime_error("Boundary particle redistribution count exceeds MPI int range.");
	}

	return static_cast<int>(Size());
}

int ParticleRedistributionMgr::BoundaryExchangeBuffer::GlobalIdxAt(
	const uint64_t bufferIdx) const
{
	return globalIdx_[bufferIdx];
}

Mem::MemUsage ParticleRedistributionMgr::BoundaryExchangeBuffer::GetMemoryUsage() const
{
	Mem::MemUsage usage = Mem::MemUsageOf_Vector(globalIdx_);

	Mem::Add(usage, GetBoundaryParticleMemoryUsage(particles_));

	return usage;
}

Mem::MemUsage ParticleRedistributionMgr::GetExchangeBufferMemoryUsage() const
{
	Mem::MemUsage usage = Mem::MemUsageOf_Vector(rankBuffers_);

	for (const RankBuffers& rankBuffers : rankBuffers_)
	{
		Mem::Add(usage, rankBuffers.sendOwnedFluidParticles.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.recvOwnedFluidParticles.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.sendHaloFluidParticles.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.recvHaloFluidParticles.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.sendHaloBoundaryParticles.GetMemoryUsage());
		Mem::Add(usage, rankBuffers.recvHaloBoundaryParticles.GetMemoryUsage());
	}

	return usage;
}

Mem::MemUsage ParticleRedistributionMgr::GetScratchMemoryUsage() const
{
	Mem::MemUsage usage{};

	Mem::Add(usage, Mem::MemUsageOf_VectorOfVector(routes_.ownedFluidLocalIdxs_ByDstRank));
	Mem::Add(usage, Mem::MemUsageOf_VectorOfVector(routes_.haloFluidLocalIdxs_ByDstRank));
	Mem::Add(usage, Mem::MemUsageOf_VectorOfVector(routes_.haloBoundaryLocalIdxs_ByDstRank));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.keptOwnedFluidLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.keptOwnedBoundaryLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(routes_.localHaloFluidLocalIdxs));
	Mem::Add(usage, Mem::MemUsageOf_Vector(sendCounts_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(recvCounts_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(retainedFluidOldLocalIdx_FromNewLocalIdx_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(retainedBoundaryOldLocalIdx_FromNewLocalIdx_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(retainedOldLocalIdxFlags_));
	Mem::Add(usage, GetFluidParticleMemoryUsage(appendedFluidParticles_));
	Mem::Add(usage, GetBoundaryParticleMemoryUsage(appendedBoundaryParticles_));
	Mem::Add(usage, GetParticleStateMemoryUsage(rebuiltFluidParticleState_));
	Mem::Add(usage, GetParticleStateMemoryUsage(rebuiltBoundaryParticleState_));

	return usage;
}

#if defined(_DEBUG)
bool ParticleRedistributionMgr::IsLocalIdxInRange(
	const int localIdx,
	const uint32_t particleCount)
{
	return localIdx >= 0 &&
		static_cast<uint64_t>(localIdx) < static_cast<uint64_t>(particleCount);
}

bool ParticleRedistributionMgr::HasParticleMetadataAtLocalIdx(
	const DistParticleState& particleState,
	const int localIdx)
{
	return localIdx >= 0 &&
		static_cast<uint64_t>(localIdx) < particleState.particleClasses.size() &&
		static_cast<uint64_t>(localIdx) < particleState.particleGlobalIdx_FromLocalIdx.size();
}

bool ParticleRedistributionMgr::IsOwnedClass(
	const DistParticleClass particleClass)
{
	return particleClass == DistParticleClass::Inner ||
		   particleClass == DistParticleClass::Outer;
}
#endif

glm::vec3 ParticleRedistributionMgr::FluidPositionAtLocalIdx(
	const Sync::FluidConstView& particles,
	const int localIdx)
{
	return
	{
		particles.xPos[localIdx],
		particles.yPos[localIdx],
		particles.zPos[localIdx]
	};
}

glm::vec3 ParticleRedistributionMgr::BoundaryPositionAtLocalIdx(
	const Sync::BoundaryConstView& particles,
	const int localIdx)
{
	return
	{
		particles.xPos[localIdx],
		particles.yPos[localIdx],
		particles.zPos[localIdx]
	};
}

void ParticleRedistributionMgr::CopyFluidParticle(
	const Sync::FluidConstView& src,
	Sync::FluidParticles& dst,
	const int srcLocalIdx,
	const int dstLocalIdx)
{
	dst.id[dstLocalIdx] = src.id[srcLocalIdx];
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

void ParticleRedistributionMgr::CopyBoundaryParticle(
	const Sync::BoundaryConstView& src,
	Sync::BoundaryParticles& dst,
	const int srcLocalIdx,
	const int dstLocalIdx)
{
	dst.id[dstLocalIdx] = src.id[srcLocalIdx];
	dst.xPos[dstLocalIdx] = src.xPos[srcLocalIdx];
	dst.yPos[dstLocalIdx] = src.yPos[srcLocalIdx];
	dst.zPos[dstLocalIdx] = src.zPos[srcLocalIdx];
	dst.volB[dstLocalIdx] = src.volB[srcLocalIdx];
	dst.psiB[dstLocalIdx] = src.psiB[srcLocalIdx];
}

bool ParticleRedistributionMgr::BuildDenseCompactionMap(
	const uint32_t oldLocalParticleCount,
	const std::vector<int>& retainedOldLocalIdxs,
	std::vector<uint32_t>& oldLocalIdx_FromNewLocalIdx,
	std::vector<uint8_t>& retainedOldLocalIdxFlags,
	std::string* failMsg)
{
	const uint64_t retainedParticleCount = retainedOldLocalIdxs.size();
	if (retainedParticleCount > oldLocalParticleCount)
	{
		if (failMsg) { *failMsg = "Retained particle count exceeds old local particle count."; }
		return false;
	}

	// mark retained particles while rejecting invalid or duplicate routes
	// TODO: parallelize this with TBB?
	// TODO: look for HPC best practices in this use-case?
	retainedOldLocalIdxFlags.assign(oldLocalParticleCount, 0);
	for (const int oldLocalIdx : retainedOldLocalIdxs)
	{
		if (oldLocalIdx < 0 ||
			static_cast<uint64_t>(oldLocalIdx) >= oldLocalParticleCount ||
			retainedOldLocalIdxFlags[oldLocalIdx] != 0)
		{
			if (failMsg) { *failMsg = "Retained particle local index is invalid or duplicated."; }
			return false;
		}
		retainedOldLocalIdxFlags[oldLocalIdx] = 1;
	}

	// preserve retained prefix slots and fill each hole from the retained old tail
	oldLocalIdx_FromNewLocalIdx.resize(retainedParticleCount);
	uint32_t oldTailLocalIdx_Excl = oldLocalParticleCount;
	for (uint64_t newLocalIdx = 0; newLocalIdx != retainedParticleCount; newLocalIdx++)
	{
		if (retainedOldLocalIdxFlags[newLocalIdx] != 0)
		{
			oldLocalIdx_FromNewLocalIdx[newLocalIdx] = static_cast<uint32_t>(newLocalIdx);
			continue;
		}

		while (oldTailLocalIdx_Excl > retainedParticleCount &&
			   retainedOldLocalIdxFlags[oldTailLocalIdx_Excl - 1] == 0)
		{
			oldTailLocalIdx_Excl--;
		}
		if (oldTailLocalIdx_Excl <= retainedParticleCount)
		{
			if (failMsg) { *failMsg = "Retained particle compaction map is inconsistent."; }
			return false;
		}

		oldTailLocalIdx_Excl--;
		oldLocalIdx_FromNewLocalIdx[newLocalIdx] = oldTailLocalIdx_Excl;
	}

	return true;
}

void ParticleRedistributionMgr::PrepareRebuiltParticleState(
	DistParticleState& particleState,
	const uint32_t ownedParticleCount,
	const uint32_t haloParticleCount,
	const uint32_t localParticleCount,
	const uint32_t globalParticleCount)
{
	// clear only global mappings written by the previous scratch state
	auto& localIdx_FromGlobalIdx = particleState.particleLocalIdx_FromGlobalIdx;
	if (localIdx_FromGlobalIdx.size() == globalParticleCount)
	{
		// TODO: parallelize this with TBB?
		// TODO: look for HPC best practices in this use-case?
		for (const int globalIdx : particleState.particleGlobalIdx_FromLocalIdx)
		{
			if (globalIdx < 0 ||
				static_cast<uint64_t>(globalIdx) >= localIdx_FromGlobalIdx.size())
			{
				continue;
			}
			localIdx_FromGlobalIdx[globalIdx] = -1;
		}
	}
	else
	{
		localIdx_FromGlobalIdx.assign(globalParticleCount, -1);
	}

	// initialize compact local metadata while retaining reusable allocations
	particleState.ClearParticleClasses();
	particleState.ownedParticleCount = ownedParticleCount;
	particleState.haloParticleCount = haloParticleCount;
	particleState.particleClasses.assign(localParticleCount, DistParticleClass::None);
	particleState.particleGlobalIdx_FromLocalIdx.assign(localParticleCount, -1);
	particleState.particleLocalIdx_Owned.reserve(ownedParticleCount);
	particleState.particleLocalIdx_Inner.reserve(ownedParticleCount);
	particleState.particleLocalIdx_Outer.reserve(ownedParticleCount);
	particleState.particleLocalIdx_Halo.reserve(haloParticleCount);
}

bool ParticleRedistributionMgr::AddParticleMetadata(
	DistParticleState& particleState,
	const DistParticleClass particleClass,
	const uint32_t globalParticleCount,
	const int globalIdx,
	const int localIdx)
{
	if (globalIdx < 0 ||
		static_cast<uint64_t>(globalIdx) >= globalParticleCount ||
		particleState.particleLocalIdx_FromGlobalIdx[globalIdx] != -1)
	{
		return false;
	}

	particleState.particleGlobalIdx_FromLocalIdx[localIdx] = globalIdx;
	particleState.particleLocalIdx_FromGlobalIdx[globalIdx] = localIdx;
	particleState.particleClasses[localIdx] = particleClass;

	if (particleClass == DistParticleClass::Halo)
	{
		particleState.particleLocalIdx_Halo.push_back(localIdx);
	}
	else
	{
		particleState.particleLocalIdx_Owned.push_back(localIdx);
		if (particleClass == DistParticleClass::Inner)
		{
			particleState.particleLocalIdx_Inner.push_back(localIdx);
		}
		else
		{
			particleState.particleLocalIdx_Outer.push_back(localIdx);
		}
	}

	return true;
}

bool ParticleRedistributionMgr::CheckedUint32(
	const uint64_t value,
	uint32_t& result,
	const char* valueName,
	std::string* failMsg)
{
	if (value > std::numeric_limits<uint32_t>::max())
	{
		if (failMsg)
		{
			*failMsg = std::string(valueName) + " exceeds uint32_t range.";
		}
		return false;
	}

	result = static_cast<uint32_t>(value);
	return true;
}

bool ParticleRedistributionMgr::ValidateParticleViews(
	const Sync::FluidConstView& fluidParticles,
	const Sync::BoundaryConstView& boundaryParticles,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	std::string* failMsg)
{
	const uint32_t N_F_local = fluidParticles.N_;
	const uint32_t N_B_local = boundaryParticles.N_;

	// check that all fluid particle fields have the same size
	const auto fluidSizeMatches = [N_F_local](const auto& values)
	{
		return values.size() == N_F_local;
	};
	if (!fluidSizeMatches(fluidParticles.id) ||
		!fluidSizeMatches(fluidParticles.xPos) ||
		!fluidSizeMatches(fluidParticles.yPos) ||
		!fluidSizeMatches(fluidParticles.zPos) ||
		!fluidSizeMatches(fluidParticles.xVel) ||
		!fluidSizeMatches(fluidParticles.yVel) ||
		!fluidSizeMatches(fluidParticles.zVel) ||
		!fluidSizeMatches(fluidParticles.xVelPred) ||
		!fluidSizeMatches(fluidParticles.yVelPred) ||
		!fluidSizeMatches(fluidParticles.zVelPred) ||
		!fluidSizeMatches(fluidParticles.xAccPress) ||
		!fluidSizeMatches(fluidParticles.yAccPress) ||
		!fluidSizeMatches(fluidParticles.zAccPress) ||
		!fluidSizeMatches(fluidParticles.sourceTerm) ||
		!fluidSizeMatches(fluidParticles.diagElem) ||
		!fluidSizeMatches(fluidParticles.pressure) ||
		!fluidSizeMatches(fluidParticles.density))
	{
		if (failMsg) { *failMsg = "Backend fluid particle view has inconsistent field sizes."; }
		return false;
	}

	// check that all boundary particle fields have the same size
	const auto boundarySizeMatches = [N_B_local](const auto& values)
	{
		return values.size() == N_B_local;
	};
	if (!boundarySizeMatches(boundaryParticles.id) ||
		!boundarySizeMatches(boundaryParticles.xPos) ||
		!boundarySizeMatches(boundaryParticles.yPos) ||
		!boundarySizeMatches(boundaryParticles.zPos) ||
		!boundarySizeMatches(boundaryParticles.volB) ||
		!boundarySizeMatches(boundaryParticles.psiB))
	{
		if (failMsg) { *failMsg = "Backend boundary particle view has inconsistent field sizes."; }
		return false;
	}

	#if defined(_DEBUG)
	// exhaustively validate distributed particle metadata
	const uint32_t N_F_global = static_cast<uint32_t>(
		fluidParticleState.particleLocalIdx_FromGlobalIdx.size());
	if (!fluidParticleState.IsValidLocalParticleState(
		N_F_local, N_F_global, failMsg))
	{
		return false;
	}
	const uint32_t N_B_global = static_cast<uint32_t>(
		boundaryParticleState.particleLocalIdx_FromGlobalIdx.size());
	if (!boundaryParticleState.IsValidLocalParticleState(
		N_B_local, N_B_global, failMsg))
	{
		return false;
	}
	#else
	(void)fluidParticleState;
	(void)boundaryParticleState;
	#endif

	return true;
}

void ParticleRedistributionMgr::PrepareBuffers(const int rankCount)
{
	rankBuffers_.resize(static_cast<uint64_t>(rankCount));
	routes_.ownedFluidLocalIdxs_ByDstRank.resize(static_cast<uint64_t>(rankCount));
	routes_.haloFluidLocalIdxs_ByDstRank.resize(static_cast<uint64_t>(rankCount));
	routes_.haloBoundaryLocalIdxs_ByDstRank.resize(static_cast<uint64_t>(rankCount));

	// reset all send/recv buffers of this rank towards all other ranks
	for (int rank = 0; rank != rankCount; rank++)
	{
		auto& rankBuffers = rankBuffers_[rank];
		rankBuffers.sendOwnedFluidParticles.Clear();
		rankBuffers.recvOwnedFluidParticles.Clear();
		rankBuffers.sendHaloFluidParticles.Clear();
		rankBuffers.recvHaloFluidParticles.Clear();
		rankBuffers.sendHaloBoundaryParticles.Clear();
		rankBuffers.recvHaloBoundaryParticles.Clear();
		rankBuffers.sendOwnedFluidCount = 0;
		rankBuffers.recvOwnedFluidCount = 0;
		rankBuffers.sendHaloFluidCount = 0;
		rankBuffers.recvHaloFluidCount = 0;
		rankBuffers.sendHaloBoundaryCount = 0;
		rankBuffers.recvHaloBoundaryCount = 0;

		routes_.ownedFluidLocalIdxs_ByDstRank[rank].clear();
		routes_.haloFluidLocalIdxs_ByDstRank[rank].clear();
		routes_.haloBoundaryLocalIdxs_ByDstRank[rank].clear();
	}

	routes_.keptOwnedFluidLocalIdxs.clear();
	routes_.keptOwnedBoundaryLocalIdxs.clear();
	routes_.localHaloFluidLocalIdxs.clear();
	sendCounts_.assign(static_cast<uint64_t>(rankCount) * PARTICLE_REDISTRIBUTION_LANE_COUNT, 0);
	recvCounts_.clear();
}

bool ParticleRedistributionMgr::RouteOwnedParticles(
	const Sync::FluidConstView& fluidParticles,
	const Sync::BoundaryConstView& boundaryParticles,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	const IDomainDecomp& domainDecomp,
	const int rank, const int rankCount,
	std::string* failMsg,
	ParticleRedistributionTimingReport* timingReport)
{
	// route fluid migration and halos
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::RouteFluid);

		// TODO: parallelize this with TBB?
		for (const int localIdx : fluidParticleState.particleLocalIdx_Owned)
		{
			#if defined(_DEBUG)
			if (!IsLocalIdxInRange(localIdx, fluidParticles.N_) ||
				!HasParticleMetadataAtLocalIdx(fluidParticleState, localIdx) ||
				!IsOwnedClass(fluidParticleState.particleClasses[localIdx]))
			{
				if (failMsg) { *failMsg = "Invalid owned fluid particle metadata."; }
				return false;
			}
			#endif

			// TODO: use explicit SIMD, similar to sph_solver.cpp implementation?
			// TODO: look for HPC best practices in this use-case?
			const glm::vec3 position = FluidPositionAtLocalIdx(fluidParticles, localIdx);
			const int ownerRank = domainDecomp.OwnerRank(position);
			if (ownerRank < 0 || ownerRank >= rankCount)
			{
				if (failMsg) { *failMsg = "Fluid particle owner rank is out of range."; }
				return false;
			}

			if (ownerRank == rank)
			{
				routes_.keptOwnedFluidLocalIdxs.push_back(localIdx);
			}
			else
			{
				routes_.ownedFluidLocalIdxs_ByDstRank[ownerRank].push_back(localIdx);
			}

			for (const int haloRank : domainDecomp.NeighborRanks(ownerRank))
			{
				#if defined(_DEBUG)
				if (haloRank < 0 || haloRank >= rankCount || haloRank == ownerRank)
				{
					if (failMsg) { *failMsg = "Fluid halo destination rank is invalid."; }
					return false;
				}
				#endif

				if (!domainDecomp.HaloExtentAABB(haloRank).Contains(position)) { continue; }

				if (haloRank == rank)
				{
					routes_.localHaloFluidLocalIdxs.push_back(localIdx);
				}
				else
				{
					routes_.haloFluidLocalIdxs_ByDstRank[haloRank].push_back(localIdx);
				}
			}
		}
	}

	// route boundary halos
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::RouteBoundary);

		// TODO: parallelize this with TBB?
		for (const int localIdx : boundaryParticleState.particleLocalIdx_Owned)
		{
			#if defined(_DEBUG)
			if (!IsLocalIdxInRange(localIdx, boundaryParticles.N_) ||
				!HasParticleMetadataAtLocalIdx(boundaryParticleState, localIdx) ||
				!IsOwnedClass(boundaryParticleState.particleClasses[localIdx]))
			{
				if (failMsg) { *failMsg = "Invalid owned boundary particle metadata."; }
				return false;
			}
			#endif

			// TODO: use explicit SIMD, similar to sph_solver.cpp implementation?
			// TODO: look for HPC best practices in this use-case?
			const glm::vec3 position = BoundaryPositionAtLocalIdx(boundaryParticles, localIdx);
			if (domainDecomp.OwnerRank(position) != rank)
			{
				if (failMsg) { *failMsg = "Owned boundary particle requires migration."; }
				return false;
			}
			routes_.keptOwnedBoundaryLocalIdxs.push_back(localIdx);

			for (const int haloRank : domainDecomp.NeighborRanks(rank))
			{
				#if defined(_DEBUG)
				if (haloRank < 0 || haloRank >= rankCount || haloRank == rank)
				{
					if (failMsg) { *failMsg = "Boundary halo destination rank is invalid."; }
					return false;
				}
				#endif

				if (domainDecomp.HaloExtentAABB(haloRank).Contains(position))
				{
					routes_.haloBoundaryLocalIdxs_ByDstRank[haloRank].push_back(localIdx);
				}
			}
		}
	}

	return true;
}

bool ParticleRedistributionMgr::PackSendBuffers(
	const Sync::FluidConstView& fluidParticles,
	const Sync::BoundaryConstView& boundaryParticles,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	const int rank, const int rankCount,
	std::string* failMsg)
{
	// TODO: use uint32_t instead?
	uint64_t sentOwnedFluidCount = 0;
	uint64_t sentHaloFluidCount = 0;
	uint64_t sentHaloBoundaryCount = 0;

	// build one compact send buffer per destination rank
	for (int dstRank = 0; dstRank != rankCount; dstRank++)
	{
		auto& rankBuffers = rankBuffers_[dstRank];
		if (dstRank == rank) { continue; }

		const auto& ownedFluidLocalIdxs =
			routes_.ownedFluidLocalIdxs_ByDstRank[dstRank];
		const auto& haloFluidLocalIdxs =
			routes_.haloFluidLocalIdxs_ByDstRank[dstRank];
		const auto& haloBoundaryLocalIdxs =
			routes_.haloBoundaryLocalIdxs_ByDstRank[dstRank];

		rankBuffers.sendOwnedFluidParticles.Resize(
			static_cast<uint32_t>(ownedFluidLocalIdxs.size()));
		rankBuffers.sendHaloFluidParticles.Resize(
			static_cast<uint32_t>(haloFluidLocalIdxs.size()));
		rankBuffers.sendHaloBoundaryParticles.Resize(
			static_cast<uint32_t>(haloBoundaryLocalIdxs.size()));

		// TODO: parallelize this with TBB?
		// TODO: use explicit SIMD, similar to sph_solver.cpp implementation?
		// TODO: look for HPC best practices in this use-case?
		for (uint64_t bufferIdx = 0; bufferIdx != ownedFluidLocalIdxs.size(); bufferIdx++)
		{
			rankBuffers.sendOwnedFluidParticles.SetFromParticleView(
				fluidParticles, fluidParticleState, bufferIdx,
				ownedFluidLocalIdxs[bufferIdx]);
		}
		for (uint64_t bufferIdx = 0; bufferIdx != haloFluidLocalIdxs.size(); bufferIdx++)
		{
			rankBuffers.sendHaloFluidParticles.SetFromParticleView(
				fluidParticles, fluidParticleState, bufferIdx,
				haloFluidLocalIdxs[bufferIdx]);
		}
		for (uint64_t bufferIdx = 0; bufferIdx != haloBoundaryLocalIdxs.size(); bufferIdx++)
		{
			rankBuffers.sendHaloBoundaryParticles.SetFromParticleView(
				boundaryParticles, boundaryParticleState, bufferIdx,
				haloBoundaryLocalIdxs[bufferIdx]);
		}

		rankBuffers.sendOwnedFluidCount = rankBuffers.sendOwnedFluidParticles.CheckedCount();
		rankBuffers.sendHaloFluidCount = rankBuffers.sendHaloFluidParticles.CheckedCount();
		rankBuffers.sendHaloBoundaryCount = rankBuffers.sendHaloBoundaryParticles.CheckedCount();

		const uint64_t countBase = static_cast<uint64_t>(dstRank) * PARTICLE_REDISTRIBUTION_LANE_COUNT;
		sendCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_OWNED_FLUID] =
			rankBuffers.sendOwnedFluidCount;
		sendCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_HALO_FLUID] =
			rankBuffers.sendHaloFluidCount;
		sendCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_HALO_BOUNDARY] =
			rankBuffers.sendHaloBoundaryCount;

		sentOwnedFluidCount += rankBuffers.sendOwnedFluidCount;
		sentHaloFluidCount += rankBuffers.sendHaloFluidCount;
		sentHaloBoundaryCount += rankBuffers.sendHaloBoundaryCount;
	}

	// store aggregate traffic counts for diagnostics
	return
		CheckedUint32(
			sentOwnedFluidCount, redistStats_.sentOwnedFluidParticleCount,
			"Sent owned fluid particle count", failMsg) &&
		CheckedUint32(
			sentHaloFluidCount, redistStats_.sentHaloFluidParticleCount,
			"Sent halo fluid particle count", failMsg) &&
		CheckedUint32(
			sentHaloBoundaryCount, redistStats_.sentHaloBoundaryParticleCount,
			"Sent halo boundary particle count", failMsg);
}

bool ParticleRedistributionMgr::PrepareReceiveBuffers(
	const int rank, const int rankCount,
	std::string* failMsg)
{
	uint64_t recvOwnedFluidCount = 0;
	uint64_t recvHaloFluidCount = 0;
	uint64_t recvHaloBoundaryCount = 0;

	// configure one compact receive buffer per source rank
	for (int srcRank = 0; srcRank != rankCount; srcRank++)
	{
		auto& rankBuffers = rankBuffers_[srcRank];
		if (srcRank == rank) { continue; }

		const uint64_t countBase = static_cast<uint64_t>(srcRank) * PARTICLE_REDISTRIBUTION_LANE_COUNT;
		rankBuffers.recvOwnedFluidCount = recvCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_OWNED_FLUID];
		rankBuffers.recvHaloFluidCount = recvCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_HALO_FLUID];
		rankBuffers.recvHaloBoundaryCount = recvCounts_[
			countBase + PARTICLE_REDISTRIBUTION_LANE_HALO_BOUNDARY];

		if (rankBuffers.recvOwnedFluidCount < 0 ||
			rankBuffers.recvHaloFluidCount < 0 ||
			rankBuffers.recvHaloBoundaryCount < 0)
		{
			if (failMsg) { *failMsg = "Received particle count is negative."; }
			return false;
		}

		rankBuffers.recvOwnedFluidParticles.Resize(
			static_cast<uint32_t>(rankBuffers.recvOwnedFluidCount));
		rankBuffers.recvHaloFluidParticles.Resize(
			static_cast<uint32_t>(rankBuffers.recvHaloFluidCount));
		rankBuffers.recvHaloBoundaryParticles.Resize(
			static_cast<uint32_t>(rankBuffers.recvHaloBoundaryCount));

		recvOwnedFluidCount += rankBuffers.recvOwnedFluidCount;
		recvHaloFluidCount += rankBuffers.recvHaloFluidCount;
		recvHaloBoundaryCount += rankBuffers.recvHaloBoundaryCount;
	}

	// store aggregate traffic counts for diagnostics
	return
		CheckedUint32(
			recvOwnedFluidCount, redistStats_.recvOwnedFluidParticleCount,
			"Received owned fluid particle count", failMsg) &&
		CheckedUint32(
			recvHaloFluidCount, redistStats_.recvHaloFluidParticleCount,
			"Received halo fluid particle count", failMsg) &&
		CheckedUint32(
			recvHaloBoundaryCount, redistStats_.recvHaloBoundaryParticleCount,
			"Received halo boundary particle count", failMsg);
}

void ParticleRedistributionMgr::ExchangePayloads(
	const int rank, const int rankCount,
	const MpiContext& mpiContext)
{
	std::vector<MPI_Request> requests;

	// post receives before matching sends
	for (int srcRank = 0; srcRank != rankCount; srcRank++)
	{
		if (srcRank == rank) { continue; }

		auto& rankBuffers = rankBuffers_[srcRank];
		rankBuffers.recvOwnedFluidParticles.QueueIRecvPayload(
			mpiContext, srcRank, PARTICLE_REDISTRIBUTION_TAG_OWNED_FLUID,
			requests);
		rankBuffers.recvHaloFluidParticles.QueueIRecvPayload(
			mpiContext, srcRank, PARTICLE_REDISTRIBUTION_TAG_HALO_FLUID,
			requests);
		rankBuffers.recvHaloBoundaryParticles.QueueIRecvPayload(
			mpiContext, srcRank, PARTICLE_REDISTRIBUTION_TAG_HALO_BOUNDARY,
			requests);
	}

	// post sends for all three payload lanes
	for (int dstRank = 0; dstRank != rankCount; dstRank++)
	{
		if (dstRank == rank) { continue; }

		const auto& rankBuffers = rankBuffers_[dstRank];
		rankBuffers.sendOwnedFluidParticles.QueueISendPayload(
			mpiContext, dstRank, PARTICLE_REDISTRIBUTION_TAG_OWNED_FLUID,
			requests);
		rankBuffers.sendHaloFluidParticles.QueueISendPayload(
			mpiContext, dstRank, PARTICLE_REDISTRIBUTION_TAG_HALO_FLUID,
			requests);
		rankBuffers.sendHaloBoundaryParticles.QueueISendPayload(
			mpiContext, dstRank, PARTICLE_REDISTRIBUTION_TAG_HALO_BOUNDARY,
			requests);
	}

	mpiContext.WaitAll(requests);
}

bool ParticleRedistributionMgr::PrepareFluidParticleMutation(
	const Sync::FluidConstView& oldParticles,
	const DistParticleState& oldParticleState,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float kernelRadius,
	const uint32_t globalParticleCount,
	std::string* failMsg,
	ParticleRedistributionTimingReport* timingReport)
{
	auto& appendedParticles = appendedFluidParticles_;
	auto& rebuiltParticleState = rebuiltFluidParticleState_;
	uint32_t retainedParticleCount = 0;
	uint32_t appendedParticleCount = 0;
	uint32_t ownedParticleCount = 0;
	uint32_t haloParticleCount = 0;
	uint32_t localParticleCount = 0;

	// compute sizes and initialize append-only fluid storage and metadata
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::FluidPrep);

		const uint64_t retainedParticleCount_U64 =
			routes_.keptOwnedFluidLocalIdxs.size();
		const uint64_t ownedParticleCount_U64 =
			routes_.keptOwnedFluidLocalIdxs.size() +
			redistStats_.recvOwnedFluidParticleCount;
		const uint64_t haloParticleCount_U64 =
			routes_.localHaloFluidLocalIdxs.size() +
			redistStats_.recvHaloFluidParticleCount;
		const uint64_t appendedParticleCount_U64 =
			redistStats_.recvOwnedFluidParticleCount +
			haloParticleCount_U64;

		if (!CheckedUint32(
				retainedParticleCount_U64, retainedParticleCount,
				"Retained fluid particle count", failMsg) ||
			!CheckedUint32(
				appendedParticleCount_U64, appendedParticleCount,
				"Appended fluid particle count", failMsg) ||
			!CheckedUint32(
				ownedParticleCount_U64, ownedParticleCount,
				"Owned fluid particle count", failMsg) ||
			!CheckedUint32(
				haloParticleCount_U64, haloParticleCount,
				"Halo fluid particle count", failMsg) ||
			!CheckedUint32(
				ownedParticleCount_U64 + haloParticleCount_U64,
				localParticleCount,
				"Local fluid particle count", failMsg) ||
			!BuildDenseCompactionMap(
				oldParticles.N_,
				routes_.keptOwnedFluidLocalIdxs,
				retainedFluidOldLocalIdx_FromNewLocalIdx_,
				retainedOldLocalIdxFlags_,
				failMsg))
		{
			return false;
		}

		appendedParticles.resizeAll(appendedParticleCount);
		PrepareRebuiltParticleState(
			rebuiltParticleState,
			ownedParticleCount,
			haloParticleCount,
			localParticleCount,
			globalParticleCount);
	}

	int dstLocalIdx = 0;
	int appendLocalIdx = 0;

	// adds metadata for a particle retained in the compacted backend prefix
	const auto addRetainedParticleMetadata = [&](
		const int oldLocalIdx,
		const DistParticleClass particleClass)
	{
		const int globalIdx = oldParticleState.particleGlobalIdx_FromLocalIdx[oldLocalIdx];
		const bool added = AddParticleMetadata(
			rebuiltParticleState, particleClass, globalParticleCount, globalIdx, dstLocalIdx);
		dstLocalIdx++;
		return added;
	};

	// rebuild metadata for owned particles retained in backend storage
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::KeptFluid);

		// TODO: parallelize this with TBB?
		// TODO: use explicit SIMD, similar to sph_solver.cpp implementation?
		// TODO: look for HPC best practices in this use-case?
		for (const uint32_t oldLocalIdx : retainedFluidOldLocalIdx_FromNewLocalIdx_)
		{
			const glm::vec3 position = FluidPositionAtLocalIdx(oldParticles, static_cast<int>(oldLocalIdx));
			const DistParticleClass particleClass =
				domainDecomp.IsInnerOwnedParticle(rank, position, kernelRadius)
				? DistParticleClass::Inner
				: DistParticleClass::Outer;
			if (!addRetainedParticleMetadata(
				static_cast<int>(oldLocalIdx), particleClass))
			{
				if (failMsg) { *failMsg = "Duplicate or invalid owned fluid global index."; }
				return false;
			}
		}
	}

	// append owned particles migrated from other ranks
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::MigratedFluid);

		for (const auto& rankBuffers : rankBuffers_)
		{
			// TODO: parallelize this with TBB?
			// TODO: use explicit SIMD, similar to sph_solver.cpp implementation?
			// TODO: look for HPC best practices in this use-case?
			const auto& recvParticles = rankBuffers.recvOwnedFluidParticles;
			for (uint64_t bufferIdx = 0; bufferIdx != recvParticles.Size(); bufferIdx++)
			{
				recvParticles.CopyToLocalParticle( appendedParticles, bufferIdx, appendLocalIdx);
				const glm::vec3 position
				{
					appendedParticles.xPos[appendLocalIdx],
					appendedParticles.yPos[appendLocalIdx],
					appendedParticles.zPos[appendLocalIdx]
				};
				const DistParticleClass particleClass =
					domainDecomp.IsInnerOwnedParticle(rank, position, kernelRadius)
						? DistParticleClass::Inner
						: DistParticleClass::Outer;
				if (!AddParticleMetadata(
					rebuiltParticleState, particleClass, globalParticleCount,
					recvParticles.GlobalIdxAt(bufferIdx), dstLocalIdx))
				{
					if (failMsg) { *failMsg = "Duplicate or invalid migrated fluid global index."; }
					return false;
				}
				appendLocalIdx++;
				dstLocalIdx++;
			}
		}
	}

	// append local and received fluid halos
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::HaloFluid);

		// append halos created locally from particles migrating away
		// TODO: parallelize this with TBB?
		// TODO: use explicit SIMD, similar to sph_solver.cpp implementation?
		// TODO: look for HPC best practices in this use-case?
		for (const int oldLocalIdx : routes_.localHaloFluidLocalIdxs)
		{
			CopyFluidParticle(
				oldParticles, appendedParticles,
				oldLocalIdx, appendLocalIdx);
			if (!AddParticleMetadata(
				rebuiltParticleState, DistParticleClass::Halo,
				globalParticleCount,
				oldParticleState.particleGlobalIdx_FromLocalIdx[oldLocalIdx],
				dstLocalIdx))
			{
				if (failMsg) { *failMsg = "Duplicate or invalid local fluid halo global index."; }
				return false;
			}
			appendLocalIdx++;
			dstLocalIdx++;
		}

		// append halo particles received from other ranks
		for (const auto& rankBuffers : rankBuffers_)
		{
			// TODO: parallelize this with TBB?
			// TODO: use explicit SIMD, similar to sph_solver.cpp implementation?
			// TODO: look for HPC best practices in this use-case?
			const auto& recvParticles = rankBuffers.recvHaloFluidParticles;
			for (uint64_t bufferIdx = 0; bufferIdx != recvParticles.Size(); bufferIdx++)
			{
				recvParticles.CopyToLocalParticle(appendedParticles, bufferIdx, appendLocalIdx);
				if (!AddParticleMetadata(
					rebuiltParticleState, DistParticleClass::Halo,
					globalParticleCount, recvParticles.GlobalIdxAt(bufferIdx), dstLocalIdx))
				{
					if (failMsg) { *failMsg = "Duplicate or invalid received fluid halo global index."; }
					return false;
				}
				appendLocalIdx++;
				dstLocalIdx++;
			}
		}
	}

	// validate rebuilt fluid metadata
	// TODO: gate expensive validation partially/fully behind a debug build flag?
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::FluidPrep);
		if (static_cast<uint32_t>(appendLocalIdx) != appendedParticleCount ||
			static_cast<uint32_t>(dstLocalIdx) != localParticleCount ||
			retainedFluidOldLocalIdx_FromNewLocalIdx_.size() != retainedParticleCount)
		{
			if (failMsg) { *failMsg = "Prepared fluid particle mutation count is inconsistent."; }
			return false;
		}
		return rebuiltParticleState.IsValidLocalLayout(localParticleCount, failMsg);
	}
}

// prepare retained compaction, appended particles, and matching boundary metadata
bool ParticleRedistributionMgr::PrepareBoundaryParticleMutation(
	const Sync::BoundaryConstView& oldParticles,
	const DistParticleState& oldParticleState,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float kernelRadius,
	const uint32_t globalParticleCount,
	std::string* failMsg,
	ParticleRedistributionTimingReport* timingReport)
{
	auto& appendedParticles = appendedBoundaryParticles_;
	auto& rebuiltParticleState = rebuiltBoundaryParticleState_;
	uint32_t retainedParticleCount = 0;
	uint32_t appendedParticleCount = 0;
	uint32_t ownedParticleCount = 0;
	uint32_t haloParticleCount = 0;
	uint32_t localParticleCount = 0;

	// compute sizes and initialize append-only boundary storage and metadata
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::BoundaryPrep);

		const uint64_t retainedParticleCount_U64 =
			routes_.keptOwnedBoundaryLocalIdxs.size();
		const uint64_t haloParticleCount_U64 =
			redistStats_.recvHaloBoundaryParticleCount;
		if (!CheckedUint32(
				retainedParticleCount_U64, retainedParticleCount,
				"Retained boundary particle count", failMsg) ||
			!CheckedUint32(
				haloParticleCount_U64, appendedParticleCount,
				"Appended boundary particle count", failMsg) ||
			!CheckedUint32(
				retainedParticleCount_U64, ownedParticleCount,
				"Owned boundary particle count", failMsg) ||
			!CheckedUint32(
				haloParticleCount_U64, haloParticleCount,
				"Halo boundary particle count", failMsg) ||
			!CheckedUint32(
				retainedParticleCount_U64 + haloParticleCount_U64,
				localParticleCount,
				"Local boundary particle count", failMsg) ||
			!BuildDenseCompactionMap(
				oldParticles.N_,
				routes_.keptOwnedBoundaryLocalIdxs,
				retainedBoundaryOldLocalIdx_FromNewLocalIdx_,
				retainedOldLocalIdxFlags_,
				failMsg))
		{
			return false;
		}

		appendedParticles.resizeAll(appendedParticleCount);
		PrepareRebuiltParticleState(
			rebuiltParticleState,
			ownedParticleCount,
			haloParticleCount,
			localParticleCount,
			globalParticleCount);
	}

	int dstLocalIdx = 0;
	int appendLocalIdx = 0;

	// rebuild metadata for owned boundary particles retained in backend storage
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::KeptBoundary);

		// TODO: parallelize this with TBB?
		// TODO: use explicit SIMD, similar to sph_solver.cpp implementation?
		// TODO: look for HPC best practices in this use-case?
		for (const uint32_t oldLocalIdx : retainedBoundaryOldLocalIdx_FromNewLocalIdx_)
		{
			const glm::vec3 position = BoundaryPositionAtLocalIdx(oldParticles, static_cast<int>(oldLocalIdx));
			const DistParticleClass particleClass =
				domainDecomp.IsInnerOwnedParticle(rank, position, kernelRadius)
					? DistParticleClass::Inner
					: DistParticleClass::Outer;
			if (!AddParticleMetadata(
				rebuiltParticleState, particleClass, globalParticleCount,
				oldParticleState.particleGlobalIdx_FromLocalIdx[oldLocalIdx],
				dstLocalIdx))
			{
				if (failMsg) { *failMsg = "Duplicate or invalid owned boundary global index."; }
				return false;
			}
			dstLocalIdx++;
		}
	}

	// append boundary halo particles received from other ranks
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::HaloBoundary);

		for (const auto& rankBuffers : rankBuffers_)
		{
			// TODO: parallelize this with TBB?
			// TODO: use explicit SIMD, similar to sph_solver.cpp implementation?
			// TODO: look for HPC best practices in this use-case?
			const auto& recvParticles = rankBuffers.recvHaloBoundaryParticles;
			for (uint64_t bufferIdx = 0; bufferIdx != recvParticles.Size(); bufferIdx++)
			{
				recvParticles.CopyToLocalParticle(appendedParticles, bufferIdx, appendLocalIdx);
				if (!AddParticleMetadata(
					rebuiltParticleState, DistParticleClass::Halo,
					globalParticleCount, recvParticles.GlobalIdxAt(bufferIdx), dstLocalIdx))
				{
					if (failMsg) { *failMsg = "Duplicate or invalid received boundary halo global index."; }
					return false;
				}
				appendLocalIdx++;
				dstLocalIdx++;
			}
		}
	}

	// validate rebuilt boundary metadata
	// TODO: gate expensive validation partially/fully behind a debug build flag?
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::BoundaryPrep);
		if (static_cast<uint32_t>(appendLocalIdx) != appendedParticleCount ||
			static_cast<uint32_t>(dstLocalIdx) != localParticleCount ||
			retainedBoundaryOldLocalIdx_FromNewLocalIdx_.size() != retainedParticleCount)
		{
			if (failMsg) { *failMsg = "Prepared boundary particle mutation count is inconsistent."; }
			return false;
		}
		return rebuiltParticleState.IsValidLocalLayout(localParticleCount, failMsg);
	}
}

bool ParticleRedistributionMgr::BuildExchangeRoutes(
	const Sync::FluidConstView& fluidParticles,
	const Sync::BoundaryConstView& boundaryParticles,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const int rankCount,
	std::string* failMsg,
	ParticleRedistributionTimingReport* timingReport)
{
	// validate the rank and backend views before replacing reusable routes
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::RoutePrep);

		if (failMsg) { failMsg->clear(); }
		if (rank < 0 || rank >= rankCount)
		{
			if (failMsg) { *failMsg = "Particle redistribution rank is out of range."; }
			return false;
		}
		if (!ValidateParticleViews(
			fluidParticles, boundaryParticles,
			fluidParticleState, boundaryParticleState, failMsg))
		{
			return false;
		}

		PrepareBuffers(rankCount);
	}

	return RouteOwnedParticles(
		fluidParticles, boundaryParticles,
		fluidParticleState, boundaryParticleState,
		domainDecomp, rank, rankCount, failMsg, timingReport);
}

bool ParticleRedistributionMgr::RedistributeParticles(
	Sim::I_DistributedParticleBackend& distParticleBackend,
	DistParticleState& fluidParticleState,
	DistParticleState& boundaryParticleState,
	const IDomainDecomp& domainDecomp,
	const MpiContext& mpiContext,
	const Sim::SimParams& simParams,
	const uint32_t globalFluidParticleCount,
	const uint32_t globalBoundaryParticleCount,
	std::string* failMsg,
	ParticleRedistributionTimingReport* timingReport)
{
	if (failMsg) { failMsg->clear(); }

	// reset
	ResetRedistStats();
	const int rank = mpiContext.Rank();
	const int rankCount = mpiContext.Size();
	Sync::FluidConstView fluidParticles{};
	Sync::BoundaryConstView boundaryParticles{};
	bool localOk = true;

	// time explicit agreement between ranks independently of local work
	const auto AllRanksTrue = [&](
		const bool value,
		const ParticleRedistributionTimingOp timingOp)
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, timingOp);
		return mpiContext.AllTrue(value);
	};

	// build exchange routes directly from backend particle views
	try
	{
		{
			[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
				timingReport, ParticleRedistributionTimingOp::RoutePrep);
			fluidParticles = distParticleBackend.GetFluidParticleView();
			boundaryParticles = distParticleBackend.GetBoundaryParticleView();
		}

		localOk = BuildExchangeRoutes(
			fluidParticles, boundaryParticles,
			fluidParticleState, boundaryParticleState,
			domainDecomp, rank, rankCount, failMsg, timingReport);

		// pack outgoing migration and halo payloads
		if (localOk)
		{
			[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
				timingReport, ParticleRedistributionTimingOp::Pack);
			localOk = PackSendBuffers(
				fluidParticles, boundaryParticles,
				fluidParticleState, boundaryParticleState,
				rank, rankCount, failMsg);
		}
	}
	catch (const std::exception& ex)
	{
		if (failMsg) { *failMsg = ex.what(); }
		localOk = false;
	}

	// no rank may enter the count exchange while another rank failed validation
	if (!AllRanksTrue(
		localOk,
		ParticleRedistributionTimingOp::SyncPack)) { return false; }

	// exchange migration and halo particle counts and resize all receive buffers accordingly
	try
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::Exchange);
		mpiContext.AllToAllInts(
			sendCounts_, recvCounts_,
			PARTICLE_REDISTRIBUTION_LANE_COUNT);
		localOk = PrepareReceiveBuffers(rank, rankCount, failMsg);
	}
	catch (const std::exception& ex)
	{
		if (failMsg) { *failMsg = ex.what(); }
		localOk = false;
	}
	if (!AllRanksTrue(
		localOk,
		ParticleRedistributionTimingOp::SyncCounts)) { return false; }

	// transfer migration and halo payloads together
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::Exchange);
		ExchangePayloads(rank, rankCount, mpiContext);
	}

	// prepare construction of new fluid state in the backend data structures
	// (later in-place compaction of retained particles, appending of migration + halo particles)
	try
	{
		localOk = PrepareFluidParticleMutation(
			fluidParticles, fluidParticleState,
			domainDecomp, rank, simParams.kernelRadius,
			globalFluidParticleCount, failMsg, timingReport);
		if (localOk)
		{
			localOk = PrepareBoundaryParticleMutation(
				boundaryParticles, boundaryParticleState,
				domainDecomp, rank, simParams.kernelRadius,
				globalBoundaryParticleCount, failMsg, timingReport);
		}
	}
	catch (const std::exception& ex)
	{
		if (failMsg) { *failMsg = ex.what(); }
		localOk = false;
	}
	if (!AllRanksTrue(
		localOk,
		ParticleRedistributionTimingOp::SyncRebuild)) { return false; }

	// compact retained particles and append received migration +  halos particles
	try
	{
		[[maybe_unused]] const ParticleRedistributionTimingScope timingScope(
			timingReport, ParticleRedistributionTimingOp::Backend);
		localOk = distParticleBackend.ApplyParticleRedistribution(
			retainedFluidOldLocalIdx_FromNewLocalIdx_,
			retainedBoundaryOldLocalIdx_FromNewLocalIdx_,
			appendedFluidParticles_,
			appendedBoundaryParticles_,
			simParams);
		if (!localOk && failMsg)
		{
			*failMsg = "Backend rejected particle redistribution mutation.";
		}
	}
	catch (const std::exception& ex)
	{
		if (failMsg) { *failMsg = ex.what(); }
		localOk = false;
	}
	if (!AllRanksTrue(
		localOk,
		ParticleRedistributionTimingOp::SyncBackend)) { return false; }

	// publish matching metadata only after every backend mutation succeeded
	std::swap(fluidParticleState, rebuiltFluidParticleState_);
	std::swap(boundaryParticleState, rebuiltBoundaryParticleState_);

	return true;
}
