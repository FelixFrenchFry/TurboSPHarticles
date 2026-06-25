// halo_exchange.cpp

// external
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

// internal
#include "halo_exchange.h"
#include "mpi_context.h"



using namespace Dist;

namespace
{
	constexpr int HALO_EXCHANGE_TAG_FLUID_COUNT = 4101;
	constexpr int HALO_EXCHANGE_TAG_BOUNDARY_COUNT = 4102;
	constexpr int HALO_EXCHANGE_TAG_FLUID_PAYLOAD = 4110;
	constexpr int HALO_EXCHANGE_TAG_BOUNDARY_PAYLOAD = 4210;

	template <typename T>
	uint64_t CheckedByteSize(const std::vector<T>& values)
	{
		const uint64_t valueCount = values.size();
		const uint64_t maxCount =
			std::numeric_limits<uint64_t>::max() / sizeof(T);

		if (valueCount > maxCount)
		{
			throw std::runtime_error("halo exchange payload byte size exceeds uint64_t range.");
		}

		return valueCount * sizeof(T);
	}

	template <typename T>
	uint64_t UsedByteSize(const std::vector<T>& values)
	{
		return static_cast<uint64_t>(values.size()) * sizeof(T);
	}

	template <typename T>
	uint64_t AllocByteSize(const std::vector<T>& values)
	{
		return static_cast<uint64_t>(values.capacity()) * sizeof(T);
	}

	template <typename T>
	const void* DataOrNull(const std::vector<T>& values)
	{
		return values.empty() ? nullptr : values.data();
	}

	template <typename T>
	void* DataOrNull(std::vector<T>& values)
	{
		return values.empty() ? nullptr : values.data();
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
			DataOrNull(values),
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
			DataOrNull(values),
			CheckedByteSize(values),
			peerRank,
			tag,
			requests);
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

	glm::vec3 BoundaryPositionAtLocalIdx(
		const Sync::BoundaryParticles& boundaryParticles,
		const int localIdx)
	{
		return
		{
			boundaryParticles.xPos[localIdx],
			boundaryParticles.yPos[localIdx],
			boundaryParticles.zPos[localIdx]
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

	void CopyBoundaryParticle_LocalToLocal(
		const Sync::BoundaryParticles& src,
		Sync::BoundaryParticles& dst,
		const int srcLocalIdx,
		const int dstLocalIdx)
	{
		dst.xPos[dstLocalIdx] = src.xPos[srcLocalIdx];
		dst.yPos[dstLocalIdx] = src.yPos[srcLocalIdx];
		dst.zPos[dstLocalIdx] = src.zPos[srcLocalIdx];
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

	bool ClearLocalMapEntryIfStillMapped(
		DistParticleState& particleState,
		const int localIdx,
		const uint32_t globalParticleCount)
	{
		if (localIdx < 0 ||
			static_cast<uint64_t>(localIdx) >= particleState.particleGlobalIdx_FromLocalIdx.size())
		{
			return false;
		}

		const int globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
		if (!IsGlobalIdxInRange(globalIdx, globalParticleCount)) { return false; }
		if (static_cast<uint64_t>(globalIdx) >= particleState.particleLocalIdx_FromGlobalIdx.size())
		{
			return false;
		}

		if (particleState.particleLocalIdx_FromGlobalIdx[globalIdx] == localIdx)
		{
			particleState.particleLocalIdx_FromGlobalIdx[globalIdx] = -1;
		}

		return true;
	}

	bool OwnedParticlesAreCompactPrefix(const DistParticleState& particleState)
	{
		if (static_cast<uint64_t>(particleState.ownedParticleCount) !=
			particleState.particleLocalIdx_Owned.size())
		{
			return false;
		}

		for (uint32_t localIdx = 0;
			localIdx < particleState.ownedParticleCount;
			localIdx++)
		{
			if (particleState.particleLocalIdx_Owned[localIdx] !=
				static_cast<int>(localIdx))
			{
				return false;
			}
		}

		return true;
	}

	template <typename FailFn>
	bool PrepareHaloExchangeRankBuffers(
		HaloExchangeWorkBuffers& workBuffers,
		const std::vector<int>& neighborRanks,
		const int rankCount,
		FailFn fail)
	{
		workBuffers.rankBuffers.resize(neighborRanks.size());
		workBuffers.rankBufferIdx_FromRank.assign(
			static_cast<uint64_t>(std::max(0, rankCount)), -1);

		for (uint64_t neighborIdx = 0; neighborIdx < neighborRanks.size(); neighborIdx++)
		{
			const int neighborRank = neighborRanks[neighborIdx];
			if (neighborRank < 0 || neighborRank >= rankCount)
			{
				return fail("halo exchange neighbor rank is outside the MPI world.");
			}

			HaloExchangeWorkBuffers::RankBuffers& exchange =
				workBuffers.rankBuffers[neighborIdx];
			exchange.rank = neighborRank;
			exchange.sendFluidCount = 0;
			exchange.recvFluidCount = 0;
			exchange.sendBoundaryCount = 0;
			exchange.recvBoundaryCount = 0;
			exchange.sendBuffer.Clear();
			exchange.recvBuffer.Clear();

			workBuffers.rankBufferIdx_FromRank[static_cast<uint64_t>(neighborRank)] =
				static_cast<int>(neighborIdx);
		}

		return true;
	}

	uint64_t CountReceivedFluidHalos(
		const std::vector<HaloExchangeWorkBuffers::RankBuffers>& exchanges)
	{
		uint64_t count = 0;
		for (const HaloExchangeWorkBuffers::RankBuffers& exchange : exchanges)
		{
			count += exchange.recvBuffer.fluidParticles.Size();
		}
		return count;
	}

	uint64_t CountReceivedBoundaryHalos(
		const std::vector<HaloExchangeWorkBuffers::RankBuffers>& exchanges)
	{
		uint64_t count = 0;
		for (const HaloExchangeWorkBuffers::RankBuffers& exchange : exchanges)
		{
			count += exchange.recvBuffer.boundaryParticles.Size();
		}
		return count;
	}
}

void HaloParticleBufferF::Clear()
{
	globalIdx_.clear();
	xPos_.clear();
	yPos_.clear();
	zPos_.clear();
	xVel_.clear();
	yVel_.clear();
	zVel_.clear();
	xVelPred_.clear();
	yVelPred_.clear();
	zVelPred_.clear();
	xAccPress_.clear();
	yAccPress_.clear();
	zAccPress_.clear();
	sourceTerm_.clear();
	diagElem_.clear();
	pressure_.clear();
	density_.clear();
}

void HaloParticleBufferF::Resize(const uint64_t particleCount)
{
	globalIdx_.resize(particleCount);
	xPos_.resize(particleCount);
	yPos_.resize(particleCount);
	zPos_.resize(particleCount);
	xVel_.resize(particleCount);
	yVel_.resize(particleCount);
	zVel_.resize(particleCount);
	xVelPred_.resize(particleCount);
	yVelPred_.resize(particleCount);
	zVelPred_.resize(particleCount);
	xAccPress_.resize(particleCount);
	yAccPress_.resize(particleCount);
	zAccPress_.resize(particleCount);
	sourceTerm_.resize(particleCount);
	diagElem_.resize(particleCount);
	pressure_.resize(particleCount);
	density_.resize(particleCount);
}

int HaloParticleBufferF::GlobalIdxAt(const uint64_t bufferIdx) const
{
	return globalIdx_[bufferIdx];
}

void HaloParticleBufferF::SetFromLocalParticle(
	const Sync::FluidParticles& fluidParticles,
	const DistParticleState& particleState,
	const uint64_t bufferIdx,
	const int localIdx)
{
	globalIdx_[bufferIdx] = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
	xPos_[bufferIdx] = fluidParticles.xPos[localIdx];
	yPos_[bufferIdx] = fluidParticles.yPos[localIdx];
	zPos_[bufferIdx] = fluidParticles.zPos[localIdx];
	xVel_[bufferIdx] = fluidParticles.xVel[localIdx];
	yVel_[bufferIdx] = fluidParticles.yVel[localIdx];
	zVel_[bufferIdx] = fluidParticles.zVel[localIdx];
	xVelPred_[bufferIdx] = fluidParticles.xVelPred[localIdx];
	yVelPred_[bufferIdx] = fluidParticles.yVelPred[localIdx];
	zVelPred_[bufferIdx] = fluidParticles.zVelPred[localIdx];
	xAccPress_[bufferIdx] = fluidParticles.xAccPress[localIdx];
	yAccPress_[bufferIdx] = fluidParticles.yAccPress[localIdx];
	zAccPress_[bufferIdx] = fluidParticles.zAccPress[localIdx];
	sourceTerm_[bufferIdx] = fluidParticles.sourceTerm[localIdx];
	diagElem_[bufferIdx] = fluidParticles.diagElem[localIdx];
	pressure_[bufferIdx] = fluidParticles.pressure[localIdx];
	density_[bufferIdx] = fluidParticles.density[localIdx];
}

void HaloParticleBufferF::CopyToLocalParticle(
	Sync::FluidParticles& dst,
	const uint64_t bufferIdx,
	const int localIdx) const
{
	dst.xPos[localIdx] = xPos_[bufferIdx];
	dst.yPos[localIdx] = yPos_[bufferIdx];
	dst.zPos[localIdx] = zPos_[bufferIdx];
	dst.xVel[localIdx] = xVel_[bufferIdx];
	dst.yVel[localIdx] = yVel_[bufferIdx];
	dst.zVel[localIdx] = zVel_[bufferIdx];
	dst.xVelPred[localIdx] = xVelPred_[bufferIdx];
	dst.yVelPred[localIdx] = yVelPred_[bufferIdx];
	dst.zVelPred[localIdx] = zVelPred_[bufferIdx];
	dst.xAccPress[localIdx] = xAccPress_[bufferIdx];
	dst.yAccPress[localIdx] = yAccPress_[bufferIdx];
	dst.zAccPress[localIdx] = zAccPress_[bufferIdx];
	dst.sourceTerm[localIdx] = sourceTerm_[bufferIdx];
	dst.diagElem[localIdx] = diagElem_[bufferIdx];
	dst.pressure[localIdx] = pressure_[bufferIdx];
	dst.density[localIdx] = density_[bufferIdx];
}

void HaloParticleBufferF::QueueIRecvPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests)
{
	if (Empty()) { return; }

	QueueIRecvVector(mpiContext, globalIdx_, peerRank, tagBase + 1, requests);
	QueueIRecvVector(mpiContext, xPos_, peerRank, tagBase + 2, requests);
	QueueIRecvVector(mpiContext, yPos_, peerRank, tagBase + 3, requests);
	QueueIRecvVector(mpiContext, zPos_, peerRank, tagBase + 4, requests);
	QueueIRecvVector(mpiContext, xVel_, peerRank, tagBase + 5, requests);
	QueueIRecvVector(mpiContext, yVel_, peerRank, tagBase + 6, requests);
	QueueIRecvVector(mpiContext, zVel_, peerRank, tagBase + 7, requests);
	QueueIRecvVector(mpiContext, xVelPred_, peerRank, tagBase + 8, requests);
	QueueIRecvVector(mpiContext, yVelPred_, peerRank, tagBase + 9, requests);
	QueueIRecvVector(mpiContext, zVelPred_, peerRank, tagBase + 10, requests);
	QueueIRecvVector(mpiContext, xAccPress_, peerRank, tagBase + 11, requests);
	QueueIRecvVector(mpiContext, yAccPress_, peerRank, tagBase + 12, requests);
	QueueIRecvVector(mpiContext, zAccPress_, peerRank, tagBase + 13, requests);
	QueueIRecvVector(mpiContext, sourceTerm_, peerRank, tagBase + 14, requests);
	QueueIRecvVector(mpiContext, diagElem_, peerRank, tagBase + 15, requests);
	QueueIRecvVector(mpiContext, pressure_, peerRank, tagBase + 16, requests);
	QueueIRecvVector(mpiContext, density_, peerRank, tagBase + 17, requests);
}

void HaloParticleBufferF::QueueISendPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests) const
{
	if (Empty()) { return; }

	QueueISendVector(mpiContext, globalIdx_, peerRank, tagBase + 1, requests);
	QueueISendVector(mpiContext, xPos_, peerRank, tagBase + 2, requests);
	QueueISendVector(mpiContext, yPos_, peerRank, tagBase + 3, requests);
	QueueISendVector(mpiContext, zPos_, peerRank, tagBase + 4, requests);
	QueueISendVector(mpiContext, xVel_, peerRank, tagBase + 5, requests);
	QueueISendVector(mpiContext, yVel_, peerRank, tagBase + 6, requests);
	QueueISendVector(mpiContext, zVel_, peerRank, tagBase + 7, requests);
	QueueISendVector(mpiContext, xVelPred_, peerRank, tagBase + 8, requests);
	QueueISendVector(mpiContext, yVelPred_, peerRank, tagBase + 9, requests);
	QueueISendVector(mpiContext, zVelPred_, peerRank, tagBase + 10, requests);
	QueueISendVector(mpiContext, xAccPress_, peerRank, tagBase + 11, requests);
	QueueISendVector(mpiContext, yAccPress_, peerRank, tagBase + 12, requests);
	QueueISendVector(mpiContext, zAccPress_, peerRank, tagBase + 13, requests);
	QueueISendVector(mpiContext, sourceTerm_, peerRank, tagBase + 14, requests);
	QueueISendVector(mpiContext, diagElem_, peerRank, tagBase + 15, requests);
	QueueISendVector(mpiContext, pressure_, peerRank, tagBase + 16, requests);
	QueueISendVector(mpiContext, density_, peerRank, tagBase + 17, requests);
}

int HaloParticleBufferF::CheckedBufferCount() const
{
	if (Size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
	{
		throw std::runtime_error("halo fluid buffer count exceeds MPI int count range.");
	}

	return static_cast<int>(Size());
}

uint64_t HaloParticleBufferF::UsedBytes() const
{
	uint64_t usedBytes = 0;

	usedBytes += UsedByteSize(globalIdx_);
	usedBytes += UsedByteSize(xPos_);
	usedBytes += UsedByteSize(yPos_);
	usedBytes += UsedByteSize(zPos_);
	usedBytes += UsedByteSize(xVel_);
	usedBytes += UsedByteSize(yVel_);
	usedBytes += UsedByteSize(zVel_);
	usedBytes += UsedByteSize(xVelPred_);
	usedBytes += UsedByteSize(yVelPred_);
	usedBytes += UsedByteSize(zVelPred_);
	usedBytes += UsedByteSize(xAccPress_);
	usedBytes += UsedByteSize(yAccPress_);
	usedBytes += UsedByteSize(zAccPress_);
	usedBytes += UsedByteSize(sourceTerm_);
	usedBytes += UsedByteSize(diagElem_);
	usedBytes += UsedByteSize(pressure_);
	usedBytes += UsedByteSize(density_);

	return usedBytes;
}

uint64_t HaloParticleBufferF::AllocBytes() const
{
	uint64_t allocBytes = 0;

	allocBytes += AllocByteSize(globalIdx_);
	allocBytes += AllocByteSize(xPos_);
	allocBytes += AllocByteSize(yPos_);
	allocBytes += AllocByteSize(zPos_);
	allocBytes += AllocByteSize(xVel_);
	allocBytes += AllocByteSize(yVel_);
	allocBytes += AllocByteSize(zVel_);
	allocBytes += AllocByteSize(xVelPred_);
	allocBytes += AllocByteSize(yVelPred_);
	allocBytes += AllocByteSize(zVelPred_);
	allocBytes += AllocByteSize(xAccPress_);
	allocBytes += AllocByteSize(yAccPress_);
	allocBytes += AllocByteSize(zAccPress_);
	allocBytes += AllocByteSize(sourceTerm_);
	allocBytes += AllocByteSize(diagElem_);
	allocBytes += AllocByteSize(pressure_);
	allocBytes += AllocByteSize(density_);

	return allocBytes;
}

void HaloParticleBufferB::Clear()
{
	globalIdx_.clear();
	xPos_.clear();
	yPos_.clear();
	zPos_.clear();
}

void HaloParticleBufferB::Resize(const uint64_t particleCount)
{
	globalIdx_.resize(particleCount);
	xPos_.resize(particleCount);
	yPos_.resize(particleCount);
	zPos_.resize(particleCount);
}

int HaloParticleBufferB::GlobalIdxAt(const uint64_t bufferIdx) const
{
	return globalIdx_[bufferIdx];
}

void HaloParticleBufferB::SetFromLocalParticle(
	const Sync::BoundaryParticles& boundaryParticles,
	const DistParticleState& particleState,
	const uint64_t bufferIdx,
	const int localIdx)
{
	globalIdx_[bufferIdx] = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
	xPos_[bufferIdx] = boundaryParticles.xPos[localIdx];
	yPos_[bufferIdx] = boundaryParticles.yPos[localIdx];
	zPos_[bufferIdx] = boundaryParticles.zPos[localIdx];
}

void HaloParticleBufferB::CopyToLocalParticle(
	Sync::BoundaryParticles& dst,
	const uint64_t bufferIdx,
	const int localIdx) const
{
	dst.xPos[localIdx] = xPos_[bufferIdx];
	dst.yPos[localIdx] = yPos_[bufferIdx];
	dst.zPos[localIdx] = zPos_[bufferIdx];
}

void HaloParticleBufferB::QueueIRecvPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests)
{
	if (Empty()) { return; }

	QueueIRecvVector(mpiContext, globalIdx_, peerRank, tagBase + 1, requests);
	QueueIRecvVector(mpiContext, xPos_, peerRank, tagBase + 2, requests);
	QueueIRecvVector(mpiContext, yPos_, peerRank, tagBase + 3, requests);
	QueueIRecvVector(mpiContext, zPos_, peerRank, tagBase + 4, requests);
}

void HaloParticleBufferB::QueueISendPayload(
	const MpiContext& mpiContext,
	const int peerRank,
	const int tagBase,
	std::vector<MPI_Request>& requests) const
{
	if (Empty()) { return; }

	QueueISendVector(mpiContext, globalIdx_, peerRank, tagBase + 1, requests);
	QueueISendVector(mpiContext, xPos_, peerRank, tagBase + 2, requests);
	QueueISendVector(mpiContext, yPos_, peerRank, tagBase + 3, requests);
	QueueISendVector(mpiContext, zPos_, peerRank, tagBase + 4, requests);
}

int HaloParticleBufferB::CheckedBufferCount() const
{
	if (Size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
	{
		throw std::runtime_error("halo boundary buffer count exceeds MPI int count range.");
	}

	return static_cast<int>(Size());
}

uint64_t HaloParticleBufferB::UsedBytes() const
{
	uint64_t usedBytes = 0;

	usedBytes += UsedByteSize(globalIdx_);
	usedBytes += UsedByteSize(xPos_);
	usedBytes += UsedByteSize(yPos_);
	usedBytes += UsedByteSize(zPos_);

	return usedBytes;
}

uint64_t HaloParticleBufferB::AllocBytes() const
{
	uint64_t allocBytes = 0;

	allocBytes += AllocByteSize(globalIdx_);
	allocBytes += AllocByteSize(xPos_);
	allocBytes += AllocByteSize(yPos_);
	allocBytes += AllocByteSize(zPos_);

	return allocBytes;
}

namespace
{
	template <typename FailFn>
	bool RebuildFluidParticlesWithReceivedHalos_Copy(
		Sync::UniversalSimState& rebuiltState,
		DistParticleState& rebuiltParticleState,
		const Sync::UniversalSimState& srcState,
		const DistParticleState& srcParticleState,
		const std::vector<HaloExchangeWorkBuffers::RankBuffers>& exchanges,
		const uint32_t globalParticleCount,
		FailFn fail)
	{
		// compute new local particle counts
		const uint32_t ownedParticleCount =
			static_cast<uint32_t>(srcParticleState.particleLocalIdx_Owned.size());
		const uint64_t recvHaloCount = CountReceivedFluidHalos(exchanges);
		if (recvHaloCount >
			static_cast<uint64_t>(std::numeric_limits<uint32_t>::max() - ownedParticleCount))
		{
			return fail("received fluid halo count exceeds uint32_t local particle count range.");
		}

		const uint32_t haloParticleCount = static_cast<uint32_t>(recvHaloCount);
		const uint32_t localParticleCount = ownedParticleCount + haloParticleCount;

		// resize rebuilt state and initialize metadata structures
		rebuiltState.ResizeFluid(localParticleCount);
		rebuiltParticleState.Clear();
		rebuiltParticleState.ownedParticleCount = ownedParticleCount;
		rebuiltParticleState.haloParticleCount = haloParticleCount;
		rebuiltParticleState.particleClasses.resize(localParticleCount, DistParticleClass::None);
		rebuiltParticleState.particleLocalIdx_FromGlobalIdx.assign(globalParticleCount, -1);
		rebuiltParticleState.particleGlobalIdx_FromLocalIdx.assign(localParticleCount, -1);
		rebuiltParticleState.particleLocalIdx_Owned.reserve(ownedParticleCount);
		rebuiltParticleState.particleLocalIdx_Inner.reserve(srcParticleState.particleLocalIdx_Inner.size());
		rebuiltParticleState.particleLocalIdx_Outer.reserve(srcParticleState.particleLocalIdx_Outer.size());
		rebuiltParticleState.particleLocalIdx_Halo.reserve(haloParticleCount);

		int dstLocalIdx = 0;
		for (const int srcLocalIdx : srcParticleState.particleLocalIdx_Owned)
		{
			if (!IsLocalIdxInRange(srcLocalIdx, srcState.fluidParticles.N_))
			{
				return fail("fluid owned local index is outside the local fluid particle range.");
			}
			if (!HasParticleMetadataAtLocalIdx(srcParticleState, srcLocalIdx))
			{
				return fail("fluid owned local index has no particle metadata.");
			}

			const DistParticleClass particleClass = srcParticleState.particleClasses[srcLocalIdx];
			if (particleClass != DistParticleClass::Inner &&
				particleClass != DistParticleClass::Outer)
			{
				return fail("fluid owned local index class is not Inner or Outer.");
			}

			const int globalIdx = srcParticleState.particleGlobalIdx_FromLocalIdx[srcLocalIdx];
			if (!AddParticleMapEntry(
				rebuiltParticleState, globalIdx, dstLocalIdx, globalParticleCount))
			{
				return fail("failed to add rebuilt owned fluid particle map entry.");
			}

			CopyFluidParticle_LocalToLocal(
				srcState.fluidParticles, rebuiltState.fluidParticles,
				srcLocalIdx, dstLocalIdx);

			rebuiltParticleState.particleClasses[dstLocalIdx] = particleClass;
			rebuiltParticleState.particleLocalIdx_Owned.push_back(dstLocalIdx);
			if (particleClass == DistParticleClass::Inner)
			{
				rebuiltParticleState.particleLocalIdx_Inner.push_back(dstLocalIdx);
			}
			else
			{
				rebuiltParticleState.particleLocalIdx_Outer.push_back(dstLocalIdx);
			}

			dstLocalIdx++;
		}

		for (const HaloExchangeWorkBuffers::RankBuffers& exchange : exchanges)
		{
			const HaloParticleBufferF& recvFluidParticles =
				exchange.recvBuffer.fluidParticles;

			for (uint64_t bufferIdx = 0; bufferIdx != recvFluidParticles.Size(); bufferIdx++)
			{
				if (!AddParticleMapEntry(
					rebuiltParticleState, recvFluidParticles.GlobalIdxAt(bufferIdx),
					dstLocalIdx, globalParticleCount))
				{
					return fail("failed to add rebuilt fluid halo particle map entry.");
				}

				recvFluidParticles.CopyToLocalParticle(
					rebuiltState.fluidParticles, bufferIdx, dstLocalIdx);

				rebuiltParticleState.particleClasses[dstLocalIdx] = DistParticleClass::Halo;
				rebuiltParticleState.particleLocalIdx_Halo.push_back(dstLocalIdx);
				dstLocalIdx++;
			}
		}

	#if defined(_DEBUG)
		if (!rebuiltParticleState.IsValidLocalLayout(localParticleCount))
		{
			return fail("rebuilt fluid halo layout is invalid.");
		}
	#endif

		return true;
	}

	template <typename FailFn>
	bool RebuildBoundaryParticlesWithReceivedHalos_Copy(
		Sync::UniversalSimState& rebuiltState,
		DistParticleState& rebuiltParticleState,
		const Sync::UniversalSimState& srcState,
		const DistParticleState& srcParticleState,
		const std::vector<HaloExchangeWorkBuffers::RankBuffers>& exchanges,
		const uint32_t globalParticleCount,
		FailFn fail)
	{
		// compute new local particle counts
		const uint32_t ownedParticleCount =
			static_cast<uint32_t>(srcParticleState.particleLocalIdx_Owned.size());
		const uint64_t recvHaloCount = CountReceivedBoundaryHalos(exchanges);
		if (recvHaloCount >
			static_cast<uint64_t>(std::numeric_limits<uint32_t>::max() - ownedParticleCount))
		{
			return fail("received boundary halo count exceeds uint32_t local particle count range.");
		}

		const uint32_t haloParticleCount = static_cast<uint32_t>(recvHaloCount);
		const uint32_t localParticleCount = ownedParticleCount + haloParticleCount;

		// resize rebuilt state and initialize metadata structures
		rebuiltState.ResizeBoundary(localParticleCount);
		rebuiltParticleState.Clear();
		rebuiltParticleState.ownedParticleCount = ownedParticleCount;
		rebuiltParticleState.haloParticleCount = haloParticleCount;
		rebuiltParticleState.particleClasses.resize(localParticleCount, DistParticleClass::None);
		rebuiltParticleState.particleLocalIdx_FromGlobalIdx.assign(globalParticleCount, -1);
		rebuiltParticleState.particleGlobalIdx_FromLocalIdx.assign(localParticleCount, -1);
		rebuiltParticleState.particleLocalIdx_Owned.reserve(ownedParticleCount);
		rebuiltParticleState.particleLocalIdx_Inner.reserve(srcParticleState.particleLocalIdx_Inner.size());
		rebuiltParticleState.particleLocalIdx_Outer.reserve(srcParticleState.particleLocalIdx_Outer.size());
		rebuiltParticleState.particleLocalIdx_Halo.reserve(haloParticleCount);

		int dstLocalIdx = 0;
		for (const int srcLocalIdx : srcParticleState.particleLocalIdx_Owned)
		{
			if (!IsLocalIdxInRange(srcLocalIdx, srcState.boundaryParticles.N_))
			{
				return fail("boundary owned local index is outside the local boundary particle range.");
			}
			if (!HasParticleMetadataAtLocalIdx(srcParticleState, srcLocalIdx))
			{
				return fail("boundary owned local index has no particle metadata.");
			}

			const DistParticleClass particleClass = srcParticleState.particleClasses[srcLocalIdx];
			if (particleClass != DistParticleClass::Inner &&
				particleClass != DistParticleClass::Outer)
			{
				return fail("boundary owned local index class is not Inner or Outer.");
			}

			const int globalIdx = srcParticleState.particleGlobalIdx_FromLocalIdx[srcLocalIdx];
			if (!AddParticleMapEntry(
				rebuiltParticleState, globalIdx, dstLocalIdx, globalParticleCount))
			{
				return fail("failed to add rebuilt owned boundary particle map entry.");
			}

			CopyBoundaryParticle_LocalToLocal(
				srcState.boundaryParticles, rebuiltState.boundaryParticles,
				srcLocalIdx, dstLocalIdx);

			rebuiltParticleState.particleClasses[dstLocalIdx] = particleClass;
			rebuiltParticleState.particleLocalIdx_Owned.push_back(dstLocalIdx);
			if (particleClass == DistParticleClass::Inner)
			{
				rebuiltParticleState.particleLocalIdx_Inner.push_back(dstLocalIdx);
			}
			else
			{
				rebuiltParticleState.particleLocalIdx_Outer.push_back(dstLocalIdx);
			}

			dstLocalIdx++;
		}

		for (const HaloExchangeWorkBuffers::RankBuffers& exchange : exchanges)
		{
			const HaloParticleBufferB& recvBoundaryParticles =
				exchange.recvBuffer.boundaryParticles;

			for (uint64_t bufferIdx = 0; bufferIdx != recvBoundaryParticles.Size(); bufferIdx++)
			{
				if (!AddParticleMapEntry(
					rebuiltParticleState, recvBoundaryParticles.GlobalIdxAt(bufferIdx),
					dstLocalIdx, globalParticleCount))
				{
					return fail("failed to add rebuilt boundary halo particle map entry.");
				}

				recvBoundaryParticles.CopyToLocalParticle(
					rebuiltState.boundaryParticles, bufferIdx, dstLocalIdx);

				rebuiltParticleState.particleClasses[dstLocalIdx] = DistParticleClass::Halo;
				rebuiltParticleState.particleLocalIdx_Halo.push_back(dstLocalIdx);
				dstLocalIdx++;
			}
		}

	#if defined(_DEBUG)
		if (!rebuiltParticleState.IsValidLocalLayout(localParticleCount))
		{
			return fail("rebuilt boundary halo layout is invalid.");
		}
	#endif

		return true;
	}

	template <typename FailFn>
	bool TryRebuildFluidParticlesWithReceivedHalos_InPlace(
		bool& rebuiltInPlace,
		Sync::UniversalSimState& localState,
		DistParticleState& particleState,
		const std::vector<HaloExchangeWorkBuffers::RankBuffers>& exchanges,
		const uint32_t globalParticleCount,
		FailFn fail)
	{
		rebuiltInPlace = false;
		if (!OwnedParticlesAreCompactPrefix(particleState)) { return true; }

		const uint32_t oldLocalParticleCount = localState.fluidParticles.N_;
		const uint32_t ownedParticleCount = particleState.ownedParticleCount;
		const uint64_t recvHaloCount = CountReceivedFluidHalos(exchanges);
		if (recvHaloCount >
			static_cast<uint64_t>(std::numeric_limits<uint32_t>::max() - ownedParticleCount))
		{
			return fail("received fluid halo count exceeds uint32_t local particle count range.");
		}

		const uint32_t haloParticleCount = static_cast<uint32_t>(recvHaloCount);
		const uint32_t localParticleCount = ownedParticleCount + haloParticleCount;

		if (ownedParticleCount > oldLocalParticleCount)
		{
			return fail("fluid owned particle count exceeds local fluid particle count.");
		}
		if (static_cast<uint64_t>(particleState.particleGlobalIdx_FromLocalIdx.size()) !=
			oldLocalParticleCount)
		{
			return fail("fluid local-to-global map size does not match local fluid particle count.");
		}
		if (static_cast<uint64_t>(particleState.particleLocalIdx_FromGlobalIdx.size()) !=
			globalParticleCount)
		{
			return fail("fluid global-to-local map size does not match global fluid particle count.");
		}

		// old halos are replaced wholesale by the newly received halo tail
		for (uint32_t localIdx = ownedParticleCount;
			localIdx < oldLocalParticleCount;
			localIdx++)
		{
			if (!ClearLocalMapEntryIfStillMapped(
				particleState, static_cast<int>(localIdx), globalParticleCount))
			{
				return fail("failed to clear old fluid halo particle map entry.");
			}
		}

		localState.fluidParticles.resizeAll(localParticleCount);
		particleState.particleGlobalIdx_FromLocalIdx.resize(localParticleCount, -1);
		particleState.particleClasses.resize(localParticleCount, DistParticleClass::None);
		std::fill(
			particleState.particleGlobalIdx_FromLocalIdx.begin() + ownedParticleCount,
			particleState.particleGlobalIdx_FromLocalIdx.end(), -1);
		std::fill(
			particleState.particleClasses.begin() + ownedParticleCount,
			particleState.particleClasses.end(), DistParticleClass::None);
		particleState.particleLocalIdx_Halo.clear();
		particleState.particleLocalIdx_Halo.reserve(haloParticleCount);
		particleState.ownedParticleCount = ownedParticleCount;
		particleState.haloParticleCount = haloParticleCount;

		uint64_t recvParticleIdx = 0;
		for (const HaloExchangeWorkBuffers::RankBuffers& exchange : exchanges)
		{
			const HaloParticleBufferF& recvFluidParticles =
				exchange.recvBuffer.fluidParticles;

			for (uint64_t bufferIdx = 0; bufferIdx != recvFluidParticles.Size(); bufferIdx++)
			{
				const int dstLocalIdx =
					static_cast<int>(ownedParticleCount + recvParticleIdx);

				if (!AddParticleMapEntry(
					particleState, recvFluidParticles.GlobalIdxAt(bufferIdx),
					dstLocalIdx, globalParticleCount))
				{
					return fail("failed to add received fluid halo particle map entry.");
				}

				recvFluidParticles.CopyToLocalParticle(
					localState.fluidParticles, bufferIdx, dstLocalIdx);

				particleState.particleClasses[dstLocalIdx] = DistParticleClass::Halo;
				particleState.particleLocalIdx_Halo.push_back(dstLocalIdx);
				recvParticleIdx++;
			}
		}

	#if defined(_DEBUG)
		if (!particleState.IsValidLocalLayout(localParticleCount))
		{
			return fail("rebuilt fluid halo layout is invalid.");
		}
	#endif

		rebuiltInPlace = true;
		return true;
	}

	template <typename FailFn>
	bool TryRebuildBoundaryParticlesWithReceivedHalos_InPlace(
		bool& rebuiltInPlace,
		Sync::UniversalSimState& localState,
		DistParticleState& particleState,
		const std::vector<HaloExchangeWorkBuffers::RankBuffers>& exchanges,
		const uint32_t globalParticleCount,
		FailFn fail)
	{
		rebuiltInPlace = false;
		if (!OwnedParticlesAreCompactPrefix(particleState)) { return true; }

		const uint32_t oldLocalParticleCount = localState.boundaryParticles.N_;
		const uint32_t ownedParticleCount = particleState.ownedParticleCount;
		const uint64_t recvHaloCount = CountReceivedBoundaryHalos(exchanges);
		if (recvHaloCount >
			static_cast<uint64_t>(std::numeric_limits<uint32_t>::max() - ownedParticleCount))
		{
			return fail("received boundary halo count exceeds uint32_t local particle count range.");
		}

		const uint32_t haloParticleCount = static_cast<uint32_t>(recvHaloCount);
		const uint32_t localParticleCount = ownedParticleCount + haloParticleCount;

		if (ownedParticleCount > oldLocalParticleCount)
		{
			return fail("boundary owned particle count exceeds local boundary particle count.");
		}
		if (static_cast<uint64_t>(particleState.particleGlobalIdx_FromLocalIdx.size()) !=
			oldLocalParticleCount)
		{
			return fail("boundary local-to-global map size does not match local boundary particle count.");
		}
		if (static_cast<uint64_t>(particleState.particleLocalIdx_FromGlobalIdx.size()) !=
			globalParticleCount)
		{
			return fail("boundary global-to-local map size does not match global boundary particle count.");
		}

		// old halos are replaced wholesale by the newly received halo tail
		for (uint32_t localIdx = ownedParticleCount;
			localIdx < oldLocalParticleCount;
			localIdx++)
		{
			if (!ClearLocalMapEntryIfStillMapped(
				particleState, static_cast<int>(localIdx), globalParticleCount))
			{
				return fail("failed to clear old boundary halo particle map entry.");
			}
		}

		localState.boundaryParticles.resizeAll(localParticleCount);
		particleState.particleGlobalIdx_FromLocalIdx.resize(localParticleCount, -1);
		particleState.particleClasses.resize(localParticleCount, DistParticleClass::None);
		std::fill(
			particleState.particleGlobalIdx_FromLocalIdx.begin() + ownedParticleCount,
			particleState.particleGlobalIdx_FromLocalIdx.end(), -1);
		std::fill(
			particleState.particleClasses.begin() + ownedParticleCount,
			particleState.particleClasses.end(), DistParticleClass::None);
		particleState.particleLocalIdx_Halo.clear();
		particleState.particleLocalIdx_Halo.reserve(haloParticleCount);
		particleState.ownedParticleCount = ownedParticleCount;
		particleState.haloParticleCount = haloParticleCount;

		uint64_t recvParticleIdx = 0;
		for (const HaloExchangeWorkBuffers::RankBuffers& exchange : exchanges)
		{
			const HaloParticleBufferB& recvBoundaryParticles =
				exchange.recvBuffer.boundaryParticles;

			for (uint64_t bufferIdx = 0; bufferIdx != recvBoundaryParticles.Size(); bufferIdx++)
			{
				const int dstLocalIdx =
					static_cast<int>(ownedParticleCount + recvParticleIdx);

				if (!AddParticleMapEntry(
					particleState, recvBoundaryParticles.GlobalIdxAt(bufferIdx),
					dstLocalIdx, globalParticleCount))
				{
					return fail("failed to add received boundary halo particle map entry.");
				}

				recvBoundaryParticles.CopyToLocalParticle(
					localState.boundaryParticles, bufferIdx, dstLocalIdx);

				particleState.particleClasses[dstLocalIdx] = DistParticleClass::Halo;
				particleState.particleLocalIdx_Halo.push_back(dstLocalIdx);
				recvParticleIdx++;
			}
		}

	#if defined(_DEBUG)
		if (!particleState.IsValidLocalLayout(localParticleCount))
		{
			return fail("rebuilt boundary halo layout is invalid.");
		}
	#endif

		rebuiltInPlace = true;
		return true;
	}

} // namespace anonymous

void HaloExchangeBuffer::Clear()
{
	fluidParticles.Clear();
	boundaryParticles.Clear();
}

bool Dist::PackHaloExchangeBufferForRank(
	HaloExchangeBuffer& sendBuffer,
	const Sync::UniversalSimState& localState,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	const IDomainDecomp& domainDecomp,
	const int dstRank,
	std::string* failMsg)
{
	if (failMsg) { failMsg->clear(); }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};

	sendBuffer.Clear();

	// query domain decomp for the halo bounds of the destination rank
	const AABB dstHaloBounds = domainDecomp.HaloExtentAABB(dstRank);

	uint64_t sendFluidParticleCount = 0;
	uint64_t sendBoundaryParticleCount = 0;

	// count fluid particles that overlap the destination rank's halo bounds
	for (const int localIdx : fluidParticleState.particleLocalIdx_Outer)
	{
		// invariant checks for correct metadata
		if (!IsLocalIdxInRange(localIdx, localState.fluidParticles.N_))
		{
			return fail("fluid outer local index is outside the local fluid particle range.");
		}
		if (!HasParticleMetadataAtLocalIdx(fluidParticleState, localIdx))
		{
			return fail("fluid outer local index has no particle metadata.");
		}

		// skip if particle outside of destination halo bounds
		// (should be rare if domain decomposition is good)
		if (!dstHaloBounds.Contains(FluidPositionAtLocalIdx(localState.fluidParticles, localIdx)))
			{ continue; }

		sendFluidParticleCount++;
	}

	// count boundary particles that overlap the destination rank's halo bounds
	for (const int localIdx : boundaryParticleState.particleLocalIdx_Outer)
	{
		// invariant checks for correct metadata
		if (!IsLocalIdxInRange(localIdx, localState.boundaryParticles.N_))
		{
			return fail("boundary outer local index is outside the local boundary particle range.");
		}
		if (!HasParticleMetadataAtLocalIdx(boundaryParticleState, localIdx))
		{
			return fail("boundary outer local index has no particle metadata.");
		}

		// skip if particle outside of destination halo bounds
		// (should be rare if domain decomposition is good)
		if (!dstHaloBounds.Contains(BoundaryPositionAtLocalIdx(localState.boundaryParticles, localIdx)))
			{ continue; }

		sendBoundaryParticleCount++;
	}

	sendBuffer.fluidParticles.Resize(sendFluidParticleCount);
	sendBuffer.boundaryParticles.Resize(sendBoundaryParticleCount);

	uint64_t fluidBufferIdx = 0;
	uint64_t boundaryBufferIdx = 0;

	// pack fluid particles into pre-sized SoA payload arrays
	for (const int localIdx : fluidParticleState.particleLocalIdx_Outer)
	{
		if (!dstHaloBounds.Contains(FluidPositionAtLocalIdx(localState.fluidParticles, localIdx)))
			{ continue; }

		sendBuffer.fluidParticles.SetFromLocalParticle(
			localState.fluidParticles, fluidParticleState, fluidBufferIdx, localIdx);
		fluidBufferIdx++;
	}

	// pack boundary particles into pre-sized SoA payload arrays
	for (const int localIdx : boundaryParticleState.particleLocalIdx_Outer)
	{
		if (!dstHaloBounds.Contains(BoundaryPositionAtLocalIdx(localState.boundaryParticles, localIdx)))
			{ continue; }

		sendBuffer.boundaryParticles.SetFromLocalParticle(
			localState.boundaryParticles, boundaryParticleState, boundaryBufferIdx, localIdx);
		boundaryBufferIdx++;
	}

	return true;
}

bool Dist::ExchangeHaloParticlesWithNeighborRanks(
	const Sync::UniversalSimState& localState,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	const IDomainDecomp& domainDecomp,
	const MpiContext& mpiContext,
	HaloExchangeWorkBuffers& workBuffers,
	HaloExchangeStats* stats,
	std::string* failMsg)
{
	if (failMsg) { failMsg->clear(); }
	if (stats) { *stats = {}; }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};

	HaloExchangeStats localStats{};
	uint64_t sentFluidParticleCount = 0;
	uint64_t sentBoundaryParticleCount = 0;

	// prepare halo exchange buffers for every neighbor rank
	const std::vector<int>& neighborRanks = domainDecomp.NeighborRanks(mpiContext.Rank());
	if (!PrepareHaloExchangeRankBuffers(
		workBuffers, neighborRanks, mpiContext.Size(), fail))
	{
		return false;
	}
	std::vector<HaloExchangeWorkBuffers::RankBuffers>& exchanges =
		workBuffers.rankBuffers;

	// pack halo exchange payloads
	bool localPackOk = true;
	std::string localPackFailMsg{};
	for (HaloExchangeWorkBuffers::RankBuffers& exchange : exchanges)
	{
		std::string packFailMsg{};
		if (!PackHaloExchangeBufferForRank(
			exchange.sendBuffer, localState, fluidParticleState, boundaryParticleState,
			domainDecomp, exchange.rank, &packFailMsg))
		{
			localPackOk = false;
			localPackFailMsg =
				"failed to pack halo exchange buffer for rank " +
				std::to_string(exchange.rank) + ": " + packFailMsg;
			break;
		}

		exchange.sendFluidCount =
			exchange.sendBuffer.fluidParticles.CheckedBufferCount();
		exchange.sendBoundaryCount =
			exchange.sendBuffer.boundaryParticles.CheckedBufferCount();
		if (exchange.sendBuffer.fluidParticles.Size() >
			static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) -
				sentFluidParticleCount)
		{
			localPackOk = false;
			localPackFailMsg = "sent fluid halo particle count exceeds uint32_t range.";
			break;
		}
		if (exchange.sendBuffer.boundaryParticles.Size() >
			static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) -
				sentBoundaryParticleCount)
		{
			localPackOk = false;
			localPackFailMsg = "sent boundary halo particle count exceeds uint32_t range.";
			break;
		}

		sentFluidParticleCount += exchange.sendBuffer.fluidParticles.Size();
		sentBoundaryParticleCount += exchange.sendBuffer.boundaryParticles.Size();
	}

	if (!mpiContext.AllTrue(localPackOk))
	{
		if (!localPackOk) { return fail(localPackFailMsg); }
		return fail("at least one rank failed to pack halo exchange buffers.");
	}

	localStats.sentFluidParticleCount =
		static_cast<uint32_t>(sentFluidParticleCount);
	localStats.sentBoundaryParticleCount =
		static_cast<uint32_t>(sentBoundaryParticleCount);

	std::vector<MPI_Request> requests{};
	requests.reserve(exchanges.size() * 4);

	// exchange particle counts before exchanging raw payloads
	for (HaloExchangeWorkBuffers::RankBuffers& exchange : exchanges) // post non-blocking MPI receives
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.IRecvInt(
			exchange.recvFluidCount,
			exchange.rank,
			HALO_EXCHANGE_TAG_FLUID_COUNT,
			requests.back());
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.IRecvInt(
			exchange.recvBoundaryCount,
			exchange.rank,
			HALO_EXCHANGE_TAG_BOUNDARY_COUNT,
			requests.back());
	}
	for (HaloExchangeWorkBuffers::RankBuffers& exchange : exchanges) // post non-blocking MPI sends
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			exchange.sendFluidCount,
			exchange.rank,
			HALO_EXCHANGE_TAG_FLUID_COUNT,
			requests.back());
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			exchange.sendBoundaryCount,
			exchange.rank,
			HALO_EXCHANGE_TAG_BOUNDARY_COUNT,
			requests.back());
	}
	mpiContext.WaitAll(requests);
	requests.clear();

	// resize receive buffers according to exchanged particle counts
	for (HaloExchangeWorkBuffers::RankBuffers& exchange : exchanges)
	{
		if (exchange.recvFluidCount < 0 || exchange.recvBoundaryCount < 0)
		{
			return fail("received negative halo exchange particle count.");
		}

		exchange.recvBuffer.fluidParticles.Resize(
			static_cast<uint64_t>(exchange.recvFluidCount));
		exchange.recvBuffer.boundaryParticles.Resize(
			static_cast<uint64_t>(exchange.recvBoundaryCount));
	}

	requests.reserve(exchanges.size() * 42);

	// exchange only non-empty payload chunks after all receive buffers are sized correctly
	for (HaloExchangeWorkBuffers::RankBuffers& exchange : exchanges) // post non-blocking MPI receives
	{
		if (exchange.recvFluidCount > 0)
		{
			exchange.recvBuffer.fluidParticles.QueueIRecvPayload(
				mpiContext, exchange.rank, HALO_EXCHANGE_TAG_FLUID_PAYLOAD, requests);
		}
		if (exchange.recvBoundaryCount > 0)
		{
			exchange.recvBuffer.boundaryParticles.QueueIRecvPayload(
				mpiContext, exchange.rank, HALO_EXCHANGE_TAG_BOUNDARY_PAYLOAD, requests);
		}
	}
	for (HaloExchangeWorkBuffers::RankBuffers& exchange : exchanges) // post non-blocking MPI sends
	{
		if (exchange.sendFluidCount > 0)
		{
			exchange.sendBuffer.fluidParticles.QueueISendPayload(
				mpiContext, exchange.rank, HALO_EXCHANGE_TAG_FLUID_PAYLOAD, requests);
		}
		if (exchange.sendBoundaryCount > 0)
		{
			exchange.sendBuffer.boundaryParticles.QueueISendPayload(
				mpiContext, exchange.rank, HALO_EXCHANGE_TAG_BOUNDARY_PAYLOAD, requests);
		}
	}
	mpiContext.WaitAll(requests);

	uint64_t recvFluidParticleCount = 0;
	uint64_t recvBoundaryParticleCount = 0;
	for (const HaloExchangeWorkBuffers::RankBuffers& exchange : exchanges)
	{
		if (exchange.recvBuffer.fluidParticles.Size() >
			static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) -
				recvFluidParticleCount)
		{
			return fail("received fluid halo particle count exceeds uint32_t range.");
		}
		if (exchange.recvBuffer.boundaryParticles.Size() >
			static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) -
				recvBoundaryParticleCount)
		{
			return fail("received boundary halo particle count exceeds uint32_t range.");
		}

		recvFluidParticleCount += exchange.recvBuffer.fluidParticles.Size();
		recvBoundaryParticleCount += exchange.recvBuffer.boundaryParticles.Size();
	}

	localStats.recvFluidParticleCount =
		static_cast<uint32_t>(recvFluidParticleCount);
	localStats.recvBoundaryParticleCount =
		static_cast<uint32_t>(recvBoundaryParticleCount);
	if (stats) { *stats = localStats; }

	return true;
}

bool Dist::RebuildLocalStateWithReceivedHalos(
	const std::vector<HaloExchangeWorkBuffers::RankBuffers>& exchanges,
	Sync::UniversalSimState& localState,
	DistParticleState& fluidParticleState,
	DistParticleState& boundaryParticleState,
	const uint32_t globalFluidParticleCount,
	const uint32_t globalBoundaryParticleCount,
	HaloRebuildScratch& scratch,
	std::string* failMsg)
{
	if (failMsg) { failMsg->clear(); }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};

	bool rebuiltFluidInPlace = false;
	if (!TryRebuildFluidParticlesWithReceivedHalos_InPlace(
		rebuiltFluidInPlace, localState, fluidParticleState, exchanges,
		globalFluidParticleCount, fail))
	{
		return false;
	}

	if (!rebuiltFluidInPlace)
	{
		scratch.state.schemaVersion = localState.schemaVersion;
		scratch.state.step = localState.step;
		scratch.state.time = localState.time;

		if (!RebuildFluidParticlesWithReceivedHalos_Copy(
			scratch.state, scratch.fluidParticleState,
			localState, fluidParticleState, exchanges,
			globalFluidParticleCount, fail))
		{
			return false;
		}

		// publish fallback rebuild when owned particles are not compact enough for in-place append
		std::swap(localState.fluidParticles, scratch.state.fluidParticles);
		std::swap(fluidParticleState, scratch.fluidParticleState);
	}

	bool rebuiltBoundaryInPlace = false;
	if (!TryRebuildBoundaryParticlesWithReceivedHalos_InPlace(
		rebuiltBoundaryInPlace, localState, boundaryParticleState, exchanges,
		globalBoundaryParticleCount, fail))
	{
		return false;
	}

	if (!rebuiltBoundaryInPlace)
	{
		scratch.state.schemaVersion = localState.schemaVersion;
		scratch.state.step = localState.step;
		scratch.state.time = localState.time;

		if (!RebuildBoundaryParticlesWithReceivedHalos_Copy(
			scratch.state, scratch.boundaryParticleState,
			localState, boundaryParticleState, exchanges,
			globalBoundaryParticleCount, fail))
		{
			return false;
		}

		// publish fallback rebuild when owned particles are not compact enough for in-place append
		std::swap(localState.boundaryParticles, scratch.state.boundaryParticles);
		std::swap(boundaryParticleState, scratch.boundaryParticleState);
	}

	return true;
}
