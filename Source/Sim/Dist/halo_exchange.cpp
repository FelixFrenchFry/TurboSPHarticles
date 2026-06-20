// halo_exchange.cpp

// external
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
	constexpr int HALO_EXCHANGE_TAG_FLUID_PAYLOAD = 4103;
	constexpr int HALO_EXCHANGE_TAG_BOUNDARY_PAYLOAD = 4104;

	struct NeighborHaloExchange
	{
		int neighborRank = -1;

		HaloExchangeBuffer sendBuffer;
		HaloExchangeBuffer recvBuffer;

		int sendFluidCount = 0;
		int sendBoundaryCount = 0;
		int recvFluidCount = 0;
		int recvBoundaryCount = 0;
	};

	template <typename T>
	int CheckedPacketCount(const std::vector<T>& packets)
	{
		if (packets.size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
		{
			throw std::runtime_error("Halo exchange packet count exceeds MPI int count range.");
		}

		return static_cast<int>(packets.size());
	}

	template <typename T>
	int CheckedByteCount(const std::vector<T>& packets)
	{
		const uint64_t packetCount = packets.size();
		const uint64_t maxCount = static_cast<uint64_t>(std::numeric_limits<int>::max()) / sizeof(T);

		if (packetCount > maxCount)
		{
			throw std::runtime_error("Halo exchange payload exceeds MPI byte count range.");
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

	HaloFluidParticlePacket PackFluidParticle(
		const Sync::FluidParticles& fluidParticles,
		const DistParticleState& particleState,
		const int localIdx)
	{
		HaloFluidParticlePacket packet{};
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

	HaloBoundaryParticlePacket PackBoundaryParticle(
		const Sync::BoundaryParticles& boundaryParticles,
		const DistParticleState& particleState,
		const int localIdx)
	{
		HaloBoundaryParticlePacket packet{};
		packet.globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];

		packet.xPos = boundaryParticles.xPos[localIdx];
		packet.yPos = boundaryParticles.yPos[localIdx];
		packet.zPos = boundaryParticles.zPos[localIdx];

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

	void CopyFluidPacketToLocal(
		const HaloFluidParticlePacket& packet,
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

	void CopyBoundaryPacketToLocal(
		const HaloBoundaryParticlePacket& packet,
		Sync::BoundaryParticles& dst,
		const int localIdx)
	{
		dst.xPos[localIdx] = packet.xPos;
		dst.yPos[localIdx] = packet.yPos;
		dst.zPos[localIdx] = packet.zPos;
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

	// TODO: find a more efficient way to do this without completely rebuilding the local state?
	bool RebuildFluidParticlesWithReceivedHalos(
		Sync::UniversalSimState& rebuiltState,
		DistParticleState& rebuiltParticleState,
		const Sync::UniversalSimState& srcState,
		const DistParticleState& srcParticleState,
		const std::vector<HaloFluidParticlePacket>& recvFluidParticles,
		const uint32_t globalParticleCount)
	{
		// compute new local particle counts
		const uint32_t ownedParticleCount = static_cast<uint32_t>(srcParticleState.particleLocalIdx_Owned.size());
		const uint32_t haloParticleCount = static_cast<uint32_t>(recvFluidParticles.size());
		const uint32_t localParticleCount = ownedParticleCount + haloParticleCount;

		// resize rebuilt state and initialize metadata structures
		// TODO: use permanent scratch space instead?
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
				{ return false; }
			if (!HasParticleMetadataAtLocalIdx(srcParticleState, srcLocalIdx))
				{ return false; }

			const DistParticleClass particleClass = srcParticleState.particleClasses[srcLocalIdx];
			if (particleClass != DistParticleClass::Inner &&
				particleClass != DistParticleClass::Outer)
				{ return false; }

			const int globalIdx = srcParticleState.particleGlobalIdx_FromLocalIdx[srcLocalIdx];
			if (!AddParticleMapEntry(
				rebuiltParticleState, globalIdx, dstLocalIdx, globalParticleCount))
				{ return false; }

			CopyFluidParticle_LocalToLocal(
				srcState.fluidParticles, rebuiltState.fluidParticles, srcLocalIdx, dstLocalIdx);

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

		for (const HaloFluidParticlePacket& packet : recvFluidParticles)
		{
			if (!AddParticleMapEntry(
				rebuiltParticleState, packet.globalIdx, dstLocalIdx, globalParticleCount))
				{ return false; }

			CopyFluidPacketToLocal(packet, rebuiltState.fluidParticles, dstLocalIdx);

			rebuiltParticleState.particleClasses[dstLocalIdx] = DistParticleClass::Halo;
			rebuiltParticleState.particleLocalIdx_Halo.push_back(dstLocalIdx);
			dstLocalIdx++;
		}

		// TODO: enable validation only in debug builds?
		return rebuiltParticleState.IsValidLocalLayout(localParticleCount);
	}

	// TODO: find a more efficient way to do this without completely rebuilding the local state?
	bool RebuildBoundaryParticlesWithReceivedHalos(
		Sync::UniversalSimState& rebuiltState,
		DistParticleState& rebuiltParticleState,
		const Sync::UniversalSimState& srcState,
		const DistParticleState& srcParticleState,
		const std::vector<HaloBoundaryParticlePacket>& recvBoundaryParticles,
		const uint32_t globalParticleCount)
	{
		// compute new local particle counts
		const uint32_t ownedParticleCount = static_cast<uint32_t>(srcParticleState.particleLocalIdx_Owned.size());
		const uint32_t haloParticleCount = static_cast<uint32_t>(recvBoundaryParticles.size());
		const uint32_t localParticleCount = ownedParticleCount + haloParticleCount;

		// resize rebuilt state and initialize metadata structures
		// TODO: use permanent scratch space instead?
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
				{ return false; }
			if (!HasParticleMetadataAtLocalIdx(srcParticleState, srcLocalIdx))
				{ return false; }

			const DistParticleClass particleClass = srcParticleState.particleClasses[srcLocalIdx];
			if (particleClass != DistParticleClass::Inner &&
				particleClass != DistParticleClass::Outer)
				{ return false; }

			const int globalIdx = srcParticleState.particleGlobalIdx_FromLocalIdx[srcLocalIdx];
			if (!AddParticleMapEntry(
				rebuiltParticleState, globalIdx, dstLocalIdx, globalParticleCount))
				{ return false; }

			CopyBoundaryParticle_LocalToLocal(
				srcState.boundaryParticles, rebuiltState.boundaryParticles, srcLocalIdx, dstLocalIdx);

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

		for (const HaloBoundaryParticlePacket& packet : recvBoundaryParticles)
		{
			if (!AddParticleMapEntry(
				rebuiltParticleState, packet.globalIdx, dstLocalIdx, globalParticleCount))
				{ return false; }

			CopyBoundaryPacketToLocal(packet, rebuiltState.boundaryParticles, dstLocalIdx);

			rebuiltParticleState.particleClasses[dstLocalIdx] = DistParticleClass::Halo;
			rebuiltParticleState.particleLocalIdx_Halo.push_back(dstLocalIdx);
			dstLocalIdx++;
		}

		// TODO: enable validation only in debug builds?
		return rebuiltParticleState.IsValidLocalLayout(localParticleCount);
	}

} // namespace anonymous

void HaloExchangeBuffer::Clear()
{
	fluidParticles.clear();
	boundaryParticles.clear();
}

// TODO: re-implement with TBB (parallel count -> prefix sum -> parallel fill)?
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

	// pack fluid particles
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

		sendBuffer.fluidParticles.push_back(
			PackFluidParticle(localState.fluidParticles, fluidParticleState, localIdx));
	}

	// pack boundary particles
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

		sendBuffer.boundaryParticles.push_back(
			PackBoundaryParticle(localState.boundaryParticles, boundaryParticleState, localIdx));
	}

	return true;
}

bool Dist::ExchangeHaloParticlesWithNeighborRanks(
	HaloExchangeBuffer& recvBuffer,
	const Sync::UniversalSimState& localState,
	const DistParticleState& fluidParticleState,
	const DistParticleState& boundaryParticleState,
	const IDomainDecomp& domainDecomp,
	const MpiContext& mpiContext,
	std::string* failMsg)
{
	if (failMsg) { failMsg->clear(); }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};

	recvBuffer.Clear();

	// query domain decomp for neighbor ranks and prepare halo exchange package for each neighbor
	const std::vector<int>& neighborRanks = domainDecomp.NeighborRanks(mpiContext.Rank());
	std::vector<NeighborHaloExchange> exchanges{};
	exchanges.reserve(neighborRanks.size());

	// pack halo exchange packages
	bool localPackOk = true;
	std::string localPackFailMsg{};
	for (const int neighborRank : neighborRanks)
	{
		if (!localPackOk) { break; }

		NeighborHaloExchange exchange{};
		exchange.neighborRank = neighborRank;
		std::string packFailMsg{};
		if (!PackHaloExchangeBufferForRank(
			exchange.sendBuffer, localState, fluidParticleState, boundaryParticleState,
			domainDecomp, neighborRank, &packFailMsg))
		{
			localPackOk = false;
			localPackFailMsg =
				"Failed to pack halo exchange buffer for rank " +
				std::to_string(neighborRank) + ": " + packFailMsg;
			break;
		}

		exchange.sendFluidCount = CheckedPacketCount(exchange.sendBuffer.fluidParticles);
		exchange.sendBoundaryCount = CheckedPacketCount(exchange.sendBuffer.boundaryParticles);
		exchanges.push_back(std::move(exchange));
	}

	if (!mpiContext.AllTrue(localPackOk))
	{
		if (!localPackOk) { return fail(localPackFailMsg); }
		return fail("At least one rank failed to pack halo exchange buffers.");
	}

	std::vector<MPI_Request> requests{};
	requests.reserve(exchanges.size() * 4);

	// exchange packet counts before exchanging raw packet payloads
	for (NeighborHaloExchange& exchange : exchanges) // post non-blocking MPI receives
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.IRecvInt(
			exchange.recvFluidCount,
			exchange.neighborRank,
			HALO_EXCHANGE_TAG_FLUID_COUNT,
			requests.back());
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.IRecvInt(
			exchange.recvBoundaryCount,
			exchange.neighborRank,
			HALO_EXCHANGE_TAG_BOUNDARY_COUNT,
			requests.back());
	}
	for (NeighborHaloExchange& exchange : exchanges) // post non-blocking MPI sends
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			exchange.sendFluidCount,
			exchange.neighborRank,
			HALO_EXCHANGE_TAG_FLUID_COUNT,
			requests.back());
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			exchange.sendBoundaryCount,
			exchange.neighborRank,
			HALO_EXCHANGE_TAG_BOUNDARY_COUNT,
			requests.back());
	}
	mpiContext.WaitAll(requests);
	requests.clear();

	// resize receive buffers according to exchanged packet counts
	for (NeighborHaloExchange& exchange : exchanges)
	{
		if (exchange.recvFluidCount < 0 || exchange.recvBoundaryCount < 0)
		{
			return fail("Received negative halo exchange packet count.");
		}

		exchange.recvBuffer.fluidParticles.resize(static_cast<uint64_t>(exchange.recvFluidCount));
		exchange.recvBuffer.boundaryParticles.resize(static_cast<uint64_t>(exchange.recvBoundaryCount));
	}

	// exchange payloads after all receive buffers are sized correctly
	for (NeighborHaloExchange& exchange : exchanges) // post non-blocking MPI receives
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.IRecvBytes(
			DataOrNull(exchange.recvBuffer.fluidParticles),
			CheckedByteCount(exchange.recvBuffer.fluidParticles),
			exchange.neighborRank,
			HALO_EXCHANGE_TAG_FLUID_PAYLOAD,
			requests.back());
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.IRecvBytes(
			DataOrNull(exchange.recvBuffer.boundaryParticles),
			CheckedByteCount(exchange.recvBuffer.boundaryParticles),
			exchange.neighborRank,
			HALO_EXCHANGE_TAG_BOUNDARY_PAYLOAD,
			requests.back());
	}
	for (NeighborHaloExchange& exchange : exchanges) // post non-blocking MPI sends
	{
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendBytes(
			DataOrNull(exchange.sendBuffer.fluidParticles),
			CheckedByteCount(exchange.sendBuffer.fluidParticles),
			exchange.neighborRank,
			HALO_EXCHANGE_TAG_FLUID_PAYLOAD,
			requests.back());
		requests.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendBytes(
			DataOrNull(exchange.sendBuffer.boundaryParticles),
			CheckedByteCount(exchange.sendBuffer.boundaryParticles),
			exchange.neighborRank,
			HALO_EXCHANGE_TAG_BOUNDARY_PAYLOAD,
			requests.back());
	}
	mpiContext.WaitAll(requests);

	// combine received halo particles from all neighbors into a single receive buffer
	for (const NeighborHaloExchange& exchange : exchanges)
	{
		recvBuffer.fluidParticles.insert(
			recvBuffer.fluidParticles.end(),
			exchange.recvBuffer.fluidParticles.begin(),
			exchange.recvBuffer.fluidParticles.end());
		recvBuffer.boundaryParticles.insert(
			recvBuffer.boundaryParticles.end(),
			exchange.recvBuffer.boundaryParticles.begin(),
			exchange.recvBuffer.boundaryParticles.end());
	}

	return true;
}

bool Dist::RebuildLocalStateWithReceivedHalos(
	const HaloExchangeBuffer& recvBuffer,
	Sync::UniversalSimState& localState,
	DistParticleState& fluidParticleState,
	DistParticleState& boundaryParticleState,
	const uint32_t globalFluidParticleCount,
	const uint32_t globalBoundaryParticleCount)
{
	// create temp rebuilt state
	Sync::UniversalSimState rebuiltState{};
	DistParticleState rebuiltFluidParticleState{};
	DistParticleState rebuiltBoundaryParticleState{};

	// copy metadata into rebuilt state
	rebuiltState.schemaVersion = localState.schemaVersion;
	rebuiltState.step = localState.step;
	rebuiltState.time = localState.time;

	// rebuild fluid particle state
	if (!RebuildFluidParticlesWithReceivedHalos(
		rebuiltState, rebuiltFluidParticleState, localState, fluidParticleState,
		recvBuffer.fluidParticles, globalFluidParticleCount))
		{ return false; }

	// rebuild boundary particle state
	if (!RebuildBoundaryParticlesWithReceivedHalos(
		rebuiltState, rebuiltBoundaryParticleState, localState, boundaryParticleState,
		recvBuffer.boundaryParticles, globalBoundaryParticleCount))
		{ return false; }

	// move temp rebuilt state into live state
	localState = std::move(rebuiltState);
	fluidParticleState = std::move(rebuiltFluidParticleState);
	boundaryParticleState = std::move(rebuiltBoundaryParticleState);
	return true;
}
