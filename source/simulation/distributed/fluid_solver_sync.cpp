// fluid_solver_sync.cpp

// external
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// internal
#include "fluid_solver_sync.h"



using namespace Dist;

namespace
{
	// separate tags for plan counts, global-index validation, and solver payloads
	constexpr int FLUID_SOLVER_SYNC_TAG_COUNT = 6200;
	constexpr int FLUID_SOLVER_SYNC_TAG_GLOBAL_IDXS = 6201;
	constexpr int FLUID_SOLVER_SYNC_TAG_PAYLOAD = 6202;

	// retain the first local validation failure for rank-wide reporting
	bool SetFailure(std::string& localFailMsg, const std::string& failMsg)
	{
		if (localFailMsg.empty()) { localFailMsg = failMsg; }
		return false;
	}

	// report validation failures consistently on every rank
	bool FinishCollectiveValidation(
		const MpiContext& mpiContext,
		const bool localOk,
		const std::string& localFailMsg,
		std::string* failMsg)
	{
		if (mpiContext.AllTrue(localOk)) { return true; }

		if (failMsg)
		{
			*failMsg = localOk
				? "Fluid solver synchronization validation failed on another rank."
				: localFailMsg;
		}
		return false;
	}

	// sort local indices into the global ID order shared by both peer plans
	void SortPeerParticleIdxs(
		std::vector<uint32_t>& localIdxs,
		std::vector<uint32_t>& globalIdxs)
	{
		std::vector<std::pair<uint32_t, uint32_t>> idxPairs;
		idxPairs.reserve(localIdxs.size());
		for (uint32_t i = 0; i != localIdxs.size(); i++)
		{
			idxPairs.emplace_back(globalIdxs[i], localIdxs[i]);
		}

		std::ranges::sort(idxPairs);
		for (uint32_t i = 0; i != idxPairs.size(); i++)
		{
			globalIdxs[i] = idxPairs[i].first;
			localIdxs[i] = idxPairs[i].second;
		}
	}

	// duplicate global IDs indicate an invalid one-to-one exchange plan
	bool HasDuplicateGlobalIdxs(const std::vector<uint32_t>& globalIdxs)
	{
		return std::adjacent_find(globalIdxs.begin(), globalIdxs.end()) != globalIdxs.end();
	}

} // namespace anonymous

bool FluidSolverSyncMgr::Rebuild(
	const Sync::FluidConstView& fluidParticles,
	const DistParticleState& fluidParticleState,
	const IDomainDecomp& domainDecomp,
	const uint64_t N_expectedGlobalF,
	MpiContext& mpiContext,
	Bench::BenchmarksTracker& benchmarksTracker,
	std::string* failMsg)
{
	std::string localFailMsg;
	bool localOk = true;

	// --- [ input validation ] ----------------------------------------------------------------------

	// validate the global count, particle view, and rank-local metadata
	if (N_expectedGlobalF > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
	{
		localOk = SetFailure(localFailMsg,
			"Global fluid particle count exceeds the supported index range.");
	}
	const uint32_t N_globalF = static_cast<uint32_t>(N_expectedGlobalF);

	if (localOk &&
		(fluidParticles.xPos.size() != fluidParticles.N_ ||
		 fluidParticles.yPos.size() != fluidParticles.N_ ||
		 fluidParticles.zPos.size() != fluidParticles.N_))
	{
		localOk = SetFailure(localFailMsg,
			"Fluid particle position view size does not match its particle count.");
	}

	if (localOk && !fluidParticleState.IsValidLocalParticleState(
		fluidParticles.N_, N_globalF, &localFailMsg))
	{
		localOk = false;
	}

	// validate the domain-decomposition neighbors used by the solver plan
	const std::vector<int>& neighborRanks = domainDecomp.NeighborRanks(mpiContext.Rank());
	std::vector<int> sortedNeighborRanks = neighborRanks;
	std::ranges::sort(sortedNeighborRanks);
	if (localOk && std::adjacent_find(
		sortedNeighborRanks.begin(), sortedNeighborRanks.end()) != sortedNeighborRanks.end())
	{
		localOk = SetFailure(localFailMsg,
			"Fluid solver synchronization neighbor ranks contain duplicates.");
	}
	for (const int peerRank : sortedNeighborRanks)
	{
		if (!localOk) { break; }
		if (peerRank < 0 || peerRank >= mpiContext.Size() || peerRank == mpiContext.Rank())
		{
			localOk = SetFailure(localFailMsg,
				"Fluid solver synchronization neighbor rank is invalid.");
		}
	}

	if (!FinishCollectiveValidation(
		mpiContext, localOk, localFailMsg, failMsg))
	{
		return false;
	}

	// --- [ local peer-plan construction ] ----------------------------------------------------------

	// peer plans ordered by rank, looked up by binary search
	std::vector<PeerPlan> newPeerPlans(sortedNeighborRanks.size());
	for (uint32_t peerIdx = 0; peerIdx != newPeerPlans.size(); peerIdx++)
	{
		newPeerPlans[peerIdx].rank = sortedNeighborRanks[peerIdx];
	}

	auto findPeerPlan = [&](const int peerRank) -> PeerPlan*
	{
		auto it = std::ranges::lower_bound(
			newPeerPlans,
			peerRank,
			{},
			&PeerPlan::rank);
		return it != newPeerPlans.end() && it->rank == peerRank
			? &*it
			: nullptr;
	};

	// send owned particles to every neighboring rank whose halo contains them
	for (const uint32_t localIdx : fluidParticleState.particleLocalIdx_Owned)
	{
		const glm::vec3 position{
			fluidParticles.xPos[localIdx],
			fluidParticles.yPos[localIdx],
			fluidParticles.zPos[localIdx]
		};
		const uint32_t globalIdx =
			fluidParticleState.particleGlobalIdx_FromLocalIdx[localIdx];

		for (PeerPlan& peerPlan : newPeerPlans)
		{
			if (!domainDecomp.HaloExtentAABB(peerPlan.rank).Contains(position)) { continue; }

			peerPlan.sendLocalIdxs.push_back(localIdx);
			peerPlan.sendGlobalIdxs.push_back(globalIdx);
		}
	}

	// receive halo particles from their owning rank
	for (const uint32_t localIdx : fluidParticleState.particleLocalIdx_Halo)
	{
		const glm::vec3 position{
			fluidParticles.xPos[localIdx],
			fluidParticles.yPos[localIdx],
			fluidParticles.zPos[localIdx]
		};
		const int ownerRank = domainDecomp.OwnerRank(position);
		PeerPlan* peerPlan = findPeerPlan(ownerRank);
		if (!peerPlan)
		{
			localOk = SetFailure(localFailMsg,
				"Halo fluid particle owner is not a neighboring rank.");
			continue;
		}

		peerPlan->recvLocalIdxs.push_back(localIdx);
		peerPlan->recvGlobalIdxs.push_back(
			fluidParticleState.particleGlobalIdx_FromLocalIdx[localIdx]);
	}

	// sort by global ID and reject duplicate entries before exchanging the plan
	for (PeerPlan& peerPlan : newPeerPlans)
	{
		SortPeerParticleIdxs(peerPlan.sendLocalIdxs, peerPlan.sendGlobalIdxs);
		SortPeerParticleIdxs(peerPlan.recvLocalIdxs, peerPlan.recvGlobalIdxs);
		if (HasDuplicateGlobalIdxs(peerPlan.sendGlobalIdxs) ||
			HasDuplicateGlobalIdxs(peerPlan.recvGlobalIdxs))
		{
			localOk = SetFailure(localFailMsg,
				"Fluid solver synchronization plan contains duplicate global indices.");
		}
	}

	if (!FinishCollectiveValidation(
		mpiContext, localOk, localFailMsg, failMsg))
	{
		return false;
	}

	// --- [ peer count exchange ] -------------------------------------------------------------------

	// pack local send counts, rejecting values beyond the MPI int range
	std::vector<int> sendCounts(newPeerPlans.size(), 0);
	std::vector<int> recvCounts(newPeerPlans.size(), 0);
	requests_.clear();
	requests_.reserve(newPeerPlans.size() * 2);
	for (uint32_t peerIdx = 0; peerIdx != newPeerPlans.size(); peerIdx++)
	{
		const uint64_t N_send = newPeerPlans[peerIdx].sendGlobalIdxs.size();
		if (N_send > static_cast<uint64_t>(std::numeric_limits<int>::max()))
		{
			localOk = SetFailure(localFailMsg,
				"Fluid solver synchronization count exceeds the MPI int range.");
			continue;
		}
		sendCounts[peerIdx] = static_cast<int>(N_send);
	}
	if (!FinishCollectiveValidation(
		mpiContext, localOk, localFailMsg, failMsg))
	{
		return false;
	}

	// exchange particle counts before comparing global-index lists
	for (uint32_t peerIdx = 0; peerIdx != newPeerPlans.size(); peerIdx++)
	{
		requests_.emplace_back(MPI_REQUEST_NULL);
		mpiContext.IRecvInt(
			recvCounts[peerIdx], newPeerPlans[peerIdx].rank,
			FLUID_SOLVER_SYNC_TAG_COUNT, requests_.back());
	}
	for (uint32_t peerIdx = 0; peerIdx != newPeerPlans.size(); peerIdx++)
	{
		requests_.emplace_back(MPI_REQUEST_NULL);
		mpiContext.ISendInt(
			sendCounts[peerIdx], newPeerPlans[peerIdx].rank,
			FLUID_SOLVER_SYNC_TAG_COUNT, requests_.back());
	}
	mpiContext.WaitAll(requests_);

	for (uint32_t peerIdx = 0; peerIdx != newPeerPlans.size(); peerIdx++)
	{
		if (recvCounts[peerIdx] < 0 ||
			static_cast<uint64_t>(recvCounts[peerIdx]) !=
			newPeerPlans[peerIdx].recvGlobalIdxs.size())
		{
			localOk = SetFailure(localFailMsg,
				"Fluid solver synchronization peer particle count does not match the local halo plan.");
		}
	}
	if (!FinishCollectiveValidation(
		mpiContext, localOk, localFailMsg, failMsg))
	{
		return false;
	}

	// --- [ peer global-index exchange ] ------------------------------------------------------------

	// exchange global IDs to verify both ranks built the same plan
	std::vector<std::vector<uint32_t>> receivedGlobalIdxs(newPeerPlans.size());
	requests_.clear();
	for (uint32_t peerIdx = 0; peerIdx != newPeerPlans.size(); peerIdx++)
	{
		receivedGlobalIdxs[peerIdx].resize(static_cast<uint32_t>(recvCounts[peerIdx]));
		mpiContext.IRecvBytesChunked(
			receivedGlobalIdxs[peerIdx].data(),
			static_cast<uint64_t>(receivedGlobalIdxs[peerIdx].size()) * sizeof(uint32_t),
			newPeerPlans[peerIdx].rank,
			FLUID_SOLVER_SYNC_TAG_GLOBAL_IDXS,
			requests_);
	}
	for (const PeerPlan& peerPlan : newPeerPlans)
	{
		mpiContext.ISendBytesChunked(
			peerPlan.sendGlobalIdxs.data(),
			static_cast<uint64_t>(peerPlan.sendGlobalIdxs.size()) * sizeof(uint32_t),
			peerPlan.rank,
			FLUID_SOLVER_SYNC_TAG_GLOBAL_IDXS,
			requests_);
	}
	mpiContext.WaitAll(requests_);

	for (uint32_t peerIdx = 0; peerIdx != newPeerPlans.size(); peerIdx++)
	{
		if (receivedGlobalIdxs[peerIdx] != newPeerPlans[peerIdx].recvGlobalIdxs)
		{
			localOk = SetFailure(localFailMsg,
				"Fluid solver synchronization peer global indices do not match the local halo plan.");
		}
	}
	if (!FinishCollectiveValidation(
		mpiContext, localOk, localFailMsg, failMsg))
	{
		return false;
	}

	// publish the plan validated by every rank
	mpiContext_ = &mpiContext;
	benchmarksTracker_ = &benchmarksTracker;
	N_localF_ = fluidParticles.N_;
	N_globalF_ = N_expectedGlobalF;
	peerPlans_ = std::move(newPeerPlans);
	return true;
}

void FluidSolverSyncMgr::ApplyFluidParticleReOrder(
	const std::span<const uint32_t> oldFluidLocalIdx_FromNewLocalIdx)
{
	if (!mpiContext_)
	{
		throw std::runtime_error("Fluid solver synchronization plan is not initialized.");
	}

	// validate and invert the storage permutation
	const bool storageWasReOrdered = !oldFluidLocalIdx_FromNewLocalIdx.empty();
	bool localOk = !storageWasReOrdered ||
		oldFluidLocalIdx_FromNewLocalIdx.size() == N_localF_;
	std::vector<uint32_t> newFluidLocalIdx_FromOldLocalIdx;
	std::vector<uint8_t> oldFluidLocalIdxUsed;
	if (localOk && storageWasReOrdered)
	{
		newFluidLocalIdx_FromOldLocalIdx.resize(N_localF_);
		oldFluidLocalIdxUsed.assign(N_localF_, 0);
		for (uint32_t newLocalIdx = 0; newLocalIdx != N_localF_; newLocalIdx++)
		{
			const uint32_t oldLocalIdx = oldFluidLocalIdx_FromNewLocalIdx[newLocalIdx];
			if (oldLocalIdx >= N_localF_ || oldFluidLocalIdxUsed[oldLocalIdx] != 0)
			{
				localOk = false;
				break;
			}
			oldFluidLocalIdxUsed[oldLocalIdx] = 1;
			newFluidLocalIdx_FromOldLocalIdx[oldLocalIdx] = newLocalIdx;
		}
	}

	if (!mpiContext_->AllTrue(localOk))
	{
		throw std::runtime_error("Invalid distributed fluid particle re-order.");
	}

	if (!storageWasReOrdered) { return; }

	// remap send and receive indices, global particle IDs remain unchanged
	for (PeerPlan& peerPlan : peerPlans_)
	{
		for (uint32_t& localIdx : peerPlan.sendLocalIdxs)
		{
			localIdx = newFluidLocalIdx_FromOldLocalIdx[localIdx];
		}
		for (uint32_t& localIdx : peerPlan.recvLocalIdxs)
		{
			localIdx = newFluidLocalIdx_FromOldLocalIdx[localIdx];
		}
	}
}

void FluidSolverSyncMgr::SynchronizeFluidScalarField(
	const std::span<float> values)
{
	SynchronizeFluidFields({ values, {}, {} }, 1);
}

void FluidSolverSyncMgr::SynchronizeFluidVectorField(
	const std::span<float> xValues,
	const std::span<float> yValues,
	const std::span<float> zValues)
{
	SynchronizeFluidFields({ xValues, yValues, zValues }, 3);
}

void FluidSolverSyncMgr::SynchronizeFluidFields(
	const std::array<std::span<float>, 3>& fields,
	const uint32_t N_fields)
{
	if (!mpiContext_ || !benchmarksTracker_)
	{
		throw std::runtime_error("Fluid solver synchronization plan is not initialized.");
	}

	bool localOk = N_fields > 0 && N_fields <= fields.size();
	for (uint32_t fieldIdx = 0; fieldIdx != N_fields; fieldIdx++)
	{
		localOk = localOk && fields[fieldIdx].size() == N_localF_;
	}
	if (!mpiContext_->AllTrue(localOk))
	{
		throw std::runtime_error("Fluid solver synchronization field size does not match the local particle count.");
	}

	// pack and exchange field values with every neighboring rank
	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_StateSync);
	requests_.clear();
	for (PeerPlan& peerPlan : peerPlans_)
	{
		// pack owned values in field-major order for each peer
		peerPlan.sendBuffer.resize(
			static_cast<uint64_t>(N_fields) * peerPlan.sendLocalIdxs.size());
		peerPlan.recvBuffer.resize(
			static_cast<uint64_t>(N_fields) * peerPlan.recvLocalIdxs.size());

		for (uint32_t fieldIdx = 0; fieldIdx != N_fields; fieldIdx++)
		{
			const uint64_t fieldOffset =
				static_cast<uint64_t>(fieldIdx) * peerPlan.sendLocalIdxs.size();
			for (uint32_t particleIdx = 0; particleIdx != peerPlan.sendLocalIdxs.size(); particleIdx++)
			{
				peerPlan.sendBuffer[fieldOffset + particleIdx] =
					fields[fieldIdx][peerPlan.sendLocalIdxs[particleIdx]];
			}
		}
	}

	// receives are posted before sends on every rank
	for (PeerPlan& peerPlan : peerPlans_)
	{
		mpiContext_->IRecvBytesChunked(
			peerPlan.recvBuffer.data(),
			static_cast<uint64_t>(peerPlan.recvBuffer.size()) * sizeof(float),
			peerPlan.rank,
			FLUID_SOLVER_SYNC_TAG_PAYLOAD,
			requests_);
	}
	for (const PeerPlan& peerPlan : peerPlans_)
	{
		mpiContext_->ISendBytesChunked(
			peerPlan.sendBuffer.data(),
			static_cast<uint64_t>(peerPlan.sendBuffer.size()) * sizeof(float),
			peerPlan.rank,
			FLUID_SOLVER_SYNC_TAG_PAYLOAD,
			requests_);
	}
	mpiContext_->WaitAll(requests_);

	// unpack halo values
	for (const PeerPlan& peerPlan : peerPlans_)
	{
		for (uint32_t fieldIdx = 0; fieldIdx != N_fields; fieldIdx++)
		{
			const uint64_t fieldOffset =
				static_cast<uint64_t>(fieldIdx) * peerPlan.recvLocalIdxs.size();
			for (uint32_t particleIdx = 0; particleIdx != peerPlan.recvLocalIdxs.size(); particleIdx++)
			{
				fields[fieldIdx][peerPlan.recvLocalIdxs[particleIdx]] =
					peerPlan.recvBuffer[fieldOffset + particleIdx];
			}
		}
	}
	benchmarksTracker_->StopBench(Bench::BenchOp::Dist_StateSync);
}

Sim::PressureSolveErrorSums FluidSolverSyncMgr::GlobalizePressureSolveErrorSums(
	const Sim::PressureSolveErrorSums& localSums) const
{
	if (!mpiContext_)
	{
		throw std::runtime_error("Fluid solver synchronization plan is not initialized.");
	}

	// reduce raw, positive, and absolute errors
	std::array<double, 3> sums{ localSums.raw, localSums.pos, localSums.abs };
	mpiContext_->AllReduceSumDoubles(sums);
	return Sim::PressureSolveErrorSums
	{
		.raw = sums[0],
		.pos = sums[1],
		.abs = sums[2]
	};
}

Mem::MemUsage FluidSolverSyncMgr::GetExchangeBufferMemoryUsage() const
{
	// exchange payload buffers and MPI requests
	Mem::MemUsage usage{};

	for (const PeerPlan& peerPlan : peerPlans_)
	{
		Mem::Add(usage, Mem::MemUsageOf_Vector(peerPlan.sendBuffer));
		Mem::Add(usage, Mem::MemUsageOf_Vector(peerPlan.recvBuffer));
	}
	Mem::Add(usage, Mem::MemUsageOf_Vector(requests_));

	return usage;
}

Mem::MemUsage FluidSolverSyncMgr::GetPlanMemoryUsage() const
{
	// peer plans and their local/global index lists
	Mem::MemUsage usage = Mem::MemUsageOf_Vector(peerPlans_);

	for (const PeerPlan& peerPlan : peerPlans_)
	{
		Mem::Add(usage, Mem::MemUsageOf_Vector(peerPlan.sendLocalIdxs));
		Mem::Add(usage, Mem::MemUsageOf_Vector(peerPlan.recvLocalIdxs));
		Mem::Add(usage, Mem::MemUsageOf_Vector(peerPlan.sendGlobalIdxs));
		Mem::Add(usage, Mem::MemUsageOf_Vector(peerPlan.recvGlobalIdxs));
	}

	return usage;
}
