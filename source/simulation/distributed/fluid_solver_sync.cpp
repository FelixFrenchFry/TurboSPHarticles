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
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// internal
#include "source/base/parallel_checks.h"
#include "fluid_solver_sync.h"



using namespace Dist;

namespace
{
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
		std::vector<uint32_t>& globalIdxs,
		std::vector<std::pair<uint32_t, uint32_t>>& idxPairs)
	{
		idxPairs.clear();
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
		return Check::CountAdjacentViolations(
			0,
			static_cast<uint32_t>(globalIdxs.size()),
			[&](const uint32_t previousIdx, const uint32_t currentIdx)
			{
				return globalIdxs[previousIdx] == globalIdxs[currentIdx];
			}) != 0;
	}

} // namespace anonymous

void FluidSolverSyncMgr::AppendFieldSyncWorkItems(
	std::vector<FieldSyncWorkItem>& workItems,
	const uint32_t peerIdx,
	const uint32_t N_particles)
{
	for (uint32_t beginParticleIdx = 0; beginParticleIdx < N_particles; beginParticleIdx += FLUID_SOLVER_SYNC_BLOCK_SIZE)
	{
		workItems.push_back(FieldSyncWorkItem
		{
			.peerIdx = peerIdx,
			.beginParticleIdx = beginParticleIdx,
			.endParticleIdx = std::min(
				beginParticleIdx + FLUID_SOLVER_SYNC_BLOCK_SIZE, N_particles)
		});
	}
}

void FluidSolverSyncMgr::BuildFieldSyncSchedules()
{
	for (uint32_t haloExtentIdx = 0; haloExtentIdx != N_HALO_EXTENTS; haloExtentIdx++)
	{
		FieldSyncSchedule& schedule = fieldSyncSchedules_[haloExtentIdx];
		schedule.sendWorkItems.clear();
		schedule.recvWorkItems.clear();

		for (uint32_t peerIdx = 0; peerIdx != peerPlans_.size(); peerIdx++)
		{
			const ParticlePlan& particlePlan =
				peerPlans_[peerIdx].particlePlans[haloExtentIdx];

			AppendFieldSyncWorkItems(
				schedule.sendWorkItems, peerIdx,
				static_cast<uint32_t>(particlePlan.sendLocalIdxs.size()));
			AppendFieldSyncWorkItems(
				schedule.recvWorkItems, peerIdx,
				static_cast<uint32_t>(particlePlan.recvLocalIdxs.size()));
		}
	}
}

void FluidSolverSyncMgr::ClearParticlePlan(ParticlePlan& particlePlan)
{
	particlePlan.sendLocalIdxs.clear();
	particlePlan.recvLocalIdxs.clear();
	particlePlan.sendGlobalIdxs.clear();
	particlePlan.recvGlobalIdxs.clear();
}

bool FluidSolverSyncMgr::Rebuild(
	const Sync::FluidConstView& particlesF,
	const DistParticleState& particleStateF,
	const IDomainDecomp& domainDecomp,
	const float singleHaloRadius,
	const Base::DistHaloMode distHaloMode,
	const uint64_t N_expectedGlobalF,
	MpiContext& mpiContext,
	Bench::BenchmarksTracker& benchmarksTracker,
	std::string* failMsg)
{
	std::string localFailMsg;
	bool localOk = true;

	// --- [ input validation ] ----------------------------------------------------------------------
	if (fluidFieldExchangePending_)
	{
		localOk = SetFailure(localFailMsg,
			"Cannot rebuild a fluid solver synchronization plan while an exchange is pending.");
	}

	// validate the global count, particle view, and rank-local metadata
	if (localOk && N_expectedGlobalF > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
	{
		localOk = SetFailure(localFailMsg,
			"Global fluid particle count exceeds the supported index range.");
	}
	const uint32_t N_globalF = static_cast<uint32_t>(N_expectedGlobalF);

	// validate every backend field before publishing the exchange plan
	const std::array<std::span<const float>, 16> fluidFields{
		particlesF.xPos, particlesF.yPos, particlesF.zPos,
		particlesF.xVel, particlesF.yVel, particlesF.zVel,
		particlesF.xVelPred, particlesF.yVelPred, particlesF.zVelPred,
		particlesF.xAccPress, particlesF.yAccPress, particlesF.zAccPress,
		particlesF.sourceTerm, particlesF.diagElem,
		particlesF.pressure, particlesF.density
	};
	bool fluidFieldsValid = particlesF.id.size() == particlesF.N_;
	for (const std::span<const float> field : fluidFields)
	{
		fluidFieldsValid = fluidFieldsValid && field.size() == particlesF.N_;
	}
	if (localOk && !fluidFieldsValid)
	{
		localOk = SetFailure(localFailMsg,
			"Fluid particle field view size does not match its particle count.");
	}

	if (localOk && !particleStateF.IsValidLocalParticleState(
		particlesF.id, N_globalF, &localFailMsg))
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

	// peer plans ordered by rank, looked up by binary search, refilled from the replaced plan
	std::vector<PeerPlan>& newPeerPlans = nextPeerPlans_;
	newPeerPlans.resize(sortedNeighborRanks.size());
	for (uint32_t peerIdx = 0; peerIdx != newPeerPlans.size(); peerIdx++)
	{
		newPeerPlans[peerIdx].rank = sortedNeighborRanks[peerIdx];
		for (ParticlePlan& particlePlan : newPeerPlans[peerIdx].particlePlans)
		{
			ClearParticlePlan(particlePlan);
		}
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

	// plan slots filled by the send and receive loops below
	constexpr uint8_t singleHaloIdx = Base::GetIdx(Base::DistHaloMode::SingleHalo);
	constexpr uint8_t doubleHaloIdx = Base::GetIdx(Base::DistHaloMode::DoubleHalo);

	const bool useDoubleHalo = distHaloMode == Base::DistHaloMode::DoubleHalo;
	const AABB singleHaloBounds =
		domainDecomp.OwnedExtentAABB(mpiContext.Rank()).Expanded(singleHaloRadius);

	// send owned particles to every neighboring rank whose selected halo extent contains them
	for (const uint32_t localIdx : particleStateF.particleLocalIdx_Owned)
	{
		const glm::vec3 position{
			particlesF.xPos[localIdx],
			particlesF.yPos[localIdx],
			particlesF.zPos[localIdx]
		};
		const uint32_t globalIdx =
			particlesF.id[localIdx];

		for (PeerPlan& peerPlan : newPeerPlans)
		{
			if (domainDecomp.OwnedExtentAABB(peerPlan.rank)
				.Expanded(singleHaloRadius).Contains(position))
			{
				peerPlan.particlePlans[singleHaloIdx].sendLocalIdxs.push_back(localIdx);
				peerPlan.particlePlans[singleHaloIdx].sendGlobalIdxs.push_back(globalIdx);
			}
			if (useDoubleHalo &&
				domainDecomp.HaloExtentAABB(peerPlan.rank).Contains(position))
			{
				peerPlan.particlePlans[doubleHaloIdx].sendLocalIdxs.push_back(localIdx);
				peerPlan.particlePlans[doubleHaloIdx].sendGlobalIdxs.push_back(globalIdx);
			}
		}
	}

	// receive halo particles from their owning rank over the matching single/double halo plans
	for (const uint32_t localIdx : particleStateF.particleLocalIdx_Halo)
	{
		const glm::vec3 position{
			particlesF.xPos[localIdx],
			particlesF.yPos[localIdx],
			particlesF.zPos[localIdx]
		};
		const int ownerRank = domainDecomp.OwnerRank(position);
		PeerPlan* peerPlan = findPeerPlan(ownerRank);
		if (!peerPlan)
		{
			localOk = SetFailure(localFailMsg,
				"Remote fluid particle owner is not a neighboring rank.");
			continue;
		}

		const uint32_t globalIdx =
			particlesF.id[localIdx];
		if (singleHaloBounds.Contains(position))
		{
			peerPlan->particlePlans[singleHaloIdx].recvLocalIdxs.push_back(localIdx);
			peerPlan->particlePlans[singleHaloIdx].recvGlobalIdxs.push_back(globalIdx);
		}
		else if (!useDoubleHalo)
		{
			localOk = SetFailure(localFailMsg,
				"Single-halo fluid particle lies outside the single halo extent.");
			continue;
		}

		if (useDoubleHalo)
		{
			peerPlan->particlePlans[doubleHaloIdx].recvLocalIdxs.push_back(localIdx);
			peerPlan->particlePlans[doubleHaloIdx].recvGlobalIdxs.push_back(globalIdx);
		}
	}

	// sort both plans by global ID and reject duplicate entries before exchanging them
	for (PeerPlan& peerPlan : newPeerPlans)
	{
		for (ParticlePlan& particlePlan : peerPlan.particlePlans)
		{
			SortPeerParticleIdxs(
				particlePlan.sendLocalIdxs, particlePlan.sendGlobalIdxs,
				peerIdxSortScratch_);
			SortPeerParticleIdxs(
				particlePlan.recvLocalIdxs, particlePlan.recvGlobalIdxs,
				peerIdxSortScratch_);
			if (HasDuplicateGlobalIdxs(particlePlan.sendGlobalIdxs) ||
				HasDuplicateGlobalIdxs(particlePlan.recvGlobalIdxs))
			{
				localOk = SetFailure(localFailMsg,
					"Fluid solver synchronization plan contains duplicate global indices.");
			}
		}
	}

	if (!FinishCollectiveValidation(
		mpiContext, localOk, localFailMsg, failMsg))
	{
		return false;
	}

	// full count and global-ID validation is retained for debug/validation builds;
	// local plan checks above remain active in optimized builds
#ifndef NDEBUG
	const auto ValidatePeerPlan = [&](const Base::DistHaloMode haloExtent) -> bool
	{
		// --- [ peer count exchange ] ---------------------------------------------------------------

		std::vector<int> sendCounts(newPeerPlans.size(), 0);
		std::vector<int> recvCounts(newPeerPlans.size(), 0);
		requests_.clear();
		requests_.reserve(newPeerPlans.size() * 2);
		for (uint32_t peerIdx = 0; peerIdx != newPeerPlans.size(); peerIdx++)
		{
			const ParticlePlan& particlePlan =
				newPeerPlans[peerIdx].particlePlans[Base::GetIdx(haloExtent)];
			const uint64_t N_send = particlePlan.sendGlobalIdxs.size();
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
			const ParticlePlan& particlePlan =
				newPeerPlans[peerIdx].particlePlans[Base::GetIdx(haloExtent)];
			if (recvCounts[peerIdx] < 0 ||
				static_cast<uint64_t>(recvCounts[peerIdx]) !=
				particlePlan.recvGlobalIdxs.size())
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

		// --- [ peer global-index exchange ] --------------------------------------------------------

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
			const ParticlePlan& particlePlan = peerPlan.particlePlans[Base::GetIdx(haloExtent)];
			mpiContext.ISendBytesChunked(
				particlePlan.sendGlobalIdxs.data(),
				static_cast<uint64_t>(particlePlan.sendGlobalIdxs.size()) * sizeof(uint32_t),
				peerPlan.rank,
				FLUID_SOLVER_SYNC_TAG_GLOBAL_IDXS,
				requests_);
		}
		mpiContext.WaitAll(requests_);

		for (uint32_t peerIdx = 0; peerIdx != newPeerPlans.size(); peerIdx++)
		{
			const ParticlePlan& particlePlan =
				newPeerPlans[peerIdx].particlePlans[Base::GetIdx(haloExtent)];
			if (receivedGlobalIdxs[peerIdx] != particlePlan.recvGlobalIdxs)
			{
				localOk = SetFailure(localFailMsg,
					"Fluid solver synchronization peer global indices do not match the local halo plan.");
			}
		}
		return FinishCollectiveValidation(
			mpiContext, localOk, localFailMsg, failMsg);
	};

	if (!ValidatePeerPlan(Base::DistHaloMode::SingleHalo) ||
		(useDoubleHalo && !ValidatePeerPlan(Base::DistHaloMode::DoubleHalo)))
	{
		return false;
	}
#endif

	// publish the locally validated plan after optional full peer-plan validation
	mpiContext_ = &mpiContext;
	benchmarksTracker_ = &benchmarksTracker;
	N_localF_ = particlesF.N_;
	N_globalF_ = N_expectedGlobalF;
	distHaloMode_ = distHaloMode;
	// the replaced plan is retained as storage space for the next rebuild
	peerPlans_.swap(nextPeerPlans_);
	BuildFieldSyncSchedules();
	return true;
}

void FluidSolverSyncMgr::ApplyFluidParticleReOrder(
	const std::span<const uint32_t> newFluidLocalIdx_FromOldLocalIdx)
{
	if (!mpiContext_)
	{
		throw std::runtime_error("Fluid solver synchronization plan is not initialized.");
	}
	if (fluidFieldExchangePending_)
	{
		throw std::runtime_error(
			"Cannot re-order a fluid solver synchronization plan while an exchange is pending.");
	}

	// the caller inverted and validated the storage permutation, so only its size is checked
	const bool storageWasReOrdered = !newFluidLocalIdx_FromOldLocalIdx.empty();
	const bool localOk = !storageWasReOrdered ||
		newFluidLocalIdx_FromOldLocalIdx.size() == N_localF_;

	if (!mpiContext_->AllTrue(localOk))
	{
		throw std::runtime_error("Invalid distributed fluid particle re-order.");
	}

	if (!storageWasReOrdered) { return; }

	// remap send and receive indices, global particle IDs remain unchanged
	for (PeerPlan& peerPlan : peerPlans_)
	{
		for (ParticlePlan& particlePlan : peerPlan.particlePlans)
		{
			for (uint32_t& localIdx : particlePlan.sendLocalIdxs)
			{
				localIdx = newFluidLocalIdx_FromOldLocalIdx[localIdx];
			}
			for (uint32_t& localIdx : particlePlan.recvLocalIdxs)
			{
				localIdx = newFluidLocalIdx_FromOldLocalIdx[localIdx];
			}
		}
	}
}

void FluidSolverSyncMgr::BeginSynchronizeFluidScalarField(
	const std::span<float> values,
	const Base::DistHaloMode haloExtent)
{
	BeginSynchronizeFluidFields({ values, {}, {} }, 1, haloExtent);
}

void FluidSolverSyncMgr::BeginSynchronizeFluidVectorField(
	const std::span<float> xValues,
	const std::span<float> yValues,
	const std::span<float> zValues,
	const Base::DistHaloMode haloExtent)
{
	BeginSynchronizeFluidFields({ xValues, yValues, zValues }, 3, haloExtent);
}

void FluidSolverSyncMgr::SynchronizeFluidScalarField(
	const std::span<float> values,
	const Base::DistHaloMode haloExtent)
{
	BeginSynchronizeFluidScalarField(values, haloExtent);
	EndSynchronizeFluidFields();
}

void FluidSolverSyncMgr::SynchronizeFluidVectorField(
	const std::span<float> xValues,
	const std::span<float> yValues,
	const std::span<float> zValues,
	const Base::DistHaloMode haloExtent)
{
	BeginSynchronizeFluidVectorField(xValues, yValues, zValues, haloExtent);
	EndSynchronizeFluidFields();
}

void FluidSolverSyncMgr::BeginSynchronizeFluidFields(
	const std::array<std::span<float>, 3>& fields,
	const uint32_t N_fields,
	const Base::DistHaloMode haloExtent)
{
	if (!mpiContext_ || !benchmarksTracker_)
	{
		throw std::runtime_error("Fluid solver synchronization plan is not initialized.");
	}
	if (fluidFieldExchangePending_)
	{
		throw std::runtime_error("A fluid solver synchronization exchange is already pending.");
	}

	bool localOk = N_fields > 0 && N_fields <= fields.size() &&
		(haloExtent == Base::DistHaloMode::SingleHalo ||
			(haloExtent == Base::DistHaloMode::DoubleHalo &&
				distHaloMode_ == Base::DistHaloMode::DoubleHalo));
	for (uint32_t fieldIdx = 0; fieldIdx != N_fields; fieldIdx++)
	{
		localOk = localOk && fields[fieldIdx].size() == N_localF_;
	}
	// field sizes were checked collectively during Rebuild(); reject malformed calls locally
	if (!localOk)
	{
		throw std::runtime_error("Fluid solver synchronization field size does not match the local particle count.");
	}

	// pack and exchange field values with every neighboring rank
	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_StateSync);
	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_SolverSync);
	requests_.clear();
	// size every peer payload before the parallel pass reads their storage
	for (PeerPlan& peerPlan : peerPlans_)
	{
		const ParticlePlan& particlePlan = peerPlan.particlePlans[Base::GetIdx(haloExtent)];

		// pack owned values in field-major order for each peer
		peerPlan.sendBuffer.resize(
			static_cast<uint64_t>(N_fields) * particlePlan.sendLocalIdxs.size());
		peerPlan.recvBuffer.resize(
			static_cast<uint64_t>(N_fields) * particlePlan.recvLocalIdxs.size());
	}

	// gather every field sequentially within each parallel work item
	// (work items span every peer, so no peer waits for another one to finish packing)
	const FieldSyncSchedule& schedule = fieldSyncSchedules_[Base::GetIdx(haloExtent)];
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(
			0, static_cast<uint32_t>(schedule.sendWorkItems.size())),
		[&](const tbb::blocked_range<uint32_t>& workRange)
	{
		for (uint32_t workIdx = workRange.begin(); workIdx != workRange.end(); workIdx++)
		{
			const FieldSyncWorkItem& workItem = schedule.sendWorkItems[workIdx];
			PeerPlan& peerPlan = peerPlans_[workItem.peerIdx];
			const ParticlePlan& particlePlan = peerPlan.particlePlans[Base::GetIdx(haloExtent)];
			const uint32_t N_send = static_cast<uint32_t>(particlePlan.sendLocalIdxs.size());

			for (uint32_t fieldIdx = 0; fieldIdx != N_fields; fieldIdx++)
			{
				const uint64_t fieldOffset = static_cast<uint64_t>(fieldIdx) * N_send;
				for (uint32_t particleIdx = workItem.beginParticleIdx;
					 particleIdx != workItem.endParticleIdx;
					 particleIdx++)
				{
					peerPlan.sendBuffer[fieldOffset + particleIdx] =
						fields[fieldIdx][particlePlan.sendLocalIdxs[particleIdx]];
				}
			}
		}
	});

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
	pendingFields_ = fields;
	N_pendingFields_ = N_fields;
	pendingHaloExtent_ = haloExtent;
	fluidFieldExchangePending_ = true;
	benchmarksTracker_->StopBench(Bench::BenchOp::Dist_SolverSync);
	benchmarksTracker_->StopBench(Bench::BenchOp::Dist_StateSync);
}

void FluidSolverSyncMgr::EndSynchronizeFluidFields()
{
	if (!mpiContext_ || !benchmarksTracker_)
	{
		throw std::runtime_error("Fluid solver synchronization plan is not initialized.");
	}
	if (!fluidFieldExchangePending_)
	{
		throw std::runtime_error("No fluid solver synchronization exchange is pending.");
	}

	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_StateSync);
	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_SolverSync);
	// time the peer payload wait separately from the local packing work
	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_SolverSyncWait);
	mpiContext_->WaitAll(requests_);
	benchmarksTracker_->StopBench(Bench::BenchOp::Dist_SolverSyncWait);

	// unpack halo values
	// (every halo slot is owned by exactly one peer, so work items write disjoint slots)
	const FieldSyncSchedule& schedule = fieldSyncSchedules_[Base::GetIdx(pendingHaloExtent_)];
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(
			0, static_cast<uint32_t>(schedule.recvWorkItems.size())),
		[&](const tbb::blocked_range<uint32_t>& workRange)
	{
		for (uint32_t workIdx = workRange.begin(); workIdx != workRange.end(); workIdx++)
		{
			const FieldSyncWorkItem& workItem = schedule.recvWorkItems[workIdx];
			const PeerPlan& peerPlan = peerPlans_[workItem.peerIdx];
			const ParticlePlan& particlePlan = peerPlan.particlePlans[Base::GetIdx(pendingHaloExtent_)];
			const uint32_t N_recv = static_cast<uint32_t>(particlePlan.recvLocalIdxs.size());

			// scatter every field sequentially within each parallel work item
			for (uint32_t fieldIdx = 0; fieldIdx != N_pendingFields_; fieldIdx++)
			{
				const uint64_t fieldOffset = static_cast<uint64_t>(fieldIdx) * N_recv;
				for (uint32_t particleIdx = workItem.beginParticleIdx;
					 particleIdx != workItem.endParticleIdx;
					 particleIdx++)
				{
					pendingFields_[fieldIdx][particlePlan.recvLocalIdxs[particleIdx]] =
						peerPlan.recvBuffer[fieldOffset + particleIdx];
				}
			}
		}
	});
	pendingFields_ = {};
	N_pendingFields_ = 0;
	fluidFieldExchangePending_ = false;
	benchmarksTracker_->StopBench(Bench::BenchOp::Dist_SolverSync);
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
	// exchange payload buffers and MPI requests of the published and recycled plans
	Mem::MemUsage usage{};

	for (const std::vector<PeerPlan>* peerPlans : { &peerPlans_, &nextPeerPlans_ })
	{
		for (const PeerPlan& peerPlan : *peerPlans)
		{
			Mem::Add(usage, Mem::MemUsageOf_Vector(peerPlan.sendBuffer));
			Mem::Add(usage, Mem::MemUsageOf_Vector(peerPlan.recvBuffer));
		}
	}
	Mem::Add(usage, Mem::MemUsageOf_Vector(requests_));

	return usage;
}

Mem::MemUsage FluidSolverSyncMgr::GetPlanMemoryUsage() const
{
	// published and recycled peer plans with their local/global index lists
	Mem::MemUsage usage = Mem::MemUsageOf_Vector(peerPlans_);

	Mem::Add(usage, Mem::MemUsageOf_Vector(nextPeerPlans_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(peerIdxSortScratch_));
	for (const FieldSyncSchedule& schedule : fieldSyncSchedules_)
	{
		Mem::Add(usage, Mem::MemUsageOf_Vector(schedule.sendWorkItems));
		Mem::Add(usage, Mem::MemUsageOf_Vector(schedule.recvWorkItems));
	}

	for (const std::vector<PeerPlan>* peerPlans : { &peerPlans_, &nextPeerPlans_ })
	{
		for (const PeerPlan& peerPlan : *peerPlans)
		{
			for (const ParticlePlan& particlePlan : peerPlan.particlePlans)
			{
				Mem::Add(usage, Mem::MemUsageOf_Vector(particlePlan.sendLocalIdxs));
				Mem::Add(usage, Mem::MemUsageOf_Vector(particlePlan.recvLocalIdxs));
				Mem::Add(usage, Mem::MemUsageOf_Vector(particlePlan.sendGlobalIdxs));
				Mem::Add(usage, Mem::MemUsageOf_Vector(particlePlan.recvGlobalIdxs));
			}
		}
	}

	return usage;
}
