// fluid_solver_sync.cpp

// external
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

// internal
#include "source/base/parallel_checks.h"
#include "parallel_utils.h"
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
		if (localIdxs.empty())
		{
			idxPairs.clear();
			return;
		}

		const uint32_t N_indices = static_cast<uint32_t>(localIdxs.size());
		idxPairs.resize(N_indices);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_indices),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t i = range.begin(); i != range.end(); i++)
			{
				idxPairs[i] = { globalIdxs[i], localIdxs[i] };
			}
		});

		tbb::parallel_sort(idxPairs.begin(), idxPairs.end());

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_indices),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t i = range.begin(); i != range.end(); i++)
			{
				globalIdxs[i] = idxPairs[i].first;
				localIdxs[i] = idxPairs[i].second;
			}
		});
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

	// peer plans ordered by rank, refilled from the replaced plan
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

	// plan buckets are indexed by compact peer index and halo extent
	constexpr uint32_t singleHaloIdx = Base::GetIdx(Base::DistHaloMode::SingleHalo);
	constexpr uint32_t doubleHaloIdx = Base::GetIdx(Base::DistHaloMode::DoubleHalo);

	const bool useDoubleHalo = distHaloMode == Base::DistHaloMode::DoubleHalo;
	const AABB singleHaloBounds =
		domainDecomp.OwnedExtentAABB(mpiContext.Rank()).Expanded(singleHaloRadius);
	const uint32_t N_peerPlans = static_cast<uint32_t>(newPeerPlans.size());
	const uint32_t N_buckets = N_peerPlans * N_HALO_EXTENTS;
	const uint32_t N_owned = static_cast<uint32_t>(
		particleStateF.particleLocalIdx_Owned.size());
	const uint32_t N_halo = static_cast<uint32_t>(
		particleStateF.particleLocalIdx_Halo.size());
	const uint32_t N_sendChunks = N_owned > 0
		? 1 + (N_owned - 1) / FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE
		: 0;
	const uint32_t N_recvChunks = N_halo > 0
		? 1 + (N_halo - 1) / FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE
		: 0;
	const uint64_t N_sendChunkBuckets =
		static_cast<uint64_t>(N_sendChunks) * N_buckets;
	const uint64_t N_recvChunkBuckets =
		static_cast<uint64_t>(N_recvChunks) * N_buckets;

	PlanBuildWorkBuffers& planBuildWorkBuffers = planBuildWorkBuffers_;
	planBuildWorkBuffers.sendCounts_ByChunkBucket.resize(N_sendChunkBuckets);
	planBuildWorkBuffers.sendWriteOffsets_ByChunkBucket.resize(N_sendChunkBuckets);
	planBuildWorkBuffers.sendTotals.resize(N_buckets);
	planBuildWorkBuffers.recvCounts_ByChunkBucket.resize(N_recvChunkBuckets);
	planBuildWorkBuffers.recvWriteOffsets_ByChunkBucket.resize(N_recvChunkBuckets);
	planBuildWorkBuffers.recvTotals.resize(N_buckets);
	planBuildWorkBuffers.recvPeerIdx_ByHaloIdx.resize(N_halo);
	std::fill(
		planBuildWorkBuffers.recvPeerIdx_ByHaloIdx.begin(),
		planBuildWorkBuffers.recvPeerIdx_ByHaloIdx.end(), -1);

	// build a compact rank-to-peer lookup for the receive pass
	planBuildWorkBuffers.peerIdx_ByRank.resize(
		static_cast<uint64_t>(mpiContext.Size()));
	std::fill(
		planBuildWorkBuffers.peerIdx_ByRank.begin(),
		planBuildWorkBuffers.peerIdx_ByRank.end(), -1);
	for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
	{
		planBuildWorkBuffers.peerIdx_ByRank[
			newPeerPlans[peerIdx].rank] = static_cast<int>(peerIdx);
	}

	// cache the bounds used by every owned-particle send test
	planBuildWorkBuffers.sendSingleBounds_ByPeer.resize(N_peerPlans);
	planBuildWorkBuffers.sendDoubleBounds_ByPeer.resize(N_peerPlans);
	for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
	{
		const int peerRank = newPeerPlans[peerIdx].rank;
		planBuildWorkBuffers.sendSingleBounds_ByPeer[peerIdx] =
			domainDecomp.OwnedExtentAABB(peerRank).Expanded(singleHaloRadius);
		if (useDoubleHalo)
		{
			planBuildWorkBuffers.sendDoubleBounds_ByPeer[peerIdx] =
				domainDecomp.HaloExtentAABB(peerRank);
		}
	}

	// first failure seen by the parallel plan construction passes
	std::atomic<const char*> planFailMsg{ nullptr };

	if (N_peerPlans != 0)
	{
		// count owned particles per peer and halo extent
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_sendChunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginOwnedIdx =
					chunkIdx * FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE;
				const uint32_t endOwnedIdx =
					beginOwnedIdx + std::min(
						FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE,
						N_owned - beginOwnedIdx);
				uint32_t* const chunkCounts =
					planBuildWorkBuffers.sendCounts_ByChunkBucket.data() +
					static_cast<uint64_t>(chunkIdx) * N_buckets;
				std::fill(chunkCounts, chunkCounts + N_buckets, uint32_t{ 0 });

				for (uint32_t ownedIdx = beginOwnedIdx;
					 ownedIdx != endOwnedIdx;
					 ownedIdx++)
				{
					const uint32_t localIdx =
						particleStateF.particleLocalIdx_Owned[ownedIdx];
					const glm::vec3 position{
						particlesF.xPos[localIdx],
						particlesF.yPos[localIdx],
						particlesF.zPos[localIdx]
					};

					for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
					{
						const uint32_t bucketBase = peerIdx * N_HALO_EXTENTS;
						if (planBuildWorkBuffers.sendSingleBounds_ByPeer[
							peerIdx].Contains(position))
						{
							chunkCounts[bucketBase + singleHaloIdx]++;
						}
						if (useDoubleHalo &&
							planBuildWorkBuffers.sendDoubleBounds_ByPeer[
								peerIdx].Contains(position))
						{
							chunkCounts[bucketBase + doubleHaloIdx]++;
						}
					}
				}
			}
		});

		PrefixCountsByChunkAndBucket(
			planBuildWorkBuffers.sendCounts_ByChunkBucket,
			planBuildWorkBuffers.sendWriteOffsets_ByChunkBucket,
			planBuildWorkBuffers.sendTotals,
			planBuildWorkBuffers.prefixBlockSumsScratch,
			N_sendChunks,
			N_buckets);

		for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
		{
			for (uint32_t haloExtentIdx = 0;
				haloExtentIdx != N_HALO_EXTENTS;
				haloExtentIdx++)
			{
				const uint32_t bucketIdx =
					peerIdx * N_HALO_EXTENTS + haloExtentIdx;
				ParticlePlan& particlePlan =
					newPeerPlans[peerIdx].particlePlans[haloExtentIdx];
				particlePlan.sendLocalIdxs.resize(
					planBuildWorkBuffers.sendTotals[bucketIdx]);
				particlePlan.sendGlobalIdxs.resize(
					planBuildWorkBuffers.sendTotals[bucketIdx]);
			}
		}

		// scatter owned particles into disjoint ranges, preserving input order within each bucket
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_sendChunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginOwnedIdx =
					chunkIdx * FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE;
				const uint32_t endOwnedIdx =
					beginOwnedIdx + std::min(
						FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE,
						N_owned - beginOwnedIdx);
				uint32_t* const chunkWriteIdxs =
					planBuildWorkBuffers.sendWriteOffsets_ByChunkBucket.data() +
					static_cast<uint64_t>(chunkIdx) * N_buckets;

				for (uint32_t ownedIdx = beginOwnedIdx;
					 ownedIdx != endOwnedIdx;
					 ownedIdx++)
				{
					const uint32_t localIdx =
						particleStateF.particleLocalIdx_Owned[ownedIdx];
					const uint32_t globalIdx = particlesF.id[localIdx];
					const glm::vec3 position{
						particlesF.xPos[localIdx],
						particlesF.yPos[localIdx],
						particlesF.zPos[localIdx]
					};

					for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
					{
						const uint32_t bucketBase = peerIdx * N_HALO_EXTENTS;
						if (planBuildWorkBuffers.sendSingleBounds_ByPeer[
							peerIdx].Contains(position))
						{
							const uint32_t writeIdx =
								chunkWriteIdxs[bucketBase + singleHaloIdx]++;
							ParticlePlan& particlePlan =
								newPeerPlans[peerIdx].particlePlans[singleHaloIdx];
							particlePlan.sendLocalIdxs[writeIdx] = localIdx;
							particlePlan.sendGlobalIdxs[writeIdx] = globalIdx;
						}
						if (useDoubleHalo &&
							planBuildWorkBuffers.sendDoubleBounds_ByPeer[
								peerIdx].Contains(position))
						{
							const uint32_t writeIdx =
								chunkWriteIdxs[bucketBase + doubleHaloIdx]++;
							ParticlePlan& particlePlan =
								newPeerPlans[peerIdx].particlePlans[doubleHaloIdx];
							particlePlan.sendLocalIdxs[writeIdx] = localIdx;
							particlePlan.sendGlobalIdxs[writeIdx] = globalIdx;
						}
					}
				}
			}
		});

		// count halo particles by their owning peer and halo extent
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_recvChunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginHaloIdx =
					chunkIdx * FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE;
				const uint32_t endHaloIdx =
					beginHaloIdx + std::min(
						FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE,
						N_halo - beginHaloIdx);
				uint32_t* const chunkCounts =
					planBuildWorkBuffers.recvCounts_ByChunkBucket.data() +
					static_cast<uint64_t>(chunkIdx) * N_buckets;
				std::fill(chunkCounts, chunkCounts + N_buckets, uint32_t{ 0 });

				for (uint32_t haloIdx = beginHaloIdx;
					 haloIdx != endHaloIdx;
					 haloIdx++)
				{
					const uint32_t localIdx =
						particleStateF.particleLocalIdx_Halo[haloIdx];
					const glm::vec3 position{
						particlesF.xPos[localIdx],
						particlesF.yPos[localIdx],
						particlesF.zPos[localIdx]
					};
					const int ownerRank = domainDecomp.OwnerRank(position);
					int peerIdx = -1;
					if (ownerRank >= 0 &&
						static_cast<uint64_t>(ownerRank) <
							planBuildWorkBuffers.peerIdx_ByRank.size())
					{
						peerIdx = planBuildWorkBuffers.peerIdx_ByRank[ownerRank];
					}
					if (peerIdx < 0)
					{
						planFailMsg.store(
							"Remote fluid particle owner is not a neighboring rank.",
							std::memory_order_relaxed);
						continue;
					}

					planBuildWorkBuffers.recvPeerIdx_ByHaloIdx[haloIdx] = peerIdx;
					const uint32_t bucketBase =
						static_cast<uint32_t>(peerIdx) * N_HALO_EXTENTS;
					if (singleHaloBounds.Contains(position))
					{
						chunkCounts[bucketBase + singleHaloIdx]++;
					}
					else if (!useDoubleHalo)
					{
						planFailMsg.store(
							"Single-halo fluid particle lies outside the single halo extent.",
							std::memory_order_relaxed);
						continue;
					}

					if (useDoubleHalo)
					{
						chunkCounts[bucketBase + doubleHaloIdx]++;
					}
				}
			}
		});

		PrefixCountsByChunkAndBucket(
			planBuildWorkBuffers.recvCounts_ByChunkBucket,
			planBuildWorkBuffers.recvWriteOffsets_ByChunkBucket,
			planBuildWorkBuffers.recvTotals,
			planBuildWorkBuffers.prefixBlockSumsScratch,
			N_recvChunks,
			N_buckets);

		for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
		{
			for (uint32_t haloExtentIdx = 0;
				haloExtentIdx != N_HALO_EXTENTS;
				haloExtentIdx++)
			{
				const uint32_t bucketIdx =
					peerIdx * N_HALO_EXTENTS + haloExtentIdx;
				ParticlePlan& particlePlan =
					newPeerPlans[peerIdx].particlePlans[haloExtentIdx];
				particlePlan.recvLocalIdxs.resize(
					planBuildWorkBuffers.recvTotals[bucketIdx]);
				particlePlan.recvGlobalIdxs.resize(
					planBuildWorkBuffers.recvTotals[bucketIdx]);
			}
		}

		// scatter halo particles into disjoint ranges, preserving input order within each bucket
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_recvChunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginHaloIdx =
					chunkIdx * FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE;
				const uint32_t endHaloIdx =
					beginHaloIdx + std::min(
						FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE,
						N_halo - beginHaloIdx);
				uint32_t* const chunkWriteIdxs =
					planBuildWorkBuffers.recvWriteOffsets_ByChunkBucket.data() +
					static_cast<uint64_t>(chunkIdx) * N_buckets;

				for (uint32_t haloIdx = beginHaloIdx;
					 haloIdx != endHaloIdx;
					 haloIdx++)
				{
					const int peerIdx =
						planBuildWorkBuffers.recvPeerIdx_ByHaloIdx[haloIdx];
					if (peerIdx < 0) { continue; }

					const uint32_t localIdx =
						particleStateF.particleLocalIdx_Halo[haloIdx];
					const uint32_t globalIdx = particlesF.id[localIdx];
					const glm::vec3 position{
						particlesF.xPos[localIdx],
						particlesF.yPos[localIdx],
						particlesF.zPos[localIdx]
					};
					const uint32_t bucketBase =
						static_cast<uint32_t>(peerIdx) * N_HALO_EXTENTS;

					if (singleHaloBounds.Contains(position))
					{
						const uint32_t writeIdx =
							chunkWriteIdxs[bucketBase + singleHaloIdx]++;
						ParticlePlan& particlePlan =
							newPeerPlans[peerIdx].particlePlans[singleHaloIdx];
						particlePlan.recvLocalIdxs[writeIdx] = localIdx;
						particlePlan.recvGlobalIdxs[writeIdx] = globalIdx;
					}
					else if (!useDoubleHalo)
					{
						continue;
					}

					if (useDoubleHalo)
					{
						const uint32_t writeIdx =
							chunkWriteIdxs[bucketBase + doubleHaloIdx]++;
						ParticlePlan& particlePlan =
							newPeerPlans[peerIdx].particlePlans[doubleHaloIdx];
						particlePlan.recvLocalIdxs[writeIdx] = localIdx;
						particlePlan.recvGlobalIdxs[writeIdx] = globalIdx;
					}
				}
			}
		});
	}
	else if (N_halo != 0)
	{
		localOk = SetFailure(localFailMsg,
			"Remote fluid particle owner is not a neighboring rank.");
	}

	if (const char* const planFail =
		planFailMsg.load(std::memory_order_relaxed))
	{
		localOk = SetFailure(localFailMsg, planFail);
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

	// peer counts validated in every build,
	// the more expensive global-ID exchange is for debug/validation builds
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
		mpiContext.WaitAll(requests_, "fluid solver peer particle counts");

		for (uint32_t peerIdx = 0; peerIdx != newPeerPlans.size(); peerIdx++)
		{
			const ParticlePlan& particlePlan =
				newPeerPlans[peerIdx].particlePlans[Base::GetIdx(haloExtent)];
			const uint64_t N_expectedRecv = particlePlan.recvGlobalIdxs.size();
			if (recvCounts[peerIdx] < 0 ||
				static_cast<uint64_t>(recvCounts[peerIdx]) !=
				N_expectedRecv)
			{
				localOk = SetFailure(localFailMsg,
					"Fluid solver synchronization peer particle count does not match the local halo plan (peer rank " +
					std::to_string(newPeerPlans[peerIdx].rank) + ", " +
					(haloExtent == Base::DistHaloMode::SingleHalo ? "single" : "double") +
					" halo: peer sent " + std::to_string(recvCounts[peerIdx]) +
					", local plan expects " + std::to_string(N_expectedRecv) + ").");
			}
		}
		if (!FinishCollectiveValidation(
			mpiContext, localOk, localFailMsg, failMsg))
		{
			return false;
		}

	#ifndef NDEBUG
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
		mpiContext.WaitAll(requests_, "fluid solver peer global IDs");

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
	#else
		return true;
	#endif
	};

	if (!ValidatePeerPlan(Base::DistHaloMode::SingleHalo) ||
		(useDoubleHalo && !ValidatePeerPlan(Base::DistHaloMode::DoubleHalo)))
	{
		return false;
	}

	// publish the locally validated plan after peer-plan validation
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

void FluidSolverSyncMgr::SynchronizeFluidParticleFlagsToHalos(
	const std::span<uint8_t> flags,
	const Base::DistHaloMode haloExtent)
{
	if (!mpiContext_)
	{
		throw std::runtime_error(
			"Fluid solver synchronization plan is not initialized.");
	}

	const bool validHaloExtent =
		haloExtent == Base::DistHaloMode::SingleHalo ||
		(haloExtent == Base::DistHaloMode::DoubleHalo &&
			distHaloMode_ == Base::DistHaloMode::DoubleHalo);
	const bool localOk = !fluidFieldExchangePending_ && validHaloExtent &&
		flags.size() == N_localF_;
	if (!mpiContext_->AllTrue(localOk))
	{
		if (fluidFieldExchangePending_)
		{
			throw std::runtime_error(
				"Cannot synchronize fluid particle flags while a fluid field exchange is pending.");
		}
		throw std::runtime_error(
			"Fluid particle flags do not match the local particle plan.");
	}

	const uint8_t haloExtentIdx = Base::GetIdx(haloExtent);
	requests_.clear();
	requests_.reserve(peerPlans_.size() * 2);

	// pack every owned flag before posting communication
	for (PeerPlan& peerPlan : peerPlans_)
	{
		const ParticlePlan& particlePlan = peerPlan.particlePlans[haloExtentIdx];
		peerPlan.sendByteBuffer.resize(particlePlan.sendLocalIdxs.size());
		peerPlan.recvByteBuffer.resize(particlePlan.recvLocalIdxs.size());

		for (uint32_t particleIdx = 0; particleIdx != particlePlan.sendLocalIdxs.size(); particleIdx++)
		{
			peerPlan.sendByteBuffer[particleIdx] =
				flags[particlePlan.sendLocalIdxs[particleIdx]];
		}
	}

	// receives are posted before sends on every rank
	for (PeerPlan& peerPlan : peerPlans_)
	{
		mpiContext_->IRecvBytesChunked(
			peerPlan.recvByteBuffer.data(),
			static_cast<uint64_t>(peerPlan.recvByteBuffer.size()),
			peerPlan.rank,
			FLUID_SOLVER_SYNC_TAG_BYTE_PAYLOAD,
			requests_);
	}
	for (const PeerPlan& peerPlan : peerPlans_)
	{
		mpiContext_->ISendBytesChunked(
			peerPlan.sendByteBuffer.data(),
			static_cast<uint64_t>(peerPlan.sendByteBuffer.size()),
			peerPlan.rank,
			FLUID_SOLVER_SYNC_TAG_BYTE_PAYLOAD,
			requests_);
	}
	mpiContext_->WaitAll(requests_, "fluid solver particle flag payloads");

	// overwrite only halo slots with their owner-held flags
	for (const PeerPlan& peerPlan : peerPlans_)
	{
		const ParticlePlan& particlePlan = peerPlan.particlePlans[haloExtentIdx];
		for (uint32_t particleIdx = 0; particleIdx != particlePlan.recvLocalIdxs.size(); particleIdx++)
		{
			flags[particlePlan.recvLocalIdxs[particleIdx]] =
				peerPlan.recvByteBuffer[particleIdx];
		}
	}
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

	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_HaloFieldExchange);
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
	benchmarksTracker_->StopBench(Bench::BenchOp::Dist_HaloFieldExchange);
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

	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_HaloFieldExchange);
	mpiContext_->WaitAll(requests_, "fluid solver field payloads");
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
	benchmarksTracker_->StopBench(Bench::BenchOp::Dist_HaloFieldExchange);
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
			Mem::Add(usage, Mem::MemUsageOf_Vector(peerPlan.sendByteBuffer));
			Mem::Add(usage, Mem::MemUsageOf_Vector(peerPlan.recvByteBuffer));
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
	Mem::Add(usage, Mem::MemUsageOf_Vector(
		planBuildWorkBuffers_.sendCounts_ByChunkBucket));
	Mem::Add(usage, Mem::MemUsageOf_Vector(
		planBuildWorkBuffers_.sendWriteOffsets_ByChunkBucket));
	Mem::Add(usage, Mem::MemUsageOf_Vector(planBuildWorkBuffers_.sendTotals));
	Mem::Add(usage, Mem::MemUsageOf_Vector(
		planBuildWorkBuffers_.recvCounts_ByChunkBucket));
	Mem::Add(usage, Mem::MemUsageOf_Vector(
		planBuildWorkBuffers_.recvWriteOffsets_ByChunkBucket));
	Mem::Add(usage, Mem::MemUsageOf_Vector(planBuildWorkBuffers_.recvTotals));
	Mem::Add(usage, Mem::MemUsageOf_Vector(planBuildWorkBuffers_.peerIdx_ByRank));
	Mem::Add(usage, Mem::MemUsageOf_Vector(
		planBuildWorkBuffers_.sendSingleBounds_ByPeer));
	Mem::Add(usage, Mem::MemUsageOf_Vector(
		planBuildWorkBuffers_.sendDoubleBounds_ByPeer));
	Mem::Add(usage, Mem::MemUsageOf_Vector(
		planBuildWorkBuffers_.recvPeerIdx_ByHaloIdx));
	Mem::Add(usage, Mem::MemUsageOf_Vector(
		planBuildWorkBuffers_.prefixBlockSumsScratch));
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
