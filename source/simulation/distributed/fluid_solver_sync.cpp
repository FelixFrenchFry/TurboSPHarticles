// fluid_solver_sync.cpp

// external
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
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

std::optional<FluidSolverSyncPayloadProtocol> Dist::GetFluidSolverSyncPayloadProtocol(
	const Sim::FluidSolverSyncPhase phase) noexcept
{
	switch (phase)
	{
	case Sim::FluidSolverSyncPhase::Density:
		return FluidSolverSyncPayloadProtocol{
			.tag = FLUID_SOLVER_SYNC_TAG_DENSITY,
			.N_bytesPerParticle = sizeof(float)
		};
	case Sim::FluidSolverSyncPhase::PredictedVelocity:
		return FluidSolverSyncPayloadProtocol{
			.tag = FLUID_SOLVER_SYNC_TAG_PREDICTED_VELOCITY,
			.N_bytesPerParticle = 3 * sizeof(float)
		};
	case Sim::FluidSolverSyncPhase::InitialPressure:
		return FluidSolverSyncPayloadProtocol{
			.tag = FLUID_SOLVER_SYNC_TAG_INITIAL_PRESSURE,
			.N_bytesPerParticle = sizeof(float)
		};
	case Sim::FluidSolverSyncPhase::PressureAcceleration:
		return FluidSolverSyncPayloadProtocol{
			.tag = FLUID_SOLVER_SYNC_TAG_PRESSURE_ACCELERATION,
			.N_bytesPerParticle = 3 * sizeof(float)
		};
	case Sim::FluidSolverSyncPhase::Pressure:
		return FluidSolverSyncPayloadProtocol{
			.tag = FLUID_SOLVER_SYNC_TAG_PRESSURE,
			.N_bytesPerParticle = sizeof(float)
		};
	case Sim::FluidSolverSyncPhase::PressureErrorReduction:
		return std::nullopt;
	case Sim::FluidSolverSyncPhase::ExportSurfaceFlags:
		return FluidSolverSyncPayloadProtocol{
			.tag = FLUID_SOLVER_SYNC_TAG_BYTE_PAYLOAD,
			.N_bytesPerParticle = sizeof(uint8_t)
		};
	}

	return std::nullopt;
}

namespace
{
	const char* FluidSolverSyncPhaseName(const Sim::FluidSolverSyncPhase phase)
	{
		switch (phase)
		{
		case Sim::FluidSolverSyncPhase::Density:
			return "density";
		case Sim::FluidSolverSyncPhase::PredictedVelocity:
			return "predicted_velocity";
		case Sim::FluidSolverSyncPhase::InitialPressure:
			return "initial_pressure";
		case Sim::FluidSolverSyncPhase::PressureAcceleration:
			return "pressure_acceleration";
		case Sim::FluidSolverSyncPhase::Pressure:
			return "pressure";
		case Sim::FluidSolverSyncPhase::PressureErrorReduction:
			return "pressure_error_reduction";
		case Sim::FluidSolverSyncPhase::ExportSurfaceFlags:
			return "export_surface_flags";
		}

		return "unknown";
	}

	std::string FluidSolverSyncContext(const Sim::FluidSolverSyncPhase phase)
	{
		return "fluid solver synchronization (phase=" +
			std::string(FluidSolverSyncPhaseName(phase)) + ")";
	}

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

	// one peer slice of a peer-major plan list
	template <typename Values>
	auto PeerSlice(Values& values, const std::vector<uint64_t>& offsets_ByPeer, const uint32_t peerIdx)
	{
		return std::span(values).subspan(
			offsets_ByPeer[peerIdx], offsets_ByPeer[peerIdx + 1] - offsets_ByPeer[peerIdx]);
	}

	// sort local indices into the global ID order shared by both peer plans
	void SortPeerParticleIdxs(
		const std::span<uint32_t> localIdxs,
		const std::span<uint64_t> particleIds,
		std::vector<std::pair<uint64_t, uint32_t>>& idxPairs)
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
				idxPairs[i] = { particleIds[i], localIdxs[i] };
			}
		});

		tbb::parallel_sort(idxPairs.begin(), idxPairs.end());

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_indices),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t i = range.begin(); i != range.end(); i++)
			{
				particleIds[i] = idxPairs[i].first;
				localIdxs[i] = idxPairs[i].second;
			}
		});
	}

	// duplicate global IDs indicate an invalid one-to-one exchange plan
	bool HasDuplicateParticleIds(const std::span<const uint64_t> particleIds)
	{
		return Check::CountAdjacentViolations(
			0,
			static_cast<uint32_t>(particleIds.size()),
			[&](const uint32_t previousIdx, const uint32_t currentIdx)
			{
				return particleIds[previousIdx] == particleIds[currentIdx];
			}) != 0;
	}

	using FluidSolverPlanFingerprint = std::array<uint64_t, 2>;

	struct FluidSolverPeerPlanSummary
	{
		uint64_t N_particles = 0;
		FluidSolverPlanFingerprint fingerprint{};
	};

	// compact, deterministic fingerprint of an ordered global-particle-ID plan
	FluidSolverPlanFingerprint BuildFluidSolverPlanFingerprint(
		const std::span<const uint64_t> particleIds) noexcept
	{
		// two independent 64-bit streams over a canonical little-endian encoding
		uint64_t first = 14695981039346656037ull;
		uint64_t second = 7809847782465536322ull;
		for (const uint64_t particleId : particleIds)
		{
			for (uint32_t byteIdx = 0; byteIdx != sizeof(particleId); byteIdx++)
			{
				const uint8_t byte = static_cast<uint8_t>(particleId >> (byteIdx * 8));
				first = (first ^ byte) * 1099511628211ull;
				second = (second ^ byte) * 14029467366897019727ull;
			}
		}

		first = (first ^ particleIds.size()) * 1099511628211ull;
		second = (second ^ particleIds.size()) * 14029467366897019727ull;
		return { first, second };
	}

} // namespace anonymous

bool FluidSolverSyncMgr::Rebuild(
	const Sync::FluidConstView& particlesF,
	const DistParticleState& particleStateF,
	const std::span<const uint32_t> outerLocalIdxsF,
	const IDomainDecomp& domainDecomp,
	const float singleHaloRadius,
	const Base::DistHaloMode distHaloMode,
	const uint64_t N_expectedGlobalF,
	MpiContext& mpiContext,
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
		particlesF.id, N_expectedGlobalF, &localFailMsg))
	{
		localOk = false;
	}
	if (localOk && outerLocalIdxsF.size() != particleStateF.N_outer)
	{
		localOk = SetFailure(localFailMsg,
			"Outer fluid particle list does not match the outer particle count.");
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

	// plans are rebuilt in place, a failed rebuild stops the run
	peerRanks_ = std::move(sortedNeighborRanks);
	for (ParticlePlan& particlePlan : particlePlans_)
	{
		particlePlan.localIdxsPublished = false;
		particlePlan.sendLocalIdxs.clear();
		particlePlan.recvLocalIdxs.clear();
		particlePlan.sendParticleIds.clear();
		particlePlan.recvParticleIds.clear();
		particlePlan.sendOffsets_ByPeer.assign(peerRanks_.size() + 1, 0);
		particlePlan.recvOffsets_ByPeer.assign(peerRanks_.size() + 1, 0);
	}

	// plan buckets are indexed by compact peer index and halo extent
	constexpr uint32_t singleHaloIdx = Base::GetIdx(Base::DistHaloMode::SingleHalo);
	constexpr uint32_t doubleHaloIdx = Base::GetIdx(Base::DistHaloMode::DoubleHalo);

	const bool useDoubleHalo = distHaloMode == Base::DistHaloMode::DoubleHalo;
	const AABB singleHaloBounds =
		domainDecomp.OwnedExtentAABB(mpiContext.Rank())
			.ExtendedAtGlobalBounds(domainDecomp.GlobalExtentAABB()).Expanded(singleHaloRadius);
	const uint32_t N_peerPlans = static_cast<uint32_t>(peerRanks_.size());
	const uint32_t N_buckets = N_peerPlans * N_HALO_EXTENTS;
	const uint32_t N_outer = static_cast<uint32_t>(outerLocalIdxsF.size());
	const uint32_t N_owned = particleStateF.N_owned;
	const uint32_t N_halo = particleStateF.N_halo;
	const uint32_t N_sendChunks = N_outer > 0
		? 1 + (N_outer - 1) / FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE
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
			peerRanks_[peerIdx]] = static_cast<int>(peerIdx);
	}

	// cache the bounds used by every send test
	planBuildWorkBuffers.sendSingleBounds_ByPeer.resize(N_peerPlans);
	planBuildWorkBuffers.sendDoubleBounds_ByPeer.resize(N_peerPlans);
	for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
	{
		const int peerRank = peerRanks_[peerIdx];
		planBuildWorkBuffers.sendSingleBounds_ByPeer[peerIdx] =
			domainDecomp.OwnedExtentAABB(peerRank)
				.ExtendedAtGlobalBounds(domainDecomp.GlobalExtentAABB()).Expanded(singleHaloRadius);
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
		// count outer owned particles per peer and halo extent
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_sendChunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginOuterIdx =
					chunkIdx * FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE;
				const uint32_t endOuterIdx =
					beginOuterIdx + std::min(
						FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE,
						N_outer - beginOuterIdx);
				uint32_t* const chunkCounts =
					planBuildWorkBuffers.sendCounts_ByChunkBucket.data() +
					static_cast<uint64_t>(chunkIdx) * N_buckets;
				std::fill(chunkCounts, chunkCounts + N_buckets, uint32_t{ 0 });

				for (uint32_t outerIdx = beginOuterIdx;
					 outerIdx != endOuterIdx;
					 outerIdx++)
				{
					const uint32_t localIdx = outerLocalIdxsF[outerIdx];
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

		PrefixCountsByChunkAndBucket_ParallelScan(
			planBuildWorkBuffers.sendCounts_ByChunkBucket,
			planBuildWorkBuffers.sendWriteOffsets_ByChunkBucket,
			planBuildWorkBuffers.sendTotals,
			N_sendChunks,
			N_buckets);

		// peer slices follow each other in peer order within each halo extent
		for (uint32_t haloExtentIdx = 0; haloExtentIdx != N_HALO_EXTENTS; haloExtentIdx++)
		{
			ParticlePlan& particlePlan = particlePlans_[haloExtentIdx];
			for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
			{
				particlePlan.sendOffsets_ByPeer[peerIdx + 1] = particlePlan.sendOffsets_ByPeer[peerIdx] +
					planBuildWorkBuffers.sendTotals[peerIdx * N_HALO_EXTENTS + haloExtentIdx];
			}
			particlePlan.sendLocalIdxs.resize(particlePlan.sendOffsets_ByPeer.back());
			particlePlan.sendParticleIds.resize(particlePlan.sendOffsets_ByPeer.back());
		}

		// scatter outer owned particles into disjoint ranges, preserving input order within each bucket
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_sendChunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginOuterIdx =
					chunkIdx * FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE;
				const uint32_t endOuterIdx =
					beginOuterIdx + std::min(
						FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE,
						N_outer - beginOuterIdx);
				uint32_t* const chunkWriteIdxs =
					planBuildWorkBuffers.sendWriteOffsets_ByChunkBucket.data() +
					static_cast<uint64_t>(chunkIdx) * N_buckets;

				for (uint32_t outerIdx = beginOuterIdx;
					 outerIdx != endOuterIdx;
					 outerIdx++)
				{
					const uint32_t localIdx = outerLocalIdxsF[outerIdx];
					const uint64_t particleId = particlesF.id[localIdx];
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
							ParticlePlan& particlePlan = particlePlans_[singleHaloIdx];
							const uint64_t writeIdx = particlePlan.sendOffsets_ByPeer[peerIdx] +
								chunkWriteIdxs[bucketBase + singleHaloIdx]++;
							particlePlan.sendLocalIdxs[writeIdx] = localIdx;
							particlePlan.sendParticleIds[writeIdx] = particleId;
						}
						if (useDoubleHalo &&
							planBuildWorkBuffers.sendDoubleBounds_ByPeer[
								peerIdx].Contains(position))
						{
							ParticlePlan& particlePlan = particlePlans_[doubleHaloIdx];
							const uint64_t writeIdx = particlePlan.sendOffsets_ByPeer[peerIdx] +
								chunkWriteIdxs[bucketBase + doubleHaloIdx]++;
							particlePlan.sendLocalIdxs[writeIdx] = localIdx;
							particlePlan.sendParticleIds[writeIdx] = particleId;
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
					const uint32_t localIdx = N_owned + haloIdx;
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

		PrefixCountsByChunkAndBucket_ParallelScan(
			planBuildWorkBuffers.recvCounts_ByChunkBucket,
			planBuildWorkBuffers.recvWriteOffsets_ByChunkBucket,
			planBuildWorkBuffers.recvTotals,
			N_recvChunks,
			N_buckets);

		for (uint32_t haloExtentIdx = 0; haloExtentIdx != N_HALO_EXTENTS; haloExtentIdx++)
		{
			ParticlePlan& particlePlan = particlePlans_[haloExtentIdx];
			for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
			{
				particlePlan.recvOffsets_ByPeer[peerIdx + 1] = particlePlan.recvOffsets_ByPeer[peerIdx] +
					planBuildWorkBuffers.recvTotals[peerIdx * N_HALO_EXTENTS + haloExtentIdx];
			}
			particlePlan.recvLocalIdxs.resize(particlePlan.recvOffsets_ByPeer.back());
			particlePlan.recvParticleIds.resize(particlePlan.recvOffsets_ByPeer.back());
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

					const uint32_t localIdx = N_owned + haloIdx;
					const uint64_t particleId = particlesF.id[localIdx];
					const glm::vec3 position{
						particlesF.xPos[localIdx],
						particlesF.yPos[localIdx],
						particlesF.zPos[localIdx]
					};
					const uint32_t bucketBase =
						static_cast<uint32_t>(peerIdx) * N_HALO_EXTENTS;

					if (singleHaloBounds.Contains(position))
					{
						ParticlePlan& particlePlan = particlePlans_[singleHaloIdx];
						const uint64_t writeIdx = particlePlan.recvOffsets_ByPeer[peerIdx] +
							chunkWriteIdxs[bucketBase + singleHaloIdx]++;
						particlePlan.recvLocalIdxs[writeIdx] = localIdx;
						particlePlan.recvParticleIds[writeIdx] = particleId;
					}
					else if (!useDoubleHalo)
					{
						continue;
					}

					if (useDoubleHalo)
					{
						ParticlePlan& particlePlan = particlePlans_[doubleHaloIdx];
						const uint64_t writeIdx = particlePlan.recvOffsets_ByPeer[peerIdx] +
							chunkWriteIdxs[bucketBase + doubleHaloIdx]++;
						particlePlan.recvLocalIdxs[writeIdx] = localIdx;
						particlePlan.recvParticleIds[writeIdx] = particleId;
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
	for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
	{
		for (ParticlePlan& particlePlan : particlePlans_)
		{
			const std::span<uint64_t> sendParticleIds =
				PeerSlice(particlePlan.sendParticleIds, particlePlan.sendOffsets_ByPeer, peerIdx);
			const std::span<uint64_t> recvParticleIds =
				PeerSlice(particlePlan.recvParticleIds, particlePlan.recvOffsets_ByPeer, peerIdx);
			SortPeerParticleIdxs(
				PeerSlice(particlePlan.sendLocalIdxs, particlePlan.sendOffsets_ByPeer, peerIdx),
				sendParticleIds, peerIdxSortScratch_);
			SortPeerParticleIdxs(
				PeerSlice(particlePlan.recvLocalIdxs, particlePlan.recvOffsets_ByPeer, peerIdx),
				recvParticleIds, peerIdxSortScratch_);
			if (HasDuplicateParticleIds(sendParticleIds) ||
				HasDuplicateParticleIds(recvParticleIds))
			{
				localOk = SetFailure(localFailMsg,
					"Fluid solver synchronization plan contains duplicate particle IDs.");
			}
		}
	}

	if (!FinishCollectiveValidation(
		mpiContext, localOk, localFailMsg, failMsg))
	{
		return false;
	}

	// --- [ peer count and ordered-ID fingerprint exchange ] ----------------------------------------
	// peer counts and ordered-ID fingerprints are validated in every build,
	// the full global-ID exchange remains available in debug/validation builds

	const uint32_t N_activeHaloExtents = useDoubleHalo ? N_HALO_EXTENTS : 1;
	const uint64_t N_summaryBytes = N_activeHaloExtents * sizeof(FluidSolverPeerPlanSummary);
	std::vector<std::array<FluidSolverPeerPlanSummary, N_HALO_EXTENTS>> sendSummaries(N_peerPlans);
	std::vector<std::array<FluidSolverPeerPlanSummary, N_HALO_EXTENTS>> recvSummaries(N_peerPlans);
	requests_.clear();
	requestInfos_.clear();
	requests_.reserve(N_peerPlans * 2);
	requestInfos_.reserve(N_peerPlans * 2);
	for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
	{
		for (uint32_t haloExtentIdx = 0; haloExtentIdx != N_activeHaloExtents; haloExtentIdx++)
		{
			const ParticlePlan& particlePlan = particlePlans_[haloExtentIdx];
			const std::span<const uint64_t> sendParticleIds =
				PeerSlice(particlePlan.sendParticleIds, particlePlan.sendOffsets_ByPeer, peerIdx);
			sendSummaries[peerIdx][haloExtentIdx] = {
				.N_particles = sendParticleIds.size(),
				.fingerprint = BuildFluidSolverPlanFingerprint(sendParticleIds)
			};
		}
	}

	for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
	{
		mpiContext.IRecvBytesChunked(
			recvSummaries[peerIdx].data(),
			N_summaryBytes,
			peerRanks_[peerIdx],
			FLUID_SOLVER_SYNC_TAG_PLAN_SUMMARY,
			requests_,
			&requestInfos_);
	}
	for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
	{
		mpiContext.ISendBytesChunked(
			sendSummaries[peerIdx].data(),
			N_summaryBytes,
			peerRanks_[peerIdx],
			FLUID_SOLVER_SYNC_TAG_PLAN_SUMMARY,
			requests_,
			&requestInfos_);
	}
	mpiContext.WaitAll(
		requests_, "fluid solver peer plan summaries", &requestInfos_);

	for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
	{
		for (uint32_t haloExtentIdx = 0; haloExtentIdx != N_activeHaloExtents; haloExtentIdx++)
		{
			const ParticlePlan& particlePlan = particlePlans_[haloExtentIdx];
			const std::span<const uint64_t> recvParticleIds =
				PeerSlice(particlePlan.recvParticleIds, particlePlan.recvOffsets_ByPeer, peerIdx);
			const uint64_t N_expectedRecv = recvParticleIds.size();
			if (recvSummaries[peerIdx][haloExtentIdx].N_particles != N_expectedRecv)
			{
				localOk = SetFailure(localFailMsg,
					"Fluid solver synchronization peer particle count does not match the local halo plan (peer rank " +
					std::to_string(peerRanks_[peerIdx]) + ", " +
					(haloExtentIdx == singleHaloIdx ? "single" : "double") +
					" halo: peer sent " + std::to_string(recvSummaries[peerIdx][haloExtentIdx].N_particles) +
					", local plan expects " + std::to_string(N_expectedRecv) + ").");
			}
			else if (recvSummaries[peerIdx][haloExtentIdx].fingerprint !=
				BuildFluidSolverPlanFingerprint(recvParticleIds))
			{
				localOk = SetFailure(localFailMsg,
					"Fluid solver synchronization peer plan fingerprint does not match "
					"the local halo plan (peer rank " +
					std::to_string(peerRanks_[peerIdx]) + ", " +
					(haloExtentIdx == singleHaloIdx ? "single" : "double") +
					" halo).");
			}
		}
	}
	if (!FinishCollectiveValidation(
		mpiContext, localOk, localFailMsg, failMsg))
	{
		return false;
	}

	#ifndef NDEBUG
	const auto ValidatePeerParticleIds = [&](const Base::DistHaloMode haloExtent) -> bool
	{
		// --- [ peer global-index exchange ] --------------------------------------------------------

		const ParticlePlan& particlePlan = particlePlans_[Base::GetIdx(haloExtent)];
		std::vector<std::vector<uint64_t>> receivedParticleIds(N_peerPlans);
		requests_.clear();
		requestInfos_.clear();
		for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
		{
			receivedParticleIds[peerIdx].resize(
				PeerSlice(particlePlan.recvParticleIds, particlePlan.recvOffsets_ByPeer, peerIdx).size());
			mpiContext.IRecvBytesChunked(
				receivedParticleIds[peerIdx].data(),
				Sim::CheckedParticleMultiply(receivedParticleIds[peerIdx].size(),
					sizeof(uint64_t), "fluid solver received global ID payload"),
				peerRanks_[peerIdx],
				FLUID_SOLVER_SYNC_TAG_GLOBAL_IDXS,
				requests_,
				&requestInfos_);
		}
		for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
		{
			const std::span<const uint64_t> sendParticleIds =
				PeerSlice(particlePlan.sendParticleIds, particlePlan.sendOffsets_ByPeer, peerIdx);
			mpiContext.ISendBytesChunked(
				sendParticleIds.data(),
				Sim::CheckedParticleMultiply(sendParticleIds.size(),
					sizeof(uint64_t), "fluid solver sent global ID payload"),
				peerRanks_[peerIdx],
				FLUID_SOLVER_SYNC_TAG_GLOBAL_IDXS,
				requests_,
				&requestInfos_);
		}
		mpiContext.WaitAll(
			requests_, "fluid solver peer global IDs", &requestInfos_);

		for (uint32_t peerIdx = 0; peerIdx != N_peerPlans; peerIdx++)
		{
			if (!std::ranges::equal(receivedParticleIds[peerIdx],
				PeerSlice(particlePlan.recvParticleIds, particlePlan.recvOffsets_ByPeer, peerIdx)))
			{
				localOk = SetFailure(localFailMsg,
					"Fluid solver synchronization peer particle IDs do not match the local halo plan.");
			}
		}
		return FinishCollectiveValidation(
			mpiContext, localOk, localFailMsg, failMsg);
	};

	if (!ValidatePeerParticleIds(Base::DistHaloMode::SingleHalo) ||
		(useDoubleHalo && !ValidatePeerParticleIds(Base::DistHaloMode::DoubleHalo)))
	{
		return false;
	}
	#endif

	// publish the locally validated plan after peer-plan validation
	mpiContext_ = &mpiContext;
	N_globalF_ = N_expectedGlobalF;
	distHaloMode_ = distHaloMode;
	for (ParticlePlan& particlePlan : particlePlans_)
	{
		particlePlan.localIdxsPublished = true;
	}
	return true;
}

void FluidSolverSyncMgr::SwapFluidSyncLocalIdxs(
	const Base::DistHaloMode haloExtent,
	std::vector<uint32_t>& sendLocalIdxs,
	std::vector<uint32_t>& recvLocalIdxs)
{
	if (haloExtent != Base::DistHaloMode::SingleHalo &&
		haloExtent != Base::DistHaloMode::DoubleHalo)
	{
		throw std::runtime_error("Invalid fluid solver synchronization halo extent.");
	}
	ParticlePlan& particlePlan = particlePlans_[Base::GetIdx(haloExtent)];
	if (!particlePlan.localIdxsPublished)
	{
		throw std::runtime_error(
			"Fluid solver synchronization local indices were already handed over or never built.");
	}

	// the swapped-in storage is reused by the next rebuild
	particlePlan.sendLocalIdxs.swap(sendLocalIdxs);
	particlePlan.recvLocalIdxs.swap(recvLocalIdxs);
	particlePlan.localIdxsPublished = false;
}

void FluidSolverSyncMgr::BeginFluidHaloExchange(
	const std::span<const std::byte> sendPayload,
	const std::span<std::byte> recvPayload,
	const Base::DistHaloMode haloExtent,
	const Sim::FluidSolverSyncPhase phase)
{
	if (!mpiContext_)
	{
		throw std::runtime_error("Fluid solver synchronization plan is not initialized.");
	}
	if (fluidFieldExchangePending_)
	{
		throw std::runtime_error("A fluid solver synchronization exchange is already pending.");
	}
	const std::optional<FluidSolverSyncPayloadProtocol> protocol =
		GetFluidSolverSyncPayloadProtocol(phase);
	if (!protocol)
	{
		throw std::runtime_error(
			"Fluid solver synchronization phase does not carry a halo payload.");
	}

	const bool validHaloExtent =
		haloExtent == Base::DistHaloMode::SingleHalo ||
		(haloExtent == Base::DistHaloMode::DoubleHalo &&
			distHaloMode_ == Base::DistHaloMode::DoubleHalo);
	if (!validHaloExtent)
	{
		throw std::runtime_error("Invalid fluid solver synchronization halo extent.");
	}
	const ParticlePlan& particlePlan = particlePlans_[Base::GetIdx(haloExtent)];
	if (particlePlan.localIdxsPublished)
	{
		throw std::runtime_error(
			"The backend must take the fluid solver synchronization local indices before exchanging halos.");
	}
	const uint64_t N_bytesPerParticle = protocol->N_bytesPerParticle;
	// payload sizes follow from the published plan, reject malformed calls locally
	if (sendPayload.size() != N_bytesPerParticle * particlePlan.sendOffsets_ByPeer.back() ||
		recvPayload.size() != N_bytesPerParticle * particlePlan.recvOffsets_ByPeer.back())
	{
		throw std::runtime_error(
			"Fluid solver synchronization payload does not match the published plan.");
	}

	requests_.clear();
	requestInfos_.clear();
	// receives are posted before sends on every rank, each peer slice is one contiguous byte range
	for (uint32_t peerIdx = 0; peerIdx != peerRanks_.size(); peerIdx++)
	{
		const uint64_t beginRecvIdx = particlePlan.recvOffsets_ByPeer[peerIdx];
		const uint64_t N_recv = particlePlan.recvOffsets_ByPeer[peerIdx + 1] - beginRecvIdx;
		mpiContext_->IRecvBytesChunked(
			recvPayload.data() + N_bytesPerParticle * beginRecvIdx,
			N_bytesPerParticle * N_recv,
			peerRanks_[peerIdx],
			protocol->tag,
			requests_,
			&requestInfos_);
	}
	for (uint32_t peerIdx = 0; peerIdx != peerRanks_.size(); peerIdx++)
	{
		const uint64_t beginSendIdx = particlePlan.sendOffsets_ByPeer[peerIdx];
		const uint64_t N_send = particlePlan.sendOffsets_ByPeer[peerIdx + 1] - beginSendIdx;
		mpiContext_->ISendBytesChunked(
			sendPayload.data() + N_bytesPerParticle * beginSendIdx,
			N_bytesPerParticle * N_send,
			peerRanks_[peerIdx],
			protocol->tag,
			requests_,
			&requestInfos_);
	}
	pendingPhase_ = phase;
	fluidFieldExchangePending_ = true;
}

void FluidSolverSyncMgr::EndFluidHaloExchange()
{
	if (!mpiContext_)
	{
		throw std::runtime_error("Fluid solver synchronization plan is not initialized.");
	}
	if (!fluidFieldExchangePending_)
	{
		throw std::runtime_error("No fluid solver synchronization exchange is pending.");
	}

	const std::string waitContext = FluidSolverSyncContext(pendingPhase_);
	mpiContext_->WaitAll(requests_, waitContext.c_str(), &requestInfos_);
	fluidFieldExchangePending_ = false;
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
	const std::string waitContext = FluidSolverSyncContext(
		Sim::FluidSolverSyncPhase::PressureErrorReduction);
	mpiContext_->AllReduceSumDoubles(sums, waitContext.c_str());
	return Sim::PressureSolveErrorSums
	{
		.raw = sums[0],
		.pos = sums[1],
		.abs = sums[2]
	};
}

Mem::MemUsage FluidSolverSyncMgr::GetExchangeBufferMemoryUsage() const
{
	// MPI requests, the exchange payloads are owned by the backend
	Mem::MemUsage usage{};

	Mem::Add(usage, Mem::MemUsageOf_Vector(requests_));
	Mem::Add(usage, Mem::MemUsageOf_Vector(requestInfos_));

	return usage;
}

Mem::MemUsage FluidSolverSyncMgr::GetPlanMemoryUsage() const
{
	// peer plans with their local index lists and global IDs, and the plan build scratch
	Mem::MemUsage usage = Mem::MemUsageOf_Vector(peerRanks_);

	for (const ParticlePlan& particlePlan : particlePlans_)
	{
		Mem::Add(usage, Mem::MemUsageOf_Vector(particlePlan.sendLocalIdxs));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particlePlan.recvLocalIdxs));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particlePlan.sendParticleIds));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particlePlan.recvParticleIds));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particlePlan.sendOffsets_ByPeer));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particlePlan.recvOffsets_ByPeer));
	}
	Mem::Add(usage, Mem::MemUsageOf_Vector(planBuildWorkBuffers_.sendCounts_ByChunkBucket));
	Mem::Add(usage, Mem::MemUsageOf_Vector(planBuildWorkBuffers_.sendWriteOffsets_ByChunkBucket));
	Mem::Add(usage, Mem::MemUsageOf_Vector(planBuildWorkBuffers_.sendTotals));
	Mem::Add(usage, Mem::MemUsageOf_Vector(planBuildWorkBuffers_.recvCounts_ByChunkBucket));
	Mem::Add(usage, Mem::MemUsageOf_Vector(planBuildWorkBuffers_.recvWriteOffsets_ByChunkBucket));
	Mem::Add(usage, Mem::MemUsageOf_Vector(planBuildWorkBuffers_.recvTotals));
	Mem::Add(usage, Mem::MemUsageOf_Vector(planBuildWorkBuffers_.peerIdx_ByRank));
	Mem::Add(usage, Mem::MemUsageOf_Vector(planBuildWorkBuffers_.sendSingleBounds_ByPeer));
	Mem::Add(usage, Mem::MemUsageOf_Vector(planBuildWorkBuffers_.sendDoubleBounds_ByPeer));
	Mem::Add(usage, Mem::MemUsageOf_Vector(planBuildWorkBuffers_.recvPeerIdx_ByHaloIdx));
	Mem::Add(usage, Mem::MemUsageOf_Vector(peerIdxSortScratch_));

	return usage;
}
