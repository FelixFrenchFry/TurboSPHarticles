// dist_particle_state.cpp

// external
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// internal
#include "source/base/helpers.h"
#include "source/base/parallel_checks.h"
#include "parallel_utils.h"
#include "dist_particle_state.h"
#include "mpi_context.h"



using namespace Dist;

namespace
{
	// TODO: fine-tune chunk size by trying out 1024, 2048, 4096, 8192, 16384, 32768?
	inline constexpr uint32_t PARTICLE_CLASSIFICATION_CHUNK_SIZE = 16384;

	template <typename IsOuterAtLocalIdx>
	uint32_t CountOuterParticles_Parallel(
		const uint32_t N_owned,
		const IsOuterAtLocalIdx& isOuterAtLocalIdx)
	{
		return tbb::parallel_reduce(
			tbb::blocked_range<uint32_t>(0, N_owned),
			uint32_t{ 0 },
			[&](const tbb::blocked_range<uint32_t>& range, uint32_t N_outer)
		{
			for (uint32_t localIdx = range.begin(); localIdx != range.end(); localIdx++)
			{
				if (isOuterAtLocalIdx(localIdx)) { N_outer++; }
			}
			return N_outer;
		},
		[](const uint32_t lhs, const uint32_t rhs)
		{
			return lhs + rhs;
		});
	}

	// collects outer particle indices in ascending order and returns their count
	template <typename IsOuterAtLocalIdx>
	uint32_t CollectOuterLocalIdxs_Parallel(
		const uint32_t N_owned,
		const IsOuterAtLocalIdx& isOuterAtLocalIdx,
		ParticleClassificationWorkBuffers& workBuffers,
		std::vector<uint32_t>& outerLocalIdxs,
		const float extraAllocCapacityPct = 0.0f)
	{
		const uint32_t N_chunks = N_owned > 0
			? 1 + (N_owned - 1) / PARTICLE_CLASSIFICATION_CHUNK_SIZE
			: 0;

		Base::ResizeScratchVector(workBuffers.classCounts_ByChunkClass, N_chunks, extraAllocCapacityPct);
		Base::ResizeScratchVector(workBuffers.classWriteOffsets_ByChunkClass, N_chunks, extraAllocCapacityPct);
		Base::ResizeScratchVector(workBuffers.classTotals, 1);

		// count outer particles per chunk
		// (chunks processed in parallel, with each worker writing only its own count)
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginLocalIdx = chunkIdx * PARTICLE_CLASSIFICATION_CHUNK_SIZE;
				const uint32_t endLocalIdx = beginLocalIdx + std::min(
					PARTICLE_CLASSIFICATION_CHUNK_SIZE,
					N_owned - beginLocalIdx);

				uint32_t N_outer = 0;
				for (uint32_t localIdx = beginLocalIdx; localIdx != endLocalIdx; localIdx++)
				{
					if (isOuterAtLocalIdx(localIdx)) { N_outer++; }
				}
				workBuffers.classCounts_ByChunkClass[chunkIdx] = N_outer;
			}
		});

		// prefix chunk-local counts to compute output range of each chunk
		PrefixCountsByChunkAndBucket_ParallelScan(
			workBuffers.classCounts_ByChunkClass,
			workBuffers.classWriteOffsets_ByChunkClass,
			workBuffers.classTotals,
			N_chunks,
			1);

		const uint32_t N_outer = workBuffers.classTotals[0];

		Base::ResizeScratchVector(outerLocalIdxs, N_outer, extraAllocCapacityPct);

		// fill the outer list while preserving local-index order
		// (chunks processed in parallel, with each worker writing to disjoint ranges)
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginLocalIdx = chunkIdx * PARTICLE_CLASSIFICATION_CHUNK_SIZE;
				const uint32_t endLocalIdx = beginLocalIdx + std::min(
					PARTICLE_CLASSIFICATION_CHUNK_SIZE,
					N_owned - beginLocalIdx);
				uint32_t outerWriteIdx = workBuffers.classWriteOffsets_ByChunkClass[chunkIdx];

				for (uint32_t localIdx = beginLocalIdx; localIdx != endLocalIdx; localIdx++)
				{
					if (isOuterAtLocalIdx(localIdx))
					{
						outerLocalIdxs[outerWriteIdx++] = localIdx;
					}
				}
			}
		});

		return N_outer;
	}

	// position of a fluid or boundary particle, stored in owning particle storage or a view
	template <typename Particles>
	glm::vec3 PositionAtArrayIdx(
		const Particles& particles,
		const uint32_t arrayIdx)
	{
		return
		{
			particles.xPos[arrayIdx],
			particles.yPos[arrayIdx],
			particles.zPos[arrayIdx]
		};
	}

	template <typename Particles>
	glm::vec3 RigidPositionAtArrayIdx(
		const Particles& particlesR,
		const Sync::RigidBodies& rigidBodies,
		const uint32_t arrayIdx)
	{
		const uint32_t bodyIdx = particlesR.bodyId[arrayIdx];
		const float x = particlesR.xPosRel[arrayIdx];
		const float y = particlesR.yPosRel[arrayIdx];
		const float z = particlesR.zPosRel[arrayIdx];
		return
		{
			rigidBodies.xCM[bodyIdx] + std::fmaf(rigidBodies.a00[bodyIdx], x,
				std::fmaf(rigidBodies.a01[bodyIdx], y, rigidBodies.a02[bodyIdx] * z)),
			rigidBodies.yCM[bodyIdx] + std::fmaf(rigidBodies.a10[bodyIdx], x,
				std::fmaf(rigidBodies.a11[bodyIdx], y, rigidBodies.a12[bodyIdx] * z)),
			rigidBodies.zCM[bodyIdx] + std::fmaf(rigidBodies.a20[bodyIdx], x,
				std::fmaf(rigidBodies.a21[bodyIdx], y, rigidBodies.a22[bodyIdx] * z))
		};
	}

	template <typename PositionAtLocalIdx>
	void ClassifyOwnedParticlesIntoInnerOuter(
		DistParticleState& particleState,
		const uint32_t N_local,
		const PositionAtLocalIdx& positionAtLocalIdx,
		const IDomainDecomp& domainDecomp,
		const int rank,
		const float haloRadius,
		std::vector<uint32_t>* outerLocalIdxs,
		ParticleClassificationWorkBuffers* workBuffers)
	{
		if (particleState.N_owned > N_local)
		{
			throw std::runtime_error("Owned particle count exceeds the local particle count.");
		}

		const auto isOuterAtLocalIdx = [&](const uint32_t localIdx)
		{
			return !domainDecomp.IsInnerOwnedParticle(rank, positionAtLocalIdx(localIdx), haloRadius);
		};
		if (!outerLocalIdxs)
		{
			particleState.N_outer = CountOuterParticles_Parallel(particleState.N_owned, isOuterAtLocalIdx);
			return;
		}

		// use caller-owned scratch when available to retain allocations across calls
		ParticleClassificationWorkBuffers localWorkBuffers{};
		particleState.N_outer = CollectOuterLocalIdxs_Parallel(
			particleState.N_owned,
			isOuterAtLocalIdx,
			workBuffers ? *workBuffers : localWorkBuffers,
			*outerLocalIdxs);
	}

	bool IsHaloParticleForRank(
		const glm::vec3& position,
		const int ownerRank,
		const AABB& haloBounds,
		const int rank)
	{
		return ownerRank >= 0 && ownerRank != rank && haloBounds.Contains(position);
	}

	// TODO: reimplement the callsite of this function with bulk copies of contiguous arrays instead of per-particle copies?
	void CopyFluidParticle(
		const Sync::FluidParticles& src,
		Sync::FluidParticles& dst,
		const uint32_t srcIdx,
		const uint32_t dstIdx)
	{
		dst.id[dstIdx] = src.id[srcIdx];
		dst.xPos[dstIdx] = src.xPos[srcIdx];
		dst.yPos[dstIdx] = src.yPos[srcIdx];
		dst.zPos[dstIdx] = src.zPos[srcIdx];
		dst.xVel[dstIdx] = src.xVel[srcIdx];
		dst.yVel[dstIdx] = src.yVel[srcIdx];
		dst.zVel[dstIdx] = src.zVel[srcIdx];
		dst.xVelPred[dstIdx] = src.xVelPred[srcIdx];
		dst.yVelPred[dstIdx] = src.yVelPred[srcIdx];
		dst.zVelPred[dstIdx] = src.zVelPred[srcIdx];
		dst.xAccPress[dstIdx] = src.xAccPress[srcIdx];
		dst.yAccPress[dstIdx] = src.yAccPress[srcIdx];
		dst.zAccPress[dstIdx] = src.zAccPress[srcIdx];
		dst.sourceTerm[dstIdx] = src.sourceTerm[srcIdx];
		dst.diagElem[dstIdx] = src.diagElem[srcIdx];
		dst.pressure[dstIdx] = src.pressure[srcIdx];
		dst.density[dstIdx] = src.density[srcIdx];
	}

	// TODO: reimplement the callsite of this function with bulk copies of contiguous arrays instead of per-particle copies?
	void CopyBoundaryParticle(
		const Sync::BoundaryParticles& src,
		Sync::BoundaryParticles& dst,
		const uint32_t srcIdx,
		const uint32_t dstIdx)
	{
		dst.id[dstIdx] = src.id[srcIdx];
		dst.xPos[dstIdx] = src.xPos[srcIdx];
		dst.yPos[dstIdx] = src.yPos[srcIdx];
		dst.zPos[dstIdx] = src.zPos[srcIdx];
		dst.volB[dstIdx] = src.volB[srcIdx];
		dst.psiB[dstIdx] = src.psiB[srcIdx];
	}

	void CopyRigidParticle(
		const Sync::RigidParticles& src,
		Sync::RigidParticles& dst,
		const uint32_t srcIdx,
		const uint32_t dstIdx)
	{
		dst.id[dstIdx] = src.id[srcIdx];
		dst.bodyId[dstIdx] = src.bodyId[srcIdx];
		dst.xPosRel[dstIdx] = src.xPosRel[srcIdx];
		dst.yPosRel[dstIdx] = src.yPosRel[srcIdx];
		dst.zPosRel[dstIdx] = src.zPosRel[srcIdx];
		dst.volR[dstIdx] = src.volR[srcIdx];
		dst.psiR[dstIdx] = src.psiR[srcIdx];
	}

	template <typename PositionAtSrcIdx, typename ResizeLocalParticles,
		typename CopyParticleSourceToLocal, typename MoveSourceToLocal>
	void BuildLocalParticleRangesFromSourceState_Parallel(
		const uint32_t N_source,
		const PositionAtSrcIdx& positionAtSrcIdx,
		const ResizeLocalParticles& resizeLocalParticles,
		const CopyParticleSourceToLocal& copyParticleSourceToLocal,
		const MoveSourceToLocal& moveSourceToLocal,
		const IDomainDecomp& domainDecomp,
		const AABB& haloBounds,
		const int rank,
		uint32_t& N_owned,
		uint32_t& N_halo)
	{
		const uint32_t N_chunks = N_source > 0
			? 1 + (N_source - 1) / PARTICLE_BUILD_CHUNK_SIZE
			: 0;

		std::vector<uint32_t> classCounts_ByChunkClass(
			N_chunks * N_PARTICLE_BUILD_CLASSES, 0u);
		std::vector<uint32_t> classWriteOffsets_ByChunkClass(
			N_chunks * N_PARTICLE_BUILD_CLASSES, 0u);
		std::vector<uint32_t> classTotals(N_PARTICLE_BUILD_CLASSES, 0u);

		// count owned and halo particles per chunk from one owner query per particle
		// (chunks processed in parallel, with each worker writing only its own count row)
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginSrcIdx = chunkIdx * PARTICLE_BUILD_CHUNK_SIZE;
				const uint32_t endSrcIdx = beginSrcIdx + std::min(
					PARTICLE_BUILD_CHUNK_SIZE,
					N_source - beginSrcIdx);
				const uint32_t chunkClassBase = chunkIdx * N_PARTICLE_BUILD_CLASSES;

				for (uint32_t srcIdx = beginSrcIdx; srcIdx != endSrcIdx; srcIdx++)
				{
					const glm::vec3 position = positionAtSrcIdx(srcIdx);
					if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z))
					{
						throw std::runtime_error("Particle position is not finite.");
					}
					const int ownerRank = domainDecomp.OwnerRank(position);

					if (ownerRank == rank)
					{
						classCounts_ByChunkClass[
							chunkClassBase + PARTICLE_BUILD_CLASS_OWNED]++;
					}
					else if (IsHaloParticleForRank(position, ownerRank, haloBounds, rank))
					{
						classCounts_ByChunkClass[
							chunkClassBase + PARTICLE_BUILD_CLASS_HALO]++;
					}
				}
			}
		});

		// prefix chunk-local class counts to compute output range of each chunk
		PrefixCountsByChunkAndBucket_ParallelScan(
			classCounts_ByChunkClass,
			classWriteOffsets_ByChunkClass,
			classTotals,
			N_chunks,
			N_PARTICLE_BUILD_CLASSES);

		N_owned = classTotals[PARTICLE_BUILD_CLASS_OWNED];
		N_halo = classTotals[PARTICLE_BUILD_CLASS_HALO];

		const uint32_t N_local = Sim::CheckedCombinedLocalParticleCount(
			N_owned, N_halo, "Initial owned and halo particle count");

		// reuse source storage when all particles already belong to this rank
		if (N_owned == N_source)
		{
			moveSourceToLocal();
			return;
		}

		resizeLocalParticles(N_local);

		// copy owned particles into the compact prefix and halo particles into the suffix
		// (chunks processed in parallel, with each worker writing to disjoint ranges)
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginSrcIdx = chunkIdx * PARTICLE_BUILD_CHUNK_SIZE;
				const uint32_t endSrcIdx = beginSrcIdx + std::min(
					PARTICLE_BUILD_CHUNK_SIZE,
					N_source - beginSrcIdx);
				const uint32_t chunkClassBase = chunkIdx * N_PARTICLE_BUILD_CLASSES;
				uint32_t ownedLocalIdx =
					classWriteOffsets_ByChunkClass[
						chunkClassBase + PARTICLE_BUILD_CLASS_OWNED];
				uint32_t haloLocalIdx = N_owned +
					classWriteOffsets_ByChunkClass[
						chunkClassBase + PARTICLE_BUILD_CLASS_HALO];

				for (uint32_t srcIdx = beginSrcIdx; srcIdx != endSrcIdx; srcIdx++)
				{
					const glm::vec3 position = positionAtSrcIdx(srcIdx);
					const int ownerRank = domainDecomp.OwnerRank(position);

					if (ownerRank == rank)
					{
						copyParticleSourceToLocal(srcIdx, ownedLocalIdx++);
					}
					else if (IsHaloParticleForRank(position, ownerRank, haloBounds, rank))
					{
						copyParticleSourceToLocal(srcIdx, haloLocalIdx++);
					}
				}
			}
		});
	}

	// build compact rank-local fluid storage as [ owned particles ][ halo particles ]
	void BuildLocalFluidParticlesFromGlobalState(
		DistParticleState& particleState,
		Sync::FluidParticles& localFluid,
		Sync::FluidParticles& globalFluid,
		const IDomainDecomp& domainDecomp,
		const int rank)
	{
		const AABB haloBounds = domainDecomp.HaloExtentAABB(rank);

		// reset metadata for the new compact layout
		particleState = {};

		// fill the compact prefix with particles owned by this rank, then append
		// non-owned particles inside the halo bounds of this rank
		BuildLocalParticleRangesFromSourceState_Parallel(
			globalFluid.N_,
			[&](const uint32_t srcIdx)
			{
				return PositionAtArrayIdx(globalFluid, srcIdx);
			},
			[&](const uint32_t N_local)
			{
				localFluid.resizeAll(N_local);
			},
			[&](const uint32_t srcIdx, const uint32_t dstIdx)
			{
				CopyFluidParticle(globalFluid, localFluid, srcIdx, dstIdx);
			},
			[&]()
			{
				localFluid = std::move(globalFluid);
			},
			domainDecomp, haloBounds, rank, particleState.N_owned, particleState.N_halo);
	}

	// build compact rank-local boundary storage as [ owned particles ][ halo particles ]
	void BuildLocalBoundaryParticlesFromGlobalState(
		DistParticleState& particleState,
		Sync::BoundaryParticles& localBoundary,
		Sync::BoundaryParticles& globalBoundary,
		const IDomainDecomp& domainDecomp,
		const int rank)
	{
		const AABB haloBounds = domainDecomp.HaloExtentAABB(rank);

		// reset metadata for the new compact layout
		particleState = {};

		// fill the compact prefix with particles owned by this rank, then append
		// non-owned particles inside the halo bounds of this rank
		BuildLocalParticleRangesFromSourceState_Parallel(
			globalBoundary.N_,
			[&](const uint32_t srcIdx)
			{
				return PositionAtArrayIdx(globalBoundary, srcIdx);
			},
			[&](const uint32_t N_local)
			{
				localBoundary.resizeAll(N_local);
			},
			[&](const uint32_t srcIdx, const uint32_t dstIdx)
			{
				CopyBoundaryParticle(globalBoundary, localBoundary, srcIdx, dstIdx);
			},
			[&]()
			{
				localBoundary = std::move(globalBoundary);
			},
			domainDecomp, haloBounds, rank, particleState.N_owned, particleState.N_halo);
	}

	// build compact rank-local rigid storage as [ owned particles ][ halo particles ]
	void BuildLocalRigidParticlesFromGlobalState(
		DistParticleState& particleState,
		Sync::RigidParticles& localRigid,
		Sync::RigidParticles& globalRigid,
		const Sync::RigidBodies& rigidBodies,
		const IDomainDecomp& domainDecomp,
		const int rank)
	{
		std::string failMsg{};
		if (!ValidateLocalRigidParticleAttachments(globalRigid.constView(), rigidBodies, &failMsg))
		{
			throw std::runtime_error(failMsg);
		}

		const AABB haloBounds = domainDecomp.HaloExtentAABB(rank);

		// reset metadata for the new compact layout
		particleState = {};

		// fill the compact prefix with particles owned by this rank,
		// then append non-owned particles inside the halo bounds of this rank
		BuildLocalParticleRangesFromSourceState_Parallel(
			globalRigid.N_,
			[&](const uint32_t srcIdx)
			{
				return RigidPositionAtArrayIdx(globalRigid, rigidBodies, srcIdx);
			},
			[&](const uint32_t N_local)
			{
				localRigid.resizeAll(N_local);
			},
			[&](const uint32_t srcIdx, const uint32_t dstIdx)
			{
				CopyRigidParticle(globalRigid, localRigid, srcIdx, dstIdx);
			},
			[&]()
			{
				localRigid = std::move(globalRigid);
			},
			domainDecomp, haloBounds, rank, particleState.N_owned, particleState.N_halo);
	}

} // namespace anonymous

bool DistParticleState::IsValidLocalParticleLayout(
	const uint32_t N_local,
	std::string* failMsg) const
{
	if (failMsg) { failMsg->clear(); }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};
	if (N_local > Sim::MaxLocalParticleCount)
	{
		return fail("localParticleCount exceeds the local particle limit.");
	}

	// check that counts match the compact local range
	if (N_owned > N_local ||
		N_halo != N_local - N_owned)
	{
		return fail("ownedParticleCount + haloParticleCount does not match localParticleCount.");
	}
	if (N_outer > N_owned)
	{
		return fail("outerParticleCount exceeds ownedParticleCount.");
	}

	return true;
}

bool DistParticleState::IsValidLocalParticleState(
	const std::span<const uint64_t> particleIds,
	const uint64_t N_global,
	std::string* failMsg) const
{
	if (failMsg) { failMsg->clear(); }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};

	if (particleIds.size() > Sim::MaxLocalParticleCount)
	{
		return fail("particleIds.size() exceeds the local particle limit.");
	}

	const uint32_t N_local = static_cast<uint32_t>(particleIds.size());
	if (!IsValidLocalParticleLayout(N_local, failMsg))
	{
		return false;
	}

	#if defined(_DEBUG)
	// check that every particle ID lands inside the global particle range
	if (Check::CountViolations(0, N_local, [&](const uint32_t localIdx)
	{
		return particleIds[localIdx] >= N_global;
	}) != 0)
	{
		return fail("particle id is outside the global particle range.");
	}

	// sort only the rank-local identity list to detect duplicates
	std::vector<uint64_t> sortedIds(particleIds.begin(), particleIds.end());
	std::sort(sortedIds.begin(), sortedIds.end());
	if (std::adjacent_find(sortedIds.begin(), sortedIds.end()) != sortedIds.end())
	{
		return fail("particle IDs contain a duplicate ID.");
	}
	#else
	(void)N_global;
	#endif

	return true;
}

bool Dist::ValidateLocalRigidParticleAttachments(
	const Sync::RigidConstView& particlesR,
	const Sync::RigidBodies& rigidBodies,
	std::string* failMsg)
{
	if (failMsg) { failMsg->clear(); }

	if (particlesR.id.size() != particlesR.N_ ||
		particlesR.bodyId.size() != particlesR.N_ ||
		particlesR.xPosRel.size() != particlesR.N_ ||
		particlesR.yPosRel.size() != particlesR.N_ ||
		particlesR.zPosRel.size() != particlesR.N_ ||
		particlesR.volR.size() != particlesR.N_ ||
		particlesR.psiR.size() != particlesR.N_ ||
		rigidBodies.rigidParticleCount.size() != rigidBodies.N_)
	{
		if (failMsg) { *failMsg = "Invalid local rigid field sizes."; }
		return false;
	}

	for (const uint32_t bodyIdx : particlesR.bodyId)
	{
		if (bodyIdx >= rigidBodies.N_)
		{
			if (failMsg) { *failMsg = "Invalid local rigid body index."; }
			return false;
		}
	}
	return true;
}

bool Dist::FindFirstOwnedParticleConflict(
	const std::span<const uint64_t> ownedParticleIds,
	const uint64_t N_global,
	const MpiContext& mpiContext,
	std::optional<OwnedParticleConflict>& conflict,
	std::string* const failMsg)
{
	constexpr uint64_t WindowSize = 65'536;
	conflict.reset();
	if (failMsg) { failMsg->clear(); }

	std::vector<uint64_t> sortedOwnedIds{};
	bool localOk = true;
	try
	{
		sortedOwnedIds.reserve(ownedParticleIds.size());
		for (const uint64_t particleId : ownedParticleIds)
		{
			if (particleId >= N_global)
			{
				localOk = false;
				if (failMsg) { *failMsg = "Owned particle diagnostic input is invalid."; }
				break;
			}
			sortedOwnedIds.push_back(particleId);
		}
		std::sort(sortedOwnedIds.begin(), sortedOwnedIds.end());
	}
	catch (const std::exception& ex)
	{
		localOk = false;
		if (failMsg) { *failMsg = ex.what(); }
	}
	if (!mpiContext.AllTrue(localOk, "owned particle diagnostic input"))
	{
		if (failMsg && failMsg->empty())
		{
			*failMsg = "Owned particle diagnostic input failed on another rank.";
		}
		return false;
	}

	std::vector<uint32_t> ownerCounts{};
	try
	{
		ownerCounts.resize(static_cast<uint32_t>(std::min(WindowSize, N_global)));
	}
	catch (const std::exception& ex)
	{
		localOk = false;
		if (failMsg) { *failMsg = ex.what(); }
	}
	if (!mpiContext.AllTrue(localOk, "owned particle diagnostic scratch"))
	{
		if (failMsg && failMsg->empty())
		{
			*failMsg = "Owned particle diagnostic scratch allocation failed on another rank.";
		}
		return false;
	}

	uint64_t ownedCursor = 0;
	for (uint64_t windowBase = 0; windowBase < N_global;)
	{
		const uint32_t N_window = static_cast<uint32_t>(std::min(WindowSize, N_global - windowBase));
		ownerCounts.resize(N_window);
		std::fill(ownerCounts.begin(), ownerCounts.end(), 0u);
		const uint64_t windowEnd = windowBase + N_window;

		while (ownedCursor < sortedOwnedIds.size() &&
			sortedOwnedIds[ownedCursor] < windowEnd)
		{
			const uint64_t id = sortedOwnedIds[ownedCursor++];
			ownerCounts[static_cast<uint32_t>(id - windowBase)]++;
		}

		mpiContext.AllReduceSumUint32s(ownerCounts);
		const auto invalidOwnerIt = std::find_if(
			ownerCounts.begin(), ownerCounts.end(),
			[](const uint32_t ownerCount) { return ownerCount != 1; });
		if (invalidOwnerIt != ownerCounts.end())
		{
			conflict = OwnedParticleConflict
			{
				.id = windowBase + static_cast<uint64_t>(
					std::distance(ownerCounts.begin(), invalidOwnerIt)),
				.ownerCount = *invalidOwnerIt
			};
			return true;
		}

		windowBase = windowEnd;
	}

	return true;
}

Mem::MemUsage Dist::GetParticleMemoryUsage(
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

Mem::MemUsage Dist::GetParticleMemoryUsage(
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

Mem::MemUsage Dist::GetParticleMemoryUsage(
	const Sync::RigidParticles& particles)
{
	Mem::MemUsage usage{};

	Mem::Add(usage, Mem::MemUsageOf_Vector(particles.id));
	Mem::Add(usage, Mem::MemUsageOf_Vector(particles.bodyId));
	Mem::Add(usage, Mem::MemUsageOf_Vector(particles.xPosRel));
	Mem::Add(usage, Mem::MemUsageOf_Vector(particles.yPosRel));
	Mem::Add(usage, Mem::MemUsageOf_Vector(particles.zPosRel));
	Mem::Add(usage, Mem::MemUsageOf_Vector(particles.volR));
	Mem::Add(usage, Mem::MemUsageOf_Vector(particles.psiR));

	return usage;
}

Mem::MemUsage Dist::GetParticleMemoryUsage(
	const Sync::UniversalSimState& state)
{
	Mem::MemUsage usage{};

	Mem::Add(usage, GetParticleMemoryUsage(state.particlesF));
	Mem::Add(usage, GetParticleMemoryUsage(state.particlesB));
	Mem::Add(usage, GetParticleMemoryUsage(state.particlesR));

	return usage;
}

void Dist::ClassifyOwnedFluidParticlesIntoInnerOuter(
	DistParticleState& particleState,
	const Sync::FluidConstView& particlesF,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float haloRadius,
	std::vector<uint32_t>* outerLocalIdxs,
	ParticleClassificationWorkBuffers* workBuffers)
{
	ClassifyOwnedParticlesIntoInnerOuter(
		particleState,
		particlesF.N_,
		[&](const uint32_t localIdx)
		{
			return PositionAtArrayIdx(particlesF, localIdx);
		},
		domainDecomp,
		rank,
		haloRadius,
		outerLocalIdxs,
		workBuffers);
}

void Dist::ClassifyOwnedBoundaryParticlesIntoInnerOuter(
	DistParticleState& particleState,
	const Sync::BoundaryConstView& particlesB,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float haloRadius,
	std::vector<uint32_t>* outerLocalIdxs,
	ParticleClassificationWorkBuffers* workBuffers)
{
	ClassifyOwnedParticlesIntoInnerOuter(
		particleState,
		particlesB.N_,
		[&](const uint32_t localIdx)
		{
			return PositionAtArrayIdx(particlesB, localIdx);
		},
		domainDecomp,
		rank,
		haloRadius,
		outerLocalIdxs,
		workBuffers);
}

void Dist::ClassifyOwnedRigidParticlesIntoInnerOuter(
	DistParticleState& particleState,
	const Sync::RigidConstView& particlesR,
	const Sync::RigidBodies& rigidBodies,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float haloRadius,
	std::vector<uint32_t>* outerLocalIdxs,
	ParticleClassificationWorkBuffers* workBuffers)
{
	ClassifyOwnedParticlesIntoInnerOuter(
		particleState,
		particlesR.N_,
		[&](const uint32_t localIdx)
		{
			return RigidPositionAtArrayIdx(particlesR, rigidBodies, localIdx);
		},
		domainDecomp,
		rank,
		haloRadius,
		outerLocalIdxs,
		workBuffers);
}

uint32_t Dist::CollectPreparedOuterFluidLocalIdxs(
	const uint32_t N_owned,
	const std::span<const uint32_t> retainedOldLocalIdxs,
	const std::span<const uint8_t> isOuterOwned_ByLocalIdx,
	const Sync::FluidConstView& appendedParticles,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float haloRadius,
	ParticleClassificationWorkBuffers& workBuffers,
	std::vector<uint32_t>& outerLocalIdxs,
	const float extraAllocCapacityPct)
{
	const uint32_t N_retained = static_cast<uint32_t>(retainedOldLocalIdxs.size());
	return CollectOuterLocalIdxs_Parallel(
		N_owned,
		[&](const uint32_t localIdx)
		{
			if (localIdx < N_retained)
			{
				return isOuterOwned_ByLocalIdx[
					retainedOldLocalIdxs[localIdx]] != 0;
			}

			const uint32_t appendIdx = localIdx - N_retained;
			const glm::vec3 position
			{
				appendedParticles.xPos[appendIdx],
				appendedParticles.yPos[appendIdx],
				appendedParticles.zPos[appendIdx]
			};
			return !domainDecomp.IsInnerOwnedParticle(rank, position, haloRadius);
		},
		workBuffers,
		outerLocalIdxs,
		extraAllocCapacityPct);
}

uint32_t Dist::CountPreparedOuterBoundaryParticles(
	const uint32_t N_owned,
	const std::span<const uint32_t> retainedOldLocalIdxs,
	const std::span<const uint8_t> isOuterOwned_ByLocalIdx,
	const Sync::BoundaryConstView& appendedParticles,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float haloRadius)
{
	const uint32_t N_retained = static_cast<uint32_t>(retainedOldLocalIdxs.size());
	return CountOuterParticles_Parallel(
		N_owned,
		[&](const uint32_t localIdx)
		{
			if (localIdx < N_retained)
			{
				return isOuterOwned_ByLocalIdx[
					retainedOldLocalIdxs[localIdx]] != 0;
			}

			const uint32_t appendIdx = localIdx - N_retained;
			const glm::vec3 position
			{
				appendedParticles.xPos[appendIdx],
				appendedParticles.yPos[appendIdx],
				appendedParticles.zPos[appendIdx]
			};
			return !domainDecomp.IsInnerOwnedParticle(rank, position, haloRadius);
		});
}

uint32_t Dist::CountPreparedOuterRigidParticles(
	const uint32_t N_owned,
	const std::span<const uint32_t> retainedOldLocalIdxs,
	const Sync::RigidConstView& oldParticles,
	const Sync::RigidConstView& appendedParticles,
	const Sync::RigidBodies& rigidBodies,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float haloRadius)
{
	const uint32_t N_retained = static_cast<uint32_t>(retainedOldLocalIdxs.size());
	return CountOuterParticles_Parallel(
		N_owned,
		[&](const uint32_t localIdx)
		{
			const bool retained = localIdx < N_retained;
			const auto& src = retained ? oldParticles : appendedParticles;
			const uint32_t srcLocalIdx = retained
				? retainedOldLocalIdxs[localIdx] : localIdx - N_retained;
			return !domainDecomp.IsInnerOwnedParticle(
				rank, RigidPositionAtArrayIdx(src, rigidBodies, srcLocalIdx), haloRadius);
		});
}

void Dist::BuildLocalParticlesFromGlobalState(
	Sync::UniversalSimState& localState,
	DistParticleState& particleStateF,
	DistParticleState& particleStateB,
	DistParticleState& particleStateR,
	Sync::UniversalSimState&& globalState,
	const IDomainDecomp& domainDecomp,
	const int rank)
{
	// reset the rank-local state while preserving global step and time
	localState = {};
	localState.step = globalState.step;
	localState.time = globalState.time;
	localState.rigidBodies = globalState.rigidBodies;

	// build compact rank-local F/B/R particle layouts
	BuildLocalFluidParticlesFromGlobalState(
		particleStateF, localState.particlesF, globalState.particlesF,
		domainDecomp, rank);
	BuildLocalBoundaryParticlesFromGlobalState(
		particleStateB, localState.particlesB, globalState.particlesB,
		domainDecomp, rank);
	BuildLocalRigidParticlesFromGlobalState(
		particleStateR, localState.particlesR, globalState.particlesR, globalState.rigidBodies,
		domainDecomp, rank);

	// release source particles while retaining replicated body state
	globalState.particlesF = {};
	globalState.particlesB = {};
	globalState.particlesR = {};
}
