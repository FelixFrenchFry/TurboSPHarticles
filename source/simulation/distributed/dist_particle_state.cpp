// dist_particle_state.cpp

// external
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <span>
#include <utility>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// internal
#include "source/base/helpers.h"
#include "source/base/parallel_checks.h"
#include "parallel_utils.h"
#include "dist_particle_state.h"



using namespace Dist;

namespace
{
	glm::vec3 FluidPositionAtArrayIdx(
		const Sync::FluidParticles& particlesF,
		const uint32_t arrayIdx)
	{
		return
		{
			particlesF.xPos[arrayIdx],
			particlesF.yPos[arrayIdx],
			particlesF.zPos[arrayIdx]
		};
	}

	glm::vec3 BoundaryPositionAtArrayIdx(
		const Sync::BoundaryParticles& particlesB,
		const uint32_t arrayIdx)
	{
		return
		{
			particlesB.xPos[arrayIdx],
			particlesB.yPos[arrayIdx],
			particlesB.zPos[arrayIdx]
		};
	}

	void PrepareParticleClassification(
		DistParticleState& particleState,
		const uint32_t N_particles)
	{
		particleState.N_owned = 0;
		particleState.N_halo = 0;
		particleState.particleClasses.assign(N_particles, DistParticleClass::None);
		particleState.particleLocalIdx_Owned.clear();
		particleState.particleLocalIdx_Inner.clear();
		particleState.particleLocalIdx_Outer.clear();
		particleState.particleLocalIdx_Halo.clear();
	}

	template <typename PositionAtLocalIdx>
	void ClassifyOwnedParticlesIntoInnerOuter_Parallel(
		DistParticleState& particleState,
		const uint32_t N_particles,
		const PositionAtLocalIdx& positionAtLocalIdx,
		const IDomainDecomp& domainDecomp,
		const int rank,
		const float haloRadius,
		ParticleClassificationWorkBuffers& workBuffers)
	{
		PrepareParticleClassification(particleState, N_particles);

		const uint32_t N_chunks = N_particles > 0
			? 1 + (N_particles - 1) / PARTICLE_CLASSIFICATION_CHUNK_SIZE
			: 0;

		workBuffers.classCounts_ByChunkClass.assign(
			N_chunks * N_PARTICLE_CLASSIFICATION_CLASSES, 0u);
		workBuffers.classWriteOffsets_ByChunkClass.assign(
			N_chunks * N_PARTICLE_CLASSIFICATION_CLASSES, 0u);
		workBuffers.classTotals.assign(N_PARTICLE_CLASSIFICATION_CLASSES, 0u);

		// classify owned particles and count each class per chunk
		// (chunks processed in parallel, with each worker writing only its own count row)
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginLocalIdx = chunkIdx * PARTICLE_CLASSIFICATION_CHUNK_SIZE;
				const uint32_t endLocalIdx = beginLocalIdx + std::min(
					PARTICLE_CLASSIFICATION_CHUNK_SIZE,
					N_particles - beginLocalIdx);
				const uint32_t chunkClassBase = chunkIdx * N_PARTICLE_CLASSIFICATION_CLASSES;

				for (uint32_t localIdx = beginLocalIdx; localIdx != endLocalIdx; localIdx++)
				{
					const glm::vec3 position = positionAtLocalIdx(localIdx);
					if (domainDecomp.OwnerRank(position) != rank) { continue; }

					workBuffers.classCounts_ByChunkClass[
						chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_OWNED]++;

					// inner owned particles do not need to be exchanged as halo data
					if (domainDecomp.IsInnerOwnedParticle(rank, position, haloRadius))
					{
						particleState.particleClasses[localIdx] = DistParticleClass::Inner;
						workBuffers.classCounts_ByChunkClass[
							chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_INNER]++;
					}
					else
					{
						particleState.particleClasses[localIdx] = DistParticleClass::Outer;
						workBuffers.classCounts_ByChunkClass[
							chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_OUTER]++;
					}
				}
			}
		});

		// prefix chunk-local class counts to compute output range of each chunk
		PrefixCountsByChunkAndBucket_ParallelScan(
			workBuffers.classCounts_ByChunkClass,
			workBuffers.classWriteOffsets_ByChunkClass,
			workBuffers.classTotals,
			N_chunks,
			N_PARTICLE_CLASSIFICATION_CLASSES);

		particleState.particleLocalIdx_Owned.resize(
			workBuffers.classTotals[PARTICLE_CLASSIFICATION_CLASS_OWNED]);
		particleState.particleLocalIdx_Inner.resize(
			workBuffers.classTotals[PARTICLE_CLASSIFICATION_CLASS_INNER]);
		particleState.particleLocalIdx_Outer.resize(
			workBuffers.classTotals[PARTICLE_CLASSIFICATION_CLASS_OUTER]);

		// fill owned/inner/outer index lists while preserving local-index order
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
					N_particles - beginLocalIdx);
				const uint32_t chunkClassBase = chunkIdx * N_PARTICLE_CLASSIFICATION_CLASSES;
				uint32_t ownedWriteIdx =
					workBuffers.classWriteOffsets_ByChunkClass[
						chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_OWNED];
				uint32_t innerWriteIdx =
					workBuffers.classWriteOffsets_ByChunkClass[
						chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_INNER];
				uint32_t outerWriteIdx =
					workBuffers.classWriteOffsets_ByChunkClass[
						chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_OUTER];

				for (uint32_t localIdx = beginLocalIdx; localIdx != endLocalIdx; localIdx++)
				{
					const DistParticleClass particleClass = particleState.particleClasses[localIdx];

					if (particleClass == DistParticleClass::Inner)
					{
						particleState.particleLocalIdx_Owned[ownedWriteIdx++] = localIdx;
						particleState.particleLocalIdx_Inner[innerWriteIdx++] = localIdx;
					}
					else if (particleClass == DistParticleClass::Outer)
					{
						particleState.particleLocalIdx_Owned[ownedWriteIdx++] = localIdx;
						particleState.particleLocalIdx_Outer[outerWriteIdx++] = localIdx;
					}
				}
			}
		});

		particleState.N_owned = workBuffers.classTotals[PARTICLE_CLASSIFICATION_CLASS_OWNED];
	}

	bool IsHaloParticleForRank(
		const glm::vec3& position,
		const int ownerRank,
		const AABB& haloBounds,
		const int rank)
	{
		return ownerRank >= 0 && ownerRank != rank && haloBounds.Contains(position);
	}

	void ClassifyHaloParticleRange(
		DistParticleState& particleState,
		const uint32_t N_owned,
		const uint32_t N_local)
	{
		particleState.N_owned = N_owned;
		particleState.N_halo = N_local - N_owned;

		particleState.particleLocalIdx_Halo.resize(particleState.N_halo);
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, particleState.N_halo),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t haloIdx = range.begin(); haloIdx != range.end(); haloIdx++)
			{
				const uint32_t localIdx = N_owned + haloIdx;
				particleState.particleLocalIdx_Halo[haloIdx] = localIdx;
				particleState.particleClasses[localIdx] = DistParticleClass::Halo;
			}
		});
	}

	#if defined(_DEBUG)
	bool IsLocalIdxInRange(
		const uint32_t localIdx,
		const uint32_t firstLocalIdx,
		const uint32_t lastLocalIdx_Excl)
	{
		return localIdx >= firstLocalIdx && localIdx < lastLocalIdx_Excl;
	}

	bool LocalIdxListCoversRange(
		const std::vector<uint32_t>& localIdxs,
		const uint32_t firstLocalIdx,
		const uint32_t lastLocalIdx_Excl)
	{
		if (lastLocalIdx_Excl < firstLocalIdx) { return false; }

		const uint32_t N_expected = lastLocalIdx_Excl - firstLocalIdx;
		if (localIdxs.size() != N_expected) { return false; }

		std::vector<uint8_t> covered(N_expected, 0);
		uint32_t previousLocalIdx = 0;
		bool hasPreviousLocalIdx = false;
		for (const uint32_t localIdx : localIdxs)
		{
			if (!IsLocalIdxInRange(localIdx, firstLocalIdx, lastLocalIdx_Excl)) { return false; }
			if (hasPreviousLocalIdx && localIdx <= previousLocalIdx) { return false; }

			const uint32_t coveredIdx = localIdx - firstLocalIdx;
			if (covered[coveredIdx] != 0) { return false; }
			covered[coveredIdx] = 1;
			previousLocalIdx = localIdx;
			hasPreviousLocalIdx = true;
		}

		return true;
	}

	bool LocalIdxListsCoverRange(
		const std::vector<uint32_t>& localIdxsA,
		const std::vector<uint32_t>& localIdxsB,
		const uint32_t firstLocalIdx,
		const uint32_t lastLocalIdx_Excl)
	{
		if (lastLocalIdx_Excl < firstLocalIdx) { return false; }

		const uint32_t N_expected = lastLocalIdx_Excl - firstLocalIdx;
		if (localIdxsA.size() + localIdxsB.size() != N_expected)
		{
			return false;
		}

		std::vector<uint8_t> covered(N_expected, 0);

		const auto markLocalIdx = [&](const uint32_t localIdx) -> bool
		{
			if (!IsLocalIdxInRange(localIdx, firstLocalIdx, lastLocalIdx_Excl)) { return false; }

			const uint32_t coveredIdx = localIdx - firstLocalIdx;
			if (covered[coveredIdx] != 0) { return false; }
			covered[coveredIdx] = 1;
			return true;
		};

		const auto markLocalIdxList = [&](const std::vector<uint32_t>& localIdxs) -> bool
		{
			uint32_t previousLocalIdx = 0;
			bool hasPreviousLocalIdx = false;
			for (const uint32_t localIdx : localIdxs)
			{
				if (hasPreviousLocalIdx && localIdx <= previousLocalIdx) { return false; }
				if (!markLocalIdx(localIdx)) { return false; }

				previousLocalIdx = localIdx;
				hasPreviousLocalIdx = true;
			}
			return true;
		};

		return markLocalIdxList(localIdxsA) && markLocalIdxList(localIdxsB);
	}
	#endif

	#if defined(_DEBUG)
	bool LocalIdxReOrderCoversRange(
		const std::span<const uint32_t> oldLocalIdx_FromNewLocalIdx,
		const uint32_t N_local)
	{
		if (oldLocalIdx_FromNewLocalIdx.size() != N_local)
		{
			return false;
		}

		// check that the reorder map is a permutation of the local particle range
		if (Check::CountViolations(0, N_local, [&](const uint32_t newLocalIdx)
		{
			return oldLocalIdx_FromNewLocalIdx[newLocalIdx] >= N_local;
		}) != 0)
		{
			return false;
		}

		std::vector<uint32_t> claimBitsScratch{};
		return Check::FindFirstDuplicateIdx(
			oldLocalIdx_FromNewLocalIdx,
			N_local,
			claimBitsScratch) == N_local;
	}
	#endif

	// TODO: reimplement the callsite of this function with bulk copies of contiguous arrays instead of per-particle copies?
	void CopyFluidParticle_GlobalToLocal(
		const Sync::FluidParticles& src,
		Sync::FluidParticles& dst,
		const uint32_t globalIdx,
		const uint32_t localIdx)
	{
		dst.id[localIdx] = src.id[globalIdx];
		dst.xPos[localIdx] = src.xPos[globalIdx];
		dst.yPos[localIdx] = src.yPos[globalIdx];
		dst.zPos[localIdx] = src.zPos[globalIdx];
		dst.xVel[localIdx] = src.xVel[globalIdx];
		dst.yVel[localIdx] = src.yVel[globalIdx];
		dst.zVel[localIdx] = src.zVel[globalIdx];
		dst.xVelPred[localIdx] = src.xVelPred[globalIdx];
		dst.yVelPred[localIdx] = src.yVelPred[globalIdx];
		dst.zVelPred[localIdx] = src.zVelPred[globalIdx];
		dst.xAccPress[localIdx] = src.xAccPress[globalIdx];
		dst.yAccPress[localIdx] = src.yAccPress[globalIdx];
		dst.zAccPress[localIdx] = src.zAccPress[globalIdx];
		dst.sourceTerm[localIdx] = src.sourceTerm[globalIdx];
		dst.diagElem[localIdx] = src.diagElem[globalIdx];
		dst.pressure[localIdx] = src.pressure[globalIdx];
		dst.density[localIdx] = src.density[globalIdx];
	}

	// TODO: reimplement the callsite of this function with bulk copies of contiguous arrays instead of per-particle copies?
	void CopyBoundaryParticle_GlobalToLocal(
		const Sync::BoundaryParticles& src,
		Sync::BoundaryParticles& dst,
		const uint32_t globalIdx,
		const uint32_t localIdx)
	{
		dst.id[localIdx] = src.id[globalIdx];
		dst.xPos[localIdx] = src.xPos[globalIdx];
		dst.yPos[localIdx] = src.yPos[globalIdx];
		dst.zPos[localIdx] = src.zPos[globalIdx];
		dst.volB[localIdx] = src.volB[globalIdx];
		dst.psiB[localIdx] = src.psiB[globalIdx];
	}

	template <typename PositionAtGlobalIdx, typename ResizeLocalParticles, typename CopyParticleGlobalToLocal>
	void BuildLocalParticleRangesFromGlobalState_Parallel(
		const uint32_t N_global,
		const PositionAtGlobalIdx& positionAtGlobalIdx,
		const ResizeLocalParticles& resizeLocalParticles,
		const CopyParticleGlobalToLocal& copyParticleGlobalToLocal,
		const IDomainDecomp& domainDecomp,
		const AABB& haloBounds,
		const int rank,
		uint32_t& N_owned,
		uint32_t& N_halo)
	{
		const uint32_t N_chunks = N_global > 0
			? 1 + (N_global - 1) / PARTICLE_BUILD_CHUNK_SIZE
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
				const uint32_t beginGlobalIdx = chunkIdx * PARTICLE_BUILD_CHUNK_SIZE;
				const uint32_t endGlobalIdx = beginGlobalIdx + std::min(
					PARTICLE_BUILD_CHUNK_SIZE,
					N_global - beginGlobalIdx);
				const uint32_t chunkClassBase = chunkIdx * N_PARTICLE_BUILD_CLASSES;

				for (uint32_t globalIdx = beginGlobalIdx; globalIdx != endGlobalIdx; globalIdx++)
				{
					const glm::vec3 position = positionAtGlobalIdx(globalIdx);
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

		// allocate exact local particle storage before the compact ranges are filled
		resizeLocalParticles(N_owned + N_halo);

		// copy owned particles into the compact prefix and halo particles into the suffix
		// (chunks processed in parallel, with each worker writing to disjoint ranges)
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint32_t beginGlobalIdx = chunkIdx * PARTICLE_BUILD_CHUNK_SIZE;
				const uint32_t endGlobalIdx = beginGlobalIdx + std::min(
					PARTICLE_BUILD_CHUNK_SIZE,
					N_global - beginGlobalIdx);
				const uint32_t chunkClassBase = chunkIdx * N_PARTICLE_BUILD_CLASSES;
				uint32_t ownedLocalIdx =
					classWriteOffsets_ByChunkClass[
						chunkClassBase + PARTICLE_BUILD_CLASS_OWNED];
				uint32_t haloLocalIdx = N_owned +
					classWriteOffsets_ByChunkClass[
						chunkClassBase + PARTICLE_BUILD_CLASS_HALO];

				for (uint32_t globalIdx = beginGlobalIdx; globalIdx != endGlobalIdx; globalIdx++)
				{
					const glm::vec3 position = positionAtGlobalIdx(globalIdx);
					const int ownerRank = domainDecomp.OwnerRank(position);

					if (ownerRank == rank)
					{
						copyParticleGlobalToLocal(globalIdx, ownedLocalIdx++);
					}
					else if (IsHaloParticleForRank(position, ownerRank, haloBounds, rank))
					{
						copyParticleGlobalToLocal(globalIdx, haloLocalIdx++);
					}
				}
			}
		});
	}

	// build compact rank-local fluid storage as [ owned particles ][ halo particles ]
	void BuildLocalFluidParticlesFromGlobalState(
		DistParticleState& particleState,
		Sync::UniversalSimState& localState,
		const Sync::UniversalSimState& globalState,
		const IDomainDecomp& domainDecomp,
		const int rank,
		const float haloRadius)
	{
		const Sync::FluidParticles& globalFluid = globalState.particlesF;
		Sync::FluidParticles& localFluid = localState.particlesF;

		const AABB haloBounds = domainDecomp.HaloExtentAABB(rank);

		// reset metadata for the new compact layout
		particleState.Clear();

		// particle IDs hold the global scenario index, so a rank-local build slice still maps back globally

		// fill the compact prefix with particles owned by this rank, then append
		// non-owned particles inside the halo bounds of this rank
		uint32_t N_owned = 0;
		uint32_t N_halo = 0;
		BuildLocalParticleRangesFromGlobalState_Parallel(
			globalFluid.N_,
			[&](const uint32_t globalIdx)
			{
				return FluidPositionAtArrayIdx(globalFluid, globalIdx);
			},
			[&](const uint32_t N_local)
			{
				localState.ResizeFluid(N_local);
			},
			[&](const uint32_t globalIdx, const uint32_t localIdx)
			{
				CopyFluidParticle_GlobalToLocal(globalFluid, localFluid, globalIdx, localIdx);
			},
			domainDecomp, haloBounds, rank, N_owned, N_halo);

		// classify the compact prefix as inner or outer, then mark the suffix as halo
		ClassifyOwnedFluidParticlesIntoInnerOuter(
			particleState, localFluid, domainDecomp, rank, haloRadius);
		ClassifyHaloParticleRange(particleState, N_owned, N_owned + N_halo);
	}

	// build compact rank-local boundary storage as [ owned particles ][ halo particles ]
	void BuildLocalBoundaryParticlesFromGlobalState(
		DistParticleState& particleState,
		Sync::UniversalSimState& localState,
		const Sync::UniversalSimState& globalState,
		const IDomainDecomp& domainDecomp,
		const int rank,
		const float haloRadius)
	{
		const Sync::BoundaryParticles& globalBoundary = globalState.particlesB;
		Sync::BoundaryParticles& localBoundary = localState.particlesB;

		const AABB haloBounds = domainDecomp.HaloExtentAABB(rank);

		// reset metadata for the new compact layout
		particleState.Clear();

		// particle IDs hold the global scenario index, so a rank-local build slice still maps back globally

		// fill the compact prefix with particles owned by this rank, then append
		// non-owned particles inside the halo bounds of this rank
		uint32_t N_owned = 0;
		uint32_t N_halo = 0;
		BuildLocalParticleRangesFromGlobalState_Parallel(
			globalBoundary.N_,
			[&](const uint32_t globalIdx)
			{
				return BoundaryPositionAtArrayIdx(globalBoundary, globalIdx);
			},
			[&](const uint32_t N_local)
			{
				localState.ResizeBoundary(N_local);
			},
			[&](const uint32_t globalIdx, const uint32_t localIdx)
			{
				CopyBoundaryParticle_GlobalToLocal(
					globalBoundary, localBoundary, globalIdx, localIdx);
			},
			domainDecomp, haloBounds, rank, N_owned, N_halo);

		// classify the compact prefix as inner or outer, then mark the suffix as halo
		ClassifyOwnedBoundaryParticlesIntoInnerOuter(
			particleState, localBoundary, domainDecomp, rank, haloRadius);
		ClassifyHaloParticleRange(particleState, N_owned, N_owned + N_halo);
	}

} // namespace anonymous

void DistParticleState::Clear()
{
	N_owned = 0;
	N_halo = 0;

	particleClasses.clear();
	particleLocalIdx_Owned.clear();
	particleLocalIdx_Inner.clear();
	particleLocalIdx_Outer.clear();
	particleLocalIdx_Halo.clear();
}

bool DistParticleState::HasMetadataAtLocalIdx(const uint32_t localIdx) const
{
	return localIdx < particleClasses.size();
}

// TODO: find opportunities for parallelization with TBB?
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

	// check that counts and class storage match the compact local range
	if (particleClasses.size() != N_local)
	{
		return fail("particleClasses.size() does not match localParticleCount.");
	}
	if (N_owned > N_local ||
		N_halo != N_local - N_owned)
	{
		return fail("ownedParticleCount + haloParticleCount does not match localParticleCount.");
	}

	// check that class index lists match the compact owned and halo split
	if (particleLocalIdx_Owned.size() != N_owned)
	{
		return fail("particleLocalIdx_Owned.size() does not match ownedParticleCount.");
	}
	if (particleLocalIdx_Halo.size() != N_halo)
	{
		return fail("particleLocalIdx_Halo.size() does not match haloParticleCount.");
	}
	if (particleLocalIdx_Inner.size() +
		particleLocalIdx_Outer.size() != N_owned)
	{
		return fail(
			"particleLocalIdx_Inner.size() + particleLocalIdx_Outer.size() "
			"does not match ownedParticleCount.");
	}

	const uint32_t haloStartLocalIdx = N_owned;

	// check that particle class layout is [ owned ][ halo ] in every build
	if (Check::CountViolations(0, haloStartLocalIdx, [&](const uint32_t localIdx)
	{
		const DistParticleClass particleClass = particleClasses[localIdx];
		return particleClass != DistParticleClass::Inner &&
			   particleClass != DistParticleClass::Outer;
	}) != 0)
	{
		return fail("owned local index range contains a particle class that is not Inner or Outer.");
	}
	if (Check::CountViolations(haloStartLocalIdx, N_local, [&](const uint32_t localIdx)
	{
		return particleClasses[localIdx] != DistParticleClass::Halo;
	}) != 0)
	{
		return fail("halo local index range contains a particle class that is not Halo.");
	}

	#if defined(_DEBUG)

	// check that index lists cover the compact owned and halo ranges in ascending order
	if (!LocalIdxListCoversRange(particleLocalIdx_Owned, 0, haloStartLocalIdx))
	{
		return fail("particleLocalIdx_Owned does not cover the owned local index range in ascending order.");
	}
	if (!LocalIdxListCoversRange(particleLocalIdx_Halo, haloStartLocalIdx, N_local))
	{
		return fail("particleLocalIdx_Halo does not cover the halo local index range in ascending order.");
	}
	if (!LocalIdxListsCoverRange(particleLocalIdx_Inner, particleLocalIdx_Outer, 0, haloStartLocalIdx))
	{
		return fail(
			"particleLocalIdx_Inner and particleLocalIdx_Outer do not cover the owned "
			"local index range in ascending order.");
	}

	// check that owned list entries point into the compact owned range
	// TODO: parallelize with TBB?
	for (const uint32_t localIdx : particleLocalIdx_Owned)
	{
		if (!IsLocalIdxInRange(localIdx, 0, haloStartLocalIdx))
		{
			return fail("particleLocalIdx_Owned contains an index outside the owned local index range.");
		}
		const DistParticleClass particleClass = particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer)
		{
			return fail("particleLocalIdx_Owned contains an index whose class is not Inner or Outer.");
		}
	}

	// check that inner list entries point to owned particles marked inner
	// TODO: parallelize with TBB?
	for (const uint32_t localIdx : particleLocalIdx_Inner)
	{
		if (!IsLocalIdxInRange(localIdx, 0, haloStartLocalIdx))
		{
			return fail("particleLocalIdx_Inner contains an index outside the owned local index range.");
		}
		if (particleClasses[localIdx] != DistParticleClass::Inner)
		{
			return fail("particleLocalIdx_Inner contains an index whose class is not Inner.");
		}
	}

	// check that outer list entries point to owned particles marked outer
	// TODO: parallelize with TBB?
	for (const uint32_t localIdx : particleLocalIdx_Outer)
	{
		if (!IsLocalIdxInRange(localIdx, 0, haloStartLocalIdx))
		{
			return fail("particleLocalIdx_Outer contains an index outside the owned local index range.");
		}
		if (particleClasses[localIdx] != DistParticleClass::Outer)
		{
			return fail("particleLocalIdx_Outer contains an index whose class is not Outer.");
		}
	}

	// check that halo list entries point to non-owned particles marked halo
	// TODO: parallelize with TBB?
	for (const uint32_t localIdx : particleLocalIdx_Halo)
	{
		if (!IsLocalIdxInRange(localIdx, haloStartLocalIdx, N_local))
		{
			return fail("particleLocalIdx_Halo contains an index outside the halo local index range.");
		}
		if (particleClasses[localIdx] != DistParticleClass::Halo)
		{
			return fail("particleLocalIdx_Halo contains an index whose class is not Halo.");
		}
	}
	#endif

	return true;
}

bool DistParticleState::IsValidLocalParticleState(
	const std::span<const uint32_t> particleIds,
	const uint32_t N_global,
	std::string* failMsg) const
{
	if (failMsg) { failMsg->clear(); }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};

	if (particleIds.size() > UINT32_MAX)
	{
		return fail("particleIds.size() exceeds uint32_t range.");
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

	// mark the particle IDs in a global index bitmap to detect duplicates
	std::vector<uint32_t> claimBitsScratch{};
	if (Check::FindFirstDuplicateIdx(
		particleIds, N_global, claimBitsScratch) != N_local)
	{
		return fail("particle ids contain a duplicate global index.");
	}
	#else
	(void)N_global;
	#endif

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
	Mem::Add(usage, Mem::MemUsageOf_Vector(particles.xRelPos));
	Mem::Add(usage, Mem::MemUsageOf_Vector(particles.yRelPos));
	Mem::Add(usage, Mem::MemUsageOf_Vector(particles.zRelPos));
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

Mem::MemUsage Dist::GetParticleMemoryUsage(
	const DistParticleState& particleState)
{
	Mem::MemUsage usage{};

	Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleClasses));
	Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_Owned));
	Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_Inner));
	Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_Outer));
	Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_Halo));

	return usage;
}

Mem::MemUsage Dist::GetParticleMemoryUsage(
	const ParticleReOrderWorkBuffers& workBuffers)
{
	Mem::MemUsage usage{};

	Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.particleClasses));
	Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.particleLocalIdx_Owned));
	Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.particleLocalIdx_Inner));
	Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.particleLocalIdx_Outer));
	Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.particleLocalIdx_Halo));
	Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.classification.classCounts_ByChunkClass));
	Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.classification.classWriteOffsets_ByChunkClass));
	Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.classification.classTotals));

	return usage;
}

bool Dist::ApplyLocalParticleReOrder(
	DistParticleState& particleState,
	const std::span<const uint32_t> oldLocalIdx_FromNewLocalIdx,
	ParticleReOrderWorkBuffers* workBuffers)
{
	if (oldLocalIdx_FromNewLocalIdx.size() > UINT32_MAX) { return false; }
	const uint32_t N_local = static_cast<uint32_t>(oldLocalIdx_FromNewLocalIdx.size());

	// check that source metadata covers every old local index
	if (!particleState.IsValidLocalParticleLayout(N_local))
	{
		return false;
	}

	#if defined(_DEBUG)
	// exhaustively check that the reorder map is a complete permutation of the local particle range
	if (!LocalIdxReOrderCoversRange(oldLocalIdx_FromNewLocalIdx, N_local))
	{
		return false;
	}
	#endif

	// use caller-owned scratch when available to retain allocations across calls
	ParticleReOrderWorkBuffers localWorkBuffers{};
	ParticleReOrderWorkBuffers& reOrderWorkBuffers =
		workBuffers ? *workBuffers : localWorkBuffers;
	ParticleClassificationWorkBuffers& classification =
		reOrderWorkBuffers.classification;

	const uint32_t N_chunks = N_local > 0
		? 1 + (N_local - 1) / PARTICLE_CLASSIFICATION_CHUNK_SIZE
		: 0;

	Base::ResizeScratchVector(reOrderWorkBuffers.particleClasses, N_local);
	classification.classCounts_ByChunkClass.assign(
		N_chunks * N_PARTICLE_CLASSIFICATION_CLASSES, 0u);
	classification.classWriteOffsets_ByChunkClass.assign(
		N_chunks * N_PARTICLE_CLASSIFICATION_CLASSES, 0u);
	classification.classTotals.assign(N_PARTICLE_CLASSIFICATION_CLASSES, 0u);

	std::atomic<bool> sourceMetadataValid{ true };

	// re-order particle classes while counting each class per chunk
	// (chunks processed in parallel, with each worker writing only its own output range and count row)
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_chunks),
		[&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
		{
			const uint32_t beginLocalIdx = chunkIdx * PARTICLE_CLASSIFICATION_CHUNK_SIZE;
			const uint32_t endLocalIdx = beginLocalIdx + std::min(
				PARTICLE_CLASSIFICATION_CHUNK_SIZE,
				N_local - beginLocalIdx);
			const uint32_t chunkClassBase =
				chunkIdx * N_PARTICLE_CLASSIFICATION_CLASSES;

			for (uint32_t newLocalIdx = beginLocalIdx; newLocalIdx != endLocalIdx; newLocalIdx++)
			{
				const uint32_t oldLocalIdx = oldLocalIdx_FromNewLocalIdx[newLocalIdx];
				if (oldLocalIdx >= N_local)
				{
					sourceMetadataValid.store(false, std::memory_order_relaxed);
					continue;
				}

				const DistParticleClass particleClass =
					particleState.particleClasses[oldLocalIdx];
				if (particleClass != DistParticleClass::Inner &&
					particleClass != DistParticleClass::Outer &&
					particleClass != DistParticleClass::Halo)
				{
					sourceMetadataValid.store(false, std::memory_order_relaxed);
					continue;
				}

				reOrderWorkBuffers.particleClasses[newLocalIdx] = particleClass;

				if (particleClass == DistParticleClass::Inner)
				{
					classification.classCounts_ByChunkClass[
						chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_OWNED]++;
					classification.classCounts_ByChunkClass[
						chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_INNER]++;
				}
				else if (particleClass == DistParticleClass::Outer)
				{
					classification.classCounts_ByChunkClass[
						chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_OWNED]++;
					classification.classCounts_ByChunkClass[
						chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_OUTER]++;
				}
			}
		}
	});

	if (!sourceMetadataValid.load(std::memory_order_relaxed))
	{
		return false;
	}

	// prefix chunk-local class counts to compute output range of each chunk
	PrefixCountsByChunkAndBucket_ParallelScan(
		classification.classCounts_ByChunkClass,
		classification.classWriteOffsets_ByChunkClass,
		classification.classTotals,
		N_chunks,
		N_PARTICLE_CLASSIFICATION_CLASSES);

	const uint32_t N_owned =
		classification.classTotals[PARTICLE_CLASSIFICATION_CLASS_OWNED];
	const uint32_t N_halo = N_local - N_owned;
	Base::ResizeScratchVector(reOrderWorkBuffers.particleLocalIdx_Owned, N_owned);
	Base::ResizeScratchVector(reOrderWorkBuffers.particleLocalIdx_Inner,
		classification.classTotals[PARTICLE_CLASSIFICATION_CLASS_INNER]);
	Base::ResizeScratchVector(reOrderWorkBuffers.particleLocalIdx_Outer,
		classification.classTotals[PARTICLE_CLASSIFICATION_CLASS_OUTER]);
	Base::ResizeScratchVector(reOrderWorkBuffers.particleLocalIdx_Halo, N_halo);

	// fill owned/inner/outer/halo index lists while preserving new local-index order
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
				N_local - beginLocalIdx);
			const uint32_t chunkClassBase =
				chunkIdx * N_PARTICLE_CLASSIFICATION_CLASSES;
			uint32_t ownedWriteIdx =
				classification.classWriteOffsets_ByChunkClass[
					chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_OWNED];
			uint32_t innerWriteIdx =
				classification.classWriteOffsets_ByChunkClass[
					chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_INNER];
			uint32_t outerWriteIdx =
				classification.classWriteOffsets_ByChunkClass[
					chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_OUTER];
			uint32_t haloWriteIdx = beginLocalIdx - ownedWriteIdx;

			for (uint32_t localIdx = beginLocalIdx; localIdx != endLocalIdx; localIdx++)
			{
				const DistParticleClass particleClass =
					reOrderWorkBuffers.particleClasses[localIdx];

				if (particleClass == DistParticleClass::Inner)
				{
					reOrderWorkBuffers.particleLocalIdx_Owned[ownedWriteIdx++] = localIdx;
					reOrderWorkBuffers.particleLocalIdx_Inner[innerWriteIdx++] = localIdx;
				}
				else if (particleClass == DistParticleClass::Outer)
				{
					reOrderWorkBuffers.particleLocalIdx_Owned[ownedWriteIdx++] = localIdx;
					reOrderWorkBuffers.particleLocalIdx_Outer[outerWriteIdx++] = localIdx;
				}
				else
				{
					reOrderWorkBuffers.particleLocalIdx_Halo[haloWriteIdx++] = localIdx;
				}
			}
		}
	});

	const uint32_t oldN_owned = particleState.N_owned;
	const uint32_t oldN_halo = particleState.N_halo;

	const auto swapParticleMetadata = [&]()
	{
		std::swap(particleState.particleClasses, reOrderWorkBuffers.particleClasses);
		std::swap(particleState.particleLocalIdx_Owned, reOrderWorkBuffers.particleLocalIdx_Owned);
		std::swap(particleState.particleLocalIdx_Inner, reOrderWorkBuffers.particleLocalIdx_Inner);
		std::swap(particleState.particleLocalIdx_Outer, reOrderWorkBuffers.particleLocalIdx_Outer);
		std::swap(particleState.particleLocalIdx_Halo, reOrderWorkBuffers.particleLocalIdx_Halo);
	};

	// exchange rebuilt metadata with the old arrays retained as scratch for the next re-order
	swapParticleMetadata();
	particleState.N_owned = N_owned;
	particleState.N_halo = N_halo;

	if (!particleState.IsValidLocalParticleLayout(N_local))
	{
		// restore the original state if rebuilt metadata validation fails
		swapParticleMetadata();
		particleState.N_owned = oldN_owned;
		particleState.N_halo = oldN_halo;
		return false;
	}

	return true;
}

void Dist::ClassifyOwnedFluidParticlesIntoInnerOuter(
	DistParticleState& particleState,
	const Sync::FluidParticles& particlesF,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float haloRadius,
	ParticleClassificationWorkBuffers* workBuffers)
{
	// use caller-owned scratch when available to retain allocations across calls
	ParticleClassificationWorkBuffers localWorkBuffers{};
	ParticleClassificationWorkBuffers& classificationWorkBuffers =
		workBuffers ? *workBuffers : localWorkBuffers;

	ClassifyOwnedParticlesIntoInnerOuter_Parallel(
		particleState,
		particlesF.N_,
		[&](const uint32_t localIdx)
		{
			return FluidPositionAtArrayIdx(particlesF, localIdx);
		},
		domainDecomp,
		rank,
		haloRadius,
		classificationWorkBuffers);
}

void Dist::ClassifyOwnedBoundaryParticlesIntoInnerOuter(
	DistParticleState& particleState,
	const Sync::BoundaryParticles& particlesB,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float haloRadius,
	ParticleClassificationWorkBuffers* workBuffers)
{
	// use caller-owned scratch when available to retain allocations across calls
	ParticleClassificationWorkBuffers localWorkBuffers{};
	ParticleClassificationWorkBuffers& classificationWorkBuffers =
		workBuffers ? *workBuffers : localWorkBuffers;

	ClassifyOwnedParticlesIntoInnerOuter_Parallel(
		particleState,
		particlesB.N_,
		[&](const uint32_t localIdx)
		{
			return BoundaryPositionAtArrayIdx(particlesB, localIdx);
		},
		domainDecomp,
		rank,
		haloRadius,
		classificationWorkBuffers);
}

void Dist::BuildLocalParticlesFromGlobalState(
	Sync::UniversalSimState& localState,
	DistParticleState& particleStateF,
	DistParticleState& particleStateB,
	const Sync::UniversalSimState& globalState,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float haloRadius)
{
	// reset the rank-local state while preserving global step and time
	localState = {};
	localState.step = globalState.step;
	localState.time = globalState.time;
	localState.particlesR = globalState.particlesR;
	localState.rigidBodies = globalState.rigidBodies;

	// build compact rank-local fluid and boundary particle layouts
	BuildLocalFluidParticlesFromGlobalState(
		particleStateF, localState, globalState,
		domainDecomp, rank, haloRadius);
	BuildLocalBoundaryParticlesFromGlobalState(
		particleStateB, localState, globalState,
		domainDecomp, rank, haloRadius);
}
