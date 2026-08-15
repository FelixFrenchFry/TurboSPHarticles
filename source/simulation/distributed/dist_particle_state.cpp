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
#include "dist_particle_state.h"



using namespace Dist;

namespace
{
	glm::vec3 FluidPositionAtArrayIdx(
		const Sync::FluidParticles& fluidParticles,
		const uint32_t arrayIdx)
	{
		return
		{
			fluidParticles.xPos[arrayIdx],
			fluidParticles.yPos[arrayIdx],
			fluidParticles.zPos[arrayIdx]
		};
	}

	glm::vec3 BoundaryPositionAtArrayIdx(
		const Sync::BoundaryParticles& boundaryParticles,
		const uint32_t arrayIdx)
	{
		return
		{
			boundaryParticles.xPos[arrayIdx],
			boundaryParticles.yPos[arrayIdx],
			boundaryParticles.zPos[arrayIdx]
		};
	}

	// reset class metadata and reserve each index list for its largest possible result
	void PrepareParticleClassification(
		DistParticleState& particleState,
		const uint32_t N_particles)
	{
		particleState.ClearParticleClasses();

		particleState.particleClasses.resize(N_particles, DistParticleClass::None);
		particleState.particleLocalIdx_Owned.reserve(N_particles);
		particleState.particleLocalIdx_Inner.reserve(N_particles);
		particleState.particleLocalIdx_Outer.reserve(N_particles);
		particleState.particleLocalIdx_Halo.reserve(N_particles);
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
		// TODO: parallelize this with TBB?
		for (uint32_t classIdx = 0; classIdx != N_PARTICLE_CLASSIFICATION_CLASSES; classIdx++)
		{
			uint32_t N_class = 0;

			for (uint32_t chunkIdx = 0; chunkIdx != N_chunks; chunkIdx++)
			{
				const uint32_t chunkClassIdx = chunkIdx * N_PARTICLE_CLASSIFICATION_CLASSES + classIdx;
				workBuffers.classWriteOffsets_ByChunkClass[chunkClassIdx] = N_class;
				N_class += workBuffers.classCounts_ByChunkClass[chunkClassIdx];
			}

			workBuffers.classTotals[classIdx] = N_class;
		}

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
		const IDomainDecomp& domainDecomp,
		const AABB& haloBounds,
		const int rank)
	{
		const int ownerRank = domainDecomp.OwnerRank(position);
		return ownerRank >= 0 && ownerRank != rank && haloBounds.Contains(position);
	}

	void ClassifyHaloParticleRange(
		DistParticleState& particleState,
		const uint32_t N_owned,
		const uint32_t N_local)
	{
		particleState.N_owned = N_owned;
		particleState.N_halo = N_local - N_owned;

		// TODO: parallelize this with TBB
		for (uint32_t localIdx = N_owned; localIdx != N_local; localIdx++)
		{
			particleState.particleLocalIdx_Halo.push_back(localIdx);
			particleState.particleClasses[localIdx] = DistParticleClass::Halo;
		}
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
		for (const uint32_t localIdx : localIdxs)
		{
			if (!IsLocalIdxInRange(localIdx, firstLocalIdx, lastLocalIdx_Excl)) { return false; }

			const uint32_t coveredIdx = localIdx - firstLocalIdx;
			if (covered[coveredIdx] != 0) { return false; }
			covered[coveredIdx] = 1;
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

		for (const uint32_t localIdx : localIdxsA)
		{
			if (!markLocalIdx(localIdx)) { return false; }
		}

		for (const uint32_t localIdx : localIdxsB)
		{
			if (!markLocalIdx(localIdx)) { return false; }
		}

		return true;
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

		// mark each old local index once to verify that the reorder map is a permutation
		// TODO: parallelize the reorder map validation with TBB?
		std::vector<uint8_t> covered(N_local, 0);
		for (const uint32_t oldLocalIdx : oldLocalIdx_FromNewLocalIdx)
		{
			if (oldLocalIdx >= N_local) { return false; }
			if (covered[oldLocalIdx] != 0) { return false; }
			covered[oldLocalIdx] = 1;
		}

		return true;
	}
	#endif

	uint32_t CountOwnedFluidParticles(
		const Sync::FluidParticles& fluidParticles,
		const IDomainDecomp& domainDecomp,
		const int rank)
	{
		uint32_t N_owned = 0;

		// TODO: parallelize this global particle scan with TBB?
		for (uint32_t particleIdx = 0; particleIdx != fluidParticles.N_; particleIdx++)
		{
			if (domainDecomp.OwnerRank(FluidPositionAtArrayIdx(
				fluidParticles, particleIdx)) == rank)
			{
				N_owned++;
			}
		}

		return N_owned;
	}

	uint32_t CountHaloFluidParticles(
		const Sync::FluidParticles& fluidParticles,
		const IDomainDecomp& domainDecomp,
		const AABB& haloBounds,
		const int rank)
	{
		uint32_t N_halo = 0;

		// TODO: parallelize this global particle scan with TBB?
		for (uint32_t particleIdx = 0; particleIdx != fluidParticles.N_; particleIdx++)
		{
			if (IsHaloParticleForRank(
				FluidPositionAtArrayIdx(fluidParticles, particleIdx),
				domainDecomp, haloBounds, rank))
			{
				N_halo++;
			}
		}

		return N_halo;
	}

	uint32_t CountOwnedBoundaryParticles(
		const Sync::BoundaryParticles& boundaryParticles,
		const IDomainDecomp& domainDecomp,
		const int rank)
	{
		uint32_t N_owned = 0;

		// TODO: parallelize this global particle scan with TBB?
		for (uint32_t particleIdx = 0; particleIdx != boundaryParticles.N_; particleIdx++)
		{
			if (domainDecomp.OwnerRank(BoundaryPositionAtArrayIdx(
				boundaryParticles, particleIdx)) == rank)
			{
				N_owned++;
			}
		}

		return N_owned;
	}

	uint32_t CountHaloBoundaryParticles(
		const Sync::BoundaryParticles& boundaryParticles,
		const IDomainDecomp& domainDecomp,
		const AABB& haloBounds,
		const int rank)
	{
		uint32_t N_halo = 0;

		// TODO: parallelize this global particle scan with TBB?
		for (uint32_t particleIdx = 0; particleIdx != boundaryParticles.N_; particleIdx++)
		{
			if (IsHaloParticleForRank(
				BoundaryPositionAtArrayIdx(boundaryParticles, particleIdx),
				domainDecomp, haloBounds, rank))
			{
				N_halo++;
			}
		}

		return N_halo;
	}

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

	// build compact rank-local fluid storage as [ owned particles ][ halo particles ]
	void BuildLocalFluidParticlesFromGlobalState(
		DistParticleState& particleState,
		Sync::UniversalSimState& localState,
		const Sync::UniversalSimState& globalState,
		const IDomainDecomp& domainDecomp,
		const int rank,
		const float haloRadius,
		const uint32_t N_globalF)
	{
		const Sync::FluidParticles& globalFluid = globalState.fluidParticles;
		Sync::FluidParticles& localFluid = localState.fluidParticles;

		// count both compact ranges before allocating exact local particle storage
		const AABB haloBounds = domainDecomp.HaloExtentAABB(rank);
		const uint32_t N_owned = CountOwnedFluidParticles(globalFluid, domainDecomp, rank);
		const uint32_t N_halo = CountHaloFluidParticles(globalFluid, domainDecomp, haloBounds, rank);
		const uint32_t N_local = N_owned + N_halo;

		// reset local storage and metadata for the new compact layout
		localState.ResizeFluid(N_local);
		particleState.Clear();
		particleState.particleGlobalIdx_FromLocalIdx.reserve(N_local);
		particleState.activeGlobalIdxFlags.assign(N_globalF, 0);

		// particle IDs hold the global scenario index, so a rank-local build slice still maps back globally

		// copy particles owned by this rank into the compact prefix
		// TODO: parallelize this global particle scan with TBB?
		uint32_t localIdx = 0;
		for (uint32_t srcIdx = 0; srcIdx != globalFluid.N_; srcIdx++)
		{
			if (domainDecomp.OwnerRank(FluidPositionAtArrayIdx(globalFluid, srcIdx)) != rank) { continue; }

			CopyFluidParticle_GlobalToLocal(globalFluid, localFluid, srcIdx, localIdx);

			const uint32_t globalIdx = globalFluid.id[srcIdx];
			particleState.particleGlobalIdx_FromLocalIdx.push_back(globalIdx);
			particleState.activeGlobalIdxFlags[globalIdx] = 1;
			localIdx++;
		}

		// append non-owned particles inside the halo bounds of this rank
		// TODO: parallelize this global particle scan with TBB?
		for (uint32_t srcIdx = 0; srcIdx != globalFluid.N_; srcIdx++)
		{
			if (!IsHaloParticleForRank(
				FluidPositionAtArrayIdx(globalFluid, srcIdx),
				domainDecomp, haloBounds, rank)) { continue; }

			CopyFluidParticle_GlobalToLocal(globalFluid, localFluid, srcIdx, localIdx);

			const uint32_t globalIdx = globalFluid.id[srcIdx];
			particleState.particleGlobalIdx_FromLocalIdx.push_back(globalIdx);
			particleState.activeGlobalIdxFlags[globalIdx] = 1;
			localIdx++;
		}

		// classify the compact prefix as inner or outer, then mark the suffix as halo
		ClassifyOwnedFluidParticlesIntoInnerOuter(
			particleState, localFluid, domainDecomp, rank, haloRadius);
		ClassifyHaloParticleRange(particleState, N_owned, N_local);
	}

	// build compact rank-local boundary storage as [ owned particles ][ halo particles ]
	void BuildLocalBoundaryParticlesFromGlobalState(
		DistParticleState& particleState,
		Sync::UniversalSimState& localState,
		const Sync::UniversalSimState& globalState,
		const IDomainDecomp& domainDecomp,
		const int rank,
		const float haloRadius,
		const uint32_t N_globalB)
	{
		const Sync::BoundaryParticles& globalBoundary = globalState.boundaryParticles;
		Sync::BoundaryParticles& localBoundary = localState.boundaryParticles;

		// count both compact ranges before allocating exact local particle storage
		const AABB haloBounds = domainDecomp.HaloExtentAABB(rank);
		const uint32_t N_owned = CountOwnedBoundaryParticles(globalBoundary, domainDecomp, rank);
		const uint32_t N_halo = CountHaloBoundaryParticles(globalBoundary, domainDecomp, haloBounds, rank);
		const uint32_t N_local = N_owned + N_halo;

		// reset local storage and metadata for the new compact layout
		localState.ResizeBoundary(N_local);
		particleState.Clear();
		particleState.particleGlobalIdx_FromLocalIdx.reserve(N_local);
		particleState.activeGlobalIdxFlags.assign(N_globalB, 0);

		// particle IDs hold the global scenario index, so a rank-local build slice still maps back globally

		// copy particles owned by this rank into the compact prefix
		// TODO: parallelize this global particle scan with TBB?
		uint32_t localIdx = 0;
		for (uint32_t srcIdx = 0; srcIdx != globalBoundary.N_; srcIdx++)
		{
			if (domainDecomp.OwnerRank(BoundaryPositionAtArrayIdx(globalBoundary, srcIdx)) != rank) { continue; }

			CopyBoundaryParticle_GlobalToLocal(globalBoundary, localBoundary, srcIdx, localIdx);

			const uint32_t globalIdx = globalBoundary.id[srcIdx];
			particleState.particleGlobalIdx_FromLocalIdx.push_back(globalIdx);
			particleState.activeGlobalIdxFlags[globalIdx] = 1;
			localIdx++;
		}

		// append non-owned particles inside the halo bounds of this rank
		// TODO: parallelize this global particle scan with TBB?
		for (uint32_t srcIdx = 0; srcIdx != globalBoundary.N_; srcIdx++)
		{
			if (!IsHaloParticleForRank(
				BoundaryPositionAtArrayIdx(globalBoundary, srcIdx),
				domainDecomp, haloBounds, rank)) { continue; }

			CopyBoundaryParticle_GlobalToLocal(globalBoundary, localBoundary, srcIdx, localIdx);

			const uint32_t globalIdx = globalBoundary.id[srcIdx];
			particleState.particleGlobalIdx_FromLocalIdx.push_back(globalIdx);
			particleState.activeGlobalIdxFlags[globalIdx] = 1;
			localIdx++;
		}

		// classify the compact prefix as inner or outer, then mark the suffix as halo
		ClassifyOwnedBoundaryParticlesIntoInnerOuter(
			particleState, localBoundary, domainDecomp, rank, haloRadius);
		ClassifyHaloParticleRange(particleState, N_owned, N_local);
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
	particleGlobalIdx_FromLocalIdx.clear();
	activeGlobalIdxFlags.clear();
}

void DistParticleState::ClearParticleClasses()
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
	return localIdx < particleClasses.size() &&
		localIdx < particleGlobalIdx_FromLocalIdx.size();
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

	// check that particle class layout adheres to [ owned ][ halo ] in every build
	// TODO: parallelize with TBB?
	for (uint32_t localIdx = 0; localIdx != haloStartLocalIdx; localIdx++)
	{
		const DistParticleClass particleClass = particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer)
		{
			return fail("owned local index range contains a particle class that is not Inner or Outer.");
		}
	}
	for (uint32_t localIdx = haloStartLocalIdx; localIdx != N_local; localIdx++)
	{
		if (particleClasses[localIdx] != DistParticleClass::Halo)
		{
			return fail("halo local index range contains a particle class that is not Halo.");
		}
	}

	#if defined(_DEBUG)

	// check that index lists cover the compact owned and halo ranges
	if (!LocalIdxListCoversRange(particleLocalIdx_Owned, 0, haloStartLocalIdx))
	{
		return fail("particleLocalIdx_Owned does not cover the owned local index range.");
	}
	if (!LocalIdxListCoversRange(particleLocalIdx_Halo, haloStartLocalIdx, N_local))
	{
		return fail("particleLocalIdx_Halo does not cover the halo local index range.");
	}
	if (!LocalIdxListsCoverRange(particleLocalIdx_Inner, particleLocalIdx_Outer, 0, haloStartLocalIdx))
	{
		return fail("particleLocalIdx_Inner and particleLocalIdx_Outer do not cover the owned local index range.");
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
	const uint32_t N_local,
	const uint32_t N_global,
	std::string* failMsg) const
{
	if (failMsg) { failMsg->clear(); }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};

	// check that counts and class storage match the active local range
	if (particleClasses.size() != N_local)
	{
		return fail("particleClasses.size() does not match localParticleCount.");
	}
	if (N_owned > N_local ||
		N_halo != N_local - N_owned)
	{
		return fail("ownedParticleCount + haloParticleCount does not match localParticleCount.");
	}

	// check that class index lists match the owned and halo particle counts
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

	// check that index metadata covers the local and global particle domains
	if (particleGlobalIdx_FromLocalIdx.size() != N_local)
	{
		return fail("particleGlobalIdx_FromLocalIdx.size() does not match localParticleCount.");
	}
	if (activeGlobalIdxFlags.size() != N_global)
	{
		return fail("activeGlobalIdxFlags.size() does not match globalParticleCount.");
	}

	#if defined(_DEBUG)
	std::vector<uint8_t> activeLocalIdxs(N_local, 0);

	// check that owned list entries are unique active inner or outer particles
	for (const uint32_t localIdx : particleLocalIdx_Owned)
	{
		if (!IsLocalIdxInRange(localIdx, 0, N_local))
		{
			return fail("particleLocalIdx_Owned contains an index outside the local particle range.");
		}
		if (activeLocalIdxs[localIdx] != 0)
		{
			return fail("particleLocalIdx_Owned contains a duplicate active local index.");
		}

		const DistParticleClass particleClass = particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer)
		{
			return fail("particleLocalIdx_Owned contains an index whose class is not Inner or Outer.");
		}

		activeLocalIdxs[localIdx] = 1;
	}

	// check that halo list entries are unique active halo particles
	for (const uint32_t localIdx : particleLocalIdx_Halo)
	{
		if (!IsLocalIdxInRange(localIdx, 0, N_local))
		{
			return fail("particleLocalIdx_Halo contains an index outside the local particle range.");
		}
		if (activeLocalIdxs[localIdx] != 0)
		{
			return fail("particleLocalIdx_Halo contains a duplicate active local index.");
		}
		if (particleClasses[localIdx] != DistParticleClass::Halo)
		{
			return fail("particleLocalIdx_Halo contains an index whose class is not Halo.");
		}

		activeLocalIdxs[localIdx] = 1;
	}
	// check that every active local index appears in either owned or halo
	for (uint32_t localIdx = 0; localIdx != N_local; localIdx++)
	{
		if (activeLocalIdxs[localIdx] == 0)
		{
			return fail("owned and halo local index lists do not cover every active local index.");
		}
	}

	std::vector<uint8_t> ownedLocalIdxs(N_local, 0);

	// check that inner list entries are unique owned particles marked inner
	for (const uint32_t localIdx : particleLocalIdx_Inner)
	{
		if (!IsLocalIdxInRange(localIdx, 0, N_local))
		{
			return fail("particleLocalIdx_Inner contains an index outside the local particle range.");
		}
		if (ownedLocalIdxs[localIdx] != 0)
		{
			return fail("particleLocalIdx_Inner contains a duplicate owned local index.");
		}
		if (particleClasses[localIdx] != DistParticleClass::Inner)
		{
			return fail("particleLocalIdx_Inner contains an index whose class is not Inner.");
		}

		ownedLocalIdxs[localIdx] = 1;
	}

	// check that outer list entries are unique owned particles marked outer
	for (const uint32_t localIdx : particleLocalIdx_Outer)
	{
		if (!IsLocalIdxInRange(localIdx, 0, N_local))
		{
			return fail("particleLocalIdx_Outer contains an index outside the local particle range.");
		}
		if (ownedLocalIdxs[localIdx] != 0)
		{
			return fail("particleLocalIdx_Outer contains a duplicate owned local index.");
		}
		if (particleClasses[localIdx] != DistParticleClass::Outer)
		{
			return fail("particleLocalIdx_Outer contains an index whose class is not Outer.");
		}

		ownedLocalIdxs[localIdx] = 1;
	}

	// check that every owned local index appears in either inner or outer
	for (const uint32_t localIdx : particleLocalIdx_Owned)
	{
		if (ownedLocalIdxs[localIdx] == 0)
		{
			return fail("inner and outer local index lists do not cover every owned local index.");
		}
	}

	// check that every local-to-global entry lands on a flagged global index
	for (uint32_t localIdx = 0; localIdx != N_local; localIdx++)
	{
		const uint32_t globalIdx = particleGlobalIdx_FromLocalIdx[localIdx];
		if (globalIdx >= N_global)
		{
			return fail("particleGlobalIdx_FromLocalIdx contains an index outside the global particle range.");
		}
		if (activeGlobalIdxFlags[globalIdx] == 0)
		{
			return fail("activeGlobalIdxFlags does not mark an active globalIdx.");
		}
	}

	// count active global flags to detect duplicate local-to-global mappings
	// (distinct mappings produce exactly one active flag per local particle)
	uint32_t N_activeGlobal = 0;
	for (uint32_t globalIdx = 0; globalIdx != N_global; globalIdx++)
	{
		if (activeGlobalIdxFlags[globalIdx] != 0) { N_activeGlobal++; }
	}
	if (N_activeGlobal != N_local)
	{
		return fail("activeGlobalIdxFlags count does not match localParticleCount.");
	}
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

	Mem::Add(usage, GetParticleMemoryUsage(state.fluidParticles));
	Mem::Add(usage, GetParticleMemoryUsage(state.boundaryParticles));
	Mem::Add(usage, GetParticleMemoryUsage(state.rigidParticles));

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
	Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleGlobalIdx_FromLocalIdx));
	Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.activeGlobalIdxFlags));

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
	Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.particleGlobalIdx_FromLocalIdx));
	Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.classification.classCounts_ByChunkClass));
	Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.classification.classWriteOffsets_ByChunkClass));
	Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.classification.classTotals));

	return usage;
}

bool Dist::ApplyLocalParticleReOrder(
	DistParticleState& particleState,
	const std::span<const uint32_t> oldLocalIdx_FromNewLocalIdx,
	const uint32_t N_global,
	ParticleReOrderWorkBuffers* workBuffers)
{
	if (oldLocalIdx_FromNewLocalIdx.size() > UINT32_MAX) { return false; }
	const uint32_t N_local = static_cast<uint32_t>(oldLocalIdx_FromNewLocalIdx.size());

	// check that source metadata covers every old local index
	if (!particleState.IsValidLocalParticleState(N_local, N_global))
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

	reOrderWorkBuffers.particleClasses.resize(N_local);
	reOrderWorkBuffers.particleGlobalIdx_FromLocalIdx.resize(N_local);
	classification.classCounts_ByChunkClass.assign(
		N_chunks * N_PARTICLE_CLASSIFICATION_CLASSES, 0u);
	classification.classWriteOffsets_ByChunkClass.assign(
		N_chunks * N_PARTICLE_CLASSIFICATION_CLASSES, 0u);
	classification.classTotals.assign(N_PARTICLE_CLASSIFICATION_CLASSES, 0u);

	std::atomic<bool> sourceMetadataValid{ true };

	// re-order particle classes and local-to-global mappings while counting each class per chunk
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

				const uint32_t globalIdx =
					particleState.particleGlobalIdx_FromLocalIdx[oldLocalIdx];
				const DistParticleClass particleClass =
					particleState.particleClasses[oldLocalIdx];
				if (globalIdx >= N_global ||
					particleState.activeGlobalIdxFlags[globalIdx] == 0 ||
					(particleClass != DistParticleClass::Inner &&
					particleClass != DistParticleClass::Outer &&
					particleClass != DistParticleClass::Halo))
				{
					sourceMetadataValid.store(false, std::memory_order_relaxed);
					continue;
				}

				reOrderWorkBuffers.particleClasses[newLocalIdx] = particleClass;
				reOrderWorkBuffers.particleGlobalIdx_FromLocalIdx[newLocalIdx] = globalIdx;

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
	for (uint32_t classIdx = 0; classIdx != N_PARTICLE_CLASSIFICATION_CLASSES; classIdx++)
	{
		uint32_t N_class = 0;

		for (uint32_t chunkIdx = 0; chunkIdx != N_chunks; chunkIdx++)
		{
			const uint32_t chunkClassIdx =
				chunkIdx * N_PARTICLE_CLASSIFICATION_CLASSES + classIdx;
			classification.classWriteOffsets_ByChunkClass[chunkClassIdx] = N_class;
			N_class += classification.classCounts_ByChunkClass[chunkClassIdx];
		}

		classification.classTotals[classIdx] = N_class;
	}

	const uint32_t N_owned =
		classification.classTotals[PARTICLE_CLASSIFICATION_CLASS_OWNED];
	const uint32_t N_halo = N_local - N_owned;
	reOrderWorkBuffers.particleLocalIdx_Owned.resize(N_owned);
	reOrderWorkBuffers.particleLocalIdx_Inner.resize(
		classification.classTotals[PARTICLE_CLASSIFICATION_CLASS_INNER]);
	reOrderWorkBuffers.particleLocalIdx_Outer.resize(
		classification.classTotals[PARTICLE_CLASSIFICATION_CLASS_OUTER]);
	reOrderWorkBuffers.particleLocalIdx_Halo.resize(N_halo);

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
		std::swap(particleState.particleGlobalIdx_FromLocalIdx, reOrderWorkBuffers.particleGlobalIdx_FromLocalIdx);
	};

	// exchange rebuilt metadata with the old arrays retained as scratch for the next re-order
	swapParticleMetadata();
	particleState.N_owned = N_owned;
	particleState.N_halo = N_halo;

	if (!particleState.IsValidLocalParticleState(N_local, N_global))
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
	const Sync::FluidParticles& fluidParticles,
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
		fluidParticles.N_,
		[&](const uint32_t localIdx)
		{
			return FluidPositionAtArrayIdx(fluidParticles, localIdx);
		},
		domainDecomp,
		rank,
		haloRadius,
		classificationWorkBuffers);
}

void Dist::ClassifyOwnedBoundaryParticlesIntoInnerOuter(
	DistParticleState& particleState,
	const Sync::BoundaryParticles& boundaryParticles,
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
		boundaryParticles.N_,
		[&](const uint32_t localIdx)
		{
			return BoundaryPositionAtArrayIdx(boundaryParticles, localIdx);
		},
		domainDecomp,
		rank,
		haloRadius,
		classificationWorkBuffers);
}

void Dist::BuildLocalParticlesFromGlobalState(
	Sync::UniversalSimState& localState,
	DistParticleState& fluidParticleState,
	DistParticleState& boundaryParticleState,
	const Sync::UniversalSimState& globalState,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float haloRadius,
	const uint32_t N_globalF,
	const uint32_t N_globalB)
{
	// reset the rank-local state while preserving global step and time
	localState = {};
	localState.step = globalState.step;
	localState.time = globalState.time;
	localState.rigidParticles = globalState.rigidParticles;
	localState.rigidBodies = globalState.rigidBodies;

	// build compact rank-local fluid and boundary particle layouts
	BuildLocalFluidParticlesFromGlobalState(
		fluidParticleState, localState, globalState,
		domainDecomp, rank, haloRadius, N_globalF);
	BuildLocalBoundaryParticlesFromGlobalState(
		boundaryParticleState, localState, globalState,
		domainDecomp, rank, haloRadius, N_globalB);
}
