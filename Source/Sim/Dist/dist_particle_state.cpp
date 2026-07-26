// dist_particle_state.cpp

// external
#include <algorithm>
#include <cstdint>
#include <utility>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// internal
#include "dist_particle_state.h"



using namespace Dist;

namespace
{
	constexpr uint64_t PARTICLE_CLASSIFICATION_CHUNK_SIZE = 4096;
	constexpr uint64_t PARTICLE_CLASSIFICATION_CLASS_COUNT = 3;
	constexpr uint64_t PARTICLE_CLASSIFICATION_CLASS_OWNED = 0;
	constexpr uint64_t PARTICLE_CLASSIFICATION_CLASS_INNER = 1;
	constexpr uint64_t PARTICLE_CLASSIFICATION_CLASS_OUTER = 2;

	// builds a position vector from the given fluid particle array
	// (arrayIdx may be local or global)
	glm::vec3 FluidPositionAtArrayIdx(
		const Sync::FluidParticles& fluidParticles,
		const int arrayIdx)
	{
		return
		{
			fluidParticles.xPos[arrayIdx],
			fluidParticles.yPos[arrayIdx],
			fluidParticles.zPos[arrayIdx]
		};
	}

	// builds a position vector from the given boundary particle array
	// (arrayIdx may be local or global)
	glm::vec3 BoundaryPositionAtArrayIdx(
		const Sync::BoundaryParticles& boundaryParticles,
		const int arrayIdx)
	{
		return
		{
			boundaryParticles.xPos[arrayIdx],
			boundaryParticles.yPos[arrayIdx],
			boundaryParticles.zPos[arrayIdx]
		};
	}

	void PrepareParticleClassification(
		DistParticleState& particleState,
		const uint64_t particleCount)
	{
		particleState.ClearParticleClasses();

		particleState.particleClasses.resize(particleCount, DistParticleClass::None);
		particleState.particleLocalIdx_Owned.reserve(particleCount);
		particleState.particleLocalIdx_Inner.reserve(particleCount);
		particleState.particleLocalIdx_Outer.reserve(particleCount);
		particleState.particleLocalIdx_Halo.reserve(particleCount);
	}

	template <typename PositionAtLocalIdx>
	void ClassifyOwnedParticlesIntoInnerOuter_Parallel(
		DistParticleState& particleState,
		const uint32_t particleCount,
		const PositionAtLocalIdx& positionAtLocalIdx,
		const IDomainDecomp& domainDecomp,
		const int rank,
		const float kernelRadius,
		ParticleClassificationWorkBuffers& workBuffers)
	{
		PrepareParticleClassification(particleState, particleCount);

		const uint64_t particleCount_U64 = particleCount;
		const uint64_t chunkCount = particleCount_U64 > 0
			? (particleCount_U64 + PARTICLE_CLASSIFICATION_CHUNK_SIZE - 1) /
				PARTICLE_CLASSIFICATION_CHUNK_SIZE
			: 0;

		workBuffers.classCounts_ByChunkClass.assign(
			chunkCount * PARTICLE_CLASSIFICATION_CLASS_COUNT, 0ull);
		workBuffers.classWriteOffsets_ByChunkClass.assign(
			chunkCount * PARTICLE_CLASSIFICATION_CLASS_COUNT, 0ull);
		workBuffers.classTotals.assign(PARTICLE_CLASSIFICATION_CLASS_COUNT, 0ull);

		// classify particles and count each class per chunk without shared push_back
		tbb::parallel_for(
			tbb::blocked_range<uint64_t>(0, chunkCount),
			[&](const tbb::blocked_range<uint64_t>& range)
		{
			for (uint64_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint64_t beginLocalIdx =
					chunkIdx * PARTICLE_CLASSIFICATION_CHUNK_SIZE;
				const uint64_t endLocalIdx = std::min(
					beginLocalIdx + PARTICLE_CLASSIFICATION_CHUNK_SIZE,
					particleCount_U64);
				const uint64_t chunkClassBase =
					chunkIdx * PARTICLE_CLASSIFICATION_CLASS_COUNT;

				for (uint64_t localIdx_U64 = beginLocalIdx;
					 localIdx_U64 != endLocalIdx;
					 localIdx_U64++)
				{
					const int localIdx = static_cast<int>(localIdx_U64);
					const glm::vec3 position = positionAtLocalIdx(localIdx);
					if (domainDecomp.OwnerRank(position) != rank) { continue; }

					workBuffers.classCounts_ByChunkClass[
						chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_OWNED]++;

					// inner owned particles do not need to be exported as halo data
					if (domainDecomp.IsInnerOwnedParticle(rank, position, kernelRadius))
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

		// prefix chunk-local class counts so each worker fills disjoint output ranges
		for (uint64_t classIdx = 0; classIdx != PARTICLE_CLASSIFICATION_CLASS_COUNT; classIdx++)
		{
			uint64_t classTotal = 0;
			for (uint64_t chunkIdx = 0; chunkIdx != chunkCount; chunkIdx++)
			{
				const uint64_t chunkClassIdx = chunkIdx * PARTICLE_CLASSIFICATION_CLASS_COUNT + classIdx;
				workBuffers.classWriteOffsets_ByChunkClass[chunkClassIdx] = classTotal;
				classTotal += workBuffers.classCounts_ByChunkClass[chunkClassIdx];
			}
			workBuffers.classTotals[classIdx] = classTotal;
		}

		particleState.particleLocalIdx_Owned.resize(
			workBuffers.classTotals[PARTICLE_CLASSIFICATION_CLASS_OWNED]);
		particleState.particleLocalIdx_Inner.resize(
			workBuffers.classTotals[PARTICLE_CLASSIFICATION_CLASS_INNER]);
		particleState.particleLocalIdx_Outer.resize(
			workBuffers.classTotals[PARTICLE_CLASSIFICATION_CLASS_OUTER]);

		// fill owned / inner / outer index lists while preserving local-index order
		tbb::parallel_for(
			tbb::blocked_range<uint64_t>(0, chunkCount),
			[&](const tbb::blocked_range<uint64_t>& range)
		{
			for (uint64_t chunkIdx = range.begin(); chunkIdx != range.end(); chunkIdx++)
			{
				const uint64_t beginLocalIdx =
					chunkIdx * PARTICLE_CLASSIFICATION_CHUNK_SIZE;
				const uint64_t endLocalIdx = std::min(
					beginLocalIdx + PARTICLE_CLASSIFICATION_CHUNK_SIZE,
					particleCount_U64);
				const uint64_t chunkClassBase =
					chunkIdx * PARTICLE_CLASSIFICATION_CLASS_COUNT;
				uint64_t ownedWriteIdx =
					workBuffers.classWriteOffsets_ByChunkClass[
						chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_OWNED];
				uint64_t innerWriteIdx =
					workBuffers.classWriteOffsets_ByChunkClass[
						chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_INNER];
				uint64_t outerWriteIdx =
					workBuffers.classWriteOffsets_ByChunkClass[
						chunkClassBase + PARTICLE_CLASSIFICATION_CLASS_OUTER];

				for (uint64_t localIdx_U64 = beginLocalIdx;
					 localIdx_U64 != endLocalIdx;
					 localIdx_U64++)
				{
					const int localIdx = static_cast<int>(localIdx_U64);
					const DistParticleClass particleClass =
						particleState.particleClasses[localIdx];

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

		particleState.ownedParticleCount = static_cast<uint32_t>(
			workBuffers.classTotals[PARTICLE_CLASSIFICATION_CLASS_OWNED]);
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
		const uint32_t ownedParticleCount,
		const uint32_t localParticleCount)
	{
		particleState.ownedParticleCount = ownedParticleCount;
		particleState.haloParticleCount = localParticleCount - ownedParticleCount;

		for (uint64_t i = ownedParticleCount; i < static_cast<uint64_t>(localParticleCount); i++)
		{
			const int localIdx = static_cast<int>(i);
			particleState.particleLocalIdx_Halo.push_back(localIdx);
			particleState.particleClasses[localIdx] = DistParticleClass::Halo;
		}
	}

	bool IsLocalIdxInRange(
		const int localIdx,
		const uint32_t firstLocalIdx,
		const uint32_t lastLocalIdx_Excl)
	{
		return localIdx >= 0 &&
			static_cast<uint32_t>(localIdx) >= firstLocalIdx &&
			static_cast<uint32_t>(localIdx) < lastLocalIdx_Excl;
	}

	bool LocalIdxListCoversRange(
		const std::vector<int>& localIdxs,
		const uint32_t firstLocalIdx,
		const uint32_t lastLocalIdx_Excl)
	{
		if (lastLocalIdx_Excl < firstLocalIdx) { return false; }

		const uint32_t expectedCount = lastLocalIdx_Excl - firstLocalIdx;
		if (static_cast<uint64_t>(localIdxs.size()) != expectedCount) { return false; }

		std::vector<uint8_t> covered(expectedCount, 0);
		for (const int localIdx : localIdxs)
		{
			if (!IsLocalIdxInRange(localIdx, firstLocalIdx, lastLocalIdx_Excl)) { return false; }

			const uint32_t coveredIdx = static_cast<uint32_t>(localIdx) - firstLocalIdx;
			if (covered[coveredIdx] != 0) { return false; }
			covered[coveredIdx] = 1;
		}

		return true;
	}

	bool LocalIdxListsCoverRange(
		const std::vector<int>& localIdxsA,
		const std::vector<int>& localIdxsB,
		const uint32_t firstLocalIdx,
		const uint32_t lastLocalIdx_Excl)
	{
		if (lastLocalIdx_Excl < firstLocalIdx) { return false; }

		const uint32_t expectedCount = lastLocalIdx_Excl - firstLocalIdx;
		if (static_cast<uint64_t>(localIdxsA.size()) + localIdxsB.size() != expectedCount)
		{
			return false;
		}

		std::vector<uint8_t> covered(expectedCount, 0);

		const auto markLocalIdx = [&](const int localIdx) -> bool
		{
			if (!IsLocalIdxInRange(localIdx, firstLocalIdx, lastLocalIdx_Excl)) { return false; }

			const uint32_t coveredIdx = static_cast<uint32_t>(localIdx) - firstLocalIdx;
			if (covered[coveredIdx] != 0) { return false; }
			covered[coveredIdx] = 1;
			return true;
		};

		for (const int localIdx : localIdxsA)
		{
			if (!markLocalIdx(localIdx)) { return false; }
		}

		for (const int localIdx : localIdxsB)
		{
			if (!markLocalIdx(localIdx)) { return false; }
		}

		return true;
	}

	bool LocalIdxReOrderCoversRange(
		const std::vector<uint32_t>& oldLocalIdx_FromNewLocalIdx,
		const uint32_t localParticleCount)
	{
		if (static_cast<uint64_t>(oldLocalIdx_FromNewLocalIdx.size()) != localParticleCount)
		{
			return false;
		}

		std::vector<uint8_t> covered(localParticleCount, 0);
		for (const uint32_t oldLocalIdx : oldLocalIdx_FromNewLocalIdx)
		{
			if (oldLocalIdx >= localParticleCount) { return false; }
			if (covered[oldLocalIdx] != 0) { return false; }
			covered[oldLocalIdx] = 1;
		}

		return true;
	}

	// TODO: re-implement with TBB parallelization?
	uint32_t CountOwnedFluidParticles(
		const Sync::FluidParticles& fluidParticles,
		const IDomainDecomp& domainDecomp,
		const int rank)
	{
		uint32_t count = 0;

		for (uint64_t i = 0; i < fluidParticles.N_; i++)
		{
			if (domainDecomp.OwnerRank(FluidPositionAtArrayIdx(fluidParticles, static_cast<int>(i))) == rank)
			{
				count++;
			}
		}

		return count;
	}

	// TODO: re-implement with TBB parallelization?
	uint32_t CountHaloFluidParticles(
		const Sync::FluidParticles& fluidParticles,
		const IDomainDecomp& domainDecomp,
		const AABB& haloBounds,
		const int rank)
	{
		uint32_t count = 0;

		for (uint64_t i = 0; i < fluidParticles.N_; i++)
		{
			if (IsHaloParticleForRank(
				FluidPositionAtArrayIdx(fluidParticles, static_cast<int>(i)),
				domainDecomp, haloBounds, rank))
			{
				count++;
			}
		}

		return count;
	}

	// TODO: re-implement with TBB parallelization?
	uint32_t CountOwnedBoundaryParticles(
		const Sync::BoundaryParticles& boundaryParticles,
		const IDomainDecomp& domainDecomp,
		const int rank)
	{
		uint32_t count = 0;

		for (uint64_t i = 0; i < boundaryParticles.N_; i++)
		{
			if (domainDecomp.OwnerRank(BoundaryPositionAtArrayIdx(boundaryParticles, static_cast<int>(i))) == rank)
			{
				count++;
			}
		}

		return count;
	}

	// TODO: re-implement with TBB parallelization?
	uint32_t CountHaloBoundaryParticles(
		const Sync::BoundaryParticles& boundaryParticles,
		const IDomainDecomp& domainDecomp,
		const AABB& haloBounds,
		const int rank)
	{
		uint32_t count = 0;

		for (uint64_t i = 0; i < boundaryParticles.N_; i++)
		{
			if (IsHaloParticleForRank(
				BoundaryPositionAtArrayIdx(boundaryParticles, static_cast<int>(i)),
				domainDecomp, haloBounds, rank))
			{
				count++;
			}
		}

		return count;
	}

	void CopyFluidParticle_GlobalToLocal(
		const Sync::FluidParticles& src,
		Sync::FluidParticles& dst,
		const int globalIdx,
		const int localIdx)
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

	void CopyBoundaryParticle_GlobalToLocal(
		const Sync::BoundaryParticles& src,
		Sync::BoundaryParticles& dst,
		const int globalIdx,
		const int localIdx)
	{
		dst.id[localIdx] = src.id[globalIdx];
		dst.xPos[localIdx] = src.xPos[globalIdx];
		dst.yPos[localIdx] = src.yPos[globalIdx];
		dst.zPos[localIdx] = src.zPos[globalIdx];
		dst.volB[localIdx] = src.volB[globalIdx];
		dst.psiB[localIdx] = src.psiB[globalIdx];
	}

	// TODO: re-implement with TBB parallelization?
	void BuildLocalFluidParticlesFromGlobalState(
		DistParticleState& particleState,
		Sync::UniversalSimState& localState,
		const Sync::UniversalSimState& globalState,
		const IDomainDecomp& domainDecomp,
		const int rank,
		const float kernelRadius)
	{
		const Sync::FluidParticles& globalFluid = globalState.fluidParticles;
		Sync::FluidParticles& localFluid = localState.fluidParticles;

		const AABB haloBounds = domainDecomp.HaloExtentAABB(rank);
		const uint32_t ownedParticleCount = CountOwnedFluidParticles(globalFluid, domainDecomp, rank);
		const uint32_t haloParticleCount = CountHaloFluidParticles(globalFluid, domainDecomp, haloBounds, rank);
		const uint32_t localParticleCount = ownedParticleCount + haloParticleCount;

		localState.ResizeFluid(localParticleCount);
		particleState.Clear();
		particleState.particleGlobalIdx_FromLocalIdx.reserve(localParticleCount);
		particleState.particleLocalIdx_FromGlobalIdx.assign(globalFluid.N_, -1);

		int localIdx = 0;
		for (uint64_t i = 0; i < globalFluid.N_; i++)
		{
			const int globalIdx = static_cast<int>(i);
			if (domainDecomp.OwnerRank(FluidPositionAtArrayIdx(globalFluid, globalIdx)) != rank) { continue; }

			CopyFluidParticle_GlobalToLocal(globalFluid, localFluid, globalIdx, localIdx);

			particleState.particleGlobalIdx_FromLocalIdx.push_back(globalIdx);
			particleState.particleLocalIdx_FromGlobalIdx[globalIdx] = localIdx;
			localIdx++;
		}

		for (uint64_t i = 0; i < globalFluid.N_; i++)
		{
			const int globalIdx = static_cast<int>(i);
			if (!IsHaloParticleForRank(
				FluidPositionAtArrayIdx(globalFluid, globalIdx),
				domainDecomp, haloBounds, rank)) { continue; }

			CopyFluidParticle_GlobalToLocal(globalFluid, localFluid, globalIdx, localIdx);

			particleState.particleGlobalIdx_FromLocalIdx.push_back(globalIdx);
			particleState.particleLocalIdx_FromGlobalIdx[globalIdx] = localIdx;
			localIdx++;
		}

		// classify after compaction so cached indices are localIdx values
		ClassifyOwnedFluidParticlesIntoInnerOuter(
			particleState, localFluid, domainDecomp, rank, kernelRadius);
		ClassifyHaloParticleRange(particleState, ownedParticleCount, localParticleCount);
	}

	// TODO: re-implement with TBB parallelization?
	void BuildLocalBoundaryParticlesFromGlobalState(
		DistParticleState& particleState,
		Sync::UniversalSimState& localState,
		const Sync::UniversalSimState& globalState,
		const IDomainDecomp& domainDecomp,
		const int rank,
		const float kernelRadius)
	{
		const Sync::BoundaryParticles& globalBoundary = globalState.boundaryParticles;
		Sync::BoundaryParticles& localBoundary = localState.boundaryParticles;

		const AABB haloBounds = domainDecomp.HaloExtentAABB(rank);
		const uint32_t ownedParticleCount = CountOwnedBoundaryParticles(globalBoundary, domainDecomp, rank);
		const uint32_t haloParticleCount = CountHaloBoundaryParticles(globalBoundary, domainDecomp, haloBounds, rank);
		const uint32_t localParticleCount = ownedParticleCount + haloParticleCount;

		localState.ResizeBoundary(localParticleCount);
		particleState.Clear();
		particleState.particleGlobalIdx_FromLocalIdx.reserve(localParticleCount);
		particleState.particleLocalIdx_FromGlobalIdx.assign(globalBoundary.N_, -1);

		int localIdx = 0;
		for (uint64_t i = 0; i < globalBoundary.N_; i++)
		{
			const int globalIdx = static_cast<int>(i);
			if (domainDecomp.OwnerRank(BoundaryPositionAtArrayIdx(globalBoundary, globalIdx)) != rank) { continue; }

			CopyBoundaryParticle_GlobalToLocal(globalBoundary, localBoundary, globalIdx, localIdx);

			particleState.particleGlobalIdx_FromLocalIdx.push_back(globalIdx);
			particleState.particleLocalIdx_FromGlobalIdx[globalIdx] = localIdx;
			localIdx++;
		}

		for (uint64_t i = 0; i < globalBoundary.N_; i++)
		{
			const int globalIdx = static_cast<int>(i);
			if (!IsHaloParticleForRank(
				BoundaryPositionAtArrayIdx(globalBoundary, globalIdx),
				domainDecomp, haloBounds, rank)) { continue; }

			CopyBoundaryParticle_GlobalToLocal(globalBoundary, localBoundary, globalIdx, localIdx);

			particleState.particleGlobalIdx_FromLocalIdx.push_back(globalIdx);
			particleState.particleLocalIdx_FromGlobalIdx[globalIdx] = localIdx;
			localIdx++;
		}

		// classify after compaction so cached indices are localIdx values
		ClassifyOwnedBoundaryParticlesIntoInnerOuter(
			particleState, localBoundary, domainDecomp, rank, kernelRadius);
		ClassifyHaloParticleRange(particleState, ownedParticleCount, localParticleCount);
	}

} // namespace anonymous

void DistParticleState::Clear()
{
	ownedParticleCount = 0;
	haloParticleCount = 0;

	particleClasses.clear();
	particleLocalIdx_Owned.clear();
	particleLocalIdx_Inner.clear();
	particleLocalIdx_Outer.clear();
	particleLocalIdx_Halo.clear();
	particleGlobalIdx_FromLocalIdx.clear();
	particleLocalIdx_FromGlobalIdx.clear();
}

void DistParticleState::ClearParticleClasses()
{
	ownedParticleCount = 0;
	haloParticleCount = 0;

	particleClasses.clear();
	particleLocalIdx_Owned.clear();
	particleLocalIdx_Inner.clear();
	particleLocalIdx_Outer.clear();
	particleLocalIdx_Halo.clear();
}

bool DistParticleState::IsValidLocalLayout(
	const uint32_t localParticleCount,
	std::string* failMsg) const
{
	if (failMsg) { failMsg->clear(); }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};

	const uint32_t haloStartLocalIdx = ownedParticleCount;

	// core counts and class storage must match the compact local range
	if (static_cast<uint64_t>(particleClasses.size()) != localParticleCount)
	{
		return fail("particleClasses.size() does not match localParticleCount.");
	}
	if (static_cast<uint64_t>(ownedParticleCount) + haloParticleCount != localParticleCount)
	{
		return fail("ownedParticleCount + haloParticleCount does not match localParticleCount.");
	}

	// class index lists must have the expected compact owned / halo split
	if (static_cast<uint64_t>(particleLocalIdx_Owned.size()) != ownedParticleCount)
	{
		return fail("particleLocalIdx_Owned.size() does not match ownedParticleCount.");
	}
	if (static_cast<uint64_t>(particleLocalIdx_Halo.size()) != haloParticleCount)
	{
		return fail("particleLocalIdx_Halo.size() does not match haloParticleCount.");
	}
	if (static_cast<uint64_t>(particleLocalIdx_Inner.size()) +
		static_cast<uint64_t>(particleLocalIdx_Outer.size()) != ownedParticleCount)
	{
		return fail(
			"particleLocalIdx_Inner.size() + particleLocalIdx_Outer.size() "
			"does not match ownedParticleCount.");
	}

	// index lists must cover the compact owned and halo ranges
	if (!LocalIdxListCoversRange(particleLocalIdx_Owned, 0, haloStartLocalIdx))
	{
		return fail("particleLocalIdx_Owned does not cover the owned local index range.");
	}
	if (!LocalIdxListCoversRange(particleLocalIdx_Halo, haloStartLocalIdx, localParticleCount))
	{
		return fail("particleLocalIdx_Halo does not cover the halo local index range.");
	}
	if (!LocalIdxListsCoverRange(particleLocalIdx_Inner, particleLocalIdx_Outer, 0, haloStartLocalIdx))
	{
		return fail("particleLocalIdx_Inner and particleLocalIdx_Outer do not cover the owned local index range.");
	}

	// compact owned range must contain only inner or outer particles
	for (uint64_t i = 0; i < haloStartLocalIdx; i++)
	{
		const DistParticleClass particleClass = particleClasses[i];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer)
		{
			return fail("owned local index range contains a particle class that is not Inner or Outer.");
		}
	}

	// compact halo range must contain only halo particles
	for (uint64_t i = haloStartLocalIdx; i < localParticleCount; i++)
	{
		if (particleClasses[i] != DistParticleClass::Halo)
		{
			return fail("halo local index range contains a particle class that is not Halo.");
		}
	}

	// owned list entries must point into the compact owned range
	for (const int localIdx : particleLocalIdx_Owned)
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

	// inner list entries must point to owned particles marked inner
	for (const int localIdx : particleLocalIdx_Inner)
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

	// outer list entries must point to owned particles marked outer
	for (const int localIdx : particleLocalIdx_Outer)
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

	// halo list entries must point to non-owned particles marked halo
	for (const int localIdx : particleLocalIdx_Halo)
	{
		if (!IsLocalIdxInRange(localIdx, haloStartLocalIdx, localParticleCount))
		{
			return fail("particleLocalIdx_Halo contains an index outside the halo local index range.");
		}
		if (particleClasses[localIdx] != DistParticleClass::Halo)
		{
			return fail("particleLocalIdx_Halo contains an index whose class is not Halo.");
		}
	}

	return true;
}

bool DistParticleState::IsValidLocalParticleState(
	const uint32_t localParticleCount,
	const uint32_t globalParticleCount,
	std::string* failMsg) const
{
	if (failMsg) { failMsg->clear(); }

	const auto fail = [failMsg](std::string message) -> bool
	{
		if (failMsg) { *failMsg = std::move(message); }
		return false;
	};

	// core counts and class storage must match the active local range
	if (static_cast<uint64_t>(particleClasses.size()) != localParticleCount)
	{
		return fail("particleClasses.size() does not match localParticleCount.");
	}
	if (static_cast<uint64_t>(ownedParticleCount) + haloParticleCount != localParticleCount)
	{
		return fail("ownedParticleCount + haloParticleCount does not match localParticleCount.");
	}

	// class index lists must have the expected owned / halo split
	if (static_cast<uint64_t>(particleLocalIdx_Owned.size()) != ownedParticleCount)
	{
		return fail("particleLocalIdx_Owned.size() does not match ownedParticleCount.");
	}
	if (static_cast<uint64_t>(particleLocalIdx_Halo.size()) != haloParticleCount)
	{
		return fail("particleLocalIdx_Halo.size() does not match haloParticleCount.");
	}
	if (static_cast<uint64_t>(particleLocalIdx_Inner.size()) +
		static_cast<uint64_t>(particleLocalIdx_Outer.size()) != ownedParticleCount)
	{
		return fail(
			"particleLocalIdx_Inner.size() + particleLocalIdx_Outer.size() "
			"does not match ownedParticleCount.");
	}

	// index maps must be sized for the local and global particle domains
	if (static_cast<uint64_t>(particleGlobalIdx_FromLocalIdx.size()) != localParticleCount)
	{
		return fail("particleGlobalIdx_FromLocalIdx.size() does not match localParticleCount.");
	}
	if (static_cast<uint64_t>(particleLocalIdx_FromGlobalIdx.size()) != globalParticleCount)
	{
		return fail("particleLocalIdx_FromGlobalIdx.size() does not match globalParticleCount.");
	}

	std::vector<uint8_t> activeLocalIdxs(localParticleCount, 0);

	// owned list entries must be unique active inner or outer particles
	for (const int localIdx : particleLocalIdx_Owned)
	{
		if (!IsLocalIdxInRange(localIdx, 0, localParticleCount))
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

	// halo list entries must be unique active halo particles
	for (const int localIdx : particleLocalIdx_Halo)
	{
		if (!IsLocalIdxInRange(localIdx, 0, localParticleCount))
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

	// every active local index must appear in either owned or halo
	for (uint64_t localIdx = 0; localIdx < localParticleCount; localIdx++)
	{
		if (activeLocalIdxs[localIdx] == 0)
		{
			return fail("owned and halo local index lists do not cover every active local index.");
		}
	}

	std::vector<uint8_t> ownedLocalIdxs(localParticleCount, 0);

	// inner list entries must be unique owned particles marked inner
	for (const int localIdx : particleLocalIdx_Inner)
	{
		if (!IsLocalIdxInRange(localIdx, 0, localParticleCount))
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

	// outer list entries must be unique owned particles marked outer
	for (const int localIdx : particleLocalIdx_Outer)
	{
		if (!IsLocalIdxInRange(localIdx, 0, localParticleCount))
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

	// every owned local index must appear in either inner or outer
	for (const int localIdx : particleLocalIdx_Owned)
	{
		if (ownedLocalIdxs[localIdx] == 0)
		{
			return fail("inner and outer local index lists do not cover every owned local index.");
		}
	}

	// local-to-global entries must be unique and round-trip through the inverse map
	std::vector<uint8_t> activeGlobalIdxs(globalParticleCount, 0);
	for (uint64_t localIdx = 0; localIdx < localParticleCount; localIdx++)
	{
		const int globalIdx = particleGlobalIdx_FromLocalIdx[localIdx];
		if (globalIdx < 0 || static_cast<uint32_t>(globalIdx) >= globalParticleCount)
		{
			return fail("particleGlobalIdx_FromLocalIdx contains an index outside the global particle range.");
		}
		if (activeGlobalIdxs[globalIdx] != 0)
		{
			return fail("particleGlobalIdx_FromLocalIdx contains a duplicate active global index.");
		}
		if (particleLocalIdx_FromGlobalIdx[globalIdx] != static_cast<int>(localIdx))
		{
			return fail("particleLocalIdx_FromGlobalIdx does not map active globalIdx back to localIdx.");
		}

		activeGlobalIdxs[globalIdx] = 1;
	}

	// non-empty global-to-local entries must also round-trip through the local map
	for (uint64_t globalIdx = 0; globalIdx < globalParticleCount; globalIdx++)
	{
		const int localIdx = particleLocalIdx_FromGlobalIdx[globalIdx];
		if (localIdx == -1) { continue; }
		if (!IsLocalIdxInRange(localIdx, 0, localParticleCount))
		{
			return fail("particleLocalIdx_FromGlobalIdx contains an index outside the local particle range.");
		}
		if (particleGlobalIdx_FromLocalIdx[localIdx] != static_cast<int>(globalIdx))
		{
			return fail("particleGlobalIdx_FromLocalIdx does not map active localIdx back to globalIdx.");
		}
	}

	return true;
}

bool Dist::ApplyLocalParticleReOrder(
	DistParticleState& particleState,
	const std::vector<uint32_t>& oldLocalIdx_FromNewLocalIdx,
	const uint32_t globalParticleCount)
{
	if (oldLocalIdx_FromNewLocalIdx.size() > UINT32_MAX) { return false; }
	const uint32_t localParticleCount =
		static_cast<uint32_t>(oldLocalIdx_FromNewLocalIdx.size());

	if (!LocalIdxReOrderCoversRange(oldLocalIdx_FromNewLocalIdx, localParticleCount))
	{
		return false;
	}
	if (static_cast<uint64_t>(particleState.particleClasses.size()) != localParticleCount)
	{
		return false;
	}
	if (static_cast<uint64_t>(particleState.particleGlobalIdx_FromLocalIdx.size()) !=
		localParticleCount)
	{
		return false;
	}

	const std::vector<DistParticleClass> oldParticleClasses = particleState.particleClasses;
	const std::vector<int> oldParticleGlobalIdx_FromLocalIdx =
		particleState.particleGlobalIdx_FromLocalIdx;

	DistParticleState reOrderedParticleState{};
	reOrderedParticleState.particleClasses.resize(localParticleCount, DistParticleClass::None);
	reOrderedParticleState.particleGlobalIdx_FromLocalIdx.assign(localParticleCount, -1);
	reOrderedParticleState.particleLocalIdx_FromGlobalIdx.assign(globalParticleCount, -1);
	reOrderedParticleState.particleLocalIdx_Owned.reserve(particleState.particleLocalIdx_Owned.size());
	reOrderedParticleState.particleLocalIdx_Inner.reserve(particleState.particleLocalIdx_Inner.size());
	reOrderedParticleState.particleLocalIdx_Outer.reserve(particleState.particleLocalIdx_Outer.size());
	reOrderedParticleState.particleLocalIdx_Halo.reserve(particleState.particleLocalIdx_Halo.size());

	for (uint64_t newLocalIdx_U64 = 0; newLocalIdx_U64 < localParticleCount; newLocalIdx_U64++)
	{
		const uint32_t newLocalIdx = static_cast<uint32_t>(newLocalIdx_U64);
		const uint32_t oldLocalIdx = oldLocalIdx_FromNewLocalIdx[newLocalIdx];

		const DistParticleClass particleClass = oldParticleClasses[oldLocalIdx];
		const int globalIdx = oldParticleGlobalIdx_FromLocalIdx[oldLocalIdx];
		if (globalIdx < 0 || static_cast<uint32_t>(globalIdx) >= globalParticleCount)
		{
			return false;
		}
		if (reOrderedParticleState.particleLocalIdx_FromGlobalIdx[globalIdx] != -1)
		{
			return false;
		}

		reOrderedParticleState.particleClasses[newLocalIdx] = particleClass;
		reOrderedParticleState.particleGlobalIdx_FromLocalIdx[newLocalIdx] = globalIdx;
		reOrderedParticleState.particleLocalIdx_FromGlobalIdx[globalIdx] =
			static_cast<int>(newLocalIdx);

		switch (particleClass)
		{
		case DistParticleClass::Inner:
			reOrderedParticleState.particleLocalIdx_Owned.push_back(static_cast<int>(newLocalIdx));
			reOrderedParticleState.particleLocalIdx_Inner.push_back(static_cast<int>(newLocalIdx));
			break;

		case DistParticleClass::Outer:
			reOrderedParticleState.particleLocalIdx_Owned.push_back(static_cast<int>(newLocalIdx));
			reOrderedParticleState.particleLocalIdx_Outer.push_back(static_cast<int>(newLocalIdx));
			break;

		case DistParticleClass::Halo:
			reOrderedParticleState.particleLocalIdx_Halo.push_back(static_cast<int>(newLocalIdx));
			break;

		case DistParticleClass::None:
			return false;
		}
	}

	reOrderedParticleState.ownedParticleCount =
		static_cast<uint32_t>(reOrderedParticleState.particleLocalIdx_Owned.size());
	reOrderedParticleState.haloParticleCount =
		static_cast<uint32_t>(reOrderedParticleState.particleLocalIdx_Halo.size());

	if (!reOrderedParticleState.IsValidLocalParticleState(localParticleCount, globalParticleCount))
	{
		return false;
	}

	particleState = std::move(reOrderedParticleState);
	return true;
}

void Dist::ClassifyOwnedFluidParticlesIntoInnerOuter(
	DistParticleState& particleState,
	const Sync::FluidParticles& fluidParticles,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float kernelRadius,
	ParticleClassificationWorkBuffers* workBuffers)
{
	ParticleClassificationWorkBuffers localWorkBuffers{};
	ParticleClassificationWorkBuffers& classificationWorkBuffers =
		workBuffers ? *workBuffers : localWorkBuffers;

	ClassifyOwnedParticlesIntoInnerOuter_Parallel(
		particleState,
		fluidParticles.N_,
		[&](const int localIdx)
		{
			return FluidPositionAtArrayIdx(fluidParticles, localIdx);
		},
		domainDecomp,
		rank,
		kernelRadius,
		classificationWorkBuffers);
}

void Dist::ClassifyOwnedBoundaryParticlesIntoInnerOuter(
	DistParticleState& particleState,
	const Sync::BoundaryParticles& boundaryParticles,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float kernelRadius,
	ParticleClassificationWorkBuffers* workBuffers)
{
	ParticleClassificationWorkBuffers localWorkBuffers{};
	ParticleClassificationWorkBuffers& classificationWorkBuffers =
		workBuffers ? *workBuffers : localWorkBuffers;

	ClassifyOwnedParticlesIntoInnerOuter_Parallel(
		particleState,
		boundaryParticles.N_,
		[&](const int localIdx)
		{
			return BoundaryPositionAtArrayIdx(boundaryParticles, localIdx);
		},
		domainDecomp,
		rank,
		kernelRadius,
		classificationWorkBuffers);
}

void Dist::BuildLocalParticlesFromGlobalState(
	Sync::UniversalSimState& localState,
	DistParticleState& fluidParticleState,
	DistParticleState& boundaryParticleState,
	const Sync::UniversalSimState& globalState,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float kernelRadius)
{
	localState = {};
	localState.step = globalState.step;
	localState.time = globalState.time;

	// rigid particles and bodies are replicated on every rank
	localState.rigidParticles = globalState.rigidParticles;
	localState.rigidBodies = globalState.rigidBodies;

	BuildLocalFluidParticlesFromGlobalState(
		fluidParticleState, localState, globalState,
		domainDecomp, rank, kernelRadius);

	BuildLocalBoundaryParticlesFromGlobalState(
		boundaryParticleState, localState, globalState,
		domainDecomp, rank, kernelRadius);
}
