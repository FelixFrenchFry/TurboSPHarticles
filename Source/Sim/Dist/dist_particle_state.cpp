// dist_particle_state.cpp

// external
#include <cstdint>
#include <utility>

// internal
#include "dist_particle_state.h"



using namespace Dist;

namespace
{
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

	void ClassifyOwnedParticleAtLocalIdx(
		DistParticleState& particleState,
		const glm::vec3& position,
		const int localIdx,
		const IDomainDecomp& domainDecomp,
		const int rank,
		const float kernelRadius)
	{
		if (domainDecomp.OwnerRank(position) != rank) { return; }

		particleState.particleLocalIdx_Owned.push_back(localIdx);

		// inner owned particles do not need to be exported as halo data
		if (domainDecomp.IsInnerOwnedParticle(rank, position, kernelRadius))
		{
			particleState.particleLocalIdx_Inner.push_back(localIdx);
			particleState.particleClasses[localIdx] = DistParticleClass::Inner;
		}
		else
		{
			particleState.particleLocalIdx_Outer.push_back(localIdx);
			particleState.particleClasses[localIdx] = DistParticleClass::Outer;
		}
	}

	bool IsHaloParticleForRank(
		const glm::vec3& position,
		const IDomainDecomp& domainDecomp,
		const AABB& haloBounds,
		const int rank)
	{
		const int ownerRank = domainDecomp.OwnerRank(position);
		return ownerRank >= 0 &&
			ownerRank != rank &&
			haloBounds.Contains(position);
	}

	void ClassifyHaloParticleRange(
		DistParticleState& particleState,
		const uint32_t firstHaloLocalIdx,
		const uint32_t localParticleCount)
	{
		particleState.ownedParticleCount = firstHaloLocalIdx;
		particleState.haloParticleCount = localParticleCount - firstHaloLocalIdx;
		particleState.firstHaloLocalIdx = firstHaloLocalIdx;

		for (uint64_t i = firstHaloLocalIdx;
			 i < static_cast<uint64_t>(localParticleCount);
			 i++)
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
				domainDecomp,
				haloBounds,
				rank))
			{
				count++;
			}
		}

		return count;
	}

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
				domainDecomp,
				haloBounds,
				rank))
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
		dst.xPos[localIdx] = src.xPos[globalIdx];
		dst.yPos[localIdx] = src.yPos[globalIdx];
		dst.zPos[localIdx] = src.zPos[globalIdx];
	}

	void BuildLocalFluidParticlesFromGlobalState(
		Sync::UniversalSimState& localState,
		DistParticleState& particleState,
		const Sync::UniversalSimState& globalState,
		const IDomainDecomp& domainDecomp,
		const int rank,
		const float kernelRadius)
	{
		const Sync::FluidParticles& globalFluid = globalState.fluidParticles;
		Sync::FluidParticles& localFluid = localState.fluidParticles;

		const AABB haloBounds = domainDecomp.HaloExtentAABB(rank);
		const uint32_t ownedParticleCount =
			CountOwnedFluidParticles(globalFluid, domainDecomp, rank);
		const uint32_t haloParticleCount =
			CountHaloFluidParticles(globalFluid, domainDecomp, haloBounds, rank);
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
				domainDecomp,
				haloBounds,
				rank)) { continue; }

			CopyFluidParticle_GlobalToLocal(globalFluid, localFluid, globalIdx, localIdx);

			particleState.particleGlobalIdx_FromLocalIdx.push_back(globalIdx);
			particleState.particleLocalIdx_FromGlobalIdx[globalIdx] = localIdx;
			localIdx++;
		}

		// classify after compaction so cached indices are localIdx values
		ClassifyOwnedFluidParticlesIntoInnerOuter(
			particleState,
			localFluid,
			domainDecomp,
			rank,
			kernelRadius);
		ClassifyHaloParticleRange(particleState, ownedParticleCount, localParticleCount);
	}

	void BuildLocalBoundaryParticlesFromGlobalState(
		Sync::UniversalSimState& localState,
		DistParticleState& particleState,
		const Sync::UniversalSimState& globalState,
		const IDomainDecomp& domainDecomp,
		const int rank,
		const float kernelRadius)
	{
		const Sync::BoundaryParticles& globalBoundary = globalState.boundaryParticles;
		Sync::BoundaryParticles& localBoundary = localState.boundaryParticles;

		const AABB haloBounds = domainDecomp.HaloExtentAABB(rank);
		const uint32_t ownedParticleCount =
			CountOwnedBoundaryParticles(globalBoundary, domainDecomp, rank);
		const uint32_t haloParticleCount =
			CountHaloBoundaryParticles(globalBoundary, domainDecomp, haloBounds, rank);
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
				domainDecomp,
				haloBounds,
				rank)) { continue; }

			CopyBoundaryParticle_GlobalToLocal(globalBoundary, localBoundary, globalIdx, localIdx);

			particleState.particleGlobalIdx_FromLocalIdx.push_back(globalIdx);
			particleState.particleLocalIdx_FromGlobalIdx[globalIdx] = localIdx;
			localIdx++;
		}

		// classify after compaction so cached indices are localIdx values
		ClassifyOwnedBoundaryParticlesIntoInnerOuter(
			particleState,
			localBoundary,
			domainDecomp,
			rank,
			kernelRadius);
		ClassifyHaloParticleRange(particleState, ownedParticleCount, localParticleCount);
	}
}

void DistParticleState::Clear()
{
	ownedParticleCount = 0;
	haloParticleCount = 0;
	firstHaloLocalIdx = 0;

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
	firstHaloLocalIdx = 0;

	particleClasses.clear();
	particleLocalIdx_Owned.clear();
	particleLocalIdx_Inner.clear();
	particleLocalIdx_Outer.clear();
	particleLocalIdx_Halo.clear();
}

bool DistParticleState::IsValidLocalLayout(const uint32_t localParticleCount) const
{
	if (static_cast<uint64_t>(particleClasses.size()) != localParticleCount) { return false; }
	if (static_cast<uint64_t>(ownedParticleCount) + haloParticleCount != localParticleCount) { return false; }
	if (firstHaloLocalIdx != ownedParticleCount) { return false; }
	if (firstHaloLocalIdx > localParticleCount) { return false; }

	if (static_cast<uint64_t>(particleLocalIdx_Owned.size()) != ownedParticleCount) { return false; }
	if (static_cast<uint64_t>(particleLocalIdx_Halo.size()) != haloParticleCount) { return false; }
	if (static_cast<uint64_t>(particleLocalIdx_Inner.size()) +
		static_cast<uint64_t>(particleLocalIdx_Outer.size()) != ownedParticleCount) { return false; }

	if (!LocalIdxListCoversRange(particleLocalIdx_Owned, 0, firstHaloLocalIdx)) { return false; }
	if (!LocalIdxListCoversRange(particleLocalIdx_Halo, firstHaloLocalIdx, localParticleCount)) { return false; }
	if (!LocalIdxListsCoverRange(
		particleLocalIdx_Inner,
		particleLocalIdx_Outer,
		0,
		firstHaloLocalIdx)) { return false; }

	for (uint64_t i = 0; i < firstHaloLocalIdx; i++)
	{
		const DistParticleClass particleClass = particleClasses[i];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer) { return false; }
	}

	for (uint64_t i = firstHaloLocalIdx; i < localParticleCount; i++)
	{
		if (particleClasses[i] != DistParticleClass::Halo) { return false; }
	}

	for (const int localIdx : particleLocalIdx_Owned)
	{
		if (!IsLocalIdxInRange(localIdx, 0, firstHaloLocalIdx)) { return false; }
		const DistParticleClass particleClass = particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer) { return false; }
	}

	for (const int localIdx : particleLocalIdx_Inner)
	{
		if (!IsLocalIdxInRange(localIdx, 0, firstHaloLocalIdx)) { return false; }
		if (particleClasses[localIdx] != DistParticleClass::Inner) { return false; }
	}

	for (const int localIdx : particleLocalIdx_Outer)
	{
		if (!IsLocalIdxInRange(localIdx, 0, firstHaloLocalIdx)) { return false; }
		if (particleClasses[localIdx] != DistParticleClass::Outer) { return false; }
	}

	for (const int localIdx : particleLocalIdx_Halo)
	{
		if (!IsLocalIdxInRange(localIdx, firstHaloLocalIdx, localParticleCount)) { return false; }
		if (particleClasses[localIdx] != DistParticleClass::Halo) { return false; }
	}

	return true;
}

bool DistParticleState::IsValidLocalParticleState(
	const uint32_t localParticleCount,
	const uint32_t globalParticleCount) const
{
	if (static_cast<uint64_t>(particleClasses.size()) != localParticleCount) { return false; }
	if (static_cast<uint64_t>(ownedParticleCount) + haloParticleCount != localParticleCount) { return false; }

	if (static_cast<uint64_t>(particleLocalIdx_Owned.size()) != ownedParticleCount) { return false; }
	if (static_cast<uint64_t>(particleLocalIdx_Halo.size()) != haloParticleCount) { return false; }
	if (static_cast<uint64_t>(particleLocalIdx_Inner.size()) +
		static_cast<uint64_t>(particleLocalIdx_Outer.size()) != ownedParticleCount) { return false; }

	if (static_cast<uint64_t>(particleGlobalIdx_FromLocalIdx.size()) != localParticleCount) { return false; }
	if (static_cast<uint64_t>(particleLocalIdx_FromGlobalIdx.size()) != globalParticleCount) { return false; }

	std::vector<uint8_t> activeLocalIdxs(localParticleCount, 0);

	for (const int localIdx : particleLocalIdx_Owned)
	{
		if (!IsLocalIdxInRange(localIdx, 0, localParticleCount)) { return false; }
		if (activeLocalIdxs[localIdx] != 0) { return false; }

		const DistParticleClass particleClass = particleClasses[localIdx];
		if (particleClass != DistParticleClass::Inner &&
			particleClass != DistParticleClass::Outer) { return false; }

		activeLocalIdxs[localIdx] = 1;
	}

	for (const int localIdx : particleLocalIdx_Halo)
	{
		if (!IsLocalIdxInRange(localIdx, 0, localParticleCount)) { return false; }
		if (activeLocalIdxs[localIdx] != 0) { return false; }
		if (particleClasses[localIdx] != DistParticleClass::Halo) { return false; }

		activeLocalIdxs[localIdx] = 1;
	}

	for (uint64_t localIdx = 0; localIdx < localParticleCount; localIdx++)
	{
		if (activeLocalIdxs[localIdx] == 0) { return false; }
	}

	std::vector<uint8_t> ownedLocalIdxs(localParticleCount, 0);

	for (const int localIdx : particleLocalIdx_Inner)
	{
		if (!IsLocalIdxInRange(localIdx, 0, localParticleCount)) { return false; }
		if (ownedLocalIdxs[localIdx] != 0) { return false; }
		if (particleClasses[localIdx] != DistParticleClass::Inner) { return false; }

		ownedLocalIdxs[localIdx] = 1;
	}

	for (const int localIdx : particleLocalIdx_Outer)
	{
		if (!IsLocalIdxInRange(localIdx, 0, localParticleCount)) { return false; }
		if (ownedLocalIdxs[localIdx] != 0) { return false; }
		if (particleClasses[localIdx] != DistParticleClass::Outer) { return false; }

		ownedLocalIdxs[localIdx] = 1;
	}

	for (const int localIdx : particleLocalIdx_Owned)
	{
		if (ownedLocalIdxs[localIdx] == 0) { return false; }
	}

	std::vector<uint8_t> activeGlobalIdxs(globalParticleCount, 0);
	for (uint64_t localIdx = 0; localIdx < localParticleCount; localIdx++)
	{
		const int globalIdx = particleGlobalIdx_FromLocalIdx[localIdx];
		if (globalIdx < 0 || static_cast<uint32_t>(globalIdx) >= globalParticleCount) { return false; }
		if (activeGlobalIdxs[globalIdx] != 0) { return false; }
		if (particleLocalIdx_FromGlobalIdx[globalIdx] != static_cast<int>(localIdx)) { return false; }

		activeGlobalIdxs[globalIdx] = 1;
	}

	for (uint64_t globalIdx = 0; globalIdx < globalParticleCount; globalIdx++)
	{
		const int localIdx = particleLocalIdx_FromGlobalIdx[globalIdx];
		if (localIdx == -1) { continue; }
		if (!IsLocalIdxInRange(localIdx, 0, localParticleCount)) { return false; }
		if (particleGlobalIdx_FromLocalIdx[localIdx] != static_cast<int>(globalIdx)) { return false; }
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
	reOrderedParticleState.firstHaloLocalIdx = reOrderedParticleState.ownedParticleCount;

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
	const float kernelRadius)
{
	PrepareParticleClassification(particleState, fluidParticles.N_);

	// TODO: use TBB to parallelize this?
	for (uint64_t i = 0; i < fluidParticles.N_; i++)
	{
		const int localIdx = static_cast<int>(i);
		ClassifyOwnedParticleAtLocalIdx(
			particleState,
			FluidPositionAtArrayIdx(fluidParticles, localIdx),
			localIdx,
			domainDecomp,
			rank,
			kernelRadius);
	}

	particleState.ownedParticleCount =
		static_cast<uint32_t>(particleState.particleLocalIdx_Owned.size());
	particleState.firstHaloLocalIdx = particleState.ownedParticleCount;
}

void Dist::ClassifyOwnedBoundaryParticlesIntoInnerOuter(
	DistParticleState& particleState,
	const Sync::BoundaryParticles& boundaryParticles,
	const IDomainDecomp& domainDecomp,
	const int rank,
	const float kernelRadius)
{
	PrepareParticleClassification(particleState, boundaryParticles.N_);

	// TODO: use TBB to parallelize this?
	for (uint64_t i = 0; i < boundaryParticles.N_; i++)
	{
		const int localIdx = static_cast<int>(i);
		ClassifyOwnedParticleAtLocalIdx(
			particleState,
			BoundaryPositionAtArrayIdx(boundaryParticles, localIdx),
			localIdx,
			domainDecomp,
			rank,
			kernelRadius);
	}

	particleState.ownedParticleCount =
		static_cast<uint32_t>(particleState.particleLocalIdx_Owned.size());
	particleState.firstHaloLocalIdx = particleState.ownedParticleCount;
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
	localState.schemaVersion = globalState.schemaVersion;
	localState.step = globalState.step;
	localState.time = globalState.time;

	BuildLocalFluidParticlesFromGlobalState(
		localState,
		fluidParticleState,
		globalState,
		domainDecomp,
		rank,
		kernelRadius);

	BuildLocalBoundaryParticlesFromGlobalState(
		localState,
		boundaryParticleState,
		globalState,
		domainDecomp,
		rank,
		kernelRadius);
}
