// dist_particle_state.cpp

// external
#include <cstdint>

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
		const int firstHaloLocalIdx,
		const uint32_t localParticleCount)
	{
		for (uint64_t i = static_cast<uint64_t>(firstHaloLocalIdx);
			 i < localParticleCount;
			 i++)
		{
			const int localIdx = static_cast<int>(i);
			particleState.particleLocalIdx_Halo.push_back(localIdx);
			particleState.particleClasses[localIdx] = DistParticleClass::Halo;
		}
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
	particleClasses.clear();
	particleLocalIdx_Owned.clear();
	particleLocalIdx_Inner.clear();
	particleLocalIdx_Outer.clear();
	particleLocalIdx_Halo.clear();
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
