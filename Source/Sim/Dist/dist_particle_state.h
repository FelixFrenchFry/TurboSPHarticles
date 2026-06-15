// dist_particle_state.h

#pragma once

// external
#include <cstdint>
#include <vector>

// internal
#include "Source/Sim/Sync/state_sim.h"
#include "domain_decomp.h"



namespace Dist
{
	enum class DistParticleClass
	{
		Inner,  // particle owned by this rank, NOT exported to other ranks
		Outer,  // particle owned by this rank, exported to other ranks
		Halo,   // particle NOT owned by this rank, received from a neighbor rank
		None    // particle is not active in this rank (e.g. outside owned + halo bounds)
	};

	// stores all classification of particles (F/B) based on the domain decomp
	// local SoA layout is [ owned particles ][ halo particles ]
	struct DistParticleState
	{
		// compact local SoA ranges before backend data re-ordering
		uint32_t ownedParticleCount = 0;
		uint32_t haloParticleCount = 0;
		uint32_t firstHaloLocalIdx = 0;

		// per-particle class flag, indexed by localIdx
		std::vector<DistParticleClass> particleClasses;

		// cached local indices for each class
		std::vector<int> particleLocalIdx_Owned;
		std::vector<int> particleLocalIdx_Inner;
		std::vector<int> particleLocalIdx_Outer;
		std::vector<int> particleLocalIdx_Halo;

		// index maps between the full scenario state and compact rank-local state
		std::vector<int> particleGlobalIdx_FromLocalIdx;
		std::vector<int> particleLocalIdx_FromGlobalIdx;

		void Clear();
		void ClearParticleClasses();

		// validates compact [ owned ][ halo ] local SoA layout
		bool IsValidLocalLayout(uint32_t localParticleCount) const;

		// validates local metadata after backend data re-ordering
		bool IsValidLocalParticleState(
			uint32_t localParticleCount,
			uint32_t globalParticleCount) const;
	};

	bool ApplyLocalParticleReOrder(
		DistParticleState& particleState,
		const std::vector<uint32_t>& oldLocalIdx_FromNewLocalIdx,
		uint32_t globalParticleCount);

	void ClassifyOwnedFluidParticlesIntoInnerOuter(
		DistParticleState& particleState,
		const Sync::FluidParticles& fluidParticles,
		const IDomainDecomp& domainDecomp,
		int rank,
		float kernelRadius);

	void ClassifyOwnedBoundaryParticlesIntoInnerOuter(
		DistParticleState& particleState,
		const Sync::BoundaryParticles& boundaryParticles,
		const IDomainDecomp& domainDecomp,
		int rank,
		float kernelRadius);

	void BuildLocalParticlesFromGlobalState(
		Sync::UniversalSimState& localState,
		DistParticleState& fluidParticleState,
		DistParticleState& boundaryParticleState,
		const Sync::UniversalSimState& globalState,
		const IDomainDecomp& domainDecomp,
		int rank,
		float kernelRadius);

} // namespace Dist
