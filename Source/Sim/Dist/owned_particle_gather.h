// owned_particle_gather.h

#pragma once

// external
#include <cstdint>
#include <vector>

// internal
#include "Source/Sim/Sync/state_sim.h"
#include "dist_particle_state.h"



namespace Dist
{
	class MpiContext;

	// fluid particle payload used to gather authoritative owned particles
	struct OwnedFluidParticlePacket
	{
		int globalIdx = -1;

		float xPos = 0.0f;
		float yPos = 0.0f;
		float zPos = 0.0f;

		float xVel = 0.0f;
		float yVel = 0.0f;
		float zVel = 0.0f;

		float xVelPred = 0.0f;
		float yVelPred = 0.0f;
		float zVelPred = 0.0f;

		float xAccPress = 0.0f;
		float yAccPress = 0.0f;
		float zAccPress = 0.0f;

		float sourceTerm = 0.0f;
		float diagElem = 0.0f;
		float pressure = 0.0f;
		float density = 0.0f;
	};

	// boundary particle payload used to gather authoritative owned particles
	struct OwnedBoundaryParticlePacket
	{
		int globalIdx = -1;

		float xPos = 0.0f;
		float yPos = 0.0f;
		float zPos = 0.0f;
		float volB = 0.0f;
		float psiB = 0.0f;
	};

	struct OwnedParticleGatherBuffer
	{
		std::vector<OwnedFluidParticlePacket> fluidParticles;
		std::vector<OwnedBoundaryParticlePacket> boundaryParticles;

		void Clear();
	};

	struct OwnedParticleGatherStats
	{
		uint32_t gatheredFluidParticleCount = 0;
		uint32_t gatheredBoundaryParticleCount = 0;
	};

	bool PackOwnedParticleGatherBuffer(
		OwnedParticleGatherBuffer& gatherBuffer,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const DistParticleState& boundaryParticleState);

	bool RebuildGlobalOwnedStateFromGatheredParticles(
		Sync::UniversalSimState& globalOwnedState,
		const OwnedParticleGatherBuffer& gatherBuffer,
		uint32_t globalFluidParticleCount,
		uint32_t globalBoundaryParticleCount);

	bool GatherOwnedParticlesToRoot(
		Sync::UniversalSimState& globalOwnedState,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const DistParticleState& boundaryParticleState,
		const MpiContext& mpiContext,
		uint32_t globalFluidParticleCount,
		uint32_t globalBoundaryParticleCount,
		OwnedParticleGatherStats* stats = nullptr);

} // namespace Dist
