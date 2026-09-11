// owned_particle_gather.h
#pragma once

// external
#include <cstdint>
#include <vector>

// internal
#include "source/simulation/sync/state_sim.h"
#include "dist_particle_state.h"



namespace Dist
{
	// MPI tags for authoritative owned-particle gathering
	inline constexpr int OWNED_PARTICLE_GATHER_TAG_FLUID_COUNT = 6101;
	inline constexpr int OWNED_PARTICLE_GATHER_TAG_BOUNDARY_COUNT = 6102;
	inline constexpr int OWNED_PARTICLE_GATHER_TAG_FLUID_PAYLOAD = 6103;
	inline constexpr int OWNED_PARTICLE_GATHER_TAG_BOUNDARY_PAYLOAD = 6104;

	class MpiContext;

	// fluid particle payload used to gather authoritative owned particles
	// TODO: use SoA instead of AoS approach?
	struct OwnedFluidParticlePacket
	{
		uint32_t globalIdx = NO_GLOBAL_PARTICLE_IDX;
		uint32_t id = 0;

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
	// TODO: use SoA instead of AoS approach?
	struct OwnedBoundaryParticlePacket
	{
		uint32_t globalIdx = NO_GLOBAL_PARTICLE_IDX;
		uint32_t id = 0;

		float xPos = 0.0f;
		float yPos = 0.0f;
		float zPos = 0.0f;
		float volB = 0.0f;
		float psiB = 0.0f;
	};

	struct OwnedParticleGatherBuffer
	{
		std::vector<OwnedFluidParticlePacket> particlesF;
		std::vector<OwnedBoundaryParticlePacket> particlesB;

		void Clear();
	};

	bool PackOwnedParticleGatherBuffer(
		OwnedParticleGatherBuffer& gatherBuffer,
		const Sync::UniversalSimState& localState,
		const DistParticleState& particleStateF,
		const DistParticleState& particleStateB);

	bool GatherOwnedParticleBufferToRoot(
		Sync::UniversalSimState& globalOwnedState,
		const OwnedParticleGatherBuffer& sendBuffer,
		const Sync::UniversalSimState& localState,
		const MpiContext& mpiContext,
		uint32_t N_globalF, uint32_t N_globalB);

} // namespace Dist
