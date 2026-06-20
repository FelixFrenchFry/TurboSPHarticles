// particle_migration.h

#pragma once

// external
#include <cstdint>
#include <vector>

// internal
#include "Source/Sim/Sync/state_sim.h"
#include "dist_particle_state.h"
#include "domain_decomp.h"



namespace Dist
{
	class MpiContext;

	// fluid particle payload used for owner migration
	struct MigratingFluidParticlePacket
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

	struct ParticleMigrationBuffer
	{
		std::vector<MigratingFluidParticlePacket> fluidParticles;

		void Clear();
	};

	struct ParticleMigrationStats
	{
		uint32_t sentFluidParticleCount = 0;
		uint32_t receivedFluidParticleCount = 0;
	};

	bool PackMigratingFluidParticlesForRank(
		ParticleMigrationBuffer& sendBuffer,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const IDomainDecomp& domainDecomp,
		int dstRank);

	bool MigrateFluidParticlesToOwnerRanks(
		Sync::UniversalSimState& localState,
		DistParticleState& fluidParticleState,
		const IDomainDecomp& domainDecomp,
		const MpiContext& mpiContext,
		float kernelRadius,
		uint32_t globalFluidParticleCount,
		ParticleMigrationStats* stats = nullptr);

	bool RebuildLocalStateWithMigratedFluidParticles(
		Sync::UniversalSimState& localState,
		DistParticleState& fluidParticleState,
		const ParticleMigrationBuffer& recvBuffer,
		const IDomainDecomp& domainDecomp,
		int rank,
		float kernelRadius,
		uint32_t globalFluidParticleCount);

} // namespace Dist
