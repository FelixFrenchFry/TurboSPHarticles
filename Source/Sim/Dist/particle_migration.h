// particle_migration.h

#pragma once

// external
#include <cstdint>
#include <string>
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

		// clears packed migrating fluid particles
		void Clear();
	};

	// counts migrated fluid particles for rank-local diagnostics
	struct ParticleMigrationStats
	{
		uint32_t sentFluidParticleCount = 0;
		uint32_t recvFluidParticleCount = 0;
	};

	// reusable storage for fluid particle migration exchange and local rebuild
	struct ParticleMigrationWorkBuffers
	{
		struct RankBuffers
		{
			int rank = -1;

			ParticleMigrationBuffer sendBuffer;
			ParticleMigrationBuffer recvBuffer;

			int sendFluidCount = 0;
			int recvFluidCount = 0;
		};

		// per-peer send/receive storage reused across migration steps
		std::vector<RankBuffers> rankBuffers;

		// single receive buffer consumed by the rebuild step
		ParticleMigrationBuffer aggregatedRecvBuffer;

		// destination storage for rebuilding owned fluid SoA + metadata
		Sync::FluidParticles rebuiltFluidParticles;
		DistParticleState rebuiltParticleState;
	};

	// packs owned fluid particles whose current position belongs to destination rank;
	// returns false + msg if local particle metadata or ownership lookup is inconsistent
	bool PackMigratingFluidParticlesForRank(
		ParticleMigrationBuffer& sendBuffer,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const IDomainDecomp& domainDecomp,
		int dstRank,
		std::string* failMsg = nullptr);

	// exchanges migrating fluid particle counts and payloads with all ranks, then rebuilds
	// this rank's owned fluid state; returns false + msg if packing, exchange, or rebuild fails
	bool MigrateFluidParticlesToOwnerRanks(
		Sync::UniversalSimState& localState,
		DistParticleState& fluidParticleState,
		const IDomainDecomp& domainDecomp,
		const MpiContext& mpiContext,
		float kernelRadius,
		uint32_t globalFluidParticleCount,
		ParticleMigrationWorkBuffers& workBuffers,
		ParticleMigrationStats* stats = nullptr,
		std::string* failMsg = nullptr);

	// rebuilds local fluid SoA arrays from kept owned particles plus received migrating particles
	// and refreshes matching particle metadata
	bool RebuildLocalStateWithMigratedFluidParticles(
		const ParticleMigrationBuffer& recvBuffer,
		Sync::UniversalSimState& localState,
		DistParticleState& fluidParticleState,
		const IDomainDecomp& domainDecomp,
		int rank,
		float kernelRadius,
		uint32_t globalFluidParticleCount,
		ParticleMigrationWorkBuffers& workBuffers,
		std::string* failMsg = nullptr);

} // namespace Dist
