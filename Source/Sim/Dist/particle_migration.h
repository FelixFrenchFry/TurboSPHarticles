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
#include "particle_migration_diagnostics.h"



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

		// maps MPI ranks to rankBuffers entries, self-rank remains -1
		std::vector<int> rankBufferIdx_FromRank;

		// rank-indexed particle counts for the all-to-all migration count exchange
		std::vector<int> sendFluidCount_FromRank;
		std::vector<int> recvFluidCount_FromRank;

		// owned local particles that still belong to this rank after routing
		std::vector<int> keptOwnedFluidLocalIdx;

		// owned local particles that migrated out and leave holes in this rank-local state
		std::vector<int> migratingOutFluidLocalIdx;

		// per-owned-particle route, 0 means kept local, 1+ maps to rankBuffers index+1
		std::vector<int> migrationRouteIdx_FromOwnedIdx;

		// chunk-local route counts and write offsets used by parallel migration packing
		std::vector<uint64_t> migrationRouteCounts_ByChunkRoute;
		std::vector<uint64_t> migrationRouteWriteOffsets_ByChunkRoute;
		std::vector<uint64_t> migrationRouteTotals;
		std::vector<uint64_t> migrationOutCounts_ByChunk;
		std::vector<uint64_t> migrationOutWriteOffsets_ByChunk;

		// local-index status stamps used by in-place migration rebuild compaction
		std::vector<uint32_t> migrationLocalStatusStamp_FromLocalIdx;
		std::vector<uint8_t> migrationLocalStatus_FromLocalIdx;
		uint32_t migrationLocalStatusStamp = 0;

		// single receive buffer consumed by the rebuild step
		ParticleMigrationBuffer aggregatedRecvBuffer;
	};

	// exchanges rank-indexed migration counts, sends payloads only to active ranks,
	// then rebuilds this rank's owned fluid state; returns false + msg on failure
	bool MigrateFluidParticlesToOwnerRanks(
		Sync::UniversalSimState& localState,
		DistParticleState& fluidParticleState,
		const IDomainDecomp& domainDecomp,
		const MpiContext& mpiContext,
		uint32_t globalFluidParticleCount,
		ParticleMigrationWorkBuffers& workBuffers,
		ParticleMigrationStats* stats = nullptr,
		std::string* failMsg = nullptr,
		ParticleMigrationTimingReport* timingReport = nullptr);

	// compacts kept owned fluid particles in-place, appends received migrating particles,
	// and refreshes ownership maps; class lists are rebuilt by the distributed class phase
	bool RebuildLocalStateWithMigratedFluidParticles(
		const ParticleMigrationBuffer& recvBuffer,
		const std::vector<int>& keptOwnedFluidLocalIdx,
		const std::vector<int>& migratingOutFluidLocalIdx,
		Sync::UniversalSimState& localState,
		DistParticleState& fluidParticleState,
		const IDomainDecomp& domainDecomp,
		int rank,
		uint32_t globalFluidParticleCount,
		ParticleMigrationWorkBuffers& workBuffers,
		std::string* failMsg = nullptr);

} // namespace Dist
