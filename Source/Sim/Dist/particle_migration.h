// particle_migration.h

#pragma once

// external
#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>

// internal
#include "Source/Sim/Sync/state_sim.h"
#include "dist_particle_state.h"
#include "domain_decomp.h"



namespace Dist
{
	class MpiContext;

	// stores outgoing/incoming migrating fluid particles,
	// mirroring the SoA data layout for direct MPI exchange and rebuild
	class MigratingParticleBufferF
	{
	public:

		void Clear();
		void Resize(uint64_t particleCount);

		uint64_t Size() const { return globalIdx_.size(); }
		bool Empty() const { return Size() == 0; }

		int GlobalIdxAt(uint64_t bufferIdx) const;
		glm::vec3 PositionAt(uint64_t bufferIdx) const;

		// copies one local fluid particle into one slot of a send migration buffer
		// TODO: do this in bulk, instead of per-particle?
		void SetFromLocalParticle(
			const Sync::FluidParticles& fluidParticles,
			const DistParticleState& particleState,
			uint64_t bufferIdx,
			int localIdx);

		// copies one slot of a receive migration buffer into one local fluid particle
		// TODO: do this in bulk, instead of per-particle?
		void CopyToLocalParticle(
			Sync::FluidParticles& dst,
			uint64_t bufferIdx,
			int localIdx) const;

		// posts non-blocking MPI receives for the payload of migrating fluid particles
		void QueueIRecvPayload(
			const MpiContext& mpiContext,
			int peerRank,
			int tagBase,
			std::vector<MPI_Request>& requests);

		// posts non-blocking MPI sends for the payload of migrating fluid particles
		void QueueISendPayload(
			const MpiContext& mpiContext,
			int peerRank,
			int tagBase,
			std::vector<MPI_Request>& requests) const;

		// helpers for diagnostics and memory usage reporting
		int CheckedBufferCount() const;
		uint64_t UsedBytes() const;
		uint64_t AllocBytes() const;

	private:
		
		// metadata
		std::vector<int> globalIdx_;

		// particle data
		std::vector<uint32_t> id_;
		std::vector<float> xPos_;
		std::vector<float> yPos_;
		std::vector<float> zPos_;
		std::vector<float> xVel_;
		std::vector<float> yVel_;
		std::vector<float> zVel_;
		std::vector<float> xVelPred_;
		std::vector<float> yVelPred_;
		std::vector<float> zVelPred_;
		std::vector<float> xAccPress_;
		std::vector<float> yAccPress_;
		std::vector<float> zAccPress_;
		std::vector<float> sourceTerm_;
		std::vector<float> diagElem_;
		std::vector<float> pressure_;
		std::vector<float> density_;
	};

	struct ParticleMigrationBuffer
	{
		MigratingParticleBufferF fluidParticles;

		// clears packed migrating particles
		void Clear();
	};

	// counts migrated fluid particles for rank-local diagnostics
	struct ParticleMigrationStats
	{
		uint32_t sentFluidParticleCount = 0;
		uint32_t recvFluidParticleCount = 0;
	};

	// reusable work buffers for particle migration exchange and local rebuild
	struct MigrationWorkBuffers
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

		// local-index generation stamps used by in-place migration rebuild compaction
		std::vector<uint32_t> migrationLocalStatusStamp_FromLocalIdx;
		uint32_t migrationLocalStatusStamp = 0;
	};

	// exchanges rank-indexed migration counts, sends payloads only to active ranks,
	// then rebuilds this rank's owned fluid state; returns false + msg on failure
	bool MigrateFluidParticlesToOwnerRanks(
		Sync::UniversalSimState& localState,
		DistParticleState& fluidParticleState,
		const IDomainDecomp& domainDecomp,
		const MpiContext& mpiContext,
		uint32_t globalFluidParticleCount,
		MigrationWorkBuffers& workBuffers,
		ParticleMigrationStats* stats = nullptr,
		std::string* failMsg = nullptr);

	// compacts kept owned fluid particles in-place, appends received migrating particles,
	// and refreshes ownership maps; class lists are rebuilt by the distributed class phase
	bool RebuildLocalStateWithMigratedFluidParticles(
		const std::vector<MigrationWorkBuffers::RankBuffers>& migrations,
		const std::vector<int>& keptOwnedFluidLocalIdx,
		const std::vector<int>& migratingOutFluidLocalIdx,
		Sync::UniversalSimState& localState,
		DistParticleState& fluidParticleState,
		const IDomainDecomp& domainDecomp,
		int rank,
		uint32_t globalFluidParticleCount,
		MigrationWorkBuffers& workBuffers,
		std::string* failMsg = nullptr);

} // namespace Dist
