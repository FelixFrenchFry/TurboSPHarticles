// halo_exchange.h

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

	// stores outgoing/incoming fluid halo particles,
	// mirroring the SoA data layout for direct MPI exchange and rebuild
	class HaloParticleBufferF
	{
	public:

		void Clear();
		void Resize(uint64_t particleCount);

		uint64_t Size() const { return globalIdx_.size(); }
		bool Empty() const { return Size() == 0; }

		int GlobalIdxAt(uint64_t bufferIdx) const;

		// copies one local fluid particle into one slot of a send halo buffer
		void SetFromLocalParticle(
			const Sync::FluidParticles& fluidParticles,
			const DistParticleState& particleState,
			uint64_t bufferIdx,
			int localIdx);

		// copies one slot of a receive halo buffer into one local fluid particle
		void CopyToLocalParticle(
			Sync::FluidParticles& dst,
			uint64_t bufferIdx,
			int localIdx) const;

		// posts non-blocking MPI receives for the payload of halo fluid particles
		void QueueIRecvPayload(
			const MpiContext& mpiContext,
			int peerRank,
			int tagBase,
			std::vector<MPI_Request>& requests);

		// posts non-blocking MPI sends for the payload of halo fluid particles
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

	// stores outgoing/incoming boundary halo particles,
	// mirroring the boundary SoA data layout for direct MPI exchange and rebuild
	class HaloParticleBufferB
	{
	public:

		void Clear();
		void Resize(uint64_t particleCount);

		uint64_t Size() const { return globalIdx_.size(); }
		bool Empty() const { return Size() == 0; }

		int GlobalIdxAt(uint64_t bufferIdx) const;

		// copies one local boundary particle into one slot of a send halo buffer
		void SetFromLocalParticle(
			const Sync::BoundaryParticles& boundaryParticles,
			const DistParticleState& particleState,
			uint64_t bufferIdx,
			int localIdx);

		// copies one slot of a receive halo buffer into one local boundary particle
		void CopyToLocalParticle(
			Sync::BoundaryParticles& dst,
			uint64_t bufferIdx,
			int localIdx) const;

		// posts non-blocking MPI receives for the payload of halo boundary particles
		void QueueIRecvPayload(
			const MpiContext& mpiContext,
			int peerRank,
			int tagBase,
			std::vector<MPI_Request>& requests);

		// posts non-blocking MPI sends for the payload of halo boundary particles
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
		std::vector<float> xPos_;
		std::vector<float> yPos_;
		std::vector<float> zPos_;
		std::vector<float> volB_;
		std::vector<float> psiB_;
	};

	struct HaloExchangeBuffer
	{
		HaloParticleBufferF fluidParticles;
		HaloParticleBufferB boundaryParticles;

		// clears packed fluid and boundary halo particles
		void Clear();
	};

	// counts exchanged halo particles for rank-local diagnostics
	struct HaloExchangeStats
	{
		uint32_t sentFluidParticleCount = 0;
		uint32_t sentBoundaryParticleCount = 0;
		uint32_t recvFluidParticleCount = 0;
		uint32_t recvBoundaryParticleCount = 0;
	};

	// reusable work buffers for halo exchange and local halo rebuild
	struct HaloExchangeWorkBuffers
	{
		struct RankBuffers
		{
			int rank = -1;

			HaloExchangeBuffer sendBuffer;
			HaloExchangeBuffer recvBuffer;

			int sendFluidCount = 0;
			int recvFluidCount = 0;
			int sendBoundaryCount = 0;
			int recvBoundaryCount = 0;
		};

		// per-neighbor send/receive storage reused across halo exchange steps
		std::vector<RankBuffers> rankBuffers;

		// maps MPI ranks to rankBuffers entries, non-neighbor ranks remain -1
		std::vector<int> rankBufferIdx_FromRank;
	};

	// reusable destination storage for rebuilding [ owned ][ halo ] rank-local state
	// used only when owned particles are not already stored in a compact prefix
	struct HaloRebuildScratch
	{
		Sync::UniversalSimState state;
		DistParticleState fluidParticleState;
		DistParticleState boundaryParticleState;
	};

	// packs this rank's outer fluid/boundary particles that overlap the halo extent of destination rank
	// into a sendBuffer; returns false + msg if local particle metadata is inconsistent
	bool PackHaloExchangeBufferForRank(
		HaloExchangeBuffer& sendBuffer,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const DistParticleState& boundaryParticleState,
		const IDomainDecomp& domainDecomp,
		int dstRank,
		std::string* failMsg = nullptr);

	// exchanges halo particle counts and SoA payloads with all neighbor ranks and stores
	// received halos in per-neighbor recv buffers; returns false + msg on invalid metadata/counts
	bool ExchangeHaloParticlesWithNeighborRanks(
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const DistParticleState& boundaryParticleState,
		const IDomainDecomp& domainDecomp,
		const MpiContext& mpiContext,
		HaloExchangeWorkBuffers& workBuffers,
		HaloExchangeStats* stats = nullptr,
		std::string* failMsg = nullptr);

	// rebuilds local fluid/boundary SoA arrays as [ owned ][ received halos ] from
	// per-neighbor receive buffers and refreshes matching particle metadata
	bool RebuildLocalStateWithReceivedHalos(
		const std::vector<HaloExchangeWorkBuffers::RankBuffers>& exchanges,
		Sync::UniversalSimState& localState,
		DistParticleState& fluidParticleState,
		DistParticleState& boundaryParticleState,
		uint32_t globalFluidParticleCount,
		uint32_t globalBoundaryParticleCount,
		HaloRebuildScratch& scratch,
		std::string* failMsg = nullptr);

} // namespace Dist
