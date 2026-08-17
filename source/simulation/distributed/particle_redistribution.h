// particle_redistribution.h

#pragma once

// external
#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>

// internal
#include "source/simulation/core/backend_sim.h"
#include "dist_particle_state.h"
#include "domain_decomp.h"



namespace Dist
{
	// chunks prefixed per block to keep parallel work tied to the chunk count
	inline constexpr uint32_t PARTICLE_PREFIX_BLOCK_CHUNK_COUNT = 32;
	// minimum block stride keeping neighboring worker totals on separate cache lines
	inline constexpr uint32_t PARTICLE_PREFIX_BLOCK_STRIDE_MIN = 8;

	// owned particles routed per chunk, so workers fill disjoint output ranges
	// TODO: fine-tune chunk size by trying out 1024, 2048, 4096, 8192, 16384, 32768?
	inline constexpr uint32_t PARTICLE_ROUTING_CHUNK_SIZE = 4096;

	// particles gathered per block, keeping one block of every field cache resident
	// TODO: fine-tune chunk size by trying out 1024, 2048, 4096, 8192, 16384, 32768?
	inline constexpr uint32_t PARTICLE_GATHER_BLOCK_SIZE = 2048;

	// owned particles classified per chunk, so workers fill disjoint output ranges
	// TODO: fine-tune chunk size by trying out 1024, 2048, 4096, 8192, 16384, 32768?
	inline constexpr uint32_t OWNED_CLASSIFICATION_CHUNK_SIZE = 4096;
	inline constexpr uint32_t OWNED_CLASSIFICATION_CLASS_COUNT = 2;
	inline constexpr uint32_t OWNED_CLASSIFICATION_CLASS_INNER = 0;
	inline constexpr uint32_t OWNED_CLASSIFICATION_CLASS_OUTER = 1;

	// packed particle count lanes exchanged once per redistribution
	inline constexpr int PARTICLE_REDISTRIBUTION_LANE_COUNT = 4;
	inline constexpr int PARTICLE_REDISTRIBUTION_LANE_OWNED_FLUID = 0;
	inline constexpr int PARTICLE_REDISTRIBUTION_LANE_HALO_FLUID = 1;
	inline constexpr int PARTICLE_REDISTRIBUTION_LANE_OWNED_BOUNDARY = 2;
	inline constexpr int PARTICLE_REDISTRIBUTION_LANE_HALO_BOUNDARY = 3;

	// disjoint MPI tag ranges for the four particle payloads
	inline constexpr int PARTICLE_REDISTRIBUTION_TAG_OWNED_FLUID = 5310;
	inline constexpr int PARTICLE_REDISTRIBUTION_TAG_HALO_FLUID = 5410;
	inline constexpr int PARTICLE_REDISTRIBUTION_TAG_OWNED_BOUNDARY = 5510;
	inline constexpr int PARTICLE_REDISTRIBUTION_TAG_HALO_BOUNDARY = 5610;

	class MpiContext;
	struct ParticleRedistributionTimingReport;

	struct ParticleRedistributionRoutes
	{
		// outgoing migration and halo routes, defining what the destination rank does with them
		// (outer vector = destination rank, inner vector = local particle indices)
		std::vector<std::vector<uint32_t>> ownedFluidLocalIdxs_ByDstRank;
		std::vector<std::vector<uint32_t>> haloFluidLocalIdxs_ByDstRank;
		std::vector<std::vector<uint32_t>> ownedBoundaryLocalIdxs_ByDstRank;
		std::vector<std::vector<uint32_t>> haloBoundaryLocalIdxs_ByDstRank;

		// particles retained locally in the rebuilt backend state
		std::vector<uint32_t> keptOwnedFluidLocalIdxs;
		std::vector<uint32_t> localHaloFluidLocalIdxs;
		std::vector<uint32_t> keptOwnedBoundaryLocalIdxs;
		std::vector<uint32_t> localHaloBoundaryLocalIdxs;

		// set when any neighbor needs the particle as halo data, so a zero marks an inner owned particle
		std::vector<uint8_t> fluidIsOuterOwned_ByLocalIdx;
		std::vector<uint8_t> boundaryIsOuterOwned_ByLocalIdx;
	};

	// reusable scratch for parallel owned-particle routing
	struct ParticleRoutingWorkBuffers
	{
		// per chunk bucket counts and write offsets, indexed by [chunkIdx * N_buckets + bucketIdx]
		std::vector<uint32_t> routeCounts_ByChunkBucket;
		std::vector<uint32_t> routeWriteOffsets_ByChunkBucket;
		std::vector<uint32_t> routeTotals;

		// destinations decided by the counting pass, indexed by position in the owned index list
		std::vector<int> ownerRank_ByOwnedIdx;

		// rank halo bounds re-read by the halo destination test of every routed particle
		std::vector<AABB> haloExtentAABB_ByRank;
	};

	struct ParticleRedistributionStats
	{
		// fluid ownership migration and halo traffic
		uint32_t N_sentOwnedF = 0;
		uint32_t N_recvOwnedF = 0;
		uint32_t N_sentHaloF = 0;
		uint32_t N_recvHaloF = 0;

		// boundary ownership migration and halo traffic
		uint32_t N_sentOwnedB = 0;
		uint32_t N_recvOwnedB = 0;
		uint32_t N_sentHaloB = 0;
		uint32_t N_recvHaloB = 0;
	};

	class ParticleRedistributionMgr final
	{
	public:

		ParticleRedistributionMgr() = default;

		// builds rank-indexed routes from the current owned particles without communication
		bool BuildExchangeRoutes(
			const Sync::FluidConstView& particlesF,
			const Sync::BoundaryConstView& particlesB,
			const DistParticleState& particleStateF,
			const DistParticleState& particleStateB,
			const IDomainDecomp& domainDecomp,
			bool redistributeParticlesB,
			int rank, int N_ranks,
			uint32_t N_globalF, uint32_t N_globalB,
			std::string* failMsg = nullptr,
			ParticleRedistributionTimingReport* timingReport = nullptr);

		// orchestrator that compacts kept particles in place and appends the next migrated / halo particles
		bool RedistributeParticles(
			Sim::I_DistributedParticleBackend& distParticleBackend,
			DistParticleState& particleStateF,
			DistParticleState& particleStateB,
			const IDomainDecomp& domainDecomp,
			const MpiContext& mpiContext,
			const Sim::SimParams& simParams,
			bool redistributeParticlesB,
			uint32_t N_globalF,
			uint32_t N_globalB,
			std::string* failMsg = nullptr,
			ParticleRedistributionTimingReport* timingReport = nullptr);

		// latest routes and traffic statistics
		const ParticleRedistributionRoutes& GetRoutes() const { return routes_; }
		const ParticleRedistributionStats& GetStats() const { return redistStats_; }
		void ResetRedistStats() { redistStats_ = {}; }

		// memory usage for reusable communication and mutation scratch storage
		Mem::MemUsage GetExchangeBufferMemoryUsage() const;
		Mem::MemUsage GetScratchMemoryUsage() const;

	private:

		// extra capacity for reusable particle payload and pipeline scratch storage
		static constexpr float exchangeBufferExtraAllocCapacityPct_ = 5.0f; // TODO: fine-tune
		static constexpr float scratchBufferExtraAllocCapacityPct_ = 5.0f; // TODO: fine-tune

		// packed SoA payload for migrated or halo fluid particles, carrying only identity,
		// (the fields velPred, accPress, sourceTerm, diagElem, pressure, and density are omitted,
		// because the following solver step re-writes them before reading them)
		// TODO: keep it like this or copy them too, if solver step behavior changes?
		class FluidExchangeBuffer
		{
		public:

			void Clear();
			void Resize(uint32_t N);

			// resizes and gathers the given backend particles into the buffer
			void SetBulkFromParticleView(
				const Sync::FluidConstView& src,
				const std::vector<uint32_t>& localIdxs);

			// copies one particle from the buffer into the backend
			void CopyToLocalParticle(
				Sync::FluidParticles& dst,
				uint32_t bufferIdx,
				uint32_t localIdx) const;

			// copies the whole buffer into a contiguous backend range
			void CopyBulkToLocalParticles(
				Sync::FluidParticles& dst,
				uint32_t dstBeginLocalIdx) const;

			void QueueIRecvPayload(
				const MpiContext& mpiContext,
				int peerRank,
				int tagBase,
				std::vector<MPI_Request>& requests);
			void QueueISendPayload(
				const MpiContext& mpiContext,
				int peerRank,
				int tagBase,
				std::vector<MPI_Request>& requests) const;

			// helpers
			uint32_t Size() const { return N_particles_; }
			bool IsEmpty() const { return Size() == 0; }
			int CheckedCount() const;
			uint32_t GlobalIdxAt(uint32_t bufferIdx) const; // particle IDs are the global indices
			Mem::MemUsage GetMemoryUsage() const;

		private:

			// logical particle count, which storage only grows to hold
			uint32_t N_particles_ = 0;
			Sync::FluidParticles particles_; // fields left out of the payload stay empty
		};

		// packed SoA payload for migrated or halo boundary particles
		class BoundaryExchangeBuffer
		{
		public:

			void Clear();
			void Resize(uint32_t N);

			// resizes and gathers the given backend particles into the buffer
			void SetBulkFromParticleView(
				const Sync::BoundaryConstView& src,
				const std::vector<uint32_t>& localIdxs);

			// copies one particle from the buffer into the backend
			void CopyToLocalParticle(
				Sync::BoundaryParticles& dst,
				uint32_t bufferIdx,
				uint32_t localIdx) const;

			// copies the whole buffer into a contiguous backend range
			void CopyBulkToLocalParticles(
				Sync::BoundaryParticles& dst,
				uint32_t dstBeginLocalIdx) const;

			void QueueIRecvPayload(
				const MpiContext& mpiContext,
				int peerRank,
				int tagBase,
				std::vector<MPI_Request>& requests);
			void QueueISendPayload(
				const MpiContext& mpiContext,
				int peerRank,
				int tagBase,
				std::vector<MPI_Request>& requests) const;

			// helpers
			uint32_t Size() const { return N_particles_; }
			bool IsEmpty() const { return Size() == 0; }
			int CheckedCount() const;
			uint32_t GlobalIdxAt(uint32_t bufferIdx) const; // particle IDs are the global indices
			Mem::MemUsage GetMemoryUsage() const;

		private:

			// logical particle count, which storage only grows to hold
			uint32_t N_particles_ = 0;
			Sync::BoundaryParticles particles_;
		};

		// send and receive storage for exchange with one neighbor rank
		struct RankBuffers
		{
			// fluid migration and halo payloads
			FluidExchangeBuffer sendOwnedParticlesF;
			FluidExchangeBuffer recvOwnedParticlesF;
			FluidExchangeBuffer sendHaloParticlesF;
			FluidExchangeBuffer recvHaloParticlesF;

			// boundary migration and halo payloads
			BoundaryExchangeBuffer sendOwnedParticlesB;
			BoundaryExchangeBuffer recvOwnedParticlesB;
			BoundaryExchangeBuffer sendHaloParticlesB;
			BoundaryExchangeBuffer recvHaloParticlesB;

			// packed count lanes
			int N_sendOwnedF = 0;
			int N_recvOwnedF = 0;
			int N_sendHaloF = 0;
			int N_recvHaloF = 0;
			int N_sendOwnedB = 0;
			int N_recvOwnedB = 0;
			int N_sendHaloB = 0;
			int N_recvHaloB = 0;
		};

		// backend-resident particle position access
		static glm::vec3 FluidPositionAtLocalIdx(
			const Sync::FluidConstView& particles,
			uint32_t localIdx);
		static glm::vec3 BoundaryPositionAtLocalIdx(
			const Sync::BoundaryConstView& particles,
			uint32_t localIdx);

		// particle property copying into append-only backend mutation storage
		// (fluid particles carry the same reduced field set as FluidExchangeBuffer)
		static void CopyFluidParticle(
			const Sync::FluidConstView& src,
			Sync::FluidParticles& dst,
			uint32_t srcLocalIdx,
			uint32_t dstLocalIdx);
		static void CopyBoundaryParticle(
			const Sync::BoundaryConstView& src,
			Sync::BoundaryParticles& dst,
			uint32_t srcLocalIdx,
			uint32_t dstLocalIdx);

		// dense compaction and rebuilt metadata construction
		static bool BuildDenseCompactionMap(
			uint32_t N_oldLocal,
			const std::vector<uint32_t>& retainedOldLocalIdxs,
			std::vector<uint32_t>& oldLocalIdx_FromNewLocalIdx,
			std::string* failMsg);
		static void PrepareRebuiltParticleState(
			DistParticleState& particleState,
			uint32_t N_owned,
			uint32_t N_halo,
			uint32_t N_local);
		static bool AddHaloParticleMetadata(
			DistParticleState& particleState,
			uint32_t N_global,
			uint32_t globalIdx,
			uint32_t particleId,
			uint32_t localIdx,
			uint32_t haloIdx);
		static bool CheckedUint32(
			uint64_t value,
			uint32_t& result,
			const char* valueName,
			std::string* failMsg);
		static bool ValidateParticleViews(
			const Sync::FluidConstView& particlesF,
			const Sync::BoundaryConstView& particlesB,
			const DistParticleState& particleStateF,
			const DistParticleState& particleStateB,
			uint32_t N_globalF, uint32_t N_globalB,
			std::string* failMsg);

		// routing and communication phases
		void PrepareBuffers(int N_ranks);
		bool RouteOwnedFluidParticles(
			const Sync::FluidConstView& particlesF,
			const DistParticleState& particleStateF,
			const IDomainDecomp& domainDecomp,
			int rank, int N_ranks,
			std::string* failMsg);
		bool RouteOwnedBoundaryParticles(
			const Sync::BoundaryConstView& particlesB,
			const DistParticleState& particleStateB,
			const IDomainDecomp& domainDecomp,
			int rank, int N_ranks,
			std::string* failMsg);
		bool RouteOwnedParticles(
			const Sync::FluidConstView& particlesF,
			const Sync::BoundaryConstView& particlesB,
			const DistParticleState& particleStateF,
			const DistParticleState& particleStateB,
			const IDomainDecomp& domainDecomp,
			bool redistributeParticlesB,
			int rank, int N_ranks,
			std::string* failMsg,
			ParticleRedistributionTimingReport* timingReport);
		bool PackSendBuffers(
			const Sync::FluidConstView& particlesF,
			const Sync::BoundaryConstView& particlesB,
			bool redistributeParticlesB,
			int rank, int N_ranks,
			std::string* failMsg);
		bool PrepareReceiveBuffers(
			int rank, int N_ranks,
			std::string* failMsg);
		void ExchangePayloads(
			int rank, int N_ranks,
			const MpiContext& mpiContext);
		bool PrepareFluidParticleMutation(
			const Sync::FluidConstView& oldParticles,
			const IDomainDecomp& domainDecomp,
			int rank,
			float haloRadius,
			uint32_t N_global,
			std::string* failMsg,
			ParticleRedistributionTimingReport* timingReport);
		bool PrepareBoundaryParticleMutation(
			const Sync::BoundaryConstView& oldParticles,
			const IDomainDecomp& domainDecomp,
			int rank,
			float haloRadius,
			uint32_t N_global,
			std::string* failMsg,
			ParticleRedistributionTimingReport* timingReport);

		// peer communication routes and storage
		std::vector<RankBuffers> rankBuffers_;
		ParticleRedistributionRoutes routes_;
		ParticleRoutingWorkBuffers routingWorkBuffers_;
		ParticleClassificationWorkBuffers ownedClassificationWorkBuffers_;
		std::vector<int> sendCounts_;
		std::vector<int> recvCounts_;

		// dense compaction mappings and append-only particle storage
		std::vector<uint32_t> retainedFluidOldLocalIdx_FromNewLocalIdx_;
		std::vector<uint32_t> retainedBoundaryOldLocalIdx_FromNewLocalIdx_;
		std::vector<uint32_t> migratedGlobalIdxScratch_;
		Sync::FluidParticles appendedParticlesF_;
		Sync::BoundaryParticles appendedParticlesB_;

		// metadata matching the compacted and appended backend state
		DistParticleState rebuiltParticleStateF_;
		DistParticleState rebuiltParticleStateB_;

		// latest per-step network traffic
		ParticleRedistributionStats redistStats_{};
	};

} // namespace Dist
