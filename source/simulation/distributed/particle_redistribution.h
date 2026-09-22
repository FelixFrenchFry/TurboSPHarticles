// particle_redistribution.h
#pragma once

// external
#include <cstdint>
#include <mpi.h>
#include <span>
#include <string>
#include <vector>

// internal
#include "source/simulation/core/backend_sim.h"
#include "dist_particle_state.h"
#include "domain_decomp.h"



namespace Dist
{
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

	// packed payload field blocks, ordered as they are gathered, sent, and copied out
	inline constexpr uint32_t FLUID_PAYLOAD_FIELD_COUNT = 6;
	inline constexpr uint32_t FLUID_PAYLOAD_FIELD_X_POS = 0;
	inline constexpr uint32_t FLUID_PAYLOAD_FIELD_Y_POS = 1;
	inline constexpr uint32_t FLUID_PAYLOAD_FIELD_Z_POS = 2;
	inline constexpr uint32_t FLUID_PAYLOAD_FIELD_X_VEL = 3;
	inline constexpr uint32_t FLUID_PAYLOAD_FIELD_Y_VEL = 4;
	inline constexpr uint32_t FLUID_PAYLOAD_FIELD_Z_VEL = 5;
	inline constexpr uint32_t BOUNDARY_PAYLOAD_FIELD_COUNT = 5;
	inline constexpr uint32_t BOUNDARY_PAYLOAD_FIELD_X_POS = 0;
	inline constexpr uint32_t BOUNDARY_PAYLOAD_FIELD_Y_POS = 1;
	inline constexpr uint32_t BOUNDARY_PAYLOAD_FIELD_Z_POS = 2;
	inline constexpr uint32_t BOUNDARY_PAYLOAD_FIELD_VOL = 3;
	inline constexpr uint32_t BOUNDARY_PAYLOAD_FIELD_PSI = 4;
	inline constexpr uint32_t RIGID_PAYLOAD_FIELD_COUNT = 5;
	inline constexpr uint32_t RIGID_PAYLOAD_FIELD_X_REL_POS = 0;
	inline constexpr uint32_t RIGID_PAYLOAD_FIELD_Y_REL_POS = 1;
	inline constexpr uint32_t RIGID_PAYLOAD_FIELD_Z_REL_POS = 2;
	inline constexpr uint32_t RIGID_PAYLOAD_FIELD_VOL = 3;
	inline constexpr uint32_t RIGID_PAYLOAD_FIELD_PSI = 4;

	// packed particle count lanes exchanged once per redistribution
	inline constexpr int PARTICLE_REDISTRIBUTION_LANE_COUNT = 6;
	inline constexpr int PARTICLE_REDISTRIBUTION_LANE_OWNED_FLUID = 0;
	inline constexpr int PARTICLE_REDISTRIBUTION_LANE_HALO_FLUID = 1;
	inline constexpr int PARTICLE_REDISTRIBUTION_LANE_OWNED_BOUNDARY = 2;
	inline constexpr int PARTICLE_REDISTRIBUTION_LANE_HALO_BOUNDARY = 3;
	inline constexpr int PARTICLE_REDISTRIBUTION_LANE_OWNED_RIGID = 4;
	inline constexpr int PARTICLE_REDISTRIBUTION_LANE_HALO_RIGID = 5;

	// disjoint MPI tag ranges for the six particle payloads
	inline constexpr int PARTICLE_REDISTRIBUTION_TAG_OWNED_FLUID = 5310;
	inline constexpr int PARTICLE_REDISTRIBUTION_TAG_HALO_FLUID = 5410;
	inline constexpr int PARTICLE_REDISTRIBUTION_TAG_OWNED_BOUNDARY = 5510;
	inline constexpr int PARTICLE_REDISTRIBUTION_TAG_HALO_BOUNDARY = 5610;
	inline constexpr int PARTICLE_REDISTRIBUTION_TAG_OWNED_RIGID = 5710;
	inline constexpr int PARTICLE_REDISTRIBUTION_TAG_HALO_RIGID = 5810;

	uint64_t AggregateParticleTrafficCounts(
		std::span<const int> packedCounts,
		int lane,
		int laneCount,
		const char* countName);

	class MpiContext;

	struct ParticleRedistributionRoutes
	{
		// outgoing migration and halo routes, defining what the destination rank does with them
		// (outer vector = destination rank, inner vector = local particle indices)
		std::vector<std::vector<uint32_t>> ownedFluidLocalIdxs_ByDstRank;
		std::vector<std::vector<uint32_t>> haloFluidLocalIdxs_ByDstRank;
		std::vector<std::vector<uint32_t>> ownedBoundaryLocalIdxs_ByDstRank;
		std::vector<std::vector<uint32_t>> haloBoundaryLocalIdxs_ByDstRank;
		std::vector<std::vector<uint32_t>> ownedRigidLocalIdxs_ByDstRank;
		std::vector<std::vector<uint32_t>> haloRigidLocalIdxs_ByDstRank;

		// particles retained locally in the rebuilt backend state
		std::vector<uint32_t> keptOwnedFluidLocalIdxs;
		std::vector<uint32_t> localHaloFluidLocalIdxs;
		std::vector<uint32_t> keptOwnedBoundaryLocalIdxs;
		std::vector<uint32_t> localHaloBoundaryLocalIdxs;
		std::vector<uint32_t> keptOwnedRigidLocalIdxs;
		std::vector<uint32_t> localHaloRigidLocalIdxs;

		// set when any neighbor needs the particle as halo data, so a zero marks an inner owned particle
		std::vector<uint8_t> fluidIsOuterOwned_ByLocalIdx;
		std::vector<uint8_t> boundaryIsOuterOwned_ByLocalIdx;
		std::vector<uint8_t> rigidIsOuterOwned_ByLocalIdx;
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
		// ownership migration
		uint32_t N_sentOwnedF = 0;
		uint32_t N_recvOwnedF = 0;
		uint32_t N_sentOwnedB = 0;
		uint32_t N_recvOwnedB = 0;
		uint32_t N_sentOwnedR = 0;
		uint32_t N_recvOwnedR = 0;

		// halo exchange
		uint64_t N_sentHaloF = 0;
		uint64_t N_recvHaloF = 0;
		uint64_t N_sentHaloB = 0;
		uint64_t N_recvHaloB = 0;
		uint64_t N_sentHaloR = 0;
		uint64_t N_recvHaloR = 0;
	};

	class ParticleRedistributionMgr final
	{
	public:

		ParticleRedistributionMgr() = default;

		// builds rank-indexed routes from the current owned particles without communication
		bool BuildExchangeRoutes(
			const Sync::FluidConstView& particlesF,
			const Sync::BoundaryConstView& particlesB,
			const Sync::RigidConstView& particlesR,
			const Sync::RigidBodies& rigidBodies,
			const DistParticleState& particleStateF,
			const DistParticleState& particleStateB,
			const DistParticleState& particleStateR,
			const IDomainDecomp& domainDecomp,
			bool redistributeParticlesB,
			int rank, int N_ranks,
			uint64_t N_globalF, uint64_t N_globalB, uint64_t N_globalR,
			std::string* failMsg = nullptr);

		// routes and exchanges F/B/R particles, then updates backend storage
		bool RedistributeParticles(
			Sim::I_DistributedParticleBackend& distParticleBackend,
			DistParticleState& particleStateF,
			DistParticleState& particleStateB,
			DistParticleState& particleStateR,
			const Sync::RigidBodies& rigidBodies,
			const IDomainDecomp& domainDecomp,
			const MpiContext& mpiContext,
			const Sim::SimParams& simParams,
			bool redistributeParticlesB,
			uint64_t N_globalF,
			uint64_t N_globalB,
			uint64_t N_globalR,
			std::string* failMsg = nullptr);

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
			uint64_t ParticleIdAt(uint32_t bufferIdx) const; // returns the particle ID at the buffer index
			Mem::MemUsage GetMemoryUsage() const;

		private:

			// logical particle count, which storage only grows to hold
			uint32_t N_particles_ = 0;

			// packed payload storage, holding the particle IDs and one contiguous block
			// per float field, indexed by [fieldIdx * N_particles_ + bufferIdx]
			std::vector<uint64_t> particleIds_;
			std::vector<float> fieldValues_;
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
			uint64_t ParticleIdAt(uint32_t bufferIdx) const; // returns the particle ID at the buffer index
			Mem::MemUsage GetMemoryUsage() const;

		private:

			// logical particle count, which storage only grows to hold
			uint32_t N_particles_ = 0;

			// packed payload storage, holding the particle IDs and one contiguous block
			// per float field, indexed by [fieldIdx * N_particles_ + bufferIdx]
			std::vector<uint64_t> particleIds_;
			std::vector<float> fieldValues_;
		};

		// packed SoA payload for migrated or halo rigid particles
		class RigidExchangeBuffer
		{
		public:

			void Clear();
			void Resize(uint32_t N);

			// resizes and gathers the given backend particles into the buffer
			void SetBulkFromParticleView(
				const Sync::RigidConstView& src,
				const std::vector<uint32_t>& localIdxs);

			// copies the whole buffer into a contiguous backend range
			void CopyBulkToLocalParticles(
				Sync::RigidParticles& dst,
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
			Mem::MemUsage GetMemoryUsage() const;

		private:

			// logical particle count, which storage only grows to hold
			uint32_t N_particles_ = 0;

			// packed particle IDs, body IDs, and one contiguous block per float field,
			// indexed by [fieldIdx * N_particles_ + bufferIdx]
			std::vector<uint64_t> particleIds_;
			std::vector<uint32_t> bodyIds_;
			std::vector<float> fieldValues_;
		};

		// send and receive storage for exchange with one neighbor rank
		struct RankBuffers
		{
			// ownership migration payloads
			FluidExchangeBuffer sendOwnedParticlesF;
			FluidExchangeBuffer recvOwnedParticlesF;
			BoundaryExchangeBuffer sendOwnedParticlesB;
			BoundaryExchangeBuffer recvOwnedParticlesB;
			RigidExchangeBuffer sendOwnedParticlesR;
			RigidExchangeBuffer recvOwnedParticlesR;

			// halo exchange payloads
			FluidExchangeBuffer sendHaloParticlesF;
			FluidExchangeBuffer recvHaloParticlesF;
			BoundaryExchangeBuffer sendHaloParticlesB;
			BoundaryExchangeBuffer recvHaloParticlesB;
			RigidExchangeBuffer sendHaloParticlesR;
			RigidExchangeBuffer recvHaloParticlesR;

			// ownership migration counts
			int N_sendOwnedF = 0;
			int N_recvOwnedF = 0;
			int N_sendOwnedB = 0;
			int N_recvOwnedB = 0;
			int N_sendOwnedR = 0;
			int N_recvOwnedR = 0;

			// halo counts
			int N_sendHaloF = 0;
			int N_recvHaloF = 0;
			int N_sendHaloB = 0;
			int N_recvHaloB = 0;
			int N_sendHaloR = 0;
			int N_recvHaloR = 0;
		};

		// backend-resident particle position access
		static glm::vec3 FluidPositionAtLocalIdx(
			const Sync::FluidConstView& particles,
			uint32_t localIdx);
		static glm::vec3 BoundaryPositionAtLocalIdx(
			const Sync::BoundaryConstView& particles,
			uint32_t localIdx);
		static glm::vec3 RigidPositionAtLocalIdx(
			const Sync::RigidConstView& particles,
			const Sync::RigidBodies& bodies,
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
		static void CopyRigidParticle(
			const Sync::RigidConstView& src,
			Sync::RigidParticles& dst,
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
			uint64_t N_global,
			uint64_t particleId,
			uint32_t localIdx,
			uint32_t haloIdx);
		static bool ValidateParticleViews(
			const Sync::FluidConstView& particlesF,
			const Sync::BoundaryConstView& particlesB,
			const DistParticleState& particleStateF,
			const DistParticleState& particleStateB,
			uint64_t N_globalF, uint64_t N_globalB,
			std::string* failMsg);

		// routing and communication phases
		void PrepareBuffers(int N_ranks);
		void ClearFluidRouteOutputs();
		void ClearBoundaryRouteOutputs();
		void ClearRigidRouteOutputs();
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
		bool RouteOwnedRigidParticles(
			const Sync::RigidConstView& particlesR,
			const Sync::RigidBodies& rigidBodies,
			const DistParticleState& particleStateR,
			const IDomainDecomp& domainDecomp,
			int rank, int N_ranks,
			std::string* failMsg);
		bool RouteOwnedParticles(
			const Sync::FluidConstView& particlesF,
			const Sync::BoundaryConstView& particlesB,
			const Sync::RigidConstView& particlesR,
			const Sync::RigidBodies& rigidBodies,
			const DistParticleState& particleStateF,
			const DistParticleState& particleStateB,
			const DistParticleState& particleStateR,
			const IDomainDecomp& domainDecomp,
			bool redistributeParticlesB,
			int rank, int N_ranks,
			std::string* failMsg);
		void PackSendBuffers(
			const Sync::FluidConstView& particlesF,
			const Sync::BoundaryConstView& particlesB,
			const Sync::RigidConstView& particlesR,
			bool redistributeParticlesB,
			int rank, int N_ranks);
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
			uint64_t N_global,
			std::string* failMsg);
		bool PrepareBoundaryParticleMutation(
			const Sync::BoundaryConstView& oldParticles,
			const IDomainDecomp& domainDecomp,
			int rank,
			float haloRadius,
			uint64_t N_global,
			std::string* failMsg);
		bool PrepareRigidParticleMutation(
			const Sync::RigidConstView& oldParticles,
			const Sync::RigidBodies& rigidBodies,
			const IDomainDecomp& domainDecomp,
			int rank,
			float haloRadius,
			uint64_t N_global,
			std::string* failMsg);

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
		std::vector<uint32_t> retainedRigidOldLocalIdx_FromNewLocalIdx_;
		std::vector<uint64_t> migratedParticleIdScratch_;
		Sync::FluidParticles appendedParticlesF_;
		Sync::BoundaryParticles appendedParticlesB_;
		Sync::RigidParticles appendedParticlesR_;

		// metadata matching the compacted and appended backend state
		DistParticleState rebuiltParticleStateF_;
		DistParticleState rebuiltParticleStateB_;
		DistParticleState rebuiltParticleStateR_;

		// latest per-step network traffic
		ParticleRedistributionStats redistStats_{};
	};

} // namespace Dist
