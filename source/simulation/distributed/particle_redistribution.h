// particle_redistribution.h
#pragma once

// external
#include <array>
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
		// (kept owned particles are every owned particle without an outgoing migration route)
		uint32_t N_keptOwnedF = 0;
		uint32_t N_keptOwnedB = 0;
		uint32_t N_keptOwnedR = 0;
		std::vector<uint32_t> localHaloFluidLocalIdxs;
		std::vector<uint32_t> localHaloBoundaryLocalIdxs;
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
		std::vector<uint16_t> ownerRank_ByOwnedIdx;

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
		// (outer fluid particles index the rebuilt layout until the backend re-orders it)
		bool RedistributeParticles(
			Sim::I_DistributedParticleBackend& distParticleBackend,
			DistParticleState& particleStateF,
			DistParticleState& particleStateB,
			DistParticleState& particleStateR,
			std::vector<uint32_t>& outerLocalIdxsF,
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

		// payload float fields of migrated or halo F particles, in packed order
		// (the fields velPred, accPress, sourceTerm, diagElem, pressure, and density are omitted,
		// because the following solver step re-writes them before reading them)
		// TODO: keep it like this or copy them too, if solver step behavior changes?
		struct FluidPayload
		{
			using ConstView = Sync::FluidConstView;
			using Particles = Sync::FluidParticles;
			static constexpr const char* particleTypeName = "fluid";
			static constexpr uint32_t N_fields = 6;
			static constexpr bool hasBodyIds = false;

			static std::array<std::span<const float>, N_fields> Fields(const ConstView& particlesF)
			{
				return
				{
					particlesF.xPos, particlesF.yPos, particlesF.zPos,
					particlesF.xVel, particlesF.yVel, particlesF.zVel
				};
			}
			static std::array<std::vector<float>*, N_fields> Fields(Particles& particlesF)
			{
				return
				{
					&particlesF.xPos, &particlesF.yPos, &particlesF.zPos,
					&particlesF.xVel, &particlesF.yVel, &particlesF.zVel
				};
			}
		};

		// payload float fields of migrated or halo B particles, in packed order
		struct BoundaryPayload
		{
			using ConstView = Sync::BoundaryConstView;
			using Particles = Sync::BoundaryParticles;
			static constexpr const char* particleTypeName = "boundary";
			static constexpr uint32_t N_fields = 5;
			static constexpr bool hasBodyIds = false;

			static std::array<std::span<const float>, N_fields> Fields(const ConstView& particlesB)
			{
				return
				{
					particlesB.xPos, particlesB.yPos, particlesB.zPos,
					particlesB.volB, particlesB.psiB
				};
			}
			static std::array<std::vector<float>*, N_fields> Fields(Particles& particlesB)
			{
				return
				{
					&particlesB.xPos, &particlesB.yPos, &particlesB.zPos,
					&particlesB.volB, &particlesB.psiB
				};
			}
		};

		// payload float fields of migrated or halo R particles, in packed order
		// (rigid particles also carry their body IDs)
		struct RigidPayload
		{
			using ConstView = Sync::RigidConstView;
			using Particles = Sync::RigidParticles;
			static constexpr const char* particleTypeName = "rigid";
			static constexpr uint32_t N_fields = 5;
			static constexpr bool hasBodyIds = true;

			static std::array<std::span<const float>, N_fields> Fields(const ConstView& particlesR)
			{
				return
				{
					particlesR.xPosRel, particlesR.yPosRel, particlesR.zPosRel,
					particlesR.volR, particlesR.psiR
				};
			}
			static std::array<std::vector<float>*, N_fields> Fields(Particles& particlesR)
			{
				return
				{
					&particlesR.xPosRel, &particlesR.yPosRel, &particlesR.zPosRel,
					&particlesR.volR, &particlesR.psiR
				};
			}
		};

		// packed SoA payload for migrated or halo particles of one species
		template <typename Payload>
		class ExchangeBuffer
		{
		public:

			void Clear();
			void Resize(uint32_t N);

			// resizes and gathers the given backend particles into the buffer
			void SetBulkFromParticleView(
				const typename Payload::ConstView& src,
				const std::vector<uint32_t>& localIdxs);

			// copies the whole buffer into a contiguous backend range
			void CopyBulkToLocalParticles(
				typename Payload::Particles& dst,
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

			// packed particle IDs, body IDs (rigid only), and one contiguous block per float field,
			// indexed by [fieldIdx * N_particles_ + bufferIdx]
			std::vector<uint64_t> particleIds_;
			std::vector<uint32_t> bodyIds_;
			std::vector<float> fieldValues_;
		};

		// send and receive storage for exchange with one neighbor rank
		struct RankBuffers
		{
			// ownership migration payloads
			ExchangeBuffer<FluidPayload> sendOwnedParticlesF;
			ExchangeBuffer<FluidPayload> recvOwnedParticlesF;
			ExchangeBuffer<BoundaryPayload> sendOwnedParticlesB;
			ExchangeBuffer<BoundaryPayload> recvOwnedParticlesB;
			ExchangeBuffer<RigidPayload> sendOwnedParticlesR;
			ExchangeBuffer<RigidPayload> recvOwnedParticlesR;

			// halo exchange payloads
			ExchangeBuffer<FluidPayload> sendHaloParticlesF;
			ExchangeBuffer<FluidPayload> recvHaloParticlesF;
			ExchangeBuffer<BoundaryPayload> sendHaloParticlesB;
			ExchangeBuffer<BoundaryPayload> recvHaloParticlesB;
			ExchangeBuffer<RigidPayload> sendHaloParticlesR;
			ExchangeBuffer<RigidPayload> recvHaloParticlesR;

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

		// sparse compaction construction, filling the slots of departing owned particles
		// inside the retained prefix with the retained particles above it
		bool BuildCompactionMoves(
			uint32_t N_oldOwned,
			uint32_t N_oldLocal,
			uint32_t N_retained,
			const std::vector<std::vector<uint32_t>>& departingLocalIdxs_ByDstRank,
			std::vector<Sync::ParticleCompactionMove>& compactionMoves,
			std::string* failMsg);
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

		// prepares retained compaction, appended particles, and matching metadata of one species
		// (countOuterParticles(N_owned, N_retained) classifies the rebuilt owned prefix, rigidBodies is read for rigid only)
		template <typename Payload, typename CountOuterParticles>
		bool PrepareParticleMutation(
			const typename Payload::ConstView& oldParticles,
			uint32_t N_oldOwned,
			uint32_t N_retained,
			const std::vector<std::vector<uint32_t>>& ownedLocalIdxs_ByDstRank,
			const std::vector<uint32_t>& localHaloLocalIdxs,
			ExchangeBuffer<Payload> RankBuffers::* recvOwnedParticles,
			ExchangeBuffer<Payload> RankBuffers::* recvHaloParticles,
			uint32_t N_recvOwned,
			uint64_t N_recvHalo,
			const Sync::RigidBodies& rigidBodies,
			uint64_t N_global,
			std::vector<Sync::ParticleCompactionMove>& compactionMoves,
			typename Payload::Particles& appendedParticles,
			DistParticleState& rebuiltParticleState,
			CountOuterParticles&& countOuterParticles,
			std::string* failMsg);

		// peer communication routes and storage
		std::vector<RankBuffers> rankBuffers_;
		ParticleRedistributionRoutes routes_;
		ParticleRoutingWorkBuffers routingWorkBuffers_;
		ParticleClassificationWorkBuffers ownedClassificationWorkBuffers_;
		std::vector<int> sendCounts_;
		std::vector<int> recvCounts_;

		// sparse compaction moves and append-only particle storage
		std::vector<Sync::ParticleCompactionMove> compactionMovesF_;
		std::vector<Sync::ParticleCompactionMove> compactionMovesB_;
		std::vector<Sync::ParticleCompactionMove> compactionMovesR_;
		std::vector<uint32_t> departingOldLocalIdxScratch_;
		std::vector<uint64_t> migratedParticleIdScratch_;
		Sync::FluidParticles appendedParticlesF_;
		Sync::BoundaryParticles appendedParticlesB_;
		Sync::RigidParticles appendedParticlesR_;

		// metadata matching the compacted and appended backend state
		DistParticleState rebuiltParticleStateF_;
		DistParticleState rebuiltParticleStateB_;
		DistParticleState rebuiltParticleStateR_;
		std::vector<uint32_t> rebuiltOuterLocalIdxsF_;

		// latest per-step network traffic
		ParticleRedistributionStats redistStats_{};
	};

} // namespace Dist
