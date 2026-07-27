// particle_redistribution.h

#pragma once

// external
#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>

// internal
#include "Source/Sim/Core/backend_sim.h"
#include "dist_particle_state.h"
#include "domain_decomp.h"



namespace Dist
{
	class MpiContext;
	struct ParticleRedistributionTimingReport;

	struct ParticleRedistributionRoutes
	{
		// outgoing migration and halo routes, defining what the destination rank does with them
		// (outer vector = destination rank, inner vector = local particle indices)
		std::vector<std::vector<int>> ownedFluidLocalIdxs_ByDstRank;
		std::vector<std::vector<int>> haloFluidLocalIdxs_ByDstRank;
		std::vector<std::vector<int>> haloBoundaryLocalIdxs_ByDstRank;

		// particles retained locally in the rebuilt backend state
		std::vector<int> keptOwnedFluidLocalIdxs;
		std::vector<int> keptOwnedBoundaryLocalIdxs;
		std::vector<int> localHaloFluidLocalIdxs;
	};

	struct ParticleRedistributionStats
	{
		// ownership migration traffic
		uint32_t sentOwnedFluidParticleCount = 0;
		uint32_t recvOwnedFluidParticleCount = 0;

		// fluid halo traffic
		uint32_t sentHaloFluidParticleCount = 0;
		uint32_t recvHaloFluidParticleCount = 0;

		// boundary halo traffic
		uint32_t sentHaloBoundaryParticleCount = 0;
		uint32_t recvHaloBoundaryParticleCount = 0;
	};

	class ParticleRedistributionMgr final
	{
	public:

		ParticleRedistributionMgr() = default;

		// builds rank-indexed routes from the current owned particles without communication
		bool BuildExchangeRoutes(
			const Sync::FluidConstView& fluidParticles,
			const Sync::BoundaryConstView& boundaryParticles,
			const DistParticleState& fluidParticleState,
			const DistParticleState& boundaryParticleState,
			const IDomainDecomp& domainDecomp,
			int rank, int rankCount,
			std::string* failMsg = nullptr,
			ParticleRedistributionTimingReport* timingReport = nullptr);

		// orchestrator that compacts kept particles in place and appends the next migrated / halo particles
		bool RedistributeParticles(
			Sim::I_DistributedParticleBackend& distParticleBackend,
			DistParticleState& fluidParticleState,
			DistParticleState& boundaryParticleState,
			const IDomainDecomp& domainDecomp,
			const MpiContext& mpiContext,
			const Sim::SimParams& simParams,
			uint32_t globalFluidParticleCount,
			uint32_t globalBoundaryParticleCount,
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

		// packed SoA payload for migrated or halo fluid particles
		class FluidExchangeBuffer
		{
		public:

			void Clear();
			void Resize(uint32_t N);

			// copies one particle from the backend into the buffer
			void SetFromParticleView(
				const Sync::FluidConstView& src,
				const DistParticleState& particleState,
				uint64_t bufferIdx,
				int localIdx);

			// copies one particle from the buffer into the backend
			void CopyToLocalParticle(
				Sync::FluidParticles& dst,
				uint64_t bufferIdx,
				int localIdx) const;

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
			uint64_t Size() const { return globalIdx_.size(); }
			bool IsEmpty() const { return Size() == 0; }
			int CheckedCount() const;
			int GlobalIdxAt(uint64_t bufferIdx) const;
			Mem::MemUsage GetMemoryUsage() const;

		private:

			std::vector<int> globalIdx_;
			Sync::FluidParticles particles_;
		};

		// packed SoA payload for migrated or halo boundary particles
		class BoundaryExchangeBuffer
		{
		public:

			void Clear();
			void Resize(uint32_t N);

			// copies one particle from the backend into the buffer
			void SetFromParticleView(
				const Sync::BoundaryConstView& src,
				const DistParticleState& particleState,
				uint64_t bufferIdx,
				int localIdx);

			// copies one particle from the buffer into the backend
			void CopyToLocalParticle(
				Sync::BoundaryParticles& dst,
				uint64_t bufferIdx,
				int localIdx) const;

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
			uint64_t Size() const { return globalIdx_.size(); }
			bool IsEmpty() const { return Size() == 0; }
			int CheckedCount() const;
			int GlobalIdxAt(uint64_t bufferIdx) const;
			Mem::MemUsage GetMemoryUsage() const;

		private:

			std::vector<int> globalIdx_;
			Sync::BoundaryParticles particles_;
		};

		// send and receive storage for exchange with one neighbor rank
		struct RankBuffers
		{
			// ownership migration payloads
			FluidExchangeBuffer sendOwnedFluidParticles;
			FluidExchangeBuffer recvOwnedFluidParticles;

			// fluid and boundary halo payloads
			FluidExchangeBuffer sendHaloFluidParticles;
			FluidExchangeBuffer recvHaloFluidParticles;
			BoundaryExchangeBuffer sendHaloBoundaryParticles;
			BoundaryExchangeBuffer recvHaloBoundaryParticles;

			// packed count lanes
			int sendOwnedFluidCount = 0;
			int recvOwnedFluidCount = 0;
			int sendHaloFluidCount = 0;
			int recvHaloFluidCount = 0;
			int sendHaloBoundaryCount = 0;
			int recvHaloBoundaryCount = 0;
		};

		#if defined(_DEBUG)
		// debug particle index and class validation
		static bool IsLocalIdxInRange(int localIdx, uint32_t particleCount);
		static bool HasParticleMetadataAtLocalIdx(
			const DistParticleState& particleState,
			int localIdx);
		static bool IsOwnedClass(DistParticleClass particleClass);
		#endif

		// backend-resident particle position access
		static glm::vec3 FluidPositionAtLocalIdx(
			const Sync::FluidConstView& particles,
			int localIdx);
		static glm::vec3 BoundaryPositionAtLocalIdx(
			const Sync::BoundaryConstView& particles,
			int localIdx);

		// particle property copying into append-only backend mutation storage
		static void CopyFluidParticle(
			const Sync::FluidConstView& src,
			Sync::FluidParticles& dst,
			int srcLocalIdx,
			int dstLocalIdx);
		static void CopyBoundaryParticle(
			const Sync::BoundaryConstView& src,
			Sync::BoundaryParticles& dst,
			int srcLocalIdx,
			int dstLocalIdx);

		// dense compaction and rebuilt metadata construction
		static bool BuildDenseCompactionMap(
			uint32_t oldLocalParticleCount,
			const std::vector<int>& retainedOldLocalIdxs,
			std::vector<uint32_t>& oldLocalIdx_FromNewLocalIdx,
			std::vector<uint8_t>& retainedOldLocalIdxFlags,
			std::string* failMsg);
		static void PrepareRebuiltParticleState(
			DistParticleState& particleState,
			uint32_t ownedParticleCount,
			uint32_t haloParticleCount,
			uint32_t localParticleCount,
			uint32_t globalParticleCount);
		static bool AddParticleMetadata(
			DistParticleState& particleState,
			DistParticleClass particleClass,
			uint32_t globalParticleCount,
			int globalIdx,
			int localIdx);
		static bool CheckedUint32(
			uint64_t value,
			uint32_t& result,
			const char* valueName,
			std::string* failMsg);
		static bool ValidateParticleViews(
			const Sync::FluidConstView& fluidParticles,
			const Sync::BoundaryConstView& boundaryParticles,
			const DistParticleState& fluidParticleState,
			const DistParticleState& boundaryParticleState,
			std::string* failMsg);

		// routing and communication phases
		void PrepareBuffers(int rankCount);
		bool RouteOwnedParticles(
			const Sync::FluidConstView& fluidParticles,
			const Sync::BoundaryConstView& boundaryParticles,
			const DistParticleState& fluidParticleState,
			const DistParticleState& boundaryParticleState,
			const IDomainDecomp& domainDecomp,
			int rank, int rankCount,
			std::string* failMsg,
			ParticleRedistributionTimingReport* timingReport);
		bool PackSendBuffers(
			const Sync::FluidConstView& fluidParticles,
			const Sync::BoundaryConstView& boundaryParticles,
			const DistParticleState& fluidParticleState,
			const DistParticleState& boundaryParticleState,
			int rank, int rankCount,
			std::string* failMsg);
		bool PrepareReceiveBuffers(
			int rank, int rankCount,
			std::string* failMsg);
		void ExchangePayloads(
			int rank, int rankCount,
			const MpiContext& mpiContext);
		bool PrepareFluidParticleMutation(
			const Sync::FluidConstView& oldParticles,
			const DistParticleState& oldParticleState,
			const IDomainDecomp& domainDecomp,
			int rank,
			float kernelRadius,
			uint32_t globalParticleCount,
			std::string* failMsg,
			ParticleRedistributionTimingReport* timingReport);
		bool PrepareBoundaryParticleMutation(
			const Sync::BoundaryConstView& oldParticles,
			const DistParticleState& oldParticleState,
			const IDomainDecomp& domainDecomp,
			int rank,
			float kernelRadius,
			uint32_t globalParticleCount,
			std::string* failMsg,
			ParticleRedistributionTimingReport* timingReport);

		// peer communication routes and storage
		std::vector<RankBuffers> rankBuffers_;
		ParticleRedistributionRoutes routes_;
		std::vector<int> sendCounts_;
		std::vector<int> recvCounts_;

		// dense compaction mappings and append-only particle storage
		std::vector<uint32_t> retainedFluidOldLocalIdx_FromNewLocalIdx_;
		std::vector<uint32_t> retainedBoundaryOldLocalIdx_FromNewLocalIdx_;
		std::vector<uint8_t> retainedOldLocalIdxFlags_;
		Sync::FluidParticles appendedFluidParticles_;
		Sync::BoundaryParticles appendedBoundaryParticles_;

		// metadata matching the compacted and appended backend state
		DistParticleState rebuiltFluidParticleState_;
		DistParticleState rebuiltBoundaryParticleState_;

		// latest per-step network traffic
		ParticleRedistributionStats redistStats_{};
	};

} // namespace Dist
