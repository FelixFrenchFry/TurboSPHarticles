// fluid_solver_sync.h
#pragma once

// external
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

// internal
#include "source/eval/memory/categories.h"
#include "source/simulation/core/backend_sim.h"
#include "source/simulation/sync/state_sim.h"
#include "dist_particle_state.h"
#include "domain_decomp.h"
#include "mpi_context.h"



namespace Dist
{
	// separate tags for plan validation and solver payloads
	inline constexpr int FLUID_SOLVER_SYNC_TAG_PLAN_SUMMARY = 6200;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_GLOBAL_IDXS = 6201;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_DENSITY = 6202;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_BYTE_PAYLOAD = 6203;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_PREDICTED_VELOCITY = 6205;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_INITIAL_PRESSURE = 6206;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_PRESSURE_ACCELERATION = 6207;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_PRESSURE = 6208;

	struct FluidSolverSyncPayloadProtocol
	{
		int tag = -1;
		uint32_t N_bytesPerParticle = 0;
	};

	// phase-specific message identity and payload shape for solver field exchanges
	std::optional<FluidSolverSyncPayloadProtocol> GetFluidSolverSyncPayloadProtocol(
		Sim::FluidSolverSyncPhase phase) noexcept;

	// owned and halo particles classified per chunk, so workers fill disjoint plan ranges
	// TODO: fine-tune chunk size by trying out 2048, 4096, 8192, 16384?
	inline constexpr uint32_t FLUID_SOLVER_SYNC_PLAN_CHUNK_SIZE = 4096;

	// one exchange plan and work schedule per halo extent mode
	inline constexpr uint32_t N_HALO_EXTENTS = 2;

	// owner-to-halo fluid field synchronization over a cached per-neighbor exchange plan
	class FluidSolverSyncMgr final : public Sim::I_DistributedFluidSolverSync
	{
	public:

		// builds the peer exchange plan from the current owned and remote particles
		// (collective peer-count and fingerprint validation is enabled in every build,
		// full peer-plan validation is enabled in debug/validation builds)
		bool Rebuild(
			const Sync::FluidConstView& particlesF,
			const DistParticleState& particleStateF,
			std::span<const uint32_t> outerLocalIdxsF,
			const IDomainDecomp& domainDecomp,
			float singleHaloRadius,
			Base::DistHaloMode distHaloMode,
			uint64_t N_expectedGlobalF,
			MpiContext& mpiContext,
			std::string* failMsg = nullptr);

		// hands the local index lists of the published plan over, once per rebuild and halo extent
		void SwapFluidSyncLocalIdxs(
			Base::DistHaloMode haloExtent,
			std::vector<uint32_t>& sendLocalIdxs,
			std::vector<uint32_t>& recvLocalIdxs) override;

		// posts the receives and sends of every peer slice of the packed payloads
		void BeginFluidHaloExchange(
			std::span<const std::byte> sendPayload,
			std::span<std::byte> recvPayload,
			Base::DistHaloMode haloExtent,
			Sim::FluidSolverSyncPhase phase) override;
		void EndFluidHaloExchange() override;

		// sums rank-local solver errors, normalization happens in the solver
		Sim::PressureSolveErrorSums GlobalizePressureSolveErrorSums(
			const Sim::PressureSolveErrorSums& localSums) const override;
		uint64_t GetGlobalFluidParticleCount() const override { return N_globalF_; }

		Mem::MemUsage GetExchangeBufferMemoryUsage() const;
		Mem::MemUsage GetPlanMemoryUsage() const;

	private:

		// particle slots shared with every neighboring rank for one halo extent, peer-major
		// (sorted by global particle ID within each peer slice, local indices in the layout of the latest Rebuild())
		struct ParticlePlan
		{
			// local slots read from and written to
			std::vector<uint32_t> sendLocalIdxs;
			std::vector<uint32_t> recvLocalIdxs;

			// global particle IDs matching the local index lists, used to order and validate the plan
			std::vector<uint64_t> sendParticleIds;
			std::vector<uint64_t> recvParticleIds;

			// peer slice offsets into the lists, indexed by [peerIdx] up to N_peers (inclusive)
			// the combined send list can include the same local particle for multiple peers
			std::vector<uint64_t> sendOffsets_ByPeer;
			std::vector<uint64_t> recvOffsets_ByPeer;

			// set by Rebuild(), cleared once the local index lists are handed over
			bool localIdxsPublished = false;
		};

		// reusable scratch for parallel construction of variable-sized peer plans
		struct PlanBuildWorkBuffers
		{
			// per chunk bucket counts and write offsets,
			// indexed by [chunkIdx * N_buckets + bucketIdx]
			std::vector<uint32_t> sendCounts_ByChunkBucket;
			std::vector<uint32_t> sendWriteOffsets_ByChunkBucket;
			std::vector<uint32_t> sendTotals;
			std::vector<uint32_t> recvCounts_ByChunkBucket;
			std::vector<uint32_t> recvWriteOffsets_ByChunkBucket;
			std::vector<uint32_t> recvTotals;

			// compact peer lookup and precomputed send bounds
			std::vector<int> peerIdx_ByRank;
			std::vector<AABB> sendSingleBounds_ByPeer;
			std::vector<AABB> sendDoubleBounds_ByPeer;

			// owner peer selected during the receive count pass
			std::vector<int> recvPeerIdx_ByHaloIdx;
		};

		// set when Rebuild() publishes a validated plan
		MpiContext* mpiContext_ = nullptr;

		// particle count the current plan was built for
		uint64_t N_globalF_ = 0;
		Base::DistHaloMode distHaloMode_ = Base::DistHaloMode::SingleHalo;

		// neighboring ranks in ascending order, single/double halo plans and reusable MPI requests
		std::vector<int> peerRanks_;
		std::array<ParticlePlan, N_HALO_EXTENTS> particlePlans_;

		std::vector<MPI_Request> requests_;
		std::vector<MpiRequestInfo> requestInfos_;
		Sim::FluidSolverSyncPhase pendingPhase_ = Sim::FluidSolverSyncPhase::Density;
		bool fluidFieldExchangePending_ = false;

		// reusable rebuild storage
		PlanBuildWorkBuffers planBuildWorkBuffers_;
		std::vector<std::pair<uint64_t, uint32_t>> peerIdxSortScratch_;
	};

} // namespace Dist
