// fluid_solver_sync.h
#pragma once

// external
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

// internal
#include "source/eval/bench/benchmarks.h"
#include "source/eval/memory/categories.h"
#include "source/simulation/core/backend_sim.h"
#include "source/simulation/sync/state_sim.h"
#include "dist_particle_state.h"
#include "domain_decomp.h"
#include "mpi_context.h"



namespace Dist
{
	// separate tags for plan validation and solver payloads
	inline constexpr int FLUID_SOLVER_SYNC_TAG_COUNT = 6200;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_GLOBAL_IDXS = 6201;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_DENSITY = 6202;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_BYTE_PAYLOAD = 6203;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_FINGERPRINT = 6204;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_PREDICTED_VELOCITY = 6205;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_INITIAL_PRESSURE = 6206;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_PRESSURE_ACCELERATION = 6207;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_PRESSURE = 6208;

	struct FluidSolverSyncPayloadProtocol
	{
		int tag = -1;
		uint32_t N_fields = 0;
	};

	// phase-specific message identity and payload shape for solver field exchanges
	std::optional<FluidSolverSyncPayloadProtocol> GetFluidSolverSyncPayloadProtocol(
		Sim::FluidSolverSyncPhase phase) noexcept;

	// particles packed per block, keeping one block of every field cache resident
	// TODO: fine-tune chunk size by trying out 1024, 2048, 4096, 8192, 16384, 32768?
	inline constexpr uint32_t FLUID_SOLVER_SYNC_BLOCK_SIZE = 2048;

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
			const IDomainDecomp& domainDecomp,
			float singleHaloRadius,
			Base::DistHaloMode distHaloMode,
			uint64_t N_expectedGlobalF,
			MpiContext& mpiContext,
			Bench::BenchmarksTracker& benchmarksTracker,
			std::string* failMsg = nullptr);

		// remaps the plan onto re-ordered backend storage, leaving global particle IDs unchanged
		void ApplyFluidParticleReOrder(
			std::span<const uint32_t> newFluidLocalIdx_FromOldLocalIdx) override;

		// overwrites remote values with the ones held by their owning rank
		void BeginSynchronizeFluidScalarField(
			std::span<float> values,
			Base::DistHaloMode haloExtent,
			Sim::FluidSolverSyncPhase phase,
			uint32_t iteration) override;
		void BeginSynchronizeFluidVectorField(
			std::span<float> xValues,
			std::span<float> yValues,
			std::span<float> zValues,
			Base::DistHaloMode haloExtent,
			Sim::FluidSolverSyncPhase phase,
			uint32_t iteration) override;
		void EndSynchronizeFluidFields() override;

		void SynchronizeFluidScalarField(
			std::span<float> values,
			Base::DistHaloMode haloExtent,
			Sim::FluidSolverSyncPhase phase,
			uint32_t iteration) override;
		void SynchronizeFluidVectorField(
			std::span<float> xValues,
			std::span<float> yValues,
			std::span<float> zValues,
			Base::DistHaloMode haloExtent,
			Sim::FluidSolverSyncPhase phase,
			uint32_t iteration) override;
		void SynchronizeFluidParticleFlagsToHalos(
			std::span<uint8_t> flags,
			Base::DistHaloMode haloExtent) override;

		// sums rank-local solver errors, normalization happens in the solver
		Sim::PressureSolveErrorSums GlobalizePressureSolveErrorSums(
			const Sim::PressureSolveErrorSums& localSums,
			uint32_t iteration) const override;
		uint64_t GetGlobalFluidParticleCount() const override { return N_globalF_; }

		Mem::MemUsage GetExchangeBufferMemoryUsage() const;
		Mem::MemUsage GetPlanMemoryUsage() const;

	private:

		// particle slots shared with one neighboring rank for one halo extent
		struct ParticlePlan
		{
			// local slots read from and written to, sorted by global particle ID
			std::vector<uint32_t> sendLocalIdxs;
			std::vector<uint32_t> recvLocalIdxs;

			// global particle IDs matching the local index lists
			std::vector<uint32_t> sendGlobalIdxs;
			std::vector<uint32_t> recvGlobalIdxs;
		};

		// single/double halo plans and reusable payload storage for one neighboring rank
		struct PeerPlan
		{
			int rank = -1;
			std::array<ParticlePlan, N_HALO_EXTENTS> particlePlans;

			// reusable payload storage
			std::vector<float> sendBuffer;
			std::vector<float> recvBuffer;
			std::vector<uint8_t> sendByteBuffer;
			std::vector<uint8_t> recvByteBuffer;
		};

		// one contiguous slice of one peer payload, packed or unpacked by a single worker
		struct FieldSyncWorkItem
		{
			uint32_t peerIdx = 0;
			uint32_t beginParticleIdx = 0;
			uint32_t endParticleIdx = 0;
		};

		// flat pack and unpack work covering every peer for one halo extent
		struct FieldSyncSchedule
		{
			std::vector<FieldSyncWorkItem> sendWorkItems;
			std::vector<FieldSyncWorkItem> recvWorkItems;
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

		// packs and posts exchange of up to three fields in one round of messages
		void BeginSynchronizeFluidFields(
			const std::array<std::span<float>, 3>& fields,
			uint32_t N_fields,
			Base::DistHaloMode haloExtent,
			Sim::FluidSolverSyncPhase phase,
			uint32_t iteration);

		// rebuilds both single and double halo work schedules from the published peer plans
		void BuildFieldSyncSchedules();

		// splits one peer particle range into block-sized work items
		static void AppendFieldSyncWorkItems(
			std::vector<FieldSyncWorkItem>& workItems,
			uint32_t peerIdx,
			uint32_t N_particles);

		// clears one plan for the next rebuild
		static void ClearParticlePlan(ParticlePlan& particlePlan);

		// set when Rebuild() publishes a validated plan
		MpiContext* mpiContext_ = nullptr;
		Bench::BenchmarksTracker* benchmarksTracker_ = nullptr;

		// particle counts the current plan was built for
		uint64_t N_globalF_ = 0;
		uint32_t N_localF_ = 0;
		Base::DistHaloMode distHaloMode_ = Base::DistHaloMode::SingleHalo;

		// peer exchange plans and reusable MPI requests
		std::vector<PeerPlan> peerPlans_;

		// flat pack and unpack work covering every peer, rebuilt with the plans
		std::array<FieldSyncSchedule, N_HALO_EXTENTS> fieldSyncSchedules_;

		std::vector<MPI_Request> requests_;
		std::vector<MpiRequestInfo> requestInfos_;
		std::array<std::span<float>, 3> pendingFields_{};
		uint32_t N_pendingFields_ = 0;
		Base::DistHaloMode pendingHaloExtent_ = Base::DistHaloMode::SingleHalo;
		Sim::FluidSolverSyncPhase pendingPhase_ = Sim::FluidSolverSyncPhase::Density;
		uint32_t pendingIteration_ = 0;
		uint64_t pendingEpoch_ = 0;
		mutable uint64_t synchronizationEpoch_ = 0;
		bool fluidFieldExchangePending_ = false;

		// reusable rebuild storage
		std::vector<PeerPlan> nextPeerPlans_;
		PlanBuildWorkBuffers planBuildWorkBuffers_;
		std::vector<std::pair<uint32_t, uint32_t>> peerIdxSortScratch_;
	};

} // namespace Dist
