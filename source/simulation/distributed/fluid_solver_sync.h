// fluid_solver_sync.h

#pragma once

// external
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

// internal
#include "source/eval/bench/benchmarks.h"
#include "source/eval/memory/metrics_mem.h"
#include "source/simulation/core/backend_sim.h"
#include "source/simulation/sync/state_sim.h"
#include "dist_particle_state.h"
#include "domain_decomp.h"
#include "mpi_context.h"



namespace Dist
{
	// separate tags for plan counts, global-index validation, and solver payloads
	inline constexpr int FLUID_SOLVER_SYNC_TAG_COUNT = 6200;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_GLOBAL_IDXS = 6201;
	inline constexpr int FLUID_SOLVER_SYNC_TAG_PAYLOAD = 6202;

	// particles packed per block, keeping one block of every field cache resident
	// TODO: fine-tune chunk size by trying out 1024, 2048, 4096, 8192, 16384, 32768?
	inline constexpr uint32_t FLUID_SOLVER_SYNC_BLOCK_SIZE = 2048;

	// owner-to-halo fluid field synchronization over a cached per-neighbor exchange plan
	class FluidSolverSyncMgr final : public Sim::I_DistributedFluidSolverSync
	{
	public:

		// builds the peer exchange plan from the current owned and remote particles
		// (collective, full peer-plan validation is enabled in debug/validation builds)
		bool Rebuild(
			const Sync::FluidConstView& fluidParticles,
			const DistParticleState& fluidParticleState,
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
		void SynchronizeFluidScalarField(
			std::span<float> values,
			Base::DistHaloMode haloExtent) override;
		void SynchronizeFluidVectorField(
			std::span<float> xValues,
			std::span<float> yValues,
			std::span<float> zValues,
			Base::DistHaloMode haloExtent) override;

		// sums rank-local solver errors, normalization happens in the solver
		Sim::PressureSolveErrorSums GlobalizePressureSolveErrorSums(
			const Sim::PressureSolveErrorSums& localSums) const override;
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

		// narrow/wide plans and reusable payload storage for one neighboring rank
		struct PeerPlan
		{
			int rank = -1;
			ParticlePlan singleHalo;
			ParticlePlan doubleHalo;

			// reusable payload storage
			std::vector<float> sendBuffer;
			std::vector<float> recvBuffer;
		};

		// packs, exchanges, and unpacks up to three fields in one round of messages
		void SynchronizeFluidFields(
			const std::array<std::span<float>, 3>& fields,
			uint32_t N_fields,
			Base::DistHaloMode haloExtent);

		// selects the narrow or wide exchange plan for one peer,
		// overloads allow for mutable or read-only access as needed
		static ParticlePlan& ParticlePlanFor(
			PeerPlan& peerPlan,
			Base::DistHaloMode haloExtent);
		static const ParticlePlan& ParticlePlanFor(
			const PeerPlan& peerPlan,
			Base::DistHaloMode haloExtent);

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
		std::vector<MPI_Request> requests_;

		// reusable rebuild storage
		std::vector<PeerPlan> nextPeerPlans_;
		std::vector<std::pair<uint32_t, uint32_t>> peerIdxSortScratch_;
	};

} // namespace Dist
