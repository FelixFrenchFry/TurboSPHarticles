// fluid_solver_sync.h

#pragma once

// external
#include <array>
#include <cstdint>
#include <span>
#include <string>
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
	// owner-to-halo fluid field synchronization over a cached per-neighbor exchange plan
	class FluidSolverSyncMgr final : public Sim::I_DistributedFluidSolverSync
	{
	public:

		// builds the peer exchange plan from the current owned and halo particles
		// (collective, and validated against the plan built by every neighboring rank)
		bool Rebuild(
			const Sync::FluidConstView& fluidParticles,
			const DistParticleState& fluidParticleState,
			const IDomainDecomp& domainDecomp,
			uint64_t N_expectedGlobalF,
			MpiContext& mpiContext,
			Bench::BenchmarksTracker& benchmarksTracker,
			std::string* failMsg = nullptr);

		// remaps the plan onto re-ordered backend storage, leaving global particle IDs unchanged
		void ApplyFluidParticleReOrder(
			std::span<const uint32_t> oldFluidLocalIdx_FromNewLocalIdx) override;

		// overwrites halo values with the ones held by their owning rank
		void SynchronizeFluidScalarField(std::span<float> values) override;
		void SynchronizeFluidVectorField(
			std::span<float> xValues,
			std::span<float> yValues,
			std::span<float> zValues) override;

		// sums rank-local solver errors, normalization happens in the solver
		Sim::PressureSolveErrorSums GlobalizePressureSolveErrorSums(
			const Sim::PressureSolveErrorSums& localSums) const override;
		uint64_t GetGlobalFluidParticleCount() const override { return N_globalF_; }

		Mem::MemUsage GetExchangeBufferMemoryUsage() const;
		Mem::MemUsage GetPlanMemoryUsage() const;

	private:

		// exchange plan and payload storage for one neighboring rank
		struct PeerPlan
		{
			int rank = -1;

			// local slots read from and written to, sorted by global particle ID
			std::vector<uint32_t> sendLocalIdxs;
			std::vector<uint32_t> recvLocalIdxs;

			// global particle IDs matching the local index lists
			std::vector<uint32_t> sendGlobalIdxs;
			std::vector<uint32_t> recvGlobalIdxs;

			// reusable payload storage
			std::vector<float> sendBuffer;
			std::vector<float> recvBuffer;
		};

		// packs, exchanges, and unpacks up to three fields in one round of messages
		void SynchronizeFluidFields(
			const std::array<std::span<float>, 3>& fields,
			uint32_t N_fields);

		// set when Rebuild() publishes a validated plan
		MpiContext* mpiContext_ = nullptr;
		Bench::BenchmarksTracker* benchmarksTracker_ = nullptr;

		// particle counts the current plan was built for
		uint64_t N_globalF_ = 0;
		uint32_t N_localF_ = 0;

		// peer exchange plans and reusable MPI requests
		std::vector<PeerPlan> peerPlans_;
		std::vector<MPI_Request> requests_;
	};

} // namespace Dist
