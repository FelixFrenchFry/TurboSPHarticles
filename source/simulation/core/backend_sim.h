// backend_sim.h
#pragma once

// external
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

// internal
#include "source/base/enum_hub.h"
#include "source/eval/bench/benchmarks.h"
#include "source/eval/memory/memory.h"
#include "source/eval/stats/statistics.h"
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/sync/snapshots.h"
#include "source/simulation/sync/state_sim.h"



#ifdef TURBOSPHARTICLES_WITH_CUDA
namespace Sync
{
	class VisualFrameMailboxCUDA;

	enum class VisualPackResultCUDA : uint8_t
	{
		Unsupported,
		Skipped,
		Published,
	};

} // namespace Sync
#endif

namespace Sim
{
	// body-level force and torque accumulated during one backend step used for distributed execution
	// (accumulated by each rank, summed up and integrated at root, then broadcast back to each rank)
	struct RigidBodyForceTorque
	{
		std::vector<float> xForce;
		std::vector<float> yForce;
		std::vector<float> zForce;
		std::vector<float> xTorque;
		std::vector<float> yTorque;
		std::vector<float> zTorque;

		void Resize(const uint32_t N)
		{
			xForce.resize(N);
			yForce.resize(N);
			zForce.resize(N);
			xTorque.resize(N);
			yTorque.resize(N);
			zTorque.resize(N);
		}
	};

	struct PressureSolveErrorSums
	{
		double raw = 0.0;
		double pos = 0.0;
		double abs = 0.0;
	};

	enum class FluidSolverSyncPhase : uint8_t
	{
		Density,
		PredictedVelocity,
		InitialPressure,
		PressureAcceleration,
		Pressure,
		PressureErrorReduction,
		ExportSurfaceFlags,
	};

	// optional owner-to-halo synchronization used by distributed fluid solvers
	class I_DistributedFluidSolverSync
	{
	public:

		virtual ~I_DistributedFluidSolverSync() = default;

		// hands the send and receive local index lists of the latest plan over by swapping storage
		// (peer-major in the local layout the plan was built on, the receiver keeps them valid across re-orders)
		virtual void SwapFluidSyncLocalIdxs(
			Base::DistHaloMode haloExtent,
			std::vector<uint32_t>& sendLocalIdxs,
			std::vector<uint32_t>& recvLocalIdxs) = 0;

		// split owner-to-halo exchange of per-particle payloads packed in the order of the index lists
		// (payloads stay untouched until the exchange ends, and may be device memory under CUDA-aware MPI)
		virtual void BeginFluidHaloExchange(
			std::span<const std::byte> sendPayload,
			std::span<std::byte> recvPayload,
			Base::DistHaloMode haloExtent,
			FluidSolverSyncPhase phase) = 0;
		virtual void EndFluidHaloExchange() = 0;

		virtual PressureSolveErrorSums GlobalizePressureSolveErrorSums(
			const PressureSolveErrorSums& localSums) const = 0;
		virtual uint64_t GetGlobalFluidParticleCount() const = 0;
	};

	// shared export surface criterion, so every backend classifies fluid particles identically
	inline bool IsNearSurfaceFluidParticle(
		const uint32_t nbrCountFF,
		const uint32_t nbrCountFB,
		const uint32_t nbrCountFR,
		const uint32_t nbrCountThreshold) noexcept
	{
		// surface particle IF touching boundary OR rigid body OR low fluid neighbor count
		return nbrCountFB > 0 || nbrCountFR > 0 || nbrCountFF < nbrCountThreshold;
	}

	// optional backend capability for distributed execution (particle exchange between ranks and rank-local work)
	class I_DistributedParticleBackend
	{
	public:

		virtual ~I_DistributedParticleBackend() = default;

		// non-owning views remain valid until backend particle storage is replaced or re-ordered
		virtual Sync::FluidConstView GetFluidParticleView() const = 0;
		virtual Sync::BoundaryConstView GetBoundaryParticleView() const = 0;
		virtual Sync::RigidConstView GetRigidParticleView() const = 0;
		virtual Sync::ParticlePositionConstView GetRigidParticleWorldPositionView() const = 0;

		// compacts retained particles in place through the compaction moves and appends migration + halo particles
		// (appended fluid particles carry only id, positions, and velocities)
		virtual bool ApplyParticleRedistribution(
			uint32_t N_retainedF,
			uint32_t N_retainedB,
			uint32_t N_retainedR,
			std::span<const Sync::ParticleCompactionMove> compactionMovesF,
			std::span<const Sync::ParticleCompactionMove> compactionMovesB,
			std::span<const Sync::ParticleCompactionMove> compactionMovesR,
			const Sync::FluidParticles& appendedParticlesF,
			const Sync::BoundaryParticles& appendedParticlesB,
			const Sync::RigidParticles& appendedParticlesR,
			uint32_t N_ownedF,
			uint32_t N_ownedB,
			uint32_t N_ownedR,
			bool redistributeParticlesB,
			const SimParams& params) = 0;

		// copies/applies only dynamic rigid-body state used by distributed synchronization
		virtual void CopyRigidBodyStateTo(Sync::RigidBodies& rigidBodies) const = 0;
		virtual bool ApplyRigidBodyStateFrom(const Sync::RigidBodies& rigidBodies) = 0;

		// configures the compact [ owned ][ halo ] layout of pulled particles before any distributed re-order
		virtual bool ConfigureDistributedParticleLayout(
			uint32_t N_ownedF,
			uint32_t N_ownedB,
			uint32_t N_ownedR) = 0;

		// takes over the send and receive local index lists of a freshly rebuilt fluid solver sync plan
		virtual void TakeFluidSolverSyncLocalIdxs(
			I_DistributedFluidSolverSync& distributedFluidSolverSync) = 0;

		// configures rank-authoritative particle work for distributed execution
		// (both index lists must be strictly ascending in the current local layout)
		// (the owned fluid prefix always computes pressure acceleration, the halo list adds to it)
		virtual bool ConfigureDistributedStep(
			const std::vector<uint32_t>& outerLocalIdxsF,
			const std::vector<uint32_t>& pressureAccHaloLocalIdxsF,
			I_DistributedFluidSolverSync& distributedFluidSolverSync) = 0;

		// selects regular export surface particles, with near-surface flags of halos taken from their owners
		// (collective with the neighboring ranks, failures before or during the exchange throw)
		virtual bool BuildExportFluidParticleLocalIdxs(
			const SimParams& simParams,
			I_DistributedFluidSolverSync& distributedFluidSolverSync,
			std::vector<uint32_t>& localIdxs) const = 0;

		// reads body-level force and torque accumulated during the latest backend step
		virtual bool GetRigidBodyForceTorque(RigidBodyForceTorque& dst) const = 0;

		// applies externally synchronized body force and torque through integrator of the backend
		virtual bool ApplyRigidBodyForceTorque(
			const RigidBodyForceTorque& forceTorque,
			float timeStep) = 0;

		// computes & fills in the partial stat data for the current sim step
		virtual bool FillStatisticsPartial(Stats::StatisticsPartial& partial, const SimParams& simParams) const = 0;
	};

	// -----------------------------------------------------------------------------------------------
	// --- [ simulation backend interface ] ----------------------------------------------------------
	// -----------------------------------------------------------------------------------------------

	class I_SimBackend
	{
	public:

		virtual ~I_SimBackend() = default;

		virtual const char* Name() const = 0;

		// reports the simulation domain, so spatial acceleration structures can stay bounded by it
		// instead of by particles that left it (an empty extent leaves the backend unbounded)
		virtual void SetSimDomainBounds(const glm::vec3& min, const glm::vec3& max) = 0;

		// transfers a validated universal simulation state into -> backend-specific data structures
		virtual void PullFromUniversalSimState(Sync::UniversalSimState& src, const SimParams& params) = 0;

		// materializes backend-specific data structures into -> universal simulation state
		// (implementations may transfer storage when the backend will be retired afterwards)
		virtual void PushToUniversalSimState(Sync::UniversalSimState& dst, bool backendWillBeRetired) = 0;

		// packs the current simulation state into a snapshot used for exporting or visualization
		// (implementation is backend-specific for better performance)
		virtual void PackExportSnapshot(Sync::ExportSnapshot& snap, const SimParams& params,
			bool includeB, std::optional<uint64_t> meshIdBeginB) const = 0;
		virtual void PackVisualSnapshot(Sync::VisualSnapshot& snap, bool includeF, bool includeR) const = 0;

		// additional CUDA-OpenGL interop for visualization (if the backend supports it)
	#ifdef TURBOSPHARTICLES_WITH_CUDA
		virtual Sync::VisualPackResultCUDA TryPackVisualSnapshotCUDA(
			Sync::VisualFrameMailboxCUDA& visualFrameMailboxCUDA,
			std::shared_ptr<Sync::VisualSnapshot> auxiliarySnap, bool includeF, bool includeR) const
		{
			(void)visualFrameMailboxCUDA;
			(void)auxiliarySnap;
			(void)includeF;
			(void)includeR;
			return Sync::VisualPackResultCUDA::Unsupported;
		}
	#endif

		// computes & fills in the scalar data based on different color coding modes
		// (backend-specific so that possible transformations can be applied efficiently)
		virtual void FillScalarData(Sync::VisualSnapshot& snap, Base::ColorCodingMode mode) const = 0;

		// computes & fills in the memory usage data for the current sim step
		// (backend-specific so that the memory usage can be computed from the custom data structures efficiently)
		virtual void FillMemoryRecord(Mem::MemRecordStep& record) const = 0;

		// computes & fills in the stat data for the current sim step
		// (backend-specific so that the stats can be computed from the custom data structures efficiently)
		virtual void FillStatisticsRecord(Stats::StatRecordStep& record, const SimParams& params) const = 0;

		// exposes the distributed execution capability of the backend (null if unsupported)
		virtual I_DistributedParticleBackend* GetDistributedParticleBackend() { return nullptr; }

		// executes only the neighbor search (construction and query of the data NS structure)
		virtual void ExecuteNeighborSearch(const SimParams& params, bool bench = false) = 0;

		// advances the simulation by N_steps
		virtual void ExecuteFullSimStep(const SimParams& params, uint32_t N_steps = 1) = 0;

		// provides the list defining which operations are shown in benchmark summaries
		virtual void GetBenchOps(Bench::BenchOpList& out) const = 0;

		// info
		virtual uint32_t GetNumFluidParticles() const = 0;
		virtual uint32_t GetNumBoundaryParticles() const = 0;
		virtual uint32_t GetNumRigidParticles() const = 0;
	};

	// backend factory
	std::unique_ptr<I_SimBackend> MakeSimBackend(Base::BackendType backendType,
												 Bench::BenchmarksTracker& benchmarksTracker);

} // namespace Sim
