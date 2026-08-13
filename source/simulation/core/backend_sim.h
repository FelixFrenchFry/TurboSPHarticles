// backend_sim.h

#pragma once

// external
#include <cstdint>
#include <memory>
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
	// forward declarations
	class I_NeighborSearch;
	struct NeighborSpan;

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

	// optional owner-to-halo synchronization used by distributed fluid solvers
	class I_DistributedFluidSolverSync
	{
	public:

		virtual ~I_DistributedFluidSolverSync() = default;

		// an empty permutation means that particle storage was not re-ordered
		// (inverted by the caller, which needs it for its own index remapping anyway)
		virtual void ApplyFluidParticleReOrder(
			std::span<const uint32_t> newFluidLocalIdx_FromOldLocalIdx) = 0;

		// synchronizes owner-authoritative fluid values over the selected halo extent
		virtual void SynchronizeFluidScalarField(
			std::span<float> values,
			Base::DistHaloMode haloExtent) = 0;
		virtual void SynchronizeFluidVectorField(
			std::span<float> xValues,
			std::span<float> yValues,
			std::span<float> zValues,
			Base::DistHaloMode haloExtent) = 0;

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

	// optional backend capability for particle exchange between ranks for distributed execution
	class I_DistributedParticleBackend
	{
	public:

		virtual ~I_DistributedParticleBackend() = default;

		// non-owning views remain valid until backend particle storage is replaced or re-ordered
		virtual Sync::FluidConstView GetFluidParticleView() const = 0;
		virtual Sync::BoundaryConstView GetBoundaryParticleView() const = 0;
		virtual Sync::RigidConstView GetRigidParticleView() const = 0;

		// compacts retained particles in place and appends migration + halo particles
		// (the appended fluid particles carry only id, positions, and velocities, because the
		// following solver step re-writes the remaining fluid fields before reading them)
		virtual bool ApplyParticleRedistribution(
			const std::vector<uint32_t>& retainedFluidOldLocalIdx_FromNewLocalIdx,
			const std::vector<uint32_t>& retainedBoundaryOldLocalIdx_FromNewLocalIdx,
			const Sync::FluidParticles& appendedFluidParticles,
			const Sync::BoundaryParticles& appendedBoundaryParticles,
			bool redistributeBoundaryParticles,
			const SimParams& params) = 0;

		// copies/applies only dynamic rigid-body state used by distributed synchronization
		virtual void CopyRigidBodyStateTo(Sync::RigidBodies& rigidBodies) const = 0;
		virtual bool ApplyRigidBodyStateFrom(const Sync::RigidBodies& rigidBodies) = 0;
	};

	// -----------------------------------------------------------------------------------------------
	// --- [ simulation backend interface ] ----------------------------------------------------------
	// -----------------------------------------------------------------------------------------------

	class I_SimBackend
	{
	public: // ---------------------------------------------------------------------------------------

		virtual ~I_SimBackend() = default;

		virtual const char* Name() const = 0;

		virtual void SetNeighborSearchMethod(std::unique_ptr<I_NeighborSearch> ns) = 0;

		// reports the simulation domain, so spatial acceleration structures can stay bounded by it
		// instead of by particles that left it (an empty extent leaves the backend unbounded)
		virtual void SetSimDomainBounds(const glm::vec3& min, const glm::vec3& max) {}

		// copies & translates a universal simulation state into -> backend-specific data structures
		virtual void PullFromUniversalSimState(Sync::UniversalSimState& src, const SimParams& params) = 0;

		// materializes backend-specific data structures into -> universal simulation state
		// (implementations may transfer storage when the backend will be retired afterwards)
		virtual void PushToUniversalSimState(Sync::UniversalSimState& dst, bool backendWillBeRetired) = 0;

		// packs the current simulation state into a snapshot used for exporting or visualization
		// (implementation is backend-specific for better performance)
		virtual void PackExportSnapshot(Sync::ExportSnapshot& snap, const SimParams& params) const = 0;
		virtual void PackVisualSnapshot(Sync::VisualSnapshot& snap) const = 0;

		// additional CUDA-OpenGL interop for visualization (if the backend supports it)
	#ifdef TURBOSPHARTICLES_WITH_CUDA
		virtual Sync::VisualPackResultCUDA TryPackVisualSnapshotCUDA(
			Sync::VisualFrameMailboxCUDA& visualFrameMailboxCUDA,
			std::shared_ptr<Sync::VisualSnapshot> auxiliarySnap) const
		{
			(void)visualFrameMailboxCUDA;
			(void)auxiliarySnap;
			return Sync::VisualPackResultCUDA::Unsupported;
		}
	#endif

		// computes & fills in the scalar data based on different color coding modes
		// (backend-specific so that possible transformations can be applied efficiently)
		virtual void FillScalarData(Sync::VisualSnapshot& snap, Base::ColorCodingMode mode) const = 0;

		// computes & fills in the memory usage data for the current sim step
		// (backend-specific so that the memory usage can be computed from the custom data structures efficiently)
		virtual void FillMemoryRecord(Mem::MemRecordStep& record, const SimParams& params) const = 0;

		// computes & fills in the stat data for the current sim step
		// (backend-specific so that the stats can be computed from the custom data structures efficiently)
		virtual void FillStatisticsRecord(Stats::StatRecordStep& record, const SimParams& params) const = 0;

		// computes & fills in the partial stat data for the current sim step
		virtual bool FillStatisticsPartial(Stats::StatisticsPartial&, const SimParams&) const { return false; }

		// reports whether the backend implements all components required for distributed execution
		virtual bool SupportsDistributedSimulation() const { return false; }

		// exposes optional backend-resident particle storage used for particle exchange in distributed execution
		virtual I_DistributedParticleBackend* GetDistributedParticleBackend() { return nullptr; }

		// executes only the neighbor search (construction and query of the data NS structure)
		virtual void ExecuteNeighborSearch(const SimParams& params, bool bench = false) = 0;

		// provides views onto the latest local re-order permutation, if neighbor search exposes it
		// (non-owning, valid until the next re-order or construction)
		virtual bool TryGetLastParticleReOrder(
			std::span<const uint32_t>& oldFluidLocalIdx_FromNewLocalIdx,
			std::span<const uint32_t>& oldBoundaryLocalIdx_FromNewLocalIdx) const
		{
			(void)oldFluidLocalIdx_FromNewLocalIdx;
			(void)oldBoundaryLocalIdx_FromNewLocalIdx;
			return false;
		}

		// configures rank-authoritative particle work for distributed execution
		virtual bool ConfigureDistributedStep(
			const std::vector<uint32_t>&,
			const std::vector<uint32_t>&,
			const std::vector<uint32_t>&,
			I_DistributedFluidSolverSync&)
		{
			return false;
		}

		// classifies fluid particles against the regular export near-surface threshold
		virtual bool FillFluidParticleNearSurfaceFlags(
			const SimParams&,
			std::vector<uint8_t>&) const
		{
			return false;
		}

		// selects regular export surface particles using authoritative near-surface flags
		virtual bool BuildExportFluidParticleLocalIdxs(
			const std::vector<uint8_t>&,
			std::vector<uint32_t>&) const
		{
			return false;
		}

		// reads body-level force and torque accumulated during the latest backend step
		virtual bool GetRigidBodyForceTorque(RigidBodyForceTorque&) const { return false; }

		// applies externally synchronized body force and torque through integrator of the backend
		virtual bool ApplyRigidBodyForceTorque(
			const RigidBodyForceTorque& forceTorque,
			float timeStep)
		{
			(void)forceTorque;
			(void)timeStep;
			return false;
		}

		// advances the simulation by N_steps
		virtual void ExecuteFullSimStep(const SimParams& params, uint32_t N_steps = 1) = 0;

		// IISPH fluid solver step
		virtual void Step_IISPH(const SimParams& params) = 0;

		// provides the list defining which operations are shown in benchmark summaries
		virtual void GetBenchOps(Bench::BenchOpList& out) const = 0;

		// info
		virtual uint32_t GetNumFluidParticles() const = 0;
		virtual uint32_t GetNumBoundaryParticles() const = 0;
		virtual uint32_t GetNumRigidParticles() const = 0;
	};

	// backend factory
	std::unique_ptr<I_SimBackend> MakeSimBackend(Base::BackendVersion version,
												 Sim::SimParamsMgr& simParamsMgr,
												 Bench::BenchmarksTracker& benchmarksTracker,
												 const SimParams& params);

	// -----------------------------------------------------------------------------------------------
	// --- [ neighbor search and neighbor list interfaces ] ------------------------------------------
	// -----------------------------------------------------------------------------------------------

	class I_NeighborSearch
	{
	public: // ---------------------------------------------------------------------------------------

		virtual ~I_NeighborSearch() = default;

		virtual const char* Name() const = 0;

		virtual void Init(const SimParams& params) = 0;

		virtual void Construct(const SimParams& params) = 0;

		virtual void Query_Count(const SimParams& params) = 0;

		virtual void Query_Fill(const SimParams& params) = 0;

		virtual void DataReOrder(const SimParams& params) = 0;

		// re-orders particle fields ahead of a full solver step, which re-initializes some of them,
		// so permutations of those fields are skipped (TODO: update if solver step behavior changes)
		// (currently omitted are velPred, accPress, sourceTerm, diagElem, pressure, and density)
		virtual void DataReOrder_SolverStep(const SimParams& params) { DataReOrder(params); }

		// provides views onto the latest local re-order permutation, if neighbor search exposes it
		// (non-owning, valid until the next re-order or construction)
		virtual bool TryGetLastParticleReOrder(
			std::span<const uint32_t>& oldFluidLocalIdx_FromNewLocalIdx,
			std::span<const uint32_t>& oldBoundaryLocalIdx_FromNewLocalIdx) const
		{
			(void)oldFluidLocalIdx_FromNewLocalIdx;
			(void)oldBoundaryLocalIdx_FromNewLocalIdx;
			return false;
		}

		virtual Mem::ParticleClassMemUsage GetMemUsage(const SimParams& params) const = 0;
	};

	struct NeighborSpan
	{
		const uint32_t* data = nullptr;
		uint32_t N = 0;
	};

	class I_NeighborList
	{
	public: // ---------------------------------------------------------------------------------------

		virtual ~I_NeighborList() = default;

		virtual void Resize(uint32_t N_F, uint32_t N_B, uint32_t N_R, const SimParams& params) = 0;

		// applies optional post-processing on the data structures after the count-query
		// (e.g. construction of the CSR neighbor storage)
		virtual void Finalize_Count() = 0;

		// applies optional post-processing on the data structures after the fill-query
		// (e.g. concatenation of individual neighbor lists into CSR neighbor storage)
		virtual void Finalize_Fill() = 0;

		// sets the neighbor count of fluid particle i, computed by the count-query
		virtual void SetNbrCountFF(uint32_t i, uint32_t N_nbrs) = 0;
		virtual void SetNbrCountFB(uint32_t i, uint32_t N_nbrs) = 0;
		virtual void SetNbrCountFR(uint32_t i, uint32_t N_nbrs) = 0;
		virtual void SetNbrCountRF(uint32_t i, uint32_t N_nbrs) = 0;
		virtual void SetNbrCountRB(uint32_t i, uint32_t N_nbrs) = 0;
		virtual void SetNbrCountRR(uint32_t i, uint32_t N_nbrs) = 0;

		// sets the full neighbor list js of source particle i, accumulated in a scratch array by the fill-query
		// (optionally applying compression and other post-processing if the implementation uses it)
		virtual void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t N_nbrs) = 0;
		virtual void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t N_nbrs) = 0;
		virtual void SetNbrListFR(uint32_t i, const uint32_t* js, uint32_t N_nbrs) = 0;
		virtual void SetNbrListRF(uint32_t i, const uint32_t* js, uint32_t N_nbrs) = 0;
		virtual void SetNbrListRB(uint32_t i, const uint32_t* js, uint32_t N_nbrs) = 0;
		virtual void SetNbrListRR(uint32_t i, const uint32_t* js, uint32_t N_nbrs) = 0;

		// gets a "span" of the list of neighbors for a source particle i
		// (for compressed neighbor lists, the span points to per-thread scratch space, which
		// gets overwritten on each call, so the data must be processed before that)
		virtual NeighborSpan FF(uint32_t i) const = 0;
		virtual NeighborSpan FB(uint32_t i) const = 0;
		virtual NeighborSpan FR(uint32_t i) const = 0;
		virtual NeighborSpan RF(uint32_t i) const = 0;
		virtual NeighborSpan RB(uint32_t i) const = 0;
		virtual NeighborSpan RR(uint32_t i) const = 0;

		// default implementations for neighbor counts, overwritten for compressed neighbor lists
		virtual uint32_t CountFF(uint32_t i) const { return FF(i).N; }
		virtual uint32_t CountFB(uint32_t i) const { return FB(i).N; }
		virtual uint32_t CountFR(uint32_t i) const { return FR(i).N; }
		virtual uint32_t CountRF(uint32_t i) const { return RF(i).N; }
		virtual uint32_t CountRB(uint32_t i) const { return RB(i).N; }
		virtual uint32_t CountRR(uint32_t i) const { return RR(i).N; }

		// stats
		virtual Mem::ParticleClassMemUsage GetMemUsage(const SimParams& params) const = 0;

		// flags
		virtual bool RunCountQuery() const = 0;
		virtual bool RunFillQuery() const = 0;
	};

} // namespace Sim
