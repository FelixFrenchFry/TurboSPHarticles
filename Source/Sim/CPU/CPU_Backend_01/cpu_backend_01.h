// cpu_backend_01.h

#pragma once

// external
#include <cstdint>
#include <span>
#include <vector>
#include <tbb/partitioner.h>

// internal
#include "Source/Eval/Bench/benchmarks.h"
#include "Source/Sim/Core/backend_sim.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "neighbor_lists.h"



namespace Sim
{
	class SimBackend_CPU01 final  : public I_SimBackend, public I_DistributedParticleBackend
	{
	public: // ---------------------------------------------------------------------------------------

		SimBackend_CPU01(Sim::SimParamsMgr& simParamsMgr,
						 Bench::BenchmarksTracker& benchmarksTracker,
						 const SimParams& params);

		const char* Name() const override
			{ return "CPU01"; }

		void SetNeighborSearchMethod(std::unique_ptr<I_NeighborSearch> ns) override
			{ neighborSearch_ = std::move(ns); }

		// copies & translates a universal simulation state into -> backend-specific data structures
		void PullFromUniversalSimState(Sync::UniversalSimState& src, const SimParams& params) override;

		// copies & translates backend-specific data structures into -> universal simulation state
		void PushToUniversalSimState(Sync::UniversalSimState& dst) const override;

		// packs the current simulation state into a snapshot used for exporting or visualization
		// (implementation is backend-specific for better performance)
		void PackExportSnapshot(Sync::ExportSnapshot& snap, const SimParams& simParams) const override;
		void PackVisualSnapshot(Sync::VisualSnapshot& snap) const override;

		// fills in the scalar data based on given different color coding modes
		// (backend-specific such that possible transformations can be applied efficiently)
		void FillScalarData(Sync::VisualSnapshot& snap, Base::ColorCodingMode mode) const override;

		// computes & fills in the memory usage data for the current sim step
		// (backend-specific so that the memory usage can be computed from the custom data structures efficiently)
		void FillMemoryRecord(Mem::MemRecordStep& record, const SimParams& params) const override;

		// computes & fills in the stat data for the current sim step
		// (backend-specific so that the stats can be computed from the custom data structures efficiently)
		void FillStatisticsRecord(Stats::StatRecordStep& record, const SimParams& simParams) const override;

		// computes & fills in rank-local partial stat data for distributed execution
		bool FillStatisticsPartial(Stats::StatisticsPartial& partial, const SimParams& simParams) const override;

		// reports whether CPU01 supports distributed execution
		bool SupportsDistributedSimulation() const override { return true; }

		I_DistributedParticleBackend* GetDistributedParticleBackend() override { return this; }

		Sync::FluidConstView GetFluidParticleView() const override;
		Sync::BoundaryConstView GetBoundaryParticleView() const override;
		Sync::RigidConstView GetRigidParticleView() const override;

		bool ApplyParticleRedistribution(
			const std::vector<uint32_t>& retainedFluidOldLocalIdx_FromNewLocalIdx,
			const std::vector<uint32_t>& retainedBoundaryOldLocalIdx_FromNewLocalIdx,
			const Sync::FluidParticles& appendedFluidParticles,
			const Sync::BoundaryParticles& appendedBoundaryParticles,
			const SimParams& params) override;

		void CopyRigidBodyStateTo(Sync::RigidBodies& rigidBodies) const override;
		bool ApplyRigidBodyStateFrom(const Sync::RigidBodies& rigidBodies) override;

		// executes only the neighbor search (construction and query of the data NS structure)
		void ExecuteNeighborSearch(const SimParams& params, bool bench = false) override;

		bool TryGetLastParticleReOrder(
			std::span<const uint32_t>& oldFluidLocalIdx_FromNewLocalIdx,
			std::span<const uint32_t>& oldBoundaryLocalIdx_FromNewLocalIdx) const override;

		// configures rank-authoritative particle work for one distributed step
		bool ConfigureDistributedStep(
			const std::vector<uint32_t>& activeFluidParticleLocalIdxs,
			const std::vector<uint32_t>& activeRigidParticleLocalIdxs) override;

		// classifies local fluid particles against the export near-surface threshold
		bool FillFluidParticleNearSurfaceFlags(
			const SimParams& simParams,
			std::vector<uint8_t>& nearSurfaceFlags) const override;

		// selects local export surface particles using reconciled near-surface flags
		bool BuildExportFluidParticleLocalIdxs(
			const std::vector<uint8_t>& nearSurfaceFlags,
			std::vector<uint32_t>& localIdxs) const override;

		// reads body-level force and torque accumulated during the latest backend step
		bool GetRigidBodyForceTorque(RigidBodyForceTorque& dst) const override;

		// applies synchronized body force and torque through the rigid-body integrator
		bool ApplyRigidBodyForceTorque(const RigidBodyForceTorque& forceTorque, float timeStep) override;

		// advances the simulation by N_steps
		void ExecuteFullSimStep(const SimParams& params, uint32_t N_steps) override;

		// IISPH fluid solver step
		void Step_IISPH(const SimParams&) override;

		bool TerminationConditionSatisfied(const SimParams& params) const;

		// provides the benchmark operation display order
		void GetBenchOps(Bench::BenchOpList& out) const override { out = benchOps_; }

		// info
		uint32_t GetNumFluidParticles() const override { return N_F_; }
		uint32_t GetNumBoundaryParticles() const override { return N_B_; }
		uint32_t GetNumRigidParticles() const override { return N_R_; }

	private: // --------------------------------------------------------------------------------------

		// extra capacity for particle fields that grow during distributed redistribution
		static constexpr float particleFieldExtraAllocCapacityPct_ = 5.0f; // TODO: fine-tune

		// TODO: fine-tune chunk size by trying out 1024, 2048, 4096, 8192, 16384, 32768?
		static constexpr uint32_t particleCompactionBlockSize_ = 1024;
		static constexpr uint32_t particleAppendBlockSize_ = 4096;

		// computes rank-local statistics for all particles or active distributed particles,
		// so records can be finalized locally OR aggregated across ranks before finalization
		void ComputeStatisticsPartial(
			Stats::StatisticsPartial& partial,
			const SimParams& simParams,
			bool useActiveParticleLocalIdxs) const;

		void ApplyLastParticleReOrderToActiveFluidParticleLocalIdxs();

		void ClassifyExportSurfaceParticles(const std::vector<uint8_t>& nearSurfaceFlags) const;

		void UpdateRigidParticles();
		void ClearRigidForcesAndAddGravity(const SimParams& params);
		void ReduceRigidParticleForces();
		void IntegrateRigidBodies(float timeStep);

		// managers
		Sim::SimParamsMgr& simParamsMgr_;
		Bench::BenchmarksTracker& benchmarksTracker_;

		// neighbor search
		std::unique_ptr<I_NeighborSearch> neighborSearch_;
		std::unique_ptr<I_NeighborListSELL> neighborList_;
		friend class NS_BruteForce_CPU01;
		friend class NS_CellLinkedList_Compact_CPU01;

		// metadata
		uint64_t simStep_ = 0;
		double simTime_ = 0.0;
		uint32_t N_F_ = 0; // number of fluid particles
		uint32_t N_B_ = 0; // number of boundary particles
		uint32_t N_R_ = 0; // number of rigid particles
		uint32_t N_RB_ = 0; // number of rigid bodies
		uint32_t N_solverIters_ = 0;

		// persistent parallelization partitioner for TBB
		mutable std::unique_ptr<tbb::affinity_partitioner> affinityPart_;

		// --- [ backend-specific data structures for operating the simulation ] ---

		// fluid particles
		std::vector<uint32_t> idF_;
		std::vector<float> xPosF_, yPosF_, zPosF_;
		std::vector<float> xVelF_, yVelF_, zVelF_;
		std::vector<float> xVelPredF_, yVelPredF_, zVelPredF_; // TODO: unify with velF?
		std::vector<float> xAccPressF_, yAccPressF_, zAccPressF_;
		std::vector<float> sourceTermF_;
		std::vector<float> diagElemF_;
		std::vector<float> pressureF_;
		std::vector<float> densityF_;

		// boundary particles
		std::vector<uint32_t> idB_;
		std::vector<float> xPosB_, yPosB_, zPosB_;
		std::vector<float> volB_; // non-uniform volumes for non-uniform boundary particle spacing
		std::vector<float> psiB_; // non-uniform pseudo-masses for non-uniform boundary particle spacing

		// rigid particles
		std::vector<uint32_t> idR_;
		std::vector<uint32_t> bodyIdR_;
		std::vector<float> xPosR_, yPosR_, zPosR_;
		std::vector<float> xRelPosR_, yRelPosR_, zRelPosR_;
		std::vector<float> xVelR_, yVelR_, zVelR_;
		std::vector<float> xForceR_, yForceR_, zForceR_; // force accumulated before body reduction
		std::vector<float> volR_; // non-uniform volumes for non-uniform rigid particle spacing
		std::vector<float> psiR_; // non-uniform pseudo-masses for non-uniform rigid particle spacing

		// rigid bodies (using RB suffix to mark rigid body data structures)
		std::vector<uint32_t> idRB_;
		std::vector<uint32_t> rigidParticleBeginRB_; // first rigid particle
		std::vector<uint32_t> rigidParticleCountRB_; // number of rigid particles

		// body state (in world frame)
		std::vector<float> xCMRB_, yCMRB_, zCMRB_; // center of mass
		std::vector<float> xVelRB_, yVelRB_, zVelRB_; // linear velocity (time rate of change of center of mass)
		std::vector<float> xOmegaRB_, yOmegaRB_, zOmegaRB_; // angular velocity (time rate of change of orientation)
		std::vector<float> xAngMomRB_, yAngMomRB_, zAngMomRB_; // angular momentum (angular velocity * mass distribution)
		std::vector<float> xForceRB_, yForceRB_, zForceRB_; // force accumulated from rigid particles (time rate of change of linear momentum)
		std::vector<float> xTorqueRB_, yTorqueRB_, zTorqueRB_; // torque accumulated from rigid particles (time rate of change of angular momentum)

		// orientation matrix A, stored row-major
		std::vector<float> a00RB_, a01RB_, a02RB_;
		std::vector<float> a10RB_, a11RB_, a12RB_;
		std::vector<float> a20RB_, a21RB_, a22RB_;

		// mass
		std::vector<float> massRB_;
		std::vector<float> massInvRB_;

		// inverse inertia in body frame, stored row-major
		std::vector<float> iBodyInv00RB_, iBodyInv01RB_, iBodyInv02RB_;
		std::vector<float> iBodyInv10RB_, iBodyInv11RB_, iBodyInv12RB_;
		std::vector<float> iBodyInv20RB_, iBodyInv21RB_, iBodyInv22RB_;

		// optional distributed integration filtering
		std::vector<uint32_t> activeFluidParticleLocalIdxs_;
		std::vector<uint32_t> activeRigidParticleLocalIdxs_;
		std::vector<uint8_t> activeRigidParticleFlags_;
		bool useActiveFluidParticleLocalIdxs_ = false;
		bool useActiveRigidParticleLocalIdxs_ = false;
		bool integrateRigidBodies_ = true;
		bool applyRigidBodyExternalForces_ = true;

		// misc data
		float avgRelDensityErrorF_raw_;
		float avgRelDensityErrorF_pos_;
		float avgRelDensityErrorF_abs_;

		// re-usable work (scratch) buffers for temp storage
		// TODO: rework this?
		mutable std::vector<float> workBufferF_;
		mutable std::vector<float> workBufferB_;
		mutable std::vector<uint32_t> workBufferIdF_;
		mutable std::vector<uint32_t> workBufferIdB_;
		mutable std::vector<uint8_t> expSurfaceFlagsScratch_;
		mutable std::vector<uint8_t> expNearSurfaceFlagsScratch_;
		mutable Stats::StatisticsPartial statisticsPartialScratch_;

		// operation benchmarking order
		Bench::BenchOpList benchOps_ = Bench::MakeDefaultSimBenchOps();
	};

} // namespace Sim
