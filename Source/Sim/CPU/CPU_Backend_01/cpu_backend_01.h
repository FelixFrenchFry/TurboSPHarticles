// cpu_backend_01.h

#pragma once

// external
#include <vector>
#include <tbb/partitioner.h>

// internal
#include "Source/Eval/Bench/benchmarks.h"
#include "Source/Sim/Core/backend_sim.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "neighbor_lists.h"



namespace Sim
{
	class SimBackend_CPU01 final : public I_SimBackend
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

		// executes only the neighbor search (construction and query of the data NS structure)
		void ExecuteNeighborSearch(const SimParams& params, bool bench = false) override;

		bool TryGetLastParticleReOrder(
			std::vector<uint32_t>& oldFluidLocalIdx_FromNewLocalIdx,
			std::vector<uint32_t>& oldBoundaryLocalIdx_FromNewLocalIdx) const override;

		bool SetActiveFluidParticleLocalIdxs(const std::vector<int>& localIdxs) override;
		void ClearActiveFluidParticleLocalIdxs() override;

		// advances the simulation by N_steps
		void ExecuteFullSimStep(const SimParams& params, uint32_t N_steps) override;

		// IISPH fluid solver step
		void Step_IISPH(const SimParams&) override;

		bool TerminationConditionSatisfied(const SimParams& params) const;

		// provides the list defining how computation operations are benchmarked in groups and/or individually
		// (backend-specific so that important cache-efficiency optimization can be applied)
		void GetBenchOpGroups(std::vector<Bench::BenchOpGroupDef>& out) const override
			{ out = benchOpGroups_; }

		// info
		uint32_t GetNumFluidParticles() const override { return N_F_; }
		uint32_t GetNumBoundaryParticles() const override { return N_B_; }

	private: // --------------------------------------------------------------------------------------

		void ApplyLastParticleReOrderToActiveFluidParticleLocalIdxs();

		void UpdateRigidBoundaryParticles();
		void ClearRigidForcesAndAddGravity(const SimParams& params);
		void ClearRigidBoundaryParticleForces();
		void ReduceRigidBoundaryParticleForces();
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
		uint32_t simStep_ = 0;
		double simTime_ = 0.0;
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// persistent parallelization partitioner for TBB
		mutable std::unique_ptr<tbb::affinity_partitioner> affinityPart_;

		// --- [ backend-specific data structures for operating the simulation ] -------------------------

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
		std::vector<uint32_t> bodyIdB_;
		std::vector<uint32_t> rigidBoundaryParticleIdxB_;
		std::vector<float> xPosB_, yPosB_, zPosB_;
		std::vector<float> xVelB_, yVelB_, zVelB_;
		std::vector<float> volB_; // non-uniform volumes for non-uniform boundary particle spacing
		std::vector<float> psiB_; // non-uniform pseudo-masses for non-uniform boundary particle spacing

		// rigid bodies
		uint32_t N_rigidBodies_ = 0;
		uint32_t N_rigidBoundaryParticles_ = 0;
		std::vector<uint32_t> idR_; // per-rigid-body ID
		std::vector<uint32_t> rigidBoundaryParticleBeginR_; // first rigid boundary particle
		std::vector<uint32_t> rigidBoundaryParticleCountR_; // number of rigid boundary particles

		// body state (in world frame)
		std::vector<float> xCMR_, yCMR_, zCMR_; // center of mass
		std::vector<float> xVelR_, yVelR_, zVelR_; // linear velocity
		std::vector<float> xOmegaR_, yOmegaR_, zOmegaR_; // angular velocity
		std::vector<float> xAngMomR_, yAngMomR_, zAngMomR_; // angular momentum
		std::vector<float> xForceR_, yForceR_, zForceR_; // force accumulated from rigid boundary particles
		std::vector<float> xTorqueR_, yTorqueR_, zTorqueR_; // torque accumulated from rigid boundary particles

		// force accumulated per rigid boundary particle before body reduction
		std::vector<float> xForceRigidBoundaryParticle_;
		std::vector<float> yForceRigidBoundaryParticle_;
		std::vector<float> zForceRigidBoundaryParticle_;

		// orientation matrix A, stored row-major
		std::vector<float> a00R_, a01R_, a02R_;
		std::vector<float> a10R_, a11R_, a12R_;
		std::vector<float> a20R_, a21R_, a22R_;

		// mass
		std::vector<float> massR_;
		std::vector<float> massInvR_;

		// inverse inertia in body frame, stored row-major
		std::vector<float> iBodyInv00R_, iBodyInv01R_, iBodyInv02R_;
		std::vector<float> iBodyInv10R_, iBodyInv11R_, iBodyInv12R_;
		std::vector<float> iBodyInv20R_, iBodyInv21R_, iBodyInv22R_;

		// local rigid boundary particle offsets from center of mass
		std::vector<float> xLocalRigidBoundaryParticle_;
		std::vector<float> yLocalRigidBoundaryParticle_;
		std::vector<float> zLocalRigidBoundaryParticle_;

		// optional distributed integration filtering
		bool useActiveFluidParticleLocalIdxs_ = false;
		std::vector<uint32_t> activeFluidParticleLocalIdxs_;

		// misc data
		uint32_t N_solverIterCount_;
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

		// operation benchmarking groups
		std::vector<Bench::BenchOpGroupDef> benchOpGroups_ =
		{
			{
				.ops = { Bench::BenchOp::Data_Ordering }
			},
			{
				.ops = { Bench::BenchOp::Neighbors_Construction }
			},
			{
				.ops = { Bench::BenchOp::Neighbors_Query }
			},
			{
				.ops = { Bench::BenchOp::Density_Solving }
			},
			{
				.ops = { Bench::BenchOp::Viscosity_Acceleration,
						   Bench::BenchOp::External_Acceleration }
			},
			{
				.ops = { Bench::BenchOp::Pressure_SourceTerm,
						   Bench::BenchOp::Pressure_DiagonalElem }
			},
			{
				.ops = { Bench::BenchOp::Pressure_JacobiSolver }
			},
			{
				.ops = { Bench::BenchOp::Pressure_Acceleration }
			},
			{
				.ops = { Bench::BenchOp::Rigid_ForceAccumulation }
			},
			{
				.ops = { Bench::BenchOp::VelocityUpdate,
							Bench::BenchOp::PositionUpdate }
			},
			{
				.ops = { Bench::BenchOp::Rigid_Update }
			},
			{
				.ops = { Bench::BenchOp::Statistics_Particles,
						   Bench::BenchOp::Statistics_Density,
						   Bench::BenchOp::Statistics_Velocity }
			},
			{
				.ops = { Bench::BenchOp::Snapshot_Export }
			},
			{
				.ops = { Bench::BenchOp::Snapshot_Visual }
			},
			{
				.ops = { Bench::BenchOp::Serialization }
			},
			{
				.ops = { Bench::BenchOp::Total_Sim }
			}
		};
	};

} // namespace Sim
