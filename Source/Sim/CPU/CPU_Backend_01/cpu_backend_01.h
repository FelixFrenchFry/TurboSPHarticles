// cpu_backend_01.h
//
// TODO: comment

#pragma once

// external
#include <boost/container/small_vector.hpp>
#include <vector>
#include <tbb/partitioner.h>

// internal
#include "Source/Eval/Bench/benchmarks.h"
#include "Source/Sim/Core/backend_sim.h"
#include "Source/Sim/Core/parameters_sim.h"



namespace Sim
{
	class SimBackend_CPU01 final : public I_SimBackend
	{
	public: // -----------------------------------------------------------------

		SimBackend_CPU01(Sim::SimParamsMgr& simParamsMgr,
						 Bench::BenchmarksTracker& benchmarksTracker,
						 const SimParams& params);

		const char* Name() const override { return "CPU01"; }

		void SetNeighborSearchStrategy(std::unique_ptr<I_NeighborSearchStrat> strat) override;

		// copies & translates a universal simulation state into -> backend-specific data structures
		void PullFromUniversalSimState(Sync::UniversalSimState& src) override;

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

		// advances the simulation by N_steps
		void ExecuteFullSimStep(const SimParams& params, uint32_t N_steps) override;

		// IISPH fluid solver step
		void Step_IISPH(const SimParams&) override;

		bool TerminationConditionSatisfied(const SimParams& params) const;

		// provides the list defining how computation operations are benchmarked in groups and/or individually
		// (backend-specific so that important cache-efficiency optimization can be applied)
		void GetBenchOpGroups(std::vector<Bench::BenchOpGroupDef>& out) const override { out = benchOpGroups_; }

		// info
		uint32_t GetNumFluidParticles() const override { return N_F_; }
		uint32_t GetNumBoundaryParticles() const override { return N_B_; }

	private: // ----------------------------------------------------------------

		// managers
		Sim::SimParamsMgr& simParamsMgr_;
		Bench::BenchmarksTracker& benchmarksTracker_;

		// neighbor search
		std::unique_ptr<I_NeighborSearchStrat> neighborSearchStrat_;
		friend class NS_BruteForce_CPU01;
		friend class NS_SpatialHashing_Basic_CPU01;
		friend class NS_CellLinkedList_Basic_CPU01;
		friend class NS_CellLinkedList_Compact_CPU01;

		// metadata
		uint32_t simStep_ = 0;
		double simTime_ = 0.0;
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// persistent parallelization partitioner for TBB
		mutable std::unique_ptr<tbb::affinity_partitioner> affinityPart_;

		// --- [ backend-specific data structures for operating the simulation ]

		// fluid particles
		std::vector<float> xPosF_, yPosF_, zPosF_;
		std::vector<float> xVelF_, yVelF_, zVelF_;
		std::vector<float> xVelPredF_, yVelPredF_, zVelPredF_;
		std::vector<float> xAccPressF_, yAccPressF_, zAccPressF_;
		std::vector<float> xAccNonPressF_, yAccNonPressF_, zAccNonPressF_;
		std::vector<float> sourceTermF_;
		std::vector<float> diagElemF_;
		std::vector<float> pressureF_;
		std::vector<float> densityF_;
		std::vector<uint32_t> idF_; // TODO: unused?

		// boundary particles
		// TODO: add boundary particle velocities?
		std::vector<float> xPosB_, yPosB_, zPosB_;
		std::vector<uint32_t> idB_; // TODO: unused?

		// misc data
		uint32_t N_solverIterCount_;
		float avgRelDensityErrorF_raw_;
		float avgRelDensityErrorF_pos_;
		float avgRelDensityErrorF_abs_;

		// neighbor storage
		std::unique_ptr<I_NeighborLists> neighborLists_;

		// re-usable work (scratch) buffer for temp storage
		mutable std::vector<float> workBuffer_;

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
				.ops = { Bench::BenchOp::VelocityUpdate,
						   Bench::BenchOp::PositionUpdate }
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
