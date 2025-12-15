// cpu_backend_01.h
//
// TODO: comment

#pragma once

// external
#include <boost/container/small_vector.hpp>
#include <vector>

// internal
#include "Source/Eval/Bench/benchmarks.h"
#include "Source/Sim/Core/backend_sim.h"
#include "Source/Sim/Core/parameters_sim.h"



namespace Sim
{
	class SimBackend_CPU01 final : public I_SimBackend
	{
	public: // -----------------------------------------------------------------

		SimBackend_CPU01(Bench::BenchmarksMgr& benchmarksMgr);

		const char* Name() const override { return "CPU01"; }

		void SetNeighborSearchStrategy(std::unique_ptr<I_NeighborSearchStrat> strat) override;

		// copies & translates a universal simulation state into -> backend-specific data structures
		void PullFromUniversalSimState(const Sync::UniversalSimState& src) override;

		// copies & translates backend-specific data structures into -> universal simulation state
		void PushToUniversalSimState(Sync::UniversalSimState& dst) const override;

		// packs the current simulation state into a snapshot used for exporting or visualization
		// (implementation is backend-specific for better performance)
		void PackExportSnapshot(Sync::ExportSnapshot& snap, const SimParams& simParams) const override;
		void PackVisualSnapshot(Sync::VisualSnapshot& snap) const override;

		// fills in the scalar data based on given different color coding modes
		// (backend-specific such that possible transformations can be applied efficiently)
		void FillScalarData(Sync::VisualSnapshot& snap, Base::ColorCodingMode mode) const override;

		// computes & fills in the stat data for the current sim step
		// (backend-specific so that the stats can be computed from the custom data structures efficiently)
		void FillStatisticsRecord(Stats::StatRecordStep& record, const SimParams& simParams) const override;

		// advances the simulation by N_steps
		void ExecuteSimStep(const SimParams& params, uint32_t N_steps) override;

		// IISPH fluid solver step
		void Step_IISPH(const SimParams&) override;

		bool TerminationConditionSatisfied(uint32_t iterCount);

		// provides the list defining how computation operations are benchmarked in groups and/or individually
		// (backend-specific so that important cache-efficiency optimization can be applied)
		void GetBenchOpGroups(std::vector<Bench::BenchOpGroupDef>& out) const override { out = benchOpGroups_; }

	private: // ----------------------------------------------------------------

		// managers
		Bench::BenchmarksMgr& benchmarksMgr_;

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

		// TODO: manage this in the param manager
		uint32_t dataReOrderInterval_ = 50;

		// --- [ backend-specific data structures for operating the simulation ]

		// fluid particles
		std::vector<float> xPosF_, yPosF_, zPosF_;
		std::vector<float> xVelF_, yVelF_, zVelF_;
		std::vector<float> xVelPredF_, yVelPredF_, zVelPredF_;
		std::vector<float> xAccF_, yAccF_, zAccF_;
		std::vector<float> xAccPressF_, yAccPressF_, zAccPressF_;
		std::vector<float> xAccViscF_, yAccViscF_, zAccViscF_;
		std::vector<float> xAccExtF_, yAccExtF_, zAccExtF_;
		std::vector<float> sourceTermF_;
		std::vector<float> diagElemF_;
		std::vector<float> sysVecElemF_;
		std::vector<float> pressureF_;
		std::vector<float> densityF_;
		std::vector<uint32_t> idF_;

		// boundary particles
		std::vector<float> xPosB_, yPosB_, zPosB_;
		std::vector<float> xVelB_, yVelB_, zVelB_; // TODO: remove velocities because unused?
		std::vector<uint32_t> idB_;

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
