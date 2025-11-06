// cpu_backend_01.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>
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
		void PackExportSnapshot(Sync::ExportSnapshot& snap) const override;
		void PackVisualSnapshot(Sync::VisualSnapshot& snap) const override;

		// fills in the scalar data based on given different color coding modes
		// (backend-specific such that possible transformations can be applied efficiently)
		void FillScalarData(Sync::VisualSnapshot& snap, Base::ColorCodingMode mode) const override;

		// computes & fills in the stat data for the current sim step
		// (backend-specific so that the stats can be computed from the custom data structures efficiently)
		void FillStatisticsRecord(Stats::StatRecordStep& record, const SimParams& simParams) const override;

		// advances the simulation by N_steps
		void ExecuteSimStep(SimParams params, int64_t N_steps) override;

		// IISPH fluid solver step
		void Step_IISPH(const SimParams&) override;

		// provides the list defining how computation operations are benchmarked in groups and/or individually
		// (backend-specific so that important cache-efficiency optimization can be applied)
		void GetBenchOpGroups(std::vector<Bench::BenchOpGroupDef>& out) const override { out = benchOpGroups_; }

	private: // ----------------------------------------------------------------

		// managers
		Bench::BenchmarksMgr& benchmarksMgr_;

		// neighbor search
		std::unique_ptr<I_NeighborSearchStrat> neighborSearchStrat_;
		friend class NS_BruteForce_CPU01;
		friend class NS_SpatialHashingBasic_CPU01;

		// metadata
		int64_t simStep_ = 0;
		double simTime_ = 0.0;
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// --- [ backend-specific data structures for operating the simulation ]

		// fluid particles
		std::vector<float> xPosF_, yPosF_, zPosF_;
		std::vector<float> xVelF_, yVelF_, zVelF_;
		std::vector<float> xAccF_, yAccF_, zAccF_;
		std::vector<float> xAccPressF_, yAccPressF_, zAccPressF_;
		std::vector<float> xAccViscF_, yAccViscF_, zAccViscF_;
		std::vector<float> xAccExtF_, yAccExtF_, zAccExtF_;
		std::vector<float> densityF_;
		std::vector<float> pressureF_;
		std::vector<float> massF_;
		std::vector<float> sizeF_;
		std::vector<uint32_t> idF_;

		// boundary particles
		std::vector<float> xPosB_, yPosB_, zPosB_;
		std::vector<float> xVelB_, yVelB_, zVelB_;
		std::vector<float> densityB_;
		std::vector<float> pressureB_;
		std::vector<float> massB_;
		std::vector<float> sizeB_;
		std::vector<uint32_t> idB_;

		// neighbor storage
		std::vector<std::vector<uint32_t>> neighborsFF_;
		std::vector<std::vector<uint32_t>> neighborsFB_;

		// kernel weight storage
		std::vector<std::vector<float>> kernelWeightsFF_;
		std::vector<std::vector<float>> kernelWeightsFB_;
		std::vector<std::vector<glm::vec3>> kernelGradientsFF_;
		std::vector<std::vector<glm::vec3>> kernelGradientsFB_;

		// re-usable work (scratch) buffer for temp storage
		mutable std::vector<float> workBuffer_;

		// operation benchmarking groups
		std::vector<Bench::BenchOpGroupDef> benchOpGroups_ =
		{
			{
				.ops = {	Bench::BenchOp::Neighbors_Construction }
			},
			{
				.ops = {	Bench::BenchOp::Neighbors_Query }
			},
			{
				.ops = {  Bench::BenchOp::Kernel_Weights,
							Bench::BenchOp::Kernel_Gradients }
			},
			{
				.ops = {	Bench::BenchOp::Density_Solving,
							Bench::BenchOp::Pressure_Solving }
			},
			{
				.ops = {	Bench::BenchOp::Pressure_Acceleration,
							Bench::BenchOp::Viscosity_Acceleration,
							Bench::BenchOp::External_Acceleration,
							Bench::BenchOp::AccelerationUpdate }
			},
			{
				.ops = {	Bench::BenchOp::VelocityUpdate,
							Bench::BenchOp::PositionUpdate }
			},
			{
				.ops = {  Bench::BenchOp::Statistics_Particles }
			},
			{
				.ops = {  Bench::BenchOp::Statistics_Density }
			},
			{
				.ops = {  Bench::BenchOp::Statistics_Velocity }
			},
			{
				.ops = {  Bench::BenchOp::Snapshot_Visual }
			},
			{
				.ops = {  Bench::BenchOp::Total_Sim }
			},
			{
				.ops = {  Bench::BenchOp::Misc_Sim }
			}
		};
	};

} // namespace Sim
