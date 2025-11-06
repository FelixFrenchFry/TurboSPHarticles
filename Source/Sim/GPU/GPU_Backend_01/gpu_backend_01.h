// gpu_backend_01.h
//
// TODO: comment

#pragma once

// external
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>

// internal
#include "Source/Eval/Bench/benchmarks.h"
#include "Source/Sim/Core/backend_sim.h"
#include "Source/Sim/Core/parameters_sim.h"



namespace Sim
{
	// vector struct for device-side 3D floats
	struct Vec3f
	{
		float x, y, z;
		__host__ __device__ Vec3f() : x(0.f), y(0.f), z(0.f) {}
		__host__ __device__ Vec3f(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
	};

	template <class T>
	class DeviceBuffer
	{
	public: // -----------------------------------------------------------------

		DeviceBuffer() = default;
		~DeviceBuffer() { Release(); }

		DeviceBuffer(const DeviceBuffer&) = delete;
		DeviceBuffer& operator=(const DeviceBuffer&) = delete;

		DeviceBuffer(DeviceBuffer&& other) noexcept { Swap_(other); }
		DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
		{
			if (this != &other) Swap_(other);
			return *this;
		}

		void Resize(size_t n)
		{
			if (n == size_) { return; }

			Release();

			if (n)
			{
				cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T));
				if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
			}

			size_ = n;
		}

		T* Data() { return ptr_; }
		const T* Data() const { return ptr_; }
		size_t Size() const { return size_; }

		void Release()
		{
			if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; }

			size_ = 0;
		}

	private: // ----------------------------------------------------------------

		void Swap_(DeviceBuffer& o) noexcept
		{
			std::swap(ptr_, o.ptr_);
			std::swap(size_, o.size_);
		}

		T* ptr_ = nullptr;
		size_t size_ = 0;
	};

	class SimBackend_GPU01 final : public I_SimBackend
	{
	public: // -----------------------------------------------------------------

		SimBackend_GPU01(Bench::BenchmarksMgr& benchmarksMgr);
		~SimBackend_GPU01() override;

		const char* Name() const override { return "GPU01"; }

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
		friend class NS_BruteForce_GPU01;
		friend class NS_SpatialHashingBasic_GPU01;

		// CUDA stream
		cudaStream_t stream_ = nullptr;

		// metadata
		int64_t simStep_ = 0;
		double simTime_ = 0.0;
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// --- [ backend-specific data structures for operating the simulation ]

		// fluid particles
		DeviceBuffer<float> d_xPosF_, d_yPosF_, d_zPosF_;
		DeviceBuffer<float> d_xVelF_, d_yVelF_, d_zVelF_;
		DeviceBuffer<float> d_xAccF_, d_yAccF_, d_zAccF_;
		DeviceBuffer<float> d_xAccPressF_, d_yAccPressF_, d_zAccPressF_;
		DeviceBuffer<float> d_xAccViscF_, d_yAccViscF_, d_zAccViscF_;
		DeviceBuffer<float> d_xAccExtF_, d_yAccExtF_, d_zAccExtF_;
		DeviceBuffer<float> d_densityF_;
		DeviceBuffer<float> d_pressureF_;
		DeviceBuffer<float> d_massF_;
		DeviceBuffer<float> d_sizeF_;
		DeviceBuffer<uint32_t> d_idF_;

		// boundary particles
		DeviceBuffer<float> d_xPosB_, d_yPosB_, d_zPosB_;
		DeviceBuffer<float> d_xVelB_, d_yVelB_, d_zVelB_;
		DeviceBuffer<float> d_densityB_;
		DeviceBuffer<float> d_pressureB_;
		DeviceBuffer<float> d_massB_;
		DeviceBuffer<float> d_sizeB_;
		DeviceBuffer<uint32_t> d_idB_;

		// neighbors
		static constexpr uint32_t N_MAX_FF_ = 96;
		static constexpr uint32_t N_MAX_FB_ = 64;

		DeviceBuffer<uint32_t> d_neighCountFF_;
		DeviceBuffer<uint32_t> d_neighCountFB_;
		DeviceBuffer<uint32_t> d_neighListFF_; // N_F_ * N_MAX_NEIGH_FF_
		DeviceBuffer<uint32_t> d_neighListFB_; // N_F_ * N_MAX_NEIGH_FB_
		DeviceBuffer<uint32_t> d_neighOverflowFlag_; // TODO: use smaller data type?

		// re-usable work (scratch) buffer for temp storage
		DeviceBuffer<float> d_workBuffer_;

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

		// resizes all device buffers when particle counts change
		// (does not preserve data)
		void EnsureCapacity(uint32_t N_F_new, uint32_t N_B_new, bool force = false)
		{
			const bool needOverflow = (d_neighOverflowFlag_.Size() != 1);

			const bool needFluid =
				force ||
				(N_F_new != N_F_) ||
				d_xPosF_.Size() != N_F_new || d_yPosF_.Size() != N_F_new || d_zPosF_.Size() != N_F_new ||
				d_xVelF_.Size() != N_F_new || d_yVelF_.Size() != N_F_new || d_zVelF_.Size() != N_F_new ||
				d_xAccF_.Size() != N_F_new || d_yAccF_.Size() != N_F_new || d_zAccF_.Size() != N_F_new ||
				d_xAccPressF_.Size() != N_F_new || d_yAccPressF_.Size() != N_F_new || d_zAccPressF_.Size() != N_F_new ||
				d_xAccViscF_.Size() != N_F_new || d_yAccViscF_.Size() != N_F_new || d_zAccViscF_.Size() != N_F_new ||
				d_xAccExtF_.Size() != N_F_new || d_yAccExtF_.Size() != N_F_new || d_zAccExtF_.Size() != N_F_new ||
				d_densityF_.Size() != N_F_new || d_pressureF_.Size() != N_F_new ||
				d_massF_.Size() != N_F_new || d_sizeF_.Size() != N_F_new || d_idF_.Size() != N_F_new ||
				d_neighCountFF_.Size() != N_F_new || d_neighCountFB_.Size() != N_F_new ||
				d_neighListFF_.Size() != static_cast<size_t>(N_F_new) * N_MAX_FF_ ||
				d_neighListFB_.Size() != static_cast<size_t>(N_F_new) * N_MAX_FB_;

			const bool needBoundary =
				force ||
				(N_B_new != N_B_) ||
				d_xPosB_.Size() != N_B_new || d_yPosB_.Size() != N_B_new || d_zPosB_.Size() != N_B_new ||
				d_xVelB_.Size() != N_B_new || d_yVelB_.Size() != N_B_new || d_zVelB_.Size() != N_B_new ||
				d_densityB_.Size() != N_B_new || d_pressureB_.Size() != N_B_new ||
				d_massB_.Size() != N_B_new || d_sizeB_.Size() != N_B_new || d_idB_.Size() != N_B_new;

			// detect if (re)allocation of buffers is needed
			// (might be due to change in particle counts or lack of initialization)
			if (!(needFluid || needBoundary || needOverflow)) { return; }

			N_F_ = N_F_new;
			N_B_ = N_B_new;

			// fluid
			d_xPosF_.Resize(N_F_); d_yPosF_.Resize(N_F_); d_zPosF_.Resize(N_F_);
			d_xVelF_.Resize(N_F_); d_yVelF_.Resize(N_F_); d_zVelF_.Resize(N_F_);
			d_xAccF_.Resize(N_F_); d_yAccF_.Resize(N_F_); d_zAccF_.Resize(N_F_);
			d_xAccPressF_.Resize(N_F_); d_yAccPressF_.Resize(N_F_); d_zAccPressF_.Resize(N_F_);
			d_xAccViscF_.Resize(N_F_);  d_yAccViscF_.Resize(N_F_);  d_zAccViscF_.Resize(N_F_);
			d_xAccExtF_.Resize(N_F_);   d_yAccExtF_.Resize(N_F_);   d_zAccExtF_.Resize(N_F_);
			d_densityF_.Resize(N_F_);
			d_pressureF_.Resize(N_F_);
			d_massF_.Resize(N_F_);
			d_sizeF_.Resize(N_F_);
			d_idF_.Resize(N_F_);

			// boundary
			d_xPosB_.Resize(N_B_); d_yPosB_.Resize(N_B_); d_zPosB_.Resize(N_B_);
			d_xVelB_.Resize(N_B_); d_yVelB_.Resize(N_B_); d_zVelB_.Resize(N_B_);
			d_densityB_.Resize(N_B_);
			d_pressureB_.Resize(N_B_);
			d_massB_.Resize(N_B_);
			d_sizeB_.Resize(N_B_);
			d_idB_.Resize(N_B_);

			// neighbors
			d_neighCountFF_.Resize(N_F_);
			d_neighCountFB_.Resize(N_F_);
			d_neighListFF_.Resize(N_F_ * N_MAX_FF_);
			d_neighListFB_.Resize(N_F_ * N_MAX_FB_);
			d_neighOverflowFlag_.Resize(1);
		}
	};

} // namespace Sim
