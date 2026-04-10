// gpu_backend_01.h
//
// TODO: comment

#pragma once
#ifdef TURBOSPHARTICLES_WITH_CUDA

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
	public: // ---------------------------------------------------------------------------------------

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

	private: // --------------------------------------------------------------------------------------

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
	public: // ---------------------------------------------------------------------------------------

		SimBackend_GPU01(Sim::SimParamsMgr& simParamsMgr,
						 Bench::BenchmarksTracker& benchmarksTracker,
						 const SimParams& params);
		~SimBackend_GPU01() override;

		const char* Name() const override
			{ return "GPU01"; }

		void SetNeighborSearchMethod(std::unique_ptr<I_NeighborSearch> ns) override
			{ neighborSearch_ = std::move(ns); }

		// copies & translates a universal simulation state into -> backend-specific data structures
		void PullFromUniversalSimState(Sync::UniversalSimState& src, const SimParams& params) override;

		// copies & translates backend-specific data structures into -> universal simulation state
		void PushToUniversalSimState(Sync::UniversalSimState& dst) const override;

		// packs the current simulation state into a snapshot used for exporting or visualization
		// (implementation is backend-specific for better performance)
		void PackExportSnapshot(Sync::ExportSnapshot& snap, const SimParams& params) const override;
		void PackVisualSnapshot(Sync::VisualSnapshot& snap) const override;

		// fills in the scalar data based on given different color coding modes
		// (backend-specific such that possible transformations can be applied efficiently)
		void FillScalarData(Sync::VisualSnapshot& snap, Base::ColorCodingMode mode) const override;

		// computes & fills in the memory usage data for the current sim step
		// (backend-specific so that the memory usage can be computed from the custom data structures efficiently)
		void FillMemoryRecord(Mem::MemRecordStep& record, const SimParams& params) const override;

		// computes & fills in the stat data for the current sim step
		// (backend-specific so that the stats can be computed from the custom data structures efficiently)
		void FillStatisticsRecord(Stats::StatRecordStep& record, const SimParams& params) const override;

		// executes only the neighbor search (construction and query of the data NS structure)
		void ExecuteNeighborSearch(const SimParams& params, bool bench = false) override;

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

		// managers
		Sim::SimParamsMgr& simParamsMgr_;
		Bench::BenchmarksTracker& benchmarksTracker_;

		// neighbor search
		std::unique_ptr<I_NeighborSearch> neighborSearch_;
		std::unique_ptr<I_NeighborList> neighborList_; // TODO: implement different neighbor list methods?
		friend class NS_BruteForce_GPU01;
		friend class NS_CellLinkedList_Compact_GPU01;

		// CUDA stream
		cudaStream_t stream_ = nullptr;

		// metadata
		int64_t simStep_ = 0;
		double simTime_ = 0.0;
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;
		uint32_t N_solverIterCount_ = 0;

		// --- [ backend-specific data structures for operating the simulation ]

		// fluid particles
		DeviceBuffer<float> d_xPosF_, d_yPosF_, d_zPosF_;
		DeviceBuffer<float> d_xVelF_, d_yVelF_, d_zVelF_;
		DeviceBuffer<float> d_xVelPredF_, d_yVelPredF_, d_zVelPredF_;
		DeviceBuffer<float> d_xAccPressF_, d_yAccPressF_, d_zAccPressF_;
		DeviceBuffer<float> d_sourceTermF_;
		DeviceBuffer<float> d_diagElemF_;
		DeviceBuffer<float> d_pressureF_;
		DeviceBuffer<float> d_densityF_;

		// boundary particles
		DeviceBuffer<float> d_xPosB_, d_yPosB_, d_zPosB_;

		// neighbor list storage (SELL32_DoubleQuery style)
		DeviceBuffer<uint16_t> d_nbrCountFF_; // temporary counts written by count-query
		DeviceBuffer<uint16_t> d_nbrCountFB_; // temporary counts written by count-query
		DeviceBuffer<uint16_t> d_nbrChunkMaxCountFF_; // per-SELL32-chunk max FF neighbor count
		DeviceBuffer<uint16_t> d_nbrChunkMaxCountFB_; // per-SELL32-chunk max FB neighbor count
		DeviceBuffer<uint64_t> d_nbrChunkOffsetsFF_; // SELL32 chunk offsets (N_chunks + 1)
		DeviceBuffer<uint64_t> d_nbrChunkOffsetsFB_; // SELL32 chunk offsets (N_chunks + 1)
		DeviceBuffer<uint32_t> d_nbrListFF_; // packed SELL32 FF neighbor list
		DeviceBuffer<uint32_t> d_nbrListFB_; // packed SELL32 FB neighbor list
		DeviceBuffer<uint32_t> d_nbrOverflowFlag_; // TODO: use smaller data type?
		uint64_t N_totalFF_ = 0; // actual FF neighbor count sum over all particles
		uint64_t N_totalFB_ = 0; // actual FB neighbor count sum over all particles
		uint64_t N_packedFF_ = 0; // padded SELL32 FF storage length
		uint64_t N_packedFB_ = 0; // padded SELL32 FB storage length
		const uint32_t extraAllocCapacityPct_ = 2; // TODO: fine-tune

		// re-usable work (scratch) buffer for temp storage
		DeviceBuffer<float> d_workBuffer_;

		// re-usable host staging buffers for export snapshot packing
		// (reduces per-export allocations and allocator overhead)
		mutable std::vector<uint16_t> h_expNbrCountFF_;
		mutable std::vector<uint16_t> h_expNbrCountFB_;
		mutable std::vector<uint64_t> h_expNbrChunkOffsetsFF_;
		mutable std::vector<uint32_t> h_expNbrListFF_;
		mutable std::vector<float> h_expXPosF_;
		mutable std::vector<float> h_expYPosF_;
		mutable std::vector<float> h_expZPosF_;
		mutable std::vector<float> h_expXVelF_;
		mutable std::vector<float> h_expYVelF_;
		mutable std::vector<float> h_expZVelF_;
		mutable std::vector<uint8_t> h_expSurfaceFlagsScratch_;

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
						   Bench::BenchOp::Statistics_Velocity}
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
			},
			{
				.ops = { Bench::BenchOp::Misc_Sim }
			}
		};

		// resizes all device buffers when particle counts change
		// (does not preserve data)
		void EnsureCapacity(uint32_t N_F_new, uint32_t N_B_new, bool force = false)
		{
			const uint32_t N_chunks_new =
				(N_F_new == 0u) ? 0u : ((N_F_new - 1u) / 32u) + 1u;
			const bool needOverflow = (d_nbrOverflowFlag_.Size() != 1);

			const bool needFluid =
				force ||
				(N_F_new != N_F_) ||
				d_xPosF_.Size() != N_F_new || d_yPosF_.Size() != N_F_new || d_zPosF_.Size() != N_F_new ||
				d_xVelF_.Size() != N_F_new || d_yVelF_.Size() != N_F_new || d_zVelF_.Size() != N_F_new ||
				d_xVelPredF_.Size() != N_F_new || d_yVelPredF_.Size() != N_F_new || d_zVelPredF_.Size() != N_F_new ||
				d_xAccPressF_.Size() != N_F_new || d_yAccPressF_.Size() != N_F_new || d_zAccPressF_.Size() != N_F_new ||
				d_sourceTermF_.Size() != N_F_new || d_diagElemF_.Size() != N_F_new ||
				d_pressureF_.Size() != N_F_new || d_densityF_.Size() != N_F_new ||
				d_nbrCountFF_.Size() != N_F_new || d_nbrCountFB_.Size() != N_F_new ||
				d_nbrChunkMaxCountFF_.Size() != N_chunks_new ||
				d_nbrChunkMaxCountFB_.Size() != N_chunks_new ||
				d_nbrChunkOffsetsFF_.Size() != static_cast<size_t>(N_chunks_new) + 1 ||
				d_nbrChunkOffsetsFB_.Size() != static_cast<size_t>(N_chunks_new) + 1;

			const bool needBoundary =
				force ||
				(N_B_new != N_B_) ||
				d_xPosB_.Size() != N_B_new || d_yPosB_.Size() != N_B_new || d_zPosB_.Size() != N_B_new;

			// detect if (re)allocation of buffers is needed
			// (might be due to change in particle counts or lack of initialization)
			if (!(needFluid || needBoundary || needOverflow)) { return; }

			N_F_ = N_F_new;
			N_B_ = N_B_new;

			// fluid
			d_xPosF_.Resize(N_F_); d_yPosF_.Resize(N_F_); d_zPosF_.Resize(N_F_);
			d_xVelF_.Resize(N_F_); d_yVelF_.Resize(N_F_); d_zVelF_.Resize(N_F_);
			d_xVelPredF_.Resize(N_F_); d_yVelPredF_.Resize(N_F_); d_zVelPredF_.Resize(N_F_);
			d_xAccPressF_.Resize(N_F_); d_yAccPressF_.Resize(N_F_); d_zAccPressF_.Resize(N_F_);
			d_sourceTermF_.Resize(N_F_);
			d_diagElemF_.Resize(N_F_);
			d_pressureF_.Resize(N_F_);
			d_densityF_.Resize(N_F_);

			// boundary
			d_xPosB_.Resize(N_B_); d_yPosB_.Resize(N_B_); d_zPosB_.Resize(N_B_);

			// neighbors
			d_nbrCountFF_.Resize(N_F_);
			d_nbrCountFB_.Resize(N_F_);
			d_nbrChunkMaxCountFF_.Resize(N_chunks_new);
			d_nbrChunkMaxCountFB_.Resize(N_chunks_new);
			d_nbrChunkOffsetsFF_.Resize(static_cast<size_t>(N_chunks_new) + 1ull);
			d_nbrChunkOffsetsFB_.Resize(static_cast<size_t>(N_chunks_new) + 1ull);
			d_nbrOverflowFlag_.Resize(1);
			N_totalFF_ = 0;
			N_totalFB_ = 0;
			N_packedFF_ = 0;
			N_packedFB_ = 0;
		}

		// turns temporary count-query output into SELL32 chunk metadata and ensures
		// capacity of packed SELL32 neighbor list storage for the fill-query.
		void FinalizeNeighborLists_Count();
	};

} // namespace Sim

#endif
