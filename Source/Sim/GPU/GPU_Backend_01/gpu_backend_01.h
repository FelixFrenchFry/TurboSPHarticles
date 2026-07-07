// gpu_backend_01.h
//
// TODO: comment

#pragma once
#ifdef TURBOSPHARTICLES_WITH_CUDA

// external
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>

// internal
#include "Source/Eval/Bench/benchmarks.h"
#include "Source/Sim/Core/backend_sim.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "statistics_kernels.cuh"



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
		friend class NS_BruteForce_GPU01;
		friend class NS_CellLinkedList_Compact_GPU01;

		// CUDA stream
		cudaStream_t stream_ = nullptr;

		// metadata
		int64_t simStep_ = 0;
		double simTime_ = 0.0;
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;
		uint32_t N_rigidBodies_ = 0;
		uint32_t N_rigidBoundaryParticles_ = 0;
		uint32_t N_solverIterCount_ = 0;

		// --- [ backend-specific data structures for operating the simulation ]

		// fluid particles
		DeviceBuffer<uint32_t> d_idF_;
		DeviceBuffer<float> d_xPosF_, d_yPosF_, d_zPosF_;
		DeviceBuffer<float> d_xVelF_, d_yVelF_, d_zVelF_;
		DeviceBuffer<float> d_xVelPredF_, d_yVelPredF_, d_zVelPredF_;
		DeviceBuffer<float> d_xAccPressF_, d_yAccPressF_, d_zAccPressF_;
		DeviceBuffer<float> d_sourceTermF_;
		DeviceBuffer<float> d_diagElemF_;
		DeviceBuffer<float> d_pressureF_;
		DeviceBuffer<float> d_densityF_;

		// boundary particles
		DeviceBuffer<uint32_t> d_idB_;
		DeviceBuffer<uint32_t> d_bodyIdB_;
		DeviceBuffer<uint32_t> d_rigidBoundaryParticleIdxB_;
		DeviceBuffer<float> d_xPosB_, d_yPosB_, d_zPosB_;
		DeviceBuffer<float> d_xVelB_, d_yVelB_, d_zVelB_;
		DeviceBuffer<float> d_volB_; // non-uniform volumes for non-uniform boundary particle spacing
		DeviceBuffer<float> d_psiB_; // non-uniform pseudo-masses for non-uniform boundary particle spacing

		// rigid bodies
		DeviceBuffer<uint32_t> d_idR_; // per-rigid-body ID
		DeviceBuffer<uint32_t> d_rigidBoundaryParticleBeginR_; // first rigid boundary particle
		DeviceBuffer<uint32_t> d_rigidBoundaryParticleCountR_; // number of rigid boundary particles

		// body state (in world frame)
		DeviceBuffer<float> d_xCMR_, d_yCMR_, d_zCMR_; // center of mass
		DeviceBuffer<float> d_xVelR_, d_yVelR_, d_zVelR_; // linear velocity
		DeviceBuffer<float> d_xOmegaR_, d_yOmegaR_, d_zOmegaR_; // angular velocity
		DeviceBuffer<float> d_xAngMomR_, d_yAngMomR_, d_zAngMomR_; // angular momentum
		DeviceBuffer<float> d_xForceR_, d_yForceR_, d_zForceR_; // force accumulated from rigid boundary particles
		DeviceBuffer<float> d_xTorqueR_, d_yTorqueR_, d_zTorqueR_; // torque accumulated from rigid boundary particles

		// orientation matrix A, stored row-major
		DeviceBuffer<float> d_a00R_, d_a01R_, d_a02R_;
		DeviceBuffer<float> d_a10R_, d_a11R_, d_a12R_;
		DeviceBuffer<float> d_a20R_, d_a21R_, d_a22R_;

		// mass
		DeviceBuffer<float> d_massR_;
		DeviceBuffer<float> d_massInvR_;

		// inverse inertia in body frame, stored row-major
		DeviceBuffer<float> d_iBodyInv00R_, d_iBodyInv01R_, d_iBodyInv02R_;
		DeviceBuffer<float> d_iBodyInv10R_, d_iBodyInv11R_, d_iBodyInv12R_;
		DeviceBuffer<float> d_iBodyInv20R_, d_iBodyInv21R_, d_iBodyInv22R_;

		// local rigid boundary particle offsets from center of mass
		DeviceBuffer<float> d_xLocalRigidBoundaryParticle_, d_yLocalRigidBoundaryParticle_, d_zLocalRigidBoundaryParticle_;

		// neighbor list storage
		// (compressed SELL32 for FF and FB neighbors)
		DeviceBuffer<uint16_t> d_nbrCountFF_; // temporary counts written by count-query
		DeviceBuffer<uint16_t> d_nbrCountFB_;
		DeviceBuffer<uint16_t> d_nbrChunkMaxCountFF_; // per-SELL32-chunk max neighbor count (pitch)
		DeviceBuffer<uint16_t> d_nbrChunkMaxCountFB_;
		DeviceBuffer<uint32_t> d_nbrChunkBaseIdxFF_; // per-chunk base index used for delta storage
		DeviceBuffer<uint32_t> d_nbrChunkBaseIdxFB_;
		DeviceBuffer<uint8_t> d_nbrChunkFormatFF_; // per-chunk storage format
		DeviceBuffer<uint8_t> d_nbrChunkFormatFB_;
		DeviceBuffer<uint64_t> d_nbrChunkOffsetsFF_; // per-chunk starting index into the selected format buffer
		DeviceBuffer<uint64_t> d_nbrChunkOffsetsFB_;
		DeviceBuffer<uint8_t> d_nbrListDelta8FF_; // contiguous delta8 neighbor storage
		DeviceBuffer<uint8_t> d_nbrListDelta8FB_;
		DeviceBuffer<uint16_t> d_nbrListDelta16FF_; // contiguous delta16 neighbor storage
		DeviceBuffer<uint16_t> d_nbrListDelta16FB_;
		DeviceBuffer<uint32_t> d_nbrListAbs32FF_; // contiguous abs32 neighbor storage
		DeviceBuffer<uint32_t> d_nbrListAbs32FB_;
		DeviceBuffer<uint32_t> d_nbrOverflowFlag_; // TODO: use smaller data type?
		uint64_t N_totalFF_ = 0; // actual neighbor count sum over all particles
		uint64_t N_totalFB_ = 0;
		uint64_t N_packedFF_ = 0; // total compressed storage length in bytes
		uint64_t N_packedFB_ = 0;
		uint64_t N_delta8FF_ = 0; // compressed delta8 storage length in refs
		uint64_t N_delta8FB_ = 0;
		uint64_t N_delta16FF_ = 0; // compressed delta16 storage length in refs
		uint64_t N_delta16FB_ = 0;
		uint64_t N_abs32FF_ = 0; // compressed abs32 storage length in refs
		uint64_t N_abs32FB_ = 0;
		const uint32_t extraAllocCapacityPct_ = 2; // TODO: fine-tune
		DeviceBuffer<uint8_t> d_nsSharedScratch_; // shared scratch arena for neighbor search/query/finalize
		DeviceBuffer<uint32_t> d_reorderIdScratch_;

		// reusable device scratch for particle statistics
		mutable DeviceBuffer<uint32_t> d_nbrCountHistFF_;
		mutable DeviceBuffer<uint32_t> d_nbrCountHistFB_;
		mutable DeviceBuffer<uint32_t> d_nbrCountHistChunkFF_;
		mutable DeviceBuffer<uint32_t> d_nbrCountHistChunkFB_;
		mutable DeviceBuffer<GPU01::NeighborCountStats> d_nbrCountStats_;
		mutable GPU01::NeighborCountStats h_nbrCountStats_;

		// re-usable host staging buffers for export snapshot packing
		// (reduces per-export allocations and allocator overhead)
		mutable std::vector<uint16_t> h_expNbrCountFF_;
		mutable std::vector<uint16_t> h_expNbrCountFB_;
		mutable std::vector<uint64_t> h_expNbrChunkOffsetsFF_;
		mutable std::vector<uint32_t> h_expNbrChunkBaseIdxFF_;
		mutable std::vector<uint8_t> h_expNbrChunkFormatFF_;
		mutable std::vector<uint8_t> h_expNbrListDelta8FF_;
		mutable std::vector<uint16_t> h_expNbrListDelta16FF_;
		mutable std::vector<uint32_t> h_expNbrListAbs32FF_;
		mutable std::vector<float> h_expXPosF_;
		mutable std::vector<float> h_expYPosF_;
		mutable std::vector<float> h_expZPosF_;
		mutable std::vector<float> h_expXVelF_;
		mutable std::vector<float> h_expYVelF_;
		mutable std::vector<float> h_expZVelF_;
		mutable std::vector<uint32_t> h_expIdF_;
		mutable std::vector<uint32_t> h_expIdB_;
		mutable std::vector<uint32_t> h_expBodyIdB_;
		mutable std::vector<float> h_expXPosB_;
		mutable std::vector<float> h_expYPosB_;
		mutable std::vector<float> h_expZPosB_;
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
			const uint32_t N_reorder_scratch_new = std::max(N_F_new, N_B_new);
			const bool needOverflow = (d_nbrOverflowFlag_.Size() != 1);
			const bool needIdScratch =
				d_reorderIdScratch_.Size() != N_reorder_scratch_new;
			const bool needFluid =
				force ||
				(N_F_new != N_F_) ||
				d_idF_.Size() != N_F_new ||
				d_xPosF_.Size() != N_F_new || d_yPosF_.Size() != N_F_new || d_zPosF_.Size() != N_F_new ||
				d_xVelF_.Size() != N_F_new || d_yVelF_.Size() != N_F_new || d_zVelF_.Size() != N_F_new ||
				d_xVelPredF_.Size() != N_F_new || d_yVelPredF_.Size() != N_F_new || d_zVelPredF_.Size() != N_F_new ||
				d_xAccPressF_.Size() != N_F_new || d_yAccPressF_.Size() != N_F_new || d_zAccPressF_.Size() != N_F_new ||
				d_sourceTermF_.Size() != N_F_new || d_diagElemF_.Size() != N_F_new ||
				d_pressureF_.Size() != N_F_new || d_densityF_.Size() != std::max(N_F_new, N_B_new) ||
				d_nbrCountFF_.Size() != N_F_new || d_nbrCountFB_.Size() != N_F_new ||
				d_nbrChunkMaxCountFF_.Size() != N_chunks_new ||
				d_nbrChunkMaxCountFB_.Size() != N_chunks_new ||
				d_nbrChunkBaseIdxFF_.Size() != N_chunks_new ||
				d_nbrChunkBaseIdxFB_.Size() != N_chunks_new ||
				d_nbrChunkFormatFF_.Size() != N_chunks_new ||
				d_nbrChunkFormatFB_.Size() != N_chunks_new ||
				d_nbrChunkOffsetsFF_.Size() != static_cast<size_t>(N_chunks_new) + 1 ||
				d_nbrChunkOffsetsFB_.Size() != static_cast<size_t>(N_chunks_new) + 1;

			const bool needBoundary =
				force ||
				(N_B_new != N_B_) ||
				d_idB_.Size() != N_B_new ||
				d_bodyIdB_.Size() != N_B_new ||
				d_rigidBoundaryParticleIdxB_.Size() != N_B_new ||
				d_xPosB_.Size() != N_B_new || d_yPosB_.Size() != N_B_new || d_zPosB_.Size() != N_B_new ||
				d_xVelB_.Size() != N_B_new || d_yVelB_.Size() != N_B_new || d_zVelB_.Size() != N_B_new ||
				d_volB_.Size() != N_B_new || d_psiB_.Size() != N_B_new;

			// detect if (re)allocation of buffers is needed
			// (might be due to change in particle counts or lack of initialization)
			if (!(needFluid || needBoundary || needOverflow || needIdScratch)) { return; }

			N_F_ = N_F_new;
			N_B_ = N_B_new;

			// fluid
			d_idF_.Resize(N_F_);
			d_xPosF_.Resize(N_F_); d_yPosF_.Resize(N_F_); d_zPosF_.Resize(N_F_);
			d_xVelF_.Resize(N_F_); d_yVelF_.Resize(N_F_); d_zVelF_.Resize(N_F_);
			d_xVelPredF_.Resize(N_F_); d_yVelPredF_.Resize(N_F_); d_zVelPredF_.Resize(N_F_);
			d_xAccPressF_.Resize(N_F_); d_yAccPressF_.Resize(N_F_); d_zAccPressF_.Resize(N_F_);
			d_sourceTermF_.Resize(N_F_);
			d_diagElemF_.Resize(N_F_);
			d_pressureF_.Resize(N_F_);
			d_densityF_.Resize(std::max(N_F_, N_B_));

			// boundary
			d_idB_.Resize(N_B_);
			d_bodyIdB_.Resize(N_B_);
			d_rigidBoundaryParticleIdxB_.Resize(N_B_);
			d_xPosB_.Resize(N_B_); d_yPosB_.Resize(N_B_); d_zPosB_.Resize(N_B_);
			d_xVelB_.Resize(N_B_); d_yVelB_.Resize(N_B_); d_zVelB_.Resize(N_B_);
			d_volB_.Resize(N_B_);
			d_psiB_.Resize(N_B_);
			d_reorderIdScratch_.Resize(std::max(N_F_, N_B_));

			// neighbors
			d_nbrCountFF_.Resize(N_F_);
			d_nbrCountFB_.Resize(N_F_);
			d_nbrChunkMaxCountFF_.Resize(N_chunks_new);
			d_nbrChunkMaxCountFB_.Resize(N_chunks_new);
			d_nbrChunkBaseIdxFF_.Resize(N_chunks_new);
			d_nbrChunkBaseIdxFB_.Resize(N_chunks_new);
			d_nbrChunkFormatFF_.Resize(N_chunks_new);
			d_nbrChunkFormatFB_.Resize(N_chunks_new);
			d_nbrChunkOffsetsFF_.Resize(static_cast<size_t>(N_chunks_new) + 1ull);
			d_nbrChunkOffsetsFB_.Resize(static_cast<size_t>(N_chunks_new) + 1ull);
			d_nbrOverflowFlag_.Resize(1);
			N_totalFF_ = 0;
			N_totalFB_ = 0;
			N_packedFF_ = 0;
			N_packedFB_ = 0;
			N_delta8FF_ = 0;
			N_delta8FB_ = 0;
			N_delta16FF_ = 0;
			N_delta16FB_ = 0;
			N_abs32FF_ = 0;
			N_abs32FB_ = 0;
		}

		void EnsureRigidCapacity(uint32_t N_rigidBodies_new, uint32_t N_rigidBoundaryParticles_new, bool force = false)
		{
			const bool needBodies =
				force ||
				(N_rigidBodies_new != N_rigidBodies_) ||
				d_idR_.Size() != N_rigidBodies_new ||
				d_rigidBoundaryParticleBeginR_.Size() != N_rigidBodies_new ||
				d_rigidBoundaryParticleCountR_.Size() != N_rigidBodies_new ||
				d_xCMR_.Size() != N_rigidBodies_new || d_yCMR_.Size() != N_rigidBodies_new || d_zCMR_.Size() != N_rigidBodies_new ||
				d_xVelR_.Size() != N_rigidBodies_new || d_yVelR_.Size() != N_rigidBodies_new || d_zVelR_.Size() != N_rigidBodies_new ||
				d_xOmegaR_.Size() != N_rigidBodies_new || d_yOmegaR_.Size() != N_rigidBodies_new || d_zOmegaR_.Size() != N_rigidBodies_new ||
				d_xAngMomR_.Size() != N_rigidBodies_new || d_yAngMomR_.Size() != N_rigidBodies_new || d_zAngMomR_.Size() != N_rigidBodies_new ||
				d_xForceR_.Size() != N_rigidBodies_new || d_yForceR_.Size() != N_rigidBodies_new || d_zForceR_.Size() != N_rigidBodies_new ||
				d_xTorqueR_.Size() != N_rigidBodies_new || d_yTorqueR_.Size() != N_rigidBodies_new || d_zTorqueR_.Size() != N_rigidBodies_new ||
				d_a00R_.Size() != N_rigidBodies_new || d_a01R_.Size() != N_rigidBodies_new || d_a02R_.Size() != N_rigidBodies_new ||
				d_a10R_.Size() != N_rigidBodies_new || d_a11R_.Size() != N_rigidBodies_new || d_a12R_.Size() != N_rigidBodies_new ||
				d_a20R_.Size() != N_rigidBodies_new || d_a21R_.Size() != N_rigidBodies_new || d_a22R_.Size() != N_rigidBodies_new ||
				d_massR_.Size() != N_rigidBodies_new || d_massInvR_.Size() != N_rigidBodies_new ||
				d_iBodyInv00R_.Size() != N_rigidBodies_new || d_iBodyInv01R_.Size() != N_rigidBodies_new || d_iBodyInv02R_.Size() != N_rigidBodies_new ||
				d_iBodyInv10R_.Size() != N_rigidBodies_new || d_iBodyInv11R_.Size() != N_rigidBodies_new || d_iBodyInv12R_.Size() != N_rigidBodies_new ||
				d_iBodyInv20R_.Size() != N_rigidBodies_new || d_iBodyInv21R_.Size() != N_rigidBodies_new || d_iBodyInv22R_.Size() != N_rigidBodies_new;

			const bool needRigidBoundaryParticles =
				force ||
				(N_rigidBoundaryParticles_new != N_rigidBoundaryParticles_) ||
				d_xLocalRigidBoundaryParticle_.Size() != N_rigidBoundaryParticles_new ||
				d_yLocalRigidBoundaryParticle_.Size() != N_rigidBoundaryParticles_new ||
				d_zLocalRigidBoundaryParticle_.Size() != N_rigidBoundaryParticles_new;

			if (!(needBodies || needRigidBoundaryParticles)) { return; }

			N_rigidBodies_ = N_rigidBodies_new;
			N_rigidBoundaryParticles_ = N_rigidBoundaryParticles_new;

			d_idR_.Resize(N_rigidBodies_);
			d_rigidBoundaryParticleBeginR_.Resize(N_rigidBodies_);
			d_rigidBoundaryParticleCountR_.Resize(N_rigidBodies_);

			d_xCMR_.Resize(N_rigidBodies_); d_yCMR_.Resize(N_rigidBodies_); d_zCMR_.Resize(N_rigidBodies_);
			d_xVelR_.Resize(N_rigidBodies_); d_yVelR_.Resize(N_rigidBodies_); d_zVelR_.Resize(N_rigidBodies_);
			d_xOmegaR_.Resize(N_rigidBodies_); d_yOmegaR_.Resize(N_rigidBodies_); d_zOmegaR_.Resize(N_rigidBodies_);
			d_xAngMomR_.Resize(N_rigidBodies_); d_yAngMomR_.Resize(N_rigidBodies_); d_zAngMomR_.Resize(N_rigidBodies_);
			d_xForceR_.Resize(N_rigidBodies_); d_yForceR_.Resize(N_rigidBodies_); d_zForceR_.Resize(N_rigidBodies_);
			d_xTorqueR_.Resize(N_rigidBodies_); d_yTorqueR_.Resize(N_rigidBodies_); d_zTorqueR_.Resize(N_rigidBodies_);

			d_a00R_.Resize(N_rigidBodies_); d_a01R_.Resize(N_rigidBodies_); d_a02R_.Resize(N_rigidBodies_);
			d_a10R_.Resize(N_rigidBodies_); d_a11R_.Resize(N_rigidBodies_); d_a12R_.Resize(N_rigidBodies_);
			d_a20R_.Resize(N_rigidBodies_); d_a21R_.Resize(N_rigidBodies_); d_a22R_.Resize(N_rigidBodies_);

			d_massR_.Resize(N_rigidBodies_);
			d_massInvR_.Resize(N_rigidBodies_);

			d_iBodyInv00R_.Resize(N_rigidBodies_); d_iBodyInv01R_.Resize(N_rigidBodies_); d_iBodyInv02R_.Resize(N_rigidBodies_);
			d_iBodyInv10R_.Resize(N_rigidBodies_); d_iBodyInv11R_.Resize(N_rigidBodies_); d_iBodyInv12R_.Resize(N_rigidBodies_);
			d_iBodyInv20R_.Resize(N_rigidBodies_); d_iBodyInv21R_.Resize(N_rigidBodies_); d_iBodyInv22R_.Resize(N_rigidBodies_);

			d_xLocalRigidBoundaryParticle_.Resize(N_rigidBoundaryParticles_);
			d_yLocalRigidBoundaryParticle_.Resize(N_rigidBoundaryParticles_);
			d_zLocalRigidBoundaryParticle_.Resize(N_rigidBoundaryParticles_);
		}

		// turns temporary count-query output into compressed SELL32 metadata
		// and ensures list-storage capacity for the fill-query.
		void FinalizeNeighborLists_Count();

		void EnsureNsSharedScratchCapacity(const size_t requiredBytes)
		{
			if (requiredBytes <= d_nsSharedScratch_.Size()) { return; }

			const size_t capacity = requiredBytes
				+ (requiredBytes * static_cast<size_t>(extraAllocCapacityPct_)) / size_t{ 100 };
			d_nsSharedScratch_.Resize(capacity);
		}
	};

} // namespace Sim

#endif
