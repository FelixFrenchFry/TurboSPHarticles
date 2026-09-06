// gpu_backend.h
//
// TODO: comment

#pragma once
#ifdef TURBOSPHARTICLES_WITH_CUDA

// external
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>

// internal
#include "common.cuh"
#include "source/base/helpers.h"
#include "source/eval/bench/benchmarks.h"
#include "source/simulation/core/backend_sim.h"
#include "source/simulation/core/parameters_sim.h"
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
				if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed.");

				Mem::TrackDeviceAlloc(n * sizeof(T));
			}

			size_ = n;
		}

		T* Data() { return ptr_; }
		const T* Data() const { return ptr_; }
		size_t Size() const { return size_; }

		void Release()
		{
			if (ptr_)
			{
				cudaFree(ptr_);
				ptr_ = nullptr;
				Mem::TrackDeviceFree(size_ * sizeof(T));
			}

			size_ = 0;
		}

		void Swap(DeviceBuffer& other) noexcept { Swap_(other); }

	private: // --------------------------------------------------------------------------------------

		void Swap_(DeviceBuffer& o) noexcept
		{
			std::swap(ptr_, o.ptr_);
			std::swap(size_, o.size_);
		}

		T* ptr_ = nullptr;
		size_t size_ = 0;
	};

	class SimBackend_GPU final : public I_SimBackend
	{
	public: // ---------------------------------------------------------------------------------------

		SimBackend_GPU(Sim::SimParamsMgr& simParamsMgr,
						 Bench::BenchmarksTracker& benchmarksTracker,
						 const SimParams& params);
		~SimBackend_GPU() override;

		const char* Name() const override
			{ return "GPU"; }

		void SetNeighborSearchMethod(std::unique_ptr<I_NeighborSearch> ns) override
			{ neighborSearch_ = std::move(ns); }

		void SetSimDomainBounds(const glm::vec3& min, const glm::vec3& max) override;

		// copies & translates a universal simulation state into -> backend-specific data structures
		void PullFromUniversalSimState(Sync::UniversalSimState& src, const SimParams& params) override;

		// materializes backend-specific data structures into -> universal simulation state
		void PushToUniversalSimState(Sync::UniversalSimState& dst, bool backendWillBeRetired) override;

		// packs the current simulation state into a snapshot used for exporting or visualization
		// (implementation is backend-specific for better performance)
		void PackExportSnapshot(Sync::ExportSnapshot& snap, const SimParams& params) const override;
		void PackVisualSnapshot(Sync::VisualSnapshot& snap) const override;
		Sync::VisualPackResultCUDA TryPackVisualSnapshotCUDA(
			Sync::VisualFrameMailboxCUDA& visualFrameMailboxCUDA,
			std::shared_ptr<Sync::VisualSnapshot> auxiliarySnap) const override;

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

		// provides the benchmark operation display order
		void GetBenchOps(Bench::BenchOpList& out) const override
			{ out = benchOps_; }

		// info
		uint32_t GetNumFluidParticles() const override { return N_F_; }
		uint32_t GetNumBoundaryParticles() const override { return N_B_; }
		uint32_t GetNumRigidParticles() const override { return N_R_; }

	private: // --------------------------------------------------------------------------------------

		// managers
		Sim::SimParamsMgr& simParamsMgr_;
		Bench::BenchmarksTracker& benchmarksTracker_;

		// neighbor search
		std::unique_ptr<I_NeighborSearch> neighborSearch_;
		friend class NS_BruteForce_GPU;
		friend class NS_CellLinkedList_Compact_GPU;

		// CUDA stream
		cudaStream_t stream_ = nullptr;

		// metadata
		uint64_t simStep_ = 0;
		double simTime_ = 0.0;
		uint32_t N_F_ = 0; // number of fluid particles
		uint32_t N_B_ = 0; // number of boundary particles
		uint32_t N_R_ = 0; // number of rigid particles
		uint32_t N_RB_ = 0; // number of rigid bodies
		uint32_t N_solverIters_ = 0;
		Vec3f simDomainMin_{};
		Vec3f simDomainMax_{};
		bool simDomainBoundsValid_ = false;

		// --- [ backend-specific data structures for operating the simulation ] ---

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
		DeviceBuffer<float> d_xPosB_, d_yPosB_, d_zPosB_;
		DeviceBuffer<float> d_volB_; // non-uniform volumes for non-uniform boundary particle spacing
		DeviceBuffer<float> d_psiB_; // non-uniform pseudo-masses for non-uniform boundary particle spacing

		// rigid particles
		DeviceBuffer<uint32_t> d_idR_;
		DeviceBuffer<uint32_t> d_bodyIdR_;
		DeviceBuffer<float> d_xPosR_, d_yPosR_, d_zPosR_;
		DeviceBuffer<float> d_xRelPosR_, d_yRelPosR_, d_zRelPosR_;
		DeviceBuffer<float> d_xVelR_, d_yVelR_, d_zVelR_;
		DeviceBuffer<float> d_xForceR_, d_yForceR_, d_zForceR_;
		DeviceBuffer<float> d_volR_;
		DeviceBuffer<float> d_psiR_;

		// rigid bodies
		DeviceBuffer<uint32_t> d_idRB_;
		DeviceBuffer<uint32_t> d_rigidParticleBeginRB_;
		DeviceBuffer<uint32_t> d_rigidParticleCountRB_;

		// body state (in world frame)
		DeviceBuffer<float> d_xCMRB_, d_yCMRB_, d_zCMRB_; // center of mass
		DeviceBuffer<float> d_xVelRB_, d_yVelRB_, d_zVelRB_; // linear velocity
		DeviceBuffer<float> d_xOmegaRB_, d_yOmegaRB_, d_zOmegaRB_; // angular velocity
		DeviceBuffer<float> d_xAngMomRB_, d_yAngMomRB_, d_zAngMomRB_; // angular momentum
		DeviceBuffer<float> d_xForceRB_, d_yForceRB_, d_zForceRB_; // force accumulated from rigid particles
		DeviceBuffer<float> d_xTorqueRB_, d_yTorqueRB_, d_zTorqueRB_; // torque accumulated from rigid particles

		// orientation matrix A, stored row-major
		DeviceBuffer<float> d_a00RB_, d_a01RB_, d_a02RB_;
		DeviceBuffer<float> d_a10RB_, d_a11RB_, d_a12RB_;
		DeviceBuffer<float> d_a20RB_, d_a21RB_, d_a22RB_;

		// mass
		DeviceBuffer<float> d_massRB_;
		DeviceBuffer<float> d_massInvRB_;

		// inverse inertia in body frame, stored row-major
		DeviceBuffer<float> d_iBodyInv00RB_, d_iBodyInv01RB_, d_iBodyInv02RB_;
		DeviceBuffer<float> d_iBodyInv10RB_, d_iBodyInv11RB_, d_iBodyInv12RB_;
		DeviceBuffer<float> d_iBodyInv20RB_, d_iBodyInv21RB_, d_iBodyInv22RB_;

		// neighbor list storage
		// (compressed SELL32 for FF, FB, and FR neighbors / CSR for RF, RB, and RR)
		DeviceBuffer<uint16_t> d_nbrCountFF_; // temporary counts written by count-query
		DeviceBuffer<uint16_t> d_nbrCountFB_;
		DeviceBuffer<uint16_t> d_nbrCountFR_;
		DeviceBuffer<uint16_t> d_nbrCountRF_;
		DeviceBuffer<uint16_t> d_nbrCountRB_;
		DeviceBuffer<uint16_t> d_nbrCountRR_;
		DeviceBuffer<uint16_t> d_nbrChunkMaxCountFF_; // per-SELL32-chunk max neighbor count (pitch)
		DeviceBuffer<uint16_t> d_nbrChunkMaxCountFB_;
		DeviceBuffer<uint16_t> d_nbrChunkMaxCountFR_;
		DeviceBuffer<uint32_t> d_nbrChunkBaseIdxFF_; // per-chunk base index used for delta storage
		DeviceBuffer<uint32_t> d_nbrChunkBaseIdxFB_;
		DeviceBuffer<uint32_t> d_nbrChunkBaseIdxFR_;
		DeviceBuffer<uint8_t> d_nbrChunkFormatFF_; // per-chunk storage format
		DeviceBuffer<uint8_t> d_nbrChunkFormatFB_;
		DeviceBuffer<uint8_t> d_nbrChunkFormatFR_;
		DeviceBuffer<uint64_t> d_nbrChunkOffsetsFF_; // per-chunk starting index into the selected format buffer
		DeviceBuffer<uint64_t> d_nbrChunkOffsetsFB_;
		DeviceBuffer<uint64_t> d_nbrChunkOffsetsFR_;
		DeviceBuffer<uint64_t> d_nbrOffsetsRF_;
		DeviceBuffer<uint64_t> d_nbrOffsetsRB_;
		DeviceBuffer<uint64_t> d_nbrOffsetsRR_;
		DeviceBuffer<uint8_t> d_nbrListDelta8FF_; // contiguous delta8 neighbor storage
		DeviceBuffer<uint8_t> d_nbrListDelta8FB_;
		DeviceBuffer<uint8_t> d_nbrListDelta8FR_;
		DeviceBuffer<uint16_t> d_nbrListDelta16FF_; // contiguous delta16 neighbor storage
		DeviceBuffer<uint16_t> d_nbrListDelta16FB_;
		DeviceBuffer<uint16_t> d_nbrListDelta16FR_;
		DeviceBuffer<uint32_t> d_nbrListAbs32FF_; // contiguous abs32 neighbor storage
		DeviceBuffer<uint32_t> d_nbrListAbs32FB_;
		DeviceBuffer<uint32_t> d_nbrListAbs32FR_;
		DeviceBuffer<uint32_t> d_nbrListRF_;
		DeviceBuffer<uint32_t> d_nbrListRB_;
		DeviceBuffer<uint32_t> d_nbrListRR_;
		DeviceBuffer<uint32_t> d_nbrOverflowFlag_; // TODO: use smaller data type?
		uint64_t N_totalFF_ = 0; // actual neighbor count sum over all particles
		uint64_t N_totalFB_ = 0;
		uint64_t N_totalFR_ = 0;
		uint64_t N_totalRF_ = 0;
		uint64_t N_totalRB_ = 0;
		uint64_t N_totalRR_ = 0;
		uint64_t N_packedFF_ = 0; // total compressed storage length in bytes
		uint64_t N_packedFB_ = 0;
		uint64_t N_packedFR_ = 0;
		uint64_t N_delta8FF_ = 0; // compressed delta8 storage length in refs
		uint64_t N_delta8FB_ = 0;
		uint64_t N_delta8FR_ = 0;
		uint64_t N_delta16FF_ = 0; // compressed delta16 storage length in refs
		uint64_t N_delta16FB_ = 0;
		uint64_t N_delta16FR_ = 0;
		uint64_t N_abs32FF_ = 0; // compressed abs32 storage length in refs
		uint64_t N_abs32FB_ = 0;
		uint64_t N_abs32FR_ = 0;
		const float extraAllocCapacityPct_ = 2.0f; // TODO: fine-tune
		DeviceBuffer<uint8_t> d_nsSharedScratch_; // shared scratch arena for neighbor search/query/finalize
		DeviceBuffer<uint32_t> d_reorderIdScratch_;

		// reusable device scratch for particle statistics
		mutable DeviceBuffer<uint32_t> d_nbrCountHist_;
		mutable DeviceBuffer<uint32_t> d_nbrCountHistChunk_;
		mutable DeviceBuffer<GPU::NeighborCountStats> d_nbrCountStats_;
		mutable std::array<GPU::NeighborCountStats, 6> h_nbrCountStats_{};

		// scratch for density stats reduction, ceil(N_F / 256) entries -> one result
		mutable DeviceBuffer<GPU::DensityStats> d_densityStatsPartials_;
		mutable DeviceBuffer<GPU::DensityStats> d_densityStats_;
		mutable GPU::DensityStats h_densityStats_;
		// scratch for velocity stats reduction, ceil(N_F / 256) entries -> one result
		mutable DeviceBuffer<GPU::VelocityStats> d_velocityStatsPartials_;
		mutable DeviceBuffer<GPU::VelocityStats> d_velocityStats_;
		mutable GPU::VelocityStats h_velocityStats_;

		// re-usable host staging buffers for export snapshot packing
		// (reduces per-export allocations and allocator overhead)
		mutable std::vector<uint16_t> h_expNbrCountFF_;
		mutable std::vector<uint16_t> h_expNbrCountFB_;
		mutable std::vector<uint16_t> h_expNbrCountFR_;
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
		mutable std::vector<uint8_t> h_expSurfaceFlagsScratch_;

		// operation benchmarking order
		Bench::BenchOpList benchOps_ = Bench::MakeDefaultSimBenchOps();

		// pack rigid body state metadata into host snapshot (for both CUDA-OpenGL interop path and otherwise)
		void PackVisualRigidBodyState(Sync::VisualSnapshot& snap) const;

		void UpdateRigidParticles();

		// resizes all device buffers when particle counts change
		// (does not preserve data)
		void EnsureCapacity(uint32_t N_newF, uint32_t N_newB, bool force = false)
		{
			const uint32_t N_newChunks = (N_newF == 0u)
				? 0u : ((N_newF - 1u) / GPU::N_SELL_WIDTH) + 1u;
			const uint32_t N_newReorderScratch = std::max(N_newF, N_newB);
			const bool needOverflow = d_nbrOverflowFlag_.Size() != 1;
			const bool needFluid = force || N_newF != N_F_ ||
				d_idF_.Size() != N_newF ||
				d_xPosF_.Size() != N_newF || d_yPosF_.Size() != N_newF || d_zPosF_.Size() != N_newF ||
				d_xVelF_.Size() != N_newF || d_yVelF_.Size() != N_newF || d_zVelF_.Size() != N_newF ||
				d_xVelPredF_.Size() != N_newF || d_yVelPredF_.Size() != N_newF || d_zVelPredF_.Size() != N_newF ||
				d_xAccPressF_.Size() != N_newF || d_yAccPressF_.Size() != N_newF || d_zAccPressF_.Size() != N_newF ||
				d_sourceTermF_.Size() != N_newF || d_diagElemF_.Size() != N_newF ||
				d_pressureF_.Size() != N_newF || d_densityF_.Size() != N_newReorderScratch ||
				d_nbrCountFF_.Size() != N_newF || d_nbrCountFB_.Size() != N_newF || d_nbrCountFR_.Size() != N_newF ||
				d_nbrChunkMaxCountFF_.Size() != N_newChunks || d_nbrChunkMaxCountFB_.Size() != N_newChunks || d_nbrChunkMaxCountFR_.Size() != N_newChunks ||
				d_nbrChunkBaseIdxFF_.Size() != N_newChunks || d_nbrChunkBaseIdxFB_.Size() != N_newChunks || d_nbrChunkBaseIdxFR_.Size() != N_newChunks ||
				d_nbrChunkFormatFF_.Size() != N_newChunks || d_nbrChunkFormatFB_.Size() != N_newChunks || d_nbrChunkFormatFR_.Size() != N_newChunks ||
				d_nbrChunkOffsetsFF_.Size() != static_cast<size_t>(N_newChunks) + 1 ||
				d_nbrChunkOffsetsFB_.Size() != static_cast<size_t>(N_newChunks) + 1 ||
				d_nbrChunkOffsetsFR_.Size() != static_cast<size_t>(N_newChunks) + 1 ||
				d_reorderIdScratch_.Size() != N_newReorderScratch;
			const bool needBoundary = force || N_newB != N_B_ ||
				d_idB_.Size() != N_newB ||
				d_xPosB_.Size() != N_newB || d_yPosB_.Size() != N_newB || d_zPosB_.Size() != N_newB ||
				d_volB_.Size() != N_newB || d_psiB_.Size() != N_newB;
			if (!(needFluid || needBoundary || needOverflow)) { return; }

			N_F_ = N_newF;
			N_B_ = N_newB;
			d_idF_.Resize(N_F_);
			d_xPosF_.Resize(N_F_); d_yPosF_.Resize(N_F_); d_zPosF_.Resize(N_F_);
			d_xVelF_.Resize(N_F_); d_yVelF_.Resize(N_F_); d_zVelF_.Resize(N_F_);
			d_xVelPredF_.Resize(N_F_); d_yVelPredF_.Resize(N_F_); d_zVelPredF_.Resize(N_F_);
			d_xAccPressF_.Resize(N_F_); d_yAccPressF_.Resize(N_F_); d_zAccPressF_.Resize(N_F_);
			d_sourceTermF_.Resize(N_F_); d_diagElemF_.Resize(N_F_);
			d_pressureF_.Resize(N_F_); d_densityF_.Resize(N_newReorderScratch);

			d_idB_.Resize(N_B_);
			d_xPosB_.Resize(N_B_); d_yPosB_.Resize(N_B_); d_zPosB_.Resize(N_B_);
			d_volB_.Resize(N_B_); d_psiB_.Resize(N_B_);
			d_reorderIdScratch_.Resize(N_newReorderScratch);

			d_nbrCountFF_.Resize(N_F_); d_nbrCountFB_.Resize(N_F_); d_nbrCountFR_.Resize(N_F_);
			d_nbrChunkMaxCountFF_.Resize(N_newChunks); d_nbrChunkMaxCountFB_.Resize(N_newChunks); d_nbrChunkMaxCountFR_.Resize(N_newChunks);
			d_nbrChunkBaseIdxFF_.Resize(N_newChunks); d_nbrChunkBaseIdxFB_.Resize(N_newChunks); d_nbrChunkBaseIdxFR_.Resize(N_newChunks);
			d_nbrChunkFormatFF_.Resize(N_newChunks); d_nbrChunkFormatFB_.Resize(N_newChunks); d_nbrChunkFormatFR_.Resize(N_newChunks);
			d_nbrChunkOffsetsFF_.Resize(static_cast<size_t>(N_newChunks) + 1);
			d_nbrChunkOffsetsFB_.Resize(static_cast<size_t>(N_newChunks) + 1);
			d_nbrChunkOffsetsFR_.Resize(static_cast<size_t>(N_newChunks) + 1);
			d_nbrOverflowFlag_.Resize(1);
		}

		void EnsureRigidCapacity(
			uint32_t N_newRB, uint32_t N_newR, bool force = false)
		{
			const bool needRigidBodies = force || N_newRB != N_RB_ ||
				d_idRB_.Size() != N_newRB ||
				d_rigidParticleBeginRB_.Size() != N_newRB || d_rigidParticleCountRB_.Size() != N_newRB ||
				d_xCMRB_.Size() != N_newRB || d_yCMRB_.Size() != N_newRB || d_zCMRB_.Size() != N_newRB ||
				d_xVelRB_.Size() != N_newRB || d_yVelRB_.Size() != N_newRB || d_zVelRB_.Size() != N_newRB ||
				d_xOmegaRB_.Size() != N_newRB || d_yOmegaRB_.Size() != N_newRB || d_zOmegaRB_.Size() != N_newRB ||
				d_xAngMomRB_.Size() != N_newRB || d_yAngMomRB_.Size() != N_newRB || d_zAngMomRB_.Size() != N_newRB ||
				d_xForceRB_.Size() != N_newRB || d_yForceRB_.Size() != N_newRB || d_zForceRB_.Size() != N_newRB ||
				d_xTorqueRB_.Size() != N_newRB || d_yTorqueRB_.Size() != N_newRB || d_zTorqueRB_.Size() != N_newRB ||
				d_a00RB_.Size() != N_newRB || d_a01RB_.Size() != N_newRB || d_a02RB_.Size() != N_newRB ||
				d_a10RB_.Size() != N_newRB || d_a11RB_.Size() != N_newRB || d_a12RB_.Size() != N_newRB ||
				d_a20RB_.Size() != N_newRB || d_a21RB_.Size() != N_newRB || d_a22RB_.Size() != N_newRB ||
				d_massRB_.Size() != N_newRB || d_massInvRB_.Size() != N_newRB ||
				d_iBodyInv00RB_.Size() != N_newRB || d_iBodyInv01RB_.Size() != N_newRB || d_iBodyInv02RB_.Size() != N_newRB ||
				d_iBodyInv10RB_.Size() != N_newRB || d_iBodyInv11RB_.Size() != N_newRB || d_iBodyInv12RB_.Size() != N_newRB ||
				d_iBodyInv20RB_.Size() != N_newRB || d_iBodyInv21RB_.Size() != N_newRB || d_iBodyInv22RB_.Size() != N_newRB;
			const bool needParticlesR = force || N_newR != N_R_ ||
				d_idR_.Size() != N_newR || d_bodyIdR_.Size() != N_newR ||
				d_xPosR_.Size() != N_newR || d_yPosR_.Size() != N_newR || d_zPosR_.Size() != N_newR ||
				d_xRelPosR_.Size() != N_newR || d_yRelPosR_.Size() != N_newR || d_zRelPosR_.Size() != N_newR ||
				d_xVelR_.Size() != N_newR || d_yVelR_.Size() != N_newR || d_zVelR_.Size() != N_newR ||
				d_xForceR_.Size() != N_newR || d_yForceR_.Size() != N_newR || d_zForceR_.Size() != N_newR ||
				d_volR_.Size() != N_newR || d_psiR_.Size() != N_newR ||
				d_nbrCountRF_.Size() != N_newR || d_nbrCountRB_.Size() != N_newR ||
				d_nbrCountRR_.Size() != N_newR ||
				d_nbrOffsetsRF_.Size() != static_cast<size_t>(N_newR) + 1 ||
				d_nbrOffsetsRB_.Size() != static_cast<size_t>(N_newR) + 1 ||
				d_nbrOffsetsRR_.Size() != static_cast<size_t>(N_newR) + 1;
			if (!(needRigidBodies || needParticlesR)) { return; }

			N_RB_ = N_newRB;
			N_R_ = N_newR;
			d_idR_.Resize(N_R_); d_bodyIdR_.Resize(N_R_);
			d_xPosR_.Resize(N_R_); d_yPosR_.Resize(N_R_); d_zPosR_.Resize(N_R_);
			d_xRelPosR_.Resize(N_R_); d_yRelPosR_.Resize(N_R_); d_zRelPosR_.Resize(N_R_);
			d_xVelR_.Resize(N_R_); d_yVelR_.Resize(N_R_); d_zVelR_.Resize(N_R_);
			d_xForceR_.Resize(N_R_); d_yForceR_.Resize(N_R_); d_zForceR_.Resize(N_R_);
			d_volR_.Resize(N_R_); d_psiR_.Resize(N_R_);
			d_nbrCountRF_.Resize(N_R_); d_nbrCountRB_.Resize(N_R_); d_nbrCountRR_.Resize(N_R_);
			d_nbrOffsetsRF_.Resize(static_cast<size_t>(N_R_) + 1);
			d_nbrOffsetsRB_.Resize(static_cast<size_t>(N_R_) + 1);
			d_nbrOffsetsRR_.Resize(static_cast<size_t>(N_R_) + 1);

			d_idRB_.Resize(N_RB_);
			d_rigidParticleBeginRB_.Resize(N_RB_); d_rigidParticleCountRB_.Resize(N_RB_);
			d_xCMRB_.Resize(N_RB_); d_yCMRB_.Resize(N_RB_); d_zCMRB_.Resize(N_RB_);
			d_xVelRB_.Resize(N_RB_); d_yVelRB_.Resize(N_RB_); d_zVelRB_.Resize(N_RB_);
			d_xOmegaRB_.Resize(N_RB_); d_yOmegaRB_.Resize(N_RB_); d_zOmegaRB_.Resize(N_RB_);
			d_xAngMomRB_.Resize(N_RB_); d_yAngMomRB_.Resize(N_RB_); d_zAngMomRB_.Resize(N_RB_);
			d_xForceRB_.Resize(N_RB_); d_yForceRB_.Resize(N_RB_); d_zForceRB_.Resize(N_RB_);
			d_xTorqueRB_.Resize(N_RB_); d_yTorqueRB_.Resize(N_RB_); d_zTorqueRB_.Resize(N_RB_);
			d_a00RB_.Resize(N_RB_); d_a01RB_.Resize(N_RB_); d_a02RB_.Resize(N_RB_);
			d_a10RB_.Resize(N_RB_); d_a11RB_.Resize(N_RB_); d_a12RB_.Resize(N_RB_);
			d_a20RB_.Resize(N_RB_); d_a21RB_.Resize(N_RB_); d_a22RB_.Resize(N_RB_);
			d_massRB_.Resize(N_RB_); d_massInvRB_.Resize(N_RB_);
			d_iBodyInv00RB_.Resize(N_RB_); d_iBodyInv01RB_.Resize(N_RB_); d_iBodyInv02RB_.Resize(N_RB_);
			d_iBodyInv10RB_.Resize(N_RB_); d_iBodyInv11RB_.Resize(N_RB_); d_iBodyInv12RB_.Resize(N_RB_);
			d_iBodyInv20RB_.Resize(N_RB_); d_iBodyInv21RB_.Resize(N_RB_); d_iBodyInv22RB_.Resize(N_RB_);
		}

		// turns temporary count-query output into compressed SELL32 metadata
		// and ensures list-storage capacity for the fill-query.
		void FinalizeNeighborLists_Count();

		void EnsureNsSharedScratchCapacity(const size_t N_requiredBytes)
		{
			if (N_requiredBytes <= d_nsSharedScratch_.Size()) { return; }

			const size_t capacity = N_requiredBytes +
				static_cast<size_t>(Base::ExtraPercent(
					N_requiredBytes,
					extraAllocCapacityPct_));
			d_nsSharedScratch_.Resize(capacity);
		}
	};

} // namespace Sim

#endif
