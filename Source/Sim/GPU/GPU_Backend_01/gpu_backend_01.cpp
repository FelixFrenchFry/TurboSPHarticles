// gpu_backend_01.cpp

// external
#include <glm/vec3.hpp>
#include <cmath>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// internal
#include "Source/Base/helpers.h"
#include "Source/Sim/GPU/gpu_utils.h"
#include "gpu_backend_01.h"
#include "neighbor_search.cuh"
#include "sph_kernels.cuh"



using namespace Sim;

namespace
{
	template <typename T>
	inline Mem::MemUsage MemUsageOf_DeviceBuffer(
		const DeviceBuffer<T>& src) noexcept
	{
		const uint64_t bytes = static_cast<uint64_t>(src.Size()) * sizeof(T);

		return Mem::MemUsage{ bytes, bytes, bytes };
	}
}

SimBackend_GPU01::SimBackend_GPU01(
	Sim::SimParamsMgr& simParamsMgr, Bench::BenchmarksTracker& benchmarksTracker, const SimParams& params)
	: simParamsMgr_{ simParamsMgr }, benchmarksTracker_ { benchmarksTracker }
{
	GPU::Check(cudaStreamCreate(&stream_), "cudaStreamCreate()");

	// select neighbor search method
	if (params.neighborSearchMethod ==
		Base::NeighborSearchMethod::BruteForce)
	{
		neighborSearchStrat_ = std::make_unique<NS_BruteForce_GPU01>(*this);
	}
	else if (params.neighborSearchMethod ==
		Base::NeighborSearchMethod::SpatialHashing_Basic)
	{
		throw std::runtime_error("Unsupported 'NeighborSearchMethod' requested.");
	}
	else if (params.neighborSearchMethod ==
		Base::NeighborSearchMethod::CellLinkedList_Basic)
	{
		neighborSearchStrat_ = std::make_unique<NS_CellLinkedList_Basic_GPU01>(*this);
	}
	else if (params.neighborSearchMethod ==
		Base::NeighborSearchMethod::CellLinkedList_Compact)
	{
		neighborSearchStrat_ = std::make_unique<NS_CellLinkedList_Compact_GPU01>(*this);
	}
	else if (params.neighborSearchMethod ==
		Base::NeighborSearchMethod::BoundVolHier_Linear_LBVH)
	{
		neighborSearchStrat_ = std::make_unique<NS_BoundVolHier_Linear_LBVH_GPU01>(*this);
	}
	else
	{
		throw std::runtime_error("Unsupported 'NeighborSearchMethod' requested.");
	}
}

SimBackend_GPU01::~SimBackend_GPU01()
{
	if (stream_)
	{
		GPU::Check(cudaStreamDestroy(stream_), "cudaStreamDestroy()");
	}
}

void SimBackend_GPU01::SetNeighborSearchStrategy(std::unique_ptr<I_NeighborSearchStrat> strat)
{
	neighborSearchStrat_ = std::move(strat);
}

// TODO: re-implement this to free up memory as soon as copied in!
void SimBackend_GPU01::PullFromUniversalSimState(
	Sync::UniversalSimState& src, const SimParams& params)
{
	auto& F = src.fluidParticles;
	auto& B = src.boundaryParticles;
	N_F_ = F.N_;
	N_B_ = B.N_;
	simStep_ = src.step;

	EnsureCapacity(N_F_, N_B_);

	const bool hasF = N_F_ > 0;
	const bool hasB = N_B_ > 0;

	// copy fluid particle data to device
	if (hasF)
	{
		GPU::Copy_HostToDevice(d_xPosF_.Data(), F.xPos.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_yPosF_.Data(), F.yPos.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_zPosF_.Data(), F.zPos.data(), N_F_, stream_);

		GPU::Copy_HostToDevice(d_xVelF_.Data(), F.xVel.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_yVelF_.Data(), F.yVel.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_zVelF_.Data(), F.zVel.data(), N_F_, stream_);

		GPU::Copy_HostToDevice(d_xAccPressF_.Data(), F.xAccPress.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_yAccPressF_.Data(), F.yAccPress.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_zAccPressF_.Data(), F.zAccPress.data(), N_F_, stream_);

		GPU::Copy_HostToDevice(d_sourceTermF_.Data(), F.sourceTerm.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_pressureF_.Data(), F.pressure.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_densityF_.Data(), F.density.data(), N_F_, stream_);
	}

	// copy boundary particle data to device
	if (hasB)
	{
		GPU::Copy_HostToDevice(d_xPosB_.Data(), B.xPos.data(), N_B_, stream_);
		GPU::Copy_HostToDevice(d_yPosB_.Data(), B.yPos.data(), N_B_, stream_);
		GPU::Copy_HostToDevice(d_zPosB_.Data(), B.zPos.data(), N_B_, stream_);
	}

	// reset neighbor counts for the next construction step
	GPU::MemsetZero(d_nbrCountFF_.Data(), N_F_, stream_);
	GPU::MemsetZero(d_nbrCountFB_.Data(), N_F_, stream_);
	GPU::MemsetZero(d_nbrOverflowFlag_.Data(), 1, stream_);
	N_totalFF_ = 0;
	N_totalFB_ = 0;

	GPU::StreamSync(stream_);
}

// TODO: re-implement this to free up memory as soon as copied out!
void SimBackend_GPU01::PushToUniversalSimState(Sync::UniversalSimState& dst) const
{
	auto& F = dst.fluidParticles;
	auto& B = dst.boundaryParticles;

	// make sure the destination data structures have enough capacity
	dst.ResizeFluid(N_F_);
	dst.ResizeBoundary(N_B_);

	const bool hasF = N_F_ > 0;
	const bool hasB = N_B_ > 0;

	// copy fluid particle data into the host's universal state
	if (hasF)
	{
		GPU::Copy_DeviceToHost(F.xPos.data(), d_xPosF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.yPos.data(), d_yPosF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.zPos.data(), d_zPosF_.Data(), N_F_, stream_);

		GPU::Copy_DeviceToHost(F.xVel.data(), d_xVelF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.yVel.data(), d_yVelF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.zVel.data(), d_zVelF_.Data(), N_F_, stream_);

		GPU::Copy_DeviceToHost(F.xAccPress.data(), d_xAccPressF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.yAccPress.data(), d_yAccPressF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.zAccPress.data(), d_zAccPressF_.Data(), N_F_, stream_);

		GPU::Copy_DeviceToHost(F.sourceTerm.data(), d_sourceTermF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.diagElem.data(), d_diagElemF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.pressure.data(), d_pressureF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.density.data(), d_densityF_.Data(), N_F_, stream_);
	}

	// copy boundary particle data into the host's universal state
	if (hasB)
	{
		GPU::Copy_DeviceToHost(B.xPos.data(), d_xPosB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(B.yPos.data(), d_yPosB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(B.zPos.data(), d_zPosB_.Data(), N_B_, stream_);
	}

	GPU::StreamSync(stream_);

	F.N_ = N_F_;
	B.N_ = N_B_;
}

void SimBackend_GPU01::PackExportSnapshot(
	Sync::ExportSnapshot& snap, const SimParams& simParams) const
{
	(void)simParams;

	// TODO: only export surface particles?

	if (N_F_ == 0)
	{
		snap.simStep = simStep_;
		snap.simTime = simTime_;
		snap.N_F = 0;
		snap.xPosF.clear();
		snap.yPosF.clear();
		snap.zPosF.clear();
		snap.velMagF.clear();

		return;
	}

	// host staging for fluid neighbor list data
	auto& h_nbrCountFF = h_expNbrCountFF_;
	auto& h_nbrCountFB = h_expNbrCountFB_;
	auto& h_nbrOffsetsFF = h_expNbrOffsetsFF_;
	auto& h_nbrListFF = h_expNbrListFF_;

	h_nbrCountFF.resize(N_F_);
	h_nbrCountFB.resize(N_F_);
	h_nbrOffsetsFF.resize(N_F_);
	h_nbrListFF.resize(static_cast<size_t>(N_totalFF_));

	// host staging for fluid positions and velocities
	auto& h_xPosF = h_expXPosF_;
	auto& h_yPosF = h_expYPosF_;
	auto& h_zPosF = h_expZPosF_;
	auto& h_xVelF = h_expXVelF_;
	auto& h_yVelF = h_expYVelF_;
	auto& h_zVelF = h_expZVelF_;

	h_xPosF.resize(N_F_);
	h_yPosF.resize(N_F_);
	h_zPosF.resize(N_F_);
	h_xVelF.resize(N_F_);
	h_yVelF.resize(N_F_);
	h_zVelF.resize(N_F_);

	// launch async device to host copies
	GPU::Copy_DeviceToHost(h_nbrCountFF.data(), d_nbrCountFF_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_nbrCountFB.data(), d_nbrCountFB_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_nbrOffsetsFF.data(), d_nbrOffsetsFF_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_nbrListFF.data(), d_nbrListFF_.Data(),
						   static_cast<size_t>(N_totalFF_), stream_);

	GPU::Copy_DeviceToHost(h_xPosF.data(), d_xPosF_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_yPosF.data(), d_yPosF_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_zPosF.data(), d_zPosF_.Data(), N_F_, stream_);

	GPU::Copy_DeviceToHost(h_xVelF.data(), d_xVelF_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_yVelF.data(), d_yVelF_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_zVelF.data(), d_zVelF_.Data(), N_F_, stream_);

	// wait for all copies to finish
	GPU::StreamSync(stream_);

	// classify surface particles
	auto& surfaceFlags = h_expSurfaceFlagsScratch_;
	surfaceFlags.resize(N_F_);

	const uint32_t N_surface = tbb::parallel_reduce(
		tbb::blocked_range<uint32_t>(0, N_F_), uint32_t{ 0 },
		[&](const tbb::blocked_range<uint32_t>& r, uint32_t localCount)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// particle considered part of the surface
				// if touching boundary OR low fluid neighbor count
				bool surface = h_nbrCountFB[i] > 0 ||
							   h_nbrCountFF[i] < 28;

				// if not surface itself, check its fluid neighbors
				if (!surface)
				{
					const uint64_t startFF = h_nbrOffsetsFF[i];
					const uint32_t N_FF = h_nbrCountFF[i];

					for (uint32_t k = 0; k < N_FF; k++)
					{
						const uint32_t j = h_nbrListFF[static_cast<size_t>(startFF) + k];

						if (h_nbrCountFB[j] > 0 ||
							h_nbrCountFF[j] < 24)
						{
							surface = true;
							break;
						}
					}
				}

				surfaceFlags[i] = static_cast<uint8_t>(surface);
				if (surface) { localCount++; }
			}

			return localCount;
		},
		[](uint32_t a, uint32_t b) -> uint32_t
		{
			return a + b;
		}
	);

	if (N_surface == 0) // no particles to export
	{
		snap.simStep = simStep_;
		snap.simTime = simTime_;
		snap.N_F = 0;
		snap.xPosF.clear();
		snap.yPosF.clear();
		snap.zPosF.clear();
		snap.velMagF.clear();

		return;
	}

	// prepare snapshot buffers
	snap.simStep = simStep_;
	snap.simTime = simTime_;
	snap.N_F = N_surface;

	snap.xPosF.resize(N_surface);
	snap.yPosF.resize(N_surface);
	snap.zPosF.resize(N_surface);
	snap.velMagF.resize(N_surface);

	// copy only surface particles
	uint32_t dst = 0;
	for (uint32_t i = 0; i < N_F_; i++)
	{
		if (!surfaceFlags[i]) { continue; }

		snap.xPosF[dst] = h_xPosF[i];
		snap.yPosF[dst] = h_yPosF[i];
		snap.zPosF[dst] = h_zPosF[i];

		const float vx = h_xVelF[i];
		const float vy = h_yVelF[i];
		const float vz = h_zVelF[i];

		snap.velMagF[dst] = std::sqrt(vx * vx + vy * vy + vz * vz);

		dst++;
	}
}

void SimBackend_GPU01::PackVisualSnapshot(Sync::VisualSnapshot& snap) const
{
	// TODO: add boundary particles

	snap.N_F = N_F_;
	snap.simStep = simStep_;

	if (N_F_ == 0) // empty sim
	{
		snap.posF.clear();
		snap.scalarDataF.clear();

		// TODO: add boundary particles

		return;
	}

	snap.posF.resize(N_F_);

	// host staging for positions
	std::vector<float> h_xPosF(N_F_);
	std::vector<float> h_yPosF(N_F_);
	std::vector<float> h_zPosF(N_F_);

	// launch async device to host copies
	GPU::Copy_DeviceToHost(h_xPosF.data(), d_xPosF_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_yPosF.data(), d_yPosF_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_zPosF.data(), d_zPosF_.Data(), N_F_, stream_);

	// wait for all copies to finish
	GPU::StreamSync(stream_);

	// pack into the visual snapshot's data structures
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				snap.posF[i] = glm::vec3{ h_xPosF[i], h_yPosF[i], h_zPosF[i] };
			}
		},
		tbb::auto_partitioner{}
	);

	FillScalarData(snap, snap.scalarModeF);
}

void SimBackend_GPU01::FillScalarData(
	Sync::VisualSnapshot& snap,
	Base::ColorCodingMode mode) const
{
	if (N_F_ == 0) // empty sim
	{
		snap.scalarDataF.clear();

		// TODO: add boundary particles

		return;
	}

	snap.scalarDataF.resize(N_F_);

	switch (mode)
	{
	case Base::ColorCodingMode::None:
		{
			snap.scalarDataF.clear();
			return;
		}
	case Base::ColorCodingMode::VelocityMag:
		{
			// host staging for velocities
			std::vector<float> h_xVelF(N_F_);
			std::vector<float> h_yVelF(N_F_);
			std::vector<float> h_zVelF(N_F_);

			// launch async device to host copies
			GPU::Copy_DeviceToHost(h_xVelF.data(), d_xVelF_.Data(), N_F_, stream_);
			GPU::Copy_DeviceToHost(h_yVelF.data(), d_yVelF_.Data(), N_F_, stream_);
			GPU::Copy_DeviceToHost(h_zVelF.data(), d_zVelF_.Data(), N_F_, stream_);

			// wait for all copies to finish
			GPU::StreamSync(stream_);

			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0, N_F_),
				[&](const tbb::blocked_range<uint32_t>& r)
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						const float vxi = h_xVelF[i];
						const float vyi = h_yVelF[i];
						const float vzi = h_zVelF[i];

						// magnitude of velocity vector
						snap.scalarDataF[i] = std::sqrt(vxi * vxi
							+ vyi * vyi
							+ vzi * vzi);
					}
				},
				tbb::auto_partitioner{}
			);
			return;
		}
	case Base::ColorCodingMode::Density:
		{
			// copy density data directly into the host snapshot
			GPU::Copy_DeviceToHost(snap.scalarDataF.data(), d_densityF_.Data(), N_F_, stream_);
			GPU::StreamSync(stream_);
			return;
		}
	case Base::ColorCodingMode::Pressure:
		{
			// copy pressure data directly into the host snapshot
			GPU::Copy_DeviceToHost(snap.scalarDataF.data(), d_pressureF_.Data(), N_F_, stream_);
			GPU::StreamSync(stream_);
			return;
		}
	}
	
	SPDLOG_ERROR("Unsupported scalar data mode in backend GPU01");
	snap.scalarDataF.clear();
}

void SimBackend_GPU01::FillMemoryRecord(
	Mem::MemRecordStep& record,
	const SimParams& params) const
{
	using namespace Mem;

	// --- [ particle properties ] -------------------------------------------------------------------

	// fluid
	{
		MemUsage usage{};

		Add(usage, MemUsageOf_DeviceBuffer(d_xPosF_));
		Add(usage, MemUsageOf_DeviceBuffer(d_yPosF_));
		Add(usage, MemUsageOf_DeviceBuffer(d_zPosF_));

		Add(usage, MemUsageOf_DeviceBuffer(d_xVelF_));
		Add(usage, MemUsageOf_DeviceBuffer(d_yVelF_));
		Add(usage, MemUsageOf_DeviceBuffer(d_zVelF_));

		Add(usage, MemUsageOf_DeviceBuffer(d_xVelPredF_));
		Add(usage, MemUsageOf_DeviceBuffer(d_yVelPredF_));
		Add(usage, MemUsageOf_DeviceBuffer(d_zVelPredF_));

		Add(usage, MemUsageOf_DeviceBuffer(d_xAccPressF_));
		Add(usage, MemUsageOf_DeviceBuffer(d_yAccPressF_));
		Add(usage, MemUsageOf_DeviceBuffer(d_zAccPressF_));

		Add(usage, MemUsageOf_DeviceBuffer(d_sourceTermF_));
		Add(usage, MemUsageOf_DeviceBuffer(d_diagElemF_));
		Add(usage, MemUsageOf_DeviceBuffer(d_pressureF_));
		Add(usage, MemUsageOf_DeviceBuffer(d_densityF_));

		// finalize
		Set(record, MemBucket::ParticlePropertiesF, usage);
	}

	// boundary
	{
		MemUsage usage{};

		Add(usage, MemUsageOf_DeviceBuffer(d_xPosB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_yPosB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_zPosB_));

		// finalize
		Set(record, MemBucket::ParticlePropertiesB, usage);
	}

	// --- [ neighbor search data structures ] -------------------------------------------------------

	{
		MemUsage usageF{};
		MemUsage usageB{};
		std::pair<MemUsage, MemUsage> usageFB = neighborSearchStrat_->GetMemUsage(params);

		Add(usageF, usageFB.first);
		Add(usageB, usageFB.second);

		// finalize
		Set(record, MemBucket::Neighbors_SearchDataStructF, usageF);
		Set(record, MemBucket::Neighbors_SearchDataStructB, usageB);
	}

	// --- [ neighbor list storage ] -----------------------------------------------------------------

	{
		MemUsage usageF{};
		MemUsage usageB{};

		Add(usageF, MemUsageOf_DeviceBuffer(d_nbrCountFF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_nbrOffsetsFF_));

		Add(usageB, MemUsageOf_DeviceBuffer(d_nbrCountFB_));
		Add(usageB, MemUsageOf_DeviceBuffer(d_nbrOffsetsFB_));

		// packed CSR list storage uses explicit "used" and "allocated" amounts
		// (used follows current total neighbor count, allocated follows device capacity).
		MemUsage listFF{};
		listFF.usedBytes = static_cast<uint64_t>(N_totalFF_) * sizeof(uint32_t);
		listFF.allocBytes = static_cast<uint64_t>(d_nbrListFF_.Size()) * sizeof(uint32_t);
		listFF.occupiedBytes = listFF.allocBytes;
		Add(usageF, listFF);

		MemUsage listFB{};
		listFB.usedBytes = static_cast<uint64_t>(N_totalFB_) * sizeof(uint32_t);
		listFB.allocBytes = static_cast<uint64_t>(d_nbrListFB_.Size()) * sizeof(uint32_t);
		listFB.occupiedBytes = listFB.allocBytes;
		Add(usageB, listFB);

		// shared overflow flag is accounted in the fluid-side bucket
		// to avoid double-counting.
		Add(usageF, MemUsageOf_DeviceBuffer(d_nbrOverflowFlag_));

		// finalize
		Set(record, MemBucket::Neighbors_ListStorageF, usageF);
		Set(record, MemBucket::Neighbors_ListStorageB, usageB);
	}

	// --- [ misc ] ----------------------------------------------------------------------------------

	{
		MemUsage usage{};

		// (only includes VRAM buffers, not accounting for RAM)
		Add(usage, MemUsageOf_DeviceBuffer(d_workBuffer_));

		// finalize
		Set(record, MemBucket::Misc, usage);
	}
}

void SimBackend_GPU01::FillStatisticsRecord(
	Stats::StatRecordStep& record,
	const SimParams& simParams) const
{
	using namespace Stats;

	const uint32_t NF = N_F_;
	const uint32_t NB = N_B_;

	Set(record, StatMetric::NumF, static_cast<float>(NF));
	Set(record, StatMetric::NumB, static_cast<float>(NB));

	const float invNF = NF ? 1.0f / static_cast<float>(NF) : 0.0f;

	// --- [ particle counts ] -----------------------------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Particles)])
	{
		benchmarksTracker_.StartBench({ Bench::BenchOp::Statistics_Particles });

		float sumFF = 0.0f;
		float nbrFF_p95 = 0.0f;
		float nbrFF_p99 = 0.0f;
		float minFF = std::numeric_limits<float>::max();
		float maxFF = 0.0f;

		float sumFB = 0.0f;
		float nbrFB_p95 = 0.0f;
		float nbrFB_p99 = 0.0f;
		float minFB = std::numeric_limits<float>::max();
		float maxFB = 0.0f;

		if (NF > 0)
		{
			
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		/*
		if (NF > 0)
		{
			struct NeighStatsLocal
			{
				float sumFF;
				float minFF;
				float maxFF;
				float sumFB;
				float minFB;
				float maxFB;
			};

			NeighStatsLocal globalNeigh = tbb::parallel_reduce(
				tbb::blocked_range<uint32_t>(0, NF),
				NeighStatsLocal
				{
					.sumFF = 0.0f,
					.minFF = std::numeric_limits<float>::max(),
					.maxFF = 0.0f,
					.sumFB = 0.0f,
					.minFB = std::numeric_limits<float>::max(),
					.maxFB = 0.0f
				},
				[&](const tbb::blocked_range<uint32_t>& r, NeighStatsLocal local) -> NeighStatsLocal
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						const float N_FF = static_cast<float>(neighborLists_->CountFF(i));
						const float N_FB = static_cast<float>(neighborLists_->CountFB(i));

						local.sumFF += N_FF;
						if (N_FF < local.minFF) [[unlikely]] { local.minFF = N_FF; }
						if (N_FF > local.maxFF) [[unlikely]] { local.maxFF = N_FF; }

						local.sumFB += N_FB;
						if (N_FB < local.minFB) [[unlikely]] { local.minFB = N_FB; }
						if (N_FB > local.maxFB) [[unlikely]] { local.maxFB = N_FB; }
					}
					return local;
				},
				[](const NeighStatsLocal& a, const NeighStatsLocal& b) -> NeighStatsLocal
				{
					NeighStatsLocal r;

					r.sumFF = a.sumFF + b.sumFF;
					r.minFF = std::min(a.minFF, b.minFF);
					r.maxFF = std::max(a.maxFF, b.maxFF);
					r.sumFB = a.sumFB + b.sumFB;
					r.minFB = std::min(a.minFB, b.minFB);
					r.maxFB = std::max(a.maxFB, b.maxFB);

					return r;
				},
				tbb::auto_partitioner{}
			);

			sumFF = globalNeigh.sumFF;
			minFF = globalNeigh.minFF;
			maxFF = globalNeigh.maxFF;
			sumFB = globalNeigh.sumFB;
			minFB = globalNeigh.minFB;
			maxFB = globalNeigh.maxFB;
		}
		if (NF == 0) // reasonable defaults for empty sim
		{
			minFF = 0.0f;
			minFB = 0.0f;
		}

		if (NF > 0) // determine percentiles
		{
			// reuse workBuffer_ for neighbor counts
			workBuffer_.resize(NF);

			// FF
			for (uint32_t i = 0; i < NF; i++)
			{
				workBuffer_[i] = static_cast<float>(neighborLists_->CountFF(i));
			}

			{
				// indices of the 95th and 99th percentiles
				const size_t i95 = (95ull * (NF - 1)) / 100ull;
				const size_t i99 = (99ull * (NF - 1)) / 100ull;

				auto begin = workBuffer_.begin();
				auto end = workBuffer_.end();
				auto nth95 = begin + i95;
				auto nth99 = begin + i99;

				// select p95 within the full range
				std::nth_element(begin, nth95, end);
				nbrFF_p95 = *nth95;

				if (i95 == i99) // handle same index edge case
				{
					nbrFF_p99 = *nth95;
				}
				else
				{
					// select p99 only within the upper part
					std::nth_element(nth95, nth99, end);
					nbrFF_p99 = *nth99;
				}
			}

			// FB
			for (uint32_t i = 0; i < NF; ++i)
			{
				workBuffer_[i] = static_cast<float>(neighborLists_->CountFB(i));
			}

			{
				// indices of the 95th and 99th percentiles
				const size_t i95 = (95ull * (NF - 1)) / 100ull;
				const size_t i99 = (99ull * (NF - 1)) / 100ull;

				auto begin = workBuffer_.begin();
				auto end = workBuffer_.end();
				auto nth95 = begin + i95;
				auto nth99 = begin + i99;

				// select p95 within the full range
				std::nth_element(begin, nth95, end);
				nbrFB_p95 = *nth95;

				if (i95 == i99) // handle same index edge case
				{
					nbrFB_p99 = *nth95;
				}
				else
				{
					// select p99 only within the upper part
					std::nth_element(nth95, nth99, end);
					nbrFB_p99 = *nth99;
				}
			}
		}
		else // reasonable defaults for empty sim
		{
			nbrFF_p95 = nbrFF_p99 = 0.0f;
			nbrFB_p95 = nbrFB_p99 = 0.0f;
		}

		Set(record, StatMetric::NumNbrFF_Avg, sumFF * invNF);
		Set(record, StatMetric::NumNbrFF_P95, nbrFF_p95);
		Set(record, StatMetric::NumNbrFF_P99, nbrFF_p99);
		Set(record, StatMetric::NumNbrFF_Min, minFF);
		Set(record, StatMetric::NumNbrFF_Max, maxFF);

		Set(record, StatMetric::NumNbrFB_Avg, sumFB * invNF);
		Set(record, StatMetric::NumNbrFB_P95, nbrFB_p95);
		Set(record, StatMetric::NumNbrFB_P99, nbrFB_p99);
		Set(record, StatMetric::NumNbrFB_Min, minFB);
		Set(record, StatMetric::NumNbrFB_Max, maxFB);
		*/

		benchmarksTracker_.StopBench({ Bench::BenchOp::Statistics_Particles });
	}
}

void SimBackend_GPU01::ExecuteNeighborSearch(const SimParams& params, bool bench)
{
	if (N_F_ == 0)
	{
		N_totalFF_ = 0;
		N_totalFB_ = 0;
		return;
	}

	GPU01::ReloadConstants(params, N_F_, N_B_, stream_);

	if (bench)
	{
		benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Construction });
	}
	neighborSearchStrat_->Construct(params);
	if (bench)
	{
		benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Construction });
	}

	if (bench)
	{
		benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Query });
	}
	neighborSearchStrat_->Query_Count(params);
	FinalizeNeighborLists_Count();
	neighborSearchStrat_->Query_Fill(params);
	if (bench)
	{
		benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Query });
	}
}

void SimBackend_GPU01::ExecuteFullSimStep(const SimParams& params, uint32_t N_steps)
{
	if (N_F_ == 0) // empty sim
	{
		simStep_ += N_steps;
		simTime_ += N_steps * params.timeStep;

		return;
	}

	for (int64_t step = 0; step < N_steps; step++)
	{
		GPU01::ReloadConstants(params, N_F_, N_B_, stream_);

		// --- [ neighbor search ] ----------------------------------------------------------------------

		benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Construction });
		neighborSearchStrat_->Construct(params);
		benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Construction });

		// --- [ particle data re-ordering ] -----------------------------------------------------------

		const int64_t simStep = simStep_ + step;
		if (params.dataReOrderInterval > 0 &&
			simStep % static_cast<int64_t>(params.dataReOrderInterval) == 0)
		{
			benchmarksTracker_.StartBench({ Bench::BenchOp::Data_Ordering });
			neighborSearchStrat_->DataReOrder(params);
			benchmarksTracker_.StopBench({ Bench::BenchOp::Data_Ordering });
		}

		benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Query });
		neighborSearchStrat_->Query_Count(params);
		FinalizeNeighborLists_Count();
		neighborSearchStrat_->Query_Fill(params);
		benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Query });

		// --- [ density computations ] -----------------------------------------------------------------

		benchmarksTracker_.StartBench({ Bench::BenchOp::Density_Solving });
		GPU01::Launch_Density(
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
			d_nbrOffsetsFF_.Data(), d_nbrOffsetsFB_.Data(),
			d_nbrListFF_.Data(), d_nbrListFB_.Data(),
			N_F_, N_B_, stream_,
			d_densityF_.Data());

		GPU::StreamSync(stream_);
		benchmarksTracker_.StopBench({ Bench::BenchOp::Density_Solving });

		// --- [ non-pressure accelerations ] -----------------------------------------------------------

		benchmarksTracker_.StartBench({ Bench::BenchOp::Viscosity_Acceleration, Bench::BenchOp::External_Acceleration });
		GPU01::Launch_NonPressureAcceleration(
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
			d_xVelF_.Data(), d_yVelF_.Data(), d_zVelF_.Data(),
			d_densityF_.Data(),
			d_nbrOffsetsFF_.Data(), d_nbrOffsetsFB_.Data(),
			d_nbrListFF_.Data(), d_nbrListFB_.Data(),
			N_F_, N_B_, stream_,
			d_xVelPredF_.Data(), d_yVelPredF_.Data(), d_zVelPredF_.Data());

		GPU::StreamSync(stream_);
		benchmarksTracker_.StopBench({ Bench::BenchOp::Viscosity_Acceleration, Bench::BenchOp::External_Acceleration });

		// --- [ source-term (density invariance) and diagonal elements ] -------------------------------

		benchmarksTracker_.StartBench({ Bench::BenchOp::Pressure_SourceTerm, Bench::BenchOp::Pressure_DiagonalElem });
		GPU01::Launch_SourceTermAndDiagonalElementAndPressureJacobiInit(
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
			d_xVelPredF_.Data(), d_yVelPredF_.Data(), d_zVelPredF_.Data(),
			d_densityF_.Data(),
			d_nbrOffsetsFF_.Data(), d_nbrOffsetsFB_.Data(),
			d_nbrListFF_.Data(), d_nbrListFB_.Data(),
			N_F_, N_B_, stream_,
			d_sourceTermF_.Data(), d_diagElemF_.Data(), d_pressureF_.Data());

		GPU::StreamSync(stream_);
		benchmarksTracker_.StopBench({ Bench::BenchOp::Pressure_SourceTerm, Bench::BenchOp::Pressure_DiagonalElem });

		// --- [ relaxed jacobi solver for pressure ] ---------------------------------------------------

		benchmarksTracker_.StartBench({ Bench::BenchOp::Pressure_JacobiSolver });

		// rest solver-related variables
		N_solverIterCount_ = 1;

		do
		{
			// [ loop 1: pressure acceleration ] --------------------------------------------------------
			GPU01::Launch_PressureAcceleration(
				d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
				d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
				d_pressureF_.Data(),
				d_nbrOffsetsFF_.Data(), d_nbrOffsetsFB_.Data(),
				d_nbrListFF_.Data(), d_nbrListFB_.Data(),
				N_F_, N_B_, stream_,
				d_xAccPressF_.Data(), d_yAccPressF_.Data(), d_zAccPressF_.Data());

			GPU::StreamSync(stream_);

			// [ loop 2: relaxed jacobi solver ] --------------------------------------------------------
			GPU01::Launch_PressureRelaxedJacobiSolver(
				d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
				d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
				d_xAccPressF_.Data(), d_yAccPressF_.Data(), d_zAccPressF_.Data(),
				d_sourceTermF_.Data(), d_diagElemF_.Data(),
				d_nbrOffsetsFF_.Data(), d_nbrOffsetsFB_.Data(),
				d_nbrListFF_.Data(), d_nbrListFB_.Data(),
				N_F_, N_B_, stream_,
				d_pressureF_.Data());

			GPU::StreamSync(stream_);

			N_solverIterCount_++;
		} while (!TerminationConditionSatisfied(params));
		benchmarksTracker_.StopBench({ Bench::BenchOp::Pressure_JacobiSolver });

		// --- [ finalize pressure acceleration for integration ] ---------------------------------------

		benchmarksTracker_.StartBench({ Bench::BenchOp::Pressure_Acceleration });
		GPU01::Launch_PressureAcceleration(
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
			d_pressureF_.Data(),
			d_nbrOffsetsFF_.Data(), d_nbrOffsetsFB_.Data(),
			d_nbrListFF_.Data(), d_nbrListFB_.Data(),
			N_F_, N_B_, stream_,
			d_xAccPressF_.Data(), d_yAccPressF_.Data(), d_zAccPressF_.Data());

		GPU::StreamSync(stream_);
		benchmarksTracker_.StopBench({ Bench::BenchOp::Pressure_Acceleration });

		// --- [ velocity and position updates ] --------------------------------------------------------

		benchmarksTracker_.StartBench({ Bench::BenchOp::VelocityUpdate, Bench::BenchOp::PositionUpdate });
		GPU01::Launch_Integration(
			d_xVelPredF_.Data(), d_yVelPredF_.Data(), d_zVelPredF_.Data(),
			d_xAccPressF_.Data(), d_yAccPressF_.Data(), d_zAccPressF_.Data(),
			N_F_, N_B_, stream_,
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xVelF_.Data(), d_yVelF_.Data(), d_zVelF_.Data());

		GPU::StreamSync(stream_);
		benchmarksTracker_.StopBench({ Bench::BenchOp::VelocityUpdate, Bench::BenchOp::PositionUpdate });
	}

	simStep_ += N_steps;
	simTime_ += N_steps * params.timeStep;
}

bool SimBackend_GPU01::TerminationConditionSatisfied(const SimParams& params) const
{
	// check min/max iteration condition for potential early-out
	if (N_solverIterCount_ < params.minSolverIters) { return false; }
	if (N_solverIterCount_ >= params.maxSolverIters) { return true; }

	bool satisfied = false;

	// check actual termination condition
	switch (params.solverTerminationCond)
	{
	case Base::SolverTerminationCond::FixedIters:
		if (N_solverIterCount_ >= params.fixedIters)
		{
			satisfied = true;
		}
		break;

	case Base::SolverTerminationCond::RelDensityError_Raw:
	case Base::SolverTerminationCond::RelDensityError_Pos:
	case Base::SolverTerminationCond::RelDensityError_Abs:
	{
		// TODO: implement
		static bool warnedNotImplemented = false;
		if (!warnedNotImplemented)
		{
			SPDLOG_WARN("GPU01 solver termination condition '{}' is currently not implemented. "
				"Falling back to maxSolverIters.",
				Base::ToCString(params.solverTerminationCond));
			warnedNotImplemented = true;
		}
	}
		break;

	default:
		// unknown enum value
		static bool warnedUnknown = false;
		if (!warnedUnknown)
		{
			SPDLOG_WARN("GPU01 solver termination condition '<unknown>' is currently not implemented. "
				"Falling back to maxSolverIters.");
			warnedUnknown = true;
		}
		break;
	}

	return satisfied;
}
