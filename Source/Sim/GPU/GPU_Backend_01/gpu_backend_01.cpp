// gpu_backend_01.cpp

// external
#include <glm/vec3.hpp>
#include <cmath>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>

// internal
#include "Source/Base/helpers.h"
#include "Source/Sim/GPU/gpu_utils.h"
#include "gpu_backend_01.h"
#include "neighbor_search.cuh"
#include "sph_kernels.cuh"



using namespace Sim;

SimBackend_GPU01::SimBackend_GPU01(Bench::BenchmarksMgr& benchmarksMgr,
								   const SimParams& params)
	: benchmarksMgr_{ benchmarksMgr }
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

void SimBackend_GPU01::PullFromUniversalSimState(const Sync::UniversalSimState& src)
{
	const auto& F = src.fluidParticles;
	const auto& B = src.boundaryParticles;
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

		GPU::Copy_HostToDevice(d_xAccF_.Data(), F.xAcc.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_yAccF_.Data(), F.yAcc.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_zAccF_.Data(), F.zAcc.data(), N_F_, stream_);

		GPU::Copy_HostToDevice(d_xAccPressF_.Data(), F.xAccPress.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_yAccPressF_.Data(), F.yAccPress.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_zAccPressF_.Data(), F.zAccPress.data(), N_F_, stream_);

		GPU::Copy_HostToDevice(d_xAccViscF_.Data(), F.xAccVisc.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_yAccViscF_.Data(), F.yAccVisc.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_zAccViscF_.Data(), F.zAccVisc.data(), N_F_, stream_);

		GPU::Copy_HostToDevice(d_xAccExtF_.Data(), F.xAccExt.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_yAccExtF_.Data(), F.yAccExt.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_zAccExtF_.Data(), F.zAccExt.data(), N_F_, stream_);

		GPU::Copy_HostToDevice(d_sourceTermF_.Data(), F.sourceTerm.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_pressureF_.Data(), F.pressure.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_densityF_.Data(), F.density.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_idF_.Data(), F.id.data(), N_F_, stream_);
	}

	// copy boundary particle data to device
	if (hasB)
	{
		GPU::Copy_HostToDevice(d_xPosB_.Data(), B.xPos.data(), N_B_, stream_);
		GPU::Copy_HostToDevice(d_yPosB_.Data(), B.yPos.data(), N_B_, stream_);
		GPU::Copy_HostToDevice(d_zPosB_.Data(), B.zPos.data(), N_B_, stream_);

		GPU::Copy_HostToDevice(d_xVelB_.Data(), B.xVel.data(), N_B_, stream_);
		GPU::Copy_HostToDevice(d_yVelB_.Data(), B.yVel.data(), N_B_, stream_);
		GPU::Copy_HostToDevice(d_zVelB_.Data(), B.zVel.data(), N_B_, stream_);

		GPU::Copy_HostToDevice(d_idB_.Data(), B.id.data(), N_B_, stream_);
	}

	// reset neighbor counts for the next construction step
	GPU::MemsetZero(d_neighCountFF_.Data(), N_F_, stream_);
	GPU::MemsetZero(d_neighCountFB_.Data(), N_F_, stream_);
	GPU::MemsetZero(d_neighOverflowFlag_.Data(), 1, stream_);

	GPU::StreamSync(stream_);

	//SPDLOG_INFO("Pulled universal sim state with {} fluid and "
	//			"{} boundary particles into data structures of backend GPU01",
	//			N_F_, N_B_);
}

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

		GPU::Copy_DeviceToHost(F.xAcc.data(), d_xAccF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.yAcc.data(), d_yAccF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.zAcc.data(), d_zAccF_.Data(), N_F_, stream_);

		GPU::Copy_DeviceToHost(F.xAccPress.data(), d_xAccPressF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.yAccPress.data(), d_yAccPressF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.zAccPress.data(), d_zAccPressF_.Data(), N_F_, stream_);

		GPU::Copy_DeviceToHost(F.xAccVisc.data(), d_xAccViscF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.yAccVisc.data(), d_yAccViscF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.zAccVisc.data(), d_zAccViscF_.Data(), N_F_, stream_);

		GPU::Copy_DeviceToHost(F.xAccExt.data(), d_xAccExtF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.yAccExt.data(), d_yAccExtF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.zAccExt.data(), d_zAccExtF_.Data(), N_F_, stream_);

		GPU::Copy_DeviceToHost(F.sourceTerm.data(), d_sourceTermF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.diagElem.data(), d_diagElemF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.sysVecElem.data(), d_sysVecElemF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.pressure.data(), d_pressureF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.density.data(), d_densityF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.id.data(), d_idF_.Data(), N_F_, stream_);
	}

	// copy boundary particle data into the host's universal state
	if (hasB)
	{
		GPU::Copy_DeviceToHost(B.xPos.data(), d_xPosB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(B.yPos.data(), d_yPosB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(B.zPos.data(), d_zPosB_.Data(), N_B_, stream_);

		GPU::Copy_DeviceToHost(B.xVel.data(), d_xVelB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(B.yVel.data(), d_yVelB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(B.zVel.data(), d_zVelB_.Data(), N_B_, stream_);

		GPU::Copy_DeviceToHost(B.id.data(), d_idB_.Data(), N_B_, stream_);
	}

	GPU::StreamSync(stream_);

	F.N_ = N_F_;
	B.N_ = N_B_;

	//SPDLOG_INFO("Pushed from data structures of backend GPU01 with "
	//			"{} fluid and {} boundary particles to universal sim state",
	//			N_F_, N_B_);
}

void SimBackend_GPU01::PackExportSnapshot(
	Sync::ExportSnapshot& snap, const SimParams& simParams) const
{
	// TODO: implement
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

// TODO: add flags to enable/disable certain stats?
void SimBackend_GPU01::FillStatisticsRecord(Stats::StatRecordStep& record,
											const SimParams& simParams) const
{
	using namespace Stats;

	// TODO: implement
}

void SimBackend_GPU01::ExecuteSimStep(const SimParams& params, uint32_t N_steps)
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

		// --- [ neighbor search ] ---------------------------------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Construction });
		neighborSearchStrat_->Construct(params);
		benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Construction });

		benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Query });
		neighborSearchStrat_->Query(params);
		benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Query });

		// --- [ density computations ] ----------------------------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Density_Solving });
		GPU01::Launch_Density(
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
			d_neighCountFF_.Data(), d_neighCountFB_.Data(),
			d_neighListFF_.Data(), d_neighListFB_.Data(),
			N_F_, N_B_, N_MAX_FF_, N_MAX_FB_, stream_, 
			d_densityF_.Data());

		GPU::StreamSync(stream_);
		benchmarksMgr_.StopBench({ Bench::BenchOp::Density_Solving });

		// --- [ non-pressure accelerations ] ----------------------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Viscosity_Acceleration,
										Bench::BenchOp::External_Acceleration });
		GPU01::Launch_NonPressureAcceleration(
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
			d_xVelF_.Data(), d_yVelF_.Data(), d_zVelF_.Data(),
			d_densityF_.Data(),
			d_neighCountFF_.Data(), d_neighCountFB_.Data(),
			d_neighListFF_.Data(), d_neighListFB_.Data(),
			N_F_, N_B_, N_MAX_FF_, N_MAX_FB_, stream_,
			d_xVelPredF_.Data(), d_yVelPredF_.Data(), d_zVelPredF_.Data(),
			d_xAccViscF_.Data(), d_yAccViscF_.Data(), d_zAccViscF_.Data(),
			d_xAccExtF_.Data(), d_yAccExtF_.Data(), d_zAccExtF_.Data());

		GPU::StreamSync(stream_);
		benchmarksMgr_.StopBench({ Bench::BenchOp::Viscosity_Acceleration,
									   Bench::BenchOp::External_Acceleration });

		// --- [ source-term (density invariance) and diagonal elements ] ------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_SourceTerm,
										Bench::BenchOp::Pressure_DiagonalElem });
		GPU01::Launch_SourceTermAndDiagonalElementAndPressureJacobiInit(
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
			d_xVelPredF_.Data(), d_yVelPredF_.Data(), d_zVelPredF_.Data(),
			d_densityF_.Data(),
			d_neighCountFF_.Data(), d_neighCountFB_.Data(),
			d_neighListFF_.Data(), d_neighListFB_.Data(),
			N_F_, N_B_, N_MAX_FF_, N_MAX_FB_, stream_,
			d_sourceTermF_.Data(), d_diagElemF_.Data(), d_pressureF_.Data());

		GPU::StreamSync(stream_);
		benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_SourceTerm,
									   Bench::BenchOp::Pressure_DiagonalElem });

		// --- [ relaxed jacobi solver for pressure ] --------------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_JacobiSolver });
		int cnt = 0;
		while (cnt < 2)
		{
			// [ loop 1: pressure acceleration ] -------------------------------
			GPU01::Launch_PressureAcceleration(
				d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
				d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
				d_pressureF_.Data(),
				d_neighCountFF_.Data(), d_neighCountFB_.Data(),
				d_neighListFF_.Data(), d_neighListFB_.Data(),
				N_F_, N_B_, N_MAX_FF_, N_MAX_FB_, stream_,
				d_xAccPressF_.Data(), d_yAccPressF_.Data(), d_zAccPressF_.Data());

			GPU::StreamSync(stream_);

			// [ loop 2: relaxed jacobi solver ] -------------------------------
			GPU01::Launch_PressureRelaxedJacobiSolver(
				d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
				d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
				d_xAccPressF_.Data(), d_yAccPressF_.Data(), d_zAccPressF_.Data(),
				d_sourceTermF_.Data(), d_diagElemF_.Data(),
				d_neighCountFF_.Data(), d_neighCountFB_.Data(),
				d_neighListFF_.Data(), d_neighListFB_.Data(),
				N_F_, N_B_, N_MAX_FF_, N_MAX_FB_, stream_,
				d_sysVecElemF_.Data(), d_pressureF_.Data());

			GPU::StreamSync(stream_);

			cnt++;
		}
		benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_JacobiSolver });

		// --- [ finalize pressure acceleration for integration ] --------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_Acceleration });
		GPU01::Launch_PressureAcceleration(
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
			d_pressureF_.Data(),
			d_neighCountFF_.Data(), d_neighCountFB_.Data(),
			d_neighListFF_.Data(), d_neighListFB_.Data(),
			N_F_, N_B_, N_MAX_FF_, N_MAX_FB_, stream_,
			d_xAccPressF_.Data(), d_yAccPressF_.Data(), d_zAccPressF_.Data());

		GPU::StreamSync(stream_);
		benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_Acceleration });

		// --- [ velocity and position updates ] -------------------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::VelocityUpdate,
										Bench::BenchOp::PositionUpdate });
		GPU01::Launch_Integration(
			d_xVelPredF_.Data(), d_yVelPredF_.Data(), d_zVelPredF_.Data(),
			d_xAccPressF_.Data(), d_yAccPressF_.Data(), d_zAccPressF_.Data(),
			N_F_, N_B_, stream_,
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xVelF_.Data(), d_yVelF_.Data(), d_zVelF_.Data());

		GPU::StreamSync(stream_);
		benchmarksMgr_.StopBench({ Bench::BenchOp::VelocityUpdate,
									   Bench::BenchOp::PositionUpdate });
	}

	simStep_ += N_steps;
	simTime_ += N_steps * params.timeStep;
}
