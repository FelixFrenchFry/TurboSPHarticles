// sph_solver.cpp

// external
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/partitioner.h>

// internal
#include "Source/Base/helpers.h"
#include "neighbor_lists.h"
#include "neighbor_search.h"
#include "cpu_backend_01.h"
#include "sph_solver.h"

#if defined(_MSC_VER)
	#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
	#define RESTRICT __restrict__
#else
	#define RESTRICT
#endif



constexpr float PI = std::numbers::pi_v<float>;
using namespace Sim;

void SimBackend_CPU01::Step_IISPH(const SimParams& params)
{
	// --- [ temporary pre-computed constants ] --------------------------------

	// misc hoisted constants
	const float timeStep = params.timeStep;
	const float sizeInvF = params.particleSizeInvF;
	const float sizeSquF = params.particleSizeF * params.particleSizeF;
	const float kernelRadiusSqu = params.kernelRadiusSqu;
	const float cubicSplineAlpha = static_cast<float>(
		(1.0 / (4.0 * static_cast<double>(PI))) * static_cast<double>(sizeInvF) *
		static_cast<double>(sizeInvF) * static_cast<double>(sizeInvF));
	const float restDensitySquInvF = 1.0f / (params.restDensityF * params.restDensityF);
	const float massF = params.massF;
	const float massB = params.massB;

	// restricted pointers for SoA arrays to support compiler optimizations
	float* RESTRICT xPosF = xPosF_.data();
	float* RESTRICT yPosF = yPosF_.data();
	float* RESTRICT zPosF = zPosF_.data();
	float* RESTRICT xPosB = xPosB_.data();
	float* RESTRICT yPosB = yPosB_.data();
	float* RESTRICT zPosB = zPosB_.data();
	float* RESTRICT xVelF = xVelF_.data();
	float* RESTRICT yVelF = yVelF_.data();
	float* RESTRICT zVelF = zVelF_.data();
	float* RESTRICT xVelPredF = xVelPredF_.data();
	float* RESTRICT yVelPredF = yVelPredF_.data();
	float* RESTRICT zVelPredF = zVelPredF_.data();
	float* RESTRICT xAccF = xAccF_.data();
	float* RESTRICT yAccF = yAccF_.data();
	float* RESTRICT zAccF = zAccF_.data();
	float* RESTRICT xAccPressF = xAccPressF_.data();
	float* RESTRICT yAccPressF = yAccPressF_.data();
	float* RESTRICT zAccPressF = zAccPressF_.data();
	float* RESTRICT xAccViscF = xAccViscF_.data();
	float* RESTRICT yAccViscF = yAccViscF_.data();
	float* RESTRICT zAccViscF = zAccViscF_.data();
	float* RESTRICT xAccExtF = xAccExtF_.data();
	float* RESTRICT yAccExtF = yAccExtF_.data();
	float* RESTRICT zAccExtF = zAccExtF_.data();
	float* RESTRICT sourceTermF = sourceTermF_.data();
	float* RESTRICT diagElemF = diagElemF_.data();
	float* RESTRICT sysVecElemF = sysVecElemF_.data();
	float* RESTRICT pressureF = pressureF_.data();
	float* RESTRICT densityF = densityF_.data();
	I_NeighborLists& neighborLists = *neighborLists_;

	// --- [ neighbor search ] -------------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Construction });
	neighborSearchStrat_->Construct(params);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Construction });

	// --- [ neighbor search query ] -------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Query });
	neighborSearchStrat_->Query(params);
	neighborLists_->Finalize();
	benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Query });

	// --- [ density computations ] --------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Density_Solving });

	CPU01::Density(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		neighborLists,
		sizeInvF, kernelRadiusSqu, cubicSplineAlpha, massF, massB, N_F_,
		densityF);

	benchmarksMgr_.StopBench({ Bench::BenchOp::Density_Solving });

	// --- [ non-pressure accelerations ] --------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Viscosity_Acceleration,
									Bench::BenchOp::External_Acceleration });

	CPU01::NonPressureAcceleration(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		xVelF, yVelF, zVelF,
		densityF,
		neighborLists,
		timeStep, sizeInvF, sizeSquF, kernelRadiusSqu, cubicSplineAlpha,
		params.restDensityB, massF, massB, params.viscosityF, params.viscosityB,
		params.gravity.x, params.gravity.y, params.gravity.z, N_F_,
		xAccViscF, yAccViscF, zAccViscF,
		xAccExtF, yAccExtF, zAccExtF,
		xVelPredF, yVelPredF, zVelPredF);

	benchmarksMgr_.StopBench({ Bench::BenchOp::Viscosity_Acceleration,
								   Bench::BenchOp::External_Acceleration });

	// --- [ source-term (density invariance) and diagonal elements ] ----------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_SourceTerm,
									Bench::BenchOp::Pressure_DiagonalElem });

	CPU01::SourceTermAndDiagonalElement(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		xVelPredF, yVelPredF, zVelPredF,
		densityF,
		neighborLists,
		timeStep, sizeInvF, kernelRadiusSqu, cubicSplineAlpha,
		params.restDensityF, restDensitySquInvF,
		massF, massB, params.gammaPress, N_F_,
		sourceTermF,
		diagElemF);

	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_SourceTerm,
								   Bench::BenchOp::Pressure_DiagonalElem });

	// --- [ relaxed Jacobi solver for pressure ] ------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_JacobiSolver });

	// initial pressure
	CPU01::PressureJacobiInit(
		diagElemF,
		sourceTermF,
		params.jacobiOmega, N_F_,
		pressureF);

	// rest solver-related variables
	N_solverIterCount_ = 1;
	avgRelDensityErrorF_raw_ = 0.0f;
	avgRelDensityErrorF_pos_ = 0.0f;
	avgRelDensityErrorF_abs_ = 0.0f;

	do
	{
		// compute pressure acceleration from current pressure field
		CPU01::PressureAcceleration(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			pressureF,
			neighborLists,
			sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensityF,
			massF, massB, params.gammaPress, N_F_,
			xAccPressF, yAccPressF, zAccPressF);

		// update system vector and pressure
		CPU01::PressureRelaxedJacobiSolver(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			xAccPressF, yAccPressF, zAccPressF,
			sourceTermF, diagElemF,
			neighborLists,
			timeStep, sizeInvF, kernelRadiusSqu, cubicSplineAlpha,
			params.restDensityF, massF, massB, params.jacobiOmega, N_F_,
			sysVecElemF, pressureF, avgRelDensityErrorF_raw_, 
			avgRelDensityErrorF_pos_, avgRelDensityErrorF_abs_);

		N_solverIterCount_++;
	} while (!TerminationConditionSatisfied(N_solverIterCount_));

	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_JacobiSolver });

	// --- [ finalize pressure acceleration ] ----------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_Acceleration });

	CPU01::PressureAcceleration(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		pressureF,
		neighborLists,
		sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensityF,
		massF, massB, params.gammaPress, N_F_,
		xAccPressF, yAccPressF, zAccPressF);

	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_Acceleration });

	// --- [ integrate velocity and position ] ---------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::VelocityUpdate,
									Bench::BenchOp::PositionUpdate });

	CPU01::Integration(
		xVelPredF, yVelPredF, zVelPredF,
		xAccPressF, yAccPressF, zAccPressF,
		xAccViscF, yAccViscF, zAccViscF,
		xAccExtF, yAccExtF, zAccExtF,
		timeStep, N_F_,
		xPosF, yPosF, zPosF,
		xVelF, yVelF, zVelF,
		xAccF, yAccF, zAccF);

	benchmarksMgr_.StopBench({ Bench::BenchOp::VelocityUpdate,
								   Bench::BenchOp::PositionUpdate });
}

bool SimBackend_CPU01::TerminationConditionSatisfied(uint32_t iterCount)
{
	// TODO: implement properly with parameters for choosing condition

	bool satisfied = false;

	if (iterCount > 5) { satisfied = true; }
	// if (avgRelDensityErrorF_raw_ <= 0.001f) { satisfied = true; }
	// if (avgRelDensityErrorF_pos_ <= 0.01f) { satisfied = true; }
	// if (avgRelDensityErrorF_abs_ <= 0.01f) { satisfied = true; }

	return satisfied;
}

inline void CPU01::KernelWeight_CubicSpline(
	const float dx, const float dy, const float dz,
	const float particleSizeInv, const float kernelRadiusSqu,
	const float cubicSplineAlpha,
	// destinations
	float& kw) noexcept
{
	const float distSqu = dx * dx + dy * dy + dz * dz;

	// outside support radius -> no weight
	if (distSqu >= kernelRadiusSqu) [[unlikely]]
	{
		kw = 0.0f;
		return;
	}

	// no distance -> shortcut computation
	if (distSqu < ZERO_EQU_LIM) [[unlikely]]
	{
		kw = 4.0f * cubicSplineAlpha;
		return;
	}

	const float dist = std::sqrt(distSqu);
	const float distNorm = dist * particleSizeInv;

	// clamp t1 to [0, 1]
	// no clamp for t2 needed, distNorm < 2 here
	float t1 = 1.0f - distNorm;
	t1 = std::max(t1, 0.0f);
	const float t2 = 2.0f - distNorm;

	// common sub-expressions
	const float t1_2 = t1 * t1;
	const float t1_3 = t1_2 * t1;
	const float t2_2 = t2 * t2;
	const float t2_3 = t2_2 * t2;

	// kernel weight
	kw = cubicSplineAlpha * (t2_3 - 4.0f * t1_3);
}

inline void CPU01::KernelGradient_CubicSpline(
	const float dx, const float dy, const float dz,
	const float particleSizeInv, const float kernelRadiusSqu,
	const float cubicSplineAlpha,
	// destinations
	float& xkg, float& ykg, float& zkg) noexcept
{
	const float distSqu = dx * dx + dy * dy + dz * dz;

	// outside support radius -> no gradient
	// no distance -> also no gradient
	if (distSqu >= kernelRadiusSqu ||
		distSqu < ZERO_EQU_LIM) [[unlikely]]
	{
		xkg = 0.0f;
		ykg = 0.0f;
		zkg = 0.0f;
		return;
	}

	const float dist = std::sqrt(distSqu);
	const float distNorm = dist * particleSizeInv;

	// clamp t1 to [0, 1]
	// no clamp for t2 needed, distNorm < 2 here
	float t1 = 1.0f - distNorm;
	t1 = std::max(t1, 0.0f);
	const float t2 = 2.0f - distNorm;

	// common sub-expressions
	const float t1_2 = t1 * t1;
	const float t2_2 = t2 * t2;

	float commonFac = cubicSplineAlpha
					* (-3.0f * t2_2 + 12.0f * t1_2)
					* (particleSizeInv / dist);

	// kernel gradient
	xkg = commonFac * dx;
	ykg = commonFac * dy;
	zkg = commonFac * dz;
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::Density(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const I_NeighborLists& neighborLists,
	const float particleSizeInvF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float massF, const float massB,
	const uint32_t N_F,
	// destinations
	float* RESTRICT densityF)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				float tmp_densityF = 0.0f;
				float tmp_densityB = 0.0f;

				// contributions from fluid neighbors
				NeighborSpan nbrListFF = neighborLists.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.count;
				#pragma omp simd reduction(+:tmp_densityF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// kernel weight
					float kw = 0.0f;
					KernelWeight_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu, cubicSplineAlpha, kw);

					tmp_densityF += kw;
				}
				tmp_densityF *= massF;

				// contributions from boundary neighbors
				NeighborSpan nbrListFB = neighborLists.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.count;
				#pragma omp simd reduction(+:tmp_densityB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFB[k];

					// position difference
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// kernel weight
					float kw = 0.0f;
					KernelWeight_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu, cubicSplineAlpha, kw);

					tmp_densityB += kw;
				}
				tmp_densityB *= massB;

				// finalize density
				densityF[i] = tmp_densityF + tmp_densityB;
			}
		},
		tbb::auto_partitioner{}
	);
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::NonPressureAcceleration(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT xVelF,
	const float* RESTRICT yVelF,
	const float* RESTRICT zVelF,
	const float* RESTRICT densityF,
	const I_NeighborLists& neighborLists,
	const float timeStep, const float particleSizeInvF,
	const float particleSizeSquF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float restDensityB,
	const float massF, const float massB, const float viscosityF, const float viscosityB,
	const float xGravity, const float yGravity, const float zGravity,
	const uint32_t N_F,
	// destinations
	float* RESTRICT xAccViscF,
	float* RESTRICT yAccViscF,
	float* RESTRICT zAccViscF,
	float* RESTRICT xAccExtF,
	float* RESTRICT yAccExtF,
	float* RESTRICT zAccExtF,
	float* RESTRICT xVelPredF,
	float* RESTRICT yVelPredF,
	float* RESTRICT zVelPredF)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				const float vxi = xVelF[i];
				const float vyi = yVelF[i];
				const float vzi = zVelF[i];

				float tmp_xAccViscF = 0.0f;
				float tmp_yAccViscF = 0.0f;
				float tmp_zAccViscF = 0.0f;

				// contributions from fluid neighbors
				NeighborSpan nbrListFF = neighborLists.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.count;
				#pragma omp simd reduction(+:tmp_xAccViscF,tmp_yAccViscF,tmp_zAccViscF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// velocity difference
					const float dvx = vxi - xVelF[j];
					const float dvy = vyi - yVelF[j];
					const float dvz = vzi - zVelF[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// viscosity acc contribution from fluid neighbor j
					const float accCoeff = (massF / fmaxf(densityF[j], 1e-12f))
										 * (dvx * dx + dvy * dy + dvz * dz)
										 / (dx * dx + dy * dy + dz * dz
										 + 0.01f * particleSizeSquF);

					tmp_xAccViscF += accCoeff * xkg;
					tmp_yAccViscF += accCoeff * ykg;
					tmp_zAccViscF += accCoeff * zkg;
				}

				float tmp_xAccViscB = 0.0f;
				float tmp_yAccViscB = 0.0f;
				float tmp_zAccViscB = 0.0f;

				// contributions from boundary neighbors
				NeighborSpan nbrListFB = neighborLists.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.count;
				#pragma omp simd reduction(+:tmp_xAccViscB,tmp_yAccViscB,tmp_zAccViscB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFB[k];

					// position difference
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// viscosity acc contribution from boundary neighbor j
					const float accCoeff = (massB / restDensityB)
										 * (vxi * dx + vyi * dy + vzi * dz)
										 / (dx * dx + dy * dy + dz * dz
										 + 0.01f * particleSizeSquF);

					tmp_xAccViscB += accCoeff * xkg;
					tmp_yAccViscB += accCoeff * ykg;
					tmp_zAccViscB += accCoeff * zkg;
				}

				// finalize accelerations
				// (3D -> factor 2 * (dim + 2))
				tmp_xAccViscF *= 10.0f * viscosityF;
				tmp_yAccViscF *= 10.0f * viscosityF;
				tmp_zAccViscF *= 10.0f * viscosityF;
				tmp_xAccViscB *= 10.0f * viscosityB;
				tmp_yAccViscB *= 10.0f * viscosityB;
				tmp_zAccViscB *= 10.0f * viscosityB;

				xAccViscF[i] = tmp_xAccViscF + tmp_xAccViscB;
				yAccViscF[i] = tmp_yAccViscF + tmp_yAccViscB;
				zAccViscF[i] = tmp_zAccViscF + tmp_zAccViscB;

				// TODO: add surface tension acc contribution
				xAccExtF[i] = xGravity;
				yAccExtF[i] = yGravity;
				zAccExtF[i] = zGravity;

				// compute predicted velocity from non-pressure accelerations
				xVelPredF[i] = xVelF[i] + timeStep * (xAccViscF[i] + xAccExtF[i]);
				yVelPredF[i] = yVelF[i] + timeStep * (yAccViscF[i] + yAccExtF[i]);
				zVelPredF[i] = zVelF[i] + timeStep * (zAccViscF[i] + zAccExtF[i]);
			}
		},
		tbb::auto_partitioner{}
	);
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::SourceTermAndDiagonalElement(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT xVelPredF,
	const float* RESTRICT yVelPredF,
	const float* RESTRICT zVelPredF,
	const float* RESTRICT densityF,
	const I_NeighborLists& neighborLists,
	const float timeStep, const float particleSizeInvF,
	const float kernelRadiusSqu, const float cubicSplineAlpha,
	const float restDensityF, const float restDensitySquInvF,
	const float massF, const float massB, const float gammaPress,
	const uint32_t N_F,
	// destinations
	float* RESTRICT sourceTermF,
	float* RESTRICT diagElemF)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				const float vpxi = xVelPredF[i];
				const float vpyi = yVelPredF[i];
				const float vpzi = zVelPredF[i];

				// source term accumulators
				float tmp_SourceTermF = 0.0f;
				float tmp_SourceTermB = 0.0f;

				// c-term accumulators
				float xCTermF = 0.0f;
				float yCTermF = 0.0f;
				float zCTermF = 0.0f;

				// contributions from fluid neighbors
				NeighborSpan nbrListFF = neighborLists.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.count;
				#pragma omp simd reduction(+:tmp_SourceTermF,xCTermF,yCTermF,zCTermF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// predicted velocity difference
					const float dvpx = vpxi - xVelPredF[j];
					const float dvpy = vpyi - yVelPredF[j];
					const float dvpz = vpzi - zVelPredF[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// source term contribution from fluid neighbor
					tmp_SourceTermF += dvpx * xkg
									 + dvpy * ykg
									 + dvpz * zkg;

					xCTermF += xkg;
					yCTermF += ykg;
					zCTermF += zkg;
				}
				tmp_SourceTermF *= massF;

				float xCTermB = 0.0f;
				float yCTermB = 0.0f;
				float zCTermB = 0.0f;

				// contributions from boundary neighbors
				NeighborSpan nbrListFB = neighborLists.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.count;
				#pragma omp simd reduction(+:tmp_SourceTermB,xCTermB,yCTermB,zCTermB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFB[k];

					// position difference
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// source term contribution from boundary neighbor
					// TODO: subtract actual boundary velocity when adding non-static boundaries?
					tmp_SourceTermB += vpxi * xkg
									 + vpyi * ykg
									 + vpzi * zkg;

					xCTermB += xkg;
					yCTermB += ykg;
					zCTermB += zkg;
				}
				tmp_SourceTermB *= massB;

				// finalize source term
				sourceTermF[i] = restDensityF - densityF[i] - timeStep
							   * (tmp_SourceTermF + tmp_SourceTermB);

				// finalize c-term
				const float xCTerm = -1.0f * (xCTermF * restDensitySquInvF * massF)
					- 2.0f * gammaPress * (xCTermB * restDensitySquInvF * massB);
				const float yCTerm = -1.0f * (yCTermF * restDensitySquInvF * massF)
					- 2.0f * gammaPress * (yCTermB * restDensitySquInvF * massB);
				const float zCTerm = -1.0f * (zCTermF * restDensitySquInvF * massF)
					- 2.0f * gammaPress * (zCTermB * restDensitySquInvF * massB);

				float tmp_DiagElemA = 0.0f;
				float tmp_DiagElemB = 0.0f;
				float tmp_DiagElemC = 0.0f;

				// part A
				// TODO: fuze A, B, and C loops into one loop?
				#pragma omp simd reduction(+:tmp_DiagElemA)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					tmp_DiagElemA += xCTerm * xkg
								   + yCTerm * ykg
								   + zCTerm * zkg;
				}
				tmp_DiagElemA *= massF;

				// part B
				// TODO: fuze A, B, and C loops into one loop?
				#pragma omp simd reduction(+:tmp_DiagElemB)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					const float tmp_x = (massF * restDensitySquInvF) * (-xkg);
					const float tmp_y = (massF * restDensitySquInvF) * (-ykg);
					const float tmp_z = (massF * restDensitySquInvF) * (-zkg);

					tmp_DiagElemB += tmp_x * xkg
								   + tmp_y * ykg
								   + tmp_z * zkg;
				}
				tmp_DiagElemB *= massF;

				// part C
				// TODO: fuze A, B, and C loops into one loop?
				#pragma omp simd reduction(+:tmp_DiagElemC)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFB[k];

					// position difference
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					tmp_DiagElemC += xCTerm * xkg
								   + yCTerm * ykg
								   + zCTerm * zkg;
				}
				tmp_DiagElemC *= massB;

				// finalize diagonal element
				diagElemF[i] = timeStep * timeStep
							 * (tmp_DiagElemA + tmp_DiagElemB + tmp_DiagElemC);
			}
		},
		tbb::auto_partitioner{}
	);
}

void CPU01::PressureJacobiInit(
	const float* RESTRICT diagElemF,
	const float* RESTRICT sourceTermF,
	const float jacobiOmega,
	const uint32_t N_F,
	// destinations
	float* RESTRICT pressureF)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float tmp_PressureF = 0.0f;

				// only update pressure if particle has neighbors
				if (!Base::ApproxEqu(diagElemF[i], 0.0f))
				{
					tmp_PressureF = jacobiOmega * (sourceTermF[i] / diagElemF[i]);
				}

				// clamp pressure to non-negative values
				pressureF[i] = fmaxf(tmp_PressureF, 0.0f);
			}
		},
		tbb::auto_partitioner{}
	);
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::PressureAcceleration(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT pressureF,
	const I_NeighborLists& neighborLists,
	const float particleSizeInvF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float restDensityF,
	const float massF, const float massB, const float gammaPress,
	const uint32_t N_F,
	// destinations
	float* RESTRICT xAccPressF,
	float* RESTRICT yAccPressF,
	float* RESTRICT zAccPressF)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				const float presi = pressureF[i];
				const float densi = restDensityF;
				const float densiInvSqu = 1.0f / (densi * densi);
				const float accCoeffF_outer = massF * presi * densiInvSqu;

				float tmp_xAccPressF = 0.0f;
				float tmp_yAccPressF = 0.0f;
				float tmp_zAccPressF = 0.0f;

				// contributions from fluid neighbors
				NeighborSpan nbrListFF = neighborLists.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.count;
				#pragma omp simd reduction(+:tmp_xAccPressF,tmp_yAccPressF,tmp_zAccPressF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					const float accCoeffF = accCoeffF_outer + massF
										  * pressureF[j] * densiInvSqu;

					// pressure acc contribution from fluid neighbor j
					tmp_xAccPressF += accCoeffF * xkg;
					tmp_yAccPressF += accCoeffF * ykg;
					tmp_zAccPressF += accCoeffF * zkg;
				}

				const float accCoeffB = gammaPress * massB
									  * 2.0f * presi * densiInvSqu;

				// contributions from boundary neighbors
				NeighborSpan nbrListFB = neighborLists.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.count;
				#pragma omp simd reduction(+:tmp_xAccPressF,tmp_yAccPressF,tmp_zAccPressF)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFB[k];

					// position difference
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// pressure acc contribution from boundary neighbor j
					tmp_xAccPressF += accCoeffB * xkg;
					tmp_yAccPressF += accCoeffB * ykg;
					tmp_zAccPressF += accCoeffB * zkg;
				}

				// finalize pressure acceleration
				tmp_xAccPressF *= -1.0f;
				tmp_yAccPressF *= -1.0f;
				tmp_zAccPressF *= -1.0f;

				xAccPressF[i] = tmp_xAccPressF;
				yAccPressF[i] = tmp_yAccPressF;
				zAccPressF[i] = tmp_zAccPressF;
			}
		},
		tbb::auto_partitioner{}
	);
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::PressureRelaxedJacobiSolver(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT xAccPressF,
	const float* RESTRICT yAccPressF,
	const float* RESTRICT zAccPressF,
	const float* RESTRICT sourceTermF,
	const float* RESTRICT diagElemF,
	const I_NeighborLists& neighborLists,
	const float timeStep, const float particleSizeInvF,
	const float kernelRadiusSqu, const float cubicSplineAlpha,
	const float restDensityF, const float massF, const float massB,
	const float jacobiOmega,
	const uint32_t N_F,
	// destinations
	float* RESTRICT sysVecElemF,
	float* RESTRICT pressureF,
	float& avgRelDensityErrorF_raw,
	float& avgRelDensityErrorF_pos,
	float& avgRelDensityErrorF_abs)
{
	struct DensityErrorAccum
	{
		double sumRaw = 0.0;
		double sumPos = 0.0;
		double sumAbs = 0.0;
		uint64_t cntAll = 0;
		uint64_t cntPos = 0;

		DensityErrorAccum() = default;

		DensityErrorAccum(DensityErrorAccum& other, tbb::split)
		{
			sumRaw = 0.0;
			sumPos = 0.0;
			sumAbs = 0.0;
			cntAll = 0;
			cntPos = 0;
		}

		void join(const DensityErrorAccum& rhs)
		{
			sumRaw += rhs.sumRaw;
			sumPos += rhs.sumPos;
			sumAbs += rhs.sumAbs;
			cntAll += rhs.cntAll;
			cntPos += rhs.cntPos;
		}
	};

	DensityErrorAccum acc = tbb::parallel_reduce(
		tbb::blocked_range<uint32_t>(0, N_F), DensityErrorAccum{},
		[&](const tbb::blocked_range<uint32_t>& r, DensityErrorAccum local) -> DensityErrorAccum
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
			    const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				const float axi = xAccPressF[i];
				const float ayi = yAccPressF[i];
				const float azi = zAccPressF[i];

				float tmp_SysVecElemF = 0.0f;
				float tmp_SysVecElemB = 0.0f;

				// contributions from fluid neighbors
				NeighborSpan nbrListFF = neighborLists.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.count;
				#pragma omp simd reduction(+:tmp_SysVecElemF)
				for (uint32_t k = 0; k < N_FF; ++k)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// pressure acc difference
					const float dapx = axi - xAccPressF[j];
					const float dapy = ayi - yAccPressF[j];
					const float dapz = azi - zAccPressF[j];

					tmp_SysVecElemF += dapx * xkg
						+ dapy * ykg
						+ dapz * zkg;
				}
				tmp_SysVecElemF *= massF;

				// contributions from boundary neighbors
				NeighborSpan nbrListFB = neighborLists.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.count;
				#pragma omp simd reduction(+:tmp_SysVecElemB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFB[k];

					// position difference
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// TODO: subtract actual boundary pressure acc when adding non-static boundaries?
					tmp_SysVecElemB += axi * xkg
						+ ayi * ykg
						+ azi * zkg;
				}
				tmp_SysVecElemB *= massB;

				// finalize system vector element
				sysVecElemF[i] = timeStep * timeStep
							   * (tmp_SysVecElemF + tmp_SysVecElemB);

				const float densityError = sourceTermF[i] - sysVecElemF[i];
				float tmp_PressureF = 0.0f;

				// only update pressure and error if particle has neighbors
				if (!Base::ApproxEqu(diagElemF[i], 0.0f))
				{
					// relaxed Jacobi solver update
					tmp_PressureF = pressureF[i] + jacobiOmega
						* (densityError / diagElemF[i]);

					// relative density error
					const float relErr = densityError / restDensityF;
					const float absErr = std::fabs(relErr);

					local.sumRaw += static_cast<double>(relErr);
					local.sumAbs += static_cast<double>(absErr);
					local.cntAll++;

					if (relErr > 0.0f)
					{
						local.sumPos += static_cast<double>(relErr);
						local.cntPos++;
					}
				}

				// clamp pressure to non-negative values
				pressureF[i] = fmaxf(tmp_PressureF, 0.0f);
			}

			return local;
		},
		[](const DensityErrorAccum& a, const DensityErrorAccum& b) -> DensityErrorAccum
		{
			DensityErrorAccum c = a;
			c.join(b);
			return c;
		}
	);

	// finalize averages and handle degenerate cases
	if (acc.cntAll > 0)
	{
		avgRelDensityErrorF_raw = static_cast<float>(acc.sumRaw 
							    / static_cast<double>(acc.cntAll));
		avgRelDensityErrorF_abs = static_cast<float>(acc.sumAbs
							    / static_cast<double>(acc.cntAll));
	}
	else
	{
		avgRelDensityErrorF_raw = 0.0f;
		avgRelDensityErrorF_abs = 0.0f;
	}

	if (acc.cntPos > 0)
	{
		avgRelDensityErrorF_pos = static_cast<float>(acc.sumPos 
								/ static_cast<double>(acc.cntPos));
	}
	else
	{
		avgRelDensityErrorF_pos = 0.0f;
	}
}

void CPU01::Integration(
	const float* RESTRICT xVelPredF,
	const float* RESTRICT yVelPredF,
	const float* RESTRICT zVelPredF,
	const float* RESTRICT xAccPressF,
	const float* RESTRICT yAccPressF,
	const float* RESTRICT zAccPressF,
	const float* RESTRICT xAccViscF,
	const float* RESTRICT yAccViscF,
	const float* RESTRICT zAccViscF,
	const float* RESTRICT xAccExtF,
	const float* RESTRICT yAccExtF,
	const float* RESTRICT zAccExtF,
	const float timeStep,
	const uint32_t N_F,
	// destinations
	float* RESTRICT xPosF,
	float* RESTRICT yPosF,
	float* RESTRICT zPosF,
	float* RESTRICT xVelF,
	float* RESTRICT yVelF,
	float* RESTRICT zVelF,
	float* RESTRICT xAccF,
	float* RESTRICT yAccF,
	float* RESTRICT zAccF)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// total acceleration update
				xAccF[i] = xAccPressF[i] + xAccViscF[i] + xAccExtF[i];
				yAccF[i] = yAccPressF[i] + yAccViscF[i] + yAccExtF[i];
				zAccF[i] = zAccPressF[i] + zAccViscF[i] + zAccExtF[i];

				// explicit Euler integration
				// (refining predicted velocity with pressure acc)
				xVelF[i] = xVelPredF[i] + timeStep * xAccPressF[i];
				yVelF[i] = yVelPredF[i] + timeStep * yAccPressF[i];
				zVelF[i] = zVelPredF[i] + timeStep * zAccPressF[i];

				// semi-implicit Euler integration
				// (using updated velocity for position update)
				xPosF[i] = xPosF[i] + timeStep * xVelF[i];
				yPosF[i] = yPosF[i] + timeStep * yVelF[i];
				zPosF[i] = zPosF[i] + timeStep * zVelF[i];
			}
		},
		tbb::auto_partitioner{}
	);
}
