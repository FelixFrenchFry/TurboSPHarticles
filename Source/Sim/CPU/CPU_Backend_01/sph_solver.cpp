// sph_solver.cpp

// external
#include <algorithm>
#include <cmath>
#include <numbers>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/partitioner.h>

// internal
#include "Source/Base/helpers.h"
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

	// cubic spline alpha (computed in double precision for accuracy)
	const double size = static_cast<double>(params.particleSizeF);
	const double sizeInv = 1.0 / size;
	const float  cubicSplineAlpha = static_cast<float>(
		(1.0 / (4.0 * static_cast<double>(PI))) * sizeInv * sizeInv * sizeInv);

	// misc hoisted constants
	const float sizeF = params.particleSizeF;
	const float massF = params.massF;
	const float massB = params.massB;
	const float restDensitySquInvF = 1.0f / (params.restDensityF * params.restDensityF);

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

	// references for neighbors
	auto& neighborsFF = neighborsFF_;
	auto& neighborsFB = neighborsFB_;

	// --- [ neighbor search ] -------------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Construction });
	neighborSearchStrat_->Construct(params);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Construction });

	// --- [ neighbor search query ] -------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Query });
	neighborSearchStrat_->Query(params);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Query });

	// --- [ density computations ] --------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Density_Solving });

	CPU01::Density(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		neighborsFF, neighborsFB,
		cubicSplineAlpha, sizeF,
		massF, massB, N_F_,
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
		neighborsFF, neighborsFB,
		params.timeStep, cubicSplineAlpha, sizeF, params.restDensityB,
		massF, massB, params.viscosityF, params.viscosityB,
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
		neighborsFF_, neighborsFB_,
		params.timeStep, cubicSplineAlpha, sizeF,
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

	int cnt = 0;
	while (cnt < 2) // TODO: add proper termination condition
	{
		// compute pressure acceleration from current pressure field
		CPU01::PressureAcceleration(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			pressureF,
			neighborsFF, neighborsFB,
			cubicSplineAlpha, sizeF, params.restDensityF,
			massF, massB, params.gammaPress, N_F_,
			xAccPressF, yAccPressF, zAccPressF);

		// update system vector and pressure
		CPU01::PressureRelaxedJacobiSolver(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			xAccPressF, yAccPressF, zAccPressF,
			sourceTermF, diagElemF, sysVecElemF,
			neighborsFF, neighborsFB,
			params.timeStep, cubicSplineAlpha, sizeF, massF, massB,
			params.jacobiOmega, N_F_,
			pressureF);

		cnt++;
	}

	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_JacobiSolver });

	// --- [ finalize pressure acc and integrate velocity and position ] -------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_Acceleration,
								    Bench::BenchOp::VelocityUpdate,
								    Bench::BenchOp::PositionUpdate });

	CPU01::PressureAccelerationAndIntegration(
		xPosB, yPosB, zPosB,
		xVelPredF, yVelPredF, zVelPredF,
		xAccViscF, yAccViscF, zAccViscF,
		xAccExtF, yAccExtF, zAccExtF,
		pressureF,
		neighborsFF, neighborsFB,
		params.timeStep, cubicSplineAlpha, sizeF, params.restDensityF,
		massF, massB, params.gammaPress, N_F_,
		xPosF, yPosF, zPosF,
		xVelF, yVelF, zVelF,
		xAccF, yAccF, zAccF,
		xAccPressF, yAccPressF, zAccPressF);

	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_Acceleration,
								   Bench::BenchOp::VelocityUpdate,
								   Bench::BenchOp::PositionUpdate });
}

inline void CPU01::KernelWeight_CubicSpline(
	const float dx, const float dy, const float dz,
	const float cubicSplineAlpha, const float particleSize,
	// destinations
	float& kw) noexcept
{
	const float distSqu = dx * dx + dy * dy + dz * dz;
	const float dist = std::sqrt(distSqu);
	const float distNorm = dist / particleSize;

	float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
	float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

	// kernel weight
	kw = cubicSplineAlpha * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);
}

inline void CPU01::KernelGradient_CubicSpline(
	const float dx, const float dy, const float dz,
	const float cubicSplineAlpha, const float particleSize,
	// destinations
	float& xkg, float& ykg, float& zkg) noexcept
{
	const float distSqu = dx * dx + dy * dy + dz * dz;
	const float dist = std::sqrt(distSqu);

	// avoid division by zero in gradient computation
	if (Base::ApproxEqu(dist, 0.0f)) [[unlikely]]
	{
		xkg = 0.0f;
		ykg = 0.0f;
		zkg = 0.0f;

		return;
	}

	const float distNorm = dist / particleSize;

	float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
	float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

	float commonFac = cubicSplineAlpha
					* (-3.0f * t2 * t2 + 12.0f * t1 * t1)
					/ (distNorm * particleSize * particleSize);

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
	const std::vector<std::vector<std::uint32_t>>& neighborsFF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFB,
	float cubicSplineAlpha, float particleSizeF,
	float massF, float massB,
	uint32_t N_F,
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
				const uint32_t* RESTRICT nbrFF = neighborsFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF[i].size());
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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, kw);

					tmp_densityF += kw;
				}
				tmp_densityF *= massF;

				// contributions from boundary neighbors
				const uint32_t* RESTRICT nbrFB = neighborsFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB[i].size());
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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, kw);

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
	const std::vector<std::vector<std::uint32_t>>& neighborsFF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFB,
	const float timeStep, const float cubicSplineAlpha,
	const float particleSizeF, const float restDensityB,
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
				const uint32_t* RESTRICT nbrFF = neighborsFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF[i].size());
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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, xkg, ykg, zkg);

					// viscosity acc contribution from fluid neighbor j
					const float accCoeff = (massF / fmaxf(densityF[j], 1e-12f))
										 * (dvx * dx + dvy * dy + dvz * dz)
										 / (dx * dx + dy * dy + dz * dz
										 + 0.01f * particleSizeF * particleSizeF);

					tmp_xAccViscF += accCoeff * xkg;
					tmp_yAccViscF += accCoeff * ykg;
					tmp_zAccViscF += accCoeff * zkg;
				}

				float tmp_xAccViscB = 0.0f;
				float tmp_yAccViscB = 0.0f;
				float tmp_zAccViscB = 0.0f;

				// contributions from boundary neighbors
				const uint32_t* RESTRICT nbrFB = neighborsFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB[i].size());
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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, xkg, ykg, zkg);

					// viscosity acc contribution from boundary neighbor j
					const float accCoeff = (massB / restDensityB)
										 * (vxi * dx + vyi * dy + vzi * dz)
										 / (dx * dx + dy * dy + dz * dz
										 + 0.01f * particleSizeF * particleSizeF);

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
	const std::vector<std::vector<std::uint32_t>>& neighborsFF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFB,
	const float timeStep, const float cubicSplineAlpha, const float particleSizeF,
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
				const uint32_t* RESTRICT nbrFF = neighborsFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF[i].size());
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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, xkg, ykg, zkg);

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
				const uint32_t* RESTRICT nbrFB = neighborsFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB[i].size());
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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, xkg, ykg, zkg);

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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, xkg, ykg, zkg);

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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, xkg, ykg, zkg);

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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, xkg, ykg, zkg);

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
	const std::vector<std::vector<std::uint32_t>>& neighborsFF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFB,
	const float cubicSplineAlpha, const float particleSizeF,
	const float restDensityF, const float massF, const float massB,
	const float gammaPress,
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

				float tmp_xAccPressF = 0.0f;
				float tmp_yAccPressF = 0.0f;
				float tmp_zAccPressF = 0.0f;

				// contributions from fluid neighbors
				const uint32_t* RESTRICT nbrFF = neighborsFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF[i].size());
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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, xkg, ykg, zkg);

					// pressure acc contribution from fluid neighbor j
					// TODO: put most of this coefficient computation outside the loop?
					const float accCoeff = massF * (presi / (densi * densi)
										 + pressureF[j] / (densi * densi));

					tmp_xAccPressF += accCoeff * xkg;
					tmp_yAccPressF += accCoeff * ykg;
					tmp_zAccPressF += accCoeff * zkg;
				}

				// contributions from boundary neighbors
				const uint32_t* RESTRICT nbrFB = neighborsFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB[i].size());
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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, xkg, ykg, zkg);

					// pressure acc contribution from boundary neighbor j
					// TODO: put full coefficient computation outside the loop?
					const float accCoeff = gammaPress * massB
										 * 2.0f * (presi / (densi * densi));

					tmp_xAccPressF += accCoeff * xkg;
					tmp_yAccPressF += accCoeff * ykg;
					tmp_zAccPressF += accCoeff * zkg;
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
	float* RESTRICT sysVecElemF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFB,
	const float timeStep, const float cubicSplineAlpha,
	const float particleSizeF, const float massF, const float massB,
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
				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				const float axi = xAccPressF[i];
				const float ayi = yAccPressF[i];
				const float azi = zAccPressF[i];

				float tmp_SysVecElemF = 0.0f;
				float tmp_SysVecElemB = 0.0f;

				// contributions from fluid neighbors
				const uint32_t* RESTRICT nbrFF = neighborsFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF[i].size());
				#pragma omp simd reduction(+:tmp_SysVecElemF)
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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, xkg, ykg, zkg);

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
				const uint32_t* RESTRICT nbrFB = neighborsFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB[i].size());
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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, xkg, ykg, zkg);

					// TODO: subtract actual boundary pressure acc when adding non-static boundaries?
					tmp_SysVecElemB += axi * xkg
									 + ayi * ykg
									 + azi * zkg;
				}
				tmp_SysVecElemB *= massB;

				// finalize system vector element
				sysVecElemF[i] = timeStep * timeStep
							   * (tmp_SysVecElemF + tmp_SysVecElemB);

				float tmp_PressureF = 0.0f;

				// only update pressure if particle has neighbors
				if (!Base::ApproxEqu(diagElemF[i], 0.0f))
				{
					tmp_PressureF = pressureF[i] + jacobiOmega
								  * ((sourceTermF[i] - sysVecElemF[i]) / diagElemF[i]);
				}

				// clamp pressure to non-negative values
				pressureF[i] = fmaxf(tmp_PressureF, 0.0f);
			}
		},
		tbb::auto_partitioner{}
	);
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::PressureAccelerationAndIntegration(
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT xVelPredF,
	const float* RESTRICT yVelPredF,
	const float* RESTRICT zVelPredF,
	const float* RESTRICT xAccViscF,
	const float* RESTRICT yAccViscF,
	const float* RESTRICT zAccViscF,
	const float* RESTRICT xAccExtF,
	const float* RESTRICT yAccExtF,
	const float* RESTRICT zAccExtF,
	const float* RESTRICT pressureF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFB,
	const float timeStep, const float cubicSplineAlpha,
	const float particleSizeF, const float restDensityF,
	const float massF, const float massB, const float gammaPress,
	uint32_t N_F,
	// destinations
	float* RESTRICT xPosF,
	float* RESTRICT yPosF,
	float* RESTRICT zPosF,
	float* RESTRICT xVelF,
	float* RESTRICT yVelF,
	float* RESTRICT zVelF,
	float* RESTRICT xAccF,
	float* RESTRICT yAccF,
	float* RESTRICT zAccF,
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

				float tmp_xAccPressF = 0.0f;
				float tmp_yAccPressF = 0.0f;
				float tmp_zAccPressF = 0.0f;

				// contributions from fluid neighbors
				const uint32_t* RESTRICT nbrFF = neighborsFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF[i].size());
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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, xkg, ykg, zkg);

					// pressure acc contribution from fluid neighbor j
					// TODO: put most of this coefficient computation outside the loop?
					const float accCoeff = massF * (presi / (densi * densi)
										 + pressureF[j] / (densi * densi));

					tmp_xAccPressF += accCoeff * xkg;
					tmp_yAccPressF += accCoeff * ykg;
					tmp_zAccPressF += accCoeff * zkg;
				}

				// contributions from boundary neighbors
				const uint32_t* RESTRICT nbrFB = neighborsFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB[i].size());
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
						dx, dy, dz, cubicSplineAlpha, particleSizeF, xkg, ykg, zkg);

					// pressure acc contribution from boundary neighbor j
					// TODO: put full coefficient computation outside the loop?
					const float accCoeff = gammaPress * massB
										 * 2.0f * (presi / (densi * densi));

					tmp_xAccPressF += accCoeff * xkg;
					tmp_yAccPressF += accCoeff * ykg;
					tmp_zAccPressF += accCoeff * zkg;
				}

				// finalize pressure acceleration
				tmp_xAccPressF *= -1.0f;
				tmp_yAccPressF *= -1.0f;
				tmp_zAccPressF *= -1.0f;

				xAccPressF[i] = tmp_xAccPressF;
				yAccPressF[i] = tmp_yAccPressF;
				zAccPressF[i] = tmp_zAccPressF;

				// total acceleration update
				xAccF[i] = xAccPressF[i] + xAccViscF[i] + xAccExtF[i];
				yAccF[i] = yAccPressF[i] + yAccViscF[i] + yAccExtF[i];
				zAccF[i] = zAccPressF[i] + zAccViscF[i] + zAccExtF[i];

				// explicit Euler integration
				// (now only refining predicted velocity with pressure acc)
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
