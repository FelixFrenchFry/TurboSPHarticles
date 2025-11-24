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

	// convenience references for neighbor / kernel containers
	auto& neighborsFF = neighborsFF_;
	auto& neighborsFB = neighborsFB_;
	auto& kwFF = kernelWeightsFF_;
	auto& kwFB = kernelWeightsFB_;
	auto& xGradFF = xKernelGradFF_;
	auto& yGradFF = yKernelGradFF_;
	auto& zGradFF = zKernelGradFF_;
	auto& xGradFB = xKernelGradFB_;
	auto& yGradFB = yKernelGradFB_;
	auto& zGradFB = zKernelGradFB_;

	// --- [ neighbor search ] -------------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Construction });
	neighborSearchStrat_->Construct(params);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Construction });

	// --- [ neighbor search query ] -------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Query });
	neighborSearchStrat_->Query(params);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Query });

	// --- [ kernel weight and gradient computations ] -------------------------

	/*
	benchmarksMgr_.StartBench({ Bench::BenchOp::Kernel_Weights,
								    Bench::BenchOp::Kernel_Gradients });

	CPU01::KernelWeightsAndGradients(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		neighborsFF, neighborsFB,
		sizeF, cubicSplineAlpha, N_F_,
		kwFF, kwFB,
		xGradFF, yGradFF, zGradFF,
		xGradFB, yGradFB, zGradFB);

	benchmarksMgr_.StopBench({ Bench::BenchOp::Kernel_Weights,
								   Bench::BenchOp::Kernel_Gradients });

	// --- [ density computations ] --------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Density_Solving });

	CPU01::Density(
		kwFF, kwFB,
		massF, massB, N_F_,
		densityF);

	benchmarksMgr_.StopBench({ Bench::BenchOp::Density_Solving });
	*/

	// --- [ kernel weight, gradient, and density computations ] ---------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Kernel_Weights,
									Bench::BenchOp::Kernel_Gradients,
									Bench::BenchOp::Density_Solving });

	CPU01::KernelWeightsAndGradientsAndDensity(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		neighborsFF, neighborsFB,
		sizeF, massF, massB, cubicSplineAlpha, N_F_,
		kwFF, kwFB,
		xGradFF, yGradFF, zGradFF,
		xGradFB, yGradFB, zGradFB,
		densityF);

	benchmarksMgr_.StopBench({ Bench::BenchOp::Kernel_Weights,
								   Bench::BenchOp::Kernel_Gradients,
								   Bench::BenchOp::Density_Solving });

	// --- [ non-pressure accelerations ] --------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Viscosity_Acceleration,
									Bench::BenchOp::External_Acceleration });

	CPU01::NonPressureAcceleration(
		xPosF, yPosF, zPosF,
		xVelF, yVelF, zVelF,
		xPosB, yPosB, zPosB,
		densityF,
		neighborsFF, neighborsFB,
		xGradFF, yGradFF, zGradFF,
		xGradFB, yGradFB, zGradFB,
		params.timeStep, sizeF, params.restDensityB,
		massF, massB, params.viscosityF, params.viscosityB,
		params.gravity.x, params.gravity.y, params.gravity.z,
		N_F_,
		xAccViscF, yAccViscF, zAccViscF,
		xAccExtF, yAccExtF, zAccExtF,
		xVelPredF, yVelPredF, zVelPredF);

	benchmarksMgr_.StopBench({ Bench::BenchOp::Viscosity_Acceleration,
								   Bench::BenchOp::External_Acceleration });

	// --- [ source-term for density invariance ] ------------------------------

	/*
	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_SourceTerm });

	CPU01::SourceTerm(
		xVelPredF, yVelPredF, zVelPredF,
		densityF,
		neighborsFF, neighborsFB,
		xGradFF, yGradFF, zGradFF,
		xGradFB, yGradFB, zGradFB,
		params.timeStep, params.restDensityF, massF, massB, N_F_,
		sourceTermF);

	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_SourceTerm });

	// --- [ diagonal elements of system matrix ] ------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_DiagonalElem });

	CPU01::DiagonalElement(
		neighborsFF, neighborsFB,
		xGradFF, yGradFF, zGradFF,
		xGradFB, yGradFB, zGradFB,
		params.timeStep, restDensitySquInvF, massF, massB, params.gammaPress, N_F_,
		diagElemF);

	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_DiagonalElem });
	*/

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_SourceTerm,
									Bench::BenchOp::Pressure_DiagonalElem });

	CPU01::SourceTermAndDiagonalElement(
		xVelPredF, yVelPredF, zVelPredF,
		densityF,
		neighborsFF_, neighborsFB_,
		xKernelGradFF_, yKernelGradFF_, zKernelGradFF_,
		xKernelGradFB_, yKernelGradFB_, zKernelGradFB_,
		params.timeStep, params.restDensityF, restDensitySquInvF,
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
			pressureF,
			neighborsFF, neighborsFB,
			xGradFF, yGradFF, zGradFF,
			xGradFB, yGradFB, zGradFB,
			params.restDensityF,
			massF, massB,
			params.gammaPress,
			N_F_,
			xAccPressF, yAccPressF, zAccPressF);

		// update system vector and pressure
		CPU01::PressureRelaxedJacobiSolver(
			xAccPressF, yAccPressF, zAccPressF,
			sourceTermF, diagElemF, sysVecElemF,
			neighborsFF, neighborsFB,
			xGradFF, yGradFF, zGradFF,
			xGradFB, yGradFB, zGradFB,
			params.timeStep, massF, massB, params.jacobiOmega, N_F_,
			pressureF);

		cnt++;
	}

	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_JacobiSolver });

	// --- [ finalize pressure acceleration for integration ] ------------------

	/*
	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_Acceleration });

	CPU01::PressureAcceleration(
		pressureF,
		neighborsFF, neighborsFB,
		xGradFF, yGradFF, zGradFF,
		xGradFB, yGradFB, zGradFB,
		params.restDensityF, massF, massB, params.gammaPress, N_F_,
		xAccPressF, yAccPressF, zAccPressF); 

	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_Acceleration });

	// --- [ velocity and position updates ] -----------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::VelocityUpdate,
									Bench::BenchOp::PositionUpdate });

	CPU01::Integration(
		xVelPredF, yVelPredF, zVelPredF,
		xAccPressF, yAccPressF, zAccPressF,
		xAccViscF, yAccViscF, zAccViscF,
		xAccExtF, yAccExtF, zAccExtF,
		params.timeStep, N_F_,
		xPosF, yPosF, zPosF,
		xVelF, yVelF, zVelF,
		xAccF, yAccF, zAccF);

	benchmarksMgr_.StopBench({ Bench::BenchOp::VelocityUpdate,
								   Bench::BenchOp::PositionUpdate });

	*/

	// --- [ finalize pressure acc and integrate velocity and position ] -------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_Acceleration,
								    Bench::BenchOp::VelocityUpdate,
								    Bench::BenchOp::PositionUpdate });

	CPU01::PressureAccelerationAndIntegration(
		xVelPredF, yVelPredF, zVelPredF,
		xAccViscF, yAccViscF, zAccViscF,
		xAccExtF, yAccExtF, zAccExtF,
		pressureF,
		neighborsFF, neighborsFB,
		xGradFF, yGradFF, zGradFF,
		xGradFB, yGradFB, zGradFB,
		params.timeStep, params.restDensityF, massF, massB,
		params.gammaPress, N_F_,
		xPosF, yPosF, zPosF,
		xVelF, yVelF, zVelF,
		xAccF, yAccF, zAccF,
		xAccPressF, yAccPressF, zAccPressF);

	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_Acceleration,
								   Bench::BenchOp::VelocityUpdate,
								   Bench::BenchOp::PositionUpdate });
}

void CPU01::KernelWeightsAndGradients(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const std::vector<std::vector<uint32_t>>& neighborsFF,
	const std::vector<std::vector<uint32_t>>& neighborsFB,
	const float particleSizeF, const float cubicSplineAlpha,
	const uint32_t N_F,
	// destinations
	std::vector<std::vector<float>>& kernelWeightsFF,
	std::vector<std::vector<float>>& kernelWeightsFB,
	std::vector<std::vector<float>>& xKernelGradFF,
	std::vector<std::vector<float>>& yKernelGradFF,
	std::vector<std::vector<float>>& zKernelGradFF,
	std::vector<std::vector<float>>& xKernelGradFB,
	std::vector<std::vector<float>>& yKernelGradFB,
	std::vector<std::vector<float>>& zKernelGradFB)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// resize kernel weight/gradient storage to current number of neighbors
				kernelWeightsFF[i].resize(neighborsFF[i].size());
				kernelWeightsFB[i].resize(neighborsFB[i].size());
				xKernelGradFF[i].resize(neighborsFF[i].size());
				yKernelGradFF[i].resize(neighborsFF[i].size());
				zKernelGradFF[i].resize(neighborsFF[i].size());
				xKernelGradFB[i].resize(neighborsFB[i].size());
				yKernelGradFB[i].resize(neighborsFB[i].size());
				zKernelGradFB[i].resize(neighborsFB[i].size());

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				// contributions from fluid neighbors
				const uint32_t* RESTRICT nbrFF = neighborsFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF[i].size());
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					const float distNorm = std::sqrt(dx * dx + dy * dy + dz * dz)
										 / particleSizeF;

					// avoid division by zero in gradient computation
					if (Base::ApproxEqu(distNorm, 0.0f)) [[unlikely]]
					{
						kernelWeightsFF[i][k] = cubicSplineAlpha * 4.0f;
						xKernelGradFF[i][k] = 0.0f;
						yKernelGradFF[i][k] = 0.0f;
						zKernelGradFF[i][k] = 0.0f;

						continue;
					}

					const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
					const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

					// kernel weight
					kernelWeightsFF[i][k] = cubicSplineAlpha
										  * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);

					// kernel gradient
					// TODO: multiply distNorm with size once or twice?
					const float gradCoeff = cubicSplineAlpha
										  * (-3.0f * t2 * t2 + 12.0f * t1 * t1)
										  / (distNorm * particleSizeF * particleSizeF);

					xKernelGradFF[i][k] = gradCoeff * dx;
					yKernelGradFF[i][k] = gradCoeff * dy;
					zKernelGradFF[i][k] = gradCoeff * dz;
				}

				// contributions from boundary neighbors
				const uint32_t* RESTRICT nbrFB = neighborsFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB[i].size());
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFB[k];

					// position difference
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					const float distNorm = std::sqrt(dx * dx + dy * dy + dz * dz)
										 / particleSizeF;

					// avoid division by zero in gradient computation
					if (Base::ApproxEqu(distNorm, 0.0f)) [[unlikely]]
					{
						kernelWeightsFB[i][k] = cubicSplineAlpha * 4.0f;
						xKernelGradFB[i][k] = 0.0f;
						yKernelGradFB[i][k] = 0.0f;
						zKernelGradFB[i][k] = 0.0f;

						continue;
					}

					const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
					const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

					// kernel weight
					kernelWeightsFB[i][k] = cubicSplineAlpha
										  * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);

					// kernel gradient
					// TODO: multiply distNorm with size once or twice?
					const float gradCoeff = cubicSplineAlpha
										  * (-3.0f * t2 * t2 + 12.0f * t1 * t1)
										  / (distNorm * particleSizeF * particleSizeF);

					xKernelGradFB[i][k] = gradCoeff * dx;
					yKernelGradFB[i][k] = gradCoeff * dy;
					zKernelGradFB[i][k] = gradCoeff * dz;
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::Density(
	const std::vector<std::vector<float>>& kernelWeightsFF,
	const std::vector<std::vector<float>>& kernelWeightsFB,
	const float massF, const float massB,
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
				float tmp_densityF = 0.0f;
				float tmp_densityB = 0.0f;

				// contributions from fluid neighbors
				const float* RESTRICT kwFF = kernelWeightsFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(kernelWeightsFF[i].size());
				#pragma omp simd reduction(+:tmp_densityF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					tmp_densityF += kwFF[k];
				}
				tmp_densityF *= massF;

				// contributions from boundary neighbors
				const float* RESTRICT kwFB = kernelWeightsFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(kernelWeightsFB[i].size());
				#pragma omp simd reduction(+:tmp_densityB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					tmp_densityB += kwFB[k];
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
void CPU01::KernelWeightsAndGradientsAndDensity(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const std::vector<std::vector<uint32_t>>& neighborsFF,
	const std::vector<std::vector<uint32_t>>& neighborsFB,
	const float particleSizeF, const float massF, const float massB,
	const float cubicSplineAlpha,
	const uint32_t N_F,
	// destinations
	std::vector<std::vector<float>>& kernelWeightsFF,
	std::vector<std::vector<float>>& kernelWeightsFB,
	std::vector<std::vector<float>>& xKernelGradFF,
	std::vector<std::vector<float>>& yKernelGradFF,
	std::vector<std::vector<float>>& zKernelGradFF,
	std::vector<std::vector<float>>& xKernelGradFB,
	std::vector<std::vector<float>>& yKernelGradFB,
	std::vector<std::vector<float>>& zKernelGradFB,
	float* RESTRICT densityF)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// resize kernel weight/gradient storage to current number of neighbors
				kernelWeightsFF[i].resize(neighborsFF[i].size());
				kernelWeightsFB[i].resize(neighborsFB[i].size());
				xKernelGradFF[i].resize(neighborsFF[i].size());
				yKernelGradFF[i].resize(neighborsFF[i].size());
				zKernelGradFF[i].resize(neighborsFF[i].size());
				xKernelGradFB[i].resize(neighborsFB[i].size());
				yKernelGradFB[i].resize(neighborsFB[i].size());
				zKernelGradFB[i].resize(neighborsFB[i].size());

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

					const float distNorm = std::sqrt(dx * dx + dy * dy + dz * dz)
										 / particleSizeF;

					// avoid division by zero in gradient computation
					if (Base::ApproxEqu(distNorm, 0.0f)) [[unlikely]]
					{
						const float kw0 = cubicSplineAlpha * 4.0f;
						kernelWeightsFF[i][k] = kw0;
						tmp_densityF += kw0;
						xKernelGradFF[i][k] = 0.0f;
						yKernelGradFF[i][k] = 0.0f;
						zKernelGradFF[i][k] = 0.0f;

						continue;
					}

					const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
					const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

					// kernel weight and density contribution
					const float kwFF = cubicSplineAlpha
									 * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);
					kernelWeightsFF[i][k] = kwFF;
					tmp_densityF += kwFF;

					// kernel gradient
					// TODO: multiply distNorm with size once or twice?
					const float gradCoeff = cubicSplineAlpha
										  * (-3.0f * t2 * t2 + 12.0f * t1 * t1)
										  / (distNorm * particleSizeF * particleSizeF);

					xKernelGradFF[i][k] = gradCoeff * dx;
					yKernelGradFF[i][k] = gradCoeff * dy;
					zKernelGradFF[i][k] = gradCoeff * dz;
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

					const float distNorm = std::sqrt(dx * dx + dy * dy + dz * dz)
										 / particleSizeF;

					// avoid division by zero in gradient computation
					if (Base::ApproxEqu(distNorm, 0.0f)) [[unlikely]]
					{
						const float kw0 = cubicSplineAlpha * 4.0f;
						kernelWeightsFB[i][k] = kw0;
						tmp_densityB += kw0;
						xKernelGradFB[i][k] = 0.0f;
						yKernelGradFB[i][k] = 0.0f;
						zKernelGradFB[i][k] = 0.0f;

						continue;
					}

					const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
					const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

					// kernel weight and density contribution
					const float kwFB = cubicSplineAlpha
									 * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);
					kernelWeightsFB[i][k] = kwFB;
					tmp_densityB += kwFB;

					// kernel gradient
					// TODO: multiply distNorm with size once or twice?
					const float gradCoeff = cubicSplineAlpha
										  * (-3.0f * t2 * t2 + 12.0f * t1 * t1)
										  / (distNorm * particleSizeF * particleSizeF);

					xKernelGradFB[i][k] = gradCoeff * dx;
					yKernelGradFB[i][k] = gradCoeff * dy;
					zKernelGradFB[i][k] = gradCoeff * dz;
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
	const float* RESTRICT xVelF,
	const float* RESTRICT yVelF,
	const float* RESTRICT zVelF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT densityF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFB,
	const std::vector<std::vector<float>>& xKernelGradFF,
	const std::vector<std::vector<float>>& yKernelGradFF,
	const std::vector<std::vector<float>>& zKernelGradFF,
	const std::vector<std::vector<float>>& xKernelGradFB,
	const std::vector<std::vector<float>>& yKernelGradFB,
	const std::vector<std::vector<float>>& zKernelGradFB,
	const float timeStep, const float particleSizeF, const float restDensityB,
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
				const float* RESTRICT xkgFF = xKernelGradFF[i].data();
				const float* RESTRICT ykgFF = yKernelGradFF[i].data();
				const float* RESTRICT zkgFF = zKernelGradFF[i].data();
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

					// viscosity acc contribution from fluid neighbor j
					const float accCoeff = (massF / fmaxf(densityF[j], 1e-12f))
										 * (dvx * dx + dvy * dy + dvz * dz)
										 / (dx * dx + dy * dy + dz * dz
										 + 0.01f * particleSizeF * particleSizeF);

					tmp_xAccViscF += accCoeff * xkgFF[k];
					tmp_yAccViscF += accCoeff * ykgFF[k];
					tmp_zAccViscF += accCoeff * zkgFF[k];
				}

				float tmp_xAccViscB = 0.0f;
				float tmp_yAccViscB = 0.0f;
				float tmp_zAccViscB = 0.0f;

				// contributions from boundary neighbors
				const float* RESTRICT xkgFB = xKernelGradFB[i].data();
				const float* RESTRICT ykgFB = yKernelGradFB[i].data();
				const float* RESTRICT zkgFB = zKernelGradFB[i].data();
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

					// viscosity acc contribution from boundary neighbor j
					const float accCoeff = (massB / restDensityB)
										 * (vxi * dx + vyi * dy + vzi * dz)
										 / (dx * dx + dy * dy + dz * dz
										 + 0.01f * particleSizeF * particleSizeF);

					tmp_xAccViscB += accCoeff * xkgFB[k];
					tmp_yAccViscB += accCoeff * ykgFB[k];
					tmp_zAccViscB += accCoeff * zkgFB[k];
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
void CPU01::SourceTerm(
	const float* RESTRICT xVelPredF,
	const float* RESTRICT yVelPredF,
	const float* RESTRICT zVelPredF,
	const float* RESTRICT densityF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFB,
	const std::vector<std::vector<float>>& xKernelGradFF,
	const std::vector<std::vector<float>>& yKernelGradFF,
	const std::vector<std::vector<float>>& zKernelGradFF,
	const std::vector<std::vector<float>>& xKernelGradFB,
	const std::vector<std::vector<float>>& yKernelGradFB,
	const std::vector<std::vector<float>>& zKernelGradFB,
	const float timeStep, const float restDensityF,
	const float massF, const float massB,
	const uint32_t N_F,
	// destinations
	float* RESTRICT sourceTermF)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const float vpxi = xVelPredF[i];
				const float vpyi = yVelPredF[i];
				const float vpzi = zVelPredF[i];

				float tmp_SourceTermF = 0.0f;
				float tmp_SourceTermB = 0.0f;

				// contributions from fluid neighbors
				const float* RESTRICT xkgFF = xKernelGradFF[i].data();
				const float* RESTRICT ykgFF = yKernelGradFF[i].data();
				const float* RESTRICT zkgFF = zKernelGradFF[i].data();
				const uint32_t* RESTRICT nbrFF = neighborsFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF[i].size());
				#pragma omp simd reduction(+:tmp_SourceTermF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// predicted velocity difference
					const float dvpx = vpxi - xVelPredF[j];
					const float dvpy = vpyi - yVelPredF[j];
					const float dvpz = vpzi - zVelPredF[j];

					tmp_SourceTermF += dvpx * xkgFF[k]
									 + dvpy * ykgFF[k]
									 + dvpz * zkgFF[k];
				}
				tmp_SourceTermF *= massF;

				// contributions from boundary neighbors
				const float* RESTRICT xkgFB = xKernelGradFB[i].data();
				const float* RESTRICT ykgFB = yKernelGradFB[i].data();
				const float* RESTRICT zkgFB = zKernelGradFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB[i].size());
				#pragma omp simd reduction(+:tmp_SourceTermB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// TODO: subtract actual boundary velocity when adding non-static boundaries?
					tmp_SourceTermB += vpxi * xkgFB[k]
									 + vpyi * ykgFB[k]
									 + vpzi * zkgFB[k];
				}
				tmp_SourceTermB *= massB;

				// finalize source term
				sourceTermF[i] = restDensityF - densityF[i] - timeStep
							   * (tmp_SourceTermF + tmp_SourceTermB);
			}
		},
		tbb::auto_partitioner{}
	);
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::DiagonalElement(
	const std::vector<std::vector<std::uint32_t>>& neighborsFF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFB,
	const std::vector<std::vector<float>>& xKernelGradFF,
	const std::vector<std::vector<float>>& yKernelGradFF,
	const std::vector<std::vector<float>>& zKernelGradFF,
	const std::vector<std::vector<float>>& xKernelGradFB,
	const std::vector<std::vector<float>>& yKernelGradFB,
	const std::vector<std::vector<float>>& zKernelGradFB,
	const float timeStep, const float restDensitySquInvF,
	const float massF, const float massB, const float gammaPress,
	const uint32_t N_F,
	// destinations
	float* RESTRICT diagElemF)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float xCTermF = 0.0f;
				float yCTermF = 0.0f;
				float zCTermF = 0.0f;

				// add c-term contributions from fluid neighbors
				const float* RESTRICT xkgFF = xKernelGradFF[i].data();
				const float* RESTRICT ykgFF = yKernelGradFF[i].data();
				const float* RESTRICT zkgFF = zKernelGradFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF[i].size());
				#pragma omp simd reduction(+:xCTermF,yCTermF,zCTermF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					xCTermF += xkgFF[k];
					yCTermF += ykgFF[k];
					zCTermF += zkgFF[k];
				}

				float xCTermB = 0.0f;
				float yCTermB = 0.0f;
				float zCTermB = 0.0f;

				// add c-term contributions from boundary neighbors
				const float* RESTRICT xkgFB = xKernelGradFB[i].data();
				const float* RESTRICT ykgFB = yKernelGradFB[i].data();
				const float* RESTRICT zkgFB = zKernelGradFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB[i].size());
				#pragma omp simd reduction(+:xCTermB,yCTermB,zCTermB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					xCTermB += xkgFB[k];
					yCTermB += ykgFB[k];
					zCTermB += zkgFB[k];
				}

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
				#pragma omp simd reduction(+:tmp_DiagElemA)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					tmp_DiagElemA += xCTerm * xkgFF[k]
								   + yCTerm * ykgFF[k]
								   + zCTerm * zkgFF[k];
				}
				tmp_DiagElemA *= massF;

				// part B
				#pragma omp simd reduction(+:tmp_DiagElemB)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					const float tmp_x = (massF * restDensitySquInvF) * (-xkgFF[k]);
					const float tmp_y = (massF * restDensitySquInvF) * (-ykgFF[k]);
					const float tmp_z = (massF * restDensitySquInvF) * (-zkgFF[k]);

					tmp_DiagElemB += tmp_x * xkgFF[k]
								   + tmp_y * ykgFF[k]
								   + tmp_z * zkgFF[k];
				}
				tmp_DiagElemB *= massF;

				// part C
				#pragma omp simd reduction(+:tmp_DiagElemC)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					tmp_DiagElemC += xCTerm * xkgFB[k]
								   + yCTerm * ykgFB[k]
								   + zCTerm * zkgFB[k];
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

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::SourceTermAndDiagonalElement(
	const float* RESTRICT xVelPredF,
	const float* RESTRICT yVelPredF,
	const float* RESTRICT zVelPredF,
	const float* RESTRICT densityF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFB,
	const std::vector<std::vector<float>>& xKernelGradFF,
	const std::vector<std::vector<float>>& yKernelGradFF,
	const std::vector<std::vector<float>>& zKernelGradFF,
	const std::vector<std::vector<float>>& xKernelGradFB,
	const std::vector<std::vector<float>>& yKernelGradFB,
	const std::vector<std::vector<float>>& zKernelGradFB,
	const float timeStep, const float restDensityF, const float restDensitySquInvF,
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
				const float* RESTRICT xkgFF = xKernelGradFF[i].data();
				const float* RESTRICT ykgFF = yKernelGradFF[i].data();
				const float* RESTRICT zkgFF = zKernelGradFF[i].data();
				const uint32_t* RESTRICT nbrFF = neighborsFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF[i].size());
				#pragma omp simd reduction(+:tmp_SourceTermF,xCTermF,yCTermF,zCTermF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// predicted velocity difference
					const float dvpx = vpxi - xVelPredF[j];
					const float dvpy = vpyi - yVelPredF[j];
					const float dvpz = vpzi - zVelPredF[j];

					// source term contribution from fluid neighbor
					tmp_SourceTermF += dvpx * xkgFF[k]
									 + dvpy * ykgFF[k]
									 + dvpz * zkgFF[k];

					xCTermF += xkgFF[k];
					yCTermF += ykgFF[k];
					zCTermF += zkgFF[k];
				}
				tmp_SourceTermF *= massF;

				float xCTermB = 0.0f;
				float yCTermB = 0.0f;
				float zCTermB = 0.0f;

				// contributions from boundary neighbors
				const float* RESTRICT xkgFB = xKernelGradFB[i].data();
				const float* RESTRICT ykgFB = yKernelGradFB[i].data();
				const float* RESTRICT zkgFB = zKernelGradFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB[i].size());
				#pragma omp simd reduction(+:tmp_SourceTermB,xCTermB,yCTermB,zCTermB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// source term contribution from boundary neighbor
					// TODO: subtract actual boundary velocity when adding non-static boundaries?
					tmp_SourceTermB += vpxi * xkgFB[k]
									 + vpyi * ykgFB[k]
									 + vpzi * zkgFB[k];

					xCTermB += xkgFB[k];
					yCTermB += ykgFB[k];
					zCTermB += zkgFB[k];
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
				#pragma omp simd reduction(+:tmp_DiagElemA)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					tmp_DiagElemA += xCTerm * xkgFF[k]
								   + yCTerm * ykgFF[k]
								   + zCTerm * zkgFF[k];
				}
				tmp_DiagElemA *= massF;

				// part B
				#pragma omp simd reduction(+:tmp_DiagElemB)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					const float tmp_x = (massF * restDensitySquInvF) * (-xkgFF[k]);
					const float tmp_y = (massF * restDensitySquInvF) * (-ykgFF[k]);
					const float tmp_z = (massF * restDensitySquInvF) * (-zkgFF[k]);

					tmp_DiagElemB += tmp_x * xkgFF[k]
								   + tmp_y * ykgFF[k]
								   + tmp_z * zkgFF[k];
				}
				tmp_DiagElemB *= massF;

				// part C
				#pragma omp simd reduction(+:tmp_DiagElemC)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					tmp_DiagElemC += xCTerm * xkgFB[k]
								   + yCTerm * ykgFB[k]
								   + zCTerm * zkgFB[k];
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
	const float* RESTRICT pressureF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFB,
	const std::vector<std::vector<float>>& xKernelGradFF,
	const std::vector<std::vector<float>>& yKernelGradFF,
	const std::vector<std::vector<float>>& zKernelGradFF,
	const std::vector<std::vector<float>>& xKernelGradFB,
	const std::vector<std::vector<float>>& yKernelGradFB,
	const std::vector<std::vector<float>>& zKernelGradFB,
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
				const float presi = pressureF[i];
				const float densi = restDensityF;

				float tmp_xAccPressF = 0.0f;
				float tmp_yAccPressF = 0.0f;
				float tmp_zAccPressF = 0.0f;

				// contributions from fluid neighbors
				const float* RESTRICT xkgFF = xKernelGradFF[i].data();
				const float* RESTRICT ykgFF = yKernelGradFF[i].data();
				const float* RESTRICT zkgFF = zKernelGradFF[i].data();
				const uint32_t* RESTRICT nbrFF = neighborsFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF[i].size());
				#pragma omp simd reduction(+:tmp_xAccPressF,tmp_yAccPressF,tmp_zAccPressF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// pressure acc contribution from fluid neighbor j
					// TODO: put most of this coefficient computation outside the loop?
					const float accCoeff = massF * (presi / (densi * densi)
										 + pressureF[j] / (densi * densi));

					tmp_xAccPressF += accCoeff * xkgFF[k];
					tmp_yAccPressF += accCoeff * ykgFF[k];
					tmp_zAccPressF += accCoeff * zkgFF[k];
				}

				// contributions from boundary neighbors
				const float* RESTRICT xkgFB = xKernelGradFB[i].data();
				const float* RESTRICT ykgFB = yKernelGradFB[i].data();
				const float* RESTRICT zkgFB = zKernelGradFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB[i].size());
				#pragma omp simd reduction(+:tmp_xAccPressF,tmp_yAccPressF,tmp_zAccPressF)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// pressure acc contribution from boundary neighbor j
					// TODO: put full coefficient computation outside the loop?
					const float accCoeff = gammaPress * massB
										 * 2.0f * (presi / (densi * densi));

					tmp_xAccPressF += accCoeff * xkgFB[k];
					tmp_yAccPressF += accCoeff * ykgFB[k];
					tmp_zAccPressF += accCoeff * zkgFB[k];
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
	const float* RESTRICT xAccPressF,
	const float* RESTRICT yAccPressF,
	const float* RESTRICT zAccPressF,
	const float* RESTRICT sourceTermF,
	const float* RESTRICT diagElemF,
	float* RESTRICT sysVecElemF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFF,
	const std::vector<std::vector<std::uint32_t>>& neighborsFB,
	const std::vector<std::vector<float>>& xKernelGradFF,
	const std::vector<std::vector<float>>& yKernelGradFF,
	const std::vector<std::vector<float>>& zKernelGradFF,
	const std::vector<std::vector<float>>& xKernelGradFB,
	const std::vector<std::vector<float>>& yKernelGradFB,
	const std::vector<std::vector<float>>& zKernelGradFB,
	const float timeStep, const float massF, const float massB,
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
				const float axi = xAccPressF[i];
				const float ayi = yAccPressF[i];
				const float azi = zAccPressF[i];

				float tmp_SysVecElemF = 0.0f;
				float tmp_SysVecElemB = 0.0f;

				// contributions from fluid neighbors
				const float* RESTRICT xkgFF = xKernelGradFF[i].data();
				const float* RESTRICT ykgFF = yKernelGradFF[i].data();
				const float* RESTRICT zkgFF = zKernelGradFF[i].data();
				const uint32_t* RESTRICT nbrFF = neighborsFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF[i].size());
				#pragma omp simd reduction(+:tmp_SysVecElemF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// pressure acc difference
					const float dapx = axi - xAccPressF[j];
					const float dapy = ayi - yAccPressF[j];
					const float dapz = azi - zAccPressF[j];

					tmp_SysVecElemF += dapx * xkgFF[k]
									 + dapy * ykgFF[k]
									 + dapz * zkgFF[k];
				}
				tmp_SysVecElemF *= massF;

				// contributions from boundary neighbors
				const float* RESTRICT xkgFB = xKernelGradFB[i].data();
				const float* RESTRICT ykgFB = yKernelGradFB[i].data();
				const float* RESTRICT zkgFB = zKernelGradFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB[i].size());
				#pragma omp simd reduction(+:tmp_SysVecElemB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// TODO: subtract actual boundary pressure acc when adding non-static boundaries?
					tmp_SysVecElemB += axi * xkgFB[k]
									 + ayi * ykgFB[k]
									 + azi * zkgFB[k];
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
				// (now only refining predicted velocity with pressure acc)
				xVelF[i] = xVelPredF[i] + timeStep * xAccPressF[i];
				yVelF[i] = yVelPredF[i] + timeStep * yAccPressF[i];
				zVelF[i] = zVelPredF[i] + timeStep * zAccPressF[i];

				// semi-implicit Euler integration
				// (using updated velocity for position update)
				xPosF[i] = xPosF[i] + timeStep * xVelF[i];
				yPosF[i] = yPosF[i] + timeStep * yVelF[i];
				zPosF[i] = zPosF[i] + timeStep * zVelF[i];

				// TODO: just leave total acceleration un-updated all the time?
			}
		},
		tbb::auto_partitioner{}
	);
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::PressureAccelerationAndIntegration(
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
	const std::vector<std::vector<float>>& xKernelGradFF,
	const std::vector<std::vector<float>>& yKernelGradFF,
	const std::vector<std::vector<float>>& zKernelGradFF,
	const std::vector<std::vector<float>>& xKernelGradFB,
	const std::vector<std::vector<float>>& yKernelGradFB,
	const std::vector<std::vector<float>>& zKernelGradFB,
	float timeStep, float restDensityF,
	float massF, float massB, float gammaPress,
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
				const float presi = pressureF[i];
				const float densi = restDensityF;

				float tmp_xAccPressF = 0.0f;
				float tmp_yAccPressF = 0.0f;
				float tmp_zAccPressF = 0.0f;

				// contributions from fluid neighbors
				const float* RESTRICT xkgFF = xKernelGradFF[i].data();
				const float* RESTRICT ykgFF = yKernelGradFF[i].data();
				const float* RESTRICT zkgFF = zKernelGradFF[i].data();
				const uint32_t* RESTRICT nbrFF = neighborsFF[i].data();
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF[i].size());
				#pragma omp simd reduction(+:tmp_xAccPressF,tmp_yAccPressF,tmp_zAccPressF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// pressure acc contribution from fluid neighbor j
					// TODO: put most of this coefficient computation outside the loop?
					const float accCoeff = massF * (presi / (densi * densi)
										 + pressureF[j] / (densi * densi));

					tmp_xAccPressF += accCoeff * xkgFF[k];
					tmp_yAccPressF += accCoeff * ykgFF[k];
					tmp_zAccPressF += accCoeff * zkgFF[k];
				}

				// contributions from boundary neighbors
				const float* RESTRICT xkgFB = xKernelGradFB[i].data();
				const float* RESTRICT ykgFB = yKernelGradFB[i].data();
				const float* RESTRICT zkgFB = zKernelGradFB[i].data();
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB[i].size());
				#pragma omp simd reduction(+:tmp_xAccPressF,tmp_yAccPressF,tmp_zAccPressF)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// pressure acc contribution from boundary neighbor j
					// TODO: put full coefficient computation outside the loop?
					const float accCoeff = gammaPress * massB
										 * 2.0f * (presi / (densi * densi));

					tmp_xAccPressF += accCoeff * xkgFB[k];
					tmp_yAccPressF += accCoeff * ykgFB[k];
					tmp_zAccPressF += accCoeff * zkgFB[k];
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
