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



constexpr float PI = std::numbers::pi_v<float>;
using namespace Sim;

void SimBackend_CPU01::Step_IISPH(const SimParams& params)
{
	// --- [ temporary pre-computed constants ] --------------------------------

	// cubic spline alpha (computed in double precision for accuracy)
	const double size = static_cast<double>(params.particleSizeF);
	const double sizeInv = 1.0 / size;
	const float cubicSplineAlpha = static_cast<float>(
		(1.0 / (4.0 * static_cast<double>(PI))) * sizeInv * sizeInv * sizeInv);

	// misc hoisted constants
	const float sizeF = params.particleSizeF;
	const float massF = params.massF;
	const float massB = params.massB;
	const float restDensitySquInvF = 1.0f / (params.restDensityF
								   * params.restDensityF);

	// --- [ neighbor search ] -------------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Construction });
	neighborSearchStrat_->Construct(params);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Construction });

	// --- [ neighbor search query ] -------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Query });
	neighborSearchStrat_->Query(params);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Query });

	// --- [ kernel weight and gradient computations ] -------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Kernel_Weights,
									Bench::BenchOp::Kernel_Gradients });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// resize kernel weight/gradient storage to current number of neighbors
				kernelWeightsFF_[i].resize(neighborsFF_[i].size());
				kernelWeightsFB_[i].resize(neighborsFB_[i].size());
				xKernelGradFF_[i].resize(neighborsFF_[i].size());
				yKernelGradFF_[i].resize(neighborsFF_[i].size());
				zKernelGradFF_[i].resize(neighborsFF_[i].size());
				xKernelGradFB_[i].resize(neighborsFB_[i].size());
				yKernelGradFB_[i].resize(neighborsFB_[i].size());
				zKernelGradFB_[i].resize(neighborsFB_[i].size());

				const float xi = xPosF_[i];
				const float yi = yPosF_[i];
				const float zi = zPosF_[i];

				// contributions from fluid neighbors
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFF_[i][k];

					// position difference
					const float dx = xi - xPosF_[j];
					const float dy = yi - yPosF_[j];
					const float dz = zi - zPosF_[j];

					const float distNorm = std::sqrt(dx * dx + dy * dy + dz * dz)
										 / params.particleSizeF;

					// avoid division by zero in gradient computation
					if (Base::ApproxEqu(distNorm, 0.0f)) [[unlikely]]
					{
						kernelWeightsFF_[i][k] = cubicSplineAlpha * 4.0f;
						xKernelGradFF_[i][k] = 0.0f;
						yKernelGradFF_[i][k] = 0.0f;
						zKernelGradFF_[i][k] = 0.0f;

						continue;
					}

					const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
					const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

					// kernel weight
					kernelWeightsFF_[i][k] = cubicSplineAlpha
										   * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);

					// kernel gradient
					// TODO: multiply distNorm with size once or twice?
					const float gradCoeff = cubicSplineAlpha
										  * (-3.0f * t2 * t2 + 12.0f * t1 * t1)
										  / (distNorm * sizeF * sizeF);

					xKernelGradFF_[i][k] = gradCoeff * dx;
					yKernelGradFF_[i][k] = gradCoeff * dy;
					zKernelGradFF_[i][k] = gradCoeff * dz;
				}

				// contributions from boundary neighbors
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFB_[i][k];

					// position difference
					const float dx = xi - xPosB_[j];
					const float dy = yi - yPosB_[j];
					const float dz = zi - zPosB_[j];

					const float distNorm = std::sqrt(dx * dx + dy * dy + dz * dz)
										 / params.particleSizeF;

					// avoid division by zero in gradient computation
					if (Base::ApproxEqu(distNorm, 0.0f)) [[unlikely]]
					{
						kernelWeightsFB_[i][k] = cubicSplineAlpha * 4.0f;
						xKernelGradFB_[i][k] = 0.0f;
						yKernelGradFB_[i][k] = 0.0f;
						zKernelGradFB_[i][k] = 0.0f;

						continue;
					}

					const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
					const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

					// kernel weight
					kernelWeightsFB_[i][k] = cubicSplineAlpha
										   * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);

					// kernel gradient
					// TODO: multiply distNorm with size once or twice?
					const float gradCoeff = cubicSplineAlpha
										  * (-3.0f * t2 * t2 + 12.0f * t1 * t1)
										  / (distNorm * sizeF * sizeF);

					xKernelGradFB_[i][k] = gradCoeff * dx;
					yKernelGradFB_[i][k] = gradCoeff * dy;
					zKernelGradFB_[i][k] = gradCoeff * dz;
				}
			}
		},
		tbb::auto_partitioner{}
	);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Kernel_Weights,
								   Bench::BenchOp::Kernel_Gradients });

	// --- [ density computations ] --------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Density_Solving });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float density = 0.0f;

				// contributions from fluid neighbors
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
				#pragma omp simd reduction(+:density)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					density += massF * kernelWeightsFF_[i][k];
				}

				// contributions from boundary neighbors
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
				#pragma omp simd reduction(+:density)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					density += massB * kernelWeightsFB_[i][k];
				}

				densityF_[i] = density;
			}
		},
		tbb::auto_partitioner{}
	);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Density_Solving });

	// --- [ non-pressure accelerations ] --------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Viscosity_Acceleration,
									Bench::BenchOp::External_Acceleration });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const float xi = xPosF_[i];
				const float yi = yPosF_[i];
				const float zi = zPosF_[i];

				const float vxi = xVelF_[i];
				const float vyi = yVelF_[i];
				const float vzi = zVelF_[i];

				float tmp_xAccViscF = 0.0f;
				float tmp_yAccViscF = 0.0f;
				float tmp_zAccViscF = 0.0f;

				// contributions from fluid neighbors
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
				#pragma omp simd reduction(+:tmp_xAccViscF,tmp_yAccViscF,tmp_zAccViscF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFF_[i][k];

					// position difference
					const float dx = xi - xPosF_[j];
					const float dy = yi - yPosF_[j];
					const float dz = zi - zPosF_[j];

					// velocity difference
					const float dvx = vxi - xVelF_[j];
					const float dvy = vyi - yVelF_[j];
					const float dvz = vzi - zVelF_[j];

					// viscosity acc contribution from fluid neighbor j
					const float accCoeff = (massF / fmaxf(densityF_[j], 1e-12f))
										 * (dvx * dx + dvy * dy + dvz * dz)
										 / (dx * dx + dy * dy + dz * dz
										 + 0.01f * sizeF * sizeF);

					tmp_xAccViscF += accCoeff * xKernelGradFF_[i][k];
					tmp_yAccViscF += accCoeff * yKernelGradFF_[i][k];
					tmp_zAccViscF += accCoeff * zKernelGradFF_[i][k];
				}

				float tmp_xAccViscB = 0.0f;
				float tmp_yAccViscB = 0.0f;
				float tmp_zAccViscB = 0.0f;

				// contributions from boundary neighbors
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
				#pragma omp simd reduction(+:tmp_xAccViscB,tmp_yAccViscB,tmp_zAccViscB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFB_[i][k];

					// position difference
					const float dx = xi - xPosB_[j];
					const float dy = yi - yPosB_[j];
					const float dz = zi - zPosB_[j];

					// viscosity acc contribution from boundary neighbor j
					const float accCoeff = (massB / params.restDensityB)
										 * (vxi * dx + vyi * dy + vzi * dz)
										 / (dx * dx + dy * dy + dz * dz
										 + 0.01f * sizeF * sizeF);

					tmp_xAccViscB += accCoeff * xKernelGradFB_[i][k];
					tmp_yAccViscB += accCoeff * yKernelGradFB_[i][k];
					tmp_zAccViscB += accCoeff * zKernelGradFB_[i][k];
				}

				// finalize accelerations
				// (3D -> factor 2 * (dim + 2))
				tmp_xAccViscF *= 2.0f * (3 + 2) * params.viscosityF;
				tmp_yAccViscF *= 2.0f * (3 + 2) * params.viscosityF;
				tmp_zAccViscF *= 2.0f * (3 + 2) * params.viscosityF;
				tmp_xAccViscB *= 2.0f * (3 + 2) * params.viscosityB;
				tmp_yAccViscB *= 2.0f * (3 + 2) * params.viscosityB;
				tmp_zAccViscB *= 2.0f * (3 + 2) * params.viscosityB;

				xAccViscF_[i] = tmp_xAccViscF + tmp_xAccViscB;
				yAccViscF_[i] = tmp_yAccViscF + tmp_yAccViscB;
				zAccViscF_[i] = tmp_zAccViscF + tmp_zAccViscB;

				// TODO: add surface tension acc contribution
				xAccExtF_[i] = params.gravity.x;
				yAccExtF_[i] = params.gravity.y;
				zAccExtF_[i] = params.gravity.z;

				// compute predicted velocity from non-pressure accelerations
				xVelPredF_[i] = xVelF_[i] + params.timeStep * (xAccViscF_[i] + xAccExtF_[i]);
				yVelPredF_[i] = yVelF_[i] + params.timeStep * (yAccViscF_[i] + yAccExtF_[i]);
				zVelPredF_[i] = zVelF_[i] + params.timeStep * (zAccViscF_[i] + zAccExtF_[i]);
			}
		},
		tbb::auto_partitioner{}
	);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Viscosity_Acceleration,
								   Bench::BenchOp::External_Acceleration });

	// --- [ source-term for density invariance ] ------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_SourceTerm });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float tmp_SourceTerm = 0.0f;

				const float vpxi = xVelPredF_[i];
				const float vpyi = yVelPredF_[i];
				const float vpzi = zVelPredF_[i];

				// contributions from fluid neighbors
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
				#pragma omp simd reduction(+:tmp_SourceTerm)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFF_[i][k];

					// predicted velocity difference
					const float dvpx = vpxi - xVelPredF_[j];
					const float dvpy = vpyi - yVelPredF_[j];
					const float dvpz = vpzi - zVelPredF_[j];

					tmp_SourceTerm += massF * (dvpx * xKernelGradFF_[i][k]
											+ dvpy * yKernelGradFF_[i][k]
											+ dvpz * zKernelGradFF_[i][k]);
				}

				// contributions from boundary neighbors
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
				#pragma omp simd reduction(+:tmp_SourceTerm)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// TODO: subtract actual boundary velocity when adding non-static boundaries?
					tmp_SourceTerm += massB * (vpxi * xKernelGradFB_[i][k]
											+ vpyi * yKernelGradFB_[i][k]
											+ vpzi * zKernelGradFB_[i][k]);
				}

				// finalize source term
				sourceTermF_[i] = params.restDensityF - densityF_[i]
								- params.timeStep * tmp_SourceTerm;
			}
		},
		tbb::auto_partitioner{}
	);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_SourceTerm });

	// --- [ diagonal elements of system matrix ] ------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_DiagonalElem });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float tmp_DiagElem = 0.0f;

				float xCTermF = 0.0f;
				float yCTermF = 0.0f;
				float zCTermF = 0.0f;

				// add c-term contributions from fluid neighbors
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
				#pragma omp simd reduction(+:xCTermF,yCTermF,zCTermF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					xCTermF += massF * xKernelGradFF_[i][k];
					yCTermF += massF * yKernelGradFF_[i][k];
					zCTermF += massF * zKernelGradFF_[i][k];
				}

				float xCTermB = 0.0f;
				float yCTermB = 0.0f;
				float zCTermB = 0.0f;

				// add c-term contributions from boundary neighbors
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
				#pragma omp simd reduction(+:xCTermB,yCTermB,zCTermB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					xCTermB += massB * xKernelGradFB_[i][k];
					yCTermB += massB * yKernelGradFB_[i][k];
					zCTermB += massB * zKernelGradFB_[i][k];
				}

				// finalize c-term
				const float xCTerm = -1.0f * (xCTermF * restDensitySquInvF)
								   - 2.0f * params.gammaPress * (xCTermB * restDensitySquInvF);
				const float yCTerm = -1.0f * (yCTermF * restDensitySquInvF)
								   - 2.0f * params.gammaPress * (yCTermB * restDensitySquInvF);
				const float zCTerm = -1.0f * (zCTermF * restDensitySquInvF)
								   - 2.0f * params.gammaPress * (zCTermB * restDensitySquInvF);

				// part 1
				#pragma omp simd reduction(+:tmp_DiagElem)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					tmp_DiagElem += massF * (xCTerm * xKernelGradFF_[i][k]
										  + yCTerm * yKernelGradFF_[i][k]
										  + zCTerm * zKernelGradFF_[i][k]);
				}

				// part 2
				#pragma omp simd reduction(+:tmp_DiagElem)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					const float tmp_x = (massF * restDensitySquInvF) * (-xKernelGradFF_[i][k]);
					const float tmp_y = (massF * restDensitySquInvF) * (-yKernelGradFF_[i][k]);
					const float tmp_z = (massF * restDensitySquInvF) * (-zKernelGradFF_[i][k]);

					tmp_DiagElem += massF * (tmp_x * xKernelGradFF_[i][k] 
										  + tmp_y * yKernelGradFF_[i][k]
										  + tmp_z * zKernelGradFF_[i][k]);
				}

				// part 3
				#pragma omp simd reduction(+:tmp_DiagElem)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					tmp_DiagElem += massB * (xCTerm * xKernelGradFB_[i][k] 
										  + yCTerm * yKernelGradFB_[i][k]
										  + zCTerm * zKernelGradFB_[i][k]);
				}

				// finalize diagonal element
				diagElemF_[i] = params.timeStep * params.timeStep * tmp_DiagElem;
			}
		},
		tbb::auto_partitioner{}
	);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_DiagonalElem });

	// --- [ initialize pressure for jacobi solver ] -----------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_JacobiSolver });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float tmpPressure = 0.0f;

				// only update pressure if particle has neighbors
				if (!Base::ApproxEqu(diagElemF_[i], 0.0f))
				{
					tmpPressure = params.jacobiOmega * (sourceTermF_[i] / diagElemF_[i]);
				}

				// clamp pressure to non-negative values
				pressureF_[i] = fmaxf(tmpPressure, 0.0f);
			}
		},
		tbb::auto_partitioner{}
	);

	// --- [ relaxed jacobi solver for pressure ] --------------------------

	int cnt = 0;
	while (cnt < 2) // TODO: add proper termination condition
	{
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_F_),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const float presi = pressureF_[i];
					const float densi = params.restDensityF;

					float tmp_xAccPressF = 0.0f;
					float tmp_yAccPressF = 0.0f;
					float tmp_zAccPressF = 0.0f;

					// contributions from fluid neighbors
					const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
					#pragma omp simd reduction(+:tmp_xAccPressF,tmp_yAccPressF,tmp_zAccPressF)
					for (uint32_t k = 0; k < N_FF; k++)
					{
						// global index of neighbor particle
						const uint32_t j = neighborsFF_[i][k];

						// pressure acc contribution from fluid neighbor j
						const float accCoeff = massF * (presi / (densi * densi) 
											 + pressureF_[j] / (densi * densi));

						tmp_xAccPressF += accCoeff * xKernelGradFF_[i][k];
						tmp_yAccPressF += accCoeff * yKernelGradFF_[i][k];
						tmp_zAccPressF += accCoeff * zKernelGradFF_[i][k];
					}

					// contributions from boundary neighbors
					const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
					#pragma omp simd reduction(+:tmp_xAccPressF,tmp_yAccPressF,tmp_zAccPressF)
					for (uint32_t k = 0; k < N_FB; k++)
					{
						// pressure acc contribution from boundary neighbor j
						const float accCoeff = params.gammaPress * massB
											 * 2.0f * (presi / (densi * densi));

						tmp_xAccPressF += accCoeff * xKernelGradFB_[i][k];
						tmp_yAccPressF += accCoeff * yKernelGradFB_[i][k];
						tmp_zAccPressF += accCoeff * zKernelGradFB_[i][k];
					}

					// finalize pressure acceleration
					tmp_xAccPressF *= -1.0f;
					tmp_yAccPressF *= -1.0f;
					tmp_zAccPressF *= -1.0f;

					xAccPressF_[i] = tmp_xAccPressF;
					yAccPressF_[i] = tmp_yAccPressF;
					zAccPressF_[i] = tmp_zAccPressF;
				}
			},
			tbb::auto_partitioner{}
		);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_F_),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					float tmp_SysVecElem = 0.0f;

					const float axi = xAccPressF_[i];
					const float ayi = yAccPressF_[i];
					const float azi = zAccPressF_[i];

					// contributions from fluid neighbors
					const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
					#pragma omp simd reduction(+:tmp_SysVecElem)
					for (uint32_t k = 0; k < N_FF; k++)
					{
						// global index of neighbor particle
						const uint32_t j = neighborsFF_[i][k];

						// pressure acc difference
						const float dapx = axi - xAccPressF_[j];
						const float dapy = ayi - yAccPressF_[j];
						const float dapz = azi - zAccPressF_[j];

						tmp_SysVecElem += massF * (dapx * xKernelGradFF_[i][k] 
												+ dapy * yKernelGradFF_[i][k]
												+ dapz * zKernelGradFF_[i][k]);
					}

					// contributions from boundary neighbors
					const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
					#pragma omp simd reduction(+:tmp_SysVecElem)
					for (uint32_t k = 0; k < N_FB; k++)
					{
						// TODO: subtract actual boundary pressure acc when adding non-static boundaries?
						tmp_SysVecElem += massB * (axi * xKernelGradFB_[i][k] 
												+ ayi * yKernelGradFB_[i][k]
												+ azi * zKernelGradFB_[i][k]);
					}

					// finalize system vector element
					sysVecElemF_[i] = params.timeStep * params.timeStep * tmp_SysVecElem;

					float tmpPressure = 0.0f;

					// only update pressure if particle has neighbors
					if (!Base::ApproxEqu(diagElemF_[i], 0.0f))
					{
						tmpPressure = pressureF_[i] + params.jacobiOmega
									* ((sourceTermF_[i] - sysVecElemF_[i]) / diagElemF_[i]);
					}

					// clamp pressure to non-negative values
					pressureF_[i] = fmaxf(tmpPressure, 0.0f);
				}
			},
			tbb::auto_partitioner{}
		);

		cnt++;
	}
	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_JacobiSolver });

	// --- [ finalize pressure acceleration for integration ] ------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_Acceleration });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const float presi = pressureF_[i];
				const float densi = params.restDensityF;

				float tmp_xAccPressF = 0.0f;
				float tmp_yAccPressF = 0.0f;
				float tmp_zAccPressF = 0.0f;

				// contributions from fluid neighbors
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
				#pragma omp simd reduction(+:tmp_xAccPressF,tmp_yAccPressF,tmp_zAccPressF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFF_[i][k];

					// pressure acc contribution from fluid neighbor j
					const float accCoeff = massF * (presi / (densi * densi) 
										 + pressureF_[j] / (densi * densi));

					tmp_xAccPressF += accCoeff * xKernelGradFF_[i][k];
					tmp_yAccPressF += accCoeff * yKernelGradFF_[i][k];
					tmp_zAccPressF += accCoeff * zKernelGradFF_[i][k];
				}

				// contributions from boundary neighbors
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
				#pragma omp simd reduction(+:tmp_xAccPressF,tmp_yAccPressF,tmp_zAccPressF)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// pressure acc contribution from boundary neighbor j
					const float accCoeff = params.gammaPress * massB
										 * 2.0f * (presi / (densi * densi));

					tmp_xAccPressF += accCoeff * xKernelGradFB_[i][k];
					tmp_yAccPressF += accCoeff * yKernelGradFB_[i][k];
					tmp_zAccPressF += accCoeff * zKernelGradFB_[i][k];
				}

				// finalize pressure acceleration
				tmp_xAccPressF *= -1.0f;
				tmp_yAccPressF *= -1.0f;
				tmp_zAccPressF *= -1.0f;

				xAccPressF_[i] = tmp_xAccPressF;
				yAccPressF_[i] = tmp_yAccPressF;
				zAccPressF_[i] = tmp_zAccPressF;

				// finalize total acceleration
				xAccF_[i] = tmp_xAccPressF + xAccViscF_[i] + xAccExtF_[i];
				yAccF_[i] = tmp_yAccPressF + yAccViscF_[i] + yAccExtF_[i];
				zAccF_[i] = tmp_zAccPressF + zAccViscF_[i] + zAccExtF_[i];
			}
		},
		tbb::auto_partitioner{}
	);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_Acceleration });

	// --- [ velocity and position updates ] -----------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::VelocityUpdate,
									Bench::BenchOp::PositionUpdate });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// explicit Euler integration
				// (now only refining predicted velocity with pressure acc)
				xVelF_[i] = xVelPredF_[i] + params.timeStep * xAccPressF_[i];
				yVelF_[i] = yVelPredF_[i] + params.timeStep * yAccPressF_[i];
				zVelF_[i] = zVelPredF_[i] + params.timeStep * zAccPressF_[i];

				// semi-implicit Euler integration
				// (using updated velocity for position update)
				xPosF_[i] = xPosF_[i] + params.timeStep * xVelF_[i];
				yPosF_[i] = yPosF_[i] + params.timeStep * yVelF_[i];
				zPosF_[i] = zPosF_[i] + params.timeStep * zVelF_[i];
			}
		},
		tbb::auto_partitioner{}
	);

	benchmarksMgr_.StopBench({ Bench::BenchOp::VelocityUpdate,
								   Bench::BenchOp::PositionUpdate });
}
