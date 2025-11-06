// sph_solver.cpp

// external
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <omp.h>
#include <glm/vec3.hpp>

// internal
#include "Source/Base/helpers.h"
#include "neighbor_search.h"
#include "cpu_backend_01.h"



constexpr float PI = std::numbers::pi_v<float>;
using namespace Sim;

/*
void SimBackend_CPU01::Step_WCSPH(const SimParams& params)
{
	// one simulation step with one parallel region
	#pragma omp parallel
	{
		// --- [ temporary pre-computed constants ] ------------------------

		const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

		// cubic spline alpha (computed in double precision for accuracy)
		const double size = static_cast<double>(params.particleSizeF);
		const double sizeInv = 1.0 / size;
		const float cubicSplineAlpha = static_cast<float>(
			(1.0 / (4.0 * static_cast<double>(PI))) * sizeInv * sizeInv * sizeInv);

		const float sizeF = params.particleSizeF;

		const float sizeFSqu = params.particleSizeF * params.particleSizeF;

		const float massF = params.massF;
		const float massB = params.massB;

		// [ neighbor search ] ---------------------------------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Construction });
		#pragma omp single
		{
			neighborSearchStrat_->Construct();
		}
		#pragma omp barrier
		benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Construction });

		benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Query });
		neighborSearchStrat_->Query();
		benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Query });

		// [ kernel weight and gradient computations ] ---------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Kernel_Weights,
										Bench::BenchOp::Kernel_Gradients });
		#pragma omp for schedule(static)
		for (uint32_t i = 0; i < N_F_; i++)
		{
			// resize to current number of neighbors
			kernelWeightsFF_[i].resize(neighborsFF_[i].size());
			kernelWeightsFB_[i].resize(neighborsFB_[i].size());
			kernelGradientsFF_[i].resize(neighborsFF_[i].size());
			kernelGradientsFB_[i].resize(neighborsFB_[i].size());

			const float xi = xPosF_[i];
			const float yi = yPosF_[i];
			const float zi = zPosF_[i];

			// loop over all fluid neighbors
			for (uint32_t k = 0; k < neighborsFF_[i].size(); k++)
			{
				// global index of neighbor particle
				const uint32_t j = neighborsFF_[i][k];

				const float dx = xi - xPosF_[j];
				const float dy = yi - yPosF_[j];
				const float dz = zi - zPosF_[j];

				const float distNorm = std::sqrt(dx * dx + dy * dy + dz * dz)
									 / params.particleSizeF;

				// avoid division by zero in gradient computation
				if (Base::ApproxEqu(distNorm, 0.0f)) [[unlikely]]
				{
					kernelWeightsFF_[i][k] = cubicSplineAlpha * 4.0f;
					kernelGradientsFF_[i][k] = glm::vec3{ 0.0f, 0.0f, 0.0f };

					continue;
				}

				const glm::vec3 posDiff = { dx, dy, dz };

				const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
				const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

				// kernel weight
				kernelWeightsFF_[i][k] = cubicSplineAlpha
									   * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);

				// kernel gradient
				// TODO: multiply distNorm with size once or twice?
				kernelGradientsFF_[i][k] = cubicSplineAlpha
										 * (posDiff / (distNorm * sizeF * sizeF))
										 * (-3.0f * t2 * t2 + 12 * t1 * t1);
			}

			// loop over all boundary neighbors
			for (uint32_t k = 0; k < neighborsFB_[i].size(); k++)
			{
				// global index of neighbor particle
				const uint32_t j = neighborsFB_[i][k];

				const float dx = xi - xPosB_[j];
				const float dy = yi - yPosB_[j];
				const float dz = zi - zPosB_[j];

				const float distNorm = std::sqrt(dx * dx + dy * dy + dz * dz)
					/ params.particleSizeF;

				// avoid division by zero in gradient computation
				if (Base::ApproxEqu(distNorm, 0.0f)) [[unlikely]]
				{
					kernelWeightsFB_[i][k] = cubicSplineAlpha * 4.0f;
					kernelGradientsFB_[i][k] = glm::vec3{ 0.0f, 0.0f, 0.0f };

					continue;
				}

				const glm::vec3 posDiff = { dx, dy, dz };

				const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
				const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

				// kernel weight
				kernelWeightsFB_[i][k] = cubicSplineAlpha
					* (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);

				// kernel gradient
				// TODO: multiply distNorm with size once or twice?
				kernelGradientsFB_[i][k] = cubicSplineAlpha
										 * (posDiff / (distNorm * sizeF * sizeF))
										 * (-3.0f * t2 * t2 + 12 * t1 * t1);
			}
		}
		benchmarksMgr_.StopBench({ Bench::BenchOp::Kernel_Weights,
									   Bench::BenchOp::Kernel_Gradients });

		// [ density and pressure computations ] ---------------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Density_Solving,
										Bench::BenchOp::Pressure_Solving });
		#pragma omp for schedule(static)
		for (uint32_t i = 0; i < N_F_; i++)
		{
			float density = 0.0f;

			// loop over all fluid neighbors
			for (uint32_t k = 0; k < neighborsFF_[i].size(); k++)
			{
				// global index of neighbor particle
				const uint32_t j = neighborsFF_[i][k];

				density += massF * kernelWeightsFF_[i][k];
			}

			// loop over all boundary neighbors
			for (uint32_t k = 0; k < neighborsFB_[i].size(); k++)
			{
				// global index of neighbor particle
				const uint32_t j = neighborsFB_[i][k];

				density += massB * kernelWeightsFB_[i][k];
			}

			densityF_[i] = density;

			const float pressure = params.stiffness * (std::powf(
				density / params.restDensityF, params.taitExponent) - 1.0f);

			// TODO: do hard clamp, allow negative pressure, or use other fix?
			pressureF_[i] = std::max(pressure, 0.0f);
		}
		benchmarksMgr_.StopBench({ Bench::BenchOp::Density_Solving,
									   Bench::BenchOp::Pressure_Solving });

		// [ acceleration computations ] -----------------------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_Acceleration,
										Bench::BenchOp::Viscosity_Acceleration,
										Bench::BenchOp::External_Acceleration,
										Bench::BenchOp::AccelerationUpdate });
		#pragma omp for schedule(static)
		for (uint32_t i = 0; i < N_F_; i++)
		{
			glm::vec3 accPress = { 0.0f, 0.0f, 0.0f };
			glm::vec3 accViscF = { 0.0f, 0.0f, 0.0f };
			glm::vec3 accViscB = { 0.0f, 0.0f, 0.0f };

			const float xi = xPosF_[i];
			const float yi = yPosF_[i];
			const float zi = zPosF_[i];

			const float vxi = xVelF_[i];
			const float vyi = yVelF_[i];
			const float vzi = zVelF_[i];

			const float presi = pressureF_[i];
			const float densi = densityF_[i];

			// loop over all fluid neighbors
			for (uint32_t k = 0; k < neighborsFF_[i].size(); k++)
			{
				// global index of neighbor particle
				const uint32_t j = neighborsFF_[i][k];

				const float dx = xi - xPosF_[j];
				const float dy = yi - yPosF_[j];
				const float dz = zi - zPosF_[j];

				const glm::vec3 posDiff = { dx, dy, dz };
				const glm::vec3 velDiff = { vxi - xVelF_[j],
											vyi - yVelF_[j],
											vzi - zVelF_[j] };

				// viscosity acc contribution from fluid neighbor j
				accViscF += (massF / densityF_[j])
						  * (glm::dot(velDiff, posDiff)
						  / (glm::dot(posDiff, posDiff)
						  + 0.01f * sizeFSqu))
						  * kernelGradientsFF_[i][k];

				// pressure acc contribution from fluid neighbor j
				accPress += massF
						  * (presi / (densi * densi)
						  + pressureF_[j] / (densityF_[j] * densityF_[j]))
						  * kernelGradientsFF_[i][k];
			}

			// loop over all boundary neighbors
			for (uint32_t k = 0; k < neighborsFB_[i].size(); k++)
			{
				// global index of neighbor particle
				const uint32_t j = neighborsFB_[i][k];

				const float dx = xi - xPosB_[j];
				const float dy = yi - yPosB_[j];
				const float dz = zi - zPosB_[j];

				const glm::vec3 posDiff = { dx, dy, dz };
				const glm::vec3 velDiff = { vxi - xVelB_[j],
											vyi - yVelB_[j],
											vzi - zVelB_[j] };

				// viscosity acc contribution from boundary neighbor j
				accViscB += (massB / params.restDensityB)
						  * (glm::dot(velDiff, posDiff)
						  / (glm::dot(posDiff, posDiff)
						  + 0.01f * sizeFSqu))
						  * kernelGradientsFB_[i][k];

				// pressure acc contribution from boundary neighbor j
				accPress += massB
						  * (presi / (densi * densi)
						  + pressureB_[j] / (params.restDensityB * params.restDensityB))
						  * kernelGradientsFB_[i][k];
			}

			// finalize accelerations
			// (3D -> factor 2 * (dim + 2))
			accViscF *= 2.0f * (3 + 2) * params.viscosityF;
			accViscB *= 2.0f * (3 + 2) * params.viscosityB;
			accPress *= -1.0f;

			xAccViscF_[i] = accViscF.x + accViscB.x;
			yAccViscF_[i] = accViscF.y + accViscB.y;
			zAccViscF_[i] = accViscF.z + accViscB.z;

			xAccPressF_[i] = accPress.x;
			yAccPressF_[i] = accPress.y;
			zAccPressF_[i] = accPress.z;

			// TODO: add surface tension acc contribution
			xAccExtF_[i] = params.gravity.x;
			yAccExtF_[i] = params.gravity.y;
			zAccExtF_[i] = params.gravity.z;

			// sum final acceleration components
			xAccF_[i] = xAccPressF_[i] + xAccViscF_[i] + xAccExtF_[i];
			yAccF_[i] = yAccPressF_[i] + yAccViscF_[i] + yAccExtF_[i];
			zAccF_[i] = zAccPressF_[i] + zAccViscF_[i] + zAccExtF_[i];
		}
		benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_Acceleration,
									   Bench::BenchOp::Viscosity_Acceleration,
									   Bench::BenchOp::External_Acceleration,
									   Bench::BenchOp::AccelerationUpdate });

		// [ velocity and position updates ] -------------------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::VelocityUpdate,
										Bench::BenchOp::PositionUpdate });
		#pragma omp for schedule(static)
		for (uint32_t i = 0; i < N_F_; i++)
		{
			// explicit Euler integration
			xVelF_[i] = xVelF_[i] + params.timeStep * xAccF_[i];
			yVelF_[i] = yVelF_[i] + params.timeStep * yAccF_[i];
			zVelF_[i] = zVelF_[i] + params.timeStep * zAccF_[i];

			// semi-implicit Euler integration
			// (using updated velocity for position update)
			xPosF_[i] = xPosF_[i] + params.timeStep * xVelF_[i];
			yPosF_[i] = yPosF_[i] + params.timeStep * yVelF_[i];
			zPosF_[i] = zPosF_[i] + params.timeStep * zVelF_[i];
		}
		benchmarksMgr_.StopBench({ Bench::BenchOp::VelocityUpdate,
									   Bench::BenchOp::PositionUpdate });
	}
}
*/

void SimBackend_CPU01::Step_IISPH(const SimParams& params)
{
	// one simulation step with one parallel region
	#pragma omp parallel
	{
		// --- [ temporary pre-computed constants ] ------------------------

		const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

		// cubic spline alpha (computed in double precision for accuracy)
		const double size = static_cast<double>(params.particleSizeF);
		const double sizeInv = 1.0 / size;
		const float cubicSplineAlpha = static_cast<float>(
			(1.0 / (4.0 * static_cast<double>(PI))) * sizeInv * sizeInv * sizeInv);

		const float sizeF = params.particleSizeF;

		const float sizeFSqu = params.particleSizeF * params.particleSizeF;

		const float massF = params.massF;
		const float massB = params.massB;

		// [ neighbor search ] ---------------------------------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Construction });
		#pragma omp single
		{
			neighborSearchStrat_->Construct(params);
		}
		#pragma omp barrier
		benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Construction });

		benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Query });
		neighborSearchStrat_->Query(params);
		benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Query });

		// [ kernel weight and gradient computations ] ---------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Kernel_Weights,
										Bench::BenchOp::Kernel_Gradients });
		#pragma omp for schedule(static)
		for (uint32_t i = 0; i < N_F_; i++)
		{
			// resize to current number of neighbors
			kernelWeightsFF_[i].resize(neighborsFF_[i].size());
			kernelWeightsFB_[i].resize(neighborsFB_[i].size());
			kernelGradientsFF_[i].resize(neighborsFF_[i].size());
			kernelGradientsFB_[i].resize(neighborsFB_[i].size());

			const float xi = xPosF_[i];
			const float yi = yPosF_[i];
			const float zi = zPosF_[i];

			// loop over all fluid neighbors
			for (uint32_t k = 0; k < neighborsFF_[i].size(); k++)
			{
				// global index of neighbor particle
				const uint32_t j = neighborsFF_[i][k];

				const float dx = xi - xPosF_[j];
				const float dy = yi - yPosF_[j];
				const float dz = zi - zPosF_[j];

				const float distNorm = std::sqrt(dx * dx + dy * dy + dz * dz)
									 / params.particleSizeF;

				// avoid division by zero in gradient computation
				if (Base::ApproxEqu(distNorm, 0.0f)) [[unlikely]]
				{
					kernelWeightsFF_[i][k] = cubicSplineAlpha * 4.0f;
					kernelGradientsFF_[i][k] = glm::vec3{ 0.0f, 0.0f, 0.0f };

					continue;
				}

				const glm::vec3 posDiff = { dx, dy, dz };

				const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
				const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

				// kernel weight
				kernelWeightsFF_[i][k] = cubicSplineAlpha
									   * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);

				// kernel gradient
				// TODO: multiply distNorm with size once or twice?
				kernelGradientsFF_[i][k] = cubicSplineAlpha
										 * (posDiff / (distNorm * sizeF * sizeF))
										 * (-3.0f * t2 * t2 + 12 * t1 * t1);
			}

			// loop over all boundary neighbors
			for (uint32_t k = 0; k < neighborsFB_[i].size(); k++)
			{
				// global index of neighbor particle
				const uint32_t j = neighborsFB_[i][k];

				const float dx = xi - xPosB_[j];
				const float dy = yi - yPosB_[j];
				const float dz = zi - zPosB_[j];

				const float distNorm = std::sqrt(dx * dx + dy * dy + dz * dz)
									 / params.particleSizeF;

				// avoid division by zero in gradient computation
				if (Base::ApproxEqu(distNorm, 0.0f)) [[unlikely]]
				{
					kernelWeightsFB_[i][k] = cubicSplineAlpha * 4.0f;
					kernelGradientsFB_[i][k] = glm::vec3{ 0.0f, 0.0f, 0.0f };

					continue;
				}

				const glm::vec3 posDiff = { dx, dy, dz };

				const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
				const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

				// kernel weight
				kernelWeightsFB_[i][k] = cubicSplineAlpha
									   * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);

				// kernel gradient
				// TODO: multiply distNorm with size once or twice?
				kernelGradientsFB_[i][k] = cubicSplineAlpha
										 * (posDiff / (distNorm * sizeF * sizeF))
										 * (-3.0f * t2 * t2 + 12 * t1 * t1);
			}
		}
		benchmarksMgr_.StopBench({ Bench::BenchOp::Kernel_Weights,
									   Bench::BenchOp::Kernel_Gradients });

		// [ density and pressure computations ] ---------------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Density_Solving,
										Bench::BenchOp::Pressure_Solving });
		#pragma omp for schedule(static)
		for (uint32_t i = 0; i < N_F_; i++)
		{
			float density = 0.0f;

			// loop over all fluid neighbors
			for (uint32_t k = 0; k < neighborsFF_[i].size(); k++)
			{
				// global index of neighbor particle
				const uint32_t j = neighborsFF_[i][k];

				density += massF * kernelWeightsFF_[i][k];
			}

			// loop over all boundary neighbors
			for (uint32_t k = 0; k < neighborsFB_[i].size(); k++)
			{
				// global index of neighbor particle
				const uint32_t j = neighborsFB_[i][k];

				density += massB * kernelWeightsFB_[i][k];
			}

			densityF_[i] = density;

			const float pressure = params.stiffness * (std::powf(
				density / params.restDensityF, params.taitExponent) - 1.0f);

			// TODO: do hard clamp, allow negative pressure, or use other fix?
			pressureF_[i] = std::max(pressure, 0.0f);
		}
		benchmarksMgr_.StopBench({ Bench::BenchOp::Density_Solving,
									   Bench::BenchOp::Pressure_Solving });

		// [ acceleration computations ] -----------------------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_Acceleration,
										Bench::BenchOp::Viscosity_Acceleration,
										Bench::BenchOp::External_Acceleration,
										Bench::BenchOp::AccelerationUpdate });
		#pragma omp for schedule(static)
		for (uint32_t i = 0; i < N_F_; i++)
		{
			glm::vec3 accPress = { 0.0f, 0.0f, 0.0f };
			glm::vec3 accViscF = { 0.0f, 0.0f, 0.0f };
			glm::vec3 accViscB = { 0.0f, 0.0f, 0.0f };

			const float xi = xPosF_[i];
			const float yi = yPosF_[i];
			const float zi = zPosF_[i];

			const float vxi = xVelF_[i];
			const float vyi = yVelF_[i];
			const float vzi = zVelF_[i];

			const float presi = pressureF_[i];
			const float densi = densityF_[i];

			// loop over all fluid neighbors
			for (uint32_t k = 0; k < neighborsFF_[i].size(); k++)
			{
				// global index of neighbor particle
				const uint32_t j = neighborsFF_[i][k];

				const float dx = xi - xPosF_[j];
				const float dy = yi - yPosF_[j];
				const float dz = zi - zPosF_[j];

				const glm::vec3 posDiff = { dx, dy, dz };
				const glm::vec3 velDiff = { vxi - xVelF_[j],
											vyi - yVelF_[j],
											vzi - zVelF_[j] };

				// viscosity acc contribution from fluid neighbor j
				accViscF += (massF / densityF_[j])
						  * (glm::dot(velDiff, posDiff)
						  / (glm::dot(posDiff, posDiff)
						  + 0.01f * sizeFSqu))
						  * kernelGradientsFF_[i][k];

				// pressure acc contribution from fluid neighbor j
				accPress += massF
						  * (presi / (densi * densi)
						  + pressureF_[j] / (densityF_[j] * densityF_[j]))
						  * kernelGradientsFF_[i][k];
			}

			// loop over all boundary neighbors
			for (uint32_t k = 0; k < neighborsFB_[i].size(); k++)
			{
				// global index of neighbor particle
				const uint32_t j = neighborsFB_[i][k];

				const float dx = xi - xPosB_[j];
				const float dy = yi - yPosB_[j];
				const float dz = zi - zPosB_[j];

				const glm::vec3 posDiff = { dx, dy, dz };
				const glm::vec3 velDiff = { vxi - xVelB_[j],
											vyi - yVelB_[j],
											vzi - zVelB_[j] };

				// viscosity acc contribution from boundary neighbor j
				accViscB += (massB / params.restDensityB)
						  * (glm::dot(velDiff, posDiff)
						  / (glm::dot(posDiff, posDiff)
						  + 0.01f * sizeFSqu))
						  * kernelGradientsFB_[i][k];

				// pressure acc contribution from boundary neighbor j
				accPress += massB
						  * (presi / (densi * densi)
						  + pressureB_[j] / (params.restDensityB * params.restDensityB))
						  * kernelGradientsFB_[i][k];
			}

			// finalize accelerations
			// (3D -> factor 2 * (dim + 2))
			accViscF *= 2.0f * (3 + 2) * params.viscosityF;
			accViscB *= 2.0f * (3 + 2) * params.viscosityB;
			accPress *= -1.0f;

			xAccViscF_[i] = accViscF.x + accViscB.x;
			yAccViscF_[i] = accViscF.y + accViscB.y;
			zAccViscF_[i] = accViscF.z + accViscB.z;

			xAccPressF_[i] = accPress.x;
			yAccPressF_[i] = accPress.y;
			zAccPressF_[i] = accPress.z;

			// TODO: add surface tension acc contribution
			xAccExtF_[i] = params.gravity.x;
			yAccExtF_[i] = params.gravity.y;
			zAccExtF_[i] = params.gravity.z;

			// sum final acceleration components
			xAccF_[i] = xAccPressF_[i] + xAccViscF_[i] + xAccExtF_[i];
			yAccF_[i] = yAccPressF_[i] + yAccViscF_[i] + yAccExtF_[i];
			zAccF_[i] = zAccPressF_[i] + zAccViscF_[i] + zAccExtF_[i];
		}
		benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_Acceleration,
									   Bench::BenchOp::Viscosity_Acceleration,
									   Bench::BenchOp::External_Acceleration,
									   Bench::BenchOp::AccelerationUpdate });

		// [ velocity and position updates ] -------------------------------

		benchmarksMgr_.StartBench({ Bench::BenchOp::VelocityUpdate,
										Bench::BenchOp::PositionUpdate });
		#pragma omp for schedule(static)
		for (uint32_t i = 0; i < N_F_; i++)
		{
			// explicit Euler integration
			xVelF_[i] = xVelF_[i] + params.timeStep * xAccF_[i];
			yVelF_[i] = yVelF_[i] + params.timeStep * yAccF_[i];
			zVelF_[i] = zVelF_[i] + params.timeStep * zAccF_[i];

			// semi-implicit Euler integration
			// (using updated velocity for position update)
			xPosF_[i] = xPosF_[i] + params.timeStep * xVelF_[i];
			yPosF_[i] = yPosF_[i] + params.timeStep * yVelF_[i];
			zPosF_[i] = zPosF_[i] + params.timeStep * zVelF_[i];
		}
		benchmarksMgr_.StopBench({ Bench::BenchOp::VelocityUpdate,
									   Bench::BenchOp::PositionUpdate });
	}
}
