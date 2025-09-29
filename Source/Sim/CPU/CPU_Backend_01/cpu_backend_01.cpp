// cpu_backend_01.h

// external
#include <glm/exponential.hpp>
#include <glm/vec3.hpp>
#include <numbers>
#include <omp.h>
#include <spdlog/spdlog.h>

// internal
#include "Source/Base/helpers.h"
#include "cpu_backend_01.h"

constexpr float  PI = std::numbers::pi_v<float>;



using namespace Sim;

void SimBackendCPU_01::PullFromUniversalSimState(const Sync::UniversalSimState& src)
{
	const auto& F = src.fluidParticles;
	const auto& B = src.boundaryParticles;
	N_F_ = F.N_;
	N_B_ = B.N_;
	simStep_ = src.step;

	// resize backend fluid data structures
	xPosF_.resize(N_F_); yPosF_.resize(N_F_); zPosF_.resize(N_F_);
	xVelF_.resize(N_F_); yVelF_.resize(N_F_); zVelF_.resize(N_F_);
	xAccF_.resize(N_F_); yAccF_.resize(N_F_); zAccF_.resize(N_F_);
	xAccPressF_.resize(N_F_); yAccPressF_.resize(N_F_); zAccPressF_.resize(N_F_);
	xAccViscF_.resize(N_F_); yAccViscF_.resize(N_F_); zAccViscF_.resize(N_F_);
	xAccExtF_.resize(N_F_); yAccExtF_.resize(N_F_); zAccExtF_.resize(N_F_);
	densityF_.resize(N_F_);
	pressureF_.resize(N_F_);
	massF_.resize(N_F_);
	sizeF_.resize(N_F_);
	idF_.resize(N_F_);

	// resize backend boundary data structures
	xPosB_.resize(N_B_); yPosB_.resize(N_B_); zPosB_.resize(N_B_);
	xVelB_.resize(N_B_); yVelB_.resize(N_B_); zVelB_.resize(N_B_);
	densityB_.resize(N_B_);
	pressureB_.resize(N_B_);
	massB_.resize(N_B_);
	sizeB_.resize(N_B_);
	idB_.resize(N_B_);

	// parallel region for the copy
	#pragma omp parallel
	{
		// fluid particles
		#pragma omp for schedule(static)
		for (uint32_t i = 0; i < N_F_; i++)
		{
			xPosF_[i] = F.xPos[i]; yPosF_[i] = F.yPos[i]; zPosF_[i] = F.zPos[i];
			xVelF_[i] = F.xVel[i]; yVelF_[i] = F.yVel[i]; zVelF_[i] = F.zVel[i];
			xAccF_[i] = F.xAcc[i]; yAccF_[i] = F.yAcc[i]; zAccF_[i] = F.zAcc[i];
			xAccPressF_[i] = F.xAccPress[i]; yAccPressF_[i] = F.yAccPress[i]; zAccPressF_[i] = F.zAccPress[i];
			xAccViscF_[i] = F.xAccVisc[i]; yAccViscF_[i] = F.yAccVisc[i]; zAccViscF_[i] = F.zAccVisc[i];
			xAccExtF_[i] = F.xAccExt[i]; yAccExtF_[i] = F.yAccExt[i]; zAccExtF_[i] = F.zAccExt[i];
			densityF_[i] = F.density[i];
			pressureF_[i] = F.pressure[i];
			massF_[i] = F.mass[i];
			sizeF_[i] = F.size[i];
			idF_[i] = F.id[i];
		}

		// boundary particles
		#pragma omp for schedule(static)
		for (uint32_t i = 0; i < N_B_; i++)
		{
			xPosB_[i] = B.xPos[i]; yPosB_[i] = B.yPos[i]; zPosB_[i] = B.zPos[i];
			xVelB_[i] = B.xVel[i]; yVelB_[i] = B.yVel[i]; zVelB_[i] = B.zVel[i];
			densityB_[i] = B.density[i];
			pressureB_[i] = B.pressure[i];
			massB_[i] = B.mass[i];
			sizeB_[i] = B.size[i];
			idB_[i] = B.id[i];
		}
	}

	SPDLOG_INFO("Pulled universal sim state with {} fluid and "
				"{} boundary particles into data structures of backend CPU_01",
				N_F_, N_B_);
}

void SimBackendCPU_01::PushToUniversalSimState(Sync::UniversalSimState& dst) const
{
	auto& F = dst.fluidParticles;
	auto& B = dst.boundaryParticles;
	dst.ResizeFluid(N_F_);
	dst.ResizeBoundary(N_B_);

	// parallel region for the copy
	#pragma omp parallel
	{
		// fluid particles
		#pragma omp for schedule(static)
		for (uint32_t i = 0; i < N_F_; i++)
		{
			F.xPos[i] = xPosF_[i]; F.yPos[i] = yPosF_[i]; F.zPos[i] = zPosF_[i];
			F.xVel[i] = xVelF_[i]; F.yVel[i] = yVelF_[i]; F.zVel[i] = zVelF_[i];
			F.xAcc[i] = xAccF_[i]; F.yAcc[i] = yAccF_[i]; F.zAcc[i] = zAccF_[i];
			F.xAccPress[i] = xAccPressF_[i]; F.yAccPress[i] = yAccPressF_[i]; F.zAccPress[i] = zAccPressF_[i];
			F.xAccVisc[i] = xAccViscF_[i]; F.yAccVisc[i] = yAccViscF_[i]; F.zAccVisc[i] = zAccViscF_[i];
			F.xAccExt[i] = xAccExtF_[i]; F.yAccExt[i] = yAccExtF_[i]; F.zAccExt[i] = zAccExtF_[i];
			F.density[i] = densityF_[i];
			F.pressure[i] = pressureF_[i];
			F.mass[i] = massF_[i];
			F.size[i] = sizeF_[i];
			F.id[i] = idF_[i];
		}

		// boundary particles
		#pragma omp for schedule(static)
		for (uint32_t i = 0; i < N_B_; i++)
		{
			B.xPos[i] = xPosB_[i]; B.yPos[i] = yPosB_[i]; B.zPos[i] = zPosB_[i];
			B.xVel[i] = xVelB_[i]; B.yVel[i] = yVelB_[i]; B.zVel[i] = zVelB_[i];
			B.density[i] = densityB_[i];
			B.pressure[i] = pressureB_[i];
			B.mass[i] = massB_[i];
			B.size[i] = sizeB_[i];
			B.id[i] = idB_[i];
		}
	}

	F.N_ = N_F_;
	B.N_ = N_B_;

	SPDLOG_INFO("Pushed from data structures of backend CPU_01 with "
				"{} fluid and {} boundary particles to universal sim state",
				N_F_, N_B_);
}

void SimBackendCPU_01::PackExportSnapshot(Sync::ExportSnapshot& snap) const
{
	// TODO: implement
}

void SimBackendCPU_01::PackVisualSnapshot(Sync::VisualSnapshot& snap) const
{
	// TODO: add boundary particles

	snap.posF.resize(N_F_);
	snap.scalarDataF.resize(N_F_);
	snap.N_F = N_F_;
	snap.simStep = simStep_;

	#pragma omp parallel for schedule(static)
	for (uint32_t i = 0; i < N_F_; i++)
	{
		snap.posF[i] = glm::vec3{ xPosF_[i], yPosF_[i], zPosF_[i] };
	}

	FillScalarData(snap, snap.scalarModeF);
}

void SimBackendCPU_01::FillScalarData(Sync::VisualSnapshot& snap,
									  Base::ColorCodingMode mode) const
{
	switch (mode)
	{
	case Base::None:
	{
		snap.scalarDataF.clear();
		break;
	}
	case Base::VelocityMag:
	{
		#pragma omp parallel for schedule(static)
		for (uint32_t i = 0; i < N_F_; i++)
		{
			// magnitude of velocity vector
			snap.scalarDataF[i] = glm::sqrt(xVelF_[i] * xVelF_[i]
											 + yVelF_[i] * yVelF_[i]
											 + zVelF_[i] * zVelF_[i]);
		}
		break;
	}
	case Base::Density:
	{
		snap.scalarDataF = densityF_;
		break;
	}
	case Base::Pressure:
	{
		snap.scalarDataF = pressureF_;
		break;
	}
	default:
		SPDLOG_ERROR("Unsupported scalar data mode in backend CPU_01");
		break;
	}
}

void SimBackendCPU_01::SimStep(SimParams params, int64_t N_steps)
{
	// initialization and checks -----------------------------------------------

	// ensure neighbor storage has enough capacity
	if (N_F_ != static_cast<uint32_t>(neighborsFF_.size()) ||
		N_F_ != static_cast<uint32_t>(neighborsFB_.size()) ||
		N_F_ != static_cast<uint32_t>(kernelWeightsFF_.size()) ||
		N_F_ != static_cast<uint32_t>(kernelWeightsFB_.size()) ||
		N_F_ != static_cast<uint32_t>(kernelGradientsFF_.size()) ||
		N_F_ != static_cast<uint32_t>(kernelGradientsFB_.size()))
	{
		neighborsFF_.resize(N_F_);
		neighborsFB_.resize(N_F_);
		kernelWeightsFF_.resize(N_F_);
		kernelWeightsFB_.resize(N_F_);
		kernelGradientsFF_.resize(N_F_);
		kernelGradientsFB_.resize(N_F_);

		// expected number of fluid and boundary neighbors
		const uint32_t N_expectedNumFF =
			std::ceilf(2.0f * PI *
			std::powf(params.kernelRadius, 3.0f));

		const uint32_t N_expectedNumFB =
			std::ceilf(0.5f * PI *
			std::powf(params.kernelRadius, 3.0f));

		// heuristic reserves (kept and re-used across steps)
		#pragma omp parallel for schedule(static)
		for (uint32_t i = 0; i < N_F_; i++)
		{
			neighborsFF_[i].reserve(N_expectedNumFF);
			neighborsFB_[i].reserve(N_expectedNumFB);
			kernelWeightsFF_[i].reserve(N_expectedNumFF);
			kernelWeightsFB_[i].reserve(N_expectedNumFB);
			kernelGradientsFF_[i].reserve(N_expectedNumFF);
			kernelGradientsFB_[i].reserve(N_expectedNumFB);
		}
	}

	for (int64_t step = 0; step < N_steps; step++)
	{
		// one simulation step with one parallel region
		#pragma omp parallel
		{
			// temporary pre-computed constants --------------------------------

			const float kernelRadiusSqu =
				params.kernelRadius * params.kernelRadius;

			// cubic spline alpha (computed in double precision for accuracy)
			const double size = static_cast<double>(params.particleSizeF);
			const double sizeInv = 1.0 / size;
			const float cubicSplineAlpha = static_cast<float>(
				(1.0 / (4.0 * static_cast<double>(PI))) * sizeInv * sizeInv * sizeInv);

			const float sizeF = params.particleSizeF;

			const float sizeFSqu = params.particleSizeF * params.particleSizeF;

			const float massF = params.massF;
			const float massB = params.massB;

			// neighbor search -------------------------------------------------

			// naive brute-force neighbor search
			#pragma omp for schedule(static)
			for (uint32_t i = 0; i < N_F_; i++)
			{
				// neighbor lists for particle i
				// (delete contents, but keep capacity)
				auto& ff = neighborsFF_[i];
				auto& fb = neighborsFB_[i];
				ff.clear();
				fb.clear();

				const float xi = xPosF_[i];
				const float yi = yPosF_[i];
				const float zi = zPosF_[i];

				// check all fluid particles j
				for (uint32_t j = 0; j < N_F_; j++)
				{
					const float dx = xi - xPosF_[j];
					const float dy = yi - yPosF_[j];
					const float dz = zi - zPosF_[j];

					// squared distance between particles i and j
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						ff.push_back(j);

						// TODO: re-use distance to compute kernel weights?
					}
				}

				// check all boundary particles j
				for (uint32_t j = 0; j < N_B_; j++)
				{
					const float dx = xi - xPosB_[j];
					const float dy = yi - yPosB_[j];
					const float dz = zi - zPosB_[j];

					// squared distance between particles i and j
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						fb.push_back(j);

						// TODO: re-use distance to compute kernel weights?
					}
				}
			}

			// kernel weight and gradient computations -------------------------

			#pragma omp for schedule(static)
			for (uint32_t i = 0; i < N_F_; i++)
			{
				// resize to current number of neighbors
				kernelWeightsFF_[i].resize(neighborsFF_[i].size());
				kernelWeightsFB_[i].resize(neighborsFB_[i].size());
				kernelGradientsFF_[i].resize(neighborsFF_[i].size());
				kernelGradientsFB_[i].resize(neighborsFB_[i].size());

				// loop over all fluid neighbors
				for (uint32_t k = 0; k < neighborsFF_[i].size(); k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFF_[i][k];

					const float xi = xPosF_[i];
					const float yi = yPosF_[i];
					const float zi = zPosF_[i];

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
					kernelGradientsFF_[i][k] = cubicSplineAlpha
											 * (posDiff / (distNorm * sizeF * sizeF))
											 * (-3.0f * t2 * t2 + 12 * t1 * t1);
				}

				// loop over all boundary neighbors
				for (uint32_t k = 0; k < neighborsFB_[i].size(); k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFB_[i][k];

					const float xi = xPosF_[i];
					const float yi = yPosF_[i];
					const float zi = zPosF_[i];

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
					kernelGradientsFB_[i][k] = cubicSplineAlpha
											 * (posDiff / (distNorm * sizeFSqu))
											 * (-3.0f * t2 * t2 + 12 * t1 * t1);
				}
			}

			// density and pressure computations -------------------------------

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

				const float pressure = params.stiffness * std::powf(
					density / params.restDensityF, params.taitExponent) - 1.0f;

				pressureF_[i] = std::max(pressure, 0.0f);
			}

			// acceleration computations ---------------------------------------

			#pragma omp for schedule(static)
			for (uint32_t i = 0; i < N_F_; i++)
			{
				glm::vec3 accPress = { 0.0f, 0.0f, 0.0f };
				glm::vec3 accViscF = { 0.0f, 0.0f, 0.0f };
				glm::vec3 accViscB = { 0.0f, 0.0f, 0.0f };

				// loop over all fluid neighbors
				for (uint32_t k = 0; k < neighborsFF_[i].size(); k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFF_[i][k];

					const float xi = xPosF_[i];
					const float yi = yPosF_[i];
					const float zi = zPosF_[i];

					const float dx = xi - xPosF_[j];
					const float dy = yi - yPosF_[j];
					const float dz = zi - zPosF_[j];

					const glm::vec3 posDiff = { dx, dy, dz };
					const glm::vec3 velDiff = { xVelF_[i] - xVelF_[j],
											    yVelF_[i] - yVelF_[j],
											    zVelF_[i] - zVelF_[j] };

					// viscosity acc contribution from fluid neighbor j
					accViscF += (massF / densityF_[j])
							 * (glm::dot(velDiff, posDiff) 
							 / (glm::dot(posDiff, posDiff) 
							 + 0.01f * sizeFSqu))
							 * kernelGradientsFF_[i][k];

					// pressure acc contribution from fluid neighbor j
					accPress += massF
							 * (pressureF_[i] / (densityF_[i] * densityF_[i])
							 + pressureF_[j] / (densityF_[j] * densityF_[j]))
							 * kernelGradientsFF_[i][k];
				}

				// loop over all boundary neighbors
				for (uint32_t k = 0; k < neighborsFB_[i].size(); k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFB_[i][k];

					const float xi = xPosF_[i];
					const float yi = yPosF_[i];
					const float zi = zPosF_[i];

					const float dx = xi - xPosB_[j];
					const float dy = yi - yPosB_[j];
					const float dz = zi - zPosB_[j];

					const glm::vec3 posDiff = { dx, dy, dz };
					const glm::vec3 velDiff = { xVelF_[i] - xVelB_[j],
												yVelF_[i] - yVelB_[j],
												zVelF_[i] - zVelB_[j] };

					// viscosity acc contribution from boundary neighbor j
					accViscB += (massB / params.restDensityB)
							 * (glm::dot(velDiff, posDiff)
							 / (glm::dot(posDiff, posDiff)
							 + 0.01f * sizeFSqu))
							 * kernelGradientsFB_[i][k];

					// pressure acc contribution from boundary neighbor j
					accPress += massB
							 * (pressureF_[i] / (densityF_[i] * densityF_[i])
							 + pressureB_[j] / (params.restDensityB * params.restDensityB))
							 * kernelGradientsFB_[i][k];
				}

				// 3D -> factor 2 * (dim + 2)
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

				// accumulate final acceleration components
				xAccF_[i] = xAccPressF_[i] + xAccViscF_[i] + xAccExtF_[i];
				yAccF_[i] = yAccPressF_[i] + yAccViscF_[i] + yAccExtF_[i];
				zAccF_[i] = zAccPressF_[i] + zAccViscF_[i] + zAccExtF_[i];
			}

			// velocity and position updates -----------------------------------

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
		}
	}

	simStep_ += 1;
	simTime_ += N_steps * params.timeStep;
}
