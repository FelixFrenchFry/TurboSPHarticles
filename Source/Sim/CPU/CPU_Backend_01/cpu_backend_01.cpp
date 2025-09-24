// cpu_backend_01.h

// external
#include <glm/exponential.hpp>
#include <glm/vec3.hpp>
#include <omp.h>
#include <spdlog/spdlog.h>

// internal
#include "cpu_backend_01.h"



using namespace Sim;

SimBackendCPU_01::SimBackendCPU_01(const SimParams& simParams)
	: simParams_{ simParams }
{
	// nothing to do
}

void SimBackendCPU_01::PullFromUniversalSimState(const Sync::UniversalSimState& src)
{
	const auto& F = src.fluidParticles;
	const auto& B = src.boundaryParticles;
	N_F_ = F.N_;
	N_B_ = B.N_;
	step_ = src.step;

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
		for (std::int64_t i = 0; i < N_F_; i++)
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
		for (std::int64_t i = 0; i < N_B_; i++)
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
		for (std::int64_t i = 0; i < N_F_; i++)
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
		for (std::int64_t i = 0; i < N_B_; i++)
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

	snap.position.resize(N_F_);
	snap.scalarData.resize(N_F_);
	snap.N_particles = N_F_;
	snap.step = step_;

	#pragma omp parallel for schedule(static)
	for (std::int64_t i = 0; i < N_F_; i++)
	{
		snap.position[i] = glm::vec3{ xPosF_[i], yPosF_[i], zPosF_[i] };
	}

	FillScalarData(snap, snap.scalarMode);
}

void SimBackendCPU_01::FillScalarData(Sync::VisualSnapshot& snap,
									  Base::ColorCodingMode mode) const
{
	switch (mode)
	{
	case Base::None:
	{
		snap.scalarData.clear();
		break;
	}
	case Base::VelocityMag:
	{
		#pragma omp parallel for schedule(static)
		for (std::int64_t i = 0; i < N_F_; i++)
		{
			// magnitude of velocity vector
			snap.scalarData[i] = glm::sqrt(xVelF_[i] * xVelF_[i]
											+ yVelF_[i] * yVelF_[i]
											+ zVelF_[i] * zVelF_[i]);
		}
		break;
	}
	case Base::Density:
	{
		snap.scalarData = densityF_;
		break;
	}
	case Base::Pressure:
	{
		snap.scalarData = pressureF_;
		break;
	}
	default:
		SPDLOG_ERROR("Unsupported scalar data mode in backend CPU_01");
		break;
	}
}

void SimBackendCPU_01::SimStep(std::int64_t N_steps)
{
	for (std::int64_t step = 0; step < N_steps; step++)
	{
		// one simulation step with one parallel region
		#pragma omp parallel
		{
			// TODO: use OpenMP for/single/etc... for SPH computations
		}

		step_++;
	}
}
