// sph_solver_legacy.h

#pragma once

// external
#include <cstdint>
#include <tbb/partitioner.h>

// internal
#include "source/simulation/CPU/backend/sph_solver.h"



namespace CPU
{
	void Density(
		const float* RESTRICT xPosF,
		const float* RESTRICT yPosF,
		const float* RESTRICT zPosF,
		const float* RESTRICT xPosB,
		const float* RESTRICT yPosB,
		const float* RESTRICT zPosB,
		const Sim::I_NeighborList& neighborList,
		float particleSizeInvF, float kernelRadiusSqu,
		float cubicSplineAlpha, float massF, float massB,
		uint32_t N_F,
		// destinations
		float* RESTRICT densityF);

	void NonPressureAcceleration(
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
		const Sim::I_NeighborList& neighborList,
		float timeStep, float particleSizeInvF,
		float particleSizeSquF, float kernelRadiusSqu,
		float cubicSplineAlpha, float restDensity,
		float massF, float massB,
		float viscosityF, float viscosityB,
		float xGravity, float yGravity, float zGravity,
		uint32_t N_F,
		// destinations
		float* RESTRICT xVelPredF,
		float* RESTRICT yVelPredF,
		float* RESTRICT zVelPredF);

	void SourceTermAndDiagonalElementAndPressureJacobiInit(
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
		const Sim::I_NeighborList& neighborList,
		float timeStep, float particleSizeInvF,
		float kernelRadiusSqu, float cubicSplineAlpha,
		float restDensity, float restDensitySquInv,
		float massF, float massB,
		float pressGamma, float jacobiOmega,
		uint32_t N_F,
		// destinations
		float* RESTRICT sourceTermF,
		float* RESTRICT diagElemF,
		float* RESTRICT pressureF);

	void PressureAcceleration(
		const float* RESTRICT xPosF,
		const float* RESTRICT yPosF,
		const float* RESTRICT zPosF,
		const float* RESTRICT xPosB,
		const float* RESTRICT yPosB,
		const float* RESTRICT zPosB,
		const float* RESTRICT pressureF,
		const Sim::I_NeighborList& neighborList,
		float particleSizeInvF, float kernelRadiusSqu,
		float cubicSplineAlpha, float restDensity,
		float massF, float massB, float pressGamma,
		uint32_t N_F,
		tbb::affinity_partitioner& affPart,
		// destinations
		float* RESTRICT xAccPressF,
		float* RESTRICT yAccPressF,
		float* RESTRICT zAccPressF);

	void PressureRelaxedJacobiSolver(
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
		const Sim::I_NeighborList& neighborList,
		float timeStep, float particleSizeInvF,
		float kernelRadiusSqu, float cubicSplineAlpha,
		float restDensity, float massF, float massB,
		float jacobiOmega,
		uint32_t N_F,
		tbb::affinity_partitioner& affPart,
		// destinations
		float* RESTRICT pressureF,
		float& avgRelDensityErrorF_raw,
		float& avgRelDensityErrorF_pos,
		float& avgRelDensityErrorF_abs);

} // namespace CPU
