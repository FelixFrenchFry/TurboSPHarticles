// sph_solver.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>

// internal
#include "neighbor_lists.h"

#if defined(_MSC_VER)
	#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
	#define RESTRICT __restrict__
#else
	#define RESTRICT
#endif



namespace CPU01
{
	constexpr float ZERO_EQU_LIM_F = 1e-7f;
	constexpr double ZERO_EQU_LIM_D = 1e-14;

	inline void KernelWeight_CubicSpline(
		float dx, float dy, float dz,
		float particleSizeInv, float kernelRadiusSqu,
		float cubicSplineAlpha,
		// destinations
		float& kw) noexcept;

	inline void KernelGradient_CubicSpline(
		float dx, float dy, float dz,
		float particleSizeInv, float kernelRadiusSqu,
		float cubicSplineAlpha,
		// destinations
		float& xkg, float& ykg, float& zkg) noexcept;

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

	void Density_ChunkedSELL8(
		const float* RESTRICT xPosF,
		const float* RESTRICT yPosF,
		const float* RESTRICT zPosF,
		const float* RESTRICT xPosB,
		const float* RESTRICT yPosB,
		const float* RESTRICT zPosB,
		const Sim::I_NeighborListSELL8& neighborList,
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
		float cubicSplineAlpha, float restDensityF,
		float restDensityB, float massF, float massB,
		float viscosityF, float viscosityB,
		float xGravity, float yGravity, float zGravity,
		uint32_t N_F,
		// destinations
		float* RESTRICT xVelPredF,
		float* RESTRICT yVelPredF,
		float* RESTRICT zVelPredF);

	void NonPressureAcceleration_ChunkedSELL8(
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
		const Sim::I_NeighborListSELL8& neighborList,
		float timeStep, float particleSizeInvF,
		float particleSizeSquF, float kernelRadiusSqu,
		float cubicSplineAlpha, float restDensityF,
		float restDensityB, float massF, float massB,
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
		float restDensityF, float restDensitySquInvF,
		float massF, float massB,
		float pressGamma, float jacobiOmega,
		uint32_t N_F,
		// destinations
		float* RESTRICT sourceTermF,
		float* RESTRICT diagElemF,
		float* RESTRICT pressureF);

	void SourceTermAndDiagonalElementAndPressureJacobiInit_ChunkedSELL8(
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
		const Sim::I_NeighborListSELL8& neighborList,
		float timeStep, float particleSizeInvF,
		float kernelRadiusSqu, float cubicSplineAlpha,
		float restDensityF, float restDensitySquInvF,
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
		float cubicSplineAlpha, float restDensityF,
		float massF, float massB, float pressGamma,
		uint32_t N_F,
		tbb::affinity_partitioner& affPart,
		// destinations
		float* RESTRICT xAccPressF,
		float* RESTRICT yAccPressF,
		float* RESTRICT zAccPressF);

	void PressureAcceleration_ChunkedSELL8(
		const float* RESTRICT xPosF,
		const float* RESTRICT yPosF,
		const float* RESTRICT zPosF,
		const float* RESTRICT xPosB,
		const float* RESTRICT yPosB,
		const float* RESTRICT zPosB,
		const float* RESTRICT pressureF,
		const Sim::I_NeighborListSELL8& neighborList,
		float particleSizeInvF, float kernelRadiusSqu,
		float cubicSplineAlpha, float restDensityF,
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
		float restDensityF, float massF, float massB,
		float jacobiOmega,
		uint32_t N_F,
		tbb::affinity_partitioner& affPart,
		// destinations
		float* RESTRICT pressureF,
		float& avgRelDensityErrorF_raw,
		float& avgRelDensityErrorF_pos,
		float& avgRelDensityErrorF_abs);

	void PressureRelaxedJacobiSolver_ChunkedSELL8(
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
		const Sim::I_NeighborListSELL8& neighborList,
		float timeStep, float particleSizeInvF,
		float kernelRadiusSqu, float cubicSplineAlpha,
		float restDensityF, float massF, float massB,
		float jacobiOmega,
		uint32_t N_F,
		tbb::affinity_partitioner& affPart,
		// destinations
		float* RESTRICT pressureF,
		float& avgRelDensityErrorF_raw,
		float& avgRelDensityErrorF_pos,
		float& avgRelDensityErrorF_abs);

	void Integration(
		const float* RESTRICT xVelPredF,
		const float* RESTRICT yVelPredF,
		const float* RESTRICT zVelPredF,
		const float* RESTRICT xAccPressF,
		const float* RESTRICT yAccPressF,
		const float* RESTRICT zAccPressF,
		float timeStep,
		uint32_t N_F,
		// destinations
		float* RESTRICT xPosF,
		float* RESTRICT yPosF,
		float* RESTRICT zPosF,
		float* RESTRICT xVelF,
		float* RESTRICT yVelF,
		float* RESTRICT zVelF);

} // namespace CPU01
