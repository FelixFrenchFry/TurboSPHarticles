// sph_solver.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>
#include <vector>

// internal
#include "Source/Sim/Core/parameters_sim.h"

#if defined(_MSC_VER)
	#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
	#define RESTRICT __restrict__
#else
	#define RESTRICT
#endif



namespace CPU01
{
	void KernelWeightsAndGradients(
		const float* RESTRICT xPosF,
		const float* RESTRICT yPosF,
		const float* RESTRICT zPosF,
		const float* RESTRICT xPosB,
		const float* RESTRICT yPosB,
		const float* RESTRICT zPosB,
		const std::vector<std::vector<uint32_t>>& neighborsFF,
		const std::vector<std::vector<uint32_t>>& neighborsFB,
		float particleSizeF, float cubicSplineAlpha,
		uint32_t N_F,
		// destinations
		std::vector<std::vector<float>>& kernelWeightsFF,
		std::vector<std::vector<float>>& kernelWeightsFB,
		std::vector<std::vector<float>>& xKernelGradFF,
		std::vector<std::vector<float>>& yKernelGradFF,
		std::vector<std::vector<float>>& zKernelGradFF,
		std::vector<std::vector<float>>& xKernelGradFB,
		std::vector<std::vector<float>>& yKernelGradFB,
		std::vector<std::vector<float>>& zKernelGradFB);

	void Density(
		const std::vector<std::vector<float>>& kernelWeightsFF,
		const std::vector<std::vector<float>>& kernelWeightsFB,
		float massF, float massB,
		uint32_t N_F,
		// destinations
		float* RESTRICT densityF);

	void NonPressureAcceleration(
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
		float timeStep, float particleSizeF, float restDensityB,
		float massF, float massB, float viscosityF, float viscosityB,
		float xGravity, float yGravity, float zGravity,
		uint32_t N_F,
		// destinations
		float* RESTRICT xAccViscF,
		float* RESTRICT yAccViscF,
		float* RESTRICT zAccViscF,
		float* RESTRICT xAccExtF,
		float* RESTRICT yAccExtF,
		float* RESTRICT zAccExtF,
		float* RESTRICT xVelPredF,
		float* RESTRICT yVelPredF,
		float* RESTRICT zVelPredF);

	void SourceTerm(
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
		float timeStep, float restDensityF,
		float massF, float massB,
		uint32_t N_F,
		// destinations
		float* RESTRICT sourceTermF);

	void DiagonalElement(
		const std::vector<std::vector<std::uint32_t>>& neighborsFF,
		const std::vector<std::vector<std::uint32_t>>& neighborsFB,
		const std::vector<std::vector<float>>& xKernelGradFF,
		const std::vector<std::vector<float>>& yKernelGradFF,
		const std::vector<std::vector<float>>& zKernelGradFF,
		const std::vector<std::vector<float>>& xKernelGradFB,
		const std::vector<std::vector<float>>& yKernelGradFB,
		const std::vector<std::vector<float>>& zKernelGradFB,
		float timeStep, float restDensitySquInvF,
		float massF, float massB, float gammaPress,
		uint32_t N_F,
		// destinations
		float* RESTRICT diagElemF);

	void SourceTermAndDiagonalElement(
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
		float timeStep, float restDensityF, float restDensitySquInvF,
		float massF, float massB, float gammaPress,
		uint32_t N_F,
		// destinations
		float* RESTRICT sourceTermF,
		float* RESTRICT diagElemF);

	void PressureJacobiInit(
		const float* RESTRICT diagElemF,
		const float* RESTRICT sourceTermF,
		float jacobiOmega,
		uint32_t N_F,
		// destinations
		float* RESTRICT pressureF);

	void PressureAcceleration(
		const float* RESTRICT pressureF,
		const std::vector<std::vector<std::uint32_t>>& neighborsFF,
		const std::vector<std::vector<std::uint32_t>>& neighborsFB,
		const std::vector<std::vector<float>>& xKernelGradFF,
		const std::vector<std::vector<float>>& yKernelGradFF,
		const std::vector<std::vector<float>>& zKernelGradFF,
		const std::vector<std::vector<float>>& xKernelGradFB,
		const std::vector<std::vector<float>>& yKernelGradFB,
		const std::vector<std::vector<float>>& zKernelGradFB,
		float restDensityF, float massF, float massB,
		float gammaPress,
		uint32_t N_F,
		// destinations
		float* RESTRICT xAccPressF,
		float* RESTRICT yAccPressF,
		float* RESTRICT zAccPressF);

	void PressureRelaxedJacobiSolver(
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
		float timeStep, float massF, float massB,
		float jacobiOmega,
		uint32_t N_F,
		// destinations
		float* RESTRICT pressureF);

	void Integration(
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
		float timeStep,
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
		float* RESTRICT zAccF);

} // namespace CPU01
