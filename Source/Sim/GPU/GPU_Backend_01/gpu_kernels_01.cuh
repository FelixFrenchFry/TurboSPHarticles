// gpu_kernels_01.cuh
//
// TODO: comment

#pragma once

// external
#include <cuda_runtime.h>
#include <cstdint>

// internal
#include "Source/Sim/Core/parameters_sim.h"



namespace GPU_01
{
	void ReloadConstants(const Sim::SimParams& params, uint32_t N_F, uint32_t N_B, cudaStream_t stream);

	// TODO: split into construction and query
	// TODO: make adhere to neighbor search interface
	void Launch_NeighborSearch_BruteForce(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t N_F, const uint32_t N_B,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		cudaStream_t stream,
		// destinations
		uint32_t* neighCountFF,
		uint32_t* neighCountFB,
		uint32_t* neighListFF,
		uint32_t* neighListFB,
		uint32_t* overflowFlag);

	void Launch_DensityPressure(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* neighCountFF,
		const uint32_t* neighCountFB,
		const uint32_t* neighListFF,
		const uint32_t* neighListFB,
		const uint32_t N_F, const uint32_t N_B,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		cudaStream_t stream,
		// destinations
		float* densityF,
		float* pressureF);

	void Launch_Accelerations(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* xVelB, const float* yVelB, const float* zVelB,
		const float* densityF,
		const float* pressureF,
		const float* pressureB,
		const uint32_t* neighCountFF,
		const uint32_t* neighCountFB,
		const uint32_t* neighListFF,
		const uint32_t* neighListFB,
		const uint32_t N_F, const uint32_t N_B,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		cudaStream_t stream,
		// destinations
		float* xAccF, float* yAccF, float* zAccF,
		float* xAccPressF, float* yAccPressF, float* zAccPressF,
		float* xAccViscF, float* yAccViscF, float* zAccViscF,
		float* xAccExtF, float* yAccExtF, float* zAccExtF);

	void Launch_Integration(
		const float* xAccF, const float* yAccF, const float* zAccF,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* xPosF, float* yPosF, float* zPosF,
		float* xVelF, float* yVelF, float* zVelF);

} // namespace GPU_01
