// sph_kernels.cuh
//
// TODO: comment

#pragma once

// external
#include <cuda_runtime.h>
#include <cstdint>

// internal
#include "Source/Sim/Core/parameters_sim.h"



namespace GPU01
{
	void ReloadConstants(const Sim::SimParams& params, uint32_t N_F, uint32_t N_B, cudaStream_t stream);

	void Launch_Density(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* nbrCountFF,
		const uint32_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrListFF,
		const uint32_t* nbrListFB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* densityF);

	void Launch_NonPressureAcceleration(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* densityF,
		const uint32_t* nbrCountFF,
		const uint32_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrListFF,
		const uint32_t* nbrListFB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* xVelPredF, float* yVelPredF, float* zVelPredF);

	void Launch_SourceTermAndDiagonalElementAndPressureJacobiInit(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xVelPredF, const float* yVelPredF, const float* zVelPredF,
		const float* densityF,
		const uint32_t* nbrCountFF,
		const uint32_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrListFF,
		const uint32_t* nbrListFB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* sourceTermF,
		float* diagElemF,
		float* pressureF);

	void Launch_PressureAcceleration(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* pressureF,
		const uint32_t* nbrCountFF,
		const uint32_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrListFF,
		const uint32_t* nbrListFB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* xAccPressF,
		float* yAccPressF,
		float* zAccPressF);

	void Launch_PressureRelaxedJacobiSolver(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xAccPressF, const float* yAccPressF, const float* zAccPressF,
		const float* sourceTermF,
		const float* diagElemF,
		const uint32_t* nbrCountFF,
		const uint32_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrListFF,
		const uint32_t* nbrListFB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* pressureF);

	void Launch_Integration(
		const float* xVelPredF, const float* yVelPredF, const float* zVelPredF,
		const float* xAccPressF, const float* yAccPressF, const float* zAccPressF,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* xPosF, float* yPosF, float* zPosF,
		float* xVelF, float* yVelF, float* zVelF);

} // namespace GPU01
