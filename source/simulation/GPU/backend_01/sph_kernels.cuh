// sph_kernels.cuh

#pragma once

// external
#include <cuda_runtime.h>
#include <cstdint>

// internal
#include "common.cuh"
#include "source/simulation/core/parameters_sim.h"



namespace GPU01
{
	void ReloadConstants(const Sim::SimParams& params, uint32_t N_F, uint32_t N_B, cudaStream_t stream);

	void Launch_Density(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const float* psiB, const float* psiR,
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint16_t* nbrCountFR,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint64_t* nbrChunkOffsetsFR,
		const uint32_t* nbrChunkBaseIdxFF,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint32_t* nbrChunkBaseIdxFR,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrChunkFormatFR,
		const uint8_t* nbrListDelta8FF,
		const uint8_t* nbrListDelta8FB,
		const uint8_t* nbrListDelta8FR,
		const uint16_t* nbrListDelta16FF,
		const uint16_t* nbrListDelta16FB,
		const uint16_t* nbrListDelta16FR,
		const uint32_t* nbrListAbs32FF,
		const uint32_t* nbrListAbs32FB,
		const uint32_t* nbrListAbs32FR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		float* densityF);

	void Launch_NonPressureAcceleration(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* xVelR, const float* yVelR, const float* zVelR,
		const float* densityF,
		const float* volB, const float* volR,
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint16_t* nbrCountFR,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint64_t* nbrChunkOffsetsFR,
		const uint32_t* nbrChunkBaseIdxFF,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint32_t* nbrChunkBaseIdxFR,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrChunkFormatFR,
		const uint8_t* nbrListDelta8FF,
		const uint8_t* nbrListDelta8FB,
		const uint8_t* nbrListDelta8FR,
		const uint16_t* nbrListDelta16FF,
		const uint16_t* nbrListDelta16FB,
		const uint16_t* nbrListDelta16FR,
		const uint32_t* nbrListAbs32FF,
		const uint32_t* nbrListAbs32FB,
		const uint32_t* nbrListAbs32FR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		float* xVelPredF, float* yVelPredF, float* zVelPredF);

	void Launch_SourceTermAndDiagonalElementAndPressureJacobiInit(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const float* xVelPredF, const float* yVelPredF, const float* zVelPredF,
		const float* xVelR, const float* yVelR, const float* zVelR,
		const float* densityF,
		const float* psiB, const float* psiR,
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint16_t* nbrCountFR,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint64_t* nbrChunkOffsetsFR,
		const uint32_t* nbrChunkBaseIdxFF,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint32_t* nbrChunkBaseIdxFR,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrChunkFormatFR,
		const uint8_t* nbrListDelta8FF,
		const uint8_t* nbrListDelta8FB,
		const uint8_t* nbrListDelta8FR,
		const uint16_t* nbrListDelta16FF,
		const uint16_t* nbrListDelta16FB,
		const uint16_t* nbrListDelta16FR,
		const uint32_t* nbrListAbs32FF,
		const uint32_t* nbrListAbs32FB,
		const uint32_t* nbrListAbs32FR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		float* sourceTermF,
		float* diagElemF,
		float* pressureF);

	void Launch_PressureAcceleration(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const float* pressureF,
		const float* psiB, const float* psiR,
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint16_t* nbrCountFR,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint64_t* nbrChunkOffsetsFR,
		const uint32_t* nbrChunkBaseIdxFF,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint32_t* nbrChunkBaseIdxFR,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrChunkFormatFR,
		const uint8_t* nbrListDelta8FF,
		const uint8_t* nbrListDelta8FB,
		const uint8_t* nbrListDelta8FR,
		const uint16_t* nbrListDelta16FF,
		const uint16_t* nbrListDelta16FB,
		const uint16_t* nbrListDelta16FR,
		const uint32_t* nbrListAbs32FF,
		const uint32_t* nbrListAbs32FB,
		const uint32_t* nbrListAbs32FR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		float* xAccPressF,
		float* yAccPressF,
		float* zAccPressF);

	void Launch_PressureRelaxedJacobiSolver(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const float* xAccPressF, const float* yAccPressF, const float* zAccPressF,
		const float* sourceTermF,
		const float* diagElemF,
		const float* psiB, const float* psiR,
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint16_t* nbrCountFR,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint64_t* nbrChunkOffsetsFR,
		const uint32_t* nbrChunkBaseIdxFF,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint32_t* nbrChunkBaseIdxFR,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrChunkFormatFR,
		const uint8_t* nbrListDelta8FF,
		const uint8_t* nbrListDelta8FB,
		const uint8_t* nbrListDelta8FR,
		const uint16_t* nbrListDelta16FF,
		const uint16_t* nbrListDelta16FB,
		const uint16_t* nbrListDelta16FR,
		const uint32_t* nbrListAbs32FF,
		const uint32_t* nbrListAbs32FB,
		const uint32_t* nbrListAbs32FR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
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
