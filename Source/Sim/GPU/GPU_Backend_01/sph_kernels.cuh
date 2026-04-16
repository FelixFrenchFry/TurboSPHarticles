// sph_kernels.cuh
//
// TODO: comment

#pragma once

// external
#include <cuda_runtime.h>
#include <cstdint>

// internal
#include "common.cuh"
#include "Source/Sim/Core/parameters_sim.h"



namespace GPU01
{
	void ReloadConstants(const Sim::SimParams& params, uint32_t N_F, uint32_t N_B, cudaStream_t stream);

	void Launch_Density(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrChunkBaseIdxFF,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrListDelta8FF,
		const uint8_t* nbrListDelta8FB,
		const uint16_t* nbrListDelta16FF,
		const uint16_t* nbrListDelta16FB,
		const uint32_t* nbrListAbs32FF,
		const uint32_t* nbrListAbs32FB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* densityF);

	void Launch_Density_CellQueryCache_Implicit(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* cellRefFByAny, const uint32_t* cellRefBByAny,
		const uint32_t* cellStartIdxF, const uint32_t* cellStartIdxB,
		const uint32_t* compactCellIdxByParticleF,
		const uint32_t* queryCellRefAnyByCellF,
		const uint32_t queryCellSlotCount,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* densityF);

	void Launch_NonPressureAcceleration(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* densityF,
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrChunkBaseIdxFF,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrListDelta8FF,
		const uint8_t* nbrListDelta8FB,
		const uint16_t* nbrListDelta16FF,
		const uint16_t* nbrListDelta16FB,
		const uint32_t* nbrListAbs32FF,
		const uint32_t* nbrListAbs32FB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* xVelPredF, float* yVelPredF, float* zVelPredF);

	void Launch_NonPressureAcceleration_CellQueryCache_Implicit(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* densityF,
		const uint32_t* cellRefFByAny, const uint32_t* cellRefBByAny,
		const uint32_t* cellStartIdxF, const uint32_t* cellStartIdxB,
		const uint32_t* compactCellIdxByParticleF,
		const uint32_t* queryCellRefAnyByCellF,
		const uint32_t queryCellSlotCount,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* xVelPredF, float* yVelPredF, float* zVelPredF);

	void Launch_SourceTermAndDiagonalElementAndPressureJacobiInit(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xVelPredF, const float* yVelPredF, const float* zVelPredF,
		const float* densityF,
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrChunkBaseIdxFF,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrListDelta8FF,
		const uint8_t* nbrListDelta8FB,
		const uint16_t* nbrListDelta16FF,
		const uint16_t* nbrListDelta16FB,
		const uint32_t* nbrListAbs32FF,
		const uint32_t* nbrListAbs32FB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* sourceTermF,
		float* diagElemF,
		float* pressureF);

	void Launch_SourceTermAndDiagonalElementAndPressureJacobiInit_CellQueryCache_Implicit(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xVelPredF, const float* yVelPredF, const float* zVelPredF,
		const float* densityF,
		const uint32_t* cellRefFByAny, const uint32_t* cellRefBByAny,
		const uint32_t* cellStartIdxF, const uint32_t* cellStartIdxB,
		const uint32_t* compactCellIdxByParticleF,
		const uint32_t* queryCellRefAnyByCellF,
		const uint32_t queryCellSlotCount,
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
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrChunkBaseIdxFF,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrListDelta8FF,
		const uint8_t* nbrListDelta8FB,
		const uint16_t* nbrListDelta16FF,
		const uint16_t* nbrListDelta16FB,
		const uint32_t* nbrListAbs32FF,
		const uint32_t* nbrListAbs32FB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* xAccPressF,
		float* yAccPressF,
		float* zAccPressF);

	void Launch_PressureAcceleration_CellQueryCache_Implicit(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* pressureF,
		const uint32_t* cellRefFByAny, const uint32_t* cellRefBByAny,
		const uint32_t* cellStartIdxF, const uint32_t* cellStartIdxB,
		const uint32_t* compactCellIdxByParticleF,
		const uint32_t* queryCellRefAnyByCellF,
		const uint32_t queryCellSlotCount,
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
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrChunkBaseIdxFF,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrListDelta8FF,
		const uint8_t* nbrListDelta8FB,
		const uint16_t* nbrListDelta16FF,
		const uint16_t* nbrListDelta16FB,
		const uint32_t* nbrListAbs32FF,
		const uint32_t* nbrListAbs32FB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* pressureF);

	void Launch_PressureRelaxedJacobiSolver_CellQueryCache_Implicit(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xAccPressF, const float* yAccPressF, const float* zAccPressF,
		const float* sourceTermF,
		const float* diagElemF,
		const uint32_t* cellRefFByAny, const uint32_t* cellRefBByAny,
		const uint32_t* cellStartIdxF, const uint32_t* cellStartIdxB,
		const uint32_t* compactCellIdxByParticleF,
		const uint32_t* queryCellRefAnyByCellF,
		const uint32_t queryCellSlotCount,
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
