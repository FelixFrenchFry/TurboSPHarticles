// snapshot_kernels.cuh

#pragma once

// external
#include <cstdint>
#include <cuda_runtime.h>

// internal
#include "Source/Base/enum_hub.h"
#include "Source/Sim/Sync/mailboxes_cuda.h"



namespace GPU01
{
	void Launch_PackVisualSnapshotCUDA(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* densityF, const float* pressureF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* volB,
		const uint32_t* rigidBoundaryParticleIdxB,
		const Base::ColorCodingMode scalarMode,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_RBP,
		cudaStream_t stream,
		// destinations
		Sync::VisualParticleSample* fluidDst,
		Sync::VisualParticleSample* rigidDst);

} // namespace GPU01
