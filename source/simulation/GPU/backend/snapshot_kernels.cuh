// snapshot_kernels.cuh

#pragma once

// external
#include <cstdint>
#include <cuda_runtime.h>

// internal
#include "source/base/enum_hub.h"
#include "source/simulation/sync/mailboxes_cuda.h"



namespace GPU
{
	void Launch_PackVisualSnapshotCUDA(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* densityF, const float* pressureF,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const float* volR,
		const Base::ColorCodingMode scalarMode,
		const uint32_t N_F, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		Sync::VisualParticleSample* fluidDst,
		Sync::VisualParticleSample* rigidDst);

} // namespace GPU
