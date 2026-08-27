// snapshot_kernels.cu

// external
#include <cuda_runtime.h>
#include <cmath>

// internal
#include "source/simulation/GPU/gpu_utils.h"
#include "snapshot_kernels.cuh"



namespace GPU
{
	namespace
	{
		__global__
		void K_PackFluidVisualSamples(
			const float* __restrict__ xPos,
			const float* __restrict__ yPos,
			const float* __restrict__ zPos,
			const float* __restrict__ xVel,
			const float* __restrict__ yVel,
			const float* __restrict__ zVel,
			const float* __restrict__ density,
			const float* __restrict__ pressure,
			const Base::ColorCodingMode scalarMode,
			const uint32_t N,
			Sync::VisualParticleSample* __restrict__ dst)
		{
			const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
			if (i >= N) { return; }

			// compute scalar value used for fluid color coding
			float scalar = 0.0f;
			switch (scalarMode)
			{
			case Base::ColorCodingMode::VelocityMag:
				{
					const float vx = xVel[i];
					const float vy = yVel[i];
					const float vz = zVel[i];
					scalar = sqrtf(vx * vx + vy * vy + vz * vz);
					break;
				}
			case Base::ColorCodingMode::Density:
				scalar = density[i];
				break;
			case Base::ColorCodingMode::Pressure:
				scalar = pressure[i];
				break;
			case Base::ColorCodingMode::None:
			default:
				break;
			}

			// pack fluid particle into visual sample
			dst[i] = Sync::VisualParticleSample{ xPos[i], yPos[i], zPos[i], scalar };
		}

		__global__
		void K_PackRigidVisualSamples(
			const float* __restrict__ xPos,
			const float* __restrict__ yPos,
			const float* __restrict__ zPos,
			const float* __restrict__ volume,
			const uint32_t N,
			Sync::VisualParticleSample* __restrict__ dst)
		{
			const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
			if (i >= N) { return; }

			// pack rigid particle into visual sample
			dst[i] = Sync::VisualParticleSample{ xPos[i], yPos[i], zPos[i], volume[i] };
		}

	} // namespace anonymous

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
		Sync::VisualParticleSample* rigidDst)
	{
		const uint32_t N_BLOCK_SIZE = 256;

		// pack fluid particles into visual samples
		if (N_F > 0)
		{
			const uint32_t N_GRID_SIZE = (N_F + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
			K_PackFluidVisualSamples<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
				xPosF, yPosF, zPosF, xVelF, yVelF, zVelF, densityF, pressureF,
				scalarMode, N_F, fluidDst);
			GPU::CheckLastKernel("K_PackFluidVisualSamples()");
		}

		// pack rigid particles into visual samples
		if (N_R > 0)
		{
			const uint32_t N_GRID_SIZE = (N_R + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
			K_PackRigidVisualSamples<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
				xPosR, yPosR, zPosR, volR, N_R, rigidDst);
			GPU::CheckLastKernel("K_PackRigidVisualSamples()");
		}
	}

} // namespace GPU
