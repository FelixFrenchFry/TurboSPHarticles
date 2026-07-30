// cuda_interop.cu

#ifdef TURBOSPHARTICLES_WITH_CUDA

// external
#include <cuda_runtime.h>
#include <cmath>

// internal
#include "Source/Sim/GPU/gpu_utils.h"
#include "Source/Sim/Sync/mailboxes_cuda.h"
#include "cuda_interop.cuh"
#include "rendering.h"



namespace Render
{
	__global__
	void K_PackFluidInstancesCUDA(
		const Sync::VisualParticleSample* __restrict__ src,
		const float radius,
		const uint32_t N,
		InstanceData* __restrict__ dst)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N) { return; }

		const auto sample = src[i];

		// pack fluid particle into OpenGL instance data
		dst[i] = InstanceData{
			sample.x, sample.y, sample.z, radius, sample.value,
			{ 0.0f, 0.0f, 0.0f } };
	}

	__global__
	void K_PackRigidInstancesCUDA(
		const Sync::VisualParticleSample* __restrict__ src,
		const float particleSizeB,
		const float radiusScale,
		const uint32_t N,
		const uint32_t dstOffset,
		InstanceData* __restrict__ dst)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N) { return; }

		const auto sample = src[i];

		// compute rigid particle radius
		const float nonNegativeScale = fmaxf(radiusScale, 0.0f);
		float radius = particleSizeB * 0.5f * nonNegativeScale;
		if (isfinite(sample.value) && sample.value > 0.0f)
		{
			const float computed = 0.5f * cbrtf(sample.value) * nonNegativeScale;
			if (isfinite(computed)) { radius = computed; }
		}

		// pack rigid particle into OpenGL instance data
		dst[dstOffset + i] = InstanceData{
			sample.x, sample.y, sample.z, radius, 0.0f,
			{ 0.0f, 0.0f, 0.0f } };
	}

	void Launch_PackVisualInstancesCUDA(
		const Sync::VisualParticleSample* fluid,
		const Sync::VisualParticleSample* rigid,
		const float fluidRadius,
		const float particleSizeB,
		const float rigidRadiusScale,
		const uint32_t N_F, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		InstanceData* dst)
	{
		const uint32_t N_BLOCK_SIZE = 256;

		// pack fluid particles into OpenGL instance data
		if (N_F > 0)
		{
			const uint32_t N_GRID_SIZE = (N_F + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
			K_PackFluidInstancesCUDA<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
				fluid, fluidRadius, N_F, dst);
			GPU::CheckLastKernel("K_PackFluidInstancesCUDA()");
		}

		// append rigid particles to OpenGL instance data
		if (N_R > 0)
		{
			const uint32_t N_GRID_SIZE = (N_R + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
			K_PackRigidInstancesCUDA<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
				rigid, particleSizeB, rigidRadiusScale, N_R, N_F, dst);
			GPU::CheckLastKernel("K_PackRigidInstancesCUDA()");
		}
	}

} // namespace Render

#endif
