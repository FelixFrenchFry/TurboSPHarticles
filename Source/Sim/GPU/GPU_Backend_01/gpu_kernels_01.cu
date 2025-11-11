// gpu_kernels_01.cu

// external
#include <cuda_runtime.h>
#include <iostream>
#include <numbers>

// internal
#include "Source/Sim/GPU/gpu_utils.h"
#include "gpu_kernels_01.cuh"

constexpr float  PI = std::numbers::pi_v<float>;



namespace GPU01
{
	__constant__ uint32_t c_N_F;
	__constant__ uint32_t c_N_B;
	__constant__ float c_timeStep;
	__constant__ float c_sizeF;
	__constant__ float c_sizeB;
	__constant__ float c_restDensityF;
	__constant__ float c_restDensityB;
	__constant__ float c_restDensitySquInvF;
	__constant__ float c_restDensitySquInvB;
	__constant__ float c_massF;
	__constant__ float c_massB;
	__constant__ float c_gammaPress;
	__constant__ float c_jacobiOmega;
	__constant__ float c_viscosityF;
	__constant__ float c_viscosityB;
	__constant__ float c_xGravity;
	__constant__ float c_yGravity;
	__constant__ float c_zGravity;
	__constant__ float c_kernelRadiusSqu;
	__constant__ float c_cubicSplineAlpha;

	void ReloadConstants(const Sim::SimParams& params, uint32_t N_F, uint32_t N_B, cudaStream_t stream)
	{
		const float sizeInv = 1.0 / params.particleSizeF;
		const float cubicSplineAlpha = static_cast<float>(
			(1.0 / (4.0 * static_cast<double>(PI))) * sizeInv * sizeInv * sizeInv);
		const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;
		const float restDensitySquInvF = 1.0f / (params.restDensityF * params.restDensityF);
		const float restDensitySquInvB = 1.0f / (params.restDensityB * params.restDensityB);

		// store values in constant memory on the GPU
		// (efficient read-only broadcast access for all threads)
		GPU::Check(cudaMemcpyToSymbol(c_N_F, &N_F, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_N_B, &N_B, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_timeStep, &params.timeStep, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_sizeF, &params.particleSizeF, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_sizeB, &params.particleSizeB, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_restDensityF, &params.restDensityF, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_restDensityB, &params.restDensityB, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_restDensitySquInvF, &restDensitySquInvF, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_restDensitySquInvB, &restDensitySquInvB, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_massF, &params.massF, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_massB, &params.massB, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_gammaPress, &params.gammaPress, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_jacobiOmega, &params.jacobiOmega, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_viscosityF, &params.viscosityF, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_viscosityB, &params.viscosityB, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_xGravity, &params.gravity.x, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_yGravity, &params.gravity.y, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_zGravity, &params.gravity.z, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_kernelRadiusSqu, &kernelRadiusSqu, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_cubicSplineAlpha, &cubicSplineAlpha, sizeof(float)));

		GPU::StreamSync(stream);
	}

	__device__ __forceinline__
	void KernelGradient_CubicSpline(
		float dx, float dy, float dz,
		float& gx, float& gy, float& gz)
	{
		float distSqu = dx * dx + dy * dy + dz * dz;
		float dist = sqrtf(distSqu);
		float distNorm = c_sizeF > 0.0f ? dist / c_sizeF : 0.0f;

		float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
		float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

		// kernel gradient
		if (dist > 1e-12f && distNorm > 0.0f)
		{
			float commonFac = c_cubicSplineAlpha
							* (-3.0f * t2 * t2 + 12.0f * t1 * t1)
							/ (distNorm * c_sizeF * c_sizeF);

			gx = commonFac * dx;
			gy = commonFac * dy;
			gz = commonFac * dz;
		}
		else
		{
			gx = 0.0f;
			gy = 0.0f;
			gz = 0.0f;
		}
	}

	__device__ __forceinline__
	void KernelWeightAndGradient_CubicSpline(
		float dx, float dy, float dz,
		float& w, float& gx, float& gy, float& gz)
	{
		float distSqu = dx * dx + dy * dy + dz * dz;
		float dist = sqrtf(distSqu);
		float distNorm = c_sizeF > 0.0f ? dist / c_sizeF : 0.0f;

		float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
		float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

		// kernel weight
		w = c_cubicSplineAlpha * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);

		// kernel gradient
		if (dist > 1e-12f && distNorm > 0.0f)
		{
			float commonFac = c_cubicSplineAlpha
						 * (-3.0f * t2 * t2 + 12.0f * t1 * t1)
						 / (distNorm * c_sizeF * c_sizeF);

			gx = commonFac * dx;
			gy = commonFac * dy;
			gz = commonFac * dz;
		}
		else
		{
			gx = 0.0f;
			gy = 0.0f;
			gz = 0.0f;
		}
	}

	__global__
	void K_NeighborSearch_BruteForce(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		// destinations
		uint32_t* __restrict__ neighCountFF,
		uint32_t* __restrict__ neighCountFB,
		uint32_t* __restrict__ neighListFF,
		uint32_t* __restrict__ neighListFB,
		uint32_t* __restrict__ overflowFlag)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_N_F) { return; }

		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];

		const uint32_t baseFF = i * N_MAX_FF;
		const uint32_t baseFB = i * N_MAX_FB;

		uint32_t cntFF = 0;
		uint32_t cntFB = 0;

		// FF
		for (uint32_t j = 0; j < c_N_F; j++)
		{
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// squared distance between particles i and j
			const float distSqu = dx * dx + dy * dy + dz * dz;

			if (distSqu <= c_kernelRadiusSqu) // neighbor found
			{
				if (cntFF < N_MAX_FF)
				{
					// add fluid neighbor to the list
					// (global memory write)
					neighListFF[baseFF + cntFF] = j;
					cntFF++;
				}
				else // overflow
				{
					atomicExch(overflowFlag, 1u);
				}
			}
		}

		// FB
		for (uint32_t j = 0; j < c_N_B; j++)
		{
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// squared distance between particles i and j
			const float distSqu = dx * dx + dy * dy + dz * dz;

			if (distSqu <= c_kernelRadiusSqu) // neighbor found
			{
				if (cntFB < N_MAX_FB)
				{
					// add boundary neighbor to the list
					// (global memory write)
					neighListFB[baseFB + cntFB] = j;
					cntFB++;
				}
				else // overflow
				{
					atomicExch(overflowFlag, 1u);
				}
			}
		}

		// update neighbor counts
		// (global memory write)
		neighCountFF[i] = cntFF;
		neighCountFB[i] = cntFB;
	}

	__global__
	void K_Density(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const uint32_t* __restrict__ neighCountFF,
		const uint32_t* __restrict__ neighCountFB,
		const uint32_t* __restrict__ neighListFF,
		const uint32_t* __restrict__ neighListFB,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		// destinations
		float* __restrict__ densityF)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_N_F) { return; }

		// particle i data
		// (global memory reads)
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];

		// temp accumulation variable
		float tmp_density = 0.0f;

		const uint32_t baseFF = i * N_MAX_FF;
		const uint32_t cntFF = neighCountFF[i];

		// FF
		for (uint32_t k = 0; k < cntFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFF[baseFF + k];

			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// kernel weight and gradient
			float w = 0.0f;
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;
			KernelWeightAndGradient_CubicSpline(
				dx, dy, dz, w, gx, gy, gz);

			// density contribution from fluid neighbor j
			tmp_density += c_massF * w;
		}

		const uint32_t baseFB = i * N_MAX_FB;
		const uint32_t cntFB = neighCountFB[i];

		// FB
		for (uint32_t k = 0; k < cntFB; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFB[baseFB + k];

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel weight and gradient
			float w = 0.0f;
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;
			KernelWeightAndGradient_CubicSpline(
				dx, dy, dz, w, gx, gy, gz);

			// density contribution from boundary neighbor j
			tmp_density += c_massB * w;
		}

		densityF[i] = tmp_density;
	}

	__global__
	void K_NonPressureAcc(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const float* __restrict__ xVelF,
		const float* __restrict__ yVelF,
		const float* __restrict__ zVelF,
		const float* __restrict__ densityF,
		const uint32_t* __restrict__ neighCountFF,
		const uint32_t* __restrict__ neighCountFB,
		const uint32_t* __restrict__ neighListFF,
		const uint32_t* __restrict__ neighListFB,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		// destinations
		float* __restrict__ xVelPredF,
		float* __restrict__ yVelPredF,
		float* __restrict__ zVelPredF,
		float* __restrict__ xAccViscF,
		float* __restrict__ yAccViscF,
		float* __restrict__ zAccViscF,
		float* __restrict__ xAccExtF,
		float* __restrict__ yAccExtF,
		float* __restrict__ zAccExtF)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_N_F) { return; }

		// particle i data
		// (global memory reads)
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];
		const float vxi = xVelF[i];
		const float vyi = yVelF[i];
		const float vzi = zVelF[i];

		// temp accumulation variables
		float tmp_xAccViscF = 0.0f;
		float tmp_yAccViscF = 0.0f;
		float tmp_zAccViscF = 0.0f;

		const uint32_t baseFF = i * N_MAX_FF;
		const uint32_t cntFF = neighCountFF[i];

		// FF
		for (uint32_t k = 0; k < cntFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFF[baseFF + k];

			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// velocity difference
			const float dvx = vxi - xVelF[j];
			const float dvy = vyi - yVelF[j];
			const float dvz = vzi - zVelF[j];

			// kernel weight and gradient
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, gx, gy, gz);

			// viscosity acc contribution from fluid neighbor j
			const float accViscCoeff = (c_massF / densityF[j])
									 * (dvx * dx + dvy * dy + dvz * dz)
									 / (dx * dx + dy * dy + dz * dz
									 + 0.01f * c_sizeF * c_sizeF);

			tmp_xAccViscF += accViscCoeff * gx;
			tmp_yAccViscF += accViscCoeff * gy;
			tmp_zAccViscF += accViscCoeff * gz;
		}

		// temp accumulation variables
		float tmp_xAccViscB = 0.0f;
		float tmp_yAccViscB = 0.0f;
		float tmp_zAccViscB = 0.0f;

		const uint32_t baseFB = i * N_MAX_FB;
		const uint32_t cntFB = neighCountFB[i];

		// FB
		for (uint32_t k = 0; k < cntFB; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFB[baseFB + k];

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel weight and gradient
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;

			KernelGradient_CubicSpline(dx, dy, dz, gx, gy, gz);

			// viscosity acc contribution from boundary neighbor j
			const float accViscCoeff = (c_massB / c_restDensityB)
									 * (vxi * dx + vyi * dy + vzi * dz)
									 / (dx * dx + dy * dy + dz * dz
									 + 0.01f * c_sizeF * c_sizeF);

			tmp_xAccViscB += accViscCoeff * gx;
			tmp_yAccViscB += accViscCoeff * gy;
			tmp_zAccViscB += accViscCoeff * gz;
		}

		// finalize accelerations
		// (3D -> factor 2 * (dim + 2))
		tmp_xAccViscF *= 2.0f * (3 + 2) * c_viscosityF;
		tmp_yAccViscF *= 2.0f * (3 + 2) * c_viscosityF;
		tmp_zAccViscF *= 2.0f * (3 + 2) * c_viscosityF;
		tmp_xAccViscB *= 2.0f * (3 + 2) * c_viscosityB;
		tmp_yAccViscB *= 2.0f * (3 + 2) * c_viscosityB;
		tmp_zAccViscB *= 2.0f * (3 + 2) * c_viscosityB;

		xAccViscF[i] = tmp_xAccViscF + tmp_xAccViscB;
		yAccViscF[i] = tmp_yAccViscF + tmp_yAccViscB;
		zAccViscF[i] = tmp_zAccViscF + tmp_zAccViscB;

		xAccExtF[i] = c_xGravity;
		yAccExtF[i] = c_yGravity;
		zAccExtF[i] = c_zGravity;

		// compute predicted velocity from non-pressure accelerations
		xVelPredF[i] = xVelF[i] + c_timeStep * (xAccViscF[i] + xAccExtF[i]);
		yVelPredF[i] = yVelF[i] + c_timeStep * (yAccViscF[i] + yAccExtF[i]);
		zVelPredF[i] = zVelF[i] + c_timeStep * (zAccViscF[i] + zAccExtF[i]);
	}

	__global__
	void K_SourceTerm(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const float* __restrict__ xVelPredF,
		const float* __restrict__ yVelPredF,
		const float* __restrict__ zVelPredF,
		const float* __restrict__ densityF,
		const uint32_t* __restrict__ neighCountFF,
		const uint32_t* __restrict__ neighCountFB,
		const uint32_t* __restrict__ neighListFF,
		const uint32_t* __restrict__ neighListFB,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		// destinations
		float* __restrict__ sourceTermF)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_N_F) { return; }

		// particle i data
		// (global memory reads)
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];
		const float vpxi = xVelPredF[i];
		const float vpyi = yVelPredF[i];
		const float vpzi = zVelPredF[i];

		// temp accumulation variable
		float tmp_sourceTermF = 0.0f;

		const uint32_t baseFF = i * N_MAX_FF;
		const uint32_t cntFF = neighCountFF[i];

		// FF
		for (uint32_t k = 0; k < cntFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFF[baseFF + k];

			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// kernel weight and gradient
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, gx, gy, gz);

			// source term contribution from fluid neighbor j
			const float dvx = vpxi - xVelPredF[j];
			const float dvy = vpyi - yVelPredF[j];
			const float dvz = vpzi - zVelPredF[j];

			tmp_sourceTermF += c_massF * (dvx * gx + dvy * gy + dvz * gz);
		}

		const uint32_t baseFB = i * N_MAX_FB;
		const uint32_t cntFB = neighCountFB[i];

		// FB
		for (uint32_t k = 0; k < cntFB; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFB[baseFB + k];

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel weight and gradient
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, gx, gy, gz);

			// source term contribution from boundary neighbor j
			tmp_sourceTermF += c_massB * (vpxi * gx + vpyi * gy + vpzi * gz);
		}

		// finalize source term
		sourceTermF[i] = c_restDensityF - densityF[i]
					   - c_timeStep * tmp_sourceTermF;
	}

	__global__
	void K_DiagonalElement(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const uint32_t* __restrict__ neighCountFF,
		const uint32_t* __restrict__ neighCountFB,
		const uint32_t* __restrict__ neighListFF,
		const uint32_t* __restrict__ neighListFB,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		// destinations
		float* __restrict__ diagElemF)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_N_F) { return; }

		// particle i data
		// (global memory reads)
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];

		// temp accumulation variables
		float tmp_DiagElemF = 0.0f;
		float temp_cxF = 0.0f;
		float temp_cyF = 0.0f;
		float temp_czF = 0.0f;
		float temp_cxB = 0.0f;
		float temp_cyB = 0.0f;
		float temp_czB = 0.0f;

		const uint32_t baseFF = i * N_MAX_FF;
		const uint32_t cntFF = neighCountFF[i];

		// FF
		for (uint32_t k = 0; k < cntFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFF[baseFF + k];

			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// kernel weight and gradient
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, gx, gy, gz);

			// add c-term contributions from fluid neighbors
			temp_cxF += c_massF * gx;
			temp_cyF += c_massF * gy;
			temp_czF += c_massF * gz;
		}

		const uint32_t baseFB = i * N_MAX_FB;
		const uint32_t cntFB = neighCountFB[i];

		// FB
		for (uint32_t k = 0; k < cntFB; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFB[baseFB + k];

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel weight and gradient
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, gx, gy, gz);

			// add c-term contributions from boundary neighbors
			temp_cxB += c_massB * gx;
			temp_cyB += c_massB * gy;
			temp_czB += c_massB * gz;
		}

		// finalize c-term
		const float cx = -1.0f * (temp_cxF * c_restDensitySquInvF)
					   - 2.0f * c_gammaPress * (temp_cxB * c_restDensitySquInvF);
		const float cy = -1.0f * (temp_cyF * c_restDensitySquInvF)
					   - 2.0f * c_gammaPress * (temp_cyB * c_restDensitySquInvF);
		const float cz = -1.0f * (temp_czF * c_restDensitySquInvF)
					   - 2.0f * c_gammaPress * (temp_czB * c_restDensitySquInvF);

		// FF
		for (uint32_t k = 0; k < cntFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFF[baseFF + k];

			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// kernel weight and gradient
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, gx, gy, gz);

			// part 1 (top) of the summation
			tmp_DiagElemF += c_massF * (cx * gx + cy * gy + cz * gz);

			// part 2 (middle) of the summation
			const float tmp_x = (c_massF * c_restDensitySquInvF) * (-1.0f * gx);
			const float tmp_y = (c_massF * c_restDensitySquInvF) * (-1.0f * gy);
			const float tmp_z = (c_massF * c_restDensitySquInvF) * (-1.0f * gz);
			tmp_DiagElemF += c_massF * (tmp_x * gx + tmp_y * gy + tmp_z * gz);
		}

		// FB
		for (uint32_t k = 0; k < cntFB; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFB[baseFB + k];

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel weight and gradient
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, gx, gy, gz);

			// part 3 (bottom) of the summation
			tmp_DiagElemF += c_massB * (cx * gx + cy * gy + cz * gz);
		}

		// finalize diagonal element
		diagElemF[i] = -1.0f * c_timeStep * c_timeStep * tmp_DiagElemF;
	}

	void K_PressureJacobiSolver(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const float* __restrict__ densityF,
		const float* __restrict__ sourceTermF,
		const float* __restrict__ diagElemF,
		const uint32_t* __restrict__ neighCountFF,
		const uint32_t* __restrict__ neighCountFB,
		const uint32_t* __restrict__ neighListFF,
		const uint32_t* __restrict__ neighListFB,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		// destinations
		float* __restrict__ xAccPressF,
		float* __restrict__ yAccPressF,
		float* __restrict__ zAccPressF,
		float* __restrict__ sysVecElemF,
		float* __restrict__ pressureF)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_N_F) { return; }

		// particle i data
		// (global memory reads)
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];
		const float pressi = pressureF[i];
		const float densi = densityF[i];

		// temp accumulation variables
		float tmp_xAccPressF = 0.0f;
		float tmp_yAccPressF = 0.0f;
		float tmp_zAccPressF = 0.0f;

		const uint32_t baseFF = i * N_MAX_FF;
		const uint32_t cntFF = neighCountFF[i];

		// FF
		for (uint32_t k = 0; k < cntFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFF[baseFF + k];

			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// kernel weight and gradient
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, gx, gy, gz);

			// pressure acc contribution from fluid neighbor j
			const float accPressCoeff = (c_massF * (pressi / (densi * densi)
									  + pressureF[j] / (densi * densi)));

			tmp_xAccPressF += accPressCoeff * gx;
			tmp_yAccPressF += accPressCoeff * gy;
			tmp_zAccPressF += accPressCoeff * gz;
		}

		const uint32_t baseFB = i * N_MAX_FB;
		const uint32_t cntFB = neighCountFB[i];

		// FB
		for (uint32_t k = 0; k < cntFB; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFB[baseFB + k];

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel weight and gradient
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, gx, gy, gz);

			// pressure acc contribution from boundary neighbor j
			const float accPressCoeff = c_gammaPress * (c_massB * 2.0f 
									  * (pressi / (densi * densi)));

			tmp_xAccPressF += accPressCoeff * gx;
			tmp_yAccPressF += accPressCoeff * gy;
			tmp_zAccPressF += accPressCoeff * gz;
		}

		// finalize pressure acceleration
		tmp_xAccPressF *= -1.0f;
		tmp_yAccPressF *= -1.0f;
		tmp_zAccPressF *= -1.0f;

		xAccPressF[i] = tmp_xAccPressF;
		yAccPressF[i] = tmp_yAccPressF;
		zAccPressF[i] = tmp_zAccPressF;
	}

	__global__
	void K_Accelerations(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const float* __restrict__ xVelF,
		const float* __restrict__ yVelF,
		const float* __restrict__ zVelF,
		const float* __restrict__ xVelB,
		const float* __restrict__ yVelB,
		const float* __restrict__ zVelB,
		const float* __restrict__ densityF,
		const float* __restrict__ pressureF,
		const uint32_t* __restrict__ neighCountFF,
		const uint32_t* __restrict__ neighCountFB,
		const uint32_t* __restrict__ neighListFF,
		const uint32_t* __restrict__ neighListFB,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		// destinations
		float* __restrict__ xAccF,
		float* __restrict__ yAccF,
		float* __restrict__ zAccF,
		float* __restrict__ xAccPressF,
		float* __restrict__ yAccPressF,
		float* __restrict__ zAccPressF,
		float* __restrict__ xAccViscF,
		float* __restrict__ yAccViscF,
		float* __restrict__ zAccViscF,
		float* __restrict__ xAccExtF,
		float* __restrict__ yAccExtF,
		float* __restrict__ zAccExtF)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_N_F) { return; }

		// particle i data
		// (global memory reads)
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];
		const float vxi = xVelF[i];
		const float vyi = yVelF[i];
		const float vzi = zVelF[i];

		const uint32_t baseFF = i * N_MAX_FF;
		const uint32_t cntFF = neighCountFF[i];

		// accumulators
		float sum_xAccPress = 0.0f;
		float sum_yAccPress = 0.0f;
		float sum_zAccPress = 0.0f;
		float sum_xAccViscF = 0.0f;
		float sum_yAccViscF = 0.0f;
		float sum_zAccViscF = 0.0f;

		// FF
		for (uint32_t k = 0; k < cntFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFF[baseFF + k];

			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// kernel weight and gradient
			float w = 0.0f;
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;

			KernelWeightAndGradient_CubicSpline(
				dx, dy, dz, w, gx, gy, gz);

			// velocity difference
			const float dvx = vxi - xVelF[j];
			const float dvy = vyi - yVelF[j];
			const float dvz = vzi - zVelF[j];

			const float velDiff_dot_posDiff = dvx * dx
										    + dvy * dy
										    + dvz * dz;

			const float posDiff_dot_posDiff = dx * dx
										    + dy * dy
										    + dz * dz;

			// viscosity acc contribution from fluid neighbor j
			const float viscCoeff = (c_massF / densityF[j])
								  * velDiff_dot_posDiff / (posDiff_dot_posDiff
								  + 0.01f * c_sizeF * c_sizeF);

			sum_xAccViscF += viscCoeff * gx;
			sum_yAccViscF += viscCoeff * gy;
			sum_zAccViscF += viscCoeff * gz;

			// pressure acc contribution from fluid neighbor j
			const float pressCoeff = (c_massF * (pressureF[i] / (densityF[i] * densityF[i])
									+ pressureF[j] / (densityF[j] * densityF[j])));

			sum_xAccPress += pressCoeff * gx;
			sum_yAccPress += pressCoeff * gy;
			sum_zAccPress += pressCoeff * gz;
		}

		const uint32_t baseFB = i * N_MAX_FB;
		const uint32_t cntFB = neighCountFB[i];

		// accumulators
		float sum_xAccViscB = 0.0f;
		float sum_yAccViscB = 0.0f;
		float sum_zAccViscB = 0.0f;

		// FB
		for (uint32_t k = 0; k < cntFB; k++)
		{
			// global index of neighbor particle
			const uint32_t j = neighListFB[baseFB + k];

			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel weight and gradient
			float w = 0.0f;
			float gx = 0.0f;
			float gy = 0.0f;
			float gz = 0.0f;

			KernelWeightAndGradient_CubicSpline(
				dx, dy, dz, w, gx, gy, gz);

			// velocity difference
			const float dvx = vxi - xVelB[j];
			const float dvy = vyi - yVelB[j];
			const float dvz = vzi - zVelB[j];

			const float velDiff_dot_posDiff = dvx * dx
											+ dvy * dy
											+ dvz * dz;

			const float posDiff_dot_posDiff = dx * dx
											+ dy * dy
											+ dz * dz;

			// viscosity acc contribution from boundary neighbor j
			const float viscCoeff = (c_massB / c_restDensityB)
								  * velDiff_dot_posDiff / (posDiff_dot_posDiff
								  + 0.01f * c_sizeF * c_sizeF);

			sum_xAccViscB += viscCoeff * gx;
			sum_yAccViscB += viscCoeff * gy;
			sum_zAccViscB += viscCoeff * gz;

			// pressure acc contribution from boundary neighbor j
			// TODO: pressure mirroring applied correctly?
			const float pressCoeff = (c_massB * (pressureF[i] / (densityF[i] * densityF[i])
								   + pressureF[i] / (c_restDensityB * c_restDensityB)));

			sum_xAccPress += pressCoeff * gx;
			sum_yAccPress += pressCoeff * gy;
			sum_zAccPress += pressCoeff * gz;
		}

		// finalize accelerations
		// (3D -> factor 2 * (dim + 2))
		const float viscScaleF = 2.0f * (3 + 2) * c_viscosityF;
		const float viscScaleB = 2.0f * (3 + 2) * c_viscosityB;
		sum_xAccPress *= -1.0f;
		sum_yAccPress *= -1.0f;
		sum_zAccPress *= -1.0f;

		xAccViscF[i] = sum_xAccViscF * viscScaleF + sum_xAccViscB * viscScaleB;
		yAccViscF[i] = sum_yAccViscF * viscScaleF + sum_yAccViscB * viscScaleB;
		zAccViscF[i] = sum_zAccViscF * viscScaleF + sum_zAccViscB * viscScaleB;

		xAccPressF[i] = sum_xAccPress;
		yAccPressF[i] = sum_yAccPress;
		zAccPressF[i] = sum_zAccPress;

		xAccExtF[i] = c_xGravity;
		yAccExtF[i] = c_yGravity;
		zAccExtF[i] = c_zGravity;

		// sum final acceleration components
		xAccF[i] = xAccPressF[i] + xAccViscF[i] + xAccExtF[i];
		yAccF[i] = yAccPressF[i] + yAccViscF[i] + yAccExtF[i];
		zAccF[i] = zAccPressF[i] + zAccViscF[i] + zAccExtF[i];
	}

	__global__
	void K_Integration(
		const float* __restrict__ axF,
		const float* __restrict__ ayF,
		const float* __restrict__ azF,
		// destinations
		float* __restrict__ xF,
		float* __restrict__ yF,
		float* __restrict__ zF,
		float* __restrict__ vxF,
		float* __restrict__ vyF,
		float* __restrict__ vzF)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_N_F) { return; }

		// explicit Euler integration for velocity
		vxF[i] = vxF[i] + c_timeStep * axF[i];
		vyF[i] = vyF[i] + c_timeStep * ayF[i];
		vzF[i] = vzF[i] + c_timeStep * azF[i];

		// semi-implicit Euler integration for position
		xF[i] = xF[i] + c_timeStep * vxF[i];
		yF[i] = yF[i] + c_timeStep * vyF[i];
		zF[i] = zF[i] + c_timeStep * vzF[i];
	}

} // namespace GPU01

// -----------------------------------------------------------------------------
// --- [ kernel launchers ] ----------------------------------------------------
// -----------------------------------------------------------------------------

namespace GPU01
{
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
		uint32_t* overflowFlag)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_NeighborSearch_BruteForce<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, N_MAX_FF, N_MAX_FB, neighCountFF, neighCountFB,
			neighListFF, neighListFB, overflowFlag);

		GPU::CheckLastKernel("K_NeighborSearch_BruteForce()");
	}

	void Launch_Density(
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
		float* densityF)
	{
		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_Density<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, neighCountFF, neighCountFB,
			neighListFF, neighListFB, N_MAX_FF, N_MAX_FB, densityF);

		GPU::CheckLastKernel("K_Density()");
	}

	void Launch_NonPressureAcc(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* densityF,
		const uint32_t* neighCountFF,
		const uint32_t* neighCountFB,
		const uint32_t* neighListFF,
		const uint32_t* neighListFB,
		const uint32_t N_F, const uint32_t N_B,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		cudaStream_t stream,
		// destinations
		float* xVelPredF, float* yVelPredF, float* zVelPredF,
		float* xAccViscF, float* yAccViscF, float* zAccViscF,
		float* xAccExtF, float* yAccExtF, float* zAccExtF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_NonPressureAcc<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xVelF, yVelF, zVelF, densityF,
			neighCountFF, neighCountFB, neighListFF, neighListFB, N_MAX_FF, N_MAX_FB,
			xVelPredF, yVelPredF, zVelPredF, xAccViscF, yAccViscF, zAccViscF, xAccExtF, yAccExtF, zAccExtF);

		GPU::CheckLastKernel("K_NonPressureAcc()");
	}

	void Launch_SourceTerm(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xVelPredF, const float* yVelPredF, const float* zVelPredF,
		const float* densityF,
		const uint32_t* neighCountFF,
		const uint32_t* neighCountFB,
		const uint32_t* neighListFF,
		const uint32_t* neighListFB,
		const uint32_t N_F, const uint32_t N_B,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		cudaStream_t stream,
		// destinations
		float* sourceTermF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_SourceTerm<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xVelPredF, yVelPredF, zVelPredF, densityF,
			neighCountFF, neighCountFB, neighListFF, neighListFB, N_MAX_FF, N_MAX_FB, sourceTermF);

		GPU::CheckLastKernel("K_SourceTerm()");
	}

	void Launch_DiagonalElement(
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
		float* diagElemF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_DiagonalElement<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, neighCountFF, neighCountFB,
			neighListFF, neighListFB, N_MAX_FF, N_MAX_FB, diagElemF);

		GPU::CheckLastKernel("K_DiagonalElement()");
	}

	void Launch_PressureJacobiSolver(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* densityF,
		const float* sourceTermF,
		const float* diagElemF,
		const uint32_t* neighCountFF,
		const uint32_t* neighCountFB,
		const uint32_t* neighListFF,
		const uint32_t* neighListFB,
		const uint32_t N_F, const uint32_t N_B,
		const uint32_t N_MAX_FF, const uint32_t N_MAX_FB,
		cudaStream_t stream,
		// destinations
		float* xAccPressF,
		float* yAccPressF,
		float* zAccPressF,
		float* sysVecElemF,
		float* pressureF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_PressureJacobiSolver<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, densityF, sourceTermF, diagElemF,
			neighCountFF, neighCountFB, neighListFF, neighListFB, N_MAX_FF, N_MAX_FB,
			xAccPressF, yAccPressF, zAccPressF, sysVecElemF, pressureF);

		GPU::CheckLastKernel("K_PressureJacobiSolver()");
	}

	void Launch_Accelerations(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* xVelB, const float* yVelB, const float* zVelB,
		const float* densityF,
		const float* pressureF,
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
		float* xAccExtF, float* yAccExtF, float* zAccExtF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_Accelerations<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xVelF, yVelF, zVelF, xVelB, yVelB, zVelB,
			densityF, pressureF, neighCountFF, neighCountFB, neighListFF, neighListFB,
			N_MAX_FF, N_MAX_FB, xAccF, yAccF, zAccF, xAccPressF, yAccPressF, zAccPressF,
			xAccViscF, yAccViscF, zAccViscF, xAccExtF, yAccExtF, zAccExtF);

		GPU::CheckLastKernel("K_Accelerations()");
	}

	void Launch_Integration(
		const float* xAccF, const float* yAccF, const float* zAccF,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* xPosF, float* yPosF, float* zPosF,
		float* xVelF, float* yVelF, float* zVelF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_Integration<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xAccF, yAccF, zAccF, xPosF, yPosF, zPosF, xVelF, yVelF, zVelF);

		GPU::CheckLastKernel("K_Integration()");
	}

} // namespace GPU01
