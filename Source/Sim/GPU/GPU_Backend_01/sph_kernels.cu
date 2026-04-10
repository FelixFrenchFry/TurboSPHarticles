// sph_kernels.cu

// external
#include <cuda_runtime.h>
#include <numbers>

// internal
#include "Source/Sim/GPU/gpu_utils.h"
#include "sph_kernels.cuh"

constexpr float  PI = std::numbers::pi_v<float>;



namespace GPU01
{
	static constexpr uint32_t N_SellWidth = 32u;

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
	__constant__ float c_pressGamma;
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
		GPU::Check(cudaMemcpyToSymbol(c_pressGamma, &params.pressGamma, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_jacobiOmega, &params.jacobiOmega, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_viscosityF, &params.viscosityF, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_viscosityB, &params.viscosityB, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_xGravity, &params.gravity.x, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_yGravity, &params.gravity.y, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_zGravity, &params.gravity.z, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_kernelRadiusSqu, &kernelRadiusSqu, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_cubicSplineAlpha, &cubicSplineAlpha, sizeof(float)));

		//GPU::StreamSync(stream);
	}

	// ------------------------------------------------------------------------------------------------
	// --- [ helpers ] --------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------

	__device__ __forceinline__
	void KernelWeight_CubicSpline(
		float dx, float dy, float dz,
		float& w)
	{
		float distSqu = dx * dx + dy * dy + dz * dz;
		float dist = sqrtf(distSqu);
		float distNorm = c_sizeF > 0.0f ? dist / c_sizeF : 0.0f;

		float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
		float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

		// kernel weight
		w = c_cubicSplineAlpha * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);
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
		if (dist > 1e-7f && distNorm > 0.0f)
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
	uint32_t Sell32ReadNeighbor(
		const uint64_t chunkBase,
		const uint32_t lane,
		const uint32_t k,
		const uint32_t* __restrict__ nbrList)
	{
		return nbrList[chunkBase + static_cast<uint64_t>(k) * N_SellWidth + lane];
	}

	// ------------------------------------------------------------------------------------------------
	// --- [ kernels launched by host ] ---------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------

	__global__
	void K_Density(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const uint16_t* __restrict__ nbrCountFF,
		const uint16_t* __restrict__ nbrCountFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint32_t* __restrict__ nbrListFF,
		const uint32_t* __restrict__ nbrListFB,
		// destinations
		float* __restrict__ densityF)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_N_F) { return; }

		// particle i data
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];

		// temp accumulation variable
		float tmp_density = 0.0f;

		const uint32_t chunk = i / N_SellWidth;
		const uint32_t lane = i % N_SellWidth;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);

		// FF
		for (uint32_t k = 0; k < countFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(baseFF, lane, k, nbrListFF);

			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// kernel weight
			float kw = 0.0f;
			KernelWeight_CubicSpline(dx, dy, dz, kw);

			// density contribution from fluid neighbor j
			tmp_density = fmaf(c_massF,kw, tmp_density);
		}

		// FB
		for (uint32_t k = 0; k < countFB; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(baseFB, lane, k, nbrListFB);

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel weight
			float kw = 0.0f;
			KernelWeight_CubicSpline(dx, dy, dz, kw);

			// density contribution from boundary neighbor j
			tmp_density = fmaf(c_massB, kw, tmp_density);
		}

		densityF[i] = tmp_density;
	}

	__global__
	void K_NonPressureAcceleration(
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
		const uint16_t* __restrict__ nbrCountFF,
		const uint16_t* __restrict__ nbrCountFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint32_t* __restrict__ nbrListFF,
		const uint32_t* __restrict__ nbrListFB,
		// destinations
		float* __restrict__ xVelPredF,
		float* __restrict__ yVelPredF,
		float* __restrict__ zVelPredF)
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

		const uint32_t chunk = i / N_SellWidth;
		const uint32_t lane = i % N_SellWidth;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);

		// FF
		for (uint32_t k = 0; k < countFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(baseFF, lane, k, nbrListFF);

			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// velocity difference
			const float dvx = vxi - xVelF[j];
			const float dvy = vyi - yVelF[j];
			const float dvz = vzi - zVelF[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			// viscosity acc contribution from fluid neighbor j
			const float accViscCoeff = (c_massF / fmaxf(densityF[j], 1e-7f))
									 * fmaf(dvx, dx, fmaf(dvy, dy, dvz * dz))
									 / (fmaf(dx, dx, fmaf(dy, dy, dz * dz))
									 + 0.01f * c_sizeF * c_sizeF);

			tmp_xAccViscF = fmaf(accViscCoeff, xkg, tmp_xAccViscF);
			tmp_yAccViscF = fmaf(accViscCoeff, ykg, tmp_yAccViscF);
			tmp_zAccViscF = fmaf(accViscCoeff, zkg, tmp_zAccViscF);
		}

		// temp accumulation variables
		float tmp_xAccViscB = 0.0f;
		float tmp_yAccViscB = 0.0f;
		float tmp_zAccViscB = 0.0f;

		// FB
		for (uint32_t k = 0; k < countFB; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(baseFB, lane, k, nbrListFB);

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			// viscosity acc contribution from boundary neighbor j
			const float accViscCoeff = (c_massB / c_restDensityB)
									 * fmaf(vxi, dx, fmaf(vyi, dy, vzi * dz))
									 / (fmaf(dx, dx, fmaf(dy, dy, dz * dz))
									 + 0.01f * c_sizeF * c_sizeF);

			tmp_xAccViscB = fmaf(accViscCoeff, xkg, tmp_xAccViscB);
			tmp_yAccViscB = fmaf(accViscCoeff, ykg, tmp_yAccViscB);
			tmp_zAccViscB = fmaf(accViscCoeff, zkg, tmp_zAccViscB);
		}

		// finalize accelerations
		// (3D -> factor 2 * (dim + 2))
		tmp_xAccViscF *= 2.0f * (3 + 2) * c_viscosityF;
		tmp_yAccViscF *= 2.0f * (3 + 2) * c_viscosityF;
		tmp_zAccViscF *= 2.0f * (3 + 2) * c_viscosityF;
		tmp_xAccViscB *= 2.0f * (3 + 2) * c_viscosityB;
		tmp_yAccViscB *= 2.0f * (3 + 2) * c_viscosityB;
		tmp_zAccViscB *= 2.0f * (3 + 2) * c_viscosityB;

		// TODO: add surface tension acc contribution
		const float xAccNonPressF = tmp_xAccViscF + tmp_xAccViscB + c_xGravity;
		const float yAccNonPressF = tmp_yAccViscF + tmp_yAccViscB + c_yGravity;
		const float zAccNonPressF = tmp_zAccViscF + tmp_zAccViscB + c_zGravity;

		// compute predicted velocity from non-pressure accelerations
		xVelPredF[i] = fmaf(c_timeStep, xAccNonPressF, xVelF[i]);
		yVelPredF[i] = fmaf(c_timeStep, yAccNonPressF, yVelF[i]);
		zVelPredF[i] = fmaf(c_timeStep, zAccNonPressF, zVelF[i]);
	}

	__global__
	void K_SourceTermAndDiagonalElementAndPressureJacobiInit(
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
		const uint16_t* __restrict__ nbrCountFF,
		const uint16_t* __restrict__ nbrCountFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint32_t* __restrict__ nbrListFF,
		const uint32_t* __restrict__ nbrListFB,
		// destinations
		float* __restrict__ sourceTermF,
		float* __restrict__ diagElemF,
		float* __restrict__ pressureF)
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

		// temp accumulation variables for fluid
		float tmp_sourceTermF = 0.0f;
		float xCTermF = 0.0f;
		float yCTermF = 0.0f;
		float zCTermF = 0.0f;
		const uint32_t chunk = i / N_SellWidth;
		const uint32_t lane = i % N_SellWidth;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);

		// FF
		for (uint32_t k = 0; k < countFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(baseFF, lane, k, nbrListFF);

			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// predicted velocity difference
			const float dvpx = vpxi - xVelPredF[j];
			const float dvpy = vpyi - yVelPredF[j];
			const float dvpz = vpzi - zVelPredF[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			// source term contribution from fluid neighbor
			tmp_sourceTermF += fmaf(dvpx, xkg, fmaf(dvpy, ykg, dvpz * zkg));

			xCTermF += xkg;
			yCTermF += ykg;
			zCTermF += zkg;
		}
		tmp_sourceTermF *= c_massF;

		// temp accumulation variables for boundary
		float tmp_sourceTermB = 0.0f;
		float xCTermB = 0.0f;
		float yCTermB = 0.0f;
		float zCTermB = 0.0f;

		// FB
		for (uint32_t k = 0; k < countFB; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(baseFB, lane, k, nbrListFB);

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			// source term contribution from boundary neighbor
			// TODO: subtract actual boundary velocity when adding non-static boundaries?
			tmp_sourceTermB += fmaf(vpxi, xkg, fmaf(vpyi, ykg, vpzi * zkg));

			xCTermB += xkg;
			yCTermB += ykg;
			zCTermB += zkg;
		}
		tmp_sourceTermB *= c_massB;

		// finalize source term
		const float fin_sourceTermF = c_restDensityF - densityF[i] - c_timeStep
									* (tmp_sourceTermF + tmp_sourceTermB);

		// finalize c-term
		const float xCTerm = -1.0f * (xCTermF * c_restDensitySquInvF * c_massF)
			- 2.0f * c_pressGamma * (xCTermB * c_restDensitySquInvF * c_massB);
		const float yCTerm = -1.0f * (yCTermF * c_restDensitySquInvF * c_massF)
			- 2.0f * c_pressGamma * (yCTermB * c_restDensitySquInvF * c_massB);
		const float zCTerm = -1.0f * (zCTermF * c_restDensitySquInvF * c_massF)
			- 2.0f * c_pressGamma * (zCTermB * c_restDensitySquInvF * c_massB);

		float tmp_diagElemA = 0.0f;
		float tmp_diagElemB = 0.0f;
		float tmp_diagElemC = 0.0f;

		// part A and B
		for (uint32_t k = 0; k < countFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(baseFF, lane, k, nbrListFF);

			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			tmp_diagElemA += fmaf(xCTerm, xkg, fmaf(yCTerm, ykg, zCTerm * zkg));

			const float tmp_x = (c_massF * c_restDensitySquInvF) * (-xkg);
			const float tmp_y = (c_massF * c_restDensitySquInvF) * (-ykg);
			const float tmp_z = (c_massF * c_restDensitySquInvF) * (-zkg);

			tmp_diagElemB += fmaf(tmp_x, xkg, fmaf(tmp_y, ykg, tmp_z * zkg));
		}
		tmp_diagElemA *= c_massF;
		tmp_diagElemB *= c_massF;

		// part C
		for (uint32_t k = 0; k < countFB; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(baseFB, lane, k, nbrListFB);

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			tmp_diagElemC += fmaf(xCTerm, xkg, fmaf(yCTerm, ykg, zCTerm * zkg));
		}
		tmp_diagElemC *= c_massB;

		// finalize diagonal element
		const float fin_diagElemF = c_timeStep * c_timeStep
								  * (tmp_diagElemA + tmp_diagElemB + tmp_diagElemC);

		float tmp_pressureF = 0.0f;

		// only update pressure if particle has neighbors
		if (!(fabsf(fin_diagElemF) <= 1e-7f))
		{
			tmp_pressureF = c_jacobiOmega * (fin_sourceTermF / fin_diagElemF);
		}

		// finalize pressure value by clamping to non-negative
		// (and perform final global memory writes)
		pressureF[i] = fmaxf(tmp_pressureF, 0.0f);
		sourceTermF[i] = fin_sourceTermF;
		diagElemF[i] = fin_diagElemF;
	}

	__global__
	void K_PressureAcceleration(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const float* __restrict__ pressureF,
		const uint16_t* __restrict__ nbrCountFF,
		const uint16_t* __restrict__ nbrCountFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint32_t* __restrict__ nbrListFF,
		const uint32_t* __restrict__ nbrListFB,
		// destinations
		float* __restrict__ xAccPressF,
		float* __restrict__ yAccPressF,
		float* __restrict__ zAccPressF)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_N_F) { return; }

		// particle i data
		// (global memory reads)
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];
		const float presi = pressureF[i];
		const float densi = c_restDensityF;
		const float densiInvSqu = 1.0f / (densi * densi);
		const float accCoeffF_outer = c_massF * presi * densiInvSqu;

		// temp accumulation variables
		float tmp_xAccPressF = 0.0f;
		float tmp_yAccPressF = 0.0f;
		float tmp_zAccPressF = 0.0f;
		const uint32_t chunk = i / N_SellWidth;
		const uint32_t lane = i % N_SellWidth;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);

		// FF
		for (uint32_t k = 0; k < countFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(baseFF, lane, k, nbrListFF);

			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			const float accCoeffF = accCoeffF_outer + c_massF
								  * pressureF[j] * densiInvSqu;

			// pressure acc contribution from fluid neighbor j
			tmp_xAccPressF = fmaf(accCoeffF, xkg, tmp_xAccPressF);
			tmp_yAccPressF = fmaf(accCoeffF, ykg, tmp_yAccPressF);
			tmp_zAccPressF = fmaf(accCoeffF, zkg, tmp_zAccPressF);
		}

		const float accCoeffB = c_pressGamma * c_massB
							  * 2.0f * presi * densiInvSqu;

		// FB
		for (uint32_t k = 0; k < countFB; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(baseFB, lane, k, nbrListFB);

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			// pressure acc contribution from boundary neighbor j
			tmp_xAccPressF = fmaf(accCoeffB, xkg, tmp_xAccPressF);
			tmp_yAccPressF = fmaf(accCoeffB, ykg, tmp_yAccPressF);
			tmp_zAccPressF = fmaf(accCoeffB, zkg, tmp_zAccPressF);
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
	void K_PressureRelaxedJacobiSolver(
		const float* __restrict__ xPosF,
		const float* __restrict__ yPosF,
		const float* __restrict__ zPosF,
		const float* __restrict__ xPosB,
		const float* __restrict__ yPosB,
		const float* __restrict__ zPosB,
		const float* __restrict__ xAccPressF,
		const float* __restrict__ yAccPressF,
		const float* __restrict__ zAccPressF,
		const float* __restrict__ sourceTermF,
		const float* __restrict__ diagElemF,
		const uint16_t* __restrict__ nbrCountFF,
		const uint16_t* __restrict__ nbrCountFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint32_t* __restrict__ nbrListFF,
		const uint32_t* __restrict__ nbrListFB,
		// destinations
		float* __restrict__ pressureF)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_N_F) { return; }

		// particle i data
		// (global memory reads)
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];
		const float axi = xAccPressF[i];
		const float ayi = yAccPressF[i];
		const float azi = zAccPressF[i];

		// temp accumulation variables
		float tmp_xSysVecElem = 0.0f;
		float tmp_ySysVecElem = 0.0f;
		float tmp_zSysVecElem = 0.0f;
		const uint32_t chunk = i / N_SellWidth;
		const uint32_t lane = i % N_SellWidth;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);

		// FF
		for (uint32_t k = 0; k < countFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(baseFF, lane, k, nbrListFF);

			// position difference
			const float dx = xi - xPosF[j];
			const float dy = yi - yPosF[j];
			const float dz = zi - zPosF[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			// system vector element contribution from fluid neighbor j
			tmp_xSysVecElem = fmaf(c_massF * (axi - xAccPressF[j]), xkg, tmp_xSysVecElem);
			tmp_ySysVecElem = fmaf(c_massF * (ayi - yAccPressF[j]), ykg, tmp_ySysVecElem);
			tmp_zSysVecElem = fmaf(c_massF * (azi - zAccPressF[j]), zkg, tmp_zSysVecElem);
		}

		// FB
		for (uint32_t k = 0; k < countFB; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(baseFB, lane, k, nbrListFB);

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			// system vector element contribution from boundary neighbor j
			tmp_xSysVecElem = fmaf(c_massB * axi, xkg, tmp_xSysVecElem);
			tmp_ySysVecElem = fmaf(c_massB * ayi, ykg, tmp_ySysVecElem);
			tmp_zSysVecElem = fmaf(c_massB * azi, zkg, tmp_zSysVecElem);
		}

		float tmp_pressureF = 0.0f;

		// only update pressure if particle has neighbors
		if (!(fabsf(diagElemF[i]) <= 1e-7f))
		{
			tmp_pressureF = pressureF[i] + c_jacobiOmega
						  * ((sourceTermF[i] - (c_timeStep * c_timeStep
						  * (tmp_xSysVecElem + tmp_ySysVecElem + tmp_zSysVecElem)))
						  / diagElemF[i]);
		}

		// finalize pressure value by clamping to non-negative
		pressureF[i] = fmaxf(tmp_pressureF, 0.0f);
	}

	__global__
	void K_Integration(
		const float* __restrict__ xVelPredF,
		const float* __restrict__ yVelPredF,
		const float* __restrict__ zVelPredF,
		const float* __restrict__ xAccPressF,
		const float* __restrict__ yAccPressF,
		const float* __restrict__ zAccPressF,
		// destinations
		float* __restrict__ xPosF,
		float* __restrict__ yPosF,
		float* __restrict__ zPosF,
		float* __restrict__ xVelF,
		float* __restrict__ yVelF,
		float* __restrict__ zVelF)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= c_N_F) { return; }

		// explicit Euler integration for velocity
		// (now only refining predicted velocity with pressure acc)
		xVelF[i] = fmaf(c_timeStep, xAccPressF[i], xVelPredF[i]);
		yVelF[i] = fmaf(c_timeStep, yAccPressF[i], yVelPredF[i]);
		zVelF[i] = fmaf(c_timeStep, zAccPressF[i], zVelPredF[i]);

		// semi-implicit Euler integration
		// (using updated velocity for position update)
		xPosF[i] = fmaf(c_timeStep, xVelF[i], xPosF[i]);
		yPosF[i] = fmaf(c_timeStep, yVelF[i], yPosF[i]);
		zPosF[i] = fmaf(c_timeStep, zVelF[i], zPosF[i]);
	}

} // namespace GPU01

// -------------------------------------------------------------------------------------------------
// --- [ kernel launchers ] ------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

namespace GPU01
{
	void Launch_Density(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrListFF,
		const uint32_t* nbrListFB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* densityF)
	{
		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_Density<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, nbrCountFF, nbrCountFB,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrListFF, nbrListFB, densityF);

		GPU::CheckLastKernel("K_Density()");
	}

	void Launch_NonPressureAcceleration(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* densityF,
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrListFF,
		const uint32_t* nbrListFB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* xVelPredF, float* yVelPredF, float* zVelPredF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_NonPressureAcceleration<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xVelF, yVelF, zVelF, densityF,
			nbrCountFF, nbrCountFB, nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrListFF, nbrListFB,
			xVelPredF, yVelPredF, zVelPredF);

		GPU::CheckLastKernel("K_NonPressureAcceleration()");
	}

	void Launch_SourceTermAndDiagonalElementAndPressureJacobiInit(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* xVelPredF, const float* yVelPredF, const float* zVelPredF,
		const float* densityF,
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrListFF,
		const uint32_t* nbrListFB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* sourceTermF,
		float* diagElemF,
		float* pressureF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_SourceTermAndDiagonalElementAndPressureJacobiInit<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xVelPredF, yVelPredF, zVelPredF, densityF,
			nbrCountFF, nbrCountFB, nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrListFF, nbrListFB,
			sourceTermF, diagElemF, pressureF);

		GPU::CheckLastKernel("K_SourceTermAndDiagonalElementAndPressureJacobiInit()");
	}

	void Launch_PressureAcceleration(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const float* pressureF,
		const uint16_t* nbrCountFF,
		const uint16_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFF,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrListFF,
		const uint32_t* nbrListFB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* xAccPressF,
		float* yAccPressF,
		float* zAccPressF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_PressureAcceleration<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, pressureF,
			nbrCountFF, nbrCountFB, nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrListFF, nbrListFB,
			xAccPressF, yAccPressF, zAccPressF);

		GPU::CheckLastKernel("K_PressureAcceleration()");
	}

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
		const uint32_t* nbrListFF,
		const uint32_t* nbrListFB,
		const uint32_t N_F, const uint32_t N_B,
		cudaStream_t stream,
		// destinations
		float* pressureF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_PressureRelaxedJacobiSolver<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xAccPressF, yAccPressF, zAccPressF,
			sourceTermF, diagElemF, nbrCountFF, nbrCountFB,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrListFF, nbrListFB,
			pressureF);

		GPU::CheckLastKernel("K_PressureRelaxedJacobiSolver()");
	}

	void Launch_Integration(
		const float* xVelPredF, const float* yVelPredF, const float* zVelPredF,
		const float* xAccPressF, const float* yAccPressF, const float* zAccPressF,
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
			xVelPredF, yVelPredF, zVelPredF, xAccPressF, yAccPressF, zAccPressF,
			xPosF, yPosF, zPosF, xVelF, yVelF, zVelF);

		GPU::CheckLastKernel("K_Integration()");
	}

} // namespace GPU01
