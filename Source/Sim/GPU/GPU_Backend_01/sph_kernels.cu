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
	__constant__ float c_restDensity;
	__constant__ float c_restDensitySquInv;
	__constant__ float c_massF;
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
		const float restDensitySquInv = 1.0f / (params.restDensity * params.restDensity);

		// store values in constant memory on the GPU
		GPU::Check(cudaMemcpyToSymbol(c_N_F, &N_F, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_N_B, &N_B, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_timeStep, &params.timeStep, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_sizeF, &params.particleSizeF, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_sizeB, &params.particleSizeB, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_restDensity, &params.restDensity, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_restDensitySquInv, &restDensitySquInv, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_massF, &params.massF, sizeof(float)));
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
	uint32_t Sell32ReadNeighbor_Delta8(
		const uint64_t chunkBase,
		const uint32_t chunkBaseIdx,
		const uint32_t lane,
		const uint32_t k,
		const uint8_t* __restrict__ nbrListDelta8)
	{
		return chunkBaseIdx + static_cast<uint32_t>(
			nbrListDelta8[chunkBase + static_cast<uint64_t>(k) * N_SellWidth + lane]);
	}

	__device__ __forceinline__
	uint32_t Sell32ReadNeighbor_Delta16(
		const uint64_t chunkBase,
		const uint32_t chunkBaseIdx,
		const uint32_t lane,
		const uint32_t k,
		const uint16_t* __restrict__ nbrListDelta16)
	{
		return chunkBaseIdx + static_cast<uint32_t>(
			nbrListDelta16[chunkBase + static_cast<uint64_t>(k) * N_SellWidth + lane]);
	}

	__device__ __forceinline__
	uint32_t Sell32ReadNeighbor_Abs32(
		const uint64_t chunkBase,
		const uint32_t lane,
		const uint32_t k,
		const uint32_t* __restrict__ nbrListAbs32)
	{
		return nbrListAbs32[chunkBase + static_cast<uint64_t>(k) * N_SellWidth + lane];
	}

	__device__ __forceinline__
	uint32_t Sell32ReadNeighbor(
		const uint64_t chunkBase,
		const uint32_t chunkBaseIdx,
		const uint8_t chunkFormat,
		const uint32_t lane,
		const uint32_t k,
		const uint8_t* __restrict__ nbrListDelta8,
		const uint16_t* __restrict__ nbrListDelta16,
		const uint32_t* __restrict__ nbrListAbs32)
	{
		if (chunkFormat == static_cast<uint8_t>(NbrChunkFormat::Delta8))
		{
			return Sell32ReadNeighbor_Delta8(
				chunkBase, chunkBaseIdx, lane, k, nbrListDelta8);
		}

		if (chunkFormat == static_cast<uint8_t>(NbrChunkFormat::Delta16))
		{
			return Sell32ReadNeighbor_Delta16(
				chunkBase, chunkBaseIdx, lane, k, nbrListDelta16);
		}

		return Sell32ReadNeighbor_Abs32(
			chunkBase, lane, k, nbrListAbs32);
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
		const float* __restrict__ xPosR,
		const float* __restrict__ yPosR,
		const float* __restrict__ zPosR,
		const float* __restrict__ psiB,
		const float* __restrict__ psiR,
		const uint16_t* __restrict__ nbrCountFF,
		const uint16_t* __restrict__ nbrCountFB,
		const uint16_t* __restrict__ nbrCountFR,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFR,
		const uint32_t* __restrict__ nbrChunkBaseIdxFF,
		const uint32_t* __restrict__ nbrChunkBaseIdxFB,
		const uint32_t* __restrict__ nbrChunkBaseIdxFR,
		const uint8_t* __restrict__ nbrChunkFormatFF,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrChunkFormatFR,
		const uint8_t* __restrict__ nbrListDelta8FF,
		const uint8_t* __restrict__ nbrListDelta8FB,
		const uint8_t* __restrict__ nbrListDelta8FR,
		const uint16_t* __restrict__ nbrListDelta16FF,
		const uint16_t* __restrict__ nbrListDelta16FB,
		const uint16_t* __restrict__ nbrListDelta16FR,
		const uint32_t* __restrict__ nbrListAbs32FF,
		const uint32_t* __restrict__ nbrListAbs32FB,
		const uint32_t* __restrict__ nbrListAbs32FR,
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
		const uint64_t baseFR = nbrChunkOffsetsFR[chunk];
		const uint32_t baseIdxFF = nbrChunkBaseIdxFF[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint32_t baseIdxFR = nbrChunkBaseIdxFR[chunk];
		const uint8_t formatFF = nbrChunkFormatFF[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint8_t formatFR = nbrChunkFormatFR[chunk];
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);
		const uint32_t countFR = static_cast<uint32_t>(nbrCountFR[i]);

		// FF
		for (uint32_t k = 0; k < countFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(
				baseFF, baseIdxFF, formatFF, lane, k,
				nbrListDelta8FF, nbrListDelta16FF, nbrListAbs32FF);

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
			const uint32_t j = Sell32ReadNeighbor(
				baseFB, baseIdxFB, formatFB, lane, k,
				nbrListDelta8FB, nbrListDelta16FB, nbrListAbs32FB);

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel weight
			float kw = 0.0f;
			KernelWeight_CubicSpline(dx, dy, dz, kw);

			// density contribution from boundary neighbor j
			tmp_density = fmaf(psiB[j], kw, tmp_density);
		}

		// FR
		for (uint32_t k = 0; k < countFR; k++)
		{
			const uint32_t j = Sell32ReadNeighbor(
				baseFR, baseIdxFR, formatFR, lane, k,
				nbrListDelta8FR, nbrListDelta16FR, nbrListAbs32FR);

			// position difference
			const float dx = xi - xPosR[j];
			const float dy = yi - yPosR[j];
			const float dz = zi - zPosR[j];

			// kernel weight
			float kw = 0.0f;
			KernelWeight_CubicSpline(dx, dy, dz, kw);

			// density contribution from rigid neighbor j
			tmp_density = fmaf(psiR[j], kw, tmp_density);
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
		const float* __restrict__ xPosR,
		const float* __restrict__ yPosR,
		const float* __restrict__ zPosR,
		const float* __restrict__ xVelF,
		const float* __restrict__ yVelF,
		const float* __restrict__ zVelF,
		const float* __restrict__ xVelR,
		const float* __restrict__ yVelR,
		const float* __restrict__ zVelR,
		const float* __restrict__ densityF,
		const float* __restrict__ volB,
		const float* __restrict__ volR,
		const uint16_t* __restrict__ nbrCountFF,
		const uint16_t* __restrict__ nbrCountFB,
		const uint16_t* __restrict__ nbrCountFR,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFR,
		const uint32_t* __restrict__ nbrChunkBaseIdxFF,
		const uint32_t* __restrict__ nbrChunkBaseIdxFB,
		const uint32_t* __restrict__ nbrChunkBaseIdxFR,
		const uint8_t* __restrict__ nbrChunkFormatFF,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrChunkFormatFR,
		const uint8_t* __restrict__ nbrListDelta8FF,
		const uint8_t* __restrict__ nbrListDelta8FB,
		const uint8_t* __restrict__ nbrListDelta8FR,
		const uint16_t* __restrict__ nbrListDelta16FF,
		const uint16_t* __restrict__ nbrListDelta16FB,
		const uint16_t* __restrict__ nbrListDelta16FR,
		const uint32_t* __restrict__ nbrListAbs32FF,
		const uint32_t* __restrict__ nbrListAbs32FB,
		const uint32_t* __restrict__ nbrListAbs32FR,
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
		const uint64_t baseFR = nbrChunkOffsetsFR[chunk];
		const uint32_t baseIdxFF = nbrChunkBaseIdxFF[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint32_t baseIdxFR = nbrChunkBaseIdxFR[chunk];
		const uint8_t formatFF = nbrChunkFormatFF[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint8_t formatFR = nbrChunkFormatFR[chunk];
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);
		const uint32_t countFR = static_cast<uint32_t>(nbrCountFR[i]);

		// FF
		for (uint32_t k = 0; k < countFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(
				baseFF, baseIdxFF, formatFF, lane, k,
				nbrListDelta8FF, nbrListDelta16FF, nbrListAbs32FF);

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
			const uint32_t j = Sell32ReadNeighbor(
				baseFB, baseIdxFB, formatFB, lane, k,
				nbrListDelta8FB, nbrListDelta16FB, nbrListAbs32FB);

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];
			const float dvx = vxi;
			const float dvy = vyi;
			const float dvz = vzi;

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			// viscosity acc contribution from boundary neighbor j
			const float accViscCoeff = volB[j]
									 * fmaf(dvx, dx, fmaf(dvy, dy, dvz * dz))
									 / (fmaf(dx, dx, fmaf(dy, dy, dz * dz))
									 + 0.01f * c_sizeF * c_sizeF);

			tmp_xAccViscB = fmaf(accViscCoeff, xkg, tmp_xAccViscB);
			tmp_yAccViscB = fmaf(accViscCoeff, ykg, tmp_yAccViscB);
			tmp_zAccViscB = fmaf(accViscCoeff, zkg, tmp_zAccViscB);
		}

		// FR
		for (uint32_t k = 0; k < countFR; k++)
		{
			const uint32_t j = Sell32ReadNeighbor(
				baseFR, baseIdxFR, formatFR, lane, k,
				nbrListDelta8FR, nbrListDelta16FR, nbrListAbs32FR);

			// position difference
			const float dx = xi - xPosR[j];
			const float dy = yi - yPosR[j];
			const float dz = zi - zPosR[j];
			const float dvx = vxi - xVelR[j];
			const float dvy = vyi - yVelR[j];
			const float dvz = vzi - zVelR[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			// viscosity acc contribution from rigid neighbor j
			const float accViscCoeff = volR[j]
									 * fmaf(dvx, dx, fmaf(dvy, dy, dvz * dz))
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
		const float* __restrict__ xPosR,
		const float* __restrict__ yPosR,
		const float* __restrict__ zPosR,
		const float* __restrict__ xVelPredF,
		const float* __restrict__ yVelPredF,
		const float* __restrict__ zVelPredF,
		const float* __restrict__ xVelR,
		const float* __restrict__ yVelR,
		const float* __restrict__ zVelR,
		const float* __restrict__ densityF,
		const float* __restrict__ psiB,
		const float* __restrict__ psiR,
		const uint16_t* __restrict__ nbrCountFF,
		const uint16_t* __restrict__ nbrCountFB,
		const uint16_t* __restrict__ nbrCountFR,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFR,
		const uint32_t* __restrict__ nbrChunkBaseIdxFF,
		const uint32_t* __restrict__ nbrChunkBaseIdxFB,
		const uint32_t* __restrict__ nbrChunkBaseIdxFR,
		const uint8_t* __restrict__ nbrChunkFormatFF,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrChunkFormatFR,
		const uint8_t* __restrict__ nbrListDelta8FF,
		const uint8_t* __restrict__ nbrListDelta8FB,
		const uint8_t* __restrict__ nbrListDelta8FR,
		const uint16_t* __restrict__ nbrListDelta16FF,
		const uint16_t* __restrict__ nbrListDelta16FB,
		const uint16_t* __restrict__ nbrListDelta16FR,
		const uint32_t* __restrict__ nbrListAbs32FF,
		const uint32_t* __restrict__ nbrListAbs32FB,
		const uint32_t* __restrict__ nbrListAbs32FR,
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
		float gradSquSumF = 0.0f;
		const uint32_t chunk = i / N_SellWidth;
		const uint32_t lane = i % N_SellWidth;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint64_t baseFR = nbrChunkOffsetsFR[chunk];
		const uint32_t baseIdxFF = nbrChunkBaseIdxFF[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint32_t baseIdxFR = nbrChunkBaseIdxFR[chunk];
		const uint8_t formatFF = nbrChunkFormatFF[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint8_t formatFR = nbrChunkFormatFR[chunk];
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);
		const uint32_t countFR = static_cast<uint32_t>(nbrCountFR[i]);

		// FF
		for (uint32_t k = 0; k < countFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(
				baseFF, baseIdxFF, formatFF, lane, k,
				nbrListDelta8FF, nbrListDelta16FF, nbrListAbs32FF);

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
			gradSquSumF += fmaf(xkg, xkg, fmaf(ykg, ykg, zkg * zkg));
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
			const uint32_t j = Sell32ReadNeighbor(
				baseFB, baseIdxFB, formatFB, lane, k,
				nbrListDelta8FB, nbrListDelta16FB, nbrListAbs32FB);

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];
			const float dvpx = vpxi;
			const float dvpy = vpyi;
			const float dvpz = vpzi;

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			// source term contribution from boundary neighbor
			const float psiBj = psiB[j];
			tmp_sourceTermB += psiBj * fmaf(dvpx, xkg, fmaf(dvpy, ykg, dvpz * zkg));

			xCTermB += psiBj * xkg;
			yCTermB += psiBj * ykg;
			zCTermB += psiBj * zkg;
		}

		// FR
		for (uint32_t k = 0; k < countFR; k++)
		{
			const uint32_t j = Sell32ReadNeighbor(
				baseFR, baseIdxFR, formatFR, lane, k,
				nbrListDelta8FR, nbrListDelta16FR, nbrListAbs32FR);

			// position difference
			const float dx = xi - xPosR[j];
			const float dy = yi - yPosR[j];
			const float dz = zi - zPosR[j];
			const float dvpx = vpxi - xVelR[j];
			const float dvpy = vpyi - yVelR[j];
			const float dvpz = vpzi - zVelR[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			// source term contribution from rigid neighbor
			const float psiRj = psiR[j];
			tmp_sourceTermB += psiRj * fmaf(dvpx, xkg, fmaf(dvpy, ykg, dvpz * zkg));

			xCTermB += psiRj * xkg;
			yCTermB += psiRj * ykg;
			zCTermB += psiRj * zkg;
		}

		// finalize source term
		const float fin_sourceTermF = c_restDensity - densityF[i] - c_timeStep
									* (tmp_sourceTermF + tmp_sourceTermB);

		// finalize c-term
		const float xCTerm = -1.0f * (xCTermF * c_restDensitySquInv * c_massF)
			- 2.0f * c_pressGamma * (xCTermB * c_restDensitySquInv);
		const float yCTerm = -1.0f * (yCTermF * c_restDensitySquInv * c_massF)
			- 2.0f * c_pressGamma * (yCTermB * c_restDensitySquInv);
		const float zCTerm = -1.0f * (zCTermF * c_restDensitySquInv * c_massF)
			- 2.0f * c_pressGamma * (zCTermB * c_restDensitySquInv);

		// part A of diagonal element
		const float tmp_diagElemA = c_massF *
			fmaf(xCTerm, xCTermF, fmaf(yCTerm, yCTermF, zCTerm * zCTermF));
		// part B of diagonal element
		const float tmp_diagElemB =
			-c_massF * c_massF * c_restDensitySquInv * gradSquSumF;
		// part C of diagonal element
		const float tmp_diagElemC =
			fmaf(xCTerm, xCTermB, fmaf(yCTerm, yCTermB, zCTerm * zCTermB));

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
		const float* __restrict__ xPosR,
		const float* __restrict__ yPosR,
		const float* __restrict__ zPosR,
		const float* __restrict__ pressureF,
		const float* __restrict__ psiB,
		const float* __restrict__ psiR,
		const uint16_t* __restrict__ nbrCountFF,
		const uint16_t* __restrict__ nbrCountFB,
		const uint16_t* __restrict__ nbrCountFR,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFR,
		const uint32_t* __restrict__ nbrChunkBaseIdxFF,
		const uint32_t* __restrict__ nbrChunkBaseIdxFB,
		const uint32_t* __restrict__ nbrChunkBaseIdxFR,
		const uint8_t* __restrict__ nbrChunkFormatFF,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrChunkFormatFR,
		const uint8_t* __restrict__ nbrListDelta8FF,
		const uint8_t* __restrict__ nbrListDelta8FB,
		const uint8_t* __restrict__ nbrListDelta8FR,
		const uint16_t* __restrict__ nbrListDelta16FF,
		const uint16_t* __restrict__ nbrListDelta16FB,
		const uint16_t* __restrict__ nbrListDelta16FR,
		const uint32_t* __restrict__ nbrListAbs32FF,
		const uint32_t* __restrict__ nbrListAbs32FB,
		const uint32_t* __restrict__ nbrListAbs32FR,
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
		const float densi = c_restDensity;
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
		const uint64_t baseFR = nbrChunkOffsetsFR[chunk];
		const uint32_t baseIdxFF = nbrChunkBaseIdxFF[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint32_t baseIdxFR = nbrChunkBaseIdxFR[chunk];
		const uint8_t formatFF = nbrChunkFormatFF[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint8_t formatFR = nbrChunkFormatFR[chunk];
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);
		const uint32_t countFR = static_cast<uint32_t>(nbrCountFR[i]);

		// FF
		for (uint32_t k = 0; k < countFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(
				baseFF, baseIdxFF, formatFF, lane, k,
				nbrListDelta8FF, nbrListDelta16FF, nbrListAbs32FF);

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

		// FB
		for (uint32_t k = 0; k < countFB; k++)
		{
			const uint32_t j = Sell32ReadNeighbor(
				baseFB, baseIdxFB, formatFB, lane, k,
				nbrListDelta8FB, nbrListDelta16FB, nbrListAbs32FB);

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			const float accCoeffB = c_pressGamma * 2.0f
								  * psiB[j] * presi * densiInvSqu;

			// pressure acc contribution from boundary neighbor j
			tmp_xAccPressF = fmaf(accCoeffB, xkg, tmp_xAccPressF);
			tmp_yAccPressF = fmaf(accCoeffB, ykg, tmp_yAccPressF);
			tmp_zAccPressF = fmaf(accCoeffB, zkg, tmp_zAccPressF);
		}

		// FR
		for (uint32_t k = 0; k < countFR; k++)
		{
			const uint32_t j = Sell32ReadNeighbor(
				baseFR, baseIdxFR, formatFR, lane, k,
				nbrListDelta8FR, nbrListDelta16FR, nbrListAbs32FR);

			// position difference
			const float dx = xi - xPosR[j];
			const float dy = yi - yPosR[j];
			const float dz = zi - zPosR[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			const float accCoeffB = c_pressGamma * 2.0f
								  * psiR[j] * presi * densiInvSqu;

			// pressure acc contribution from rigid neighbor j
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
		const float* __restrict__ xPosR,
		const float* __restrict__ yPosR,
		const float* __restrict__ zPosR,
		const float* __restrict__ xAccPressF,
		const float* __restrict__ yAccPressF,
		const float* __restrict__ zAccPressF,
		const float* __restrict__ sourceTermF,
		const float* __restrict__ diagElemF,
		const float* __restrict__ psiB,
		const float* __restrict__ psiR,
		const uint16_t* __restrict__ nbrCountFF,
		const uint16_t* __restrict__ nbrCountFB,
		const uint16_t* __restrict__ nbrCountFR,
		const uint64_t* __restrict__ nbrChunkOffsetsFF,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFR,
		const uint32_t* __restrict__ nbrChunkBaseIdxFF,
		const uint32_t* __restrict__ nbrChunkBaseIdxFB,
		const uint32_t* __restrict__ nbrChunkBaseIdxFR,
		const uint8_t* __restrict__ nbrChunkFormatFF,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrChunkFormatFR,
		const uint8_t* __restrict__ nbrListDelta8FF,
		const uint8_t* __restrict__ nbrListDelta8FB,
		const uint8_t* __restrict__ nbrListDelta8FR,
		const uint16_t* __restrict__ nbrListDelta16FF,
		const uint16_t* __restrict__ nbrListDelta16FB,
		const uint16_t* __restrict__ nbrListDelta16FR,
		const uint32_t* __restrict__ nbrListAbs32FF,
		const uint32_t* __restrict__ nbrListAbs32FB,
		const uint32_t* __restrict__ nbrListAbs32FR,
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
		const uint64_t baseFR = nbrChunkOffsetsFR[chunk];
		const uint32_t baseIdxFF = nbrChunkBaseIdxFF[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint32_t baseIdxFR = nbrChunkBaseIdxFR[chunk];
		const uint8_t formatFF = nbrChunkFormatFF[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint8_t formatFR = nbrChunkFormatFR[chunk];
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);
		const uint32_t countFR = static_cast<uint32_t>(nbrCountFR[i]);

		// FF
		for (uint32_t k = 0; k < countFF; k++)
		{
			// global index of neighbor particle
			const uint32_t j = Sell32ReadNeighbor(
				baseFF, baseIdxFF, formatFF, lane, k,
				nbrListDelta8FF, nbrListDelta16FF, nbrListAbs32FF);

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
			const uint32_t j = Sell32ReadNeighbor(
				baseFB, baseIdxFB, formatFB, lane, k,
				nbrListDelta8FB, nbrListDelta16FB, nbrListAbs32FB);

			// position difference
			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			const float psiBj = psiB[j];

			// system vector element contribution from boundary neighbor j
			tmp_xSysVecElem = fmaf(psiBj * axi, xkg, tmp_xSysVecElem);
			tmp_ySysVecElem = fmaf(psiBj * ayi, ykg, tmp_ySysVecElem);
			tmp_zSysVecElem = fmaf(psiBj * azi, zkg, tmp_zSysVecElem);
		}

		// FR
		for (uint32_t k = 0; k < countFR; k++)
		{
			const uint32_t j = Sell32ReadNeighbor(
				baseFR, baseIdxFR, formatFR, lane, k,
				nbrListDelta8FR, nbrListDelta16FR, nbrListAbs32FR);

			// position difference
			const float dx = xi - xPosR[j];
			const float dy = yi - yPosR[j];
			const float dz = zi - zPosR[j];

			// kernel gradient
			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(dx, dy, dz, xkg, ykg, zkg);

			const float psiRj = psiR[j];

			// system vector element contribution from rigid neighbor j
			tmp_xSysVecElem = fmaf(psiRj * axi, xkg, tmp_xSysVecElem);
			tmp_ySysVecElem = fmaf(psiRj * ayi, ykg, tmp_ySysVecElem);
			tmp_zSysVecElem = fmaf(psiRj * azi, zkg, tmp_zSysVecElem);
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
		float* densityF)
	{
		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_Density<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR, psiB, psiR,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrChunkOffsetsFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR,
			nbrListDelta8FF, nbrListDelta8FB, nbrListDelta8FR,
			nbrListDelta16FF, nbrListDelta16FB, nbrListDelta16FR,
			nbrListAbs32FF, nbrListAbs32FB, nbrListAbs32FR, densityF);

		GPU::CheckLastKernel("K_Density()");
	}

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
		float* xVelPredF, float* yVelPredF, float* zVelPredF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_NonPressureAcceleration<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR,
			xVelF, yVelF, zVelF, xVelR, yVelR, zVelR, densityF, volB, volR,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrChunkOffsetsFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR,
			nbrListDelta8FF, nbrListDelta8FB, nbrListDelta8FR,
			nbrListDelta16FF, nbrListDelta16FB, nbrListDelta16FR,
			nbrListAbs32FF, nbrListAbs32FB, nbrListAbs32FR,
			xVelPredF, yVelPredF, zVelPredF);

		GPU::CheckLastKernel("K_NonPressureAcceleration()");
	}

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
		float* pressureF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_SourceTermAndDiagonalElementAndPressureJacobiInit<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR,
			xVelPredF, yVelPredF, zVelPredF, xVelR, yVelR, zVelR, densityF, psiB, psiR,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrChunkOffsetsFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR,
			nbrListDelta8FF, nbrListDelta8FB, nbrListDelta8FR,
			nbrListDelta16FF, nbrListDelta16FB, nbrListDelta16FR,
			nbrListAbs32FF, nbrListAbs32FB, nbrListAbs32FR,
			sourceTermF, diagElemF, pressureF);

		GPU::CheckLastKernel("K_SourceTermAndDiagonalElementAndPressureJacobiInit()");
	}

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
		float* zAccPressF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_PressureAcceleration<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR, pressureF, psiB, psiR,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrChunkOffsetsFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR,
			nbrListDelta8FF, nbrListDelta8FB, nbrListDelta8FR,
			nbrListDelta16FF, nbrListDelta16FB, nbrListDelta16FR,
			nbrListAbs32FF, nbrListAbs32FB, nbrListAbs32FR,
			xAccPressF, yAccPressF, zAccPressF);

		GPU::CheckLastKernel("K_PressureAcceleration()");
	}

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
		float* pressureF)
	{
		// TODO: initialize constants if needed

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_PressureRelaxedJacobiSolver<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR,
			xAccPressF, yAccPressF, zAccPressF, sourceTermF, diagElemF, psiB, psiR,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrChunkOffsetsFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR,
			nbrListDelta8FF, nbrListDelta8FB, nbrListDelta8FR,
			nbrListDelta16FF, nbrListDelta16FB, nbrListDelta16FR,
			nbrListAbs32FF, nbrListAbs32FB, nbrListAbs32FR, pressureF);

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
