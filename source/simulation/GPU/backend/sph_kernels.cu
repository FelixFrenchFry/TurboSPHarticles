// sph_kernels.cu

// external
#include <cfloat>
#include <cuda_runtime.h>

// internal
#include "source/simulation/core/solver_constants.h"
#include "source/simulation/GPU/gpu_utils.h"
#include "sph_kernels.cuh"




namespace GPU
{
	__constant__ uint32_t c_N_F;
	__constant__ uint32_t c_N_B;
	__constant__ float c_timeStep;
	__constant__ float c_sizeB;
	__constant__ float c_restDensity;
	__constant__ float c_restDensitySquInv;
	__constant__ float c_massF;
	__constant__ float c_pressGamma;
	__constant__ float c_jacobiOmega;
	__constant__ float c_jacobiWarmStart;
	__constant__ float c_viscosityF;
	__constant__ float c_viscosityB;
	__constant__ float c_xGravity;
	__constant__ float c_yGravity;
	__constant__ float c_zGravity;
	__constant__ float c_sizeSquEps;
	__constant__ float c_weightScale;

	// cubic-spline gradient coefficients per unit distance
	struct KernelConstants_CubicSpline
	{
		float size;
		float twoSize;
		float innerSlope;
		float innerOffset;
		float outerScale;
	};
	__constant__ KernelConstants_CubicSpline c_kernelConstants;

	void ReloadConstants(const Sim::SimParams& params, uint32_t N_F, uint32_t N_B, cudaStream_t stream)
	{
		const Sim::SolverConstants solverConstants = Sim::BuildSolverConstants(params.particleSizeF, params.restDensity);
		const float sizeInvF = solverConstants.sizeInvF;
		const float cubicSplineAlpha = solverConstants.cubicSplineAlpha;
		const float restDensitySquInv = solverConstants.restDensitySquInv;
		const float sizeSquEps = 0.01f * solverConstants.sizeSquF;
		const float weightScale = cubicSplineAlpha * sizeInvF * sizeInvF * sizeInvF;
		const KernelConstants_CubicSpline kernelConstants{
			1.0f / sizeInvF,
			2.0f / sizeInvF,
			9.0f * cubicSplineAlpha * sizeInvF * sizeInvF * sizeInvF,
			-12.0f * cubicSplineAlpha * sizeInvF * sizeInvF,
			-3.0f * cubicSplineAlpha * sizeInvF * sizeInvF * sizeInvF };

		// store values in constant memory on the GPU
		GPU::Check(cudaMemcpyToSymbol(c_N_F, &N_F, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_N_B, &N_B, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_timeStep, &params.timeStep, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_sizeB, &params.particleSizeB, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_restDensity, &params.restDensity, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_restDensitySquInv, &restDensitySquInv, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_massF, &params.massF, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_pressGamma, &params.pressGamma, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_jacobiOmega, &params.jacobiOmega, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_jacobiWarmStart, &params.jacobiWarmStart, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_viscosityF, &params.viscosityF, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_viscosityB, &params.viscosityB, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_xGravity, &params.gravity.x, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_yGravity, &params.gravity.y, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_zGravity, &params.gravity.z, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_sizeSquEps, &sizeSquEps, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_weightScale, &weightScale, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_kernelConstants, &kernelConstants, sizeof(KernelConstants_CubicSpline)));
	}

	// ------------------------------------------------------------------------------------------------
	// --- [ helpers ] --------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------

	// unscaled cubic-spline kernel weight in distance units
	// (zero at and beyond the kernel radius, sums take the alpha / size^3 factor once)
	__device__ __forceinline__
	float KernelWeightUnscaled_CubicSpline(const float distSqu)
	{
		// (clamped rsqrt, exact zero distance at the self pair)
		const float dist = distSqu * rsqrtf(fmaxf(distSqu, FLT_MIN));
		const float t1 = fmaxf(c_kernelConstants.size - dist, 0.0f);
		const float t2 = fmaxf(c_kernelConstants.twoSize - dist, 0.0f);

		return fmaf(-4.0f, (t1 * t1) * t1, (t2 * t2) * t2);
	}

	// cubic-spline kernel gradient factor without the min-dist guard, the gradient is factor * (dx, dy, dz)
	// (finite at zero distance, for sums that multiply it by a term vanishing there)
	__device__ __forceinline__
	float KernelGradientFactorNoMinDist_CubicSpline(const float distSqu)
	{
		// (clamped rsqrt, exact zero distance at the self pair)
		const float distInv = rsqrtf(fmaxf(distSqu, FLT_MIN));
		const float dist = distSqu * distInv;

		// factor q * (9q - 12) and cancel q / dist to avoid cancellation near zero
		const float innerFac = fmaf(c_kernelConstants.innerSlope, dist, c_kernelConstants.innerOffset);
		const float t2 = fmaxf(c_kernelConstants.twoSize - dist, 0.0f);
		const float outerFac = (t2 * t2) * (c_kernelConstants.outerScale * distInv);

		return (dist < c_kernelConstants.size) ? innerFac : outerFac;
	}

	__device__ __forceinline__
	uint32_t SELL32ReadNeighbor_Delta16x2(
		const uint64_t chunkBase,
		const uint32_t chunkBaseIdx,
		const uint32_t chunkBaseIdx2,
		const uint32_t lane,
		const uint32_t k,
		const uint16_t* __restrict__ nbrListDelta16)
	{
		// the top bit selects the second base index
		const uint32_t delta = static_cast<uint32_t>(
			nbrListDelta16[chunkBase + static_cast<uint64_t>(k) * N_SELL_WIDTH + lane]);
		return ((delta & 0x8000u) ? chunkBaseIdx2 : chunkBaseIdx) + (delta & 0x7fffu);
	}

	__device__ __forceinline__
	uint32_t SELL32ReadNeighbor_Abs32(
		const uint64_t chunkBase,
		const uint32_t lane,
		const uint32_t k,
		const uint32_t* __restrict__ nbrListAbs32)
	{
		return nbrListAbs32[chunkBase + static_cast<uint64_t>(k) * N_SELL_WIDTH + lane];
	}

	template <typename F>
	__device__ __forceinline__
	void ComputeForEachNeighbor_SELL32(
		const uint64_t chunkBase,
		const uint32_t chunkBaseIdx,
		const uint32_t chunkBaseIdx2,
		const uint8_t chunkFormat,
		const uint32_t lane,
		const uint32_t count,
		const uint16_t* __restrict__ nbrListDelta16,
		const uint32_t* __restrict__ nbrListAbs32,
		F&& func)
	{
		if (chunkFormat == static_cast<uint8_t>(NbrChunkFormat::Delta16x2))
		{
			for (uint32_t k = 0; k < count; k++)
			{
				const uint32_t j = SELL32ReadNeighbor_Delta16x2(
					chunkBase, chunkBaseIdx, chunkBaseIdx2, lane, k, nbrListDelta16);
				func(j);
			}
		}
		else
		{
			for (uint32_t k = 0; k < count; k++)
			{
				const uint32_t j = SELL32ReadNeighbor_Abs32(
					chunkBase, lane, k, nbrListAbs32);
				func(j);
			}
		}
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
		const uint32_t* __restrict__ nbrChunkBaseIdx2FF,
		const uint8_t* __restrict__ nbrChunkFormatFF,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrChunkFormatFR,
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

		float weightSumFF = 0.0f;

		const uint32_t chunk = i / N_SELL_WIDTH;
		const uint32_t lane = i % N_SELL_WIDTH;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint64_t baseFR = nbrChunkOffsetsFR[chunk];
		const uint32_t baseIdxFF = nbrChunkBaseIdxFF[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint32_t baseIdxFR = nbrChunkBaseIdxFR[chunk];
		const uint8_t formatFF = nbrChunkFormatFF[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint8_t formatFR = nbrChunkFormatFR[chunk];
		const uint32_t baseIdx2FF = (formatFF == static_cast<uint8_t>(NbrChunkFormat::Delta16x2))
			? nbrChunkBaseIdx2FF[chunk] : 0u;
		// FB and FR chunks never split, their second base index sits 0x8000 above the first
		const uint32_t baseIdx2FB = baseIdxFB + 0x8000u;
		const uint32_t baseIdx2FR = baseIdxFR + 0x8000u;
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);
		const uint32_t countFR = static_cast<uint32_t>(nbrCountFR[i]);

		// FF density contribution
		ComputeForEachNeighbor_SELL32(
			baseFF, baseIdxFF, baseIdx2FF, formatFF, lane, countFF,
			nbrListDelta16FF, nbrListAbs32FF,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosF[j];
				const float dy = yi - yPosF[j];
				const float dz = zi - zPosF[j];

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				weightSumFF += KernelWeightUnscaled_CubicSpline(distSqu);
			});

		// mass scales the whole FF sum
		float tmp_density = weightSumFF * c_massF;

		// FB density contribution
		ComputeForEachNeighbor_SELL32(
			baseFB, baseIdxFB, baseIdx2FB, formatFB, lane, countFB,
			nbrListDelta16FB, nbrListAbs32FB,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosB[j];
				const float dy = yi - yPosB[j];
				const float dz = zi - zPosB[j];

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				tmp_density = fmaf(psiB[j], KernelWeightUnscaled_CubicSpline(distSqu), tmp_density);
			});

		// FR density contribution
		ComputeForEachNeighbor_SELL32(
			baseFR, baseIdxFR, baseIdx2FR, formatFR, lane, countFR,
			nbrListDelta16FR, nbrListAbs32FR,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosR[j];
				const float dy = yi - yPosR[j];
				const float dz = zi - zPosR[j];

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				tmp_density = fmaf(psiR[j], KernelWeightUnscaled_CubicSpline(distSqu), tmp_density);
			});

		// alpha and 1 / size^3 scale the whole sum
		densityF[i] = tmp_density * c_weightScale;
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
		const uint32_t* __restrict__ nbrChunkBaseIdx2FF,
		const uint8_t* __restrict__ nbrChunkFormatFF,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrChunkFormatFR,
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

		const uint32_t chunk = i / N_SELL_WIDTH;
		const uint32_t lane = i % N_SELL_WIDTH;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint64_t baseFR = nbrChunkOffsetsFR[chunk];
		const uint32_t baseIdxFF = nbrChunkBaseIdxFF[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint32_t baseIdxFR = nbrChunkBaseIdxFR[chunk];
		const uint8_t formatFF = nbrChunkFormatFF[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint8_t formatFR = nbrChunkFormatFR[chunk];
		const uint32_t baseIdx2FF = (formatFF == static_cast<uint8_t>(NbrChunkFormat::Delta16x2))
			? nbrChunkBaseIdx2FF[chunk] : 0u;
		// FB and FR chunks never split, their second base index sits 0x8000 above the first
		const uint32_t baseIdx2FB = baseIdxFB + 0x8000u;
		const uint32_t baseIdx2FR = baseIdxFR + 0x8000u;
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);
		const uint32_t countFR = static_cast<uint32_t>(nbrCountFR[i]);

		// FF viscosity acceleration
		ComputeForEachNeighbor_SELL32(
			baseFF, baseIdxFF, baseIdx2FF, formatFF, lane, countFF,
			nbrListDelta16FF, nbrListAbs32FF,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosF[j];
				const float dy = yi - yPosF[j];
				const float dz = zi - zPosF[j];

				const float dvx = vxi - xVelF[j];
				const float dvy = vyi - yVelF[j];
				const float dvz = vzi - zVelF[j];

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				const float kgFac = KernelGradientFactorNoMinDist_CubicSpline(distSqu);

				const float dot = fmaf(dvx, dx, fmaf(dvy, dy, dvz * dz));
				const float accViscCoeff = dot
										 / (fmaxf(densityF[j], 1e-7f) * (distSqu + c_sizeSquEps))
										 * kgFac;

				tmp_xAccViscF = fmaf(accViscCoeff, dx, tmp_xAccViscF);
				tmp_yAccViscF = fmaf(accViscCoeff, dy, tmp_yAccViscF);
				tmp_zAccViscF = fmaf(accViscCoeff, dz, tmp_zAccViscF);
			});

		// temp accumulation variables
		float tmp_xAccViscB = 0.0f;
		float tmp_yAccViscB = 0.0f;
		float tmp_zAccViscB = 0.0f;

		// FB viscosity acceleration
		ComputeForEachNeighbor_SELL32(
			baseFB, baseIdxFB, baseIdx2FB, formatFB, lane, countFB,
			nbrListDelta16FB, nbrListAbs32FB,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosB[j];
				const float dy = yi - yPosB[j];
				const float dz = zi - zPosB[j];
				const float dvx = vxi;
				const float dvy = vyi;
				const float dvz = vzi;

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				const float kgFac = KernelGradientFactorNoMinDist_CubicSpline(distSqu);

				const float dot = fmaf(dvx, dx, fmaf(dvy, dy, dvz * dz));
				const float accViscCoeff = (volB[j] * dot) / (distSqu + c_sizeSquEps) * kgFac;

				tmp_xAccViscB = fmaf(accViscCoeff, dx, tmp_xAccViscB);
				tmp_yAccViscB = fmaf(accViscCoeff, dy, tmp_yAccViscB);
				tmp_zAccViscB = fmaf(accViscCoeff, dz, tmp_zAccViscB);
			});

		// FR viscosity acceleration
		ComputeForEachNeighbor_SELL32(
			baseFR, baseIdxFR, baseIdx2FR, formatFR, lane, countFR,
			nbrListDelta16FR, nbrListAbs32FR,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosR[j];
				const float dy = yi - yPosR[j];
				const float dz = zi - zPosR[j];
				const float dvx = vxi - xVelR[j];
				const float dvy = vyi - yVelR[j];
				const float dvz = vzi - zVelR[j];

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				const float kgFac = KernelGradientFactorNoMinDist_CubicSpline(distSqu);

				const float dot = fmaf(dvx, dx, fmaf(dvy, dy, dvz * dz));
				const float accViscCoeff = (volR[j] * dot) / (distSqu + c_sizeSquEps) * kgFac;

				tmp_xAccViscB = fmaf(accViscCoeff, dx, tmp_xAccViscB);
				tmp_yAccViscB = fmaf(accViscCoeff, dy, tmp_yAccViscB);
				tmp_zAccViscB = fmaf(accViscCoeff, dz, tmp_zAccViscB);
			});

		// finalize accelerations, mass scales the whole FF sum
		// (3D -> factor 2 * (dim + 2))
		tmp_xAccViscF *= c_massF * (2.0f * (3 + 2) * c_viscosityF);
		tmp_yAccViscF *= c_massF * (2.0f * (3 + 2) * c_viscosityF);
		tmp_zAccViscF *= c_massF * (2.0f * (3 + 2) * c_viscosityF);
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
		const uint32_t* __restrict__ nbrChunkBaseIdx2FF,
		const uint8_t* __restrict__ nbrChunkFormatFF,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrChunkFormatFR,
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
		const uint32_t chunk = i / N_SELL_WIDTH;
		const uint32_t lane = i % N_SELL_WIDTH;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint64_t baseFR = nbrChunkOffsetsFR[chunk];
		const uint32_t baseIdxFF = nbrChunkBaseIdxFF[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint32_t baseIdxFR = nbrChunkBaseIdxFR[chunk];
		const uint8_t formatFF = nbrChunkFormatFF[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint8_t formatFR = nbrChunkFormatFR[chunk];
		const uint32_t baseIdx2FF = (formatFF == static_cast<uint8_t>(NbrChunkFormat::Delta16x2))
			? nbrChunkBaseIdx2FF[chunk] : 0u;
		// FB and FR chunks never split, their second base index sits 0x8000 above the first
		const uint32_t baseIdx2FB = baseIdxFB + 0x8000u;
		const uint32_t baseIdx2FR = baseIdxFR + 0x8000u;
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);
		const uint32_t countFR = static_cast<uint32_t>(nbrCountFR[i]);

		// FF pressure source term and diagonal
		ComputeForEachNeighbor_SELL32(
			baseFF, baseIdxFF, baseIdx2FF, formatFF, lane, countFF,
			nbrListDelta16FF, nbrListAbs32FF,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosF[j];
				const float dy = yi - yPosF[j];
				const float dz = zi - zPosF[j];

				const float dvpx = vpxi - xVelPredF[j];
				const float dvpy = vpyi - yVelPredF[j];
				const float dvpz = vpzi - zVelPredF[j];

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				const float kgFac = KernelGradientFactorNoMinDist_CubicSpline(distSqu);

				const float sourceTerm = fmaf(dvpx, dx, fmaf(dvpy, dy, dvpz * dz));
				tmp_sourceTermF = fmaf(kgFac, sourceTerm, tmp_sourceTermF);

				xCTermF = fmaf(kgFac, dx, xCTermF);
				yCTermF = fmaf(kgFac, dy, yCTermF);
				zCTermF = fmaf(kgFac, dz, zCTermF);

				// (kgFac * distSqu first keeps the self pair exactly zero)
				gradSquSumF = fmaf(kgFac, kgFac * distSqu, gradSquSumF);
			});
		tmp_sourceTermF *= c_massF;

		// temp accumulation variables for boundary
		float tmp_sourceTermB = 0.0f;
		float xCTermB = 0.0f;
		float yCTermB = 0.0f;
		float zCTermB = 0.0f;

		// FB pressure source term and diagonal
		ComputeForEachNeighbor_SELL32(
			baseFB, baseIdxFB, baseIdx2FB, formatFB, lane, countFB,
			nbrListDelta16FB, nbrListAbs32FB,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosB[j];
				const float dy = yi - yPosB[j];
				const float dz = zi - zPosB[j];
				const float dvpx = vpxi;
				const float dvpy = vpyi;
				const float dvpz = vpzi;

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				const float kgFacWeighted = psiB[j] * KernelGradientFactorNoMinDist_CubicSpline(distSqu);

				const float sourceTerm = fmaf(dvpx, dx, fmaf(dvpy, dy, dvpz * dz));
				tmp_sourceTermB = fmaf(kgFacWeighted, sourceTerm, tmp_sourceTermB);

				xCTermB = fmaf(kgFacWeighted, dx, xCTermB);
				yCTermB = fmaf(kgFacWeighted, dy, yCTermB);
				zCTermB = fmaf(kgFacWeighted, dz, zCTermB);
			});

		// FR pressure source term and diagonal
		ComputeForEachNeighbor_SELL32(
			baseFR, baseIdxFR, baseIdx2FR, formatFR, lane, countFR,
			nbrListDelta16FR, nbrListAbs32FR,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosR[j];
				const float dy = yi - yPosR[j];
				const float dz = zi - zPosR[j];
				const float dvpx = vpxi - xVelR[j];
				const float dvpy = vpyi - yVelR[j];
				const float dvpz = vpzi - zVelR[j];

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				const float kgFacWeighted = psiR[j] * KernelGradientFactorNoMinDist_CubicSpline(distSqu);

				const float sourceTerm = fmaf(dvpx, dx, fmaf(dvpy, dy, dvpz * dz));
				tmp_sourceTermB = fmaf(kgFacWeighted, sourceTerm, tmp_sourceTermB);

				xCTermB = fmaf(kgFacWeighted, dx, xCTermB);
				yCTermB = fmaf(kgFacWeighted, dy, yCTermB);
				zCTermB = fmaf(kgFacWeighted, dz, zCTermB);
			});

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
		// (warm start from the scaled previous pressure if any, otherwise one Jacobi step from zero)
		if (!(fabsf(fin_diagElemF) <= 1e-7f))
		{
			const float pressurePrev = (c_jacobiWarmStart > 0.0f) ? pressureF[i] : 0.0f;
			tmp_pressureF = (pressurePrev > 0.0f)
				? c_jacobiWarmStart * pressurePrev
				: c_jacobiOmega * (fin_sourceTermF / fin_diagElemF);
		}

		// finalize pressure value by clamping to non-negative
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
		const uint32_t* __restrict__ nbrChunkBaseIdx2FF,
		const uint8_t* __restrict__ nbrChunkFormatFF,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrChunkFormatFR,
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
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];
		const float presi = pressureF[i];
		const float accCoeffB_i = 2.0f * c_pressGamma * c_restDensitySquInv * presi;

		// temp accumulation variables
		float tmp_xAccPressF = 0.0f;
		float tmp_yAccPressF = 0.0f;
		float tmp_zAccPressF = 0.0f;
		const uint32_t chunk = i / N_SELL_WIDTH;
		const uint32_t lane = i % N_SELL_WIDTH;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint64_t baseFR = nbrChunkOffsetsFR[chunk];
		const uint32_t baseIdxFF = nbrChunkBaseIdxFF[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint32_t baseIdxFR = nbrChunkBaseIdxFR[chunk];
		const uint8_t formatFF = nbrChunkFormatFF[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint8_t formatFR = nbrChunkFormatFR[chunk];
		const uint32_t baseIdx2FF = (formatFF == static_cast<uint8_t>(NbrChunkFormat::Delta16x2))
			? nbrChunkBaseIdx2FF[chunk] : 0u;
		// FB and FR chunks never split, their second base index sits 0x8000 above the first
		const uint32_t baseIdx2FB = baseIdxFB + 0x8000u;
		const uint32_t baseIdx2FR = baseIdxFR + 0x8000u;
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);
		const uint32_t countFR = static_cast<uint32_t>(nbrCountFR[i]);

		// FF pressure acceleration
		ComputeForEachNeighbor_SELL32(
			baseFF, baseIdxFF, baseIdx2FF, formatFF, lane, countFF,
			nbrListDelta16FF, nbrListAbs32FF,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosF[j];
				const float dy = yi - yPosF[j];
				const float dz = zi - zPosF[j];

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				const float kgFac = KernelGradientFactorNoMinDist_CubicSpline(distSqu);

				const float accCoeffF = (presi + pressureF[j]) * kgFac;

				tmp_xAccPressF = fmaf(accCoeffF, dx, tmp_xAccPressF);
				tmp_yAccPressF = fmaf(accCoeffF, dy, tmp_yAccPressF);
				tmp_zAccPressF = fmaf(accCoeffF, dz, tmp_zAccPressF);
			});

		// scale the FF sum by massF / restDensity^2, shared by all FF terms
		tmp_xAccPressF *= c_massF * c_restDensitySquInv;
		tmp_yAccPressF *= c_massF * c_restDensitySquInv;
		tmp_zAccPressF *= c_massF * c_restDensitySquInv;

		// FB pressure acceleration
		ComputeForEachNeighbor_SELL32(
			baseFB, baseIdxFB, baseIdx2FB, formatFB, lane, countFB,
			nbrListDelta16FB, nbrListAbs32FB,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosB[j];
				const float dy = yi - yPosB[j];
				const float dz = zi - zPosB[j];

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				const float kgFac = KernelGradientFactorNoMinDist_CubicSpline(distSqu);

				const float accCoeffB = accCoeffB_i * psiB[j] * kgFac;

				tmp_xAccPressF = fmaf(accCoeffB, dx, tmp_xAccPressF);
				tmp_yAccPressF = fmaf(accCoeffB, dy, tmp_yAccPressF);
				tmp_zAccPressF = fmaf(accCoeffB, dz, tmp_zAccPressF);
			});

		// FR pressure acceleration
		ComputeForEachNeighbor_SELL32(
			baseFR, baseIdxFR, baseIdx2FR, formatFR, lane, countFR,
			nbrListDelta16FR, nbrListAbs32FR,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosR[j];
				const float dy = yi - yPosR[j];
				const float dz = zi - zPosR[j];

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				const float kgFac = KernelGradientFactorNoMinDist_CubicSpline(distSqu);

				const float accCoeffB = accCoeffB_i * psiR[j] * kgFac;

				tmp_xAccPressF = fmaf(accCoeffB, dx, tmp_xAccPressF);
				tmp_yAccPressF = fmaf(accCoeffB, dy, tmp_yAccPressF);
				tmp_zAccPressF = fmaf(accCoeffB, dz, tmp_zAccPressF);
			});

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
		const uint32_t* __restrict__ nbrChunkBaseIdx2FF,
		const uint8_t* __restrict__ nbrChunkFormatFF,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrChunkFormatFR,
		const uint16_t* __restrict__ nbrListDelta16FF,
		const uint16_t* __restrict__ nbrListDelta16FB,
		const uint16_t* __restrict__ nbrListDelta16FR,
		const uint32_t* __restrict__ nbrListAbs32FF,
		const uint32_t* __restrict__ nbrListAbs32FB,
		const uint32_t* __restrict__ nbrListAbs32FR,
		// destinations
		float* __restrict__ pressureF,
		Sim::PressureSolveErrorSums* __restrict__ partialErrorSums,
		const bool computeErrors)
	{
		uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		const bool active = i < c_N_F;
		if (!active && !computeErrors) { return; }

		// out-of-range threads redo the last particle without writing, to reach the reduction barrier
		if (!active) { i = c_N_F - 1; }

		// particle i data
		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];
		const float axi = xAccPressF[i];
		const float ayi = yAccPressF[i];
		const float azi = zAccPressF[i];

		// temp accumulation variables
		float tmp_sysVecElem = 0.0f;
		const uint32_t chunk = i / N_SELL_WIDTH;
		const uint32_t lane = i % N_SELL_WIDTH;
		const uint64_t baseFF = nbrChunkOffsetsFF[chunk];
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint64_t baseFR = nbrChunkOffsetsFR[chunk];
		const uint32_t baseIdxFF = nbrChunkBaseIdxFF[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint32_t baseIdxFR = nbrChunkBaseIdxFR[chunk];
		const uint8_t formatFF = nbrChunkFormatFF[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint8_t formatFR = nbrChunkFormatFR[chunk];
		const uint32_t baseIdx2FF = (formatFF == static_cast<uint8_t>(NbrChunkFormat::Delta16x2))
			? nbrChunkBaseIdx2FF[chunk] : 0u;
		// FB and FR chunks never split, their second base index sits 0x8000 above the first
		const uint32_t baseIdx2FB = baseIdxFB + 0x8000u;
		const uint32_t baseIdx2FR = baseIdxFR + 0x8000u;
		const uint32_t countFF = static_cast<uint32_t>(nbrCountFF[i]);
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);
		const uint32_t countFR = static_cast<uint32_t>(nbrCountFR[i]);

		// FF pressure system vector element
		ComputeForEachNeighbor_SELL32(
			baseFF, baseIdxFF, baseIdx2FF, formatFF, lane, countFF,
			nbrListDelta16FF, nbrListAbs32FF,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosF[j];
				const float dy = yi - yPosF[j];
				const float dz = zi - zPosF[j];

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				const float kgFac = KernelGradientFactorNoMinDist_CubicSpline(distSqu);

				const float dotAccF = fmaf(axi - xAccPressF[j], dx,
									  fmaf(ayi - yAccPressF[j], dy,
										  (azi - zAccPressF[j]) * dz));
				tmp_sysVecElem = fmaf(kgFac, dotAccF, tmp_sysVecElem);
			});

		// mass scales the whole FF sum
		tmp_sysVecElem *= c_massF;

		// FB pressure system vector element
		ComputeForEachNeighbor_SELL32(
			baseFB, baseIdxFB, baseIdx2FB, formatFB, lane, countFB,
			nbrListDelta16FB, nbrListAbs32FB,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosB[j];
				const float dy = yi - yPosB[j];
				const float dz = zi - zPosB[j];

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				const float kgFacWeighted = psiB[j] * KernelGradientFactorNoMinDist_CubicSpline(distSqu);

				const float dotAccB = fmaf(axi, dx, fmaf(ayi, dy, azi * dz));
				tmp_sysVecElem = fmaf(kgFacWeighted, dotAccB, tmp_sysVecElem);
			});

		// FR pressure system vector element
		ComputeForEachNeighbor_SELL32(
			baseFR, baseIdxFR, baseIdx2FR, formatFR, lane, countFR,
			nbrListDelta16FR, nbrListAbs32FR,
			[&](const uint32_t j)
			{
				const float dx = xi - xPosR[j];
				const float dy = yi - yPosR[j];
				const float dz = zi - zPosR[j];

				const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
				const float kgFacWeighted = psiR[j] * KernelGradientFactorNoMinDist_CubicSpline(distSqu);

				const float dotAccR = fmaf(axi, dx, fmaf(ayi, dy, azi * dz));
				tmp_sysVecElem = fmaf(kgFacWeighted, dotAccR, tmp_sysVecElem);
			});

		float tmp_pressureF = 0.0f;

		// only update pressure if particle has neighbors
		if (!(fabsf(diagElemF[i]) <= 1e-7f))
		{
			tmp_pressureF = pressureF[i] + c_jacobiOmega
						  * ((sourceTermF[i] - (c_timeStep * c_timeStep * tmp_sysVecElem))
						  / diagElemF[i]);
		}

		// finalize pressure value by clamping to non-negative
		if (active) { pressureF[i] = fmaxf(tmp_pressureF, 0.0f); }

		// (uniform across the launch, so every thread of the block reaches the barrier)
		if (!computeErrors) { return; }

		// relative density error of particle i, zero for out-of-range threads and particles without neighbors
		float relErr = 0.0f;
		if (active && !(fabsf(diagElemF[i]) <= 1e-7f))
		{
			const float predDensityError = c_timeStep * c_timeStep * tmp_sysVecElem - sourceTermF[i];
			relErr = predDensityError / c_restDensity;
		}

		double sumRaw = static_cast<double>(relErr);
		double sumPos = fmax(0.0, sumRaw);
		double sumAbs = fabs(sumRaw);

		// reduce within each warp, then over the warps of the block in a fixed order
		for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
		{
			sumRaw += __shfl_down_sync(0xffffffffu, sumRaw, offset);
			sumPos += __shfl_down_sync(0xffffffffu, sumPos, offset);
			sumAbs += __shfl_down_sync(0xffffffffu, sumAbs, offset);
		}

		__shared__ double sRaw[N_DEFAULT_BLOCK_SIZE / 32u];
		__shared__ double sPos[N_DEFAULT_BLOCK_SIZE / 32u];
		__shared__ double sAbs[N_DEFAULT_BLOCK_SIZE / 32u];
		const uint32_t warpIdx = threadIdx.x / 32u;
		if (threadIdx.x % 32u == 0u)
		{
			sRaw[warpIdx] = sumRaw;
			sPos[warpIdx] = sumPos;
			sAbs[warpIdx] = sumAbs;
		}
		__syncthreads();

		if (threadIdx.x == 0u)
		{
			Sim::PressureSolveErrorSums out{};
			for (uint32_t w = 0; w < blockDim.x / 32u; w++)
			{
				out.raw += sRaw[w];
				out.pos += sPos[w];
				out.abs += sAbs[w];
			}

			partialErrorSums[blockIdx.x] = out;
		}
	}

	__global__
	void K_FinalizePressureSolveErrorSums(
		const Sim::PressureSolveErrorSums* __restrict__ partialErrorSums,
		const uint32_t N_partials,
		Sim::PressureSolveErrorSums* __restrict__ errorSums)
	{
		const uint32_t tid = threadIdx.x;

		// each thread sums a fixed strided subset of the block partials
		double sumRaw = 0.0;
		double sumPos = 0.0;
		double sumAbs = 0.0;
		for (uint32_t partialIdx = tid; partialIdx < N_partials; partialIdx += blockDim.x)
		{
			sumRaw += partialErrorSums[partialIdx].raw;
			sumPos += partialErrorSums[partialIdx].pos;
			sumAbs += partialErrorSums[partialIdx].abs;
		}

		// reduce within each warp, then over the warps of the block in a fixed order
		for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
		{
			sumRaw += __shfl_down_sync(0xffffffffu, sumRaw, offset);
			sumPos += __shfl_down_sync(0xffffffffu, sumPos, offset);
			sumAbs += __shfl_down_sync(0xffffffffu, sumAbs, offset);
		}

		__shared__ double sRaw[N_DEFAULT_BLOCK_SIZE / 32u];
		__shared__ double sPos[N_DEFAULT_BLOCK_SIZE / 32u];
		__shared__ double sAbs[N_DEFAULT_BLOCK_SIZE / 32u];
		const uint32_t warpIdx = tid / 32u;
		if (tid % 32u == 0u)
		{
			sRaw[warpIdx] = sumRaw;
			sPos[warpIdx] = sumPos;
			sAbs[warpIdx] = sumAbs;
		}
		__syncthreads();

		if (tid == 0u)
		{
			Sim::PressureSolveErrorSums out{};
			for (uint32_t w = 0; w < blockDim.x / 32u; w++)
			{
				out.raw += sRaw[w];
				out.pos += sPos[w];
				out.abs += sAbs[w];
			}

			*errorSums = out;
		}
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

} // namespace GPU

// -------------------------------------------------------------------------------------------------
// --- [ kernel launchers ] ------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

namespace GPU
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
		const uint32_t* nbrChunkBaseIdx2FF,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrChunkFormatFR,
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
		const uint32_t N_BLOCK_SIZE = N_DEFAULT_BLOCK_SIZE;
		const uint32_t N_GRID_SIZE = (N_F + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_Density<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR, psiB, psiR,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrChunkOffsetsFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkBaseIdx2FF,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR,
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
		const uint32_t* nbrChunkBaseIdx2FF,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrChunkFormatFR,
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
		const uint32_t N_BLOCK_SIZE = N_DEFAULT_BLOCK_SIZE;
		const uint32_t N_GRID_SIZE = (N_F + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_NonPressureAcceleration<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR,
			xVelF, yVelF, zVelF, xVelR, yVelR, zVelR, densityF, volB, volR,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrChunkOffsetsFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkBaseIdx2FF,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR,
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
		const uint32_t* nbrChunkBaseIdx2FF,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrChunkFormatFR,
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
		const uint32_t N_BLOCK_SIZE = N_DEFAULT_BLOCK_SIZE;
		const uint32_t N_GRID_SIZE = (N_F + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_SourceTermAndDiagonalElementAndPressureJacobiInit<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR,
			xVelPredF, yVelPredF, zVelPredF, xVelR, yVelR, zVelR, densityF, psiB, psiR,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrChunkOffsetsFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkBaseIdx2FF,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR,
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
		const uint32_t* nbrChunkBaseIdx2FF,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrChunkFormatFR,
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
		const uint32_t N_BLOCK_SIZE = N_DEFAULT_BLOCK_SIZE;
		const uint32_t N_GRID_SIZE = (N_F + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_PressureAcceleration<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR, pressureF, psiB, psiR,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrChunkOffsetsFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkBaseIdx2FF,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR,
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
		const uint32_t* nbrChunkBaseIdx2FF,
		const uint8_t* nbrChunkFormatFF,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrChunkFormatFR,
		const uint16_t* nbrListDelta16FF,
		const uint16_t* nbrListDelta16FB,
		const uint16_t* nbrListDelta16FR,
		const uint32_t* nbrListAbs32FF,
		const uint32_t* nbrListAbs32FB,
		const uint32_t* nbrListAbs32FR,
		const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		float* pressureF,
		Sim::PressureSolveErrorSums* partialErrorSums,
		Sim::PressureSolveErrorSums* errorSums,
		const bool computeErrors)
	{
		const uint32_t N_BLOCK_SIZE = N_DEFAULT_BLOCK_SIZE;
		const uint32_t N_GRID_SIZE = PressureSolveErrorPartialCount(N_F);

		K_PressureRelaxedJacobiSolver<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosB, yPosB, zPosB, xPosR, yPosR, zPosR,
			xAccPressF, yAccPressF, zAccPressF, sourceTermF, diagElemF, psiB, psiR,
			nbrCountFF, nbrCountFB, nbrCountFR,
			nbrChunkOffsetsFF, nbrChunkOffsetsFB, nbrChunkOffsetsFR,
			nbrChunkBaseIdxFF, nbrChunkBaseIdxFB, nbrChunkBaseIdxFR,
			nbrChunkBaseIdx2FF,
			nbrChunkFormatFF, nbrChunkFormatFB, nbrChunkFormatFR,
			nbrListDelta16FF, nbrListDelta16FB, nbrListDelta16FR,
			nbrListAbs32FF, nbrListAbs32FB, nbrListAbs32FR, pressureF, partialErrorSums, computeErrors);
		GPU::CheckLastKernel("K_PressureRelaxedJacobiSolver()");

		if (!computeErrors) { return; }

		// reduce the block partials on device and leave only the final sums for the host
		K_FinalizePressureSolveErrorSums<<<1, N_BLOCK_SIZE, 0, stream>>>(
			partialErrorSums, N_GRID_SIZE, errorSums);
		GPU::CheckLastKernel("K_FinalizePressureSolveErrorSums()");
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
		const uint32_t N_BLOCK_SIZE = N_DEFAULT_BLOCK_SIZE;
		const uint32_t N_GRID_SIZE = (N_F + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		K_Integration<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream>>>(
			xVelPredF, yVelPredF, zVelPredF, xAccPressF, yAccPressF, zAccPressF,
			xPosF, yPosF, zPosF, xVelF, yVelF, zVelF);
		GPU::CheckLastKernel("K_Integration()");
	}

} // namespace GPU
