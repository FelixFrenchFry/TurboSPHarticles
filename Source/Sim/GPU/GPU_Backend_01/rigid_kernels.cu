// rigid_kernels.cu

// external
#include <cuda_runtime.h>
#include <cmath>

// internal
#include "common.cuh"
#include "rigid_kernels.cuh"
#include "Source/Sim/GPU/gpu_utils.h"



namespace GPU01
{
	static constexpr uint32_t N_SellWidth = 32u;
	static constexpr uint32_t InvalidRigidBodyId = 0xffffffffu;
	static constexpr float RigidOrthonormalEps = 1e-12f;

	__device__ __forceinline__
	float Dot3(
		const float ax, const float ay, const float az,
		const float bx, const float by, const float bz)
	{
		return fmaf(ax, bx, fmaf(ay, by, az * bz));
	}

	__device__ __forceinline__
	bool Normalize3(float& x, float& y, float& z)
	{
		const float lenSqu = Dot3(x, y, z, x, y, z);
		if (lenSqu <= RigidOrthonormalEps) { return false; }

		const float lenInv = rsqrtf(lenSqu);
		x *= lenInv;
		y *= lenInv;
		z *= lenInv;

		return true;
	}

	__device__ __forceinline__
	void ReOrthonormalizeBody(
		const uint32_t bodyIdx,
		float* __restrict__ a00R, float* __restrict__ a01R, float* __restrict__ a02R,
		float* __restrict__ a10R, float* __restrict__ a11R, float* __restrict__ a12R,
		float* __restrict__ a20R, float* __restrict__ a21R, float* __restrict__ a22R)
	{
		// use Gram-Schmidt algo on orientation columns
		float b1x = a00R[bodyIdx];
		float b1y = a10R[bodyIdx];
		float b1z = a20R[bodyIdx];
		if (!Normalize3(b1x, b1y, b1z))
		{
			a00R[bodyIdx] = 1.0f; a01R[bodyIdx] = 0.0f; a02R[bodyIdx] = 0.0f;
			a10R[bodyIdx] = 0.0f; a11R[bodyIdx] = 1.0f; a12R[bodyIdx] = 0.0f;
			a20R[bodyIdx] = 0.0f; a21R[bodyIdx] = 0.0f; a22R[bodyIdx] = 1.0f;
			return;
		}

		float b2x = a01R[bodyIdx];
		float b2y = a11R[bodyIdx];
		float b2z = a21R[bodyIdx];
		const float b1DotA2 = Dot3(b1x, b1y, b1z, b2x, b2y, b2z);
		b2x -= b1DotA2 * b1x;
		b2y -= b1DotA2 * b1y;
		b2z -= b1DotA2 * b1z;
		if (!Normalize3(b2x, b2y, b2z))
		{
			a00R[bodyIdx] = 1.0f; a01R[bodyIdx] = 0.0f; a02R[bodyIdx] = 0.0f;
			a10R[bodyIdx] = 0.0f; a11R[bodyIdx] = 1.0f; a12R[bodyIdx] = 0.0f;
			a20R[bodyIdx] = 0.0f; a21R[bodyIdx] = 0.0f; a22R[bodyIdx] = 1.0f;
			return;
		}

		float b3x = a02R[bodyIdx];
		float b3y = a12R[bodyIdx];
		float b3z = a22R[bodyIdx];
		const float b1DotA3 = Dot3(b1x, b1y, b1z, b3x, b3y, b3z);
		const float b2DotA3 = Dot3(b2x, b2y, b2z, b3x, b3y, b3z);
		b3x -= b1DotA3 * b1x + b2DotA3 * b2x;
		b3y -= b1DotA3 * b1y + b2DotA3 * b2y;
		b3z -= b1DotA3 * b1z + b2DotA3 * b2z;
		if (!Normalize3(b3x, b3y, b3z))
		{
			a00R[bodyIdx] = 1.0f; a01R[bodyIdx] = 0.0f; a02R[bodyIdx] = 0.0f;
			a10R[bodyIdx] = 0.0f; a11R[bodyIdx] = 1.0f; a12R[bodyIdx] = 0.0f;
			a20R[bodyIdx] = 0.0f; a21R[bodyIdx] = 0.0f; a22R[bodyIdx] = 1.0f;
			return;
		}

		a00R[bodyIdx] = b1x; a01R[bodyIdx] = b2x; a02R[bodyIdx] = b3x;
		a10R[bodyIdx] = b1y; a11R[bodyIdx] = b2y; a12R[bodyIdx] = b3y;
		a20R[bodyIdx] = b1z; a21R[bodyIdx] = b2z; a22R[bodyIdx] = b3z;
	}

	__device__ __forceinline__
	void ComputeWorldInertiaInv(
		const uint32_t bodyIdx,
		const float* __restrict__ a00R, const float* __restrict__ a01R, const float* __restrict__ a02R,
		const float* __restrict__ a10R, const float* __restrict__ a11R, const float* __restrict__ a12R,
		const float* __restrict__ a20R, const float* __restrict__ a21R, const float* __restrict__ a22R,
		const float* __restrict__ iBodyInv00R, const float* __restrict__ iBodyInv01R, const float* __restrict__ iBodyInv02R,
		const float* __restrict__ iBodyInv10R, const float* __restrict__ iBodyInv11R, const float* __restrict__ iBodyInv12R,
		const float* __restrict__ iBodyInv20R, const float* __restrict__ iBodyInv21R, const float* __restrict__ iBodyInv22R,
		float& i00, float& i01, float& i02,
		float& i10, float& i11, float& i12,
		float& i20, float& i21, float& i22)
	{
		const float a00 = a00R[bodyIdx], a01 = a01R[bodyIdx], a02 = a02R[bodyIdx];
		const float a10 = a10R[bodyIdx], a11 = a11R[bodyIdx], a12 = a12R[bodyIdx];
		const float a20 = a20R[bodyIdx], a21 = a21R[bodyIdx], a22 = a22R[bodyIdx];

		const float b00 = iBodyInv00R[bodyIdx], b01 = iBodyInv01R[bodyIdx], b02 = iBodyInv02R[bodyIdx];
		const float b10 = iBodyInv10R[bodyIdx], b11 = iBodyInv11R[bodyIdx], b12 = iBodyInv12R[bodyIdx];
		const float b20 = iBodyInv20R[bodyIdx], b21 = iBodyInv21R[bodyIdx], b22 = iBodyInv22R[bodyIdx];

		const float t00 = a00 * b00 + a01 * b10 + a02 * b20;
		const float t01 = a00 * b01 + a01 * b11 + a02 * b21;
		const float t02 = a00 * b02 + a01 * b12 + a02 * b22;
		const float t10 = a10 * b00 + a11 * b10 + a12 * b20;
		const float t11 = a10 * b01 + a11 * b11 + a12 * b21;
		const float t12 = a10 * b02 + a11 * b12 + a12 * b22;
		const float t20 = a20 * b00 + a21 * b10 + a22 * b20;
		const float t21 = a20 * b01 + a21 * b11 + a22 * b21;
		const float t22 = a20 * b02 + a21 * b12 + a22 * b22;

		i00 = t00 * a00 + t01 * a01 + t02 * a02;
		i01 = t00 * a10 + t01 * a11 + t02 * a12;
		i02 = t00 * a20 + t01 * a21 + t02 * a22;
		i10 = t10 * a00 + t11 * a01 + t12 * a02;
		i11 = t10 * a10 + t11 * a11 + t12 * a12;
		i12 = t10 * a20 + t11 * a21 + t12 * a22;
		i20 = t20 * a00 + t21 * a01 + t22 * a02;
		i21 = t20 * a10 + t21 * a11 + t22 * a12;
		i22 = t20 * a20 + t21 * a21 + t22 * a22;
	}

	__device__ __forceinline__
	void AddForceToRigidBoundaryParticle(
		const uint32_t rigidBoundaryParticleIdx,
		const float xForce, const float yForce, const float zForce,
		float* __restrict__ xForceRigidBoundaryParticle,
		float* __restrict__ yForceRigidBoundaryParticle,
		float* __restrict__ zForceRigidBoundaryParticle)
	{
		atomicAdd(xForceRigidBoundaryParticle + rigidBoundaryParticleIdx, xForce);
		atomicAdd(yForceRigidBoundaryParticle + rigidBoundaryParticleIdx, yForce);
		atomicAdd(zForceRigidBoundaryParticle + rigidBoundaryParticleIdx, zForce);
	}

	__device__ __forceinline__
	void KernelGradient_CubicSpline(
		const float dx, const float dy, const float dz,
		const float particleSizeInvF,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		float& gx, float& gy, float& gz)
	{
		const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
		if (distSqu >= kernelRadiusSqu || distSqu < 1e-7f)
		{
			gx = 0.0f;
			gy = 0.0f;
			gz = 0.0f;
			return;
		}

		const float dist = sqrtf(distSqu);
		const float distNorm = dist * particleSizeInvF;
		const float t1 = fmaxf(1.0f - distNorm, 0.0f);
		const float t2 = fmaxf(2.0f - distNorm, 0.0f);
		const float commonFac = cubicSplineAlpha
			* (-3.0f * t2 * t2 + 12.0f * t1 * t1)
			* (particleSizeInvF / dist);

		gx = commonFac * dx;
		gy = commonFac * dy;
		gz = commonFac * dz;
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
	void K_UpdateRigidBoundaryParticles(
		const uint32_t* __restrict__ bodyIdB,
		const uint32_t* __restrict__ rigidBoundaryParticleIdxB,
		const uint32_t* __restrict__ rigidBoundaryParticleBeginR,
		const uint32_t* __restrict__ rigidBoundaryParticleCountR,
		const float* __restrict__ xCMR, const float* __restrict__ yCMR, const float* __restrict__ zCMR,
		const float* __restrict__ xVelR, const float* __restrict__ yVelR, const float* __restrict__ zVelR,
		const float* __restrict__ xOmegaR, const float* __restrict__ yOmegaR, const float* __restrict__ zOmegaR,
		const float* __restrict__ a00R, const float* __restrict__ a01R, const float* __restrict__ a02R,
		const float* __restrict__ a10R, const float* __restrict__ a11R, const float* __restrict__ a12R,
		const float* __restrict__ a20R, const float* __restrict__ a21R, const float* __restrict__ a22R,
		const float* __restrict__ xLocalRigidBoundaryParticle,
		const float* __restrict__ yLocalRigidBoundaryParticle,
		const float* __restrict__ zLocalRigidBoundaryParticle,
		const uint32_t N_B, const uint32_t N_rigidBodies, const uint32_t N_rigidBoundaryParticles,
		// destinations
		float* __restrict__ xPosB, float* __restrict__ yPosB, float* __restrict__ zPosB,
		float* __restrict__ xVelB, float* __restrict__ yVelB, float* __restrict__ zVelB)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_B) { return; }

		const uint32_t bodyIdx = bodyIdB[i];
		if (bodyIdx == InvalidRigidBodyId)
		{
			xVelB[i] = 0.0f;
			yVelB[i] = 0.0f;
			zVelB[i] = 0.0f;
			return;
		}

		const uint32_t rigidBoundaryParticleIdx = rigidBoundaryParticleIdxB[i];
		if (bodyIdx >= N_rigidBodies || rigidBoundaryParticleIdx >= N_rigidBoundaryParticles)
		{
			xVelB[i] = 0.0f;
			yVelB[i] = 0.0f;
			zVelB[i] = 0.0f;
			return;
		}

		const uint32_t rigidBoundaryParticleBegin = rigidBoundaryParticleBeginR[bodyIdx];
		const uint32_t rigidBoundaryParticleCount = rigidBoundaryParticleCountR[bodyIdx];
		if (rigidBoundaryParticleBegin > N_rigidBoundaryParticles || rigidBoundaryParticleCount > N_rigidBoundaryParticles - rigidBoundaryParticleBegin ||
			rigidBoundaryParticleIdx < rigidBoundaryParticleBegin || rigidBoundaryParticleIdx >= rigidBoundaryParticleBegin + rigidBoundaryParticleCount)
		{
			xVelB[i] = 0.0f;
			yVelB[i] = 0.0f;
			zVelB[i] = 0.0f;
			return;
		}

		const float lx = xLocalRigidBoundaryParticle[rigidBoundaryParticleIdx];
		const float ly = yLocalRigidBoundaryParticle[rigidBoundaryParticleIdx];
		const float lz = zLocalRigidBoundaryParticle[rigidBoundaryParticleIdx];

		const float rx = fmaf(a00R[bodyIdx], lx, fmaf(a01R[bodyIdx], ly, a02R[bodyIdx] * lz));
		const float ry = fmaf(a10R[bodyIdx], lx, fmaf(a11R[bodyIdx], ly, a12R[bodyIdx] * lz));
		const float rz = fmaf(a20R[bodyIdx], lx, fmaf(a21R[bodyIdx], ly, a22R[bodyIdx] * lz));

		xPosB[i] = xCMR[bodyIdx] + rx;
		yPosB[i] = yCMR[bodyIdx] + ry;
		zPosB[i] = zCMR[bodyIdx] + rz;

		const float wx = xOmegaR[bodyIdx];
		const float wy = yOmegaR[bodyIdx];
		const float wz = zOmegaR[bodyIdx];

		xVelB[i] = xVelR[bodyIdx] + wy * rz - wz * ry;
		yVelB[i] = yVelR[bodyIdx] + wz * rx - wx * rz;
		zVelB[i] = zVelR[bodyIdx] + wx * ry - wy * rx;
	}

	__global__
	void K_ClearRigidForcesAndAddGravity(
		const float* __restrict__ massR,
		const float* __restrict__ massInvR,
		const float xGravity, const float yGravity, const float zGravity,
		const uint32_t N_rigidBodies,
		// destinations
		float* __restrict__ xForceR, float* __restrict__ yForceR, float* __restrict__ zForceR,
		float* __restrict__ xTorqueR, float* __restrict__ yTorqueR, float* __restrict__ zTorqueR)
	{
		const uint32_t bodyIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (bodyIdx >= N_rigidBodies) { return; }

		xForceR[bodyIdx] = 0.0f;
		yForceR[bodyIdx] = 0.0f;
		zForceR[bodyIdx] = 0.0f;
		xTorqueR[bodyIdx] = 0.0f;
		yTorqueR[bodyIdx] = 0.0f;
		zTorqueR[bodyIdx] = 0.0f;

		if (massInvR[bodyIdx] <= 0.0f || massR[bodyIdx] <= 0.0f) { return; }

		const float mass = massR[bodyIdx];
		xForceR[bodyIdx] = mass * xGravity;
		yForceR[bodyIdx] = mass * yGravity;
		zForceR[bodyIdx] = mass * zGravity;
	}

	__global__
	void K_ClearRigidBoundaryParticleForces(
		const uint32_t N_rigidBoundaryParticles,
		// destinations
		float* __restrict__ xForceRigidBoundaryParticle,
		float* __restrict__ yForceRigidBoundaryParticle,
		float* __restrict__ zForceRigidBoundaryParticle)
	{
		const uint32_t rigidBoundaryParticleIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (rigidBoundaryParticleIdx >= N_rigidBoundaryParticles) { return; }

		xForceRigidBoundaryParticle[rigidBoundaryParticleIdx] = 0.0f;
		yForceRigidBoundaryParticle[rigidBoundaryParticleIdx] = 0.0f;
		zForceRigidBoundaryParticle[rigidBoundaryParticleIdx] = 0.0f;
	}

	__global__
	void K_AccumulateRigidBodyNonPressureForces(
		const float* __restrict__ xPosF, const float* __restrict__ yPosF, const float* __restrict__ zPosF,
		const float* __restrict__ xPosB, const float* __restrict__ yPosB, const float* __restrict__ zPosB,
		const uint32_t* __restrict__ bodyIdB,
		const uint32_t* __restrict__ rigidBoundaryParticleIdxB,
		const float* __restrict__ volB,
		const float* __restrict__ xVelF, const float* __restrict__ yVelF, const float* __restrict__ zVelF,
		const float* __restrict__ xVelB, const float* __restrict__ yVelB, const float* __restrict__ zVelB,
		const uint16_t* __restrict__ nbrCountFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint32_t* __restrict__ nbrChunkBaseIdxFB,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrListDelta8FB,
		const uint16_t* __restrict__ nbrListDelta16FB,
		const uint32_t* __restrict__ nbrListAbs32FB,
		const float particleSizeInvF,
		const float particleSizeSquF,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		const float massF,
		const float viscosityB,
		const uint32_t N_F, const uint32_t N_rigidBoundaryParticles,
		// destinations
		float* __restrict__ xForceRigidBoundaryParticle,
		float* __restrict__ yForceRigidBoundaryParticle,
		float* __restrict__ zForceRigidBoundaryParticle)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_F) { return; }

		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];
		const float vxi = xVelF[i];
		const float vyi = yVelF[i];
		const float vzi = zVelF[i];

		const uint32_t chunk = i / N_SellWidth;
		const uint32_t lane = i % N_SellWidth;
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);
		const float particleSizeSquEps = 0.01f * particleSizeSquF;
		const float viscosityFacB = 10.0f * viscosityB;

		for (uint32_t k = 0; k < countFB; k++)
		{
			const uint32_t j = Sell32ReadNeighbor(
				baseFB, baseIdxFB, formatFB, lane, k,
				nbrListDelta8FB, nbrListDelta16FB, nbrListAbs32FB);

			const uint32_t bodyIdx = bodyIdB[j];
			if (bodyIdx == InvalidRigidBodyId) { continue; }

			const uint32_t rigidBoundaryParticleIdx = rigidBoundaryParticleIdxB[j];
			#ifndef NDEBUG
			if (rigidBoundaryParticleIdx >= N_rigidBoundaryParticles) { continue; }
			#endif

			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];
			const float dvx = vxi - xVelB[j];
			const float dvy = vyi - yVelB[j];
			const float dvz = vzi - zVelB[j];

			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(
				dx, dy, dz,
				particleSizeInvF, kernelRadiusSqu, cubicSplineAlpha,
				xkg, ykg, zkg);

			const float dot = fmaf(dvx, dx, fmaf(dvy, dy, dvz * dz));
			const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
			const float accCoeff = (volB[j] * dot) / (distSqu + particleSizeSquEps);
			const float forceCoeff = -massF * viscosityFacB * accCoeff;

			// opposite of the boundary viscosity/friction force applied to fluid particle i
			AddForceToRigidBoundaryParticle(
				rigidBoundaryParticleIdx,
				forceCoeff * xkg,
				forceCoeff * ykg,
				forceCoeff * zkg,
				xForceRigidBoundaryParticle,
				yForceRigidBoundaryParticle,
				zForceRigidBoundaryParticle);
		}
	}

	__global__
	void K_AccumulateRigidBodyPressureForces(
		const float* __restrict__ xPosF, const float* __restrict__ yPosF, const float* __restrict__ zPosF,
		const float* __restrict__ xPosB, const float* __restrict__ yPosB, const float* __restrict__ zPosB,
		const uint32_t* __restrict__ bodyIdB,
		const uint32_t* __restrict__ rigidBoundaryParticleIdxB,
		const float* __restrict__ psiB,
		const float* __restrict__ pressureF,
		const uint16_t* __restrict__ nbrCountFB,
		const uint64_t* __restrict__ nbrChunkOffsetsFB,
		const uint32_t* __restrict__ nbrChunkBaseIdxFB,
		const uint8_t* __restrict__ nbrChunkFormatFB,
		const uint8_t* __restrict__ nbrListDelta8FB,
		const uint16_t* __restrict__ nbrListDelta16FB,
		const uint32_t* __restrict__ nbrListAbs32FB,
		const float particleSizeInvF,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		const float restDensity,
		const float massF,
		const float pressGamma,
		const uint32_t N_F, const uint32_t N_rigidBoundaryParticles,
		// destinations
		float* __restrict__ xForceRigidBoundaryParticle,
		float* __restrict__ yForceRigidBoundaryParticle,
		float* __restrict__ zForceRigidBoundaryParticle)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_F) { return; }

		const float xi = xPosF[i];
		const float yi = yPosF[i];
		const float zi = zPosF[i];
		const float presi = pressureF[i];

		const uint32_t chunk = i / N_SellWidth;
		const uint32_t lane = i % N_SellWidth;
		const uint64_t baseFB = nbrChunkOffsetsFB[chunk];
		const uint32_t baseIdxFB = nbrChunkBaseIdxFB[chunk];
		const uint8_t formatFB = nbrChunkFormatFB[chunk];
		const uint32_t countFB = static_cast<uint32_t>(nbrCountFB[i]);
		const float densiInvSqu = 1.0f / (restDensity * restDensity);
		const float accCoeffB_outer = 2.0f * pressGamma * densiInvSqu;

		for (uint32_t k = 0; k < countFB; k++)
		{
			const uint32_t j = Sell32ReadNeighbor(
				baseFB, baseIdxFB, formatFB, lane, k,
				nbrListDelta8FB, nbrListDelta16FB, nbrListAbs32FB);

			const uint32_t bodyIdx = bodyIdB[j];
			if (bodyIdx == InvalidRigidBodyId) { continue; }

			const uint32_t rigidBoundaryParticleIdx = rigidBoundaryParticleIdxB[j];
			#ifndef NDEBUG
			if (rigidBoundaryParticleIdx >= N_rigidBoundaryParticles) { continue; }
			#endif

			const float dx = xi - xPosB[j];
			const float dy = yi - yPosB[j];
			const float dz = zi - zPosB[j];

			float xkg = 0.0f;
			float ykg = 0.0f;
			float zkg = 0.0f;
			KernelGradient_CubicSpline(
				dx, dy, dz,
				particleSizeInvF, kernelRadiusSqu, cubicSplineAlpha,
				xkg, ykg, zkg);

			const float accCoeffB = accCoeffB_outer * psiB[j] * presi;

			// opposite of the boundary pressure force applied to fluid particle i
			AddForceToRigidBoundaryParticle(
				rigidBoundaryParticleIdx,
				massF * accCoeffB * xkg,
				massF * accCoeffB * ykg,
				massF * accCoeffB * zkg,
				xForceRigidBoundaryParticle,
				yForceRigidBoundaryParticle,
				zForceRigidBoundaryParticle);
		}
	}

	__global__
	void K_ReduceRigidBoundaryParticleForces(
		const uint32_t* __restrict__ rigidBoundaryParticleBeginR,
		const uint32_t* __restrict__ rigidBoundaryParticleCountR,
		const float* __restrict__ xLocalRigidBoundaryParticle,
		const float* __restrict__ yLocalRigidBoundaryParticle,
		const float* __restrict__ zLocalRigidBoundaryParticle,
		const float* __restrict__ a00R, const float* __restrict__ a01R, const float* __restrict__ a02R,
		const float* __restrict__ a10R, const float* __restrict__ a11R, const float* __restrict__ a12R,
		const float* __restrict__ a20R, const float* __restrict__ a21R, const float* __restrict__ a22R,
		const float* __restrict__ xForceRigidBoundaryParticle,
		const float* __restrict__ yForceRigidBoundaryParticle,
		const float* __restrict__ zForceRigidBoundaryParticle,
		const uint32_t N_rigidBodies,
		const uint32_t N_rigidBoundaryParticles,
		// destinations
		float* __restrict__ xForceR, float* __restrict__ yForceR, float* __restrict__ zForceR,
		float* __restrict__ xTorqueR, float* __restrict__ yTorqueR, float* __restrict__ zTorqueR)
	{
		const uint32_t bodyIdx = blockIdx.x;
		const uint32_t tid = threadIdx.x;
		if (bodyIdx >= N_rigidBodies) { return; }

		const uint32_t begin = rigidBoundaryParticleBeginR[bodyIdx];
		const uint32_t count = rigidBoundaryParticleCountR[bodyIdx];
		#ifndef NDEBUG
		if (begin > N_rigidBoundaryParticles || count > N_rigidBoundaryParticles - begin) { return; }
		#endif
		const uint32_t end = begin + count;

		float sumXForce = 0.0f;
		float sumYForce = 0.0f;
		float sumZForce = 0.0f;
		float sumXTorque = 0.0f;
		float sumYTorque = 0.0f;
		float sumZTorque = 0.0f;

		for (uint32_t rigidBoundaryParticleIdx = begin + tid;
			 rigidBoundaryParticleIdx < end;
			 rigidBoundaryParticleIdx += blockDim.x)
		{
			const float xForce = xForceRigidBoundaryParticle[rigidBoundaryParticleIdx];
			const float yForce = yForceRigidBoundaryParticle[rigidBoundaryParticleIdx];
			const float zForce = zForceRigidBoundaryParticle[rigidBoundaryParticleIdx];

			const float lx = xLocalRigidBoundaryParticle[rigidBoundaryParticleIdx];
			const float ly = yLocalRigidBoundaryParticle[rigidBoundaryParticleIdx];
			const float lz = zLocalRigidBoundaryParticle[rigidBoundaryParticleIdx];

			const float rx = fmaf(a00R[bodyIdx], lx, fmaf(a01R[bodyIdx], ly, a02R[bodyIdx] * lz));
			const float ry = fmaf(a10R[bodyIdx], lx, fmaf(a11R[bodyIdx], ly, a12R[bodyIdx] * lz));
			const float rz = fmaf(a20R[bodyIdx], lx, fmaf(a21R[bodyIdx], ly, a22R[bodyIdx] * lz));

			sumXForce += xForce;
			sumYForce += yForce;
			sumZForce += zForce;
			sumXTorque += ry * zForce - rz * yForce;
			sumYTorque += rz * xForce - rx * zForce;
			sumZTorque += rx * yForce - ry * xForce;
		}

		extern __shared__ float shared[];
		float* sXForce = shared;
		float* sYForce = sXForce + blockDim.x;
		float* sZForce = sYForce + blockDim.x;
		float* sXTorque = sZForce + blockDim.x;
		float* sYTorque = sXTorque + blockDim.x;
		float* sZTorque = sYTorque + blockDim.x;

		sXForce[tid] = sumXForce;
		sYForce[tid] = sumYForce;
		sZForce[tid] = sumZForce;
		sXTorque[tid] = sumXTorque;
		sYTorque[tid] = sumYTorque;
		sZTorque[tid] = sumZTorque;
		__syncthreads();

		for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
		{
			if (tid < stride)
			{
				sXForce[tid] += sXForce[tid + stride];
				sYForce[tid] += sYForce[tid + stride];
				sZForce[tid] += sZForce[tid + stride];
				sXTorque[tid] += sXTorque[tid + stride];
				sYTorque[tid] += sYTorque[tid + stride];
				sZTorque[tid] += sZTorque[tid + stride];
			}
			__syncthreads();
		}

		if (tid == 0u)
		{
			xForceR[bodyIdx] += sXForce[0];
			yForceR[bodyIdx] += sYForce[0];
			zForceR[bodyIdx] += sZForce[0];
			xTorqueR[bodyIdx] += sXTorque[0];
			yTorqueR[bodyIdx] += sYTorque[0];
			zTorqueR[bodyIdx] += sZTorque[0];
		}
	}

	__global__
	void K_IntegrateRigidBodies(
		const float timeStep,
		const uint32_t N_rigidBodies,
		// destinations
		float* __restrict__ xCMR, float* __restrict__ yCMR, float* __restrict__ zCMR,
		float* __restrict__ xVelR, float* __restrict__ yVelR, float* __restrict__ zVelR,
		float* __restrict__ xOmegaR, float* __restrict__ yOmegaR, float* __restrict__ zOmegaR,
		float* __restrict__ xAngMomR, float* __restrict__ yAngMomR, float* __restrict__ zAngMomR,
		const float* __restrict__ xForceR, const float* __restrict__ yForceR, const float* __restrict__ zForceR,
		const float* __restrict__ xTorqueR, const float* __restrict__ yTorqueR, const float* __restrict__ zTorqueR,
		float* __restrict__ a00R, float* __restrict__ a01R, float* __restrict__ a02R,
		float* __restrict__ a10R, float* __restrict__ a11R, float* __restrict__ a12R,
		float* __restrict__ a20R, float* __restrict__ a21R, float* __restrict__ a22R,
		const float* __restrict__ massInvR,
		const float* __restrict__ iBodyInv00R, const float* __restrict__ iBodyInv01R, const float* __restrict__ iBodyInv02R,
		const float* __restrict__ iBodyInv10R, const float* __restrict__ iBodyInv11R, const float* __restrict__ iBodyInv12R,
		const float* __restrict__ iBodyInv20R, const float* __restrict__ iBodyInv21R, const float* __restrict__ iBodyInv22R)
	{
		const uint32_t bodyIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (bodyIdx >= N_rigidBodies) { return; }

		const float massInv = massInvR[bodyIdx];

		xVelR[bodyIdx] += timeStep * xForceR[bodyIdx] * massInv;
		yVelR[bodyIdx] += timeStep * yForceR[bodyIdx] * massInv;
		zVelR[bodyIdx] += timeStep * zForceR[bodyIdx] * massInv;

		xCMR[bodyIdx] += timeStep * xVelR[bodyIdx];
		yCMR[bodyIdx] += timeStep * yVelR[bodyIdx];
		zCMR[bodyIdx] += timeStep * zVelR[bodyIdx];

		xAngMomR[bodyIdx] += timeStep * xTorqueR[bodyIdx];
		yAngMomR[bodyIdx] += timeStep * yTorqueR[bodyIdx];
		zAngMomR[bodyIdx] += timeStep * zTorqueR[bodyIdx];

		float i00, i01, i02;
		float i10, i11, i12;
		float i20, i21, i22;
		ComputeWorldInertiaInv(
			bodyIdx,
			a00R, a01R, a02R,
			a10R, a11R, a12R,
			a20R, a21R, a22R,
			iBodyInv00R, iBodyInv01R, iBodyInv02R,
			iBodyInv10R, iBodyInv11R, iBodyInv12R,
			iBodyInv20R, iBodyInv21R, iBodyInv22R,
			i00, i01, i02,
			i10, i11, i12,
			i20, i21, i22);

		const float lx = xAngMomR[bodyIdx];
		const float ly = yAngMomR[bodyIdx];
		const float lz = zAngMomR[bodyIdx];
		const float wx = fmaf(i00, lx, fmaf(i01, ly, i02 * lz));
		const float wy = fmaf(i10, lx, fmaf(i11, ly, i12 * lz));
		const float wz = fmaf(i20, lx, fmaf(i21, ly, i22 * lz));

		xOmegaR[bodyIdx] = wx;
		yOmegaR[bodyIdx] = wy;
		zOmegaR[bodyIdx] = wz;

		const float a00 = a00R[bodyIdx], a01 = a01R[bodyIdx], a02 = a02R[bodyIdx];
		const float a10 = a10R[bodyIdx], a11 = a11R[bodyIdx], a12 = a12R[bodyIdx];
		const float a20 = a20R[bodyIdx], a21 = a21R[bodyIdx], a22 = a22R[bodyIdx];

		a00R[bodyIdx] += timeStep * (-wz * a10 + wy * a20);
		a01R[bodyIdx] += timeStep * (-wz * a11 + wy * a21);
		a02R[bodyIdx] += timeStep * (-wz * a12 + wy * a22);
		a10R[bodyIdx] += timeStep * ( wz * a00 - wx * a20);
		a11R[bodyIdx] += timeStep * ( wz * a01 - wx * a21);
		a12R[bodyIdx] += timeStep * ( wz * a02 - wx * a22);
		a20R[bodyIdx] += timeStep * (-wy * a00 + wx * a10);
		a21R[bodyIdx] += timeStep * (-wy * a01 + wx * a11);
		a22R[bodyIdx] += timeStep * (-wy * a02 + wx * a12);

		ReOrthonormalizeBody(
			bodyIdx,
			a00R, a01R, a02R,
			a10R, a11R, a12R,
			a20R, a21R, a22R);
	}

} // namespace GPU01

// -------------------------------------------------------------------------------------------------
// --- [ kernel launchers ] ------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

namespace GPU01
{
	void Launch_UpdateRigidBoundaryParticles(
		const uint32_t* bodyIdB,
		const uint32_t* rigidBoundaryParticleIdxB,
		const uint32_t* rigidBoundaryParticleBeginR,
		const uint32_t* rigidBoundaryParticleCountR,
		const float* xCMR, const float* yCMR, const float* zCMR,
		const float* xVelR, const float* yVelR, const float* zVelR,
		const float* xOmegaR, const float* yOmegaR, const float* zOmegaR,
		const float* a00R, const float* a01R, const float* a02R,
		const float* a10R, const float* a11R, const float* a12R,
		const float* a20R, const float* a21R, const float* a22R,
		const float* xLocalRigidBoundaryParticle, const float* yLocalRigidBoundaryParticle, const float* zLocalRigidBoundaryParticle,
		const uint32_t N_B, const uint32_t N_rigidBodies, const uint32_t N_rigidBoundaryParticles,
		cudaStream_t stream,
		// destinations
		float* xPosB, float* yPosB, float* zPosB,
		float* xVelB, float* yVelB, float* zVelB)
	{
		if (N_B == 0) { return; }

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_B + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_UpdateRigidBoundaryParticles<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			bodyIdB, rigidBoundaryParticleIdxB,
			rigidBoundaryParticleBeginR, rigidBoundaryParticleCountR,
			xCMR, yCMR, zCMR,
			xVelR, yVelR, zVelR,
			xOmegaR, yOmegaR, zOmegaR,
			a00R, a01R, a02R,
			a10R, a11R, a12R,
			a20R, a21R, a22R,
			xLocalRigidBoundaryParticle, yLocalRigidBoundaryParticle, zLocalRigidBoundaryParticle,
			N_B, N_rigidBodies, N_rigidBoundaryParticles,
			xPosB, yPosB, zPosB,
			xVelB, yVelB, zVelB);

		GPU::CheckLastKernel("K_UpdateRigidBoundaryParticles()");
	}

	void Launch_ClearRigidForcesAndAddGravity(
		const float* massR,
		const float* massInvR,
		const float xGravity, const float yGravity, const float zGravity,
		const uint32_t N_rigidBodies,
		cudaStream_t stream,
		// destinations
		float* xForceR, float* yForceR, float* zForceR,
		float* xTorqueR, float* yTorqueR, float* zTorqueR)
	{
		if (N_rigidBodies == 0) { return; }

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_rigidBodies + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_ClearRigidForcesAndAddGravity<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			massR, massInvR,
			xGravity, yGravity, zGravity,
			N_rigidBodies,
			xForceR, yForceR, zForceR,
			xTorqueR, yTorqueR, zTorqueR);

		GPU::CheckLastKernel("K_ClearRigidForcesAndAddGravity()");
	}

	void Launch_ClearRigidBoundaryParticleForces(
		const uint32_t N_rigidBoundaryParticles,
		cudaStream_t stream,
		// destinations
		float* xForceRigidBoundaryParticle,
		float* yForceRigidBoundaryParticle,
		float* zForceRigidBoundaryParticle)
	{
		if (N_rigidBoundaryParticles == 0) { return; }

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_rigidBoundaryParticles + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_ClearRigidBoundaryParticleForces<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			N_rigidBoundaryParticles,
			xForceRigidBoundaryParticle,
			yForceRigidBoundaryParticle,
			zForceRigidBoundaryParticle);

		GPU::CheckLastKernel("K_ClearRigidBoundaryParticleForces()");
	}

	void Launch_AccumulateRigidBodyNonPressureForces(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* bodyIdB,
		const uint32_t* rigidBoundaryParticleIdxB,
		const float* volB,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* xVelB, const float* yVelB, const float* zVelB,
		const uint16_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrListDelta8FB,
		const uint16_t* nbrListDelta16FB,
		const uint32_t* nbrListAbs32FB,
		const float particleSizeInvF,
		const float particleSizeSquF,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		const float massF,
		const float viscosityB,
		const uint32_t N_F, const uint32_t N_rigidBoundaryParticles,
		cudaStream_t stream,
		// destinations
		float* xForceRigidBoundaryParticle,
		float* yForceRigidBoundaryParticle,
		float* zForceRigidBoundaryParticle)
	{
		if (N_F == 0 || N_rigidBoundaryParticles == 0) { return; }

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_AccumulateRigidBodyNonPressureForces<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			bodyIdB,
			rigidBoundaryParticleIdxB,
			volB,
			xVelF, yVelF, zVelF,
			xVelB, yVelB, zVelB,
			nbrCountFB,
			nbrChunkOffsetsFB,
			nbrChunkBaseIdxFB,
			nbrChunkFormatFB,
			nbrListDelta8FB,
			nbrListDelta16FB,
			nbrListAbs32FB,
			particleSizeInvF,
			particleSizeSquF,
			kernelRadiusSqu,
			cubicSplineAlpha,
			massF,
			viscosityB,
			N_F, N_rigidBoundaryParticles,
			xForceRigidBoundaryParticle,
			yForceRigidBoundaryParticle,
			zForceRigidBoundaryParticle);

		GPU::CheckLastKernel("K_AccumulateRigidBodyNonPressureForces()");
	}

	void Launch_AccumulateRigidBodyPressureForces(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* bodyIdB,
		const uint32_t* rigidBoundaryParticleIdxB,
		const float* psiB,
		const float* pressureF,
		const uint16_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrListDelta8FB,
		const uint16_t* nbrListDelta16FB,
		const uint32_t* nbrListAbs32FB,
		const float particleSizeInvF,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		const float restDensity,
		const float massF,
		const float pressGamma,
		const uint32_t N_F, const uint32_t N_rigidBoundaryParticles,
		cudaStream_t stream,
		// destinations
		float* xForceRigidBoundaryParticle,
		float* yForceRigidBoundaryParticle,
		float* zForceRigidBoundaryParticle)
	{
		if (N_F == 0 || N_rigidBoundaryParticles == 0) { return; }

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_F + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_AccumulateRigidBodyPressureForces<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			bodyIdB,
			rigidBoundaryParticleIdxB,
			psiB,
			pressureF,
			nbrCountFB,
			nbrChunkOffsetsFB,
			nbrChunkBaseIdxFB,
			nbrChunkFormatFB,
			nbrListDelta8FB,
			nbrListDelta16FB,
			nbrListAbs32FB,
			particleSizeInvF,
			kernelRadiusSqu,
			cubicSplineAlpha,
			restDensity,
			massF,
			pressGamma,
			N_F, N_rigidBoundaryParticles,
			xForceRigidBoundaryParticle,
			yForceRigidBoundaryParticle,
			zForceRigidBoundaryParticle);

		GPU::CheckLastKernel("K_AccumulateRigidBodyPressureForces()");
	}

	void Launch_ReduceRigidBoundaryParticleForces(
		const uint32_t* rigidBoundaryParticleBeginR,
		const uint32_t* rigidBoundaryParticleCountR,
		const float* xLocalRigidBoundaryParticle,
		const float* yLocalRigidBoundaryParticle,
		const float* zLocalRigidBoundaryParticle,
		const float* a00R, const float* a01R, const float* a02R,
		const float* a10R, const float* a11R, const float* a12R,
		const float* a20R, const float* a21R, const float* a22R,
		const float* xForceRigidBoundaryParticle,
		const float* yForceRigidBoundaryParticle,
		const float* zForceRigidBoundaryParticle,
		const uint32_t N_rigidBodies,
		const uint32_t N_rigidBoundaryParticles,
		cudaStream_t stream,
		// destinations
		float* xForceR, float* yForceR, float* zForceR,
		float* xTorqueR, float* yTorqueR, float* zTorqueR)
	{
		if (N_rigidBodies == 0 || N_rigidBoundaryParticles == 0) { return; }

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = N_rigidBodies;
		const uint32_t sharedBytes = 6u * N_BLOCKSIZE * static_cast<uint32_t>(sizeof(float));

		K_ReduceRigidBoundaryParticleForces<<<N_GRIDSIZE, N_BLOCKSIZE, sharedBytes, stream>>>(
			rigidBoundaryParticleBeginR,
			rigidBoundaryParticleCountR,
			xLocalRigidBoundaryParticle,
			yLocalRigidBoundaryParticle,
			zLocalRigidBoundaryParticle,
			a00R, a01R, a02R,
			a10R, a11R, a12R,
			a20R, a21R, a22R,
			xForceRigidBoundaryParticle,
			yForceRigidBoundaryParticle,
			zForceRigidBoundaryParticle,
			N_rigidBodies,
			N_rigidBoundaryParticles,
			xForceR, yForceR, zForceR,
			xTorqueR, yTorqueR, zTorqueR);

		GPU::CheckLastKernel("K_ReduceRigidBoundaryParticleForces()");
	}

	void Launch_IntegrateRigidBodies(
		const float timeStep,
		const uint32_t N_rigidBodies,
		cudaStream_t stream,
		// destinations
		float* xCMR, float* yCMR, float* zCMR,
		float* xVelR, float* yVelR, float* zVelR,
		float* xOmegaR, float* yOmegaR, float* zOmegaR,
		float* xAngMomR, float* yAngMomR, float* zAngMomR,
		const float* xForceR, const float* yForceR, const float* zForceR,
		const float* xTorqueR, const float* yTorqueR, const float* zTorqueR,
		float* a00R, float* a01R, float* a02R,
		float* a10R, float* a11R, float* a12R,
		float* a20R, float* a21R, float* a22R,
		const float* massInvR,
		const float* iBodyInv00R, const float* iBodyInv01R, const float* iBodyInv02R,
		const float* iBodyInv10R, const float* iBodyInv11R, const float* iBodyInv12R,
		const float* iBodyInv20R, const float* iBodyInv21R, const float* iBodyInv22R)
	{
		if (N_rigidBodies == 0) { return; }

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_rigidBodies + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_IntegrateRigidBodies<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			timeStep,
			N_rigidBodies,
			xCMR, yCMR, zCMR,
			xVelR, yVelR, zVelR,
			xOmegaR, yOmegaR, zOmegaR,
			xAngMomR, yAngMomR, zAngMomR,
			xForceR, yForceR, zForceR,
			xTorqueR, yTorqueR, zTorqueR,
			a00R, a01R, a02R,
			a10R, a11R, a12R,
			a20R, a21R, a22R,
			massInvR,
			iBodyInv00R, iBodyInv01R, iBodyInv02R,
			iBodyInv10R, iBodyInv11R, iBodyInv12R,
			iBodyInv20R, iBodyInv21R, iBodyInv22R);

		GPU::CheckLastKernel("K_IntegrateRigidBodies()");
	}

} // namespace GPU01
