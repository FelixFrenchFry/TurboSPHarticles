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
		float* __restrict__ a00RB, float* __restrict__ a01RB, float* __restrict__ a02RB,
		float* __restrict__ a10RB, float* __restrict__ a11RB, float* __restrict__ a12RB,
		float* __restrict__ a20RB, float* __restrict__ a21RB, float* __restrict__ a22RB)
	{
		// use Gram-Schmidt algo on orientation columns
		float b1x = a00RB[bodyIdx];
		float b1y = a10RB[bodyIdx];
		float b1z = a20RB[bodyIdx];
		if (!Normalize3(b1x, b1y, b1z))
		{
			a00RB[bodyIdx] = 1.0f; a01RB[bodyIdx] = 0.0f; a02RB[bodyIdx] = 0.0f;
			a10RB[bodyIdx] = 0.0f; a11RB[bodyIdx] = 1.0f; a12RB[bodyIdx] = 0.0f;
			a20RB[bodyIdx] = 0.0f; a21RB[bodyIdx] = 0.0f; a22RB[bodyIdx] = 1.0f;
			return;
		}

		float b2x = a01RB[bodyIdx];
		float b2y = a11RB[bodyIdx];
		float b2z = a21RB[bodyIdx];
		const float b1DotA2 = Dot3(b1x, b1y, b1z, b2x, b2y, b2z);
		b2x -= b1DotA2 * b1x;
		b2y -= b1DotA2 * b1y;
		b2z -= b1DotA2 * b1z;
		if (!Normalize3(b2x, b2y, b2z))
		{
			a00RB[bodyIdx] = 1.0f; a01RB[bodyIdx] = 0.0f; a02RB[bodyIdx] = 0.0f;
			a10RB[bodyIdx] = 0.0f; a11RB[bodyIdx] = 1.0f; a12RB[bodyIdx] = 0.0f;
			a20RB[bodyIdx] = 0.0f; a21RB[bodyIdx] = 0.0f; a22RB[bodyIdx] = 1.0f;
			return;
		}

		float b3x = a02RB[bodyIdx];
		float b3y = a12RB[bodyIdx];
		float b3z = a22RB[bodyIdx];
		const float b1DotA3 = Dot3(b1x, b1y, b1z, b3x, b3y, b3z);
		const float b2DotA3 = Dot3(b2x, b2y, b2z, b3x, b3y, b3z);
		b3x -= b1DotA3 * b1x + b2DotA3 * b2x;
		b3y -= b1DotA3 * b1y + b2DotA3 * b2y;
		b3z -= b1DotA3 * b1z + b2DotA3 * b2z;
		if (!Normalize3(b3x, b3y, b3z))
		{
			a00RB[bodyIdx] = 1.0f; a01RB[bodyIdx] = 0.0f; a02RB[bodyIdx] = 0.0f;
			a10RB[bodyIdx] = 0.0f; a11RB[bodyIdx] = 1.0f; a12RB[bodyIdx] = 0.0f;
			a20RB[bodyIdx] = 0.0f; a21RB[bodyIdx] = 0.0f; a22RB[bodyIdx] = 1.0f;
			return;
		}

		a00RB[bodyIdx] = b1x; a01RB[bodyIdx] = b2x; a02RB[bodyIdx] = b3x;
		a10RB[bodyIdx] = b1y; a11RB[bodyIdx] = b2y; a12RB[bodyIdx] = b3y;
		a20RB[bodyIdx] = b1z; a21RB[bodyIdx] = b2z; a22RB[bodyIdx] = b3z;
	}

	__device__ __forceinline__
	void ComputeWorldInertiaInv(
		const uint32_t bodyIdx,
		const float* __restrict__ a00RB, const float* __restrict__ a01RB, const float* __restrict__ a02RB,
		const float* __restrict__ a10RB, const float* __restrict__ a11RB, const float* __restrict__ a12RB,
		const float* __restrict__ a20RB, const float* __restrict__ a21RB, const float* __restrict__ a22RB,
		const float* __restrict__ iBodyInv00RB, const float* __restrict__ iBodyInv01RB, const float* __restrict__ iBodyInv02RB,
		const float* __restrict__ iBodyInv10RB, const float* __restrict__ iBodyInv11RB, const float* __restrict__ iBodyInv12RB,
		const float* __restrict__ iBodyInv20RB, const float* __restrict__ iBodyInv21RB, const float* __restrict__ iBodyInv22RB,
		float& i00, float& i01, float& i02,
		float& i10, float& i11, float& i12,
		float& i20, float& i21, float& i22)
	{
		const float a00 = a00RB[bodyIdx], a01 = a01RB[bodyIdx], a02 = a02RB[bodyIdx];
		const float a10 = a10RB[bodyIdx], a11 = a11RB[bodyIdx], a12 = a12RB[bodyIdx];
		const float a20 = a20RB[bodyIdx], a21 = a21RB[bodyIdx], a22 = a22RB[bodyIdx];

		const float b00 = iBodyInv00RB[bodyIdx], b01 = iBodyInv01RB[bodyIdx], b02 = iBodyInv02RB[bodyIdx];
		const float b10 = iBodyInv10RB[bodyIdx], b11 = iBodyInv11RB[bodyIdx], b12 = iBodyInv12RB[bodyIdx];
		const float b20 = iBodyInv20RB[bodyIdx], b21 = iBodyInv21RB[bodyIdx], b22 = iBodyInv22RB[bodyIdx];

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

	// ------------------------------------------------------------------------------------------------
	// --- [ kernels launched by host ] ---------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------

	__global__
	void K_UpdateRigidParticles(
		const uint32_t* __restrict__ bodyIdR,
		const float* __restrict__ xRelPosR, const float* __restrict__ yRelPosR, const float* __restrict__ zRelPosR,
		const float* __restrict__ xCMRB, const float* __restrict__ yCMRB, const float* __restrict__ zCMRB,
		const float* __restrict__ xVelRB, const float* __restrict__ yVelRB, const float* __restrict__ zVelRB,
		const float* __restrict__ xOmegaRB, const float* __restrict__ yOmegaRB, const float* __restrict__ zOmegaRB,
		const float* __restrict__ a00RB, const float* __restrict__ a01RB, const float* __restrict__ a02RB,
		const float* __restrict__ a10RB, const float* __restrict__ a11RB, const float* __restrict__ a12RB,
		const float* __restrict__ a20RB, const float* __restrict__ a21RB, const float* __restrict__ a22RB,
		const uint32_t N_R, const uint32_t N_RB,
		// destinations
		float* __restrict__ xPosR, float* __restrict__ yPosR, float* __restrict__ zPosR,
		float* __restrict__ xVelR, float* __restrict__ yVelR, float* __restrict__ zVelR)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_R) { return; }

		const uint32_t bodyIdx = bodyIdR[i];
		if (bodyIdx >= N_RB) { return; }

		const float lx = xRelPosR[i];
		const float ly = yRelPosR[i];
		const float lz = zRelPosR[i];
		const float rx = fmaf(a00RB[bodyIdx], lx, fmaf(a01RB[bodyIdx], ly, a02RB[bodyIdx] * lz));
		const float ry = fmaf(a10RB[bodyIdx], lx, fmaf(a11RB[bodyIdx], ly, a12RB[bodyIdx] * lz));
		const float rz = fmaf(a20RB[bodyIdx], lx, fmaf(a21RB[bodyIdx], ly, a22RB[bodyIdx] * lz));

		xPosR[i] = xCMRB[bodyIdx] + rx;
		yPosR[i] = yCMRB[bodyIdx] + ry;
		zPosR[i] = zCMRB[bodyIdx] + rz;

		const float wx = xOmegaRB[bodyIdx];
		const float wy = yOmegaRB[bodyIdx];
		const float wz = zOmegaRB[bodyIdx];
		xVelR[i] = xVelRB[bodyIdx] + wy * rz - wz * ry;
		yVelR[i] = yVelRB[bodyIdx] + wz * rx - wx * rz;
		zVelR[i] = zVelRB[bodyIdx] + wx * ry - wy * rx;
	}

	__global__
	void K_ClearRigidForcesAndAddGravity(
		const float* __restrict__ massRB,
		const float* __restrict__ massInvRB,
		const float xGravity, const float yGravity, const float zGravity,
		const uint32_t N_RB,
		// destinations
		float* __restrict__ xForceRB, float* __restrict__ yForceRB, float* __restrict__ zForceRB,
		float* __restrict__ xTorqueRB, float* __restrict__ yTorqueRB, float* __restrict__ zTorqueRB)
	{
		const uint32_t bodyIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (bodyIdx >= N_RB) { return; }

		xForceRB[bodyIdx] = 0.0f;
		yForceRB[bodyIdx] = 0.0f;
		zForceRB[bodyIdx] = 0.0f;
		xTorqueRB[bodyIdx] = 0.0f;
		yTorqueRB[bodyIdx] = 0.0f;
		zTorqueRB[bodyIdx] = 0.0f;

		if (massInvRB[bodyIdx] <= 0.0f || massRB[bodyIdx] <= 0.0f) { return; }

		const float mass = massRB[bodyIdx];
		xForceRB[bodyIdx] = mass * xGravity;
		yForceRB[bodyIdx] = mass * yGravity;
		zForceRB[bodyIdx] = mass * zGravity;
	}

	__global__
	void K_AccumulateRigidParticleNonPressureForces(
		const float* __restrict__ xPosF, const float* __restrict__ yPosF, const float* __restrict__ zPosF,
		const float* __restrict__ xPosR, const float* __restrict__ yPosR, const float* __restrict__ zPosR,
		const float* __restrict__ xVelF, const float* __restrict__ yVelF, const float* __restrict__ zVelF,
		const float* __restrict__ xVelR, const float* __restrict__ yVelR, const float* __restrict__ zVelR,
		const float* __restrict__ volR,
		const uint16_t* __restrict__ nbrCountRF,
		const uint64_t* __restrict__ nbrOffsetsRF,
		const uint32_t* __restrict__ nbrListRF,
		const float particleSizeInvF,
		const float particleSizeSquF,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		const float massF,
		const float viscosityB,
		const uint32_t N_R,
		// destinations
		float* __restrict__ xForceR, float* __restrict__ yForceR, float* __restrict__ zForceR)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_R) { return; }

		float xForce = 0.0f;
		float yForce = 0.0f;
		float zForce = 0.0f;
		const uint64_t begin = nbrOffsetsRF[i];
		const uint32_t count = nbrCountRF[i];
		for (uint32_t k = 0; k < count; k++)
		{
			const uint32_t j = nbrListRF[begin + k];

			const float dx = xPosF[j] - xPosR[i];
			const float dy = yPosF[j] - yPosR[i];
			const float dz = zPosF[j] - zPosR[i];

			const float dvx = xVelF[j] - xVelR[i];
			const float dvy = yVelF[j] - yVelR[i];
			const float dvz = zVelF[j] - zVelR[i];

			float xkg = 0.0f, ykg = 0.0f, zkg = 0.0f;
			KernelGradient_CubicSpline(
				dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
				cubicSplineAlpha, xkg, ykg, zkg);

			const float dot = fmaf(dvx, dx, fmaf(dvy, dy, dvz * dz));
			const float distSqu = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
			const float accCoeff = volR[i] * dot / (distSqu + 0.01f * particleSizeSquF);
			const float forceCoeff = -massF * 10.0f * viscosityB * accCoeff;

			xForce = fmaf(forceCoeff, xkg, xForce);
			yForce = fmaf(forceCoeff, ykg, yForce);
			zForce = fmaf(forceCoeff, zkg, zForce);
		}
		xForceR[i] = xForce;
		yForceR[i] = yForce;
		zForceR[i] = zForce;
	}

	__global__
	void K_AccumulateRigidParticlePressureForces(
		const float* __restrict__ xPosF, const float* __restrict__ yPosF, const float* __restrict__ zPosF,
		const float* __restrict__ xPosR, const float* __restrict__ yPosR, const float* __restrict__ zPosR,
		const float* __restrict__ pressureF,
		const float* __restrict__ psiR,
		const uint16_t* __restrict__ nbrCountRF,
		const uint64_t* __restrict__ nbrOffsetsRF,
		const uint32_t* __restrict__ nbrListRF,
		const float particleSizeInvF,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		const float restDensity,
		const float massF,
		const float pressGamma,
		const uint32_t N_R,
		// destinations
		float* __restrict__ xForceR, float* __restrict__ yForceR, float* __restrict__ zForceR)
	{
		const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N_R) { return; }

		float xForce = 0.0f;
		float yForce = 0.0f;
		float zForce = 0.0f;
		const float densityInvSqu = 1.0f / (restDensity * restDensity);
		const uint64_t begin = nbrOffsetsRF[i];
		const uint32_t count = nbrCountRF[i];
		for (uint32_t k = 0; k < count; k++)
		{
			const uint32_t j = nbrListRF[begin + k];

			const float dx = xPosF[j] - xPosR[i];
			const float dy = yPosF[j] - yPosR[i];
			const float dz = zPosF[j] - zPosR[i];

			float xkg = 0.0f, ykg = 0.0f, zkg = 0.0f;
			KernelGradient_CubicSpline(
				dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
				cubicSplineAlpha, xkg, ykg, zkg);

			const float forceCoeff = massF * pressGamma * 2.0f
				* psiR[i] * pressureF[j] * densityInvSqu;

			xForce = fmaf(forceCoeff, xkg, xForce);
			yForce = fmaf(forceCoeff, ykg, yForce);
			zForce = fmaf(forceCoeff, zkg, zForce);
		}
		xForceR[i] += xForce;
		yForceR[i] += yForce;
		zForceR[i] += zForce;
	}

	__global__
	void K_ReduceRigidParticleForces(
		const uint32_t* __restrict__ rigidParticleBeginRB,
		const uint32_t* __restrict__ rigidParticleCountRB,
		const float* __restrict__ xRelPosR,
		const float* __restrict__ yRelPosR,
		const float* __restrict__ zRelPosR,
		const float* __restrict__ xForceR,
		const float* __restrict__ yForceR,
		const float* __restrict__ zForceR,
		const float* __restrict__ a00RB, const float* __restrict__ a01RB, const float* __restrict__ a02RB,
		const float* __restrict__ a10RB, const float* __restrict__ a11RB, const float* __restrict__ a12RB,
		const float* __restrict__ a20RB, const float* __restrict__ a21RB, const float* __restrict__ a22RB,
		const uint32_t N_RB,
		const uint32_t N_R,
		// destinations
		float* __restrict__ xForceRB, float* __restrict__ yForceRB, float* __restrict__ zForceRB,
		float* __restrict__ xTorqueRB, float* __restrict__ yTorqueRB, float* __restrict__ zTorqueRB)
	{
		const uint32_t bodyIdx = blockIdx.x;
		const uint32_t tid = threadIdx.x;
		if (bodyIdx >= N_RB) { return; }

		const uint32_t begin = rigidParticleBeginRB[bodyIdx];
		const uint32_t count = rigidParticleCountRB[bodyIdx];
		#ifndef NDEBUG
		if (begin > N_R || count > N_R - begin) { return; }
		#endif
		const uint32_t end = begin + count;

		float sumXForce = 0.0f;
		float sumYForce = 0.0f;
		float sumZForce = 0.0f;
		float sumXTorque = 0.0f;
		float sumYTorque = 0.0f;
		float sumZTorque = 0.0f;

		for (uint32_t i = begin + tid;
			 i < end;
			 i += blockDim.x)
		{
			const float lx = xRelPosR[i];
			const float ly = yRelPosR[i];
			const float lz = zRelPosR[i];

			const float xForce = xForceR[i];
			const float yForce = yForceR[i];
			const float zForce = zForceR[i];

			const float rx = fmaf(a00RB[bodyIdx], lx, fmaf(a01RB[bodyIdx], ly, a02RB[bodyIdx] * lz));
			const float ry = fmaf(a10RB[bodyIdx], lx, fmaf(a11RB[bodyIdx], ly, a12RB[bodyIdx] * lz));
			const float rz = fmaf(a20RB[bodyIdx], lx, fmaf(a21RB[bodyIdx], ly, a22RB[bodyIdx] * lz));

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
			xForceRB[bodyIdx] += sXForce[0];
			yForceRB[bodyIdx] += sYForce[0];
			zForceRB[bodyIdx] += sZForce[0];
			xTorqueRB[bodyIdx] += sXTorque[0];
			yTorqueRB[bodyIdx] += sYTorque[0];
			zTorqueRB[bodyIdx] += sZTorque[0];
		}
	}

	__global__
	void K_IntegrateRigidBodies(
		const float timeStep,
		const uint32_t N_RB,
		// destinations
		float* __restrict__ xCMRB, float* __restrict__ yCMRB, float* __restrict__ zCMRB,
		float* __restrict__ xVelRB, float* __restrict__ yVelRB, float* __restrict__ zVelRB,
		float* __restrict__ xOmegaRB, float* __restrict__ yOmegaRB, float* __restrict__ zOmegaRB,
		float* __restrict__ xAngMomRB, float* __restrict__ yAngMomRB, float* __restrict__ zAngMomRB,
		const float* __restrict__ xForceRB, const float* __restrict__ yForceRB, const float* __restrict__ zForceRB,
		const float* __restrict__ xTorqueRB, const float* __restrict__ yTorqueRB, const float* __restrict__ zTorqueRB,
		float* __restrict__ a00RB, float* __restrict__ a01RB, float* __restrict__ a02RB,
		float* __restrict__ a10RB, float* __restrict__ a11RB, float* __restrict__ a12RB,
		float* __restrict__ a20RB, float* __restrict__ a21RB, float* __restrict__ a22RB,
		const float* __restrict__ massInvRB,
		const float* __restrict__ iBodyInv00RB, const float* __restrict__ iBodyInv01RB, const float* __restrict__ iBodyInv02RB,
		const float* __restrict__ iBodyInv10RB, const float* __restrict__ iBodyInv11RB, const float* __restrict__ iBodyInv12RB,
		const float* __restrict__ iBodyInv20RB, const float* __restrict__ iBodyInv21RB, const float* __restrict__ iBodyInv22RB)
	{
		const uint32_t bodyIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (bodyIdx >= N_RB) { return; }

		const float massInv = massInvRB[bodyIdx];

		xVelRB[bodyIdx] += timeStep * xForceRB[bodyIdx] * massInv;
		yVelRB[bodyIdx] += timeStep * yForceRB[bodyIdx] * massInv;
		zVelRB[bodyIdx] += timeStep * zForceRB[bodyIdx] * massInv;

		xCMRB[bodyIdx] += timeStep * xVelRB[bodyIdx];
		yCMRB[bodyIdx] += timeStep * yVelRB[bodyIdx];
		zCMRB[bodyIdx] += timeStep * zVelRB[bodyIdx];

		xAngMomRB[bodyIdx] += timeStep * xTorqueRB[bodyIdx];
		yAngMomRB[bodyIdx] += timeStep * yTorqueRB[bodyIdx];
		zAngMomRB[bodyIdx] += timeStep * zTorqueRB[bodyIdx];

		float i00, i01, i02;
		float i10, i11, i12;
		float i20, i21, i22;
		ComputeWorldInertiaInv(
			bodyIdx,
			a00RB, a01RB, a02RB,
			a10RB, a11RB, a12RB,
			a20RB, a21RB, a22RB,
			iBodyInv00RB, iBodyInv01RB, iBodyInv02RB,
			iBodyInv10RB, iBodyInv11RB, iBodyInv12RB,
			iBodyInv20RB, iBodyInv21RB, iBodyInv22RB,
			i00, i01, i02,
			i10, i11, i12,
			i20, i21, i22);

		const float lx = xAngMomRB[bodyIdx];
		const float ly = yAngMomRB[bodyIdx];
		const float lz = zAngMomRB[bodyIdx];
		const float wx = fmaf(i00, lx, fmaf(i01, ly, i02 * lz));
		const float wy = fmaf(i10, lx, fmaf(i11, ly, i12 * lz));
		const float wz = fmaf(i20, lx, fmaf(i21, ly, i22 * lz));

		xOmegaRB[bodyIdx] = wx;
		yOmegaRB[bodyIdx] = wy;
		zOmegaRB[bodyIdx] = wz;

		const float a00 = a00RB[bodyIdx], a01 = a01RB[bodyIdx], a02 = a02RB[bodyIdx];
		const float a10 = a10RB[bodyIdx], a11 = a11RB[bodyIdx], a12 = a12RB[bodyIdx];
		const float a20 = a20RB[bodyIdx], a21 = a21RB[bodyIdx], a22 = a22RB[bodyIdx];

		a00RB[bodyIdx] += timeStep * (-wz * a10 + wy * a20);
		a01RB[bodyIdx] += timeStep * (-wz * a11 + wy * a21);
		a02RB[bodyIdx] += timeStep * (-wz * a12 + wy * a22);
		a10RB[bodyIdx] += timeStep * ( wz * a00 - wx * a20);
		a11RB[bodyIdx] += timeStep * ( wz * a01 - wx * a21);
		a12RB[bodyIdx] += timeStep * ( wz * a02 - wx * a22);
		a20RB[bodyIdx] += timeStep * (-wy * a00 + wx * a10);
		a21RB[bodyIdx] += timeStep * (-wy * a01 + wx * a11);
		a22RB[bodyIdx] += timeStep * (-wy * a02 + wx * a12);

		ReOrthonormalizeBody(
			bodyIdx,
			a00RB, a01RB, a02RB,
			a10RB, a11RB, a12RB,
			a20RB, a21RB, a22RB);
	}

} // namespace GPU01

// -------------------------------------------------------------------------------------------------
// --- [ kernel launchers ] ------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

namespace GPU01
{
	void Launch_UpdateRigidParticles(
		const uint32_t* bodyIdR,
		const float* xRelPosR, const float* yRelPosR, const float* zRelPosR,
		const float* xCMRB, const float* yCMRB, const float* zCMRB,
		const float* xVelRB, const float* yVelRB, const float* zVelRB,
		const float* xOmegaRB, const float* yOmegaRB, const float* zOmegaRB,
		const float* a00RB, const float* a01RB, const float* a02RB,
		const float* a10RB, const float* a11RB, const float* a12RB,
		const float* a20RB, const float* a21RB, const float* a22RB,
		const uint32_t N_R, const uint32_t N_RB, cudaStream_t stream,
		// destinations
		float* xPosR, float* yPosR, float* zPosR,
		float* xVelR, float* yVelR, float* zVelR)
	{
		if (N_R == 0) { return; }

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_R + N_BLOCKSIZE - 1) / N_BLOCKSIZE;
		K_UpdateRigidParticles<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			bodyIdR, xRelPosR, yRelPosR, zRelPosR,
			xCMRB, yCMRB, zCMRB, xVelRB, yVelRB, zVelRB, xOmegaRB, yOmegaRB, zOmegaRB,
			a00RB, a01RB, a02RB, a10RB, a11RB, a12RB, a20RB, a21RB, a22RB,
			N_R, N_RB,
			xPosR, yPosR, zPosR, xVelR, yVelR, zVelR);

		GPU::CheckLastKernel("K_UpdateRigidParticles()");
	}

	void Launch_ClearRigidForcesAndAddGravity(
		const float* massRB,
		const float* massInvRB,
		const float xGravity, const float yGravity, const float zGravity,
		const uint32_t N_RB,
		cudaStream_t stream,
		// destinations
		float* xForceRB, float* yForceRB, float* zForceRB,
		float* xTorqueRB, float* yTorqueRB, float* zTorqueRB)
	{
		if (N_RB == 0) { return; }

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_RB + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_ClearRigidForcesAndAddGravity<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			massRB, massInvRB,
			xGravity, yGravity, zGravity,
			N_RB,
			xForceRB, yForceRB, zForceRB,
			xTorqueRB, yTorqueRB, zTorqueRB);

		GPU::CheckLastKernel("K_ClearRigidForcesAndAddGravity()");
	}

	void Launch_AccumulateRigidParticleNonPressureForces(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* xVelR, const float* yVelR, const float* zVelR,
		const float* volR,
		const uint16_t* nbrCountRF,
		const uint64_t* nbrOffsetsRF,
		const uint32_t* nbrListRF,
		const float particleSizeInvF,
		const float particleSizeSquF,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		const float massF,
		const float viscosityB,
		const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		float* xForceR, float* yForceR, float* zForceR)
	{
		if (N_R == 0) { return; }
		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_R + N_BLOCKSIZE - 1) / N_BLOCKSIZE;
		K_AccumulateRigidParticleNonPressureForces<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosR, yPosR, zPosR,
			xVelF, yVelF, zVelF, xVelR, yVelR, zVelR, volR,
			nbrCountRF, nbrOffsetsRF, nbrListRF,
			particleSizeInvF, particleSizeSquF, kernelRadiusSqu, cubicSplineAlpha,
			massF, viscosityB, N_R, xForceR, yForceR, zForceR);
		GPU::CheckLastKernel("K_AccumulateRigidParticleNonPressureForces()");
	}

	void Launch_AccumulateRigidParticlePressureForces(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const float* pressureF,
		const float* psiR,
		const uint16_t* nbrCountRF,
		const uint64_t* nbrOffsetsRF,
		const uint32_t* nbrListRF,
		const float particleSizeInvF,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		const float restDensity,
		const float massF,
		const float pressGamma,
		const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		float* xForceR, float* yForceR, float* zForceR)
	{
		if (N_R == 0) { return; }
		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_R + N_BLOCKSIZE - 1) / N_BLOCKSIZE;
		K_AccumulateRigidParticlePressureForces<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			xPosF, yPosF, zPosF, xPosR, yPosR, zPosR, pressureF, psiR,
			nbrCountRF, nbrOffsetsRF, nbrListRF,
			particleSizeInvF, kernelRadiusSqu, cubicSplineAlpha,
			restDensity, massF, pressGamma, N_R, xForceR, yForceR, zForceR);
		GPU::CheckLastKernel("K_AccumulateRigidParticlePressureForces()");
	}

	void Launch_ReduceRigidParticleForces(
		const uint32_t* rigidParticleBeginRB,
		const uint32_t* rigidParticleCountRB,
		const float* xRelPosR,
		const float* yRelPosR,
		const float* zRelPosR,
		const float* xForceR,
		const float* yForceR,
		const float* zForceR,
		const float* a00RB, const float* a01RB, const float* a02RB,
		const float* a10RB, const float* a11RB, const float* a12RB,
		const float* a20RB, const float* a21RB, const float* a22RB,
		const uint32_t N_RB,
		const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		float* xForceRB, float* yForceRB, float* zForceRB,
		float* xTorqueRB, float* yTorqueRB, float* zTorqueRB)
	{
		if (N_RB == 0 || N_R == 0) { return; }

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = N_RB;
		const uint32_t sharedBytes = 6u * N_BLOCKSIZE * static_cast<uint32_t>(sizeof(float));

		K_ReduceRigidParticleForces<<<N_GRIDSIZE, N_BLOCKSIZE, sharedBytes, stream>>>(
			rigidParticleBeginRB,
			rigidParticleCountRB,
			xRelPosR,
			yRelPosR,
			zRelPosR,
			xForceR,
			yForceR,
			zForceR,
			a00RB, a01RB, a02RB,
			a10RB, a11RB, a12RB,
			a20RB, a21RB, a22RB,
			N_RB,
			N_R,
			xForceRB, yForceRB, zForceRB,
			xTorqueRB, yTorqueRB, zTorqueRB);

		GPU::CheckLastKernel("K_ReduceRigidParticleForces()");
	}

	void Launch_IntegrateRigidBodies(
		const float timeStep,
		const uint32_t N_RB,
		cudaStream_t stream,
		// destinations
		float* xCMRB, float* yCMRB, float* zCMRB,
		float* xVelRB, float* yVelRB, float* zVelRB,
		float* xOmegaRB, float* yOmegaRB, float* zOmegaRB,
		float* xAngMomRB, float* yAngMomRB, float* zAngMomRB,
		const float* xForceRB, const float* yForceRB, const float* zForceRB,
		const float* xTorqueRB, const float* yTorqueRB, const float* zTorqueRB,
		float* a00RB, float* a01RB, float* a02RB,
		float* a10RB, float* a11RB, float* a12RB,
		float* a20RB, float* a21RB, float* a22RB,
		const float* massInvRB,
		const float* iBodyInv00RB, const float* iBodyInv01RB, const float* iBodyInv02RB,
		const float* iBodyInv10RB, const float* iBodyInv11RB, const float* iBodyInv12RB,
		const float* iBodyInv20RB, const float* iBodyInv21RB, const float* iBodyInv22RB)
	{
		if (N_RB == 0) { return; }

		const uint32_t N_BLOCKSIZE = 256;
		const uint32_t N_GRIDSIZE = (N_RB + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

		K_IntegrateRigidBodies<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream>>>(
			timeStep,
			N_RB,
			xCMRB, yCMRB, zCMRB,
			xVelRB, yVelRB, zVelRB,
			xOmegaRB, yOmegaRB, zOmegaRB,
			xAngMomRB, yAngMomRB, zAngMomRB,
			xForceRB, yForceRB, zForceRB,
			xTorqueRB, yTorqueRB, zTorqueRB,
			a00RB, a01RB, a02RB,
			a10RB, a11RB, a12RB,
			a20RB, a21RB, a22RB,
			massInvRB,
			iBodyInv00RB, iBodyInv01RB, iBodyInv02RB,
			iBodyInv10RB, iBodyInv11RB, iBodyInv12RB,
			iBodyInv20RB, iBodyInv21RB, iBodyInv22RB);

		GPU::CheckLastKernel("K_IntegrateRigidBodies()");
	}

} // namespace GPU01
