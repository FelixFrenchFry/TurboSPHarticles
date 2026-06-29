// rigid_bodies.cpp

// external
#include <algorithm>
#include <cmath>
#include <stdexcept>

// internal
#include "rigid_bodies.h"



namespace
{
	constexpr float RigidOrthonormalEps = 1e-12f;

	float Dot3(
		const float ax, const float ay, const float az,
		const float bx, const float by, const float bz)
	{
		return std::fmaf(ax, bx, std::fmaf(ay, by, az * bz));
	}

	float Length3(const float x, const float y, const float z)
	{
		return std::sqrt(Dot3(x, y, z, x, y, z));
	}

	void Normalize3(float& x, float& y, float& z)
	{
		const float len = Length3(x, y, z);
		if (len <= RigidOrthonormalEps || !std::isfinite(len))
		{
			throw std::runtime_error("Degenerate rigid body orientation.");
		}

		const float lenInv = 1.0f / len;
		x *= lenInv;
		y *= lenInv;
		z *= lenInv;
	}

	void ReorthonormalizeBody(Sync::RigidBodies& rigidBodies, const uint32_t bodyIdx)
	{
		// use Gram-Schmidt algo on orientation columns
		float b1x = rigidBodies.a00[bodyIdx];
		float b1y = rigidBodies.a10[bodyIdx];
		float b1z = rigidBodies.a20[bodyIdx];
		Normalize3(b1x, b1y, b1z);

		float b2x = rigidBodies.a01[bodyIdx];
		float b2y = rigidBodies.a11[bodyIdx];
		float b2z = rigidBodies.a21[bodyIdx];
		const float b1DotA2 = Dot3(b1x, b1y, b1z, b2x, b2y, b2z);
		b2x -= b1DotA2 * b1x;
		b2y -= b1DotA2 * b1y;
		b2z -= b1DotA2 * b1z;
		Normalize3(b2x, b2y, b2z);

		float b3x = rigidBodies.a02[bodyIdx];
		float b3y = rigidBodies.a12[bodyIdx];
		float b3z = rigidBodies.a22[bodyIdx];
		const float b1DotA3 = Dot3(b1x, b1y, b1z, b3x, b3y, b3z);
		const float b2DotA3 = Dot3(b2x, b2y, b2z, b3x, b3y, b3z);
		b3x -= b1DotA3 * b1x + b2DotA3 * b2x;
		b3y -= b1DotA3 * b1y + b2DotA3 * b2y;
		b3z -= b1DotA3 * b1z + b2DotA3 * b2z;
		Normalize3(b3x, b3y, b3z);

		rigidBodies.a00[bodyIdx] = b1x; rigidBodies.a01[bodyIdx] = b2x; rigidBodies.a02[bodyIdx] = b3x;
		rigidBodies.a10[bodyIdx] = b1y; rigidBodies.a11[bodyIdx] = b2y; rigidBodies.a12[bodyIdx] = b3y;
		rigidBodies.a20[bodyIdx] = b1z; rigidBodies.a21[bodyIdx] = b2z; rigidBodies.a22[bodyIdx] = b3z;
	}

	void ComputeWorldInertiaInv(
		const Sync::RigidBodies& rigidBodies, const uint32_t bodyIdx,
		float& i00, float& i01, float& i02,
		float& i10, float& i11, float& i12,
		float& i20, float& i21, float& i22)
	{
		const float a00 = rigidBodies.a00[bodyIdx], a01 = rigidBodies.a01[bodyIdx], a02 = rigidBodies.a02[bodyIdx];
		const float a10 = rigidBodies.a10[bodyIdx], a11 = rigidBodies.a11[bodyIdx], a12 = rigidBodies.a12[bodyIdx];
		const float a20 = rigidBodies.a20[bodyIdx], a21 = rigidBodies.a21[bodyIdx], a22 = rigidBodies.a22[bodyIdx];

		const float b00 = rigidBodies.iBodyInv00[bodyIdx], b01 = rigidBodies.iBodyInv01[bodyIdx], b02 = rigidBodies.iBodyInv02[bodyIdx];
		const float b10 = rigidBodies.iBodyInv10[bodyIdx], b11 = rigidBodies.iBodyInv11[bodyIdx], b12 = rigidBodies.iBodyInv12[bodyIdx];
		const float b20 = rigidBodies.iBodyInv20[bodyIdx], b21 = rigidBodies.iBodyInv21[bodyIdx], b22 = rigidBodies.iBodyInv22[bodyIdx];

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

} // namespace anonymous

namespace Sim
{
	void RigidBodyMgr::UpdateBoundaryParticles(
		const Sync::RigidBodies& rigidBodies,
		Sync::BoundaryParticles& boundaryParticles) const
	{
		boundaryParticles.EnsureBoundaryKinematicsPresent();

		// TODO: parallelize with TBB?
		for (uint32_t i = 0; i < boundaryParticles.N_; i++)
		{
			const uint32_t bodyIdx = boundaryParticles.bodyId[i];
			if (bodyIdx == Sync::InvalidRigidBodyId)
			{
				boundaryParticles.xVel[i] = 0.0f;
				boundaryParticles.yVel[i] = 0.0f;
				boundaryParticles.zVel[i] = 0.0f;
				continue;
			}

			const uint32_t sampleIdx = boundaryParticles.rigidSampleIdx[i];
			if (bodyIdx >= rigidBodies.N_ || sampleIdx >= rigidBodies.sampleN_)
			{
				throw std::runtime_error("Invalid rigid boundary attachment.");
			}
			const uint32_t sampleBegin = rigidBodies.sampleBegin[bodyIdx];
			const uint32_t sampleCount = rigidBodies.sampleCount[bodyIdx];
			if (sampleBegin > rigidBodies.sampleN_ || sampleCount > rigidBodies.sampleN_ - sampleBegin)
			{
				throw std::runtime_error("Invalid rigid boundary sample range.");
			}
			if (sampleIdx < sampleBegin || sampleIdx >= sampleBegin + sampleCount)
			{
				throw std::runtime_error("Rigid boundary sample belongs to a different body.");
			}

			const float lx = rigidBodies.xLocalB[sampleIdx];
			const float ly = rigidBodies.yLocalB[sampleIdx];
			const float lz = rigidBodies.zLocalB[sampleIdx];

			const float rx = std::fmaf(rigidBodies.a00[bodyIdx], lx,
				std::fmaf(rigidBodies.a01[bodyIdx], ly, rigidBodies.a02[bodyIdx] * lz));
			const float ry = std::fmaf(rigidBodies.a10[bodyIdx], lx,
				std::fmaf(rigidBodies.a11[bodyIdx], ly, rigidBodies.a12[bodyIdx] * lz));
			const float rz = std::fmaf(rigidBodies.a20[bodyIdx], lx,
				std::fmaf(rigidBodies.a21[bodyIdx], ly, rigidBodies.a22[bodyIdx] * lz));

			boundaryParticles.xPos[i] = rigidBodies.xCM[bodyIdx] + rx;
			boundaryParticles.yPos[i] = rigidBodies.yCM[bodyIdx] + ry;
			boundaryParticles.zPos[i] = rigidBodies.zCM[bodyIdx] + rz;

			const float wx = rigidBodies.xOmega[bodyIdx];
			const float wy = rigidBodies.yOmega[bodyIdx];
			const float wz = rigidBodies.zOmega[bodyIdx];

			boundaryParticles.xVel[i] = rigidBodies.xVel[bodyIdx] + wy * rz - wz * ry;
			boundaryParticles.yVel[i] = rigidBodies.yVel[bodyIdx] + wz * rx - wx * rz;
			boundaryParticles.zVel[i] = rigidBodies.zVel[bodyIdx] + wx * ry - wy * rx;
		}
	}

	void RigidBodyMgr::ClearForces(Sync::RigidBodies& rigidBodies) const
	{
		std::fill(rigidBodies.xForce.begin(), rigidBodies.xForce.end(), 0.0f);
		std::fill(rigidBodies.yForce.begin(), rigidBodies.yForce.end(), 0.0f);
		std::fill(rigidBodies.zForce.begin(), rigidBodies.zForce.end(), 0.0f);
		std::fill(rigidBodies.xTorque.begin(), rigidBodies.xTorque.end(), 0.0f);
		std::fill(rigidBodies.yTorque.begin(), rigidBodies.yTorque.end(), 0.0f);
		std::fill(rigidBodies.zTorque.begin(), rigidBodies.zTorque.end(), 0.0f);
	}

	void RigidBodyMgr::AddGravity(
		Sync::RigidBodies& rigidBodies,
		const float xGravity, const float yGravity, const float zGravity) const
	{
		// TODO: parallelize with TBB?
		// gravity acts at the center of mass
		for (uint32_t bodyIdx = 0; bodyIdx < rigidBodies.N_; bodyIdx++)
		{
			if (rigidBodies.massInv[bodyIdx] <= 0.0f) { continue; }

			const float mass = rigidBodies.mass[bodyIdx];
			if (mass <= 0.0f) { continue; }

			rigidBodies.xForce[bodyIdx] += mass * xGravity;
			rigidBodies.yForce[bodyIdx] += mass * yGravity;
			rigidBodies.zForce[bodyIdx] += mass * zGravity;
		}
	}

	void RigidBodyMgr::AddForce(
		Sync::RigidBodies& rigidBodies,
		const uint32_t bodyIdx,
		const float xPoint, const float yPoint, const float zPoint,
		const float xForce, const float yForce, const float zForce) const
	{
		if (bodyIdx >= rigidBodies.N_) { throw std::runtime_error("Invalid rigid body force target."); }

		const float rx = xPoint - rigidBodies.xCM[bodyIdx];
		const float ry = yPoint - rigidBodies.yCM[bodyIdx];
		const float rz = zPoint - rigidBodies.zCM[bodyIdx];

		rigidBodies.xForce[bodyIdx] += xForce;
		rigidBodies.yForce[bodyIdx] += yForce;
		rigidBodies.zForce[bodyIdx] += zForce;

		rigidBodies.xTorque[bodyIdx] += ry * zForce - rz * yForce;
		rigidBodies.yTorque[bodyIdx] += rz * xForce - rx * zForce;
		rigidBodies.zTorque[bodyIdx] += rx * yForce - ry * xForce;
	}

	void RigidBodyMgr::Integrate(Sync::RigidBodies& rigidBodies, const float timeStep) const
	{
		// TODO: parallelize with TBB?
		for (uint32_t bodyIdx = 0; bodyIdx < rigidBodies.N_; bodyIdx++)
		{
			const float massInv = rigidBodies.massInv[bodyIdx];

			rigidBodies.xVel[bodyIdx] += timeStep * rigidBodies.xForce[bodyIdx] * massInv;
			rigidBodies.yVel[bodyIdx] += timeStep * rigidBodies.yForce[bodyIdx] * massInv;
			rigidBodies.zVel[bodyIdx] += timeStep * rigidBodies.zForce[bodyIdx] * massInv;

			rigidBodies.xCM[bodyIdx] += timeStep * rigidBodies.xVel[bodyIdx];
			rigidBodies.yCM[bodyIdx] += timeStep * rigidBodies.yVel[bodyIdx];
			rigidBodies.zCM[bodyIdx] += timeStep * rigidBodies.zVel[bodyIdx];

			rigidBodies.xAngMom[bodyIdx] += timeStep * rigidBodies.xTorque[bodyIdx];
			rigidBodies.yAngMom[bodyIdx] += timeStep * rigidBodies.yTorque[bodyIdx];
			rigidBodies.zAngMom[bodyIdx] += timeStep * rigidBodies.zTorque[bodyIdx];

			float i00, i01, i02;
			float i10, i11, i12;
			float i20, i21, i22;
			ComputeWorldInertiaInv(rigidBodies, bodyIdx, i00, i01, i02, i10, i11, i12, i20, i21, i22);

			const float lx = rigidBodies.xAngMom[bodyIdx];
			const float ly = rigidBodies.yAngMom[bodyIdx];
			const float lz = rigidBodies.zAngMom[bodyIdx];
			const float wx = std::fmaf(i00, lx, std::fmaf(i01, ly, i02 * lz));
			const float wy = std::fmaf(i10, lx, std::fmaf(i11, ly, i12 * lz));
			const float wz = std::fmaf(i20, lx, std::fmaf(i21, ly, i22 * lz));

			rigidBodies.xOmega[bodyIdx] = wx;
			rigidBodies.yOmega[bodyIdx] = wy;
			rigidBodies.zOmega[bodyIdx] = wz;

			const float a00 = rigidBodies.a00[bodyIdx], a01 = rigidBodies.a01[bodyIdx], a02 = rigidBodies.a02[bodyIdx];
			const float a10 = rigidBodies.a10[bodyIdx], a11 = rigidBodies.a11[bodyIdx], a12 = rigidBodies.a12[bodyIdx];
			const float a20 = rigidBodies.a20[bodyIdx], a21 = rigidBodies.a21[bodyIdx], a22 = rigidBodies.a22[bodyIdx];

			rigidBodies.a00[bodyIdx] += timeStep * (-wz * a10 + wy * a20);
			rigidBodies.a01[bodyIdx] += timeStep * (-wz * a11 + wy * a21);
			rigidBodies.a02[bodyIdx] += timeStep * (-wz * a12 + wy * a22);
			rigidBodies.a10[bodyIdx] += timeStep * ( wz * a00 - wx * a20);
			rigidBodies.a11[bodyIdx] += timeStep * ( wz * a01 - wx * a21);
			rigidBodies.a12[bodyIdx] += timeStep * ( wz * a02 - wx * a22);
			rigidBodies.a20[bodyIdx] += timeStep * (-wy * a00 + wx * a10);
			rigidBodies.a21[bodyIdx] += timeStep * (-wy * a01 + wx * a11);
			rigidBodies.a22[bodyIdx] += timeStep * (-wy * a02 + wx * a12);

			ReorthonormalizeBody(rigidBodies, bodyIdx);
		}
	}

} // namespace Sim
