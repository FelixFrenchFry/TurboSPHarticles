// rigid_body_sync.cpp

// external
#include <algorithm>
#include <cstdint>
#include <vector>

// internal
#include "rigid_body_sync.h"



using namespace Dist;

namespace
{
	constexpr uint64_t N_forceTorqueComponents = 6;
	constexpr uint64_t N_rigidBodyStateComponents = 21;

	// appends one SoA field to a packed collective buffer
	void PackFloatVector(
		const std::vector<float>& src,
		std::vector<float>& dst,
		uint64_t& dstOffset)
	{
		std::copy(src.begin(), src.end(), dst.begin() + dstOffset);
		dstOffset += src.size();
	}

	// copies one packed SoA field into its destination
	void UnpackFloatVector(
		const std::vector<float>& src,
		std::vector<float>& dst,
		uint64_t& srcOffset)
	{
		std::copy_n(src.begin() + srcOffset, dst.size(), dst.begin());
		srcOffset += dst.size();
	}

} // namespace anonymous

void RigidBodySyncBuffers::Resize(const uint32_t N_RB_new)
{
	// allocate packed buffers for all rigid body components
	N_RB = N_RB_new;
	forceTorqueValues.resize(static_cast<uint64_t>(N_RB) * N_forceTorqueComponents);
	stateValues.resize(static_cast<uint64_t>(N_RB) * N_rigidBodyStateComponents);
}

void RigidBodySyncBuffers::PackForceTorque(
	const Sim::RigidBodyForceTorque& forceTorque)
{
	uint64_t dstOffset = 0;

	// pack force and torque components in a fixed order
	PackFloatVector(forceTorque.xForce, forceTorqueValues, dstOffset);
	PackFloatVector(forceTorque.yForce, forceTorqueValues, dstOffset);
	PackFloatVector(forceTorque.zForce, forceTorqueValues, dstOffset);
	PackFloatVector(forceTorque.xTorque, forceTorqueValues, dstOffset);
	PackFloatVector(forceTorque.yTorque, forceTorqueValues, dstOffset);
	PackFloatVector(forceTorque.zTorque, forceTorqueValues, dstOffset);
}

void RigidBodySyncBuffers::UnpackForceTorque(
	Sim::RigidBodyForceTorque& forceTorque) const
{
	uint64_t srcOffset = 0;

	// unpack force and torque components in the same fixed order it was packed
	UnpackFloatVector(forceTorqueValues, forceTorque.xForce, srcOffset);
	UnpackFloatVector(forceTorqueValues, forceTorque.yForce, srcOffset);
	UnpackFloatVector(forceTorqueValues, forceTorque.zForce, srcOffset);
	UnpackFloatVector(forceTorqueValues, forceTorque.xTorque, srcOffset);
	UnpackFloatVector(forceTorqueValues, forceTorque.yTorque, srcOffset);
	UnpackFloatVector(forceTorqueValues, forceTorque.zTorque, srcOffset);
}

void RigidBodySyncBuffers::PackRigidBodyState(
	const Sync::RigidBodies& rigidBodies)
{
	uint64_t dstOffset = 0;

	// pack dynamic body state in a fixed order
	PackFloatVector(rigidBodies.xCM, stateValues, dstOffset);
	PackFloatVector(rigidBodies.yCM, stateValues, dstOffset);
	PackFloatVector(rigidBodies.zCM, stateValues, dstOffset);
	PackFloatVector(rigidBodies.xVel, stateValues, dstOffset);
	PackFloatVector(rigidBodies.yVel, stateValues, dstOffset);
	PackFloatVector(rigidBodies.zVel, stateValues, dstOffset);
	PackFloatVector(rigidBodies.xOmega, stateValues, dstOffset);
	PackFloatVector(rigidBodies.yOmega, stateValues, dstOffset);
	PackFloatVector(rigidBodies.zOmega, stateValues, dstOffset);
	PackFloatVector(rigidBodies.xAngMom, stateValues, dstOffset);
	PackFloatVector(rigidBodies.yAngMom, stateValues, dstOffset);
	PackFloatVector(rigidBodies.zAngMom, stateValues, dstOffset);
	PackFloatVector(rigidBodies.a00, stateValues, dstOffset);
	PackFloatVector(rigidBodies.a01, stateValues, dstOffset);
	PackFloatVector(rigidBodies.a02, stateValues, dstOffset);
	PackFloatVector(rigidBodies.a10, stateValues, dstOffset);
	PackFloatVector(rigidBodies.a11, stateValues, dstOffset);
	PackFloatVector(rigidBodies.a12, stateValues, dstOffset);
	PackFloatVector(rigidBodies.a20, stateValues, dstOffset);
	PackFloatVector(rigidBodies.a21, stateValues, dstOffset);
	PackFloatVector(rigidBodies.a22, stateValues, dstOffset);
}

void RigidBodySyncBuffers::UnpackRigidBodyState(
	Sync::RigidBodies& rigidBodies) const
{
	uint64_t srcOffset = 0;

	// unpack dynamic body state in the same fixed order it was packed
	UnpackFloatVector(stateValues, rigidBodies.xCM, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.yCM, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.zCM, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.xVel, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.yVel, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.zVel, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.xOmega, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.yOmega, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.zOmega, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.xAngMom, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.yAngMom, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.zAngMom, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.a00, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.a01, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.a02, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.a10, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.a11, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.a12, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.a20, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.a21, srcOffset);
	UnpackFloatVector(stateValues, rigidBodies.a22, srcOffset);
}

void Dist::SumRigidBodyForceTorqueAcrossRanks(
	Sim::RigidBodyForceTorque& forceTorque,
	RigidBodySyncBuffers& buffers,
	const MpiContext& mpiContext)
{
	// reduce rank-local body loads before root integration
	buffers.PackForceTorque(forceTorque);
	mpiContext.AllReduceSumFloats(buffers.forceTorqueValues);
	buffers.UnpackForceTorque(forceTorque);
}

void Dist::AddRigidBodyGravity(
	Sim::RigidBodyForceTorque& forceTorque,
	const Sync::RigidBodies& rigidBodies,
	const glm::vec3& gravity)
{
	const uint32_t N_RB = rigidBodies.N_;

	// add gravity at center of mass of each rigid body
	for (uint32_t bodyIdx = 0; bodyIdx < N_RB; bodyIdx++)
	{
		const float mass = rigidBodies.mass[bodyIdx];
		const float massInv = rigidBodies.massInv[bodyIdx];
		if (mass <= 0.0f || massInv <= 0.0f) { continue; }

		forceTorque.xForce[bodyIdx] += mass * gravity.x;
		forceTorque.yForce[bodyIdx] += mass * gravity.y;
		forceTorque.zForce[bodyIdx] += mass * gravity.z;
	}
}

void Dist::BroadcastRigidBodyState(
	Sync::RigidBodies& rigidBodies,
	RigidBodySyncBuffers& buffers,
	const MpiContext& mpiContext)
{
	// broadcast the root-integrated dynamic body state
	if (mpiContext.IsRoot())
	{
		buffers.PackRigidBodyState(rigidBodies);
	}

	mpiContext.BroadcastFloats(buffers.stateValues, mpiContext.RootRank());

	if (!mpiContext.IsRoot())
	{
		buffers.UnpackRigidBodyState(rigidBodies);
	}
}
