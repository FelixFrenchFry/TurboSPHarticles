// rigid_body_sync.h
#pragma once

// external
#include <cstdint>
#include <vector>

// internal
#include "source/simulation/core/backend_sim.h"
#include "source/simulation/sync/state_sim.h"
#include "mpi_context.h"



namespace Dist
{
	// reusable packed storage for rigid body collectives
	struct RigidBodySyncBuffers
	{
		// number of replicated rigid bodies
		uint32_t N_RB = 0;

		// packed collective data
		std::vector<float> forceTorqueValues;
		std::vector<float> stateValues;

		void Resize(uint32_t N_RB_new);

		void PackForceTorque(const Sim::RigidBodyForceTorque& forceTorque);
		void UnpackForceTorque(Sim::RigidBodyForceTorque& forceTorque) const;
		void PackRigidBodyState(const Sync::RigidBodies& rigidBodies);
		void UnpackRigidBodyState(Sync::RigidBodies& rigidBodies) const;
	};

	// sums body-level force and torque accumulated from rank-authoritative rigid particles
	void SumRigidBodyForceTorqueAcrossRanks(
		Sim::RigidBodyForceTorque& forceTorque,
		RigidBodySyncBuffers& buffers,
		const MpiContext& mpiContext);

	// adds gravity once after body-level force and torque have been summed
	void AddRigidBodyGravity(
		Sim::RigidBodyForceTorque& forceTorque,
		const Sync::RigidBodies& rigidBodies,
		const glm::vec3& gravity);

	// broadcasts the root-integrated dynamic rigid body state to every rank
	void BroadcastRigidBodyState(
		Sync::RigidBodies& rigidBodies,
		RigidBodySyncBuffers& buffers,
		const MpiContext& mpiContext);

} // namespace Dist
