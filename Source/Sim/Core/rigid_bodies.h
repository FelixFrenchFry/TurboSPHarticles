// rigid_bodies.h

#pragma once

// external
#include <cstdint>

// internal
#include "Source/Sim/Sync/state_sim.h"



namespace Sim
{
	class RigidBodyMgr
	{
	public:

		void UpdateBoundaryParticles(
			const Sync::RigidBodies& rigidBodies,
			Sync::BoundaryParticles& boundaryParticles) const;

		void ClearForces(Sync::RigidBodies& rigidBodies) const;

		void AddGravity(
			Sync::RigidBodies& rigidBodies,
			float xGravity, float yGravity, float zGravity) const;

		void AddForce(
			Sync::RigidBodies& rigidBodies,
			uint32_t bodyIdx,
			float xPoint, float yPoint, float zPoint,
			float xForce, float yForce, float zForce) const;

		void Integrate(Sync::RigidBodies& rigidBodies, float timeStep) const;
	};

} // namespace Sim
