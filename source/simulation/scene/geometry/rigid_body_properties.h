// rigid_body_properties.h

#pragma once

// external
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>

// internal
#include "triangle_surface.h"



namespace Scene
{
	struct RigidBodyPhysicalProperties
	{
		glm::vec3 centerOfMass{ 0.0f };
		glm::mat3 bodyInertiaInv{ 0.0f };
		float volume = 0.0f;
		float mass = 0.0f;
	};

	// computes rigid-body properties from a closed, consistently oriented mesh
	class RigidBodyPropertyComputer
	{
	public:

		RigidBodyPhysicalProperties Compute(
			const TriangleSurfaceMesh& mesh,
			float density) const;
	};

} // namespace Scene
