// triangle_surface.h
#pragma once

// external
#include <glm/vec3.hpp>
#include <vector>



namespace Scene
{
	// backend-agnostic indexed triangle surface
	struct TriangleSurfaceMesh
	{
		std::vector<glm::vec3> vertices;
		std::vector<glm::uvec3> triangles;
	};

	// scales the mesh uniformly to the target extent and returns its bounding box center
	glm::vec3 ScaleToTargetExtent(TriangleSurfaceMesh& mesh, float targetExtent);

} // namespace Scene
