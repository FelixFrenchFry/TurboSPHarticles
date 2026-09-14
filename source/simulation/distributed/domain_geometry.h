// domain_geometry.h
#pragma once

// external
#include <glm/vec3.hpp>



namespace Dist
{
	// axis-aligned bounding box used by domain decomp
	struct AABB
	{
		glm::vec3 min = glm::vec3(0.0f);
		glm::vec3 max = glm::vec3(0.0f);

		[[nodiscard]] bool IsValid() const;
		[[nodiscard]] glm::vec3 Extents() const;
		[[nodiscard]] AABB Expanded(float radius) const;

		// called per particle and halo destination by the particle routing
		// (link-time optimization is off in CUDA builds)
		[[nodiscard]] bool Contains(const glm::vec3& point) const
		{
			return point.x >= min.x && point.x <= max.x
				&& point.y >= min.y && point.y <= max.y
				&& point.z >= min.z && point.z <= max.z;
		}
	};

} // namespace Dist
