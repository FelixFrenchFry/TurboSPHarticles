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
		[[nodiscard]] glm::vec3 Center() const;
		[[nodiscard]] bool Contains(const glm::vec3& point) const;
		[[nodiscard]] AABB Expanded(float radius) const;
	};

} // namespace Dist
