// domain_geometry.cpp

// external
#include <algorithm>

// internal
#include "domain_geometry.h"



using namespace Dist;

bool AABB::IsValid() const
{
	return max.x > min.x
		&& max.y > min.y
		&& max.z > min.z;
}

glm::vec3 AABB::Extents() const
{
	return max - min;
}

glm::vec3 AABB::Center() const
{
	return 0.5f * (min + max);
}

AABB AABB::Expanded(const float radius) const
{
	const float r = std::max(0.0f, radius);
	const glm::vec3 pad{ r, r, r };

	return AABB
	{
		.min = min - pad,
		.max = max + pad
	};
}
