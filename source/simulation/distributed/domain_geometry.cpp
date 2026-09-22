// domain_geometry.cpp

// external
#include <algorithm>
#include <limits>

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

AABB AABB::ExtendedAtGlobalBounds(const AABB& globalBounds) const
{
	AABB bounds = *this;
	for (int axis = 0; axis < 3; axis++)
	{
		if (min[axis] == globalBounds.min[axis])
		{
			bounds.min[axis] = -std::numeric_limits<float>::infinity();
		}
		if (max[axis] == globalBounds.max[axis])
		{
			bounds.max[axis] = std::numeric_limits<float>::infinity();
		}
	}
	return bounds;
}
