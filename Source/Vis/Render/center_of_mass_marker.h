// center_of_mass_marker.h

#pragma once

// external
#include <algorithm>
#include <array>
#include <cstdint>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>



namespace Render
{
	inline constexpr float CENTER_OF_MASS_MARKER_LENGTH_RATIO = 2.5f;
	inline constexpr float CENTER_OF_MASS_MARKER_THICKNESS_RATIO = 0.25f;

	inline std::array<glm::mat4, 3> BuildCenterOfMassMarkerTransforms(
		const glm::vec3& centerOfMass,
		const glm::mat3& orientation,
		const float particleSizeF)
	{
		const float size = std::max(particleSizeF, 0.0f);
		const float length = CENTER_OF_MASS_MARKER_LENGTH_RATIO * size;
		const float thickness = CENTER_OF_MASS_MARKER_THICKNESS_RATIO * size;
		const std::array<glm::vec3, 3> dimensions
		{
			glm::vec3{ length, thickness, thickness },
			glm::vec3{ thickness, length, thickness },
			glm::vec3{ thickness, thickness, length },
		};

		std::array<glm::mat4, 3> transforms{};
		for (uint32_t axis = 0; axis < transforms.size(); axis++)
		{
			glm::mat4 transform{ 1.0f };
			transform[0] = glm::vec4(orientation[0] * dimensions[axis].x, 0.0f);
			transform[1] = glm::vec4(orientation[1] * dimensions[axis].y, 0.0f);
			transform[2] = glm::vec4(orientation[2] * dimensions[axis].z, 0.0f);
			transform[3] = glm::vec4(centerOfMass, 1.0f);
			transforms[axis] = transform;
		}
		return transforms;
	}

} // namespace Render