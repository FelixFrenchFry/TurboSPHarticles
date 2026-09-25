// triangle_surface.cpp

// external
#include <algorithm>
#include <cmath>
#include <stdexcept>

// internal
#include "triangle_surface.h"



namespace Scene
{
	glm::vec3 ScaleToTargetExtent(TriangleSurfaceMesh& mesh, const float targetExtent)
	{
		if (!std::isfinite(targetExtent) || targetExtent <= 0.0f)
		{
			throw std::runtime_error("Mesh model target extent must be > 0.");
		}
		if (mesh.vertices.empty())
		{
			throw std::runtime_error("Cannot scale an empty triangle surface.");
		}

		// compute the raw model bounds
		glm::vec3 modelMin = mesh.vertices.front();
		glm::vec3 modelMax = mesh.vertices.front();
		for (const glm::vec3& vertex : mesh.vertices)
		{
			modelMin.x = std::min(modelMin.x, vertex.x);
			modelMin.y = std::min(modelMin.y, vertex.y);
			modelMin.z = std::min(modelMin.z, vertex.z);
			modelMax.x = std::max(modelMax.x, vertex.x);
			modelMax.y = std::max(modelMax.y, vertex.y);
			modelMax.z = std::max(modelMax.z, vertex.z);
		}

		const glm::vec3 modelExtent = modelMax - modelMin;
		const float rawModelExtent = std::max(
			modelExtent.x,
			std::max(modelExtent.y, modelExtent.z));
		if (!std::isfinite(rawModelExtent) || rawModelExtent <= 0.0f)
		{
			throw std::runtime_error("Mesh model extent must be > 0.");
		}

		// derive a uniform scale from the largest model extent
		const float modelScale = targetExtent / rawModelExtent;
		if (!std::isfinite(modelScale) || modelScale <= 0.0f)
		{
			throw std::runtime_error("Mesh model scale must be > 0.");
		}

		for (glm::vec3& vertex : mesh.vertices)
		{
			vertex *= modelScale;
		}

		return 0.5f * (modelMin + modelMax) * modelScale;
	}

} // namespace Scene
