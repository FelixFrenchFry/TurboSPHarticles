// rigid_body_properties.cpp

// external
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

// internal
#include "rigid_body_properties.h"



namespace
{
	using EdgeKey = std::array<uint32_t, 2>;

	struct EdgeData
	{
		uint32_t count = 0;
		int32_t orientation = 0;
		std::array<uint32_t, 2> triangleIdx{};
	};

	bool IsFinite(const glm::dvec3& value)
	{
		return std::isfinite(value.x) &&
			std::isfinite(value.y) &&
			std::isfinite(value.z);
	}

	bool IsFinite(const glm::dmat3& value)
	{
		for (uint32_t column = 0; column < 3; column++)
		{
			if (!IsFinite(value[column])) { return false; }
		}
		return true;
	}

	bool IsFinite(const glm::mat3& value)
	{
		for (uint32_t column = 0; column < 3; column++)
		{
			if (!std::isfinite(value[column].x) ||
				!std::isfinite(value[column].y) ||
				!std::isfinite(value[column].z))
			{
				return false;
			}
		}
		return true;
	}

	glm::dmat3 OuterProduct(
		const glm::dvec3& left,
		const glm::dvec3& right)
	{
		glm::dmat3 result{ 0.0 };
		result[0] = { left.x * right.x, left.y * right.x, left.z * right.x };
		result[1] = { left.x * right.y, left.y * right.y, left.z * right.y };
		result[2] = { left.x * right.z, left.y * right.z, left.z * right.z };
		return result;
	}

	void AddEdge(
		std::map<EdgeKey, EdgeData>& edges,
		const uint32_t a,
		const uint32_t b,
		const uint32_t triangleIdx)
	{
		const EdgeKey key = a < b
			? EdgeKey{ a, b }
			: EdgeKey{ b, a };
		const int32_t orientation = a < b ? 1 : -1;

		EdgeData& edge = edges[key];
		if (edge.count < 2) { edge.triangleIdx[edge.count] = triangleIdx; }
		edge.count++;
		edge.orientation += orientation;
		if (edge.count > 2)
		{
			throw std::runtime_error(
				"Rigid mesh contains an edge shared by more than two triangles.");
		}
	}

} // namespace anonymous

namespace Scene
{
	RigidBodyPhysicalProperties RigidBodyPropertyComputer::Compute(
		const TriangleSurfaceMesh& mesh,
		const float density) const
	{
		// validate mesh inputs and rigid body density
		if (mesh.vertices.empty() || mesh.triangles.empty())
		{
			throw std::runtime_error("Rigid mesh does not contain a triangle surface.");
		}
		if (mesh.vertices.size() > std::numeric_limits<uint32_t>::max() ||
			mesh.triangles.size() > std::numeric_limits<uint32_t>::max())
		{
			throw std::runtime_error("Rigid mesh exceeds uint32 index limits.");
		}
		if (!std::isfinite(density) || density <= 0.0f)
		{
			throw std::runtime_error("Rigid body density must be > 0.");
		}

		for (const glm::vec3& vertex : mesh.vertices)
		{
			if (!std::isfinite(vertex.x) ||
				!std::isfinite(vertex.y) ||
				!std::isfinite(vertex.z))
			{
				throw std::runtime_error("Rigid mesh contains a non-finite vertex.");
			}
		}

		// validate closed, consistently oriented triangle connectivity
		std::map<EdgeKey, EdgeData> edges;
		std::vector<std::vector<uint32_t>> triangleNeighbors(mesh.triangles.size());
		std::vector<std::vector<uint32_t>> incidentTriangles(mesh.vertices.size());
		for (uint32_t triangleIdx = 0;
			triangleIdx < static_cast<uint32_t>(mesh.triangles.size());
			triangleIdx++)
		{
			const glm::uvec3& triangle = mesh.triangles[triangleIdx];
			if (triangle.x >= mesh.vertices.size() ||
				triangle.y >= mesh.vertices.size() ||
				triangle.z >= mesh.vertices.size())
			{
				throw std::runtime_error("Rigid mesh contains an invalid vertex index.");
			}
			if (triangle.x == triangle.y ||
				triangle.x == triangle.z ||
				triangle.y == triangle.z)
			{
				throw std::runtime_error("Rigid mesh contains a degenerate triangle.");
			}

			AddEdge(edges, triangle.x, triangle.y, triangleIdx);
			AddEdge(edges, triangle.y, triangle.z, triangleIdx);
			AddEdge(edges, triangle.z, triangle.x, triangleIdx);
			incidentTriangles[triangle.x].push_back(triangleIdx);
			incidentTriangles[triangle.y].push_back(triangleIdx);
			incidentTriangles[triangle.z].push_back(triangleIdx);
		}

		// validate closed edges and build triangle adjacency
		for (const auto& edgeEntry : edges)
		{
			const EdgeKey& key = edgeEntry.first;
			const EdgeData& edge = edgeEntry.second;
			if (edge.count != 2 || edge.orientation != 0)
			{
				throw std::runtime_error(
					"Rigid mesh must be a closed consistently oriented triangle surface.");
			}

			triangleNeighbors[edge.triangleIdx[0]].push_back(edge.triangleIdx[1]);
			triangleNeighbors[edge.triangleIdx[1]].push_back(edge.triangleIdx[0]);
			(void)key;
		}

		// validate that every vertex has one connected incident triangle fan
		for (const std::vector<uint32_t>& incident : incidentTriangles)
		{
			if (incident.empty()) { continue; }

			std::vector<uint32_t> pending{ incident.front() };
			std::vector<uint32_t> visited;
			visited.reserve(incident.size());
			while (!pending.empty())
			{
				const uint32_t triangleIdx = pending.back();
				pending.pop_back();
				if (std::find(visited.begin(), visited.end(), triangleIdx) != visited.end())
				{
					continue;
				}

				visited.push_back(triangleIdx);
				for (const uint32_t neighborIdx : triangleNeighbors[triangleIdx])
				{
					if (std::binary_search(incident.begin(), incident.end(), neighborIdx))
					{
						pending.push_back(neighborIdx);
					}
				}
			}

			if (visited.size() != incident.size())
			{
				throw std::runtime_error("Rigid mesh is non-manifold at a vertex.");
			}
		}

		// accumulate signed tetrahedral volume moments around the origin
		double signedVolume = 0.0;
		glm::dvec3 firstMoment{ 0.0 };
		glm::dmat3 secondMoment{ 0.0 };

		for (const glm::uvec3& triangle : mesh.triangles)
		{
			const glm::dvec3 a = glm::dvec3(mesh.vertices[triangle.x]);
			const glm::dvec3 b = glm::dvec3(mesh.vertices[triangle.y]);
			const glm::dvec3 c = glm::dvec3(mesh.vertices[triangle.z]);
			const double determinant = glm::dot(a, glm::cross(b, c));
			const double tetraVolume = determinant / 6.0;

			if (!std::isfinite(tetraVolume))
			{
				throw std::runtime_error("Rigid mesh volume computation is not finite.");
			}

			signedVolume += tetraVolume;
			firstMoment += tetraVolume * (a + b + c) / 4.0;
			secondMoment += tetraVolume *
				(0.1 * (OuterProduct(a, a) + OuterProduct(b, b) + OuterProduct(c, c)) +
				 0.05 * (OuterProduct(a, b) + OuterProduct(b, a) +
					OuterProduct(a, c) + OuterProduct(c, a) +
					OuterProduct(b, c) + OuterProduct(c, b)));
		}

		if (!std::isfinite(signedVolume) ||
			!IsFinite(firstMoment) ||
			!IsFinite(secondMoment) ||
			std::abs(signedVolume) <= std::numeric_limits<double>::epsilon())
		{
			throw std::runtime_error("Rigid mesh enclosed volume is invalid.");
		}

		// normalize globally reversed triangle winding
		if (signedVolume < 0.0)
		{
			signedVolume = -signedVolume;
			firstMoment = -firstMoment;
			secondMoment = -secondMoment;
		}

		// shift volume moments to the center of mass and assemble the inertia tensor
		const glm::dvec3 centerOfMass = firstMoment / signedVolume;
		const glm::dmat3 centralSecondMoment =
			secondMoment - signedVolume * OuterProduct(centerOfMass, centerOfMass);
		const double secondMomentTrace =
			centralSecondMoment[0][0] +
			centralSecondMoment[1][1] +
			centralSecondMoment[2][2];

		glm::dmat3 inertia{ 0.0 };
		for (uint32_t column = 0; column < 3; column++)
		{
			for (uint32_t row = 0; row < 3; row++)
			{
				inertia[column][row] = static_cast<double>(row == column)
					* secondMomentTrace - centralSecondMoment[column][row];
			}
		}
		inertia *= static_cast<double>(density);

		// derive and validate the physical properties in double precision
		const double volume = signedVolume;
		const double mass = volume * static_cast<double>(density);
		const double inertiaDeterminant = glm::determinant(inertia);
		if (!std::isfinite(volume) ||
			!std::isfinite(mass) ||
			!IsFinite(centerOfMass) ||
			!IsFinite(inertia) ||
			!std::isfinite(inertiaDeterminant) ||
			mass > std::numeric_limits<float>::max() ||
			volume > std::numeric_limits<float>::max() ||
			inertiaDeterminant <= 0.0 ||
			inertia[0][0] <= 0.0 ||
			inertia[1][1] <= 0.0 ||
			inertia[2][2] <= 0.0)
		{
			throw std::runtime_error("Rigid mesh physical properties are invalid.");
		}

		// invert the inertia tensor in the body frame
		const glm::dmat3 bodyInertiaInv = glm::inverse(inertia);
		if (!IsFinite(bodyInertiaInv))
		{
			throw std::runtime_error("Rigid mesh inverse inertia is not finite.");
		}

		// convert validated properties into the use float types
		const float volumeFloat = static_cast<float>(volume);
		const float massFloat = static_cast<float>(mass);
		const glm::vec3 centerOfMassFloat = glm::vec3(centerOfMass);
		const glm::mat3 bodyInertiaInvFloat = glm::mat3(bodyInertiaInv);
		if (!std::isfinite(volumeFloat) || volumeFloat <= 0.0f ||
			!std::isfinite(massFloat) || massFloat <= 0.0f ||
			!IsFinite(centerOfMassFloat) ||
			!IsFinite(bodyInertiaInvFloat))
		{
			throw std::runtime_error("Rigid mesh physical properties exceed float limits.");
		}

		RigidBodyPhysicalProperties properties;
		properties.volume = volumeFloat;
		properties.mass = massFloat;
		properties.centerOfMass = centerOfMassFloat;
		properties.bodyInertiaInv = bodyInertiaInvFloat;
		return properties;
	}

} // namespace Scene
