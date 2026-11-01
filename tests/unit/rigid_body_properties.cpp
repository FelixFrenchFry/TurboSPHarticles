// rigid_body_properties.cpp

// external
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <utility>

// internal
#include "source/simulation/scene/geometry/rigid_body_properties.h"



namespace
{
	Scene::TriangleSurfaceMesh MakeBoxMesh()
	{
		Scene::TriangleSurfaceMesh mesh;
		mesh.vertices = {
			{ -1.0f, -2.0f, -3.0f }, { 1.0f, -2.0f, -3.0f },
			{ 1.0f, 2.0f, -3.0f }, { -1.0f, 2.0f, -3.0f },
			{ -1.0f, -2.0f, 3.0f }, { 1.0f, -2.0f, 3.0f },
			{ 1.0f, 2.0f, 3.0f }, { -1.0f, 2.0f, 3.0f } };
		mesh.triangles = {
			{ 0, 2, 1 }, { 0, 3, 2 }, { 4, 5, 6 }, { 4, 6, 7 },
			{ 0, 1, 5 }, { 0, 5, 4 }, { 3, 7, 6 }, { 3, 6, 2 },
			{ 0, 4, 7 }, { 0, 7, 3 }, { 1, 2, 6 }, { 1, 6, 5 } };
		return mesh;
	}

	Scene::TriangleSurfaceMesh MakeTetrahedronMesh()
	{
		Scene::TriangleSurfaceMesh mesh;
		mesh.vertices = {
			{ 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f },
			{ 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } };
		mesh.triangles = { { 0, 2, 1 }, { 0, 1, 3 }, { 0, 3, 2 }, { 1, 2, 3 } };
		return mesh;
	}

	// at density six the unit tetrahedron has mass one and this analytical inverse inertia
	glm::mat3 TetrahedronInverseInertia()
	{
		return glm::mat3{
			14.0f, -2.0f, -2.0f,
			-2.0f, 14.0f, -2.0f,
			-2.0f, -2.0f, 14.0f };
	}

	void ExpectCenterNear(const glm::vec3& actual, const glm::vec3& expected)
	{
		for (int axis = 0; axis < 3; axis++)
		{
			SCOPED_TRACE(axis);
			EXPECT_NEAR(actual[axis], expected[axis], 1e-6f);
		}
	}

	void ExpectInverseInertiaNear(const glm::mat3& actual, const glm::mat3& expected)
	{
		for (int column = 0; column < 3; column++)
		{
			SCOPED_TRACE(column);
			for (int row = 0; row < 3; row++)
			{
				SCOPED_TRACE(row);
				EXPECT_NEAR(actual[column][row], expected[column][row],
					1e-8f + 2e-6f * std::abs(expected[column][row]));
			}
		}
	}

	// unequal side lengths distinguish the three principal moments of a centered box
	TEST(RigidBodyPropertyTests, ComputesKnownBoxProperties)
	{
		const Scene::RigidBodyPropertyComputer computer;
		const auto actual = computer.Compute(MakeBoxMesh(), 3.0f);
		const glm::mat3 expected{
			1.0f / 624.0f, 0.0f, 0.0f,
			0.0f, 1.0f / 480.0f, 0.0f,
			0.0f, 0.0f, 1.0f / 240.0f };

		EXPECT_FLOAT_EQ(actual.volume, 48.0f);
		EXPECT_FLOAT_EQ(actual.mass, 144.0f);
		ExpectCenterNear(actual.centerOfMass, glm::vec3{ 0.0f });
		ExpectInverseInertiaNear(actual.bodyInertiaInv, expected);
	}

	// the tetrahedron checks a nonzero center of mass and all off-diagonal inertia terms
	TEST(RigidBodyPropertyTests, ComputesKnownTetrahedronProperties)
	{
		const Scene::RigidBodyPropertyComputer computer;
		const auto actual = computer.Compute(MakeTetrahedronMesh(), 6.0f);

		EXPECT_FLOAT_EQ(actual.volume, 1.0f / 6.0f);
		EXPECT_FLOAT_EQ(actual.mass, 1.0f);
		ExpectCenterNear(actual.centerOfMass, glm::vec3{ 0.25f });
		ExpectInverseInertiaNear(actual.bodyInertiaInv, TetrahedronInverseInertia());
	}

	// translating the mesh moves its center while preserving inertia about that center
	TEST(RigidBodyPropertyTests, TranslatesCenterWithoutChangingInverseInertia)
	{
		auto mesh = MakeTetrahedronMesh();
		for (auto& vertex : mesh.vertices) { vertex += glm::vec3{ 8.0f, -4.0f, 2.0f }; }
		const Scene::RigidBodyPropertyComputer computer;
		const auto actual = computer.Compute(mesh, 6.0f);

		EXPECT_FLOAT_EQ(actual.volume, 1.0f / 6.0f);
		EXPECT_FLOAT_EQ(actual.mass, 1.0f);
		ExpectCenterNear(actual.centerOfMass, glm::vec3{ 8.25f, -3.75f, 2.25f });
		ExpectInverseInertiaNear(actual.bodyInertiaInv, TetrahedronInverseInertia());
	}

	// rotating a translated box moves its center and mixes the unequal principal moments
	TEST(RigidBodyPropertyTests, RotatesCenterAndInverseInertiaWithTheMesh)
	{
		auto mesh = MakeBoxMesh();
		for (auto& vertex : mesh.vertices)
		{
			const float x = vertex.x + 5.0f;
			const float y = vertex.y;
			vertex.x = (3.0f * x - 4.0f * y) / 5.0f;
			vertex.y = (4.0f * x + 3.0f * y) / 5.0f;
		}
		const Scene::RigidBodyPropertyComputer computer;
		const auto actual = computer.Compute(mesh, 3.0f);
		const float ix = 1.0f / 624.0f;
		const float iy = 1.0f / 480.0f;
		const float xy = 12.0f * (ix - iy) / 25.0f;
		const glm::mat3 expected{
			(9.0f * ix + 16.0f * iy) / 25.0f, xy, 0.0f,
			xy, (16.0f * ix + 9.0f * iy) / 25.0f, 0.0f,
			0.0f, 0.0f, 1.0f / 240.0f };

		EXPECT_FLOAT_EQ(actual.volume, 48.0f);
		EXPECT_FLOAT_EQ(actual.mass, 144.0f);
		ExpectCenterNear(actual.centerOfMass, glm::vec3{ 3.0f, 4.0f, 0.0f });
		ExpectInverseInertiaNear(actual.bodyInertiaInv, expected);
	}

	// constant density gives cubic volume and mass scaling and inverse fifth-power inertia scaling
	TEST(RigidBodyPropertyTests, ScalesPropertiesWithMeshSize)
	{
		const Scene::RigidBodyPropertyComputer computer;
		for (const float scale : { 0.5f, 2.0f })
		{
			SCOPED_TRACE(scale);
			auto mesh = MakeTetrahedronMesh();
			for (auto& vertex : mesh.vertices) { vertex *= scale; }
			const auto actual = computer.Compute(mesh, 6.0f);
			const float volumeScale = scale * scale * scale;
			const float inertiaScale = volumeScale * scale * scale;

			EXPECT_FLOAT_EQ(actual.volume, volumeScale / 6.0f);
			EXPECT_FLOAT_EQ(actual.mass, volumeScale);
			ExpectCenterNear(actual.centerOfMass, glm::vec3{ 0.25f * scale });
			ExpectInverseInertiaNear(actual.bodyInertiaInv, TetrahedronInverseInertia() / inertiaScale);
		}
	}

	// changing density affects mass and inverse inertia while preserving the geometric properties
	TEST(RigidBodyPropertyTests, ScalesMassAndInverseInertiaWithDensity)
	{
		const Scene::RigidBodyPropertyComputer computer;
		for (const float density : { 3.0f, 6.0f, 12.0f })
		{
			SCOPED_TRACE(density);
			const auto actual = computer.Compute(MakeTetrahedronMesh(), density);

			EXPECT_FLOAT_EQ(actual.volume, 1.0f / 6.0f);
			EXPECT_FLOAT_EQ(actual.mass, density / 6.0f);
			ExpectCenterNear(actual.centerOfMass, glm::vec3{ 0.25f });
			ExpectInverseInertiaNear(actual.bodyInertiaInv, TetrahedronInverseInertia() * (6.0f / density));
		}
	}

	// reversing every triangle preserves positive physical properties and all inertia terms
	TEST(RigidBodyPropertyTests, AcceptsGloballyReversedWinding)
	{
		auto mesh = MakeTetrahedronMesh();
		for (auto& triangle : mesh.triangles) { std::swap(triangle.y, triangle.z); }
		const Scene::RigidBodyPropertyComputer computer;
		const auto actual = computer.Compute(mesh, 6.0f);

		EXPECT_FLOAT_EQ(actual.volume, 1.0f / 6.0f);
		EXPECT_FLOAT_EQ(actual.mass, 1.0f);
		ExpectCenterNear(actual.centerOfMass, glm::vec3{ 0.25f });
		ExpectInverseInertiaNear(actual.bodyInertiaInv, TetrahedronInverseInertia());
	}

	// both vertices and triangles are required to describe a closed surface
	TEST(RigidBodyPropertyTests, RejectsEmptySurfaces)
	{
		const Scene::RigidBodyPropertyComputer computer;
		EXPECT_THROW(computer.Compute(Scene::TriangleSurfaceMesh{}, 6.0f), std::runtime_error);
		auto mesh = MakeTetrahedronMesh();
		mesh.triangles.clear();
		EXPECT_THROW(computer.Compute(mesh, 6.0f), std::runtime_error);
		mesh = MakeTetrahedronMesh();
		mesh.vertices.clear();
		EXPECT_THROW(computer.Compute(mesh, 6.0f), std::runtime_error);
	}

	// a missing face must be rejected even when its removal preserves the signed volume
	TEST(RigidBodyPropertyTests, RejectsOpenMeshes)
	{
		auto mesh = MakeTetrahedronMesh();
		mesh.triangles.erase(mesh.triangles.begin());
		const Scene::RigidBodyPropertyComputer computer;

		EXPECT_THROW(computer.Compute(mesh, 6.0f), std::runtime_error);
	}

	// reversing a single face breaks the orientation shared by adjacent triangles
	TEST(RigidBodyPropertyTests, RejectsInconsistentTriangleWinding)
	{
		auto mesh = MakeTetrahedronMesh();
		std::swap(mesh.triangles[0].y, mesh.triangles[0].z);
		const Scene::RigidBodyPropertyComputer computer;

		EXPECT_THROW(computer.Compute(mesh, 6.0f), std::runtime_error);
	}

	// each triangle index must reference an existing vertex
	TEST(RigidBodyPropertyTests, RejectsInvalidVertexIndices)
	{
		const Scene::RigidBodyPropertyComputer computer;
		for (int axis = 0; axis < 3; axis++)
		{
			SCOPED_TRACE(axis);
			auto mesh = MakeTetrahedronMesh();
			mesh.triangles[0][axis] = static_cast<uint32_t>(mesh.vertices.size());

			EXPECT_THROW(computer.Compute(mesh, 6.0f), std::runtime_error);
		}
	}

	// repeated vertex indices cannot form a triangle
	TEST(RigidBodyPropertyTests, RejectsRepeatedTriangleIndices)
	{
		const Scene::RigidBodyPropertyComputer computer;
		for (int axis = 0; axis < 3; axis++)
		{
			SCOPED_TRACE(axis);
			auto mesh = MakeTetrahedronMesh();
			mesh.triangles[0][axis] = mesh.triangles[0][(axis + 1) % 3];

			EXPECT_THROW(computer.Compute(mesh, 6.0f), std::runtime_error);
		}
	}

	// closed connectivity alone is insufficient when the enclosed volume is zero
	TEST(RigidBodyPropertyTests, RejectsZeroVolumeMeshes)
	{
		auto mesh = MakeTetrahedronMesh();
		mesh.vertices[3] = { 1.0f, 1.0f, 0.0f };
		const Scene::RigidBodyPropertyComputer computer;

		EXPECT_THROW(computer.Compute(mesh, 6.0f), std::runtime_error);
	}

	// a duplicated face gives its edges more than two incident triangles
	TEST(RigidBodyPropertyTests, RejectsNonManifoldEdges)
	{
		auto mesh = MakeTetrahedronMesh();
		mesh.triangles.push_back(mesh.triangles[0]);
		const Scene::RigidBodyPropertyComputer computer;

		EXPECT_THROW(computer.Compute(mesh, 6.0f), std::runtime_error);
	}

	// two closed tetrahedra sharing only one vertex have disconnected incident triangle fans
	TEST(RigidBodyPropertyTests, RejectsDisconnectedVertexFans)
	{
		auto mesh = MakeTetrahedronMesh();
		mesh.vertices.insert(mesh.vertices.end(), {
			{ -1.0f, 0.0f, 0.0f }, { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, -1.0f } });
		const auto second = MakeTetrahedronMesh();
		for (auto triangle : second.triangles)
		{
			for (int axis = 0; axis < 3; axis++)
			{
				if (triangle[axis] != 0) { triangle[axis] += 3; }
			}
			std::swap(triangle.y, triangle.z);
			mesh.triangles.push_back(triangle);
		}
		const Scene::RigidBodyPropertyComputer computer;

		EXPECT_THROW(computer.Compute(mesh, 6.0f), std::runtime_error);
	}

	class RigidBodyInvalidDensityTests : public testing::TestWithParam<float> {};

	// density must be positive and finite even for a valid closed mesh
	TEST_P(RigidBodyInvalidDensityTests, RejectsInvalidDensity)
	{
		const Scene::RigidBodyPropertyComputer computer;

		EXPECT_THROW(computer.Compute(MakeTetrahedronMesh(), GetParam()), std::runtime_error);
	}

	INSTANTIATE_TEST_SUITE_P(InvalidDensity, RigidBodyInvalidDensityTests,
		testing::Values(0.0f, -1.0f, std::numeric_limits<float>::infinity(),
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()));

	class RigidBodyNonFiniteVertexTests : public testing::TestWithParam<float> {};

	// every coordinate must be finite before volume moments are accumulated
	TEST_P(RigidBodyNonFiniteVertexTests, RejectsNonFiniteCoordinates)
	{
		const Scene::RigidBodyPropertyComputer computer;
		for (int axis = 0; axis < 3; axis++)
		{
			SCOPED_TRACE(axis);
			auto mesh = MakeTetrahedronMesh();
			mesh.vertices[1][axis] = GetParam();

			EXPECT_THROW(computer.Compute(mesh, 6.0f), std::runtime_error);
		}
	}

	INSTANTIATE_TEST_SUITE_P(NonFiniteVertex, RigidBodyNonFiniteVertexTests,
		testing::Values(std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
			std::numeric_limits<float>::quiet_NaN()));
}
