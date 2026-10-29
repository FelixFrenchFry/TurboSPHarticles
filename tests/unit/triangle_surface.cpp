// triangle_surface.cpp

// external
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

// internal
#include "source/simulation/scene/geometry/triangle_surface.h"



namespace
{
	// checks uniform scaling, bounding box center, and preservation of triangle indices
	TEST(TriangleSurfaceTests, ScalesUniformlyAndReturnsScaledBoundingBoxCenter)
	{
		Scene::TriangleSurfaceMesh mesh;
		mesh.vertices = { { -2.0f, 1.0f, -1.0f }, { 6.0f, 5.0f, 3.0f }, { 4.0f, 2.0f, 1.0f } };
		mesh.triangles = { { 0, 1, 2 } };

		const glm::vec3 center = Scene::ScaleToTargetExtent(mesh, 4.0f);

		// the original largest extent is 8, so every coordinate scales by 0.5
		ASSERT_EQ(mesh.vertices.size(), 3u);
		EXPECT_FLOAT_EQ(mesh.vertices[0].x, -1.0f);
		EXPECT_FLOAT_EQ(mesh.vertices[0].y, 0.5f);
		EXPECT_FLOAT_EQ(mesh.vertices[0].z, -0.5f);
		EXPECT_FLOAT_EQ(mesh.vertices[1].x, 3.0f);
		EXPECT_FLOAT_EQ(mesh.vertices[1].y, 2.5f);
		EXPECT_FLOAT_EQ(mesh.vertices[1].z, 1.5f);
		EXPECT_FLOAT_EQ(mesh.vertices[2].x, 2.0f);
		EXPECT_FLOAT_EQ(mesh.vertices[2].y, 1.0f);
		EXPECT_FLOAT_EQ(mesh.vertices[2].z, 0.5f);
		EXPECT_FLOAT_EQ(center.x, 1.0f);
		EXPECT_FLOAT_EQ(center.y, 1.5f);
		EXPECT_FLOAT_EQ(center.z, 0.5f);
		ASSERT_EQ(mesh.triangles.size(), 1u);
		EXPECT_EQ(mesh.triangles[0].x, 0u);
		EXPECT_EQ(mesh.triangles[0].y, 1u);
		EXPECT_EQ(mesh.triangles[0].z, 2u);
	}

	// checks that planar meshes scale using their longest in-plane axis
	TEST(TriangleSurfaceTests, AcceptsPlanarMeshes)
	{
		Scene::TriangleSurfaceMesh mesh;
		mesh.vertices = { { 0.0f, 0.0f, 2.0f }, { 2.0f, 0.0f, 2.0f }, { 0.0f, 4.0f, 2.0f } };

		const glm::vec3 center = Scene::ScaleToTargetExtent(mesh, 8.0f);

		EXPECT_FLOAT_EQ(mesh.vertices[1].x, 4.0f);
		EXPECT_FLOAT_EQ(mesh.vertices[2].y, 8.0f);
		EXPECT_FLOAT_EQ(mesh.vertices[0].z, 4.0f);
		EXPECT_FLOAT_EQ(center.x, 2.0f);
		EXPECT_FLOAT_EQ(center.y, 4.0f);
		EXPECT_FLOAT_EQ(center.z, 4.0f);
	}

	// rejects meshes without vertices
	TEST(TriangleSurfaceTests, RejectsEmptyMeshes)
	{
		Scene::TriangleSurfaceMesh mesh;

		EXPECT_THROW(Scene::ScaleToTargetExtent(mesh, 1.0f), std::runtime_error);
	}

	// rejects coincident vertices because they have no positive extent to scale
	TEST(TriangleSurfaceTests, RejectsZeroExtentMeshes)
	{
		Scene::TriangleSurfaceMesh mesh;
		mesh.vertices = { { 1.0f, 2.0f, 3.0f }, { 1.0f, 2.0f, 3.0f } };

		EXPECT_THROW(Scene::ScaleToTargetExtent(mesh, 1.0f), std::runtime_error);
	}

	class TriangleSurfaceTargetExtentTests : public testing::TestWithParam<float> {};

	// rejects nonpositive and nonfinite target extents without changing vertex positions
	TEST_P(TriangleSurfaceTargetExtentTests, RejectsInvalidTargetExtents)
	{
		Scene::TriangleSurfaceMesh mesh;
		mesh.vertices = { { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f } };

		EXPECT_THROW(Scene::ScaleToTargetExtent(mesh, GetParam()), std::runtime_error);
		EXPECT_FLOAT_EQ(mesh.vertices[1].x, 1.0f);
		EXPECT_FLOAT_EQ(mesh.vertices[1].y, 1.0f);
		EXPECT_FLOAT_EQ(mesh.vertices[1].z, 1.0f);
	}

	INSTANTIATE_TEST_SUITE_P(InvalidTargets, TriangleSurfaceTargetExtentTests,
		testing::Values(0.0f, -1.0f, std::numeric_limits<float>::infinity(),
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()));
}
