// boundary_mesh_placement.cpp

// external
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

// internal
#include "source/simulation/scene/boundary_mesh_helpers.h"



namespace
{
	// asymmetric bounds check all three alignment modes in a rotated container
	TEST(BoundaryMeshPlacementTests, AlignsRotatedMeshBoundsInContainerAxes)
	{
		const Scene::OBB container
		{
			.origin = { 10.0f, 20.0f, 30.0f },
			.uHat = { 0.0f, 0.0f, 1.0f },
			.vHat = { 0.0f, 1.0f, 0.0f },
			.wHat = { -1.0f, 0.0f, 0.0f },
			.uLen = 8.0f,
			.vLen = 4.0f,
			.wLen = 6.0f,
		};
		const float cos45 = std::sqrt(0.5f);
		const glm::mat3 meshToContainer
		{
			{ cos45, 0.0f, cos45 },
			{ 0.0f, 1.0f, 0.0f },
			{ -cos45, 0.0f, cos45 },
		};
		const glm::mat3 containerToWorld{ container.uHat, container.vHat, container.wHat };
		const std::vector<glm::vec3> meshLocalVertices
		{
			{ -2.0f, -1.0f, 1.0f },
			{ 2.0f, 1.0f, -1.0f },
			{ 2.0f, -1.0f, 1.0f },
		};

		Scene::BoundaryMeshEntryConfig cfg{};
		cfg.modelContainerAlignment =
		{
			Scene::BoundaryMeshAlignment::Max,
			Scene::BoundaryMeshAlignment::Min,
			Scene::BoundaryMeshAlignment::Center,
		};
		cfg.useContainerAlignment = true;
		cfg.boundaryMeshPosition = { 1.0f, 2.0f, 3.0f };

		const glm::vec3 center = Scene::ComputeBoundaryMeshCenter(
			cfg, meshLocalVertices, containerToWorld * meshToContainer, container);

		// align uMax = 3 * cos45, vMin = -1, and w in [-cos45, 3 * cos45], then add the world-space offset
		EXPECT_NEAR(center.x, 8.0f + cos45, 1e-5f);
		EXPECT_NEAR(center.y, 23.0f, 1e-5f);
		EXPECT_NEAR(center.z, 41.0f - 3.0f * cos45, 1e-5f);
	}

	// all bundled coastlines align to the far u face, the floor, and the w midpoint
	TEST(BoundaryMeshPlacementTests, LoadsAlignmentForAllCoastlineModels)
	{
		const std::filesystem::path modelCatalogPath =
			std::filesystem::path(TSP_CONFIG_DIR).parent_path() /
			"assets" / "models" / "boundary_models.json";
		for (int version = 1; version <= 3; version++)
		{
			const std::string modelName = "RockyCoastlineV" + std::to_string(version);
			SCOPED_TRACE(modelName);
			const nlohmann::json entry =
			{
				{ "model", modelName },
				{ "modelTargetExtent", 4.0 },
				{ "boundaryMesh", {
					{ "position", { 0.0, 0.0, 0.0 } },
					{ "yawDeg", -90.0 },
					{ "pitchDeg", 0.0 },
					{ "rollDeg", 0.0 },
				} }
			};

			Scene::BoundaryMeshEntryConfig cfg = Scene::LoadBoundaryMeshEntryConfigFromJson(entry, "test");
			ASSERT_TRUE(cfg.useContainerAlignment);
			ASSERT_TRUE(Scene::TryResolveBoundaryMeshModel(modelCatalogPath, cfg, "test"));
			EXPECT_EQ(cfg.modelContainerAlignment[0], Scene::BoundaryMeshAlignment::Max);
			EXPECT_EQ(cfg.modelContainerAlignment[1], Scene::BoundaryMeshAlignment::Min);
			EXPECT_EQ(cfg.modelContainerAlignment[2], Scene::BoundaryMeshAlignment::Center);
		}
	}

	// centerOfPosition supplies the world-space center without a container transform
	TEST(BoundaryMeshPlacementTests, PreservesAbsoluteCenterFromExistingConfigs)
	{
		const nlohmann::json entry =
		{
			{ "model", "RockyCoastlineV1" },
			{ "modelTargetExtent", 4.0 },
			{ "boundaryMesh", {
				{ "centerOfPosition", { 3.60775, 0.36664, 2.0 } },
				{ "yawDeg", -90.0 },
				{ "pitchDeg", 0.0 },
				{ "rollDeg", 0.0 },
			} }
		};

		const Scene::BoundaryMeshEntryConfig cfg = Scene::LoadBoundaryMeshEntryConfigFromJson(entry, "test");
		const glm::vec3 center = Scene::ComputeBoundaryMeshCenter(
			cfg, { glm::vec3{ 0.0f } }, glm::mat3{ 1.0f }, Scene::OBB{});

		EXPECT_FLOAT_EQ(center.x, 3.60775f);
		EXPECT_FLOAT_EQ(center.y, 0.36664f);
		EXPECT_FLOAT_EQ(center.z, 2.0f);
	}
}
