// checkpoint.cpp

// external
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

// internal
#include "source/eval/serial/checkpoint.h"



namespace
{
	nlohmann::json MakeFileInfo()
	{
		return { { "file", "state.bin" }, { "bytes", 604u },
			{ "fluidCount", 2u }, { "boundaryCount", 1u },
			{ "rigidCount", 3u }, { "bodyCount", 2u } };
	}

	nlohmann::json MakeManifest(const Sim::SimParams& params = {},
		const Export::CheckpointMode mode = Export::CheckpointMode::Single)
	{
		auto manifest = Export::MakeCheckpointManifest(mode, 42, 1.25, params);
		manifest["files"].push_back(MakeFileInfo());
		return manifest;
	}

	// explicit component values check both conversion directions independently
	TEST(CheckpointConversionTests, EncodesAndDecodesKnownVector)
	{
		const glm::vec3 expected{ 1.25f, -2.5f, 0.0f };
		const nlohmann::json encoded{ 1.25, -2.5, 0.0 };
		EXPECT_EQ(Export::CheckpointVec3ToJson(expected), encoded);
		const auto actual = Export::CheckpointVec3FromJson(encoded);
		for (int axis = 0; axis < 3; axis++)
		{
			SCOPED_TRACE(axis);
			EXPECT_FLOAT_EQ(actual[axis], expected[axis]);
		}
	}

	// vectors require exactly three components in an array
	TEST(CheckpointConversionTests, RejectsMalformedVectorShapes)
	{
		for (const auto& value : { nlohmann::json(nullptr), nlohmann::json::object(),
			nlohmann::json("vector"), nlohmann::json::array(),
			nlohmann::json{ 1, 2 }, nlohmann::json{ 1, 2, 3, 4 } })
		{
			SCOPED_TRACE(value.dump());
			EXPECT_THROW(Export::CheckpointVec3FromJson(value), std::runtime_error);
		}
	}

	// a correctly sized array still requires numeric components on every axis
	TEST(CheckpointConversionTests, RejectsNonNumericVectorComponents)
	{
		for (const auto& component : { nlohmann::json("1"), nlohmann::json(true), nlohmann::json(nullptr) })
		{
			SCOPED_TRACE(component.dump());
			for (int axis = 0; axis < 3; axis++)
			{
				SCOPED_TRACE(axis);
				nlohmann::json value{ 1, 2, 3 };
				value[axis] = component;
				EXPECT_THROW(Export::CheckpointVec3FromJson(value), std::runtime_error);
			}
		}
	}

	// neither conversion direction may accept a non-finite coordinate
	TEST(CheckpointConversionTests, RejectsNonFiniteVectorComponents)
	{
		for (const float component : { std::numeric_limits<float>::infinity(),
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() })
		{
			SCOPED_TRACE(component);
			for (int axis = 0; axis < 3; axis++)
			{
				SCOPED_TRACE(axis);
				glm::vec3 value{ 1.0f, 2.0f, 3.0f };
				value[axis] = component;
				nlohmann::json encoded{ 1, 2, 3 };
				encoded[axis] = component;
				EXPECT_THROW(Export::CheckpointVec3ToJson(value), std::runtime_error);
				EXPECT_THROW(Export::CheckpointVec3FromJson(encoded), std::runtime_error);
			}
		}
	}

	// asymmetric bounds distinguish endpoint order and every coordinate
	TEST(CheckpointConversionTests, EncodesAndDecodesKnownBounds)
	{
		const glm::vec3 expectedMin{ -3.0f, -2.0f, -1.0f };
		const glm::vec3 expectedMax{ 4.0f, 5.0f, 6.0f };
		const nlohmann::json encoded{ { -3, -2, -1 }, { 4, 5, 6 } };
		EXPECT_EQ(Export::CheckpointBoundsToJson(expectedMin, expectedMax), encoded);
		glm::vec3 min{}, max{};
		Export::CheckpointBoundsFromJson(encoded, min, max);
		for (int axis = 0; axis < 3; axis++)
		{
			SCOPED_TRACE(axis);
			EXPECT_FLOAT_EQ(min[axis], expectedMin[axis]);
			EXPECT_FLOAT_EQ(max[axis], expectedMax[axis]);
		}
	}

	// bounds require two endpoints, each containing exactly three components
	TEST(CheckpointConversionTests, RejectsMalformedBoundsShapes)
	{
		const nlohmann::json endpoint{ 1, 2, 3 };
		for (const auto& value : { nlohmann::json(nullptr), nlohmann::json::object(),
			nlohmann::json("bounds"), nlohmann::json::array(),
			nlohmann::json::array({ endpoint }), nlohmann::json::array({ nlohmann::json{ 0, 0, 0 }, endpoint, endpoint }),
			nlohmann::json{ { 0, 0 }, { 1, 2, 3 } }, nlohmann::json{ { 0, 0, 0 }, { 1, 2, 3, 4 } } })
		{
			SCOPED_TRACE(value.dump());
			glm::vec3 min{}, max{};
			EXPECT_THROW(Export::CheckpointBoundsFromJson(value, min, max), std::runtime_error);
		}
	}

	// each endpoint must be finite on every axis when saving and loading bounds
	TEST(CheckpointConversionTests, RejectsNonFiniteBounds)
	{
		for (const float component : { std::numeric_limits<float>::infinity(),
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() })
		{
			SCOPED_TRACE(component);
			for (int endpoint = 0; endpoint < 2; endpoint++)
			{
				SCOPED_TRACE(endpoint);
				for (int axis = 0; axis < 3; axis++)
				{
					SCOPED_TRACE(axis);
					glm::vec3 min{ -3.0f, -2.0f, -1.0f }, max{ 4.0f, 5.0f, 6.0f };
					(endpoint == 0 ? min : max)[axis] = component;
					nlohmann::json encoded{ { -3, -2, -1 }, { 4, 5, 6 } };
					encoded[endpoint][axis] = component;
					EXPECT_THROW(Export::CheckpointBoundsToJson(min, max), std::runtime_error);
					EXPECT_THROW(Export::CheckpointBoundsFromJson(encoded, min, max), std::runtime_error);
				}
			}
		}
	}

	// zero or negative extent on any axis cannot describe a restart domain
	TEST(CheckpointConversionTests, RejectsEmptyAndInvertedBounds)
	{
		for (int axis = 0; axis < 3; axis++)
		{
			SCOPED_TRACE(axis);
			for (const int upper : { 0, -1 })
			{
				SCOPED_TRACE(upper);
				nlohmann::json encoded{ { 0, 0, 0 }, { 1, 1, 1 } };
				encoded[1][axis] = upper;
				glm::vec3 min{}, max{};
				EXPECT_THROW(Export::CheckpointBoundsFromJson(encoded, min, max), std::runtime_error);
			}
		}
	}

	// manifests identify the wire formats and retain the saved time and resolved parameters
	TEST(CheckpointManifestTests, BuildsExpectedMetadata)
	{
		Sim::SimParams params;
		params.restDensity = 900.0f;
		for (const auto mode : { Export::CheckpointMode::Single, Export::CheckpointMode::Distributed })
		{
			SCOPED_TRACE(mode == Export::CheckpointMode::Single ? "single" : "distributed");
			const auto manifest = Export::MakeCheckpointManifest(mode, 5'000'000'000ull, 1.25, params);

			EXPECT_EQ(manifest.at("format"), "TSPCheckpoint");
			EXPECT_EQ(manifest.at("version"), 3);
			EXPECT_EQ(manifest.at("stateFormat"), "TSPSIM-2");
			EXPECT_EQ(manifest.at("mode"), mode == Export::CheckpointMode::Single ? "single" : "distributed");
			EXPECT_EQ(manifest.at("step").get<uint64_t>(), 5'000'000'000ull);
			EXPECT_DOUBLE_EQ(manifest.at("time").get<double>(), 1.25);
			EXPECT_EQ(manifest.at("params"), Sim::ToJson(params));
			EXPECT_EQ(manifest.at("params").at("restDensity"), 900.0f);
			EXPECT_EQ(manifest.at("files"), nlohmann::json::array());
		}
	}

	// JSON text must retain enough information to validate either execution mode
	TEST(CheckpointManifestTests, ParsesAndValidatesMatchingSettings)
	{
		const Sim::SimParams params;
		for (const auto mode : { Export::CheckpointMode::Single, Export::CheckpointMode::Distributed })
		{
			SCOPED_TRACE(mode == Export::CheckpointMode::Single ? "single" : "distributed");
			const auto expected = MakeManifest(params, mode);
			const auto actual = Export::ParseCheckpointManifest(expected.dump());
			EXPECT_EQ(actual, expected);
			const auto info = Export::ValidateCheckpointManifest(actual, mode, params);

			EXPECT_EQ(info.step, 42u);
			EXPECT_DOUBLE_EQ(info.time, 1.25);
		}
	}

	// initial checkpoints and steps above the 32-bit range retain their exact values
	TEST(CheckpointManifestTests, AcceptsZeroAndWideStepValues)
	{
		for (const uint64_t step : { 0ull, 5'000'000'000ull })
		{
			auto manifest = MakeManifest();
			manifest["step"] = step;
			manifest["time"] = 0.0;
			const auto parsed = Export::ParseCheckpointManifest(manifest.dump());
			const auto info = Export::ValidateCheckpointManifest(parsed, Export::CheckpointMode::Single, {});

			EXPECT_EQ(info.step, step);
			EXPECT_DOUBLE_EQ(info.time, 0.0);
		}
	}

	// format tags are checked independently, including unsupported older versions
	TEST(CheckpointManifestTests, RejectsUnsupportedFormats)
	{
		const nlohmann::json changes{
			{ "format", "OtherCheckpoint" }, { "version", 2 }, { "stateFormat", "TSPSIM-1" } };
		for (const auto& [key, value] : changes.items())
		{
			SCOPED_TRACE(key);
			auto manifest = MakeManifest();
			manifest[key] = value;

			EXPECT_THROW(Export::ParseCheckpointManifest(manifest.dump()), std::runtime_error);
		}
	}

	// absent fields and malformed JSON fail at the parsing or validation boundary
	TEST(CheckpointManifestTests, RejectsMissingFieldsAndMalformedJson)
	{
		for (const char* text : { "", "{", "[]", "null" })
		{
			SCOPED_TRACE(text);
			EXPECT_THROW(Export::ParseCheckpointManifest(text), nlohmann::json::exception);
		}
		for (const char* key : { "format", "version", "stateFormat", "mode", "files", "params", "step", "time" })
		{
			SCOPED_TRACE(key);
			auto manifest = MakeManifest();
			manifest.erase(key);

			EXPECT_THROW({
				const auto parsed = Export::ParseCheckpointManifest(manifest.dump());
				Export::ValidateCheckpointManifest(parsed, Export::CheckpointMode::Single, {});
			}, nlohmann::json::exception);
		}
	}

	// a checkpoint cannot switch between single-process and distributed execution
	TEST(CheckpointManifestTests, RejectsExecutionModeChanges)
	{
		EXPECT_THROW(Export::ValidateCheckpointManifest(MakeManifest(), Export::CheckpointMode::Distributed, {}),
			std::runtime_error);
		EXPECT_THROW(Export::ValidateCheckpointManifest(MakeManifest({}, Export::CheckpointMode::Distributed),
			Export::CheckpointMode::Single, {}), std::runtime_error);
	}

	// metadata alone is not a complete checkpoint without an array of state files
	TEST(CheckpointManifestTests, RejectsMissingStateFileLists)
	{
		for (const auto& files : { nlohmann::json::array(), nlohmann::json::object(), nlohmann::json(nullptr) })
		{
			auto manifest = MakeManifest();
			manifest["files"] = files;

			EXPECT_THROW(Export::ValidateCheckpointManifest(manifest, Export::CheckpointMode::Single, {}),
				std::runtime_error);
		}
	}

	// physics and solver changes must remain incompatible in both execution modes
	TEST(CheckpointManifestTests, RejectsIncompatibleParameters)
	{
		const nlohmann::json changes{
			{ "timeStep", 0.002 }, { "restDensity", 900.0 }, { "massF", 0.002 },
			{ "gravity", { 1.0, 0.0, 0.0 } }, { "jacobiOmega", 0.75 },
			{ "fixedIters", 20 }, { "maxSolverIters", 200 },
			{ "neighborSearchMethod", "BruteForce" }, { "gridCellSizeRatio", 0.5 } };
		for (const auto mode : { Export::CheckpointMode::Single, Export::CheckpointMode::Distributed })
		{
			SCOPED_TRACE(mode == Export::CheckpointMode::Single ? "single" : "distributed");
			for (const auto& [key, value] : changes.items())
			{
				SCOPED_TRACE(key);
				auto manifest = MakeManifest({}, mode);
				ASSERT_NE(manifest["params"].at(key), value);
				manifest["params"][key] = value;

				EXPECT_THROW(Export::ValidateCheckpointManifest(manifest, mode, {}), std::runtime_error);
			}
		}
	}

	// restart output and diagnostic choices do not change the saved physical state
	TEST(CheckpointManifestTests, AcceptsOutputAndDiagnosticChanges)
	{
		Sim::SimParams params;
		params.maxSteps = 100;
		params.maxThreadsSim = 2;
		params.exportEnabled = false;
		params.saveFinalState = true;
		params.exportDir = "restart_output";
		params.exportInterval = 7;
		params.checkpointTime = 2.0;
		params.maxBenchRecs = 10;
		params.maxMemRecs = 20;
		params.maxStatRecs = 30;
		params.statGroupsEnabled.fill(true);
		params.surfaceNbrCountRatio = 0.5f;
		params.surfaceNbrCountThreshold = 18;
		for (const auto mode : { Export::CheckpointMode::Single, Export::CheckpointMode::Distributed })
		{
			SCOPED_TRACE(mode == Export::CheckpointMode::Single ? "single" : "distributed");
			EXPECT_NO_THROW(Export::ValidateCheckpointManifest(MakeManifest({}, mode), mode, params));
		}
	}

	// scene sampling settings can change because restart restores the existing particles
	TEST(CheckpointManifestTests, AcceptsSceneGenerationChanges)
	{
		Sim::SimParams params;
		params.jitterFracF = 0.1f;
		params.jitterSeedF = 123;
		params.jitterFracB = 0.2f;
		params.jitterSeedB = 456;
		params.surfaceSamplingMinDistanceRatioB = 0.7f;
		params.surfaceSamplingMinDistanceB = 0.007f;
		params.surfaceSamplingMaxDistanceRatioR = 1.1f;
		params.surfaceSamplingMaxDistanceR = 0.011f;
		params.surfaceSamplingCandidateMultiplier = 16;
		params.surfaceSamplingSeed = 789;

		EXPECT_NO_THROW(Export::ValidateCheckpointManifest(MakeManifest(), Export::CheckpointMode::Single, params));
	}

	// only distributed checkpoints permit a fresh decomposition and balancing history
	TEST(CheckpointManifestTests, AllowsLayoutChangesOnlyForDistributedRestarts)
	{
		const nlohmann::json changes{
			{ "domainDecompMethod", "AABB_Columns" }, { "distHaloMode", "DoubleHalo" },
			{ "distHaloRadius", 0.04 }, { "loadBalanceMetric", "RankOwnedParticleCount" },
			{ "loadBalanceTrigger", "Periodic" }, { "loadBalanceThreshold", 0.2 },
			{ "loadBalanceSustainSteps", 4 }, { "loadBalanceCooldownSteps", 30 },
			{ "loadBalanceIntervalSteps", 60 } };
		for (const auto& [key, value] : changes.items())
		{
			SCOPED_TRACE(key);
			auto manifest = MakeManifest();
			ASSERT_NE(manifest["params"].at(key), value);
			manifest["params"][key] = value;
			EXPECT_THROW(Export::ValidateCheckpointManifest(manifest, Export::CheckpointMode::Single, {}),
				std::runtime_error);
			manifest["mode"] = "distributed";
			EXPECT_NO_THROW(Export::ValidateCheckpointManifest(manifest, Export::CheckpointMode::Distributed, {}));
		}
	}

	// malformed parameter containers cannot be treated as compatible settings
	TEST(CheckpointManifestTests, RejectsMalformedParameters)
	{
		for (const auto& value : { nlohmann::json(nullptr), nlohmann::json::array(), nlohmann::json("params") })
		{
			auto manifest = MakeManifest();
			manifest["params"] = value;
			EXPECT_THROW(Export::ValidateCheckpointManifest(manifest, Export::CheckpointMode::Single, {}),
				std::runtime_error);
		}
	}

	// both manifest construction and restart validation reject invalid saved times
	TEST(CheckpointManifestTests, RejectsInvalidStepAndTime)
	{
		for (const double time : { -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() })
		{
			auto manifest = MakeManifest();
			manifest["time"] = time;
			EXPECT_THROW(Export::MakeCheckpointManifest(Export::CheckpointMode::Single, 42, time, {}), std::runtime_error);
			EXPECT_THROW(Export::ValidateCheckpointManifest(manifest, Export::CheckpointMode::Single, {}), std::runtime_error);
		}
		const uint64_t step = std::numeric_limits<uint64_t>::max();
		auto manifest = MakeManifest();
		manifest["step"] = step;
		EXPECT_THROW(Export::MakeCheckpointManifest(Export::CheckpointMode::Single, step, 1.25, {}), std::runtime_error);
		EXPECT_THROW(Export::ValidateCheckpointManifest(manifest, Export::CheckpointMode::Single, {}), std::runtime_error);
	}

	// a finite run limit must leave at least one step after the restored checkpoint
	TEST(CheckpointManifestTests, RequiresRunLimitBeyondSavedStep)
	{
		const auto manifest = MakeManifest();
		Sim::SimParams params;
		for (const int64_t limit : { 1, 41, 42 })
		{
			params.maxSteps = limit;
			EXPECT_THROW(Export::ValidateCheckpointManifest(manifest, Export::CheckpointMode::Single, params),
				std::runtime_error);
		}
		for (const int64_t limit : { -1, 0, 43 })
		{
			params.maxSteps = limit;
			EXPECT_NO_THROW(Export::ValidateCheckpointManifest(manifest, Export::CheckpointMode::Single, params));
		}
	}

	// saved time must be numeric rather than a coercible string or another JSON type
	TEST(CheckpointManifestTests, RejectsMalformedTimeValues)
	{
		for (const auto& value : { nlohmann::json("1.25"), nlohmann::json(nullptr),
			nlohmann::json(true), nlohmann::json::array() })
		{
			auto manifest = MakeManifest();
			manifest["time"] = value;
			EXPECT_THROW(Export::ValidateCheckpointManifest(manifest, Export::CheckpointMode::Single, {}),
				nlohmann::json::exception);
		}
	}

	// JSON integer conversion must preserve wide IDs without accepting coercions
	TEST(CheckpointManifestTests, ChecksUnsignedIntegerValues)
	{
		EXPECT_EQ(Export::CheckpointUint64(nlohmann::json(int64_t{ 0 })), 0u);
		EXPECT_EQ(Export::CheckpointUint64(nlohmann::json(int64_t{ 42 })), 42u);
		EXPECT_EQ(Export::CheckpointUint64(nlohmann::json(5'000'000'000ull)), 5'000'000'000ull);
		EXPECT_EQ(Export::CheckpointUint64(nlohmann::json(std::numeric_limits<uint64_t>::max())),
			std::numeric_limits<uint64_t>::max());
		for (const auto& value : { nlohmann::json(-1), nlohmann::json(1.5), nlohmann::json("42"),
			nlohmann::json(true), nlohmann::json(nullptr) })
		{
			SCOPED_TRACE(value.dump());
			EXPECT_THROW(Export::CheckpointUint64(value), std::runtime_error);
			auto manifest = MakeManifest();
			manifest["step"] = value;
			EXPECT_THROW(Export::ValidateCheckpointManifest(manifest, Export::CheckpointMode::Single, {}), std::runtime_error);
		}
	}

	class CheckpointFileTests : public testing::Test
	{
	protected:
		void SetUp() override
		{
			std::random_device random;
			const auto parent = std::filesystem::temp_directory_path();
			for (uint32_t attempt = 0; attempt < 16; attempt++)
			{
				const auto candidate = parent / ("tsp_checkpoint_tests_" + std::to_string(random()) + "_" + std::to_string(random()));
				if (!std::filesystem::create_directory(candidate)) { continue; }
				directory_ = candidate;
				Export::SaveOptions options;
				options.baseDir = directory_;
				serializer_ = std::make_unique<Export::SerializationMgr>(options);
				return;
			}
			FAIL() << "Could not create an isolated temporary directory";
		}

		void TearDown() override
		{
			if (directory_.empty()) { return; }
			std::error_code error;
			std::filesystem::remove_all(directory_, error);
			EXPECT_FALSE(error) << error.message();
		}

		void SaveState() const
		{
			Sync::UniversalSimState state;
			state.step = 42;
			state.time = 1.25;
			state.ResizeFluid(2);
			state.ResizeBoundary(1);
			state.ResizeRigid(3);
			state.ResizeRigidBodies(2);
			state.particlesF.id = { 101, 102 };
			state.particlesF.xPos = { 0.25f, -0.5f };
			state.particlesB.id = { 201 };
			state.particlesR.id = { 301, 302, 303 };
			state.particlesR.bodyId = { 0, 1, 0 };
			state.rigidBodies.id = { 401, 402 };
			state.rigidBodies.rigidParticleCount = { 2, 1 };
			serializer_->SaveUniversalSimState(state, "state.bin");
		}

		std::filesystem::path directory_;
		std::unique_ptr<Export::SerializationMgr> serializer_;
	};

	// completed publication makes the manifest readable and removes the temporary name
	TEST_F(CheckpointFileTests, PublishesAndReadsManifest)
	{
		const auto expected = MakeManifest();
		Export::PublishCheckpointManifest(directory_, expected);
		const auto text = Export::ReadCheckpointManifestText(directory_ / "manifest.json");

		EXPECT_EQ(Export::ParseCheckpointManifest(text), expected);
		EXPECT_FALSE(std::filesystem::exists(directory_ / "manifest.json.tmp"));
	}

	// a second publication must preserve the checkpoint already present on disk
	TEST_F(CheckpointFileTests, RejectsOverwritingManifest)
	{
		auto manifest = MakeManifest();
		Export::PublishCheckpointManifest(directory_, manifest);
		const auto expected = Export::ReadCheckpointManifestText(directory_ / "manifest.json");
		manifest["step"] = 43;

		EXPECT_THROW(Export::PublishCheckpointManifest(directory_, manifest), std::runtime_error);
		EXPECT_EQ(Export::ReadCheckpointManifestText(directory_ / "manifest.json"), expected);
	}

	// incomplete names are rejected even when they contain a valid manifest
	TEST_F(CheckpointFileTests, RejectsMissingAndUnpublishedManifests)
	{
		EXPECT_THROW(Export::ReadCheckpointManifestText(directory_ / "manifest.json"), std::runtime_error);
		Export::PublishCheckpointManifest(directory_, MakeManifest());
		std::filesystem::copy_file(directory_ / "manifest.json", directory_ / "manifest.json.tmp");

		EXPECT_THROW(Export::ReadCheckpointManifestText(directory_ / "manifest.json.tmp"), std::runtime_error);
	}

	// failed writes must not expose a completed checkpoint
	TEST_F(CheckpointFileTests, RejectsPublishingIntoMissingDirectory)
	{
		const auto missing = directory_ / "missing";

		EXPECT_THROW(Export::PublishCheckpointManifest(missing, MakeManifest()), std::runtime_error);
		EXPECT_FALSE(std::filesystem::exists(missing / "manifest.json"));
	}

	// repeated saves at the same step reserve new directories without replacing old checkpoints
	TEST_F(CheckpointFileTests, CreatesDistinctCheckpointDirectories)
	{
		const auto parent = directory_ / "checkpoints";
		const auto first = Export::CreateCheckpointDirectory(parent, 42);
		Export::PublishCheckpointManifest(first, MakeManifest());
		const auto second = Export::CreateCheckpointDirectory(parent, 42);
		const auto third = Export::CreateCheckpointDirectory(parent, 42);

		EXPECT_EQ(first, parent / "checkpoint_step_42");
		EXPECT_EQ(second, parent / "checkpoint_step_42_1");
		EXPECT_EQ(third, parent / "checkpoint_step_42_2");
		EXPECT_TRUE(std::filesystem::is_directory(second));
		EXPECT_TRUE(std::filesystem::is_directory(third));
		EXPECT_EQ(Export::ParseCheckpointManifest(Export::ReadCheckpointManifestText(first / "manifest.json")), MakeManifest());
	}

	// publication and loading agree on the saved metadata and mixed population counts
	TEST_F(CheckpointFileTests, LoadsStateFromPublishedManifest)
	{
		SaveState();
		ASSERT_EQ(std::filesystem::file_size(directory_ / "state.bin"), 604u);
		Export::PublishCheckpointManifest(directory_, MakeManifest());
		const auto manifest = Export::ParseCheckpointManifest(
			Export::ReadCheckpointManifestText(directory_ / "manifest.json"));
		const auto info = Export::ValidateCheckpointManifest(manifest, Export::CheckpointMode::Single, {});
		Sync::UniversalSimState actual;
		Export::SaveOptions options;
		options.baseDir = directory_ / "unrelated";
		const Export::SerializationMgr serializer(options);
		Export::LoadCheckpointState(serializer, actual, directory_, manifest.at("files").at(0), info);

		EXPECT_EQ(actual.step, 42u);
		EXPECT_DOUBLE_EQ(actual.time, 1.25);
		EXPECT_EQ(actual.particlesF.N_, 2u);
		EXPECT_EQ(actual.particlesB.N_, 1u);
		EXPECT_EQ(actual.particlesR.N_, 3u);
		EXPECT_EQ(actual.rigidBodies.N_, 2u);
		EXPECT_EQ(actual.particlesF.id, (std::vector<uint64_t>{ 101, 102 }));
		EXPECT_EQ(actual.particlesF.xPos, (std::vector<float>{ 0.25f, -0.5f }));
		EXPECT_EQ(actual.particlesR.bodyId, (std::vector<uint32_t>{ 0, 1, 0 }));
	}

	// manifest lengths and each population count must agree with the binary file
	TEST_F(CheckpointFileTests, RejectsMismatchedFileMetadata)
	{
		SaveState();
		for (const char* key : { "bytes", "fluidCount", "boundaryCount", "rigidCount", "bodyCount" })
		{
			SCOPED_TRACE(key);
			auto fileInfo = MakeFileInfo();
			fileInfo[key] = fileInfo.at(key).get<uint64_t>() + 1;
			Sync::UniversalSimState actual;

			EXPECT_THROW(Export::LoadCheckpointState(*serializer_, actual, directory_, fileInfo, { 42, 1.25 }),
				std::runtime_error);
		}
	}

	// matching sizes and counts do not permit state from a different checkpoint time
	TEST_F(CheckpointFileTests, RejectsMismatchedStateTime)
	{
		SaveState();
		for (const auto info : { Export::CheckpointInfo{ 43, 1.25 }, Export::CheckpointInfo{ 42, 1.5 } })
		{
			Sync::UniversalSimState actual;
			EXPECT_THROW(Export::LoadCheckpointState(*serializer_, actual, directory_, MakeFileInfo(), info),
				std::runtime_error);
		}
	}

	// metadata cannot redirect state loading through an absolute path or directory components
	TEST_F(CheckpointFileTests, RejectsInvalidStateFileNames)
	{
		SaveState();
		std::filesystem::create_directory(directory_ / "nested");
		for (const auto& name : { std::string(), std::string("."), std::string(".."),
			std::string("nested/../state.bin"), (directory_ / "state.bin").string() })
		{
			SCOPED_TRACE(name);
			auto fileInfo = MakeFileInfo();
			fileInfo["file"] = name;
			Sync::UniversalSimState actual;

			EXPECT_THROW(Export::LoadCheckpointState(*serializer_, actual, directory_, fileInfo, { 42, 1.25 }),
				std::runtime_error);
		}
	}

	// missing state files and incomplete metadata cannot produce a restored state
	TEST_F(CheckpointFileTests, RejectsMissingFilesAndMetadata)
	{
		Sync::UniversalSimState actual;
		EXPECT_THROW(Export::LoadCheckpointState(*serializer_, actual, directory_, MakeFileInfo(), { 42, 1.25 }),
			std::filesystem::filesystem_error);
		SaveState();
		for (const char* key : { "file", "bytes", "fluidCount", "boundaryCount", "rigidCount", "bodyCount" })
		{
			SCOPED_TRACE(key);
			auto fileInfo = MakeFileInfo();
			fileInfo.erase(key);

			EXPECT_THROW(Export::LoadCheckpointState(*serializer_, actual, directory_, fileInfo, { 42, 1.25 }),
				nlohmann::json::exception);
		}
	}
}
