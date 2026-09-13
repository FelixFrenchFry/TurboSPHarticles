// checkpoint.h
#pragma once

// external
#include <cstdint>
#include <filesystem>
#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>
#include <string>

// internal
#include "source/simulation/core/parameters_sim.h"
#include "serialization.h"



namespace Export
{
	enum class CheckpointMode
	{
		Single,
		Distributed
	};

	struct CheckpointInfo
	{
		uint64_t step = 0;
		double time = 0.0;
	};

	// JSON field conversion
	uint64_t CheckpointUint64(const nlohmann::json& value);
	nlohmann::json CheckpointVec3ToJson(const glm::vec3& value);
	glm::vec3 CheckpointVec3FromJson(const nlohmann::json& value);
	nlohmann::json CheckpointBoundsToJson(
		const glm::vec3& min,
		const glm::vec3& max);
	void CheckpointBoundsFromJson(
		const nlohmann::json& value,
		glm::vec3& min,
		glm::vec3& max);

	// builds common checkpoint metadata from the current simulation parameters
	nlohmann::json MakeCheckpointManifest(
		CheckpointMode mode,
		uint64_t step,
		double time,
		const Sim::SimParams& params);

	// reads the manifest JSON and checks its format version
	nlohmann::json ParseCheckpointManifest(const std::string& text);

	// checks restart settings and returns the saved simulation step and time
	CheckpointInfo ValidateCheckpointManifest(
		const nlohmann::json& manifest,
		CheckpointMode mode,
		const Sim::SimParams& params);

	// creates a checkpoint directory, adding a suffix if the name already exists
	std::filesystem::path CreateCheckpointDirectory(
		const std::filesystem::path& parent,
		uint64_t step);

	// reads manifest.json for local parsing or MPI broadcast
	std::string ReadCheckpointManifestText(const std::filesystem::path& path);

	// writes manifest.json through a temporary file after all state files are complete
	void PublishCheckpointManifest(
		const std::filesystem::path& directory,
		const nlohmann::json& manifest);

	// loads one binary state file and checks its size, counts, step and time
	void LoadCheckpointState(
		const SerializationMgr& serializer,
		Sync::UniversalSimState& state,
		const std::filesystem::path& directory,
		const nlohmann::json& fileInfo,
		const CheckpointInfo& info);

} // namespace Export
