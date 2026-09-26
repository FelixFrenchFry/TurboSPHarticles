// checkpoint.cpp

// external
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

// internal
#include "checkpoint.h"



namespace Export
{
	// manifest helpers
	static const char* GetCheckpointModeName(const CheckpointMode mode)
	{
		return mode == CheckpointMode::Single ? "single" : "distributed";
	}

	static nlohmann::json GetCheckpointComparisonParams(nlohmann::json params, const CheckpointMode mode)
	{
		if (!params.is_object()) { throw std::runtime_error("Invalid checkpoint parameters."); }

		// exclude output, diagnostics, run limits and scene generation settings
		for (const char* key : { "maxSteps", "maxThreadsSim", "exportEnabled",
			"saveFinalState", "exportDir", "exportInterval", "checkpointTime", "maxBenchRecs", "timeUnit", "maxMemRecs",
			"byteUnit", "maxStatRecs", "statGroupsEnabled", "progressReportMode",
			"surfaceNbrCountRatio", "surfaceNbrCountThreshold",
			"jitterFracF", "jitterSeedF", "jitterFracB", "jitterSeedB",
			"surfaceSamplingMinDistanceRatio", "surfaceSamplingMaxDistanceRatio",
			"surfaceSamplingCoverageProbeSpacingRatio", "surfaceSamplingMinDistance",
			"surfaceSamplingMaxDistance", "surfaceSamplingCoverageProbeSpacing",
			"surfaceSamplingMinDistanceRatioB", "surfaceSamplingMaxDistanceRatioB",
			"surfaceSamplingCoverageProbeSpacingRatioB", "surfaceSamplingMinDistanceB",
			"surfaceSamplingMaxDistanceB", "surfaceSamplingCoverageProbeSpacingB",
			"surfaceSamplingMinDistanceRatioR", "surfaceSamplingMaxDistanceRatioR",
			"surfaceSamplingCoverageProbeSpacingRatioR", "surfaceSamplingMinDistanceR",
			"surfaceSamplingMaxDistanceR", "surfaceSamplingCoverageProbeSpacingR",
			"surfaceSamplingCandidateMultiplier", "surfaceSamplingMaxTrialsPerCell",
			"surfaceSamplingSeed" })
		{
			params.erase(key);
		}
		if (mode == CheckpointMode::Distributed)
		{
			// a shared checkpoint is assigned to a new layout with fresh balancing history
			for (const char* key : { "domainDecompMethod", "distHaloMode", "distHaloRadius",
				"loadBalanceMetric", "loadBalanceTrigger", "loadBalanceWeightModel",
				"loadBalanceThreshold", "loadBalanceSustainSteps", "loadBalanceCooldownSteps",
				"loadBalanceIntervalSteps" })
			{
				params.erase(key);
			}
		}
		return params;
	}

	static void ValidateCheckpointTime(
		const uint64_t step,
		const double time)
	{
		if (step == std::numeric_limits<uint64_t>::max() || !std::isfinite(time) || time < 0.0)
		{
			throw std::runtime_error("Invalid checkpoint time/step.");
		}
	}

	// --- [ JSON field conversion ] -----------------------------------------------------------------

	uint64_t CheckpointUint64(const nlohmann::json& value)
	{
		if (!value.is_number_unsigned() &&
			(!value.is_number_integer() || value.get<int64_t>() < 0))
		{
			throw std::runtime_error("Expected a non-negative integer in checkpoint manifest.");
		}
		return value.get<uint64_t>();
	}

	nlohmann::json CheckpointVec3ToJson(const glm::vec3& value)
	{
		if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
		{
			throw std::runtime_error("Non-finite checkpoint vector.");
		}
		return { value.x, value.y, value.z };
	}

	glm::vec3 CheckpointVec3FromJson(const nlohmann::json& value)
	{
		if (!value.is_array() || value.size() != 3)
		{
			throw std::runtime_error("Invalid checkpoint vector.");
		}
		glm::vec3 result{};
		for (int axis = 0; axis < 3; axis++)
		{
			result[axis] = value[axis].get<float>();
			if (!std::isfinite(result[axis])) { throw std::runtime_error("Non-finite checkpoint vector."); }
		}
		return result;
	}

	nlohmann::json CheckpointBoundsToJson(
		const glm::vec3& min,
		const glm::vec3& max)
	{
		return { CheckpointVec3ToJson(min), CheckpointVec3ToJson(max) };
	}

	void CheckpointBoundsFromJson(
		const nlohmann::json& value,
		glm::vec3& min,
		glm::vec3& max)
	{
		if (!value.is_array() || value.size() != 2)
		{
			throw std::runtime_error("Invalid checkpoint domain bounds.");
		}
		min = CheckpointVec3FromJson(value[0]);
		max = CheckpointVec3FromJson(value[1]);
		for (int axis = 0; axis < 3; axis++)
		{
			if (min[axis] >= max[axis]) { throw std::runtime_error("Invalid checkpoint domain extent."); }
		}
	}

	// --- [ manifest metadata ] ---------------------------------------------------------------------

	nlohmann::json MakeCheckpointManifest(
		const CheckpointMode mode,
		const uint64_t step,
		const double time,
		const Sim::SimParams& params)
	{
		ValidateCheckpointTime(step, time);

		nlohmann::json manifest = nlohmann::json::object();
		manifest["format"] = "TSPCheckpoint";
		manifest["version"] = 3;
		manifest["mode"] = GetCheckpointModeName(mode);
		manifest["stateFormat"] = "TSPSIM-2";
		manifest["step"] = step;
		manifest["time"] = time;
		manifest["params"] = Sim::ToJson(params);
		manifest["files"] = nlohmann::json::array();
		return manifest;
	}

	nlohmann::json ParseCheckpointManifest(const std::string& text)
	{
		auto manifest = nlohmann::json::parse(text);
		if (manifest.at("format") != "TSPCheckpoint" ||
			manifest.at("version") != 3 ||
			manifest.at("stateFormat") != "TSPSIM-2")
		{
			throw std::runtime_error("Unsupported checkpoint format.");
		}
		return manifest;
	}

	CheckpointInfo ValidateCheckpointManifest(
		const nlohmann::json& manifest,
		const CheckpointMode mode,
		const Sim::SimParams& params)
	{
		if (manifest.at("mode") != GetCheckpointModeName(mode))
		{
			throw std::runtime_error("Checkpoint restart requires the original execution mode.");
		}
		if (!manifest.at("files").is_array() || manifest.at("files").empty())
		{
			throw std::runtime_error("Checkpoint contains no state files.");
		}
		const auto savedParams = GetCheckpointComparisonParams(manifest.at("params"), mode);
		const auto currentParams = GetCheckpointComparisonParams(Sim::ToJson(params), mode);
		if (savedParams != currentParams)
		{
			std::string message = "Checkpoint physics, solver, or decomposition parameters differ from this run.";
			for (const auto& change : nlohmann::json::diff(savedParams, currentParams))
			{
				const auto key = change.at("path").get<std::string>();
				const nlohmann::json::json_pointer path(key);
				message += "\n" + key + ": saved=" +
					(savedParams.contains(path) ? savedParams.at(path).dump() : "<missing>") +
					", current=" + (currentParams.contains(path) ? currentParams.at(path).dump() : "<missing>");
			}
			throw std::runtime_error(message);
		}

		CheckpointInfo info{};
		info.step = CheckpointUint64(manifest.at("step"));
		info.time = manifest.at("time").get<double>();
		ValidateCheckpointTime(info.step, info.time);
		if (params.maxSteps > 0 && static_cast<uint64_t>(params.maxSteps) <= info.step)
		{
			throw std::runtime_error("maxSteps must exceed the saved checkpoint step.");
		}
		return info;
	}

	// --- [ checkpoint files ] ----------------------------------------------------------------------

	std::filesystem::path CreateCheckpointDirectory(
		const std::filesystem::path& parent,
		const uint64_t step)
	{
		const auto absoluteParent = std::filesystem::absolute(parent);
		std::filesystem::create_directories(absoluteParent);
		const std::string name = "checkpoint_step_" + std::to_string(step);
		for (uint64_t suffix = 0; ; suffix++)
		{
			const auto candidate = absoluteParent / (name + (suffix == 0 ? "" : "_" + std::to_string(suffix)));
			if (std::filesystem::create_directory(candidate)) { return candidate; }
		}
	}

	std::string ReadCheckpointManifestText(const std::filesystem::path& path)
	{
		if (path.filename() != "manifest.json")
		{
			throw std::runtime_error("Restart requires a completed checkpoint's manifest.json.");
		}
		std::ifstream file(path, std::ios::binary);
		if (!file) { throw std::runtime_error("Cannot open checkpoint manifest."); }
		std::string text{ std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
		if (file.bad()) { throw std::runtime_error("Failed to read checkpoint manifest."); }
		return text;
	}

	void PublishCheckpointManifest(
		const std::filesystem::path& directory,
		const nlohmann::json& manifest)
	{
		const auto path = directory / "manifest.json";
		if (std::filesystem::exists(path)) { throw std::runtime_error("Checkpoint manifest already exists."); }
		const auto tmpPath = directory / "manifest.json.tmp";
		std::ofstream file(tmpPath, std::ios::binary | std::ios::trunc);
		file << manifest.dump(2) << '\n';
		file.close();
		if (!file) { throw std::runtime_error("Failed to finish checkpoint manifest."); }
		std::filesystem::rename(tmpPath, path);
	}

	void LoadCheckpointState(
		const SerializationMgr& serializer,
		Sync::UniversalSimState& state,
		const std::filesystem::path& directory,
		const nlohmann::json& fileInfo,
		const CheckpointInfo& info)
	{
		const std::filesystem::path fileName = fileInfo.at("file").get<std::string>();
		if (fileName.empty() || fileName.has_parent_path() || fileName == "." || fileName == "..")
		{
			throw std::runtime_error("Invalid checkpoint state file name.");
		}
		const auto path = directory / fileName;
		if (std::filesystem::file_size(path) != CheckpointUint64(fileInfo.at("bytes")))
		{
			throw std::runtime_error("Checkpoint state file length differs from the manifest.");
		}
		serializer.LoadUniversalSimState(state, std::filesystem::absolute(path).string());
		if (state.step != info.step || state.time != info.time ||
			state.particlesF.N_ != CheckpointUint64(fileInfo.at("fluidCount")) ||
			state.particlesB.N_ != CheckpointUint64(fileInfo.at("boundaryCount")) ||
			state.particlesR.N_ != CheckpointUint64(fileInfo.at("rigidCount")) ||
			state.rigidBodies.N_ != CheckpointUint64(fileInfo.at("bodyCount")))
		{
			throw std::runtime_error("Checkpoint state differs from the manifest.");
		}
	}

} // namespace Export
