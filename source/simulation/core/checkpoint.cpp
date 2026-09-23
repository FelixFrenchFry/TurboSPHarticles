// checkpoint.cpp

// external
#include <filesystem>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <utility>
#include <vector>

// internal
#include "source/eval/serial/checkpoint.h"
#include "simulation.h"



using namespace Sim;

namespace
{
	// writes the boundary boxes used by visualization
	nlohmann::json CheckpointBoundaryBoxesToJson(
		const std::vector<Scene::OBB>& boxes)
	{
		auto result = nlohmann::json::array();
		for (const auto& box : boxes)
		{
			nlohmann::json item = nlohmann::json::object();
			item["origin"] = Export::CheckpointVec3ToJson(box.origin);
			item["uDir"] = Export::CheckpointVec3ToJson(box.uHat);
			item["vDir"] = Export::CheckpointVec3ToJson(box.vHat);
			item["wDir"] = Export::CheckpointVec3ToJson(box.wHat);
			item["lengths"] = Export::CheckpointVec3ToJson({ box.uLen, box.vLen, box.wLen });
			result.push_back(std::move(item));
		}
		return result;
	}

	// reads boundary box geometry without regenerating the scenario
	std::vector<Scene::OBB> CheckpointBoundaryBoxesFromJson(
		const nlohmann::json& value)
	{
		if (!value.is_array())
		{
			throw std::runtime_error("Invalid checkpoint boundary boxes.");
		}

		std::vector<Scene::OBB> boxes{};
		for (const auto& item : value)
		{
			const glm::vec3 lengths = Export::CheckpointVec3FromJson(item.at("lengths"));
			if (lengths.x < 0.0f || lengths.y < 0.0f || lengths.z < 0.0f)
			{
				throw std::runtime_error("Invalid checkpoint boundary box lengths.");
			}

			Scene::OBB box{};
			box.origin = Export::CheckpointVec3FromJson(item.at("origin"));
			box.uHat = Export::CheckpointVec3FromJson(item.at("uDir"));
			box.vHat = Export::CheckpointVec3FromJson(item.at("vDir"));
			box.wHat = Export::CheckpointVec3FromJson(item.at("wDir"));
			box.uLen = lengths.x;
			box.vLen = lengths.y;
			box.wLen = lengths.z;
			boxes.push_back(box);
		}
		return boxes;
	}

} // namespace anonymous

void SimulationMgr::SaveCheckpoint(
	const SimParams& params)
{
	if (!simBackend_)
	{
		throw std::runtime_error("Cannot save checkpoint without a simulation backend.");
	}

	// transfer the final backend state into the universal state
	simBackend_->PushToUniversalSimState(universalSimState_, true);
	simBackend_.reset();
	universalSimState_.step = simStep_;
	universalSimState_.time = simTime_;

	// build simulation and scene metadata
	auto manifest = Export::MakeCheckpointManifest(
		Export::CheckpointMode::Single, simStep_, simTime_, params);
	manifest["globalBounds"] = domainBounds_
		? Export::CheckpointBoundsToJson(domainBounds_->min, domainBounds_->max)
		: nlohmann::json(nullptr);
	manifest["boundaryBoxes"] = CheckpointBoundaryBoxesToJson(boundaryBoxes_);

	// write the full state into a new checkpoint directory
	const auto checkpointDir = Export::CreateCheckpointDirectory(
		serializationMgr_.BaseDir() / params.exportDir, simStep_);
	const auto statePath = checkpointDir / "state.bin";
	serializationMgr_.SaveUniversalSimState(universalSimState_, statePath.string());

	nlohmann::json fileInfo = nlohmann::json::object();
	fileInfo["file"] = "state.bin";
	fileInfo["fluidCount"] = universalSimState_.particlesF.N_;
	fileInfo["boundaryCount"] = universalSimState_.particlesB.N_;
	fileInfo["rigidCount"] = universalSimState_.particlesR.N_;
	fileInfo["bodyCount"] = universalSimState_.rigidBodies.N_;
	fileInfo["bytes"] = std::filesystem::file_size(statePath);
	manifest["files"].push_back(std::move(fileInfo));

	// publish the manifest after the state file has finished writing
	Export::PublishCheckpointManifest(checkpointDir, manifest);
	SPDLOG_INFO("Saved checkpoint '{}'", (checkpointDir / "manifest.json").string());
}

void SimulationMgr::LoadCheckpoint(
	const std::string& manifestPath)
{
	// read metadata and check the current simulation settings
	const auto path = std::filesystem::absolute(manifestPath);
	const auto manifestText = Export::ReadCheckpointManifestText(path);
	const auto manifest = Export::ParseCheckpointManifest(manifestText);
	const auto info = Export::ValidateCheckpointManifest(
		manifest, Export::CheckpointMode::Single, simParams_);

	const auto& files = manifest.at("files");
	if (files.size() != 1 || files[0].at("file") != "state.bin")
	{
		throw std::runtime_error("Single-process checkpoint requires one state.bin file.");
	}

	// restore scene metadata and particle state into temporary storage
	auto boxes = CheckpointBoundaryBoxesFromJson(manifest.at("boundaryBoxes"));
	std::optional<DomainBounds> domainBounds{};
	if (!manifest.at("globalBounds").is_null())
	{
		DomainBounds bounds{};
		Export::CheckpointBoundsFromJson(manifest.at("globalBounds"), bounds.min, bounds.max);
		domainBounds = bounds;
	}
	Sync::UniversalSimState state{};
	Export::LoadCheckpointState(serializationMgr_, state, path.parent_path(), files[0], info);

	// replace the simulation state after all checkpoint data has been validated
	universalSimState_ = std::move(state);
	boundaryBoxes_ = std::move(boxes);
	domainBounds_ = domainBounds;
	simScenario_.reset();
	simStep_ = info.step;
	simTime_ = info.time;
	serializationMgr_.ResetExportDirState();
	SPDLOG_INFO("Restored checkpoint at step {} (time {})", simStep_, simTime_);
}
