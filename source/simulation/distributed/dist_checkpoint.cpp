// dist_checkpoint.cpp

// external
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

// internal
#include "source/eval/serial/checkpoint.h"
#include "dist_helpers.h"
#include "dist_simulation.h"



using namespace Dist;

namespace
{
	// packed rank metadata
	// (fluid, boundary, and rigid particle counts, rigid body count, then file size)
	constexpr size_t N_rankInfoFields = 5;

	std::string CheckpointRankFileName(const int rank)
	{
		return "rank_" + std::to_string(rank) + ".bin";
	}

	nlohmann::json CheckpointBoundsToJson(const AABB& bounds)
	{
		return Export::CheckpointBoundsToJson(bounds.min, bounds.max);
	}

	AABB CheckpointBoundsFromJson(const nlohmann::json& value)
	{
		AABB bounds{};
		Export::CheckpointBoundsFromJson(value, bounds.min, bounds.max);
		return bounds;
	}

	void RestoreCheckpointParticleMetadata(
		DistParticleState& state,
		const uint32_t N_owned)
	{
		state.Clear();
		state.N_owned = N_owned;

		// mark all restored particles for routing, including those that crossed a subdomain wall
		state.particleClasses.assign(N_owned, DistParticleClass::Outer);
		state.particleLocalIdx_Owned.resize(N_owned);
		std::iota(state.particleLocalIdx_Owned.begin(), state.particleLocalIdx_Owned.end(), 0u);
		state.particleLocalIdx_Outer = state.particleLocalIdx_Owned;
	}

	RebalanceTriggerState CheckpointTriggerStateFromJson(const nlohmann::json& value)
	{
		const uint64_t badStepCount = Export::CheckpointUint64(value.at("badStepCount"));
		const uint64_t cooldownRemaining = Export::CheckpointUint64(value.at("cooldownRemaining"));
		if (badStepCount > std::numeric_limits<uint32_t>::max() ||
			cooldownRemaining > std::numeric_limits<uint32_t>::max())
		{
			throw std::runtime_error("Checkpoint load balance counters exceed the 32-bit limit.");
		}
		return { static_cast<uint32_t>(badStepCount), static_cast<uint32_t>(cooldownRemaining) };
	}

} // namespace anonymous

bool DistSimulationMgr::SaveCheckpoint(
	const Sim::SimParams& params)
{
	bool localOk = true;
	std::string directory{};
	std::vector<uint64_t> rankInfo{};

	// validate owned particles and allocate the rank metadata buffer
	try
	{
		localOk = ValidateRankLocalSimState(true);
		rankInfo.resize(static_cast<size_t>(mpiContext_.Size()) * N_rankInfoFields, 0);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to prepare checkpoint: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }
	if (!ValidateGlobalOwnedParticleCounts()) { return false; }

	// create one checkpoint directory and share its path with every rank
	if (mpiContext_.IsRoot())
	{
		try
		{
			directory = Export::CreateCheckpointDirectory(
				serializationMgr_.BaseDir() / params.exportDir, simStep_).string();
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR("Failed to create distributed checkpoint directory: {}", ex.what());
			localOk = false;
		}
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }
	mpiContext_.BroadcastString(directory, mpiContext_.RootRank());

	try
	{
		// remove halos from the validated [ owned ][ halo ] state and write this rank's file
		localSimState_.particlesF.resizeAll(particleStateF_.N_owned);
		localSimState_.particlesB.resizeAll(particleStateB_.N_owned);
		const auto path = std::filesystem::path(directory) / CheckpointRankFileName(mpiContext_.Rank());
		serializationMgr_.SaveUniversalSimState(localSimState_, path.string());

		const size_t offset = static_cast<size_t>(mpiContext_.Rank()) * N_rankInfoFields;
		rankInfo[offset] = localSimState_.particlesF.N_;
		rankInfo[offset + 1] = localSimState_.particlesB.N_;
		rankInfo[offset + 2] = localSimState_.particlesR.N_;
		rankInfo[offset + 3] = localSimState_.rigidBodies.N_;
		rankInfo[offset + 4] = std::filesystem::file_size(path);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to write checkpoint rank file: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	// collect file sizes and particle counts after every rank file has finished writing
	mpiContext_.AllReduceSumUint64s(rankInfo);

	// publish the common manifest with the saved domain decomposition
	if (mpiContext_.IsRoot())
	{
		try
		{
			auto manifest = Export::MakeCheckpointManifest(
				Export::CheckpointMode::Distributed, simStep_, simTime_, params);
			manifest["rankCount"] = mpiContext_.Size();
			manifest["globalFluidCount"] = N_expectedGlobalF_;
			manifest["globalBoundaryCount"] = N_expectedGlobalB_;
			manifest["fluidIdChecksum"] = expectedOwnedIdChecksumF_;
			manifest["boundaryIdChecksum"] = expectedOwnedIdChecksumB_;
			manifest["globalBounds"] = CheckpointBoundsToJson(domainDecomp_->GlobalExtentAABB());

			// save root's trigger history alongside the current subdomain boundaries
			const auto triggerState = loadBalanceMgr_.GetTriggerState();
			manifest["loadBalanceState"]["badStepCount"] = triggerState.badStepCount;
			manifest["loadBalanceState"]["cooldownRemaining"] = triggerState.cooldownRemaining;

			for (int rank = 0; rank < mpiContext_.Size(); rank++)
			{
				const size_t offset = static_cast<size_t>(rank) * N_rankInfoFields;
				const auto path = std::filesystem::path(directory) / CheckpointRankFileName(rank);
				std::error_code error{};
				const auto fileSize = std::filesystem::file_size(path, error);
				if (error || fileSize != rankInfo[offset + 4])
				{
					throw std::runtime_error("Checkpoint rank file '" + path.string() +
						"' is missing, inaccessible, or has a different size on root. "
						"Distributed checkpoints require a shared output directory.");
				}
				if (rankInfo[offset + 2] != localSimState_.particlesR.N_ ||
					rankInfo[offset + 3] != localSimState_.rigidBodies.N_)
				{
					throw std::runtime_error("Checkpoint rigid counts differ across ranks.");
				}

				nlohmann::json fileInfo = nlohmann::json::object();
				fileInfo["file"] = CheckpointRankFileName(rank);
				fileInfo["fluidCount"] = rankInfo[offset];
				fileInfo["boundaryCount"] = rankInfo[offset + 1];
				fileInfo["rigidCount"] = rankInfo[offset + 2];
				fileInfo["bodyCount"] = rankInfo[offset + 3];
				fileInfo["bytes"] = rankInfo[offset + 4];
				fileInfo["ownedBounds"] = CheckpointBoundsToJson(domainDecomp_->OwnedExtentAABB(rank));
				manifest["files"].push_back(std::move(fileInfo));
			}
			Export::PublishCheckpointManifest(directory, manifest);
			SPDLOG_INFO("Saved distributed checkpoint '{}'",
				(std::filesystem::path(directory) / "manifest.json").string());
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR("Failed to publish distributed checkpoint manifest: {}", ex.what());
			localOk = false;
		}
	}
	return mpiContext_.AllTrue(localOk);
}

bool DistSimulationMgr::LoadCheckpoint(
	const std::string& manifestPath)
{
	bool localOk = true;
	std::string directory{};
	std::string manifestText{};

	// read the manifest on root and broadcast it with the checkpoint directory
	if (mpiContext_.IsRoot())
	{
		try
		{
			const auto path = std::filesystem::absolute(manifestPath);
			manifestText = Export::ReadCheckpointManifestText(path);
			directory = path.parent_path().string();
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR("Failed to read checkpoint '{}': {}", manifestPath, ex.what());
			localOk = false;
		}
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }
	mpiContext_.BroadcastString(directory, mpiContext_.RootRank());
	mpiContext_.BroadcastString(manifestText, mpiContext_.RootRank());

	try
	{
		// check restart settings and the saved rank layout
		const auto manifest = Export::ParseCheckpointManifest(manifestText);
		const auto info = Export::ValidateCheckpointManifest(
			manifest, Export::CheckpointMode::Distributed, simParams_);
		const auto& rankFiles = manifest.at("files");
		if (Export::CheckpointUint64(manifest.at("rankCount")) != static_cast<uint64_t>(mpiContext_.Size()) ||
			!rankFiles.is_array() || rankFiles.size() != static_cast<size_t>(mpiContext_.Size()))
		{
			throw std::runtime_error("Checkpoint restart requires the original MPI rank count.");
		}

		// restore trigger history after policy configuration and before the first resumed step
		RebalanceTriggerState triggerState{};
		if (const auto it = manifest.find("loadBalanceState"); it != manifest.end())
		{
			triggerState = CheckpointTriggerStateFromJson(*it);
		}
		else if (loadBalanceMgr_.Enabled() &&
			simParams_.loadBalanceTrigger == Base::LoadBalanceTrigger::SustainedImbalance)
		{
			throw std::runtime_error("Checkpoint is missing load-balancing trigger state.");
		}
		loadBalanceMgr_.RestoreTriggerState(triggerState);

		simStep_ = info.step;
		simTime_ = info.time;
		N_expectedGlobalF_ = Export::CheckpointUint64(manifest.at("globalFluidCount"));
		N_expectedGlobalB_ = Export::CheckpointUint64(manifest.at("globalBoundaryCount"));
		expectedOwnedIdChecksumF_ = Export::CheckpointUint64(manifest.at("fluidIdChecksum"));
		expectedOwnedIdChecksumB_ = Export::CheckpointUint64(manifest.at("boundaryIdChecksum"));

		// restore subdomain boundaries before loading rank-local particle state
		std::vector<AABB> ownedBounds{};
		for (int rank = 0; rank < mpiContext_.Size(); rank++)
		{
			if (rankFiles[rank].at("file") != CheckpointRankFileName(rank) ||
				rankFiles[rank].at("rigidCount") != rankFiles[0].at("rigidCount") ||
				rankFiles[rank].at("bodyCount") != rankFiles[0].at("bodyCount"))
			{
				throw std::runtime_error("Inconsistent checkpoint rank metadata.");
			}
			ownedBounds.push_back(CheckpointBoundsFromJson(rankFiles[rank].at("ownedBounds")));
		}
		const DomainDecompInput input
		{
			.globalBounds = CheckpointBoundsFromJson(manifest.at("globalBounds")),
			.haloRadius = simParams_.distHaloRadius,
			.N_ranks = mpiContext_.Size()
		};
		domainDecomp_->Restore(input, ownedBounds);
		if (!domainDecomp_->IsValidForRankCount(mpiContext_.Size()))
		{
			throw std::runtime_error("Invalid restored checkpoint decomposition.");
		}
		simBackend_->SetSimDomainBounds(input.globalBounds.min, input.globalBounds.max);

		// load this rank's file and rebuild ownership metadata for startup redistribution
		const auto& rankInfo = rankFiles[mpiContext_.Rank()];
		Export::LoadCheckpointState(serializationMgr_, localSimState_, directory, rankInfo, info);
		RestoreCheckpointParticleMetadata(particleStateF_, localSimState_.particlesF.N_);
		RestoreCheckpointParticleMetadata(particleStateB_, localSimState_.particlesB.N_);
		globalSimState_ = {};
		globalSimState_.particlesR = localSimState_.particlesR;
		globalSimState_.rigidBodies = localSimState_.rigidBodies;
		rigidBodySyncBuffers_.Resize(globalSimState_.rigidBodies.N_);
		boundaryRedistributionRequired_ = true;
		serializationMgr_.ResetExportDirState();
		exportSnapshotDir_.clear();
		localOk = ValidateRankLocalSimState(true);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to load distributed checkpoint: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }
	if (!ValidateGlobalOwnedParticleCounts()) { return false; }
	SPDLOG_INFO_MPI("Restored distributed checkpoint at step {} (time {})",
		mpiContext_, simStep_, simTime_);

	return true;
}
