// dist_checkpoint.cpp

// external
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// internal
#include "source/eval/serial/checkpoint.h"
#include "dist_checkpoint.h"
#include "dist_helpers.h"
#include "dist_simulation.h"



using namespace Dist;

namespace
{
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
} // namespace anonymous

void Dist::BuildCheckpointRepartitionProfile(
	const Export::SerializationMgr& serializationMgr,
	const std::string& path,
	const IDomainDecomp& domainDecomp,
	const Base::LoadBalanceWeightModel weightModel,
	const int rank,
	const int N_ranks,
	DomainDecompRepartitionProfile& profile)
{
	DomainDecompRepartitionParticleInput input{};
	input.weightModel = weightModel;
	input.rank = rank;

	// initialize the strategy's profile dimensions even when this rank has no particles
	domainDecomp.BuildRepartitionProfile(input, profile);
	DomainDecompRepartitionProfile chunkProfile{};
	std::vector<uint32_t> localIdxs{};

	const auto AccumulatePositions = [&](
		const Base::ParticleType particleType,
		const Export::SimStatePositionView& positions)
	{
		localIdxs.resize(positions.x.size());
		std::iota(localIdxs.begin(), localIdxs.end(), 0u);

		DomainDecompParticlePositionView particleInput{};
		particleInput.x = positions.x;
		particleInput.y = positions.y;
		particleInput.z = positions.z;
		particleInput.ownedLocalIdx = localIdxs;

		input.fluid = {};
		input.boundary = {};
		input.rigid = {};
		if (particleType == Base::ParticleType::Fluid)
		{
			input.fluid = particleInput;
		}
		else if (particleType == Base::ParticleType::Boundary)
		{
			input.boundary = particleInput;
		}
		else
		{
			input.rigid = particleInput;
		}

		domainDecomp.BuildRepartitionProfile(input, chunkProfile);

		const auto AccumulateCounts = [](auto& total, const auto& chunk)
		{
			if (total.size() != chunk.size())
			{
				throw std::runtime_error("Checkpoint profile dimensions changed.");
			}
			for (size_t i = 0; i < total.size(); i++)
			{
				total[i] = Sim::CheckedParticleAdd(total[i], chunk[i], "checkpoint profile");
			}
		};
		AccumulateCounts(profile.ownedFluidByUnit, chunkProfile.ownedFluidByUnit);
		AccumulateCounts(profile.ownedBoundaryByUnit, chunkProfile.ownedBoundaryByUnit);
		AccumulateCounts(profile.ownedRigidByUnit, chunkProfile.ownedRigidByUnit);
	};
	serializationMgr.VisitSimStatePositions(path, rank, N_ranks, AccumulatePositions);
}

Export::BinHeaderSim Dist::SaveDistributedCheckpointState(
	const Sync::UniversalSimState& state,
	const std::string& path,
	const MpiContext& mpiContext)
{
	try
	{
		// packed rank metadata: F/B/R/RB counts, step, then exact time bits
		constexpr size_t N_rankInfoFields = 6;
		std::vector<uint64_t> rankInfo(static_cast<size_t>(mpiContext.Size()) * N_rankInfoFields, 0);
		const size_t localOffset = static_cast<size_t>(mpiContext.Rank()) * N_rankInfoFields;
		rankInfo[localOffset] = state.particlesF.N_;
		rankInfo[localOffset + 1] = state.particlesB.N_;
		rankInfo[localOffset + 2] = state.particlesR.N_;
		rankInfo[localOffset + 3] = state.rigidBodies.N_;
		rankInfo[localOffset + 4] = state.step;
		rankInfo[localOffset + 5] = std::bit_cast<uint64_t>(state.time);
		mpiContext.AllReduceSumUint64s(rankInfo);

		// compute global counts and this rank's particle ranges
		uint64_t N_totalF = 0;
		uint64_t N_totalB = 0;
		uint64_t N_totalR = 0;
		uint64_t beginIdxF = 0;
		uint64_t beginIdxB = 0;
		uint64_t beginIdxR = 0;
		uint64_t N_maxF = 0;
		uint64_t N_maxB = 0;
		uint64_t N_maxR = 0;
		for (int rank = 0; rank < mpiContext.Size(); rank++)
		{
			const size_t offset = static_cast<size_t>(rank) * N_rankInfoFields;
			if (rankInfo[offset + 3] != state.rigidBodies.N_ ||
				rankInfo[offset + 4] != state.step ||
				rankInfo[offset + 5] != std::bit_cast<uint64_t>(state.time))
			{
				throw std::runtime_error("Checkpoint ranks disagree on rigid body counts or step/time.");
			}
			Sim::CheckedLocalParticleCount(rankInfo[offset], "checkpoint fluid count");
			Sim::CheckedLocalParticleCount(rankInfo[offset + 1], "checkpoint boundary count");
			Sim::CheckedLocalParticleCount(rankInfo[offset + 2], "checkpoint rigid count");
			if (rank == mpiContext.Rank())
			{
				beginIdxF = N_totalF;
				beginIdxB = N_totalB;
				beginIdxR = N_totalR;
			}
			N_totalF += rankInfo[offset];
			N_totalB += rankInfo[offset + 1];
			N_totalR += rankInfo[offset + 2];
			N_maxF = std::max(N_maxF, rankInfo[offset]);
			N_maxB = std::max(N_maxB, rankInfo[offset + 1]);
			N_maxR = std::max(N_maxR, rankInfo[offset + 2]);
		}

		// use the same field order for array validation and collective writes
		const auto VisitFluidFields = [](auto visit, auto& particles) { Export::VisitFluidStateFields(visit, particles); };
		const auto VisitBoundaryFields = [](auto visit, auto& particles) { Export::VisitBoundaryStateFields(visit, particles); };
		const auto VisitRigidFields = [](auto visit, auto& particles) { Export::VisitRigidStateFields(visit, particles); };
		const auto VisitBodyFields = [](auto visit, auto& bodies) { Export::VisitRigidBodyStateFields(visit, bodies); };
		const auto ValidateArrays = [](const auto& particles, auto visitFields)
		{
			const auto ValidateField = [&](const auto& field)
			{
				if (field.size() != particles.N_)
				{
					throw std::runtime_error("Invalid checkpoint array size.");
				}
			};
			visitFields(ValidateField, particles);
		};
		ValidateArrays(state.particlesF, VisitFluidFields);
		ValidateArrays(state.particlesB, VisitBoundaryFields);
		ValidateArrays(state.particlesR, VisitRigidFields);
		ValidateArrays(state.rigidBodies, VisitBodyFields);

		for (const auto* ids : { &state.particlesF.id, &state.particlesB.id, &state.particlesR.id })
		{
			if (std::find(ids->begin(), ids->end(), Sim::INVALID_PARTICLE_ID) != ids->end())
			{
				throw std::runtime_error("Checkpoint contains an invalid particle ID.");
			}
		}

		if (!std::isfinite(state.time) || state.time < 0 || state.step == std::numeric_limits<uint64_t>::max())
		{
			throw std::runtime_error("Invalid checkpoint step/time.");
		}

		std::vector<uint64_t> bodyParticleCounts(state.rigidBodies.N_, 0);
		for (const uint32_t bodyIdx : state.particlesR.bodyId)
		{
			if (bodyIdx >= state.rigidBodies.N_)
			{
				throw std::runtime_error("Invalid checkpoint rigid body index.");
			}
			bodyParticleCounts[bodyIdx]++;
		}

		mpiContext.AllReduceSumUint64s(bodyParticleCounts);
		for (uint32_t bodyIdx = 0; bodyIdx < state.rigidBodies.N_; bodyIdx++)
		{
			if (bodyParticleCounts[bodyIdx] != state.rigidBodies.rigidParticleCount[bodyIdx])
			{
				throw std::runtime_error("Checkpoint rigid particle memberships do not match body counts.");
			}
		}

		// root writes the header, all ranks participate in the collective call
		const auto header = Export::MakeSimStateHeader(
			state.step, state.time, N_totalF, N_totalB, N_totalR, state.rigidBodies.N_);
		const auto off = Export::ComputeSimStateOffsets(header);
		MPI_File file = mpiContext.OpenFileForWrite(path, off.totalBytes);
		mpiContext.WriteFileBytesAtAll(
			file,
			0,
			&header,
			mpiContext.IsRoot() ? static_cast<int>(sizeof(header)) : 0,
			"Checkpoint header write");

		// empty ranks participate in every chunk write
		const char dummy = 0;
		const auto WritePopulation = [&](
			const auto& particles,
			uint64_t offset,
			const uint64_t N_total,
			const uint64_t beginIdx,
			const uint64_t N_max,
			const bool writeLocal,
			auto visitFields)
		{
			const auto WriteField = [&](const auto& field)
			{
				const uint64_t valueBytes = sizeof(field[0]);
				const uint64_t N_bytes = writeLocal ? field.size() * valueBytes : 0;
				for (uint64_t byteOffset = 0; byteOffset < N_max * valueBytes; byteOffset += MPI_CONTEXT_BYTE_CHUNK_SIZE)
				{
					const uint64_t N_chunkBytes = byteOffset < N_bytes
						? std::min(MPI_CONTEXT_BYTE_CHUNK_SIZE, N_bytes - byteOffset) : 0;
					mpiContext.WriteFileBytesAtAll(
						file,
						N_chunkBytes ? offset + beginIdx * valueBytes + byteOffset : 0,
						N_chunkBytes ? reinterpret_cast<const char*>(field.data()) + byteOffset : &dummy,
						static_cast<int>(N_chunkBytes),
						"Checkpoint field write");
				}
				offset += N_total * valueBytes;
			};
			visitFields(WriteField, particles);
		};
		WritePopulation(state.particlesF, off.fluid, N_totalF, beginIdxF, N_maxF, true, VisitFluidFields);
		WritePopulation(state.particlesB, off.boundary, N_totalB, beginIdxB, N_maxB, true, VisitBoundaryFields);
		WritePopulation(state.particlesR, off.rigid, N_totalR, beginIdxR, N_maxR, true, VisitRigidFields);

		// root contributes the replicated body state
		WritePopulation(
			state.rigidBodies, off.rigidBodies, header.N_RB, 0, header.N_RB,
			mpiContext.IsRoot(), VisitBodyFields);

		mpiContext.CloseFile(file);
		// every rank must close the file before root checks its size and publishes it
		mpiContext.Barrier();
		return header;
	}
	catch (...)
	{
		mpiContext.Abort("Failed during collective checkpoint state output");
	}
}

bool DistSimulationMgr::SaveCheckpoint(
	const Sim::SimParams& params)
{
	bool localOk = true;
	std::string directory{};
	std::string writePath{};

	// validate owned particles before creating checkpoint files
	try
	{
		localOk = ValidateRankLocalSimState(true);
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
		// the validated state stores [ owned ][ halo ], halos are rebuilt on restart
		localSimState_.particlesF.resizeAll(particleStateF_.N_owned);
		localSimState_.particlesB.resizeAll(particleStateB_.N_owned);
		localSimState_.particlesR.resizeAll(particleStateR_.N_owned);
		localSimState_.step = simStep_;
		localSimState_.time = simTime_;
		writePath = (std::filesystem::path(directory) / "state.bin.tmp").string();
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to prepare shared checkpoint state: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }
	const auto header = SaveDistributedCheckpointState(localSimState_, writePath, mpiContext_);

	// every rank has closed the shared file, publish the manifest last
	if (mpiContext_.IsRoot())
	{
		try
		{
			const uint64_t N_bytes = Export::ComputeSimStateOffsets(header).totalBytes;
			const auto statePath = std::filesystem::path(directory) / "state.bin";
			if (std::filesystem::exists(statePath) || std::filesystem::file_size(writePath) != N_bytes)
			{
				throw std::runtime_error("Cannot publish incomplete or existing checkpoint state.");
			}
			std::filesystem::rename(writePath, statePath);
			auto manifest = Export::MakeCheckpointManifest(
				Export::CheckpointMode::Distributed, simStep_, simTime_, params);
			manifest["globalFluidCount"] = N_expectedGlobalF_;
			manifest["globalBoundaryCount"] = N_expectedGlobalB_;
			manifest["globalRigidCount"] = N_expectedGlobalR_;
			manifest["fluidIdChecksum"] = expectedOwnedIdChecksumF_;
			manifest["boundaryIdChecksum"] = expectedOwnedIdChecksumB_;
			manifest["rigidIdChecksum"] = expectedOwnedIdChecksumR_;
			manifest["globalBounds"] = CheckpointBoundsToJson(domainDecomp_->GlobalExtentAABB());

			nlohmann::json fileInfo = nlohmann::json::object();
			fileInfo["file"] = "state.bin";
			fileInfo["fluidCount"] = header.N_F;
			fileInfo["boundaryCount"] = header.N_B;
			fileInfo["rigidCount"] = header.N_R;
			fileInfo["bodyCount"] = header.N_RB;
			fileInfo["bytes"] = N_bytes;
			manifest["files"].push_back(std::move(fileInfo));

			Export::PublishCheckpointManifest(directory, manifest);
			SPDLOG_INFO("Saved distributed checkpoint '{}'",
				(std::filesystem::path(directory) / "manifest.json").string());
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR("Failed to publish distributed checkpoint: {}", ex.what());
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

	nlohmann::json manifest;
	Export::CheckpointInfo info{};
	DomainDecompInput input{};
	std::filesystem::path statePath;
	DomainDecompRepartitionProfile profile{};

	try
	{
		// validate the manifest and restore global simulation metadata
		manifest = Export::ParseCheckpointManifest(manifestText);
		info = Export::ValidateCheckpointManifest(
			manifest, Export::CheckpointMode::Distributed, simParams_);
		simStep_ = info.step;
		simTime_ = info.time;
		N_expectedGlobalF_ = Export::CheckpointUint64(manifest.at("globalFluidCount"));
		N_expectedGlobalB_ = Export::CheckpointUint64(manifest.at("globalBoundaryCount"));
		N_expectedGlobalR_ = Export::CheckpointUint64(manifest.at("globalRigidCount"));
		expectedOwnedIdChecksumF_ = Export::CheckpointUint64(manifest.at("fluidIdChecksum"));
		expectedOwnedIdChecksumB_ = Export::CheckpointUint64(manifest.at("boundaryIdChecksum"));
		expectedOwnedIdChecksumR_ = Export::CheckpointUint64(manifest.at("rigidIdChecksum"));

		// check the shared state file before loading particles
		const auto& files = manifest.at("files");
		if (files.size() != 1 || files[0].at("file") != "state.bin")
		{
			throw std::runtime_error("Shared checkpoint requires one state.bin file.");
		}
		const auto& fileInfo = files[0];
		statePath = std::filesystem::path(directory) / "state.bin";
		if (std::filesystem::file_size(statePath) != Export::CheckpointUint64(fileInfo.at("bytes")))
		{
			throw std::runtime_error("Checkpoint state file length differs from the manifest.");
		}

		// build the current layout using the saved global bounds
		input =
		{
			.globalBounds = CheckpointBoundsFromJson(manifest.at("globalBounds")),
			.haloRadius = simParams_.distHaloRadius,
			.N_ranks = mpiContext_.Size()
		};
		domainDecomp_->Build(input);
		if (!domainDecomp_->IsValidForRankCount(mpiContext_.Size()))
		{
			throw std::runtime_error("Invalid checkpoint restart decomposition.");
		}

		// profile saved positions before allocating any owned particle state
		BuildCheckpointRepartitionProfile(serializationMgr_, statePath.string(), *domainDecomp_,
			simParams_.loadBalanceWeightModel, mpiContext_.Rank(), mpiContext_.Size(), profile);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to build checkpoint domain decomp profile: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	// every rank must finish local reads successfully before reducing the profiles
	if (!mpiContext_.AllTrue(localOk)) { return false; }
	ApplyInitialDomainDecompProfile(profile);

	try
	{
		const auto& fileInfo = manifest.at("files")[0];

		// retain particles owned by this rank in the new layout
		const auto IsLocalParticle = [&](const glm::vec3& position)
		{
			const int ownerRank = domainDecomp_->OwnerRank(position);
			if (ownerRank < 0 || ownerRank >= mpiContext_.Size())
			{
				throw std::runtime_error("Checkpoint particle has no valid owner in the selected decomposition.");
			}
			return ownerRank == mpiContext_.Rank();
		};
		const AABB haloBounds = domainDecomp_->HaloExtentAABB(mpiContext_.Rank());
		const auto IsLocalRigidParticle = [&](const glm::vec3& position)
		{
			return IsLocalParticle(position) || haloBounds.Contains(position);
		};
		const auto header = serializationMgr_.LoadSelectedSimState(
			globalSimState_, statePath.string(), IsLocalParticle, IsLocalRigidParticle);
		if (header.step != info.step || header.time != info.time ||
			header.N_F != N_expectedGlobalF_ ||
			header.N_B != N_expectedGlobalB_ ||
			header.N_R != N_expectedGlobalR_ ||
			header.N_F != Export::CheckpointUint64(fileInfo.at("fluidCount")) ||
			header.N_B != Export::CheckpointUint64(fileInfo.at("boundaryCount")) ||
			header.N_R != Export::CheckpointUint64(fileInfo.at("rigidCount")) ||
			header.N_RB != Export::CheckpointUint64(fileInfo.at("bodyCount")))
		{
			throw std::runtime_error("Checkpoint state differs from the manifest.");
		}

		// rebuild ownership metadata for startup redistribution and halo construction
		simBackend_->SetSimDomainBounds(input.globalBounds.min, input.globalBounds.max);
		localOk = BuildRankLocalSimState();
		rigidBodySyncBuffers_.Resize(globalSimState_.rigidBodies.N_);

		boundaryRedistributionRequired_ = true;
		serializationMgr_.ResetExportDirState();
		exportSnapshotDir_.clear();
		localOk = localOk && ValidateRankLocalSimState(true);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to load distributed checkpoint: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }
	if (!ValidateGlobalOwnedParticleCounts(false)) { return false; }
	SPDLOG_INFO_MPI("Restored distributed checkpoint at step {} (time {})",
		mpiContext_, simStep_, simTime_);

	return true;
}
