// dist_simulation.cpp

// external
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

// internal
#include "Source/Base/enum_hub.h"
#include "Source/Sim/Scene/scenario_sim.h"
#include "dist_diagnostics.h"
#include "dist_simulation.h"
#include "dist_helpers.h"
#include "dist_snapshots.h"
#include "halo_exchange.h"
#include "owned_particle_gather.h"
#include "particle_migration.h"
#include "Layouts/decomp_AABB_uniform.h"



using namespace Dist;

namespace
{
	void IncludePoint(AABB& box, const glm::vec3& point)
	{
		box.min.x = std::min(box.min.x, point.x);
		box.min.y = std::min(box.min.y, point.y);
		box.min.z = std::min(box.min.z, point.z);

		box.max.x = std::max(box.max.x, point.x);
		box.max.y = std::max(box.max.y, point.y);
		box.max.z = std::max(box.max.z, point.z);
	}

	// scenarios use OBBs, while decomp starts from one global AABB
	AABB ComputeAABBFromOBB(const Scene::OBB& box)
	{
		const glm::vec3 u = box.uHat * box.uLen;
		const glm::vec3 v = box.vHat * box.vLen;
		const glm::vec3 w = box.wHat * box.wLen;

		const std::array<glm::vec3, 8> corners =
		{
			box.origin,
			box.origin + u,
			box.origin + v,
			box.origin + w,
			box.origin + u + v,
			box.origin + u + w,
			box.origin + v + w,
			box.origin + u + v + w
		};

		AABB boxAABB
		{
			.min = corners[0],
			.max = corners[0]
		};

		for (const glm::vec3& corner : corners)
		{
			IncludePoint(boxAABB, corner);
		}

		return boxAABB;
	}

	// build the global domain AABB that will be split between MPI ranks
	AABB ComputeGlobalDomainBounds(
		const std::vector<Scene::OBB>& boundaryBoxes,
		const Sim::SimParams& params)
	{
		if (boundaryBoxes.empty())
		{
			throw std::runtime_error("Scenario does not provide boundary boxes.");
		}

		AABB globalBounds = ComputeAABBFromOBB(boundaryBoxes.front());

		for (uint64_t i = 1; i < boundaryBoxes.size(); i++)
		{
			const AABB boxBounds = ComputeAABBFromOBB(boundaryBoxes[i]);
			IncludePoint(globalBounds, boxBounds.min);
			IncludePoint(globalBounds, boxBounds.max);
		}

		const float boundaryPad = params.boundaryLayers > 0
			? (static_cast<float>(params.boundaryLayers) - 0.5f) * params.particleSizeB
			: 0.0f;

		return globalBounds.Expanded(boundaryPad);
	}

	std::string FormatNeighborRanks(const std::vector<int>& neighborRanks)
	{
		std::ostringstream oss;
		oss << '[';
		for (uint64_t i = 0; i < neighborRanks.size(); i++)
		{
			if (i > 0) { oss << ", "; }
			oss << neighborRanks[i];
		}
		oss << ']';
		return oss.str();
	}

	template <std::size_t N>
	void CopyTextField(char (&dst)[N], const std::string& src)
	{
		std::snprintf(dst, N, "%s", src.c_str());
	}

	std::string FormatRankLogDirName(const int rank)
	{
		std::ostringstream oss;
		oss << "rank_" << std::setw(3) << std::setfill('0') << rank;
		return oss.str();
	}

	Export::LoggingOptions MakeRankLoggingOptions(const MpiContext& mpiContext)
	{
		Export::LoggingOptions loggingOptions{};
		// each rank writes its own CSV files to avoid shared log-file writes
		loggingOptions.baseDir /= FormatRankLogDirName(mpiContext.Rank());
		return loggingOptions;
	}

	Mem::MemSpace PrimaryMemSpace(const Base::BackendVersion backendVersion)
	{
		return backendVersion == Base::BackendVersion::GPU01
			? Mem::MemSpace::VRAM
			: Mem::MemSpace::RAM;
	}

	Mem::MemUsage SumMemoryUsage(
		const Mem::MemRecordStep& record,
		const Mem::MemSpace space)
	{
		Mem::MemUsage total{};

		for (uint8_t i = 0; i < Mem::GetIdx(Mem::MemBucket::Count); i++)
		{
			Mem::Add(total, record.mem[Mem::GetIdx(space)][i]);
		}

		return total;
	}

	Mem::MemUsage SumMaxMemoryUsage(
		const Mem::MemoryTracker& memoryTracker,
		const Mem::MemSpace space)
	{
		Mem::MemUsage total{};

		for (uint8_t i = 0; i < Mem::GetIdx(Mem::MemBucket::Count); i++)
		{
			const auto bucket = static_cast<Mem::MemBucket>(i);
			total.usedBytes += memoryTracker.GetMem_Max_Used(space, bucket);
			total.allocBytes += memoryTracker.GetMem_Max_Alloc(space, bucket);
		}

		return total;
	}

	constexpr std::array<Mem::MemBucket, 7> BACKEND_MEM_BUCKETS =
	{
		Mem::MemBucket::ParticlePropertiesF,
		Mem::MemBucket::ParticlePropertiesB,
		Mem::MemBucket::Neighbors_SearchDataStructF,
		Mem::MemBucket::Neighbors_SearchDataStructB,
		Mem::MemBucket::Neighbors_ListStorageF,
		Mem::MemBucket::Neighbors_ListStorageB,
		Mem::MemBucket::Misc
	};

	Mem::MemUsage SumBackendMaxMemoryUsage(
		const Mem::MemoryTracker& memoryTracker,
		const Mem::MemSpace space)
	{
		Mem::MemUsage total{};

		for (const Mem::MemBucket bucket : BACKEND_MEM_BUCKETS)
		{
			total.usedBytes += memoryTracker.GetMem_Max_Used(space, bucket);
			total.allocBytes += memoryTracker.GetMem_Max_Alloc(space, bucket);
		}

		return total;
	}

	Mem::MemUsage MemoryUsageOfFluidParticles(
		const Sync::FluidParticles& fluidParticles)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.xPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.yPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.zPos));

		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.xVel));
		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.yVel));
		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.zVel));

		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.xVelPred));
		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.yVelPred));
		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.zVelPred));

		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.xAccPress));
		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.yAccPress));
		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.zAccPress));

		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.sourceTerm));
		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.diagElem));
		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.pressure));
		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.density));

		return usage;
	}

	Mem::MemUsage MemoryUsageOfBoundaryParticles(
		const Sync::BoundaryParticles& boundaryParticles)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(boundaryParticles.xPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(boundaryParticles.yPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(boundaryParticles.zPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(boundaryParticles.volB));
		Mem::Add(usage, Mem::MemUsageOf_Vector(boundaryParticles.psiB));

		return usage;
	}

	Mem::MemUsage MemoryUsageOfUniversalSimState(
		const Sync::UniversalSimState& state)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, MemoryUsageOfFluidParticles(state.fluidParticles));
		Mem::Add(usage, MemoryUsageOfBoundaryParticles(state.boundaryParticles));

		return usage;
	}

	Mem::MemUsage MemoryUsageOfExportSnapshot(
		const Sync::ExportSnapshot& snap)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.idF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.xPosF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.yPosF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.zPosF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.velMagF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.idB));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.xPosB));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.yPosB));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.zPosB));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.volB));

		return usage;
	}

	Mem::MemUsage MemoryUsageOfParticleState(
		const DistParticleState& particleState)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleClasses));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_Owned));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_Inner));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_Outer));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_Halo));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleGlobalIdx_FromLocalIdx));
		Mem::Add(usage, Mem::MemUsageOf_Vector(particleState.particleLocalIdx_FromGlobalIdx));

		return usage;
	}

	Mem::MemUsage MemoryUsageOfParticleClassificationWorkBuffers(
		const ParticleClassificationWorkBuffers& workBuffers)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.classCounts_ByChunkClass));
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.classWriteOffsets_ByChunkClass));
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.classTotals));

		return usage;
	}

	Mem::MemUsage MemoryUsageOfParticleMigrationBuffer(
		const ParticleMigrationBuffer& buffer)
	{
		return
		{
			.usedBytes = buffer.fluidParticles.UsedBytes(),
			.allocBytes = buffer.fluidParticles.AllocBytes()
		};
	}

	Mem::MemUsage MemoryUsageOfHaloExchangeBuffer(
		const HaloExchangeBuffer& buffer)
	{
		Mem::MemUsage usage{};

		usage.usedBytes += buffer.fluidParticles.UsedBytes();
		usage.allocBytes += buffer.fluidParticles.AllocBytes();
		usage.usedBytes += buffer.boundaryParticles.UsedBytes();
		usage.allocBytes += buffer.boundaryParticles.AllocBytes();

		return usage;
	}

	Mem::MemUsage MemoryUsageOfParticleMigrationExchangeBuffers(
		const MigrationWorkBuffers& workBuffers)
	{
		Mem::MemUsage usage = Mem::MemUsageOf_Vector(workBuffers.rankBuffers);

		for (const MigrationWorkBuffers::RankBuffers& rankBuffers :
			workBuffers.rankBuffers)
		{
			Mem::Add(usage,
				MemoryUsageOfParticleMigrationBuffer(rankBuffers.sendBuffer));
			Mem::Add(usage,
				MemoryUsageOfParticleMigrationBuffer(rankBuffers.recvBuffer));
		}

		return usage;
	}

	Mem::MemUsage MemoryUsageOfHaloExchangeWorkBuffers(
		const HaloExchangeWorkBuffers& workBuffers)
	{
		Mem::MemUsage usage = Mem::MemUsageOf_Vector(workBuffers.rankBuffers);

		for (const HaloExchangeWorkBuffers::RankBuffers& rankBuffers :
			workBuffers.rankBuffers)
		{
			Mem::Add(usage,
				MemoryUsageOfHaloExchangeBuffer(rankBuffers.sendBuffer));
			Mem::Add(usage,
				MemoryUsageOfHaloExchangeBuffer(rankBuffers.recvBuffer));
		}
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.rankBufferIdx_FromRank));

		return usage;
	}

	Mem::MemUsage MemoryUsageOfParticleMigrationScratch(
		const MigrationWorkBuffers& workBuffers)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.rankBufferIdx_FromRank));
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.sendFluidCount_FromRank));
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.recvFluidCount_FromRank));
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.keptOwnedFluidLocalIdx));
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.migratingOutFluidLocalIdx));
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.migrationRouteIdx_FromOwnedIdx));
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.migrationRouteCounts_ByChunkRoute));
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.migrationRouteWriteOffsets_ByChunkRoute));
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.migrationRouteTotals));
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.migrationOutCounts_ByChunk));
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.migrationOutWriteOffsets_ByChunk));
		Mem::Add(usage, Mem::MemUsageOf_Vector(workBuffers.migrationLocalStatusStamp_FromLocalIdx));

		return usage;
	}

	Mem::MemUsage MemoryUsageOfHaloRebuildScratch(
		const HaloRebuildScratch& scratch)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, MemoryUsageOfUniversalSimState(scratch.state));
		Mem::Add(usage, MemoryUsageOfParticleState(scratch.fluidParticleState));
		Mem::Add(usage, MemoryUsageOfParticleState(scratch.boundaryParticleState));

		return usage;
	}

	bool AlmostEqual(const float a, const float b, const float eps = 1e-6f)
	{
		return std::fabs(a - b) <= eps;
	}

	glm::vec3 FluidPositionAtLocalIdx(
		const Sync::FluidParticles& fluidParticles,
		const int localIdx)
	{
		return
		{
			fluidParticles.xPos[localIdx],
			fluidParticles.yPos[localIdx],
			fluidParticles.zPos[localIdx]
		};
	}

	glm::vec3 BoundaryPositionAtLocalIdx(
		const Sync::BoundaryParticles& boundaryParticles,
		const int localIdx)
	{
		return
		{
			boundaryParticles.xPos[localIdx],
			boundaryParticles.yPos[localIdx],
			boundaryParticles.zPos[localIdx]
		};
	}

	bool IsLocalIdxInRange(
		const int localIdx,
		const uint32_t particleCount)
	{
		return localIdx >= 0 &&
			static_cast<uint32_t>(localIdx) < particleCount;
	}

	bool HasParticleMetadataAtLocalIdx(
		const DistParticleState& particleState,
		const int localIdx)
	{
		return localIdx >= 0 &&
			static_cast<uint64_t>(localIdx) < particleState.particleClasses.size() &&
			static_cast<uint64_t>(localIdx) < particleState.particleGlobalIdx_FromLocalIdx.size();
	}

	bool FluidParticleOrderMatchesLocalIdxMap(
		const Sync::UniversalSimState& backendState,
		const Sync::UniversalSimState& globalState,
		const DistParticleState& particleState)
	{
		const auto& backendFluid = backendState.fluidParticles;
		const auto& globalFluid = globalState.fluidParticles;

		if (particleState.particleGlobalIdx_FromLocalIdx.size() != backendFluid.N_)
		{
			return false;
		}

		for (uint64_t localIdx = 0; localIdx < backendFluid.N_; localIdx++)
		{
			const int globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
			if (globalIdx < 0 || static_cast<uint32_t>(globalIdx) >= globalFluid.N_) { return false; }

			if (!AlmostEqual(backendFluid.xPos[localIdx], globalFluid.xPos[globalIdx]) ||
				!AlmostEqual(backendFluid.yPos[localIdx], globalFluid.yPos[globalIdx]) ||
				!AlmostEqual(backendFluid.zPos[localIdx], globalFluid.zPos[globalIdx]) ||
				!AlmostEqual(backendFluid.xVel[localIdx], globalFluid.xVel[globalIdx]) ||
				!AlmostEqual(backendFluid.yVel[localIdx], globalFluid.yVel[globalIdx]) ||
				!AlmostEqual(backendFluid.zVel[localIdx], globalFluid.zVel[globalIdx]) ||
				!AlmostEqual(backendFluid.xVelPred[localIdx], globalFluid.xVelPred[globalIdx]) ||
				!AlmostEqual(backendFluid.yVelPred[localIdx], globalFluid.yVelPred[globalIdx]) ||
				!AlmostEqual(backendFluid.zVelPred[localIdx], globalFluid.zVelPred[globalIdx]) ||
				!AlmostEqual(backendFluid.xAccPress[localIdx], globalFluid.xAccPress[globalIdx]) ||
				!AlmostEqual(backendFluid.yAccPress[localIdx], globalFluid.yAccPress[globalIdx]) ||
				!AlmostEqual(backendFluid.zAccPress[localIdx], globalFluid.zAccPress[globalIdx]) ||
				!AlmostEqual(backendFluid.sourceTerm[localIdx], globalFluid.sourceTerm[globalIdx]) ||
				!AlmostEqual(backendFluid.diagElem[localIdx], globalFluid.diagElem[globalIdx]) ||
				!AlmostEqual(backendFluid.pressure[localIdx], globalFluid.pressure[globalIdx]) ||
				!AlmostEqual(backendFluid.density[localIdx], globalFluid.density[globalIdx]))
			{
				return false;
			}
		}

		return true;
	}

	bool BoundaryParticleOrderMatchesLocalIdxMap(
		const Sync::UniversalSimState& backendState,
		const Sync::UniversalSimState& globalState,
		const DistParticleState& particleState)
	{
		const auto& backendBoundary = backendState.boundaryParticles;
		const auto& globalBoundary = globalState.boundaryParticles;

		if (particleState.particleGlobalIdx_FromLocalIdx.size() != backendBoundary.N_)
		{
			return false;
		}

		for (uint64_t localIdx = 0; localIdx < backendBoundary.N_; localIdx++)
		{
			const int globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
			if (globalIdx < 0 || static_cast<uint32_t>(globalIdx) >= globalBoundary.N_) { return false; }

			if (!AlmostEqual(backendBoundary.xPos[localIdx], globalBoundary.xPos[globalIdx]) ||
				!AlmostEqual(backendBoundary.yPos[localIdx], globalBoundary.yPos[globalIdx]) ||
				!AlmostEqual(backendBoundary.zPos[localIdx], globalBoundary.zPos[globalIdx]) ||
				!AlmostEqual(backendBoundary.volB[localIdx], globalBoundary.volB[globalIdx]) ||
				!AlmostEqual(backendBoundary.psiB[localIdx], globalBoundary.psiB[globalIdx]))
			{
				return false;
			}
		}

		return true;
	}
}

DistSimulationMgr::DistSimulationMgr(MpiContext& mpiContext,
									 Sim::SimParamsMgr& simParamsMgr)
	: mpiContext_{ mpiContext }
	, simParamsMgr_{ simParamsMgr }
	, domainDecomp_{ std::make_unique<Decomp_AABB_Uniform>() }
	, loggingMgr_{ MakeRankLoggingOptions(mpiContext) }
{
	// nothing to do
}

bool DistSimulationMgr::Init(
	std::string& profile,
	const std::vector<std::string>& simOverrides,
	const bool logParamStatus)
{
	if (!LoadParamsFromProfileBroadcast(profile, logParamStatus))
	{
		return false;
	}

	configProfile_ = profile;
	paramsFile_ = configProfile_ + "/params.json";
	scenariosFile_ = configProfile_ + "/scenarios.json";

	// initial snapshot
	SyncParameters(true);

	if (!LoadParamsFromCliOverrides(simOverrides, logParamStatus))
	{
		return false;
	}
	SyncParameters(true);

	SPDLOG_INFO_MPI("Init distributed simulation state...", mpiContext_);
	SelectSimScenario(simParams_.scenarioType, configProfile_);
	// TODO: select sim backend function?
	if (!mpiContext_.AllTrue(simScenario_ != nullptr)) { return false; }

	SPDLOG_INFO_MPI("Init distributed domain decomp...", mpiContext_);
	if (!BuildDomainDecomp()) { return false; }

	SPDLOG_INFO_MPI("Init rank-local particle states...", mpiContext_);
	if (!ReBuildInitialSimState()) { return false; }

	SPDLOG_INFO_MPI("Init rank-local backend data structures...", mpiContext_);
	if (!InitRankLocalBackend()) { return false; }
	if (!RunRankLocalBackendStartupCheck()) { return false; }

	// updated snapshot
	initialized_.store(true, std::memory_order_release);
	SyncParameters(true);

	return true;
}

void DistSimulationMgr::Shutdown()
{
	RequestStop();
	EndRunLogging();

	// TODO: free memory occupied by particles, grids, etc...

	initialized_.store(false, std::memory_order_release);
	running_.store(false, std::memory_order_release);
	simBackend_.reset();
	memoryTracker_.reset();
	benchmarksTracker_.reset();
}

void DistSimulationMgr::SyncParameters(const bool force)
{
	uint64_t latestSimParamsVersion = simParamsMgr_.StableVersion();
	if (latestSimParamsVersion != simParamsVersion_ || force)
	{
		auto snapshot = simParamsMgr_.SnapshotParams();

		simParams_ = std::get<0>(snapshot);
		simParamsVersion_ = std::get<1>(snapshot);
	}
}

void DistSimulationMgr::SelectSimScenario(
	const Base::ScenarioType type,
	const std::string& profile)
{
	// create new scenario
	simScenario_ = Scene::MakeSimScenario(scenarioMgr_, type, profile);
	if (!simScenario_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to create sim scenario ({})",
			mpiContext_.Rank(), mpiContext_.Size(), static_cast<int>(type));
		return;
	}
}

void DistSimulationMgr::SelectSimBackend(const Base::BackendVersion version)
{
	simBackend_.reset();
	memoryTracker_.reset();
	benchmarksTracker_.reset();
	curBenchOpGroups_.clear();

	// create new metric trackers and backend over rank-local data
	benchmarksTracker_ = std::make_unique<Bench::BenchmarksTracker>(simParams_);
	memoryTracker_ = std::make_unique<Mem::MemoryTracker>(simParams_);
	simBackend_ = Sim::MakeSimBackend(version, simParamsMgr_, *benchmarksTracker_, simParams_);
	if (!simBackend_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to create sim backend ({})",
			mpiContext_.Rank(), mpiContext_.Size(), static_cast<int>(version));
		return;
	}

	simBackend_->GetBenchOpGroups(curBenchOpGroups_);
	const std::vector<Bench::BenchOpGroupDef> distBenchOpGroups =
	{
		{ .ops = { Bench::BenchOp::Dist_ParticleMigration } },
		{ .ops = { Bench::BenchOp::Dist_ParticleClass } },
		{ .ops = { Bench::BenchOp::Dist_HaloExchange } },
		{ .ops = { Bench::BenchOp::Dist_StateSync } }
	};
	auto distBenchInsertIt = curBenchOpGroups_.end();
	for (auto it = curBenchOpGroups_.begin(); it != curBenchOpGroups_.end(); ++it)
	{
		if (it->ops.size() == 1 && it->ops[0] == Bench::BenchOp::Serialization)
		{
			distBenchInsertIt = it;
			distBenchInsertIt++;
			break;
		}
	}
	if (distBenchInsertIt == curBenchOpGroups_.end())
	{
		for (auto it = curBenchOpGroups_.begin(); it != curBenchOpGroups_.end(); ++it)
		{
			if (it->ops.size() == 1 && it->ops[0] == Bench::BenchOp::Total_Sim)
			{
				distBenchInsertIt = it;
				break;
			}
		}
	}
	curBenchOpGroups_.insert(distBenchInsertIt, distBenchOpGroups.begin(), distBenchOpGroups.end());

	if (!Bench::ValidateOpGroups(curBenchOpGroups_))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid distributed benchmark operation groups "
			"for sim backend ({})",
			mpiContext_.Rank(), mpiContext_.Size(), static_cast<int>(version));
		curBenchOpGroups_.clear();
	}
}

bool DistSimulationMgr::ReBuildInitialSimState()
{
	bool localOk = true;

	try
	{
		if (!simScenario_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed to re-build initial sim state (no scenario selected)",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		if (!domainDecomp_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed to re-build initial sim state (no domain decomp)",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}

		if (localOk)
		{
			simStep_ = 0;
			simTime_ = 0.0;

			SyncParameters();

			if (!BuildGlobalSimState()) { localOk = false; }
			if (localOk && !BuildRankLocalSimState()) { localOk = false; }
			if (localOk && !ValidateRankLocalSimState()) { localOk = false; }
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to re-build initial sim state: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	return mpiContext_.AllTrue(localOk);
}

void DistSimulationMgr::RunStartupCheck() const
{
	if (!initialized_.load(std::memory_order_acquire))
	{
		SPDLOG_ERROR("DistSimulationMgr was not initialized.");
		return;
	}

	DistDiagnosticsMgr::PrintDistSimSetup(
		mpiContext_,
		BuildDistSimSetupReport(),
		BuildStartupRankReport());
}

DistSimSetupReport DistSimulationMgr::BuildDistSimSetupReport() const
{
	DistSimSetupReport report{};
	report.mpiThreadSupport = mpiContext_.ThreadSupport();

	report.domainDecompMethod = domainDecomp_ ? domainDecomp_->Name() : "<none>";
	report.backendVersion = Base::ToCString(simParams_.backendVersion);
	report.scenarioType = Base::ToCString(simParams_.scenarioType);
	report.neighborSearchMethod = Base::ToCString(simParams_.neighborSearchMethod);
	report.neighborListStorage = Base::ToEffectiveNeighborListStorageCString(
		simParams_.backendVersion,
		simParams_.neighborListMethod);

	report.globalFluidParticleCount = globalSimState_.fluidParticles.N_;
	report.globalBoundaryParticleCount = globalSimState_.boundaryParticles.N_;
	return report;
}

DistStartupRankReport DistSimulationMgr::BuildStartupRankReport() const
{
	DistStartupRankReport report{};
	report.rank = mpiContext_.Rank();

	if (domainDecomp_)
	{
		CopyTextField(
			report.neighborRanks,
			FormatNeighborRanks(domainDecomp_->NeighborRanks(mpiContext_.Rank())));
	}
	else
	{
		CopyTextField(report.neighborRanks, "[]");
	}

	report.ownedFluidParticleCount = fluidParticleState_.ownedParticleCount;
	report.ownedBoundaryParticleCount = boundaryParticleState_.ownedParticleCount;

	return report;
}

void DistSimulationMgr::RunSim_Blocking()
{
	bool localOk = true;

	if (!initialized_.load(std::memory_order_acquire))
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to launch blocking DistSimulationMgr (not initialized)",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!simBackend_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to launch blocking DistSimulationMgr (no backend selected)",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!benchmarksTracker_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to launch blocking DistSimulationMgr (no benchmark tracker)",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!memoryTracker_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to launch blocking DistSimulationMgr (no memory tracker)",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return; }

	if (running_.exchange(true))
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to launch blocking DistSimulationMgr (already running)",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk))
	{
		running_.store(false, std::memory_order_release);
		return;
	}

	stopRequested_.store(false, std::memory_order_release);

	if (mpiContext_.IsRoot())
	{
		printf("\n");
	}
	SPDLOG_INFO_MPI("Distributed simulation started running...", mpiContext_);

	// local copy for consistent maximum-step handling during this run
	const auto paramsInit = simParams_;
	distSyncTimingReport_ = {};
	BeginRunLogging(paramsInit);
	DistDiagnosticsMgr::ResetProgress();
	bool runOk = true;

	// pack and serialize export snapshot of initial state
	if (paramsInit.exportEnabled)
	{
		if (!HandleExportSnapshot(paramsInit)) { runOk = false; }
	}

	AppendInitialMemoryMetricsLog(paramsInit);

	while (true)
	{
		if (!runOk) { break; }

		SyncParameters(true);

		// local copy for step-local consistency
		const auto params = simParams_;
		const bool doSimStep = (params.simRunning || params.simRunNextStep);

		// let any rank stop the whole MPI loop cleanly
		const bool keepRunning = mpiContext_.AllTrue(
			!stopRequested_.load(std::memory_order_acquire));
		if (!keepRunning) { break; }

		// explicit maximum number of steps reached
		if (static_cast<int64_t>(simStep_) >= paramsInit.maxSteps &&
			paramsInit.maxSteps > 0) { break; }

		// small sleep to avoid busy spin when nothing to do yet
		if (!doSimStep)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		// reset data records for the new step
		benchmarksTracker_->BeginStep();
		memoryTracker_->BeginStep();
		lastParticleMigrationStats_ = {};
		lastHaloExchangeStats_ = {};

		// starting point of total sim benchmark
		benchmarksTracker_->StartBench({ Bench::BenchOp::Total_Sim });

		// perform the actual distributed simulation step
		const bool stepOk = ExecuteDistributedStep(params);

		if (stepOk)
		{
			// update metadata
			simStep_ += 1;
			simTime_ += params.timeStep;
			localSimState_.step = simStep_;
			localSimState_.time = simTime_;
		}

		if (!stepOk)
		{
			FinalizeBenchMemMeasurements(params);
			runOk = false;
			break;
		}

		// reset single-step flag
		if (simParams_.simRunNextStep) { simParamsMgr_.Set_SimRunNextStep(false); }

		// pack and serialize export snapshot at given frequency
		bool exportedStep = false;
		if (simStep_ % params.exportInterval == 0 &&
			params.exportEnabled)
		{
			exportedStep = true;
			if (!HandleExportSnapshot(params)) { runOk = false; }
		}

		// evaluate the benchmarks and memory usage for this step
		FinalizeBenchMemMeasurements(params);
		AppendLatestMemoryMetricsLog();
		AppendLatestBenchmarkMetricsLog();

		SyncParameters();

		DisplayMemoryUsageStats(params);
		DisplayDistributedProgress(params, exportedStep);
	}

	const auto paramsFinal = simParams_;
	if (mpiContext_.IsRoot())
	{
		std::printf("\n");
	}
	DisplayMemoryUsageStats(paramsFinal, true);

	if (runOk && !GatherOwnedGlobalSimStateOnRoot("after distributed run"))
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to gather owned global state after distributed run.",
			mpiContext_.Rank(), mpiContext_.Size());
	}

	running_.store(false, std::memory_order_release);
	EndRunLogging();

	SPDLOG_INFO_MPI("Distributed simulation stopped running", mpiContext_);
	if (mpiContext_.IsRoot())
	{
		printf("\n");
	}
}

void DistSimulationMgr::RequestStop()
{
	stopRequested_.store(true, std::memory_order_release);
}

bool DistSimulationMgr::LoadParamsFromProfileBroadcast(
	std::string& profile,
	const bool logStatus)
{
	std::string jsonText{};
	int loadOk = 1;

	if (mpiContext_.IsRoot())
	{
		if (!TryReadProfileJson(profile, jsonText))
		{
			if (profile != "Default")
			{
				if (logStatus)
				{
					SPDLOG_WARN_MPI(
						"Failed to load '/{}/params.json'. Falling back to '/Default/params.json'.",
						mpiContext_, profile);
				}
				profile = "Default";
			}

			if (!TryReadProfileJson(profile, jsonText))
			{
				SPDLOG_ERROR_MPI("Failed to load '/{}/params.json'.", mpiContext_, profile);
				loadOk = 0;
			}
		}
	}

	mpiContext_.BroadcastInt(loadOk);
	mpiContext_.BroadcastString(profile);
	if (loadOk == 0) { return false; }

	mpiContext_.BroadcastString(jsonText);

	bool localOk = true;
	try
	{
		const nlohmann::json j = nlohmann::json::parse(jsonText);
		localOk = simParamsMgr_.LoadSimParamsFromJson(j);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to parse broadcast params JSON: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	const bool globalOk = mpiContext_.AllTrue(localOk);
	if (globalOk && logStatus)
	{
		SPDLOG_INFO_MPI("Using config profile '{}'", mpiContext_, profile);
	}

	return globalOk;
}

bool DistSimulationMgr::LoadParamsFromCliOverrides(
	const std::vector<std::string>& simOverrides,
	const bool logStatus)
{
	bool localOk = true;
	if (!simOverrides.empty())
	{
		localOk = simParamsMgr_.LoadSimParamsFromCliOverrides(
			simOverrides,
			logStatus && mpiContext_.IsRoot());
	}

	const bool globalOk = mpiContext_.AllTrue(localOk);
	if (globalOk && logStatus && !simOverrides.empty())
	{
		SPDLOG_INFO_MPI(
			"Applied {} simulation override(s) from CLI args",
			mpiContext_,
			simOverrides.size());
	}

	return globalOk;
}

bool DistSimulationMgr::BuildDomainDecomp()
{
	bool localOk = true;

	try
	{
		if (!simScenario_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed to build distributed domain decomp (no scenario selected).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		else
		{
			const AABB globalBounds =
				ComputeGlobalDomainBounds(simScenario_->GetBoundaryBoxes(), simParams_);

			// kernel radius defines initial halo thickness for ownership queries
			const DomainDecompInput input
			{
				.globalBounds = globalBounds,
				.haloRadius = simParams_.kernelRadius,
				.rankCount = mpiContext_.Size()
			};

			domainDecomp_->Build(input);

			if (!domainDecomp_->IsValidForRankCount(mpiContext_.Size()))
			{
				SPDLOG_ERROR("[rank {}/{}] Invalid distributed domain decomp.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
			}
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to build distributed domain decomp: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	return mpiContext_.AllTrue(localOk);
}

bool DistSimulationMgr::BuildGlobalSimState()
{
	if (!simScenario_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to build global sim state (no scenario selected).",
			mpiContext_.Rank(), mpiContext_.Size());
		return false;
	}

	// reset scenario and all full-state data
	globalSimState_ = {};
	globalSimState_.schemaVersion = Sync::UniversalSimStateSchemaVersion;
	globalSimState_.step = 0;
	globalSimState_.time = 0.0;

	// all ranks build the same full scenario before rank-local compaction
	simScenario_->Build(globalSimState_, simParams_);
	globalSimState_.boundaryParticles.EnsureBoundaryCoefficientsPresent(
		simParams_.massB, simParams_.restDensityF);
	globalSimState_.ShrinkToFitAll();

	return true;
}

bool DistSimulationMgr::BuildRankLocalSimState()
{
	if (!domainDecomp_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to build rank-local sim state (no domain decomp).",
			mpiContext_.Rank(), mpiContext_.Size());
		return false;
	}

	// reset compact rank-local state and all particle classifications
	localSimState_ = {};
	fluidParticleState_.Clear();
	boundaryParticleState_.Clear();

	// keep owned particles first, then append halo particles
	BuildLocalParticlesFromGlobalState(
		localSimState_,
		fluidParticleState_,
		boundaryParticleState_,
		globalSimState_,
		*domainDecomp_,
		mpiContext_.Rank(),
		simParams_.kernelRadius);

	localSimState_.ShrinkToFitAll();

	return true;
}

bool DistSimulationMgr::ValidateRankLocalSimState() const
{
	bool localOk = true;

	const uint32_t localFluidParticleCount = localSimState_.fluidParticles.N_;
	const uint32_t localBoundaryParticleCount = localSimState_.boundaryParticles.N_;

	std::string failMsg{};
	if (!fluidParticleState_.IsValidLocalLayout(localFluidParticleCount, &failMsg))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid rank-local fluid particle layout: {}",
			mpiContext_.Rank(), mpiContext_.Size(), failMsg);
		localOk = false;
	}

	if (!boundaryParticleState_.IsValidLocalLayout(localBoundaryParticleCount, &failMsg))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid rank-local boundary particle layout: {}",
			mpiContext_.Rank(), mpiContext_.Size(), failMsg);
		localOk = false;
	}

	if (fluidParticleState_.particleGlobalIdx_FromLocalIdx.size() !=
		static_cast<uint64_t>(localFluidParticleCount))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid fluid local-to-global index map.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	if (fluidParticleState_.particleLocalIdx_FromGlobalIdx.size() !=
		static_cast<uint64_t>(globalSimState_.fluidParticles.N_))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid fluid global-to-local index map.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	if (boundaryParticleState_.particleGlobalIdx_FromLocalIdx.size() !=
		static_cast<uint64_t>(localBoundaryParticleCount))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid boundary local-to-global index map.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	if (boundaryParticleState_.particleLocalIdx_FromGlobalIdx.size() !=
		static_cast<uint64_t>(globalSimState_.boundaryParticles.N_))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid boundary global-to-local index map.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	return localOk;
}

bool DistSimulationMgr::InitRankLocalBackend()
{
	bool localOk = true;

	try
	{
		SyncParameters();

		SelectSimBackend(simParams_.backendVersion);
		if (!simBackend_)
		{
			localOk = false;
		}
		else
		{
			// transfer rank-local SoA state into backend-specific data structures
			simBackend_->PullFromUniversalSimState(localSimState_, simParams_);

			if (simBackend_->GetNumFluidParticles() != fluidParticleState_.ownedParticleCount +
				fluidParticleState_.haloParticleCount)
			{
				SPDLOG_ERROR("[rank {}/{}] Rank-local backend fluid count mismatch.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
			}

			if (simBackend_->GetNumBoundaryParticles() != boundaryParticleState_.ownedParticleCount +
				boundaryParticleState_.haloParticleCount)
			{
				SPDLOG_ERROR("[rank {}/{}] Rank-local backend boundary count mismatch.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
			}
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to initialize rank-local backend: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	return mpiContext_.AllTrue(localOk);
}

bool DistSimulationMgr::RunRankLocalBackendStartupCheck()
{
	bool localOk = true;

	try
	{
		if (!simBackend_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed rank-local backend startup check (no backend selected).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		else
		{
			// first backend smoke test on compact [ owned ][ halo ] particle data
			simBackend_->ExecuteNeighborSearch(simParams_);
			if (!ApplyLastBackendParticleReOrder()) { localOk = false; }

			Sync::UniversalSimState backendSimState{};
			simBackend_->PushToUniversalSimState(backendSimState);
			localSimState_ = std::move(backendSimState);
			localSimState_.step = simStep_;
			localSimState_.time = simTime_;

			if (localOk && !ValidateRankLocalBackendParticleOrder(
				"during backend startup check")) { localOk = false; }
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed rank-local backend startup check: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	return mpiContext_.AllTrue(localOk);
}

bool DistSimulationMgr::GatherOwnedGlobalSimStateOnRoot(const char* phaseName) const
{
	Sync::UniversalSimState globalOwnedState{};
	OwnedParticleGatherStats stats{};

	const bool gatherOk = GatherOwnedParticlesToRoot(
		globalOwnedState,
		localSimState_,
		fluidParticleState_,
		boundaryParticleState_,
		mpiContext_,
		globalSimState_.fluidParticles.N_,
		globalSimState_.boundaryParticles.N_,
		&stats);

	if (!gatherOk) { return false; }

	SPDLOG_INFO_MPI("Gathered owned global state {} (fluid={}, boundary={}, step={}, time={})",
		mpiContext_, phaseName,
		stats.gatheredFluidParticleCount,
		stats.gatheredBoundaryParticleCount,
		globalOwnedState.step,
		globalOwnedState.time);

	return true;
}

bool DistSimulationMgr::HandleExportSnapshot(const Sim::SimParams& params) const
{
	DistSnapshotGatherStats stats{};

	benchmarksTracker_->StartBench({ Bench::BenchOp::Snapshot_Export });
	// fill export snapshot
	auto& expSnap = exportSnapScratch_;
	const bool gatherOk = GatherExportSnapshotToRoot(
		expSnap,
		localSimState_,
		fluidParticleState_,
		boundaryParticleState_,
		mpiContext_,
		simStep_,
		simTime_,
		&stats);
	benchmarksTracker_->StopBench({ Bench::BenchOp::Snapshot_Export });

	if (!gatherOk) { return false; }

	if (mpiContext_.IsRoot())
	{
		benchmarksTracker_->StartBench({ Bench::BenchOp::Serialization });
		// write export snapshot to files
		serializationMgr_.SaveExportSnapshot(expSnap, params.exportDir);
		serializationMgr_.SaveExportSnapshotXdmfViews(expSnap, params.exportDir);
		benchmarksTracker_->StopBench({ Bench::BenchOp::Serialization });
	}

	return true;
}

void DistSimulationMgr::BeginRunLogging(const Sim::SimParams& params)
{
	loggingMgr_.BeginRun(
		configProfile_, paramsFile_, scenariosFile_, params, mpiContext_.IsRoot());
}

void DistSimulationMgr::EndRunLogging()
{
	loggingMgr_.EndRun();
}

void DistSimulationMgr::AppendInitialMemoryMetricsLog(const Sim::SimParams& params) const
{
	if (!loggingMgr_.IsActive()) { return; }
	if (!simBackend_) { return; }

	Mem::MemRecordStep record{};
	record.simStep = simStep_;
	record.simTime = simTime_;

	simBackend_->FillMemoryRecord(record, params);
	FillDistributedMemoryRecord(record);
	loggingMgr_.AppendMemoryRecord(record);
}

void DistSimulationMgr::AppendLatestMemoryMetricsLog() const
{
	if (!loggingMgr_.IsActive()) { return; }
	if (!memoryTracker_) { return; }

	loggingMgr_.AppendMemoryRecord(memoryTracker_->GetCurrentRecord());
}

void DistSimulationMgr::AppendLatestBenchmarkMetricsLog() const
{
	if (!loggingMgr_.IsActive()) { return; }
	if (!benchmarksTracker_) { return; }

	loggingMgr_.AppendBenchmarkRecord(benchmarksTracker_->GetCurrentRecord());
}

void DistSimulationMgr::FillDistributedMemoryRecord(
	Mem::MemRecordStep& record) const
{
	Mem::Set(
		record,
		Mem::MemSpace::RAM,
		Mem::MemBucket::Dist_GlobalState,
		MemoryUsageOfUniversalSimState(globalSimState_));
	Mem::Set(
		record,
		Mem::MemSpace::RAM,
		Mem::MemBucket::Dist_LocalState,
		MemoryUsageOfUniversalSimState(localSimState_));

	Mem::MemUsage particleMetadataUsage{};
	Mem::Add(particleMetadataUsage, MemoryUsageOfParticleState(fluidParticleState_));
	Mem::Add(particleMetadataUsage, MemoryUsageOfParticleState(boundaryParticleState_));
	Mem::Set(
		record,
		Mem::MemSpace::RAM,
		Mem::MemBucket::Dist_ParticleMetadata,
		particleMetadataUsage);

	Mem::MemUsage exchangeBufferUsage{};
	Mem::Add(exchangeBufferUsage,
		MemoryUsageOfParticleMigrationExchangeBuffers(particleMigrationWorkBuffers_));
	Mem::Add(exchangeBufferUsage,
		MemoryUsageOfHaloExchangeWorkBuffers(haloExchangeWorkBuffers_));
	Mem::Set(
		record,
		Mem::MemSpace::RAM,
		Mem::MemBucket::Dist_ExchangeBuffers,
		exchangeBufferUsage);

	Mem::MemUsage miscUsage{};
	Mem::Add(miscUsage,
		MemoryUsageOfParticleMigrationScratch(particleMigrationWorkBuffers_));
	Mem::Add(miscUsage,
		MemoryUsageOfParticleClassificationWorkBuffers(
			fluidParticleClassificationWorkBuffers_));
	Mem::Add(miscUsage,
		MemoryUsageOfParticleClassificationWorkBuffers(
			boundaryParticleClassificationWorkBuffers_));
	Mem::Add(miscUsage, MemoryUsageOfHaloRebuildScratch(haloRebuildScratch_));
	Mem::Add(miscUsage, MemoryUsageOfExportSnapshot(exportSnapScratch_));
	Mem::Set(
		record,
		Mem::MemSpace::RAM,
		Mem::MemBucket::Dist_Misc,
		miscUsage);
}

void DistSimulationMgr::FinalizeBenchMemMeasurements(
	const Sim::SimParams& params) const
{
	if (!simBackend_) { return; }
	if (!memoryTracker_) { return; }
	if (!benchmarksTracker_) { return; }

	// record memory usage, archive data and update aggregates
	simBackend_->FillMemoryRecord(memoryTracker_->GetCurrentRecord(), params);
	FillDistributedMemoryRecord(memoryTracker_->GetCurrentRecord());
	memoryTracker_->EndStep(simStep_, simTime_);

	// ending point of total sim benchmark, archive data and update aggregates
	benchmarksTracker_->StopBench({ Bench::BenchOp::Total_Sim });
	benchmarksTracker_->EndStep(simStep_, simTime_);
}

DistRankStepReport DistSimulationMgr::BuildRankStepReport(
	const Sim::SimParams& params,
	const bool exportedStep) const
{
	DistRankStepReport report{};
	report.rank = mpiContext_.Rank();
	report.step = simStep_;
	report.maxSteps = params.maxSteps;

	report.ownedFluidParticleCount = fluidParticleState_.ownedParticleCount;
	report.innerFluidParticleCount = static_cast<uint32_t>(fluidParticleState_.particleLocalIdx_Inner.size());
	report.outerFluidParticleCount = static_cast<uint32_t>(fluidParticleState_.particleLocalIdx_Outer.size());
	report.haloFluidParticleCount = fluidParticleState_.haloParticleCount;
	report.ownedBoundaryParticleCount = boundaryParticleState_.ownedParticleCount;
	report.innerBoundaryParticleCount = static_cast<uint32_t>(boundaryParticleState_.particleLocalIdx_Inner.size());
	report.outerBoundaryParticleCount = static_cast<uint32_t>(boundaryParticleState_.particleLocalIdx_Outer.size());
	report.haloBoundaryParticleCount = boundaryParticleState_.haloParticleCount;

	report.migratedFluidParticleSendCount = lastParticleMigrationStats_.sentFluidParticleCount;
	report.migratedFluidParticleRecvCount = lastParticleMigrationStats_.recvFluidParticleCount;
	report.haloFluidParticleSendCount = lastHaloExchangeStats_.sentFluidParticleCount;
	report.haloFluidParticleRecvCount = lastHaloExchangeStats_.recvFluidParticleCount;
	report.haloBoundaryParticleSendCount = lastHaloExchangeStats_.sentBoundaryParticleCount;
	report.haloBoundaryParticleRecvCount = lastHaloExchangeStats_.recvBoundaryParticleCount;

	if (benchmarksTracker_)
	{
		report.totalStepLatestNs =
			benchmarksTracker_->GetBench_Latest(Bench::BenchOp::Total_Sim);
		report.totalStepAvgNs =
			benchmarksTracker_->GetBench_Avg_OverAll(Bench::BenchOp::Total_Sim);
	}

	if (memoryTracker_)
	{
		const Mem::MemSpace memSpace = PrimaryMemSpace(params.backendVersion);
		const Mem::MemUsage latest = SumMemoryUsage(memoryTracker_->GetCurrentRecord(), memSpace);
		const Mem::MemUsage max = SumMaxMemoryUsage(*memoryTracker_, memSpace);
		const Mem::MemUsage backendMax = SumBackendMaxMemoryUsage(*memoryTracker_, memSpace);

		report.latestMemUsedBytes = latest.usedBytes;
		report.latestMemAllocBytes = latest.allocBytes;
		report.maxMemUsedBytes = max.usedBytes;
		report.maxMemAllocBytes = max.allocBytes;

		report.maxBackendMemAllocBytes = backendMax.allocBytes;
		report.maxDistGlobalStateMemAllocBytes =
			memoryTracker_->GetMem_Max_Alloc(
				Mem::MemSpace::RAM, Mem::MemBucket::Dist_GlobalState);
		report.maxDistLocalStateMemAllocBytes =
			memoryTracker_->GetMem_Max_Alloc(
				Mem::MemSpace::RAM, Mem::MemBucket::Dist_LocalState);
		report.maxDistParticleMetadataMemAllocBytes =
			memoryTracker_->GetMem_Max_Alloc(
				Mem::MemSpace::RAM, Mem::MemBucket::Dist_ParticleMetadata);
		report.maxDistExchangeBuffersMemAllocBytes =
			memoryTracker_->GetMem_Max_Alloc(
				Mem::MemSpace::RAM, Mem::MemBucket::Dist_ExchangeBuffers);
		report.maxDistMiscMemAllocBytes =
			memoryTracker_->GetMem_Max_Alloc(
				Mem::MemSpace::RAM, Mem::MemBucket::Dist_Misc);
		report.maxTotalTrackedMemAllocBytes =
			report.maxBackendMemAllocBytes +
			report.maxDistGlobalStateMemAllocBytes +
			report.maxDistLocalStateMemAllocBytes +
			report.maxDistParticleMetadataMemAllocBytes +
			report.maxDistExchangeBuffersMemAllocBytes +
			report.maxDistMiscMemAllocBytes;
	}

	report.exportedStep = exportedStep ? 1 : 0;
	return report;
}

DistRankBenchReport DistSimulationMgr::BuildRankBenchReport() const
{
	DistRankBenchReport report{};
	report.rank = mpiContext_.Rank();
	report.step = simStep_;

	if (!benchmarksTracker_) { return report; }

	for (uint8_t i = 0; i < Bench::GetIdx(Bench::BenchOp::Count); i++)
	{
		const auto op = static_cast<Bench::BenchOp>(i);
		report.latestNs[i] = benchmarksTracker_->GetBench_Latest(op);
		report.avgNs[i] = benchmarksTracker_->GetBench_Avg_OverAll(op);
	}

	return report;
}

void DistSimulationMgr::DisplayMemoryUsageStats(
	const Sim::SimParams& params,
	const bool force) const
{
	if (!force && simStep_ != 1) { return; }

	DistDiagnosticsMgr::PrintMemorySummary(
		mpiContext_, params, BuildRankStepReport(params, false));
}

void DistSimulationMgr::DisplayDistributedProgress(
	const Sim::SimParams& params,
	const bool exportedStep) const
{
	if (simStep_ == 0) { return; }

	DistDiagnosticsMgr::PrintLatestPerformanceSummaryAndProgress(
		mpiContext_, params, curBenchOpGroups_,
		BuildRankStepReport(params, exportedStep),
		BuildRankBenchReport(),
		distSyncTimingReport_);
}

bool DistSimulationMgr::ExecuteDistributedStep(const Sim::SimParams& params)
{
	bool localOk = true;

	try
	{
		if (!simBackend_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed distributed step (no backend selected).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		if (!domainDecomp_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed distributed step (no domain decomp).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		if (!mpiContext_.AllTrue(localOk)) { return false; }

		BeginDistSyncTimingReport(distSyncTimingReport_, mpiContext_.Rank());

		if (localOk)
		{
			SyncParameters();

			// boundary particles are still treated as static owned data
			if (!ValidateOwnedBoundaryParticlesStillOnRank()) { localOk = false; }
		}
		if (!mpiContext_.AllTrue(localOk)) { return false; }

		if (localOk)
		{
			benchmarksTracker_->StartBench({ Bench::BenchOp::Dist_ParticleMigration });
			const bool migrationOk = MigrateOwnedFluidParticlesToOwnerRanks();
			benchmarksTracker_->StopBench({ Bench::BenchOp::Dist_ParticleMigration });
			if (!migrationOk) { localOk = false; }
		}
		if (!mpiContext_.AllTrue(localOk)) { return false; }

		if (localOk)
		{
			benchmarksTracker_->StartBench({ Bench::BenchOp::Dist_ParticleClass });

			// refresh inner/outer lists after migration and backend order changes
			ClassifyOwnedFluidParticlesIntoInnerOuter(
				fluidParticleState_, localSimState_.fluidParticles, *domainDecomp_,
				mpiContext_.Rank(), params.kernelRadius,
				&fluidParticleClassificationWorkBuffers_);
			ClassifyOwnedBoundaryParticlesIntoInnerOuter(
				boundaryParticleState_, localSimState_.boundaryParticles, *domainDecomp_,
				mpiContext_.Rank(), params.kernelRadius,
				&boundaryParticleClassificationWorkBuffers_);

			benchmarksTracker_->StopBench({ Bench::BenchOp::Dist_ParticleClass });
		}
		if (!mpiContext_.AllTrue(localOk)) { return false; }

		if (localOk)
		{
			benchmarksTracker_->StartBench({ Bench::BenchOp::Dist_HaloExchange });
			const bool haloOk = ExchangeRankLocalHalos();
			benchmarksTracker_->StopBench({ Bench::BenchOp::Dist_HaloExchange });
			if (!haloOk) { localOk = false; }
		}
		if (localOk && !PullRankLocalStateIntoBackend()) { localOk = false; }

		if (localOk)
		{
			benchmarksTracker_->StartBench({ Bench::BenchOp::Dist_StateSync });
			bool activeIdxOk = false;
			{
				[[maybe_unused]] const DistSyncTimingScope syncTimingScope(
					&distSyncTimingReport_, DistSyncTimingOp::ActiveIdx);
				activeIdxOk =
					simBackend_->SetActiveFluidParticleLocalIdxs(fluidParticleState_.particleLocalIdx_Owned);
			}
			benchmarksTracker_->StopBench({ Bench::BenchOp::Dist_StateSync });

			if (!activeIdxOk)
			{
				SPDLOG_ERROR("[rank {}/{}] Selected backend does not support "
					"distributed active fluid particle integration.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
			}
		}

		if (localOk)
		{
			const bool solverMayReOrderParticles = simBackend_->GetNumFluidParticles() > 0;
			simBackend_->ExecuteFullSimStep(params);

			benchmarksTracker_->StartBench({ Bench::BenchOp::Dist_StateSync });

			// skip solver step if no fluid particles to avoid re-application of neighbor search particle re-order
			{
				[[maybe_unused]] const DistSyncTimingScope syncTimingScope(
					&distSyncTimingReport_, DistSyncTimingOp::Reorder);
				if (solverMayReOrderParticles && !ApplyLastBackendParticleReOrder())
				{
					localOk = false;
				}
			}

			{
				[[maybe_unused]] const DistSyncTimingScope syncTimingScope(
					&distSyncTimingReport_, DistSyncTimingOp::Push);
				Sync::UniversalSimState backendSimState{};
				simBackend_->PushToUniversalSimState(backendSimState);
				localSimState_ = std::move(backendSimState);
				localSimState_.step = simStep_;
				localSimState_.time = simTime_;
			}

			benchmarksTracker_->StopBench({ Bench::BenchOp::Dist_StateSync });

			if (localOk && !ValidateRankLocalParticleMetadata(
				"after distributed solver step")) { localOk = false; }
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed distributed step: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	return mpiContext_.AllTrue(localOk);
}

bool DistSimulationMgr::MigrateOwnedFluidParticlesToOwnerRanks()
{
	if (!domainDecomp_) { return false; }

	bool localOk = true;

	try
	{
		ParticleMigrationStats stats{};
		std::string failMsg{};
		if (!MigrateFluidParticlesToOwnerRanks(
			localSimState_,
			fluidParticleState_,
			*domainDecomp_,
			mpiContext_,
			globalSimState_.fluidParticles.N_,
			particleMigrationWorkBuffers_,
			&stats,
			&failMsg))
		{
			SPDLOG_ERROR("[rank {}/{}] Failed to migrate fluid particles: {}",
				mpiContext_.Rank(), mpiContext_.Size(), failMsg);
			localOk = false;
		}
		lastParticleMigrationStats_ = stats;
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to migrate fluid particles: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	const bool allOk = mpiContext_.AllTrue(localOk);
	return allOk;
}

bool DistSimulationMgr::ValidateOwnedBoundaryParticlesStillOnRank() const
{
	if (!domainDecomp_) { return false; }

	bool localOk = true;
	for (const int localIdx : boundaryParticleState_.particleLocalIdx_Owned)
	{
		if (!IsLocalIdxInRange(localIdx, localSimState_.boundaryParticles.N_) ||
			!HasParticleMetadataAtLocalIdx(boundaryParticleState_, localIdx))
		{
			SPDLOG_ERROR("[rank {}/{}] Invalid owned boundary particle metadata.",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
			break;
		}

		const glm::vec3 position =
			BoundaryPositionAtLocalIdx(localSimState_.boundaryParticles, localIdx);
		if (domainDecomp_->OwnerRank(position) != mpiContext_.Rank())
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Owned boundary particle needs migration "
				"(localIdx={}, globalIdx={}).",
				mpiContext_.Rank(), mpiContext_.Size(), localIdx,
				boundaryParticleState_.particleGlobalIdx_FromLocalIdx[localIdx]);
			localOk = false;
			break;
		}
	}

	return localOk;
}

bool DistSimulationMgr::PullRankLocalStateIntoBackend()
{
	if (!simBackend_) { return false; }

	// transfer rebuilt rank-local SoA state into backend-specific data structures
	benchmarksTracker_->StartBench({ Bench::BenchOp::Dist_StateSync });
	{
		[[maybe_unused]] const DistSyncTimingScope syncTimingScope(
			&distSyncTimingReport_, DistSyncTimingOp::Pull);
		simBackend_->PullFromUniversalSimState(localSimState_, simParams_);
	}
	benchmarksTracker_->StopBench({ Bench::BenchOp::Dist_StateSync });

	return true;
}

bool DistSimulationMgr::ValidateRankLocalBackendParticleOrder(const char* phaseName) const
{
	const bool fluidOrderStable =
		FluidParticleOrderMatchesLocalIdxMap(
			localSimState_, globalSimState_, fluidParticleState_);
	const bool boundaryOrderStable =
		BoundaryParticleOrderMatchesLocalIdxMap(
			localSimState_, globalSimState_, boundaryParticleState_);

	if (!fluidOrderStable || !boundaryOrderStable)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Rank-local backend particle order changed {} "
			"(fluid={}, boundary={}) and DistParticleState localIdx metadata "
			"or payload does not match the backend state.",
			mpiContext_.Rank(), mpiContext_.Size(), phaseName,
			fluidOrderStable ? "stable" : "changed",
			boundaryOrderStable ? "stable" : "changed");
		return false;
	}

	return true;
}

bool DistSimulationMgr::ValidateRankLocalParticleMetadata(const char* phaseName) const
{
	std::string fluidFailMsg{};
	const bool fluidMetadataValid =
		fluidParticleState_.IsValidLocalParticleState(
			localSimState_.fluidParticles.N_,
			globalSimState_.fluidParticles.N_,
			&fluidFailMsg);
	std::string boundaryFailMsg{};
	const bool boundaryMetadataValid =
		boundaryParticleState_.IsValidLocalParticleState(
			localSimState_.boundaryParticles.N_,
			globalSimState_.boundaryParticles.N_,
			&boundaryFailMsg);

	if (!fluidMetadataValid || !boundaryMetadataValid)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Invalid rank-local particle metadata {} "
			"(fluid={}, boundary={})",
			mpiContext_.Rank(), mpiContext_.Size(), phaseName,
			fluidMetadataValid ? "valid" : fluidFailMsg,
			boundaryMetadataValid ? "valid" : boundaryFailMsg);
		return false;
	}

	return true;
}

bool DistSimulationMgr::ExchangeRankLocalHalos()
{
	bool localOk = true;

	try
	{
		if (!domainDecomp_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed to exchange halo particles (no domain decomp).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		else
		{
			HaloExchangeStats stats{};
			std::string failMsg{};
			if (!ExchangeHaloParticlesWithNeighborRanks(
				localSimState_,
				fluidParticleState_,
				boundaryParticleState_,
				*domainDecomp_,
				mpiContext_,
				haloExchangeWorkBuffers_,
				&stats,
				&failMsg))
			{
				SPDLOG_ERROR("[rank {}/{}] Failed to exchange halo particle packets: {}",
					mpiContext_.Rank(), mpiContext_.Size(), failMsg);
				localOk = false;
			}
			lastHaloExchangeStats_ = stats;

			std::string rebuildFailMsg{};
			if (localOk && !RebuildLocalStateWithReceivedHalos(
				haloExchangeWorkBuffers_.rankBuffers,
				localSimState_,
				fluidParticleState_,
				boundaryParticleState_,
				globalSimState_.fluidParticles.N_,
				globalSimState_.boundaryParticles.N_,
				haloRebuildScratch_,
				&rebuildFailMsg))
			{
				SPDLOG_ERROR("[rank {}/{}] Failed to rebuild rank-local state with received halos: {}",
					mpiContext_.Rank(), mpiContext_.Size(), rebuildFailMsg);
				localOk = false;
			}

		#if defined(_DEBUG)
			if (localOk)
			{
				if (!ValidateRankLocalSimState()) { localOk = false; }
			}
		#endif
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to exchange halo particles: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	const bool allOk = mpiContext_.AllTrue(localOk);
	return allOk;
}

bool DistSimulationMgr::ApplyLastBackendParticleReOrder()
{
	if (!simBackend_) { return false; }

	std::vector<uint32_t> oldFluidLocalIdx_FromNewLocalIdx;
	std::vector<uint32_t> oldBoundaryLocalIdx_FromNewLocalIdx;

	const bool hasParticleReOrder =
		simBackend_->TryGetLastParticleReOrder(
			oldFluidLocalIdx_FromNewLocalIdx,
			oldBoundaryLocalIdx_FromNewLocalIdx);

	if (!hasParticleReOrder) { return true; }

	bool localOk = true;

	if (!ApplyLocalParticleReOrder(
		fluidParticleState_,
		oldFluidLocalIdx_FromNewLocalIdx,
		globalSimState_.fluidParticles.N_))
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to apply backend fluid particle re-order.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	if (!ApplyLocalParticleReOrder(
		boundaryParticleState_,
		oldBoundaryLocalIdx_FromNewLocalIdx,
		globalSimState_.boundaryParticles.N_))
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to apply backend boundary particle re-order.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	return localOk;
}

bool DistSimulationMgr::TryReadProfileJson(
	const std::string& profile,
	std::string& jsonText)
{
	const std::filesystem::path paramsPath =
		std::filesystem::path(TSP_CONFIG_DIR) / profile / "params.json";

	std::ifstream file(paramsPath, std::ios::in | std::ios::binary);
	if (!file) { return false; }

	jsonText.assign(
		std::istreambuf_iterator<char>(file),
		std::istreambuf_iterator<char>());
	return true;
}
