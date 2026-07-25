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
#include "dist_statistics.h"
#include "halo_exchange.h"
#include "owned_particle_gather.h"
#include "particle_migration.h"
#include "rigid_body_sync.h"
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

	constexpr std::array<Mem::MemBucket, 10> BACKEND_MEM_BUCKETS =
	{
		Mem::MemBucket::ParticlePropertiesF,
		Mem::MemBucket::ParticlePropertiesB,
		Mem::MemBucket::ParticlePropertiesR,
		Mem::MemBucket::Neighbors_SearchDataStructF,
		Mem::MemBucket::Neighbors_SearchDataStructB,
		Mem::MemBucket::Neighbors_SearchDataStructR,
		Mem::MemBucket::Neighbors_ListStorageF,
		Mem::MemBucket::Neighbors_ListStorageB,
		Mem::MemBucket::Neighbors_ListStorageR,
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

		Mem::Add(usage, Mem::MemUsageOf_Vector(fluidParticles.id));
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

		Mem::Add(usage, Mem::MemUsageOf_Vector(boundaryParticles.id));
		Mem::Add(usage, Mem::MemUsageOf_Vector(boundaryParticles.xPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(boundaryParticles.yPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(boundaryParticles.zPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(boundaryParticles.volB));
		Mem::Add(usage, Mem::MemUsageOf_Vector(boundaryParticles.psiB));

		return usage;
	}

	Mem::MemUsage MemoryUsageOfRigidParticles(
		const Sync::RigidParticles& rigidParticles)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(rigidParticles.id));
		Mem::Add(usage, Mem::MemUsageOf_Vector(rigidParticles.bodyId));
		Mem::Add(usage, Mem::MemUsageOf_Vector(rigidParticles.xRelPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(rigidParticles.yRelPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(rigidParticles.zRelPos));
		Mem::Add(usage, Mem::MemUsageOf_Vector(rigidParticles.volR));
		Mem::Add(usage, Mem::MemUsageOf_Vector(rigidParticles.psiR));

		return usage;
	}

	Mem::MemUsage MemoryUsageOfUniversalSimState(
		const Sync::UniversalSimState& state)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, MemoryUsageOfFluidParticles(state.fluidParticles));
		Mem::Add(usage, MemoryUsageOfBoundaryParticles(state.boundaryParticles));
		Mem::Add(usage, MemoryUsageOfRigidParticles(state.rigidParticles));

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
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.idR));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.bodyIdR));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.xPosR));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.yPosR));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.zPosR));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.volR));

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

	Mem::MemUsage MemoryUsageOfRigidBodyForceTorque(
		const Sim::RigidBodyForceTorque& forceTorque)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(forceTorque.xForce));
		Mem::Add(usage, Mem::MemUsageOf_Vector(forceTorque.yForce));
		Mem::Add(usage, Mem::MemUsageOf_Vector(forceTorque.zForce));
		Mem::Add(usage, Mem::MemUsageOf_Vector(forceTorque.xTorque));
		Mem::Add(usage, Mem::MemUsageOf_Vector(forceTorque.yTorque));
		Mem::Add(usage, Mem::MemUsageOf_Vector(forceTorque.zTorque));

		return usage;
	}

	Mem::MemUsage MemoryUsageOfRigidBodySyncBuffers(
		const RigidBodySyncBuffers& buffers)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(buffers.forceTorqueValues));
		Mem::Add(usage, Mem::MemUsageOf_Vector(buffers.stateValues));

		return usage;
	}

	Mem::MemUsage MemoryUsageOfStatisticsPartial(
		const Stats::StatisticsPartial& partial)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(partial.nbrCountFF.histogram));
		Mem::Add(usage, Mem::MemUsageOf_Vector(partial.nbrCountFB.histogram));
		Mem::Add(usage, Mem::MemUsageOf_Vector(partial.nbrCountFR.histogram));
		Mem::Add(usage, Mem::MemUsageOf_Vector(partial.nbrCountRF.histogram));
		Mem::Add(usage, Mem::MemUsageOf_Vector(partial.nbrCountRB.histogram));
		Mem::Add(usage, Mem::MemUsageOf_Vector(partial.nbrCountRR.histogram));
		Mem::Add(usage, Mem::MemUsageOf_Vector(partial.velMag.histogram));

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
		if (backendFluid.id.size() != backendFluid.N_ ||
			globalFluid.id.size() != globalFluid.N_)
		{
			return false;
		}

		for (uint64_t localIdx = 0; localIdx < backendFluid.N_; localIdx++)
		{
			const int globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
			if (globalIdx < 0 || static_cast<uint32_t>(globalIdx) >= globalFluid.N_) { return false; }

			if (backendFluid.id[localIdx] != globalFluid.id[globalIdx] ||
				!AlmostEqual(backendFluid.xPos[localIdx], globalFluid.xPos[globalIdx]) ||
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
		if (backendBoundary.id.size() != backendBoundary.N_ ||
			globalBoundary.id.size() != globalBoundary.N_)
		{
			return false;
		}

		for (uint64_t localIdx = 0; localIdx < backendBoundary.N_; localIdx++)
		{
			const int globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
			if (globalIdx < 0 || static_cast<uint32_t>(globalIdx) >= globalBoundary.N_) { return false; }

			if (backendBoundary.id[localIdx] != globalBoundary.id[globalIdx] ||
				!AlmostEqual(backendBoundary.xPos[localIdx], globalBoundary.xPos[globalIdx]) ||
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

	bool RigidTopologyMatchesGlobalState(
		const Sync::UniversalSimState& rankState,
		const Sync::UniversalSimState& globalState)
	{
		const auto& rankRigid = rankState.rigidParticles;
		const auto& globalRigid = globalState.rigidParticles;
		if (rankRigid.N_ != globalRigid.N_ ||
			rankRigid.id != globalRigid.id ||
			rankRigid.bodyId != globalRigid.bodyId ||
			rankRigid.xRelPos != globalRigid.xRelPos ||
			rankRigid.yRelPos != globalRigid.yRelPos ||
			rankRigid.zRelPos != globalRigid.zRelPos ||
			rankRigid.volR != globalRigid.volR ||
			rankRigid.psiR != globalRigid.psiR)
		{
			return false;
		}

		const auto& rankBodies = rankState.rigidBodies;
		const auto& globalBodies = globalState.rigidBodies;
		return
			rankBodies.N_ == globalBodies.N_ &&
			rankBodies.id == globalBodies.id &&
			rankBodies.rigidParticleBegin == globalBodies.rigidParticleBegin &&
			rankBodies.rigidParticleCount == globalBodies.rigidParticleCount &&
			rankBodies.mass == globalBodies.mass &&
			rankBodies.massInv == globalBodies.massInv &&
			rankBodies.iBodyInv00 == globalBodies.iBodyInv00 &&
			rankBodies.iBodyInv01 == globalBodies.iBodyInv01 &&
			rankBodies.iBodyInv02 == globalBodies.iBodyInv02 &&
			rankBodies.iBodyInv10 == globalBodies.iBodyInv10 &&
			rankBodies.iBodyInv11 == globalBodies.iBodyInv11 &&
			rankBodies.iBodyInv12 == globalBodies.iBodyInv12 &&
			rankBodies.iBodyInv20 == globalBodies.iBodyInv20 &&
			rankBodies.iBodyInv21 == globalBodies.iBodyInv21 &&
			rankBodies.iBodyInv22 == globalBodies.iBodyInv22;
	}

	bool RigidStateMatchesGlobalState(
		const Sync::UniversalSimState& rankState,
		const Sync::UniversalSimState& globalState)
	{
		const auto& rankBodies = rankState.rigidBodies;
		const auto& globalBodies = globalState.rigidBodies;
		return
			rankBodies.xCM == globalBodies.xCM &&
			rankBodies.yCM == globalBodies.yCM &&
			rankBodies.zCM == globalBodies.zCM &&
			rankBodies.xVel == globalBodies.xVel &&
			rankBodies.yVel == globalBodies.yVel &&
			rankBodies.zVel == globalBodies.zVel &&
			rankBodies.xOmega == globalBodies.xOmega &&
			rankBodies.yOmega == globalBodies.yOmega &&
			rankBodies.zOmega == globalBodies.zOmega &&
			rankBodies.xAngMom == globalBodies.xAngMom &&
			rankBodies.yAngMom == globalBodies.yAngMom &&
			rankBodies.zAngMom == globalBodies.zAngMom &&
			rankBodies.a00 == globalBodies.a00 &&
			rankBodies.a01 == globalBodies.a01 &&
			rankBodies.a02 == globalBodies.a02 &&
			rankBodies.a10 == globalBodies.a10 &&
			rankBodies.a11 == globalBodies.a11 &&
			rankBodies.a12 == globalBodies.a12 &&
			rankBodies.a20 == globalBodies.a20 &&
			rankBodies.a21 == globalBodies.a21 &&
			rankBodies.a22 == globalBodies.a22;
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

	SPDLOG_INFO_MPI("Init distributed simulation backend...", mpiContext_);
	SelectSimBackend(simParams_.backendVersion);
	bool backendSupported = simBackend_ && simBackend_->SupportsDistributedSimulation();
	if (simBackend_ && !backendSupported)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Sim backend {} does not support distributed simulation. "
			"CPU01 is currently the only distributed backend.",
			mpiContext_.Rank(), mpiContext_.Size(), simBackend_->Name());
	}
	if (!mpiContext_.AllTrue(backendSupported)) { return false; }

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
	statisticsTracker_.reset();
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
		SPDLOG_ERROR("[rank {}/{}] Failed to create sim scenario ({}).",
			mpiContext_.Rank(), mpiContext_.Size(), static_cast<int>(type));
		return;
	}
}

void DistSimulationMgr::SelectSimBackend(const Base::BackendVersion version)
{
	simBackend_.reset();
	statisticsTracker_.reset();
	memoryTracker_.reset();
	benchmarksTracker_.reset();
	curBenchOps_.clear();

	// create new metric trackers and backend over rank-local data
	benchmarksTracker_ = std::make_unique<Bench::BenchmarksTracker>(simParams_);
	memoryTracker_ = std::make_unique<Mem::MemoryTracker>(simParams_);
	statisticsTracker_ = std::make_unique<Stats::StatisticsTracker>(simParams_);
	try
	{
		simBackend_ = Sim::MakeSimBackend(
			version, simParamsMgr_, *benchmarksTracker_, simParams_);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to create distributed sim backend ({}): {} "
			"CPU01 is currently the only distributed backend.",
			mpiContext_.Rank(), mpiContext_.Size(),
			static_cast<int>(version), ex.what());
		return;
	}
	if (!simBackend_)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to create distributed sim backend ({}). "
			"CPU01 is currently the only distributed backend.",
			mpiContext_.Rank(), mpiContext_.Size(), static_cast<int>(version));
		return;
	}

	simBackend_->GetBenchOps(curBenchOps_);
	const Bench::BenchOpList distBenchOps =
	{
		Bench::BenchOp::Dist_ParticleMigration,
		Bench::BenchOp::Dist_ParticleClass,
		Bench::BenchOp::Dist_HaloExchange,
		Bench::BenchOp::Dist_StateSync
	};
	auto distBenchInsertIt = curBenchOps_.end();
	for (auto it = curBenchOps_.begin(); it != curBenchOps_.end(); it++)
	{
		if (*it == Bench::BenchOp::Visualization)
		{
			distBenchInsertIt = it;
			distBenchInsertIt++;
			break;
		}
	}
	if (distBenchInsertIt == curBenchOps_.end())
	{
		for (auto it = curBenchOps_.begin(); it != curBenchOps_.end(); it++)
		{
			if (*it == Bench::BenchOp::Total)
			{
				distBenchInsertIt = it;
				break;
			}
		}
	}
	curBenchOps_.insert(distBenchInsertIt, distBenchOps.begin(), distBenchOps.end());

	if (!Bench::ValidateOpList(curBenchOps_))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid distributed benchmark operation list "
			"for sim backend ({}).",
			mpiContext_.Rank(), mpiContext_.Size(), static_cast<int>(version));
		curBenchOps_.clear();
	}
}

bool DistSimulationMgr::ReBuildInitialSimState()
{
	bool localOk = true;

	try
	{
		if (!simScenario_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed to re-build initial sim state (no scenario selected).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		if (!domainDecomp_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed to re-build initial sim state (no domain decomp).",
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
			if (localOk && !ValidateRankLocalSimState(true)) { localOk = false; }
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to re-build initial sim state: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	if (!mpiContext_.AllTrue(localOk)) { return false; }

	expectedGlobalFluidParticleCount_ = globalSimState_.fluidParticles.N_;
	expectedGlobalBoundaryParticleCount_ = globalSimState_.boundaryParticles.N_;
	mpiContext_.BroadcastUint64(expectedGlobalFluidParticleCount_, mpiContext_.RootRank());
	mpiContext_.BroadcastUint64(expectedGlobalBoundaryParticleCount_, mpiContext_.RootRank());

	const bool globalParticleCountsMatch =
		globalSimState_.fluidParticles.N_ ==
			expectedGlobalFluidParticleCount_ &&
		globalSimState_.boundaryParticles.N_ ==
			expectedGlobalBoundaryParticleCount_;
	if (!mpiContext_.AllTrue(globalParticleCountsMatch))
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Global particle counts differ across ranks.",
			mpiContext_.Rank(), mpiContext_.Size());
		return false;
	}

	try
	{
		rigidBodySyncBuffers_.Resize(globalSimState_.rigidBodies.N_);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to prepare rigid body synchronization buffers: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	return ValidateGlobalOwnedParticleCounts("after initial rank-local build");
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
	report.globalRigidParticleCount = globalSimState_.rigidParticles.N_;
	report.globalRigidBodyCount = globalSimState_.rigidBodies.N_;

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
	report.replicatedRigidParticleCount = localSimState_.rigidParticles.N_;

	return report;
}

bool DistSimulationMgr::RunSim_Blocking()
{
	bool localOk = true;

	if (!initialized_.load(std::memory_order_acquire))
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to launch blocking DistSimulationMgr (not initialized).",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!simBackend_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to launch blocking DistSimulationMgr (no backend selected).",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!benchmarksTracker_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to launch blocking DistSimulationMgr (no benchmark tracker).",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!memoryTracker_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to launch blocking DistSimulationMgr (no memory tracker).",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!statisticsTracker_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to launch blocking DistSimulationMgr (no statistics tracker).",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	if (running_.exchange(true))
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to launch blocking DistSimulationMgr (already running).",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk))
	{
		running_.store(false, std::memory_order_release);
		return false;
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
		if (!WriteExportSnapshot(paramsInit)) { runOk = false; }
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
		if (simStep_ >= static_cast<uint64_t>(paramsInit.maxSteps) &&
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
		statisticsTracker_->BeginStep();
		lastParticleMigrationStats_ = {};
		lastHaloExchangeStats_ = {};

		// starting point of total benchmark
		benchmarksTracker_->StartBench(Bench::BenchOp::Total);

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
			if (!WriteExportSnapshot(params)) { runOk = false; }
		}

		if (runOk && !FillDistributedStatisticsRecord(params))
		{
			runOk = false;
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
		runOk = false;
	}

	running_.store(false, std::memory_order_release);
	EndRunLogging();

	SPDLOG_INFO_MPI("Distributed simulation stopped running", mpiContext_);
	if (mpiContext_.IsRoot())
	{
		printf("\n");
	}

	return runOk;
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
	globalSimState_.step = 0;
	globalSimState_.time = 0.0;

	// all ranks build the same full scenario before rank-local compaction
	simScenario_->Build(globalSimState_, simParams_);
	globalSimState_.fluidParticles.ensureParticleIdsPresent();
	globalSimState_.boundaryParticles.ensureParticleIdsPresent();
	globalSimState_.rigidParticles.ensureParticleIdsPresent();
	globalSimState_.boundaryParticles.ensureBoundaryCoefficientsPresent(
		simParams_.massB, simParams_.restDensity);
	globalSimState_.rigidParticles.ensureRigidCoefficientsPresent(
		simParams_.massB, simParams_.restDensity);
	globalSimState_.ValidateRigidAttachments();
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
	activeRigidParticleLocalIdxs_.clear();
	rigidBodyForceTorque_ = {};

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

bool DistSimulationMgr::ValidateRankLocalSimState(
	const bool includeRigidState) const
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

	if (includeRigidState)
	{
		if (!RigidTopologyMatchesGlobalState(localSimState_, globalSimState_))
		{
			SPDLOG_ERROR("[rank {}/{}] Invalid replicated rank-local rigid topology.",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		else
		{
			try
			{
				localSimState_.ValidateRigidAttachments();
			}
			catch (const std::exception& ex)
			{
				SPDLOG_ERROR("[rank {}/{}] Invalid rank-local rigid attachments: {}",
					mpiContext_.Rank(), mpiContext_.Size(), ex.what());
				localOk = false;
			}
		}
	}

	return localOk;
}

bool DistSimulationMgr::ValidateGlobalOwnedParticleCounts(
	const char* phaseName) const
{
	// the global particle counts are an initialization invariant checked once in
	// ReBuildInitialSimState(), so only the current ownership sums are re-checked per step
	const uint64_t globalOwnedFluidParticleCount =
		mpiContext_.AllReduceSumUint64(fluidParticleState_.ownedParticleCount);
	const uint64_t globalOwnedBoundaryParticleCount =
		mpiContext_.AllReduceSumUint64(boundaryParticleState_.ownedParticleCount);

	const bool countsValid =
		globalOwnedFluidParticleCount ==
			expectedGlobalFluidParticleCount_ &&
		globalOwnedBoundaryParticleCount ==
			expectedGlobalBoundaryParticleCount_;
	if (!countsValid)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Invalid global owned particle counts {} "
			"(fluid={}/{}, boundary={}/{}).",
			mpiContext_.Rank(), mpiContext_.Size(), phaseName,
			globalOwnedFluidParticleCount,
			expectedGlobalFluidParticleCount_,
			globalOwnedBoundaryParticleCount,
			expectedGlobalBoundaryParticleCount_);
	}

	return mpiContext_.AllTrue(countsValid);
}

bool DistSimulationMgr::InitRankLocalBackend()
{
	bool localOk = true;

	try
	{
		SyncParameters();

		if (!simBackend_)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to initialize rank-local backend "
				"(no distributed backend selected).",
				mpiContext_.Rank(), mpiContext_.Size());
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

			if (simBackend_->GetNumRigidParticles() != globalSimState_.rigidParticles.N_)
			{
				SPDLOG_ERROR("[rank {}/{}] Rank-local backend rigid count mismatch.",
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

bool DistSimulationMgr::WriteExportSnapshot(const Sim::SimParams& params) const
{
	DistSnapshotGatherStats stats{};

	benchmarksTracker_->StartBench(Bench::BenchOp::Serialization);

	auto& exportFluidParticleLocalIdxs = exportFluidParticleLocalIdxsScratch_;
	if (!BuildExportFluidParticleLocalIdxs(exportFluidParticleLocalIdxs))
	{
		benchmarksTracker_->StopBench(Bench::BenchOp::Serialization);
		return false;
	}

	// fill export snapshot
	auto& expSnap = exportSnapScratch_;
	const bool gatherOk = GatherExportSnapshotToRoot(
		expSnap,
		localSimState_,
		fluidParticleState_,
		boundaryParticleState_,
		exportFluidParticleLocalIdxs,
		mpiContext_,
		simStep_,
		simTime_,
		&stats);

	if (!gatherOk)
	{
		benchmarksTracker_->StopBench(Bench::BenchOp::Serialization);
		return false;
	}

	bool writeOk = true;
	if (mpiContext_.IsRoot())
	{
		try
		{
			// write export snapshot to files
			serializationMgr_.SaveExportSnapshot(expSnap, params.exportDir);
			serializationMgr_.SaveExportSnapshotXdmfViews(expSnap, params.exportDir);
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to write distributed export snapshot: {}",
				mpiContext_.Rank(), mpiContext_.Size(), ex.what());
			writeOk = false;
		}
	}

	benchmarksTracker_->StopBench(Bench::BenchOp::Serialization);

	return mpiContext_.AllTrue(writeOk);
}

bool DistSimulationMgr::BuildExportFluidParticleLocalIdxs(
	std::vector<int>& localIdxs) const
{
	bool localOk = true;

	if (!simBackend_)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to select distributed fluid surface particles "
			"(no backend selected).",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	if (localOk &&
		(!simBackend_->FillFluidParticleNearSurfaceFlags(localNearSurfaceFlagsScratch_) ||
		localNearSurfaceFlagsScratch_.size() !=
			static_cast<uint64_t>(localSimState_.fluidParticles.N_)))
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to classify rank-local fluid particles "
			"for distributed surface export.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	// sized from the global count on every rank, so the reconciliation below stays collective
	// even when a rank has already failed and only contributes its zero-initialized flags
	auto& globalNearSurfaceFlags = globalNearSurfaceFlagsScratch_;
	globalNearSurfaceFlags.assign(globalSimState_.fluidParticles.N_, 0);

	if (localOk)
	{
		for (const int localIdx : fluidParticleState_.particleLocalIdx_Owned)
		{
			if (!IsLocalIdxInRange(localIdx, localSimState_.fluidParticles.N_) ||
				!HasParticleMetadataAtLocalIdx(fluidParticleState_, localIdx))
			{
				SPDLOG_ERROR(
					"[rank {}/{}] Invalid owned fluid particle metadata "
					"during distributed surface export.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
				break;
			}

			const int globalIdx =
				fluidParticleState_.particleGlobalIdx_FromLocalIdx[localIdx];
			if (globalIdx < 0 ||
				static_cast<uint32_t>(globalIdx) >= globalSimState_.fluidParticles.N_)
			{
				SPDLOG_ERROR(
					"[rank {}/{}] Invalid owned fluid particle global index "
					"during distributed surface export (localIdx={}, globalIdx={}).",
					mpiContext_.Rank(), mpiContext_.Size(), localIdx, globalIdx);
				localOk = false;
				break;
			}

			globalNearSurfaceFlags[globalIdx] =
				localNearSurfaceFlagsScratch_[localIdx];
		}
	}

	mpiContext_.AllReduceMaxUint8s(globalNearSurfaceFlags);

	if (localOk)
	{
		for (uint32_t localIdx = 0;
			localIdx < localSimState_.fluidParticles.N_;
			localIdx++)
		{
			if (!HasParticleMetadataAtLocalIdx(
				fluidParticleState_, static_cast<int>(localIdx)))
			{
				localOk = false;
				break;
			}

			const int globalIdx =
				fluidParticleState_.particleGlobalIdx_FromLocalIdx[localIdx];
			if (globalIdx < 0 ||
				static_cast<uint32_t>(globalIdx) >= globalSimState_.fluidParticles.N_)
			{
				localOk = false;
				break;
			}

			localNearSurfaceFlagsScratch_[localIdx] =
				globalNearSurfaceFlags[globalIdx];
		}
	}

	if (localOk && !simBackend_->BuildExportFluidParticleLocalIdxs(
		localNearSurfaceFlagsScratch_,
		localIdxs))
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to select distributed fluid surface particles.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	return mpiContext_.AllTrue(localOk);
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
	Mem::Set(record, Mem::MemSpace::RAM, Mem::MemBucket::Dist_GlobalState,
		MemoryUsageOfUniversalSimState(globalSimState_));
	Mem::Set(record, Mem::MemSpace::RAM, Mem::MemBucket::Dist_LocalState,
		MemoryUsageOfUniversalSimState(localSimState_));

	Mem::MemUsage particleMetadataUsage{};
	Mem::Add(particleMetadataUsage, MemoryUsageOfParticleState(fluidParticleState_));
	Mem::Add(particleMetadataUsage, MemoryUsageOfParticleState(boundaryParticleState_));
	Mem::Set(record, Mem::MemSpace::RAM, Mem::MemBucket::Dist_ParticleMetadata,
		particleMetadataUsage);

	Mem::MemUsage exchangeBufferUsage{};
	Mem::Add(exchangeBufferUsage,
		MemoryUsageOfParticleMigrationExchangeBuffers(particleMigrationWorkBuffers_));
	Mem::Add(exchangeBufferUsage, MemoryUsageOfHaloExchangeWorkBuffers(haloExchangeWorkBuffers_));
	Mem::Set(record, Mem::MemSpace::RAM, Mem::MemBucket::Dist_ExchangeBuffers, exchangeBufferUsage);

	Mem::MemUsage miscUsage{};
	Mem::Add(miscUsage, MemoryUsageOfParticleMigrationScratch(particleMigrationWorkBuffers_));
	Mem::Add(miscUsage,
		MemoryUsageOfParticleClassificationWorkBuffers(fluidParticleClassificationWorkBuffers_));
	Mem::Add(miscUsage,
		MemoryUsageOfParticleClassificationWorkBuffers(boundaryParticleClassificationWorkBuffers_));
	Mem::Add(miscUsage, MemoryUsageOfHaloRebuildScratch(haloRebuildScratch_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(activeRigidParticleLocalIdxs_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(activeRigidParticleRankCountScratch_));
	Mem::Add(miscUsage, MemoryUsageOfRigidBodyForceTorque(rigidBodyForceTorque_));
	Mem::Add(miscUsage, MemoryUsageOfRigidBodySyncBuffers(rigidBodySyncBuffers_));
	Mem::Add(miscUsage, MemoryUsageOfStatisticsPartial(statisticsPartialScratch_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(localNearSurfaceFlagsScratch_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(globalNearSurfaceFlagsScratch_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(exportFluidParticleLocalIdxsScratch_));
	Mem::Add(miscUsage, MemoryUsageOfExportSnapshot(exportSnapScratch_));
	Mem::Set(record, Mem::MemSpace::RAM, Mem::MemBucket::Dist_Misc, miscUsage);
}

bool DistSimulationMgr::FillDistributedStatisticsRecord(
	const Sim::SimParams& params) const
{
	bool localOk = true;

	if (!simBackend_ || !statisticsTracker_)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to fill distributed statistics record "
			"(no backend or statistics tracker).",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	auto& partial = statisticsPartialScratch_;
	if (localOk && !simBackend_->FillStatisticsPartial(partial, params))
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to fill rank-local distributed "
			"statistics partial.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	partial.N_B = boundaryParticleState_.ownedParticleCount;
	FinalizeDistributedStatisticsRecord(
		statisticsTracker_->GetCurrentRecord(),
		partial,
		params,
		mpiContext_);

	statisticsTracker_->EndStep(simStep_, simTime_);
	return true;
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

	// ending point of total benchmark, archive data and update aggregates
	benchmarksTracker_->StopBench(Bench::BenchOp::Total);
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
			benchmarksTracker_->GetBench_Latest(Bench::BenchOp::Total);
		report.totalStepAvgNs =
			benchmarksTracker_->GetBench_Avg_OverAll(Bench::BenchOp::Total);
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
		mpiContext_, params, curBenchOps_,
		BuildRankStepReport(params, exportedStep),
		BuildRankBenchReport(),
		distSyncTimingReport_);
}

bool DistSimulationMgr::ExecuteDistributedStep(const Sim::SimParams& params)
{
	// results of [ rank-reduced ] helpers are returned directly, because every rank already
	// agrees on them, while rank-local results are reduced before the next collective runs
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

		SyncParameters();

		// boundary particles are still treated as static owned data
		if (!ValidateOwnedBoundaryParticlesStillOnRank()) { localOk = false; }
		if (!mpiContext_.AllTrue(localOk)) { return false; }

		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_ParticleMigration);
		const bool migrationOk = MigrateOwnedFluidParticlesToOwnerRanks();
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_ParticleMigration);
		if (!migrationOk) { return false; }

		if (!ValidateGlobalOwnedParticleCounts("after fluid migration"))
		{
			return false;
		}

		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_ParticleClass);

		// refresh inner/outer lists after migration and backend order changes
		ClassifyOwnedFluidParticlesIntoInnerOuter(
			fluidParticleState_, localSimState_.fluidParticles, *domainDecomp_,
			mpiContext_.Rank(), params.kernelRadius,
			&fluidParticleClassificationWorkBuffers_);
		ClassifyOwnedBoundaryParticlesIntoInnerOuter(
			boundaryParticleState_, localSimState_.boundaryParticles, *domainDecomp_,
			mpiContext_.Rank(), params.kernelRadius,
			&boundaryParticleClassificationWorkBuffers_);

		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_ParticleClass);

		// halo exchange only appends received halos, so ownership is unchanged here and
		// does not have to be re-validated against the global counts
		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_HaloExchange);
		const bool haloOk = ExchangeRankLocalHalos();
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_HaloExchange);
		if (!haloOk) { return false; }

		if (!PrepareRankLocalBackendStep()) { return false; }

		try
		{
			simBackend_->ExecuteFullSimStep(params);
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed rank-local backend solver step: {}",
				mpiContext_.Rank(), mpiContext_.Size(), ex.what());
			localOk = false;
		}
		if (!mpiContext_.AllTrue(localOk)) { return false; }

		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_StateSync);
		try
		{
			// skip solver step if no fluid particles to avoid re-application of
			// neighbor search particle re-order
			if (simBackend_->GetNumFluidParticles() > 0)
			{
				[[maybe_unused]] const DistSyncTimingScope syncTimingScope(
					&distSyncTimingReport_, DistSyncTimingOp::Reorder);
				if (!ApplyLastBackendParticleReOrder()) { localOk = false; }
			}
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to apply backend particle re-order: {}",
				mpiContext_.Rank(), mpiContext_.Size(), ex.what());
			localOk = false;
		}
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_StateSync);
		if (!mpiContext_.AllTrue(localOk)) { return false; }

		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_StateSync);
		const bool finishOk = FinishRankLocalBackendStep(params);
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_StateSync);
		if (!finishOk) { return false; }

		if (!ValidateRankLocalParticleMetadata("after distributed solver step")) { localOk = false; }
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
	bool localOk = true;

	try
	{
		if (!domainDecomp_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed to migrate fluid particles (no domain decomp).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}

		ParticleMigrationStats stats{};
		std::string failMsg{};
		if (localOk && !MigrateFluidParticlesToOwnerRanks(
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

bool DistSimulationMgr::PrepareRankLocalBackendStep()
{
	// derived from the replicated global topology, so every rank agrees on whether the
	// rigid body collectives below run at all
	const bool hasRigidBodies =
		globalSimState_.rigidParticles.N_ > 0 ||
		globalSimState_.rigidBodies.N_ > 0;

	bool localOk = true;
	activeRigidParticleLocalIdxs_.clear();

	try
	{
		std::string failMsg{};
		if (hasRigidBodies && !BuildActiveRigidParticleLocalIdxs(
			activeRigidParticleLocalIdxs_,
			localSimState_,
			*domainDecomp_,
			mpiContext_.Rank(),
			mpiContext_.Size(),
			&failMsg))
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to classify active rigid particles: {}",
				mpiContext_.Rank(), mpiContext_.Size(), failMsg);
			localOk = false;
		}

		if (!mpiContext_.AllTrue(localOk)) { return false; }

		if (hasRigidBodies)
		{
			auto& activeRankCounts = activeRigidParticleRankCountScratch_;
			activeRankCounts.assign(localSimState_.rigidParticles.N_, 0);
			for (const int localIdx : activeRigidParticleLocalIdxs_)
			{
				activeRankCounts[localIdx] = 1;
			}
			mpiContext_.AllReduceSumUint32s(activeRankCounts);

			const auto invalidAuthorityIt = std::find_if(
				activeRankCounts.begin(),
				activeRankCounts.end(),
				[](const uint32_t rankCount) { return rankCount != 1; });
			if (invalidAuthorityIt != activeRankCounts.end())
			{
				const uint32_t rigidParticleIdx = static_cast<uint32_t>(
					std::distance(activeRankCounts.begin(), invalidAuthorityIt));
				SPDLOG_ERROR(
					"[rank {}/{}] Rigid particle {} has {} authoritative ranks "
					"(expected exactly one).",
					mpiContext_.Rank(), mpiContext_.Size(),
					rigidParticleIdx, *invalidAuthorityIt);
				localOk = false;
			}
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to classify rank-local distributed work: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_StateSync);
	try
	{
		[[maybe_unused]] const DistSyncTimingScope syncTimingScope(
			&distSyncTimingReport_, DistSyncTimingOp::Pull);
		simBackend_->PullFromUniversalSimState(localSimState_, simParams_);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to pull rank-local backend state: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	benchmarksTracker_->StopBench(Bench::BenchOp::Dist_StateSync);
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_StateSync);
	try
	{
		bool configured = false;
		{
			[[maybe_unused]] const DistSyncTimingScope syncTimingScope(
				&distSyncTimingReport_, DistSyncTimingOp::ActiveIdx);
			configured = simBackend_->ConfigureDistributedStep(
				fluidParticleState_.particleLocalIdx_Owned,
				activeRigidParticleLocalIdxs_);
		}
		if (!configured)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to configure rank-local distributed "
				"backend step.",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to configure rank-local distributed "
			"backend step: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	benchmarksTracker_->StopBench(Bench::BenchOp::Dist_StateSync);

	return mpiContext_.AllTrue(localOk);
}

bool DistSimulationMgr::FinishRankLocalBackendStep(
	const Sim::SimParams& params)
{
	// derived from the replicated global topology, so every rank agrees on whether the
	// rigid body collectives below run at all
	const bool hasRigidBodies =
		globalSimState_.rigidParticles.N_ > 0 ||
		globalSimState_.rigidBodies.N_ > 0;
	bool localOk = true;

	if (!simBackend_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to finish distributed step (no backend selected).",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	if (localOk && hasRigidBodies)
	{
		try
		{
			if (!simBackend_->GetRigidBodyForceTorque(rigidBodyForceTorque_))
			{
				SPDLOG_ERROR(
					"[rank {}/{}] Failed to read rank-local rigid body force and torque.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
			}
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to read rank-local rigid body force and torque: {}",
				mpiContext_.Rank(), mpiContext_.Size(), ex.what());
			localOk = false;
		}
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	if (hasRigidBodies)
	{
		SumRigidBodyForceTorqueAcrossRanks(
			rigidBodyForceTorque_,
			rigidBodySyncBuffers_,
			mpiContext_);

		if (mpiContext_.IsRoot())
		{
			AddRigidBodyGravity(
				rigidBodyForceTorque_,
				globalSimState_.rigidBodies,
				params.gravity);
		}

		try
		{
			if (mpiContext_.IsRoot() &&
				!simBackend_->ApplyRigidBodyForceTorque(
					rigidBodyForceTorque_,
					params.timeStep))
			{
				SPDLOG_ERROR(
					"[rank {}/{}] Failed to apply synchronized rigid body force and torque.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
			}
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to apply synchronized rigid body force and torque: {}",
				mpiContext_.Rank(), mpiContext_.Size(), ex.what());
			localOk = false;
		}
		if (!mpiContext_.AllTrue(localOk)) { return false; }
	}

	try
	{
		[[maybe_unused]] const DistSyncTimingScope syncTimingScope(
			&distSyncTimingReport_, DistSyncTimingOp::Push);
		Sync::UniversalSimState backendSimState{};
		simBackend_->PushToUniversalSimState(backendSimState);
		localSimState_ = std::move(backendSimState);
		localSimState_.step = simStep_;
		localSimState_.time = simTime_;
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to push rank-local backend state: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	if (hasRigidBodies)
	{
		BroadcastRigidBodyState(
			localSimState_.rigidBodies,
			rigidBodySyncBuffers_,
			mpiContext_);
	}

	// every earlier exit is already reduced, so reaching this point means all ranks succeeded
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
	const bool rigidStateStable =
		RigidStateMatchesGlobalState(localSimState_, globalSimState_);

	if (!fluidOrderStable || !boundaryOrderStable || !rigidStateStable)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Rank-local backend particle order changed {} "
			"(fluid={}, boundary={}, rigid={}) and DistParticleState localIdx metadata "
			"or payload does not match the backend state.",
			mpiContext_.Rank(), mpiContext_.Size(), phaseName,
			fluidOrderStable ? "stable" : "changed",
			boundaryOrderStable ? "stable" : "changed",
			rigidStateStable ? "stable" : "changed");
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
			"(fluid={}, boundary={}).",
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
				if (!ValidateRankLocalSimState(false)) { localOk = false; }
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
