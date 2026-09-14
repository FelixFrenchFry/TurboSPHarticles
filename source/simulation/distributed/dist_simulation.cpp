// dist_simulation.cpp

// external
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <span>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <tbb/blocked_range.h>
#include <tbb/parallel_scan.h>
#include <tbb/partitioner.h>

// internal
#include "source/base/enum_hub.h"
#include "source/base/helpers.h"
#include "source/eval/memory/memory.h"
#include "source/simulation/scene/scenario_sim.h"
#include "dist_diagnostics.h"
#include "dist_simulation.h"
#include "dist_helpers.h"
#include "dist_snapshots.h"
#include "dist_statistics.h"
#include "rigid_body_sync.h"



using namespace Dist;

namespace
{
	// checksum over the particle IDs owned by this rank, summed across all ranks it has
	// to stay at the value the initial partition produced (wrap-around sums are intended)
	uint64_t OwnedParticleIdChecksum(
		const DistParticleState& particleState,
		const std::span<const uint64_t> particleIds)
	{
		uint64_t checksum = 0;

		for (const uint32_t localIdx : particleState.particleLocalIdx_Owned)
		{
			if (localIdx >= particleIds.size()) { continue; }

			// using SplitMix64-style mix
			uint64_t x = static_cast<uint64_t>(
				particleIds[localIdx]) + 0x9E3779B97F4A7C15ull;
			x ^= x >> 30;
			x *= 0xBF58476D1CE4E5B9ull;
			x ^= x >> 27;
			x *= 0x94D049BB133111EBull;
			x ^= x >> 31;

			checksum += x;
		}

		return checksum;
	}

	// build the global domain AABB that will be split between MPI ranks
	// (decomp starts from one global AABB, while scenarios describe their bounds with OBBs)
	AABB ComputeGlobalDomainBounds(
		const std::vector<Scene::OBB>& boundaryBoxes,
		const Sim::SimParams& params)
	{
		AABB globalBounds{};
		if (!Scene::ComputeSimDomainBounds(
			boundaryBoxes, params, globalBounds.min, globalBounds.max))
		{
			throw std::runtime_error("Scenario does not provide boundary boxes.");
		}

		return globalBounds;
	}

	Sim::SimParams MakeInitialDomainDecompProbeParams(const Sim::SimParams& params)
	{
		Sim::SimParams probeParams = params;
		probeParams.particleSizeF *= INITIAL_DOMAIN_DECOMP_PROFILE_PARTICLE_SIZE_MULTIPLIER;
		probeParams.particleSizeB *= INITIAL_DOMAIN_DECOMP_PROFILE_PARTICLE_SIZE_MULTIPLIER;
		probeParams.particleSizeInvF = 1.0f / probeParams.particleSizeF;
		probeParams.particleSizeInvB = 1.0f / probeParams.particleSizeB;

		probeParams.kernelRadius = probeParams.kernelSuppRatio * probeParams.particleSizeF;
		probeParams.kernelRadiusSqu = probeParams.kernelRadius * probeParams.kernelRadius;
		probeParams.distHaloRadius = probeParams.distHaloMode == Base::DistHaloMode::DoubleHalo
			? 2.0f * probeParams.kernelRadius
			: probeParams.kernelRadius;

		probeParams.massF = probeParams.restDensity * probeParams.particleSizeF
			* probeParams.particleSizeF * probeParams.particleSizeF;
		probeParams.massB = probeParams.restDensity * probeParams.particleSizeB
			* probeParams.particleSizeB * probeParams.particleSizeB;

		probeParams.surfaceSamplingMinDistance =
			probeParams.surfaceSamplingMinDistanceRatio * probeParams.particleSizeB;
		probeParams.surfaceSamplingMaxDistance =
			probeParams.surfaceSamplingMaxDistanceRatio * probeParams.particleSizeB;
		probeParams.surfaceSamplingCoverageProbeSpacing =
			probeParams.surfaceSamplingCoverageProbeSpacingRatio * probeParams.particleSizeB;

		return probeParams;
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
		loggingOptions.rankSubDir = FormatRankLogDirName(mpiContext.Rank());
		// the export directory is written by the root rank only
		loggingOptions.writeRunMetaToExportDir = mpiContext.IsRoot();
		loggingOptions.mpiRank = mpiContext.Rank();
		loggingOptions.N_mpiRanks = mpiContext.Size();
		return loggingOptions;
	}

	Mem::MemSpace PrimaryMemSpace(const Base::BackendType backendType)
	{
		return backendType == Base::BackendType::GPU
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
			total.N_usedBytes += memoryTracker.GetMem_Max_Used(space, bucket);
			total.N_allocBytes += memoryTracker.GetMem_Max_Alloc(space, bucket);
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
			total.N_usedBytes += memoryTracker.GetMem_Max_Used(space, bucket);
			total.N_allocBytes += memoryTracker.GetMem_Max_Alloc(space, bucket);
		}

		return total;
	}

	Mem::MemUsage MemoryUsageOfExportSnapshot(
		const Sync::ExportSnapshot& snap)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.xPosF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.yPosF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.zPosF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.velMagF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.rankF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.xPosB));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.yPosB));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.zPosB));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.volB));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.bodyIdR));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.xPosR));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.yPosR));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.zPosR));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.volR));

		return usage;
	}

	Mem::MemUsage MemoryUsageOfExportSnapshotGatherBuffer(
		const DistExportSnapshotBuffer& gatherBuffer)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(gatherBuffer.particlesF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(gatherBuffer.particlesB));

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

	bool RigidTopologyMatchesGlobalState(
		const Sync::UniversalSimState& rankState,
		const Sync::UniversalSimState& globalState)
	{
		const auto& rankRigid = rankState.particlesR;
		const auto& globalRigid = globalState.particlesR;
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

} // namespace anonymous

DistSimulationMgr::DistSimulationMgr(MpiContext& mpiContext,
									 Sim::SimParamsMgr& simParamsMgr)
	: mpiContext_{ mpiContext }
	, simParamsMgr_{ simParamsMgr }
	, loggingMgr_{ MakeRankLoggingOptions(mpiContext) }
{
	// nothing to do
}

bool DistSimulationMgr::Init(
	std::string& profile,
	const std::vector<std::string>& simOverrides,
	const bool logParamStatus,
	const std::string& restartCheckpoint)
{
	initTime_ = std::chrono::steady_clock::now();

	// initial snapshot
	SyncParameters(true);

	boundaryRedistributionRequired_ = true;
	if (!LoadParamsFromProfileBroadcast(profile, logParamStatus))
	{
		return false;
	}

	configProfile_ = profile;
	paramsFile_ = Base::ParamsPathForProfile(configProfile_);
	scenariosFile_ = Base::ScenariosPathForProfile(configProfile_);

	if (!LoadParamsFromCliOverrides(simOverrides, logParamStatus))
	{
		return false;
	}
	SyncParameters(true);

	// create the backend before building or restoring its domain bounds
	SPDLOG_INFO_MPI("Create distributed simulation backend...", mpiContext_);
	CreateSimBackend(simParams_.backendType);
	bool backendSupported = simBackend_ && simBackend_->SupportsDistributedSimulation();
	if (simBackend_ && !backendSupported)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Sim backend {} does not support distributed simulation. "
			"CPU is currently the only distributed backend.",
			mpiContext_.Rank(), mpiContext_.Size(), simBackend_->Name());
	}
	if (!mpiContext_.AllTrue(backendSupported)) { return false; }

	SPDLOG_INFO_MPI("Init distributed domain decomp...", mpiContext_);
	// every rank parses the same broadcast parameters, so all ranks pick the same strategy
	SelectDomainDecomp(simParams_.domainDecompMethod);
	if (!mpiContext_.AllTrue(domainDecomp_ != nullptr)) { return false; }
	if (!ConfigureLoadBalancing()) { return false; }

	SPDLOG_INFO_MPI("Init simulation state...", mpiContext_);
	if (restartCheckpoint.empty())
	{
		SelectSimScenario(simParams_.scenarioType, configProfile_);
		if (!mpiContext_.AllTrue(simScenario_ != nullptr)) { return false; }

		if (!BuildDomainDecomp()) { return false; }
		SPDLOG_INFO_MPI("Init initial domain decomp profile...", mpiContext_);
		if (!BuildInitialDomainDecomp()) { return false; }

		SPDLOG_INFO_MPI("Init rank-local particle states...", mpiContext_);
		if (!ResetAndRebuildSimState()) { return false; }
	}
	else // restart from a simulation state checkpoint
	{
		if (!LoadCheckpoint(restartCheckpoint)) { return false; }
	}
	SPDLOG_INFO_MPI("Init backend data structures...", mpiContext_);
	if (!InitializeSimBackendFromState()) { return false; }
	if (!restartCheckpoint.empty())
	{
		// redistribute restored particles and rebuild halos in backend storage
		if (!RedistributeRankLocalParticles(simParams_)) { return false; }
		if (!ValidateGlobalOwnedParticleCounts()) { return false; }
	}
	if (!RunRankLocalBackendStartupCheck()) { return false; }

	// build the fluid solver synchronization plan required by the initial export
	if (!mpiContext_.AllTrue(RebuildFluidSolverSyncPlan())) { return false; }

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

void DistSimulationMgr::CreateSimBackend(const Base::BackendType backendType)
{
	if (simBackend_)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Distributed simulation backend switching is currently disabled.",
			mpiContext_.Rank(), mpiContext_.Size());
		return;
	}

	distParticleBackend_ = nullptr;
	statisticsTracker_.reset();
	memoryTracker_.reset();
	benchmarksTracker_.reset();
	curBenchOps_.clear();

	// create metric trackers and backend, particle state is loaded separately
	benchmarksTracker_ = std::make_unique<Bench::BenchmarksTracker>(simParams_);
	memoryTracker_ = std::make_unique<Mem::MemoryTracker>(simParams_);
	statisticsTracker_ = std::make_unique<Stats::StatisticsTracker>(simParams_);
	try
	{
		simBackend_ = Sim::MakeSimBackend(
			backendType, simParamsMgr_, *benchmarksTracker_, simParams_);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to create distributed sim backend ({}): {} "
			"CPU is currently the only distributed backend.",
			mpiContext_.Rank(), mpiContext_.Size(),
			static_cast<int>(backendType), ex.what());
		return;
	}
	if (!simBackend_)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to create distributed sim backend ({}). "
			"CPU is currently the only distributed backend.",
			mpiContext_.Rank(), mpiContext_.Size(), static_cast<int>(backendType));
		return;
	}

	distParticleBackend_ = simBackend_->GetDistributedParticleBackend();
	if (!distParticleBackend_)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Sim backend ({}) does not provide distributed "
			"particle storage access. CPU is currently the only distributed backend.",
			mpiContext_.Rank(), mpiContext_.Size(), static_cast<int>(backendType));
		simBackend_.reset();
		return;
	}

	simBackend_->GetBenchOps(curBenchOps_);
	const Bench::BenchOpList distBenchOps =
	{
		Bench::BenchOp::Dist_ParticleRedistribution,
		Bench::BenchOp::Dist_PlanRebuild,
		Bench::BenchOp::Dist_HaloFieldExchange,
		Bench::BenchOp::Dist_ParticlePermutation,
		Bench::BenchOp::Dist_LoadBalancing,
		Bench::BenchOp::Dist_Misc
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
			mpiContext_.Rank(), mpiContext_.Size(), static_cast<int>(backendType));
		curBenchOps_.clear();
	}
}

void DistSimulationMgr::SelectDomainDecomp(const Base::DomainDecompMethod method)
{
	domainDecomp_.reset();

	// create new domain decomp, which still has to be built for the selected scenario
	try
	{
		domainDecomp_ = MakeDomainDecomp(method);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to create distributed domain decomp ({}): {} "
			"AABB_Slabs and AABB_Columns are the domain decomps.",
			mpiContext_.Rank(), mpiContext_.Size(),
			static_cast<int>(method), ex.what());
		return;
	}
	if (!domainDecomp_)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to create distributed domain decomp ({}). "
			"AABB_Slabs and AABB_Columns are the domain decomps.",
			mpiContext_.Rank(), mpiContext_.Size(), static_cast<int>(method));
		return;
	}
}

bool DistSimulationMgr::ResetAndRebuildSimState()
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
			serializationMgr_.ResetExportDirState();

			SyncParameters();

			if (!BuildGlobalSimState()) { localOk = false; }
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to re-build initial sim state: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	if (!mpiContext_.AllTrue(localOk)) { return false; }

	// cached before the broadcast to confirm below that every rank generated the same particles
	const uint64_t N_builtGlobalF = N_expectedGlobalF_;
	const uint64_t N_builtGlobalB = N_expectedGlobalB_;
	mpiContext_.BroadcastUint64(N_expectedGlobalF_, mpiContext_.RootRank());
	mpiContext_.BroadcastUint64(N_expectedGlobalB_, mpiContext_.RootRank());

	const bool globalParticleCountsMatch =
		N_builtGlobalF == N_expectedGlobalF_ &&
		N_builtGlobalB == N_expectedGlobalB_;

	if (!mpiContext_.AllTrue(globalParticleCountsMatch))
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Global particle counts differ across ranks.",
			mpiContext_.Rank(), mpiContext_.Size());
		return false;
	}

	try
	{
		if (!BuildRankLocalSimState()) { localOk = false; }
		if (localOk && !ValidateRankLocalSimState(true)) { localOk = false; }
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to build rank-local initial sim state: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	// use initial partition to validate if per-step ownership checksums match across ranks
	expectedOwnedIdChecksumF_ =
		mpiContext_.AllReduceSumUint64(
			OwnedParticleIdChecksum(particleStateF_, localSimState_.particlesF.id));
	expectedOwnedIdChecksumB_ =
		mpiContext_.AllReduceSumUint64(
			OwnedParticleIdChecksum(particleStateB_, localSimState_.particlesB.id));

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

	return ValidateGlobalOwnedParticleCounts();
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

	report.domainDecompMethod = Base::ToCString(simParams_.domainDecompMethod);
	report.distHaloMode = Base::ToCString(simParams_.distHaloMode);
	report.loadBalanceMetric = Base::ToCString(simParams_.loadBalanceMetric);
	report.loadBalanceTrigger = Base::ToCString(simParams_.loadBalanceTrigger);
	report.loadBalanceWeightModel = Base::ToCString(simParams_.loadBalanceWeightModel);
	report.backendType = Base::ToCString(simParams_.backendType);
	report.scenarioType = Base::ToCString(simParams_.scenarioType);
	report.neighborSearchMethod = Base::ToCString(simParams_.neighborSearchMethod);
	report.neighborListStorage = Base::ToEffectiveNeighborListStorageCString(
		simParams_.backendType,
		simParams_.neighborListMethod);

	report.N_globalF = N_expectedGlobalF_;
	report.N_globalB = N_expectedGlobalB_;
	report.N_globalR = globalSimState_.particlesR.N_;
	report.N_globalRB = globalSimState_.rigidBodies.N_;

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

	report.N_ownedF = particleStateF_.N_owned;
	report.N_ownedB = particleStateB_.N_owned;
	report.N_replicatedR = localSimState_.particlesR.N_;

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
	const uint64_t startStep = simStep_;
	BeginRunLogging(paramsInit);
	DistDiagnosticsMgr::ResetProgress(startStep);
	bool runOk = true;
	const char* stopReason = "unknown reason";
	bool checkpointDue = false;

	// restarted runs export after their next scheduled solver step,
	// startup routing rebuilds halos but does not reconstruct their pressure/density fields
	if (paramsInit.exportEnabled && simStep_ == 0)
	{
		MarkTrace(Trace::TraceStage::Dist_ExportSnapshot);
		if (!WriteExportSnapshot(paramsInit)) { runOk = false; }
	}

	AppendInitialMemoryMetricsLog(paramsInit);

	while (true)
	{
		if (!runOk)
		{
			stopReason = "failed export, statistics, or load balancing stage";
			break;
		}

		MarkTrace(Trace::TraceStage::Dist_LoopParameterSync);
		SyncParameters(true);

		// local copy for step-local consistency
		auto params = simParams_;

		// the run length is fixed at launch, so progress reports match the step limit below
		params.maxSteps = paramsInit.maxSteps;
		const bool doSimStep = (params.simRunning || params.simRunNextStep);

		// check checkpoint deadline on root
		checkpointDue = mpiContext_.IsRoot() && paramsInit.checkpointTime > 0.0 &&
			std::chrono::duration<double>(std::chrono::steady_clock::now() - initTime_).count() >=
			paramsInit.checkpointTime;

		// synchronize shutdown requests across ranks
		const bool keepRunning = mpiContext_.AllTrue(
			!stopRequested_.load(std::memory_order_acquire) && !checkpointDue);
		if (!keepRunning)
		{
			// broadcast whether shutdown requires a checkpoint
			int checkpointRequested = checkpointDue ? 1 : 0;
			mpiContext_.BroadcastInt(checkpointRequested);
			checkpointDue = checkpointRequested != 0;

			stopReason = "stop requested";
			if (checkpointDue)
			{
				stopReason = "checkpoint time reached";
				SPDLOG_INFO_MPI("Checkpoint time reached, saving checkpoint and stopping...", mpiContext_);
			}
			break;
		}

		// explicit max number of steps reached
		if (simStep_ >= static_cast<uint64_t>(paramsInit.maxSteps) &&
			paramsInit.maxSteps > 0)
		{
			stopReason = "max number of steps reached";
			break;
		}

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
		particleRedistributionMgr_.ResetRedistStats();
		traceRecorder_.BeginStep(
			simStep_ + 1,
			simTime_ + static_cast<double>(params.timeStep));

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
			stopReason = "failed simulation step";
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
			MarkTrace(Trace::TraceStage::Dist_ExportSnapshot);
			if (!WriteExportSnapshot(params)) { runOk = false; }
		}

		if (runOk)
		{
			MarkTrace(Trace::TraceStage::Dist_Statistics);
			if (!FillDistributedStatisticsRecord(params)) { runOk = false; }
		}

		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_LoadBalancing);
		if (runOk)
		{
			MarkTrace(Trace::TraceStage::Dist_LoadBalancing);
			if (!UpdateLoadBalancing()) { runOk = false; }
		}
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_LoadBalancing);

		// evaluate the benchmarks and memory usage for this step
		if (runOk) { MarkTrace(Trace::TraceStage::Dist_Metrics); }
		FinalizeBenchMemMeasurements(params);
		AppendLatestMemoryMetricsLog();
		AppendLatestBenchmarkMetricsLog();

		if (runOk) { MarkTrace(Trace::TraceStage::Dist_PostStepParameterSync); }
		SyncParameters();

		if (runOk) { MarkTrace(Trace::TraceStage::Dist_ProgressReport); }
		DisplayMemoryUsageStats(params, simStep_ == startStep + 1);

		// the progress block redraw would overwrite the error a failed stage just reported
		if (runOk) { DisplayDistributedProgress(params, exportedStep); }
		if (runOk) { MarkTrace(Trace::TraceStage::StepComplete); }
	}

	auto paramsFinal = simParams_;
	paramsFinal.maxSteps = paramsInit.maxSteps;
	const bool wantsFinalState = runOk && (paramsFinal.saveFinalState || checkpointDue);
	if (wantsFinalState)
	{
		MarkTrace(Trace::TraceStage::Dist_FinalStateSerialization);
	}
	const bool shouldSaveFinalState = mpiContext_.AllTrue(
		wantsFinalState);
	const bool finalStateOk = WriteFinalState(
		paramsFinal,
		shouldSaveFinalState);
	if (runOk && !finalStateOk)
	{
		runOk = false;
		stopReason = "failed final state serialization";
	}

	if (mpiContext_.IsRoot())
	{
		std::printf("\n");
	}
	DisplayMemoryUsageStats(paramsFinal, true);
	if (memoryTracker_)
	{
		const bool processMemoryChanged = memoryTracker_->SampleProcessMemory();
		if (processMemoryChanged) { AppendLatestMemoryMetricsLog(); }
	}

	running_.store(false, std::memory_order_release);
	EndRunLogging();

	SPDLOG_INFO_MPI("Distributed simulation stopped running ({})", mpiContext_, stopReason);
	DistDiagnosticsMgr::PrintFinalProgressPeakRam(
		mpiContext_, BuildRankStepReport(paramsFinal, false));

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
			if (profile != "default")
			{
				if (logStatus)
				{
					SPDLOG_WARN_MPI(
						"Failed to load '/{}/params.json'. Falling back to '/{}'.",
						mpiContext_, profile, Base::ParamsPathForProfile("default"));
				}
				profile = "default";
			}

			if (!TryReadProfileJson(profile, jsonText))
			{
				SPDLOG_ERROR_MPI(
					"Failed to load '/{}'.",
					mpiContext_, Base::ParamsPathForProfile(profile));
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
		else if (!domainDecomp_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed to build distributed domain decomp (no domain decomp selected).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		else
		{
			const AABB globalBounds =
				ComputeGlobalDomainBounds(simScenario_->GetBoundaryBoxes(), simParams_);

			// selected distributed halo mode defines the complete stored remote extent
			const DomainDecompInput input
			{
				.globalBounds = globalBounds,
				.haloRadius = simParams_.distHaloRadius,
				.N_ranks = mpiContext_.Size()
			};

			domainDecomp_->Build(input);

			// used to keep the backend spatial data structures bounded by the domain
			if (simBackend_)
			{
				simBackend_->SetSimDomainBounds(globalBounds.min, globalBounds.max);
			}

			if (!domainDecomp_->IsValidForRankCount(mpiContext_.Size()))
			{
				SPDLOG_ERROR("[rank {}/{}] Invalid distributed domain decomp.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
			}
			else
			{
				// layout is uniform at this stage, initial profile is applied afterwards
				constexpr const char* axisNames[] = { "X", "Y", "Z" };
				const glm::vec3 ownedExtents = domainDecomp_->OwnedExtentAABB(mpiContext_.Rank()).Extents();
				const glm::vec3 globalExtents = globalBounds.Extents();

				for (int axis = 0; axis < 3; axis++)
				{
					// only split axes shrink below the global extent
					if (ownedExtents[axis] >= globalExtents[axis] ||
						ownedExtents[axis] >= simParams_.distHaloRadius)
					{
						continue;
					}

					SPDLOG_WARN_MPI(
						"Distributed subdomains are thinner than the halo radius on the {} axis "
						"({:.4f} < {:.4f}), so their halos reach past the direct neighbors.",
						mpiContext_, axisNames[axis], ownedExtents[axis],
						simParams_.distHaloRadius);
				}
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

bool DistSimulationMgr::BuildInitialDomainDecomp()
{
	bool localOk = true;

	if (!simScenario_)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to build initial distributed domain decomp "
			"(no scenario selected).",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	else if (!domainDecomp_)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to build initial distributed domain decomp "
			"(no domain decomp selected).",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	Sync::UniversalSimState probeState{};
	std::vector<uint32_t> probeFluidLocalIdxs{};
	std::vector<uint32_t> probeBoundaryLocalIdxs{};

	try
	{
		if (mpiContext_.IsRoot())
		{
			const Sim::SimParams probeParams =
				MakeInitialDomainDecompProbeParams(simParams_);
			simScenario_->Build(probeState, probeParams);
			scenarioMgr_.FinalizeGeneratedParticleIds(probeState);

			probeFluidLocalIdxs.resize(probeState.particlesF.xPos.size());
			std::iota(probeFluidLocalIdxs.begin(), probeFluidLocalIdxs.end(), 0u);
			probeBoundaryLocalIdxs.resize(probeState.particlesB.xPos.size());
			std::iota(probeBoundaryLocalIdxs.begin(), probeBoundaryLocalIdxs.end(), 0u);
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to build initial domain decomp profile: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	DomainDecompRepartitionProfile profile{};
	try
	{
		DomainDecompRepartitionParticleInput profileInput{};
		profileInput.weightModel = simParams_.loadBalanceWeightModel;
		profileInput.rank = mpiContext_.Rank();

		if (mpiContext_.IsRoot())
		{
			// every probe particle contributes to the profile
			profileInput.fluid.x = probeState.particlesF.xPos;
			profileInput.fluid.y = probeState.particlesF.yPos;
			profileInput.fluid.z = probeState.particlesF.zPos;
			profileInput.fluid.ownedLocalIdx = probeFluidLocalIdxs;
			profileInput.boundary.x = probeState.particlesB.xPos;
			profileInput.boundary.y = probeState.particlesB.yPos;
			profileInput.boundary.z = probeState.particlesB.zPos;
			profileInput.boundary.ownedLocalIdx = probeBoundaryLocalIdxs;
		}

		domainDecomp_->BuildRepartitionProfile(profileInput, profile);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to build initial domain decomp profile: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	try
	{
		mpiContext_.AllReduceSumUint64s(profile.ownedFluidByUnit);
		mpiContext_.AllReduceSumUint64s(profile.ownedBoundaryByUnit);

		// repartition validates rank loads, despite the initial profile supplying
		// the spatial weights used to place the domain walls
		const std::vector<double> initialLoadByRank(
			static_cast<uint64_t>(mpiContext_.Size()), 1.0);
		domainDecomp_->Repartition(
			{
				.weightModel = simParams_.loadBalanceWeightModel,
				.loadByRank = initialLoadByRank,
				.ownedFluidByUnit = profile.ownedFluidByUnit,
				.ownedBoundaryByUnit = profile.ownedBoundaryByUnit
			});

		if (!domainDecomp_->IsValidForRankCount(mpiContext_.Size()))
		{
			throw std::runtime_error(
				"Initial domain decomposition produced an invalid rank layout.");
		}
	}
	catch (const std::exception& ex)
	{
		mpiContext_.AbortJob(
			"Failed to apply initial domain decomp profile: " +
			std::string(ex.what()));
	}
	catch (...)
	{
		mpiContext_.AbortJob(
			"Failed to apply initial domain decomp profile: unknown error.");
	}

	return mpiContext_.AllTrue(localOk);
}

bool DistSimulationMgr::ConfigureLoadBalancing()
{
	bool localOk = true;
	std::string failMsg{};

	try
	{
		if (!loadBalanceMgr_.Configure(simParams_, &failMsg))
		{
			localOk = false;
		}
	}
	catch (const std::exception& ex)
	{
		failMsg = ex.what();
		localOk = false;
	}

	if (!localOk)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to configure distributed load balancing: {}",
			mpiContext_.Rank(), mpiContext_.Size(), failMsg);
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
	if (!domainDecomp_)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to build global sim state (no domain decomp).",
			mpiContext_.Rank(), mpiContext_.Size());
		return false;
	}
	// reset scenario and all full-state data
	globalSimState_ = {};
	globalSimState_.step = 0;
	globalSimState_.time = 0.0;

	// expand the stored remote extent by one physical kernel radius so boundary
	// coefficient computations (psiB) still see complete halo neighborhoods
	const AABB buildRegion = domainDecomp_->HaloExtentAABB(mpiContext_.Rank())
		.Expanded(simParams_.kernelRadius);

	scenarioMgr_.BeginBuildRegion(buildRegion.min, buildRegion.max);
	simScenario_->Build(globalSimState_, simParams_);
	scenarioMgr_.FinalizeGeneratedParticleIds(globalSimState_);
	scenarioMgr_.EndBuildRegion();

	// the generated particle set does not depend on the build region, so these counts are
	// the full scenario counts and every rank arrives at the same values
	N_expectedGlobalF_ = scenarioMgr_.GetBuiltGlobalFluidParticleCount();
	N_expectedGlobalB_ = scenarioMgr_.GetBuiltGlobalBoundaryParticleCount();

	globalSimState_.particlesF.ensureParticleIdsPresent();
	globalSimState_.particlesB.ensureParticleIdsPresent();
	globalSimState_.particlesR.ensureParticleIdsPresent();
	globalSimState_.particlesB.ensureBoundaryCoefficientsPresent(
		simParams_.massB, simParams_.restDensity);
	globalSimState_.particlesR.ensureRigidCoefficientsPresent(
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
	particleStateF_.Clear();
	particleStateB_.Clear();
	pressureAccFluidParticleLocalIdxs_.clear();
	activeRigidParticleLocalIdxs_.clear();
	rigidBodyForceTorque_ = {};

	// keep owned particles first, then append halo particles
	BuildLocalParticlesFromGlobalState(
		localSimState_,
		particleStateF_,
		particleStateB_,
		globalSimState_,
		*domainDecomp_,
		mpiContext_.Rank(),
		simParams_.distHaloRadius);

	localSimState_.ShrinkToFitAll();

	// the replicated scenario payload was only a build-time source, so
	// only the replicated rigid state is kept past the rank-local build
	globalSimState_.particlesF = {};
	globalSimState_.particlesB = {};

	return true;
}

bool DistSimulationMgr::ValidateRankLocalSimState(
	const bool includeRigidState) const
{
	bool localOk = true;

	std::string failMsg{};
	if (!particleStateF_.IsValidLocalParticleState(
		localSimState_.particlesF.id,
		N_expectedGlobalF_,
		&failMsg))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid rank-local fluid particle state: {}",
			mpiContext_.Rank(), mpiContext_.Size(), failMsg);
		localOk = false;
	}

	if (!particleStateB_.IsValidLocalParticleState(
		localSimState_.particlesB.id,
		N_expectedGlobalB_,
		&failMsg))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid rank-local boundary particle state: {}",
			mpiContext_.Rank(), mpiContext_.Size(), failMsg);
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

bool DistSimulationMgr::ValidateGlobalOwnedParticleCounts() const
{
	std::span<const uint64_t> particleIdsF = localSimState_.particlesF.id;
	std::span<const uint64_t> particleIdsB = localSimState_.particlesB.id;

	if (distParticleBackend_)
	{
		const Sync::FluidConstView backendParticlesF =
			distParticleBackend_->GetFluidParticleView();
		const Sync::BoundaryConstView backendParticlesB =
			distParticleBackend_->GetBoundaryParticleView();
		if (backendParticlesF.N_ == particleStateF_.N_owned + particleStateF_.N_halo)
		{
			particleIdsF = backendParticlesF.id;
		}
		if (backendParticlesB.N_ == particleStateB_.N_owned + particleStateB_.N_halo)
		{
			particleIdsB = backendParticlesB.id;
		}
	}

	// rank-local owned particle counts and particle ID checksums, summed across all ranks
	std::array<uint64_t, 4> N_globalOwned
	{
		particleStateF_.N_owned,
		particleStateB_.N_owned,
		OwnedParticleIdChecksum(particleStateF_, particleIdsF),
		OwnedParticleIdChecksum(particleStateB_, particleIdsB)
	};
	mpiContext_.AllReduceSumUint64s(N_globalOwned);

	const bool countsValid =
		N_globalOwned[0] == N_expectedGlobalF_ &&
		N_globalOwned[1] == N_expectedGlobalB_;
	const bool checksumsValid =
		N_globalOwned[2] == expectedOwnedIdChecksumF_ &&
		N_globalOwned[3] == expectedOwnedIdChecksumB_;

	if (!countsValid)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Invalid global owned particle counts "
			"(fluid={}/{}, boundary={}/{}).",
			mpiContext_.Rank(), mpiContext_.Size(),
			N_globalOwned[0],
			N_expectedGlobalF_,
			N_globalOwned[1],
			N_expectedGlobalB_);
	}
	else if (!checksumsValid)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Global owned particle IDs are not owned by exactly one rank "
			"(fluid checksum={}/{}, boundary checksum={}/{}).",
			mpiContext_.Rank(), mpiContext_.Size(),
			N_globalOwned[2],
			expectedOwnedIdChecksumF_,
			N_globalOwned[3],
			expectedOwnedIdChecksumB_);
	}

	#if defined(_DEBUG)
	// expensive owner diagnosis runs only after every rank has observed the same
	// reduced count/checksum failure, scanning a bounded global-ID window at a time
	if (!countsValid || !checksumsValid)
	{
		const auto ReportOwnedIdConflict = [&](
			const DistParticleState& particleState,
			const std::span<const uint64_t> particleIds,
			const uint64_t N_global,
			const char* const particleTypeName) -> bool
		{
			std::optional<OwnedParticleConflict> conflict{};
			std::string diagnosticFailure{};
			if (!FindFirstOwnedParticleConflict(
				particleIds,
				particleState.particleLocalIdx_Owned,
				N_global,
				mpiContext_,
				conflict,
				&diagnosticFailure))
			{
				SPDLOG_ERROR(
					"[rank {}/{}] Failed to diagnose global {} ownership: {}",
					mpiContext_.Rank(), mpiContext_.Size(),
					particleTypeName, diagnosticFailure);
				return false;
			}
			if (conflict)
			{
				SPDLOG_ERROR(
					"[rank {}/{}] Global {} particle {} has {} owner claims "
					"(expected exactly one).",
					mpiContext_.Rank(), mpiContext_.Size(),
					particleTypeName, conflict->id, conflict->ownerCount);
				return false;
			}
			return true;
		};

		if (!ReportOwnedIdConflict(
			particleStateF_, particleIdsF, N_expectedGlobalF_, "fluid"))
		{
			return false;
		}
		if (!ReportOwnedIdConflict(
			particleStateB_, particleIdsB, N_expectedGlobalB_, "boundary"))
		{
			return false;
		}
	}
	#endif

	return countsValid && checksumsValid;
}

bool DistSimulationMgr::InitializeSimBackendFromState()
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

			if (simBackend_->GetNumFluidParticles() != particleStateF_.N_owned +
				particleStateF_.N_halo)
			{
				SPDLOG_ERROR("[rank {}/{}] Rank-local backend fluid count mismatch.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
			}

			if (simBackend_->GetNumBoundaryParticles() != particleStateB_.N_owned +
				particleStateB_.N_halo)
			{
				SPDLOG_ERROR("[rank {}/{}] Rank-local backend boundary count mismatch.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
			}

			if (simBackend_->GetNumRigidParticles() != globalSimState_.particlesR.N_)
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
			// configure compact [ owned ][ halo ] layout before the startup neighbor search
			const bool configured = simBackend_->ConfigureDistributedParticleLayout(
				particleStateF_.N_owned,
				particleStateB_.N_owned);
			if (!configured)
			{
				SPDLOG_ERROR(
					"[rank {}/{}] Failed to configure rank-local backend startup check.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
			}

			// first backend smoke test on compact [ owned ][ halo ] layout
			if (localOk)
			{
				simBackend_->ExecuteNeighborSearch(simParams_);
				if (!ApplyLastBackendParticleReOrder()) { localOk = false; }
			}

			Sync::UniversalSimState backendSimState{};
			simBackend_->PushToUniversalSimState(backendSimState, false);
			localSimState_ = std::move(backendSimState);
			localSimState_.step = simStep_;
			localSimState_.time = simTime_;

			if (localOk && !ValidateRankLocalBackendParticleState()) { localOk = false; }

			// fluid and boundary payloads remain backend-resident after startup
			localSimState_.particlesF = {};
			localSimState_.particlesB = {};
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

bool DistSimulationMgr::RebuildFluidSolverSyncPlan()
{
	bool localOk = true;
	std::string failMsg{};

	try
	{
		if (!fluidSolverSyncMgr_.Rebuild(
			distParticleBackend_->GetFluidParticleView(),
			particleStateF_,
			*domainDecomp_,
			simParams_.kernelRadius,
			simParams_.distHaloMode,
			N_expectedGlobalF_,
			mpiContext_,
			*benchmarksTracker_,
			&failMsg))
		{
			localOk = false;
		}
	}
	catch (const std::exception& ex)
	{
		mpiContext_.AbortJob(
			"Failed to rebuild distributed fluid solver synchronization: " +
			std::string(ex.what()));
	}
	catch (...)
	{
		mpiContext_.AbortJob(
			"Failed to rebuild distributed fluid solver synchronization: unknown error.");
	}

	if (!localOk)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to rebuild distributed fluid solver synchronization: {}",
			mpiContext_.Rank(), mpiContext_.Size(), failMsg);
	}
	return localOk;
}

bool DistSimulationMgr::WriteExportSnapshot(const Sim::SimParams& params)
{
	benchmarksTracker_->StartBench(Bench::BenchOp::Serialization);

	const bool backendAvailable = distParticleBackend_ != nullptr;
	if (!backendAvailable)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to write distributed export snapshot "
			"(no distributed backend selected).",
			mpiContext_.Rank(), mpiContext_.Size());
	}
	if (!mpiContext_.AllTrue(backendAvailable))
	{
		benchmarksTracker_->StopBench(Bench::BenchOp::Serialization);
		return false;
	}

	const Sync::FluidConstView particlesF =
		distParticleBackend_->GetFluidParticleView();
	const Sync::BoundaryConstView particlesB =
		distParticleBackend_->GetBoundaryParticleView();
	const Sync::RigidConstView particlesR =
		distParticleBackend_->GetRigidParticleView();

	DistSnapshotGatherStats stats{};

	auto& exportFluidParticleLocalIdxs = exportFluidParticleLocalIdxsScratch_;
	bool exportSelectionOk = true;
	try
	{
		exportSelectionOk =
			BuildExportFluidParticleLocalIdxs(particlesF, exportFluidParticleLocalIdxs);
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to select particles for distributed export: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		exportSelectionOk = false;
	}
	if (!mpiContext_.AllTrue(exportSelectionOk))
	{
		benchmarksTracker_->StopBench(Bench::BenchOp::Serialization);
		return false;
	}

	// fill export snapshot
	auto& expSnap = exportSnapScratch_;
	MarkTrace(Trace::TraceStage::Dist_ExportSnapshotGather);
	const bool gatherOk = GatherExportSnapshotToRoot(
		expSnap,
		exportSnapGatherScratch_,
		particlesF,
		particlesB,
		particlesR,
		localSimState_.rigidBodies,
		particleStateF_,
		particleStateB_,
		exportFluidParticleLocalIdxs,
		mpiContext_,
		simStep_,
		simTime_,
		&stats);

	if (!gatherOk)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to gather the distributed export snapshot on the root rank.",
			mpiContext_.Rank(), mpiContext_.Size());
		benchmarksTracker_->StopBench(Bench::BenchOp::Serialization);
		return false;
	}

	bool writeOk = true;
	if (mpiContext_.IsRoot())
	{
		MarkTrace(Trace::TraceStage::Dist_ExportSnapshotSerialization);
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

bool DistSimulationMgr::WriteFinalState(
	const Sim::SimParams& params,
	const bool enabled)
{
	bool writeOk = !enabled || MaterializeRankLocalState(true);

	// final-state output is the last backend consumer, even on failure
	simBackend_.reset();
	distParticleBackend_ = nullptr;
	if (enabled && writeOk) { writeOk = SaveCheckpoint(params); }
	localSimState_ = {};
	return writeOk;
}

bool DistSimulationMgr::BuildExportFluidParticleLocalIdxs(
	const Sync::FluidConstView& particlesF,
	std::vector<uint32_t>& localIdxs)
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
		(!simBackend_->FillFluidParticleNearSurfaceFlags(
			simParams_, localNearSurfaceFlagsScratch_) ||
		localNearSurfaceFlagsScratch_.size() !=
			static_cast<uint64_t>(particlesF.N_)))
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to classify rank-local fluid particles "
			"for distributed surface export.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	if (localOk)
	{
		for (const uint32_t localIdx : particleStateF_.particleLocalIdx_Owned)
		{
			if (localIdx >= particlesF.N_ ||
				localIdx >= particlesF.id.size() ||
				!particleStateF_.HasMetadataAtLocalIdx(localIdx))
			{
				SPDLOG_ERROR(
					"[rank {}/{}] Invalid owned fluid particle metadata "
					"during distributed surface export.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
				break;
			}

			const uint64_t particleId = particlesF.id[localIdx];
			if (particleId >= N_expectedGlobalF_)
			{
				SPDLOG_ERROR(
					"[rank {}/{}] Invalid owned fluid particle ID "
					"during distributed surface export (localIdx={}, particleId={}).",
					mpiContext_.Rank(), mpiContext_.Size(), localIdx, particleId);
				localOk = false;
				break;
			}
		}
	}

	if (localOk)
	{
		for (uint32_t localIdx = 0; localIdx < particlesF.N_; localIdx++)
		{
			if (!particleStateF_.HasMetadataAtLocalIdx(localIdx) ||
				localIdx >= particlesF.id.size())
			{
				localOk = false;
				break;
			}

			const uint64_t particleId = particlesF.id[localIdx];
			if (particleId >= N_expectedGlobalF_)
			{
				localOk = false;
				break;
			}
		}
	}

	// validate local flags on each rank before entering owner-to-halo exchange
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	fluidSolverSyncMgr_.SynchronizeFluidParticleFlagsToHalos(
		localNearSurfaceFlagsScratch_, simParams_.distHaloMode);

	if (!simBackend_->BuildExportFluidParticleLocalIdxs(
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
	// the root rank reserves the run directory shared by all ranks
	uint64_t runIdx = mpiContext_.IsRoot() ? loggingMgr_.ReserveNextRunIdx() : 0;
	mpiContext_.BroadcastUint64(runIdx, mpiContext_.RootRank());
	if (runIdx == 0)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to reserve a run log directory.",
			mpiContext_.Rank(), mpiContext_.Size());
		EndRunLogging();
		return;
	}

	loggingMgr_.BeginRun(configProfile_, paramsFile_, scenariosFile_, params,
		simParamsMgr_.CliOverrides(), mpiContext_.IsRoot(), runIdx);
	traceRecorder_.BeginRun(loggingMgr_.GetRunDir(), mpiContext_.Rank());
}

void DistSimulationMgr::MarkTrace(const Trace::TraceStage stage)
{
	if (!traceRecorder_.IsActive()) { return; }

	if (benchmarksTracker_)
	{
		benchmarksTracker_->PauseActiveTimers();
	}

	try
	{
		traceRecorder_.Mark(stage);
	}
	catch (...)
	{
		if (benchmarksTracker_)
		{
			benchmarksTracker_->ResumeActiveTimers();
		}
		throw;
	}

	if (benchmarksTracker_)
	{
		benchmarksTracker_->ResumeActiveTimers();
	}
}

void DistSimulationMgr::EndRunLogging()
{
	traceRecorder_.EndRun();
	loggingMgr_.EndRun();
}

void DistSimulationMgr::AppendInitialMemoryMetricsLog(const Sim::SimParams& params) const
{
	if (!loggingMgr_.IsActive()) { return; }
	if (!simBackend_) { return; }
	if (!memoryTracker_) { return; }

	Mem::MemRecordStep record{};
	record.simStep = simStep_;
	record.simTime = simTime_;

	simBackend_->FillMemoryRecord(record, params);
	FillDistributedMemoryRecord(record);
	memoryTracker_->SampleProcessMemory();
	record.process = memoryTracker_->GetCurrentRecord().process;
	record.device = memoryTracker_->GetCurrentRecord().device;
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
		GetParticleMemoryUsage(globalSimState_));
	Mem::Set(record, Mem::MemSpace::RAM, Mem::MemBucket::Dist_LocalState,
		GetParticleMemoryUsage(localSimState_));

	Mem::MemUsage particleMetadataUsage{};
	Mem::Add(particleMetadataUsage, GetParticleMemoryUsage(particleStateF_));
	Mem::Add(particleMetadataUsage, GetParticleMemoryUsage(particleStateB_));
	Mem::Set(record, Mem::MemSpace::RAM, Mem::MemBucket::Dist_ParticleMetadata, particleMetadataUsage);

	Mem::MemUsage exchangeBufferUsage{};
	Mem::Add(exchangeBufferUsage, particleRedistributionMgr_.GetExchangeBufferMemoryUsage());
	Mem::Add(exchangeBufferUsage, fluidSolverSyncMgr_.GetExchangeBufferMemoryUsage());
	Mem::Set(record, Mem::MemSpace::RAM, Mem::MemBucket::Dist_ExchangeBuffers, exchangeBufferUsage);

	Mem::MemUsage miscUsage{};
	Mem::Add(miscUsage, particleRedistributionMgr_.GetScratchMemoryUsage());
	Mem::Add(miscUsage, fluidSolverSyncMgr_.GetPlanMemoryUsage());
	Mem::Add(miscUsage, loadBalanceMgr_.GetScratchMemoryUsage());
	Mem::Add(miscUsage, GetParticleMemoryUsage(fluidParticleReOrderWorkBuffers_));
	Mem::Add(miscUsage, GetParticleMemoryUsage(boundaryParticleReOrderWorkBuffers_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(pressureAccFluidParticleLocalIdxs_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(activeRigidParticleLocalIdxs_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(activeRigidParticleRankCountScratch_));
	Mem::Add(miscUsage, MemoryUsageOfRigidBodyForceTorque(rigidBodyForceTorque_));
	Mem::Add(miscUsage, MemoryUsageOfRigidBodySyncBuffers(rigidBodySyncBuffers_));
	Mem::Add(miscUsage, MemoryUsageOfStatisticsPartial(statisticsPartialScratch_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(localNearSurfaceFlagsScratch_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(exportFluidParticleLocalIdxsScratch_));
	Mem::Add(miscUsage, MemoryUsageOfExportSnapshot(exportSnapScratch_));
	Mem::Add(miscUsage, MemoryUsageOfExportSnapshotGatherBuffer(exportSnapGatherScratch_));
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
			"[rank {}/{}] Failed to fill rank-local distributed statistics partial.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	partial.N_B = particleStateB_.N_owned;
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

	report.N_ownedF = particleStateF_.N_owned;
	report.N_innerF = static_cast<uint32_t>(particleStateF_.particleLocalIdx_Inner.size());
	report.N_outerF = static_cast<uint32_t>(particleStateF_.particleLocalIdx_Outer.size());
	report.N_haloF = particleStateF_.N_halo;
	report.N_ownedB = particleStateB_.N_owned;
	report.N_innerB = static_cast<uint32_t>(particleStateB_.particleLocalIdx_Inner.size());
	report.N_outerB = static_cast<uint32_t>(particleStateB_.particleLocalIdx_Outer.size());
	report.N_haloB = particleStateB_.N_halo;

	const ParticleRedistributionStats& redistributionStats =
		particleRedistributionMgr_.GetStats();
	report.N_sentMigratedF = redistributionStats.N_sentOwnedF;
	report.N_recvMigratedF = redistributionStats.N_recvOwnedF;
	report.N_sentHaloF = redistributionStats.N_sentHaloF;
	report.N_recvHaloF = redistributionStats.N_recvHaloF;
	report.N_sentMigratedB = redistributionStats.N_sentOwnedB;
	report.N_recvMigratedB = redistributionStats.N_recvOwnedB;
	report.N_sentHaloB = redistributionStats.N_sentHaloB;
	report.N_recvHaloB = redistributionStats.N_recvHaloB;

	if (benchmarksTracker_)
	{
		report.totalStepLatestNs =
			benchmarksTracker_->GetBench_Latest(Bench::BenchOp::Total);
		report.totalStepAvgNs =
			benchmarksTracker_->GetBench_Avg_OverAll(Bench::BenchOp::Total);
	}

	if (memoryTracker_)
	{
		const Mem::MemSpace memSpace = PrimaryMemSpace(params.backendType);
		const Mem::MemUsage latest = SumMemoryUsage(memoryTracker_->GetCurrentRecord(), memSpace);
		const Mem::MemUsage max = SumMaxMemoryUsage(*memoryTracker_, memSpace);
		const Mem::MemUsage backendMax = SumBackendMaxMemoryUsage(*memoryTracker_, memSpace);

		report.N_latestMemUsedBytes = latest.N_usedBytes;
		report.N_latestMemAllocBytes = latest.N_allocBytes;
		report.N_maxMemUsedBytes = max.N_usedBytes;
		report.N_maxMemAllocBytes = max.N_allocBytes;

		report.N_maxBackendMemAllocBytes = backendMax.N_allocBytes;
		report.N_maxDistGlobalStateMemAllocBytes =
			memoryTracker_->GetMem_Max_Alloc(
				Mem::MemSpace::RAM, Mem::MemBucket::Dist_GlobalState);
		report.N_maxDistLocalStateMemAllocBytes =
			memoryTracker_->GetMem_Max_Alloc(
				Mem::MemSpace::RAM, Mem::MemBucket::Dist_LocalState);
		report.N_maxDistParticleMetadataMemAllocBytes =
			memoryTracker_->GetMem_Max_Alloc(
				Mem::MemSpace::RAM, Mem::MemBucket::Dist_ParticleMetadata);
		report.N_maxDistExchangeBuffersMemAllocBytes =
			memoryTracker_->GetMem_Max_Alloc(
				Mem::MemSpace::RAM, Mem::MemBucket::Dist_ExchangeBuffers);
		report.N_maxDistMiscMemAllocBytes =
			memoryTracker_->GetMem_Max_Alloc(
				Mem::MemSpace::RAM, Mem::MemBucket::Dist_Misc);
		report.N_maxTotalTrackedMemAllocBytes =
			report.N_maxBackendMemAllocBytes +
			report.N_maxDistGlobalStateMemAllocBytes +
			report.N_maxDistLocalStateMemAllocBytes +
			report.N_maxDistParticleMetadataMemAllocBytes +
			report.N_maxDistExchangeBuffersMemAllocBytes +
			report.N_maxDistMiscMemAllocBytes;
	}

	report.exportedStep = exportedStep ? 1 : 0;
	report.N_processResidentBytes =
		memoryTracker_ ? memoryTracker_->GetCurrentRecord().process.N_residentBytes : 0;
	report.N_processPeakResidentBytes =
		memoryTracker_ ? memoryTracker_->GetCurrentRecord().process.N_peakResidentBytes : 0;
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
		BuildRankBenchReport());
}

bool DistSimulationMgr::UpdateLoadBalancing()
{
	if (!loadBalanceMgr_.Enabled()) { return true; }
	if (!domainDecomp_ || !benchmarksTracker_ || !distParticleBackend_)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to update distributed load balancing "
			"(missing domain decomposition, benchmark tracker, or backend particle storage).",
			mpiContext_.Rank(), mpiContext_.Size());
		return mpiContext_.AllTrue(false);
	}

	const RankLoadSample localSample
	{
		.step = simStep_,
		.N_ownedF = particleStateF_.N_owned,
		.N_ownedB = particleStateB_.N_owned,
		.N_haloF = particleStateF_.N_halo,
		.N_haloB = particleStateB_.N_halo,
		// (measurement is already one step old, since the record is finalized after load balancing)
		.rankBackendStepTimeNs =
			benchmarksTracker_->GetBench_Latest(Bench::BenchOp::Dist_RankBackendStep)
	};
	const Sync::FluidConstView particlesF =
		distParticleBackend_->GetFluidParticleView();
	const Sync::BoundaryConstView particlesB =
		distParticleBackend_->GetBoundaryParticleView();
	const DomainDecompRepartitionParticleInput localParticleInput
	{
		.weightModel = simParams_.loadBalanceWeightModel,
		.rank = mpiContext_.Rank(),
		.fluid =
		{
			.x = particlesF.xPos,
			.y = particlesF.yPos,
			.z = particlesF.zPos,
			.ownedLocalIdx = particleStateF_.particleLocalIdx_Owned
		},
		.boundary =
		{
			.x = particlesB.xPos,
			.y = particlesB.yPos,
			.z = particlesB.zPos,
			.ownedLocalIdx = particleStateB_.particleLocalIdx_Owned
		}
	};

	std::string failMsg{};
	bool repartitioned = false;
	const bool ok = loadBalanceMgr_.EvaluateLoadAndRebalance(
		localSample,
		localParticleInput,
		*domainDecomp_,
		mpiContext_,
		repartitioned,
		&failMsg);
	if (ok && repartitioned)
	{
		boundaryRedistributionRequired_ = true;
	}
	if (!ok)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to update distributed load balancing: {}",
			mpiContext_.Rank(), mpiContext_.Size(),
			failMsg.empty() ? "Another rank reported an error." : failMsg);
	}
	return ok;
}

bool DistSimulationMgr::ExecuteDistributedStep(const Sim::SimParams& params)
{
	// results of [ rank-reduced ] helpers are returned directly, because every rank already
	// agrees on them, while rank-local results are reduced before the next collective runs
	bool localOk = true;

	try
	{
		// check that all components are present before redistribution
		if (!simBackend_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed distributed step (no backend selected).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		if (!distParticleBackend_)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed distributed step "
				"(backend has no distributed particle storage access).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		if (!domainDecomp_)
		{
			SPDLOG_ERROR("[rank {}/{}] Failed distributed step (no domain decomp).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		MarkTrace(Trace::TraceStage::Dist_InputValidation);
		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_Misc);
		const bool inputOk = mpiContext_.AllTrue(localOk);
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_Misc);
		if (!inputOk) { return false; }

		// ensure all ranks have the same parameters before redistribution
		MarkTrace(Trace::TraceStage::Dist_ParameterSync);
		SyncParameters();

		MarkTrace(Trace::TraceStage::Dist_ParticleRedistribution);
		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_ParticleRedistribution);
		const bool redistributionOk = RedistributeRankLocalParticles(params);
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_ParticleRedistribution);
		if (!redistributionOk) { return false; }

		MarkTrace(Trace::TraceStage::Dist_GlobalCountValidation);
		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_Misc);
		const bool globalCountsOk = ValidateGlobalOwnedParticleCounts();
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_Misc);
		if (!globalCountsOk) { return false; }

		MarkTrace(Trace::TraceStage::Dist_RankParticlePreparation);
		if (!PrepareRankLocalBackendStep()) { return false; }

		MarkTrace(Trace::TraceStage::Dist_RankBackendStep);
		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_RankBackendStep);
		try
		{
			simBackend_->ExecuteFullSimStep(params);
		}
		catch (const std::exception& ex)
		{
			mpiContext_.AbortJob(
				"Failed rank-local backend solver step: " + std::string(ex.what()));
		}
		catch (...)
		{
			mpiContext_.AbortJob(
				"Failed rank-local backend solver step: unknown error.");
		}
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_RankBackendStep);
		MarkTrace(Trace::TraceStage::Dist_RankBackendStepComplete);

		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_Misc);
		const bool backendOk = mpiContext_.AllTrue(
			localOk, "distributed rank backend completion");
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_Misc);
		if (!backendOk) { return false; }

		MarkTrace(Trace::TraceStage::Dist_ParticlePermutation);
		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_ParticlePermutation);
		try
		{
			// skip solver step if no fluid particles to avoid neighbor search particle re-order
			if (simBackend_->GetNumFluidParticles() > 0)
			{
				if (!ApplyLastBackendParticleReOrder()) { localOk = false; }
			}
		}
		catch (const std::exception& ex)
		{
			mpiContext_.AbortJob(
				"Failed to apply backend particle re-order: " + std::string(ex.what()));
		}
		catch (...)
		{
			mpiContext_.AbortJob(
				"Failed to apply backend particle re-order: unknown error.");
		}
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_ParticlePermutation);

		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_Misc);
		const bool reOrderOk = mpiContext_.AllTrue(localOk);
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_Misc);
		if (!reOrderOk) { return false; }

		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_Misc);
		const bool finishOk = FinishRankLocalBackendStep(params);
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_Misc);
		if (!finishOk) { return false; }

		MarkTrace(Trace::TraceStage::Dist_ParticleMetadataValidation);
		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_Misc);
		if (!ValidateRankLocalParticleMetadata()) { localOk = false; }
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_Misc);
	}
	catch (const std::exception& ex)
	{
		mpiContext_.AbortJob(
			"Failed distributed step: " + std::string(ex.what()));
	}
	catch (...)
	{
		mpiContext_.AbortJob("Failed distributed step: unknown error.");
	}

	return mpiContext_.AllTrue(localOk);
}

bool DistSimulationMgr::RedistributeRankLocalParticles(
	const Sim::SimParams& params)
{
	bool localOk = true;

	// coordinate wrapper-level failures outside the timed redistribution phases
	const auto AllRanksTrue = [&](const bool value)
	{
		return mpiContext_.AllTrue(value);
	};

	try
	{
		if (!domainDecomp_ || !distParticleBackend_)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to redistribute particles "
				"(missing domain decomposition or backend storage access).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		if (!AllRanksTrue(localOk)) { return false; }

		std::string failMsg{};
		const bool redistributeParticlesB = boundaryRedistributionRequired_;
		if (localOk && !particleRedistributionMgr_.RedistributeParticles(
			*distParticleBackend_,
			particleStateF_,
			particleStateB_,
			*domainDecomp_,
			mpiContext_,
			params,
			redistributeParticlesB,
			N_expectedGlobalF_,
			N_expectedGlobalB_,
			&failMsg))
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to redistribute particles: {}",
				mpiContext_.Rank(), mpiContext_.Size(), failMsg);
			localOk = false;
		}
		else if (localOk && redistributeParticlesB)
		{
			boundaryRedistributionRequired_ = false;
		}
	}
	catch (const std::exception& ex)
	{
		mpiContext_.AbortJob(
			"Failed to redistribute particles: " + std::string(ex.what()));
	}
	catch (...)
	{
		mpiContext_.AbortJob("Failed to redistribute particles: unknown error.");
	}

	return AllRanksTrue(localOk);
}

bool DistSimulationMgr::MaterializeRankLocalState(const bool backendWillBeRetired)
{
	bool localOk = true;

	try
	{
		if (!simBackend_)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to materialize rank-local state "
				"(no backend selected).",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
		else
		{
			Sync::UniversalSimState backendSimState{};
			simBackend_->PushToUniversalSimState(backendSimState, backendWillBeRetired);
			localSimState_ = std::move(backendSimState);
			localSimState_.step = simStep_;
			localSimState_.time = simTime_;
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to materialize rank-local state: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	return mpiContext_.AllTrue(localOk);
}

bool DistSimulationMgr::PrepareRankLocalBackendStep()
{
	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_ParticlePermutation);

	// derived from the replicated global topology, so every rank agrees on whether the
	// rigid body collectives below run at all
	const bool hasRigidBodies =
		globalSimState_.particlesR.N_ > 0 ||
		globalSimState_.rigidBodies.N_ > 0;

	bool localOk = true;
	const bool computeFirstHaloLayerPressureAcc =
		simParams_.distHaloMode == Base::DistHaloMode::DoubleHalo;
	if (computeFirstHaloLayerPressureAcc)
	{
		// upper bound for the owned indices plus the appended first halo layer
		pressureAccFluidParticleLocalIdxs_.reserve(
			particleStateF_.N_owned + particleStateF_.N_halo);
	}
	pressureAccFluidParticleLocalIdxs_ =
		particleStateF_.particleLocalIdx_Owned;
	activeRigidParticleLocalIdxs_.clear();

	try
	{
		// computes pressure acceleration for the first halo layer locally
		if (computeFirstHaloLayerPressureAcc)
		{
			const Sync::FluidConstView particlesF =
				distParticleBackend_->GetFluidParticleView();
			const AABB singleHaloBounds =
				domainDecomp_->OwnedExtentAABB(mpiContext_.Rank())
					.Expanded(simParams_.kernelRadius);
			const auto& haloLocalIdxs = particleStateF_.particleLocalIdx_Halo;
			const uint32_t N_halo = static_cast<uint32_t>(haloLocalIdxs.size());

			if (N_halo > 0u)
			{
				const uint32_t N_owned = static_cast<uint32_t>(
					pressureAccFluidParticleLocalIdxs_.size());
				pressureAccFluidParticleLocalIdxs_.resize(N_owned + N_halo);

				const uint32_t N_firstHalo = tbb::parallel_scan(
					tbb::blocked_range<uint32_t>(0, N_halo),
					uint32_t{ 0 },
					[&](const tbb::blocked_range<uint32_t>& range,
						const uint32_t prefix,
						const bool isFinalScan)
				{
					uint32_t N_selected = prefix;
					for (uint32_t haloIdx = range.begin();
						haloIdx != range.end(); haloIdx++)
					{
						const uint32_t localIdx = haloLocalIdxs[haloIdx];
						const glm::vec3 position{
							particlesF.xPos[localIdx],
							particlesF.yPos[localIdx],
							particlesF.zPos[localIdx]
						};
						if (!singleHaloBounds.Contains(position)) { continue; }

						if (isFinalScan)
						{
							pressureAccFluidParticleLocalIdxs_[N_owned + N_selected] = localIdx;
						}
						N_selected++;
					}
					return N_selected;
				},
				[](const uint32_t lhs, const uint32_t rhs)
				{
					return lhs + rhs;
				},
				tbb::auto_partitioner{}
				);

				pressureAccFluidParticleLocalIdxs_.resize(N_owned + N_firstHalo);
			}
		}

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
			activeRankCounts.assign(localSimState_.particlesR.N_, 0);
			for (const uint32_t localIdx : activeRigidParticleLocalIdxs_)
			{
				activeRankCounts[localIdx] = 1;
			}
			MarkTrace(Trace::TraceStage::Dist_RigidParticleAuthorityValidation);
			mpiContext_.AllReduceSumUint32s(activeRankCounts);

			const auto invalidAuthorityIt = std::find_if(
				activeRankCounts.begin(),
				activeRankCounts.end(),
				[](const uint32_t N_ranks) { return N_ranks != 1; });
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
		mpiContext_.AbortJob(
			"Failed to classify rank-local distributed work: " +
			std::string(ex.what()));
	}
	catch (...)
	{
		mpiContext_.AbortJob(
			"Failed to classify rank-local distributed work: unknown error.");
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }
	benchmarksTracker_->StopBench(Bench::BenchOp::Dist_ParticlePermutation);

	const bool pressureAccIsOwnedPrefix =
		pressureAccFluidParticleLocalIdxs_.size() ==
		particleStateF_.particleLocalIdx_Owned.size();

	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_PlanRebuild);
	try
	{
		MarkTrace(Trace::TraceStage::Dist_FluidSyncPlanRebuild);
		if (!RebuildFluidSolverSyncPlan()) { localOk = false; }

		bool configured = false;
		if (localOk)
		{
			MarkTrace(Trace::TraceStage::Dist_BackendConfiguration);
			configured = simBackend_->ConfigureDistributedStep(
				particleStateF_.particleLocalIdx_Owned,
				particleStateF_.particleLocalIdx_Outer,
				pressureAccFluidParticleLocalIdxs_,
				pressureAccIsOwnedPrefix,
				particleStateB_.N_owned,
				activeRigidParticleLocalIdxs_,
				fluidSolverSyncMgr_);
		}
		if (localOk && !configured)
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to configure rank-local distributed backend step.",
				mpiContext_.Rank(), mpiContext_.Size());
			localOk = false;
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to configure rank-local distributed backend step: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	benchmarksTracker_->StopBench(Bench::BenchOp::Dist_PlanRebuild);

	return mpiContext_.AllTrue(localOk);
}

bool DistSimulationMgr::FinishRankLocalBackendStep(
	const Sim::SimParams& params)
{
	MarkTrace(Trace::TraceStage::Dist_RigidBodySynchronization);

	// derived from the replicated global topology, so every rank agrees on whether the
	// rigid body collectives below run at all
	const bool hasRigidBodies =
		globalSimState_.particlesR.N_ > 0 ||
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
		MarkTrace(Trace::TraceStage::Dist_RigidForceCollection);
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
		MarkTrace(Trace::TraceStage::Dist_RigidForceReduction);
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
			MarkTrace(Trace::TraceStage::Dist_RigidForceIntegration);
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

	if (hasRigidBodies)
	{
		try
		{
			MarkTrace(Trace::TraceStage::Dist_RigidStateBroadcast);
			if (mpiContext_.IsRoot())
			{
				distParticleBackend_->CopyRigidBodyStateTo(
					localSimState_.rigidBodies);
			}

			BroadcastRigidBodyState(
				localSimState_.rigidBodies,
				rigidBodySyncBuffers_,
				mpiContext_);
			if (!distParticleBackend_->ApplyRigidBodyStateFrom(
				localSimState_.rigidBodies))
			{
				SPDLOG_ERROR(
					"[rank {}/{}] Backend rejected synchronized rigid body state.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
			}
		}
		catch (const std::exception& ex)
		{
			mpiContext_.AbortJob(
				"Failed to synchronize rigid body state: " + std::string(ex.what()));
		}
		catch (...)
		{
			mpiContext_.AbortJob(
				"Failed to synchronize rigid body state: unknown error.");
		}
	}

	// every earlier exit is already reduced, so reaching this point means all ranks succeeded
	return mpiContext_.AllTrue(localOk);
}

bool DistSimulationMgr::ValidateRankLocalBackendParticleState() const
{
	std::string fluidFailMsg{};
	const bool fluidStateValid =
		particleStateF_.IsValidLocalParticleState(
			localSimState_.particlesF.id,
			N_expectedGlobalF_,
			&fluidFailMsg);
	std::string boundaryFailMsg{};
	const bool boundaryStateValid =
		particleStateB_.IsValidLocalParticleState(
			localSimState_.particlesB.id,
			N_expectedGlobalB_,
			&boundaryFailMsg);
	const bool rigidStateStable =
		RigidStateMatchesGlobalState(localSimState_, globalSimState_);

	if (!fluidStateValid || !boundaryStateValid || !rigidStateStable)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Rank-local backend particle identity/layout validation failed "
			"(fluid={}, boundary={}, rigid={}).",
			mpiContext_.Rank(), mpiContext_.Size(),
			fluidStateValid ? "valid" : fluidFailMsg,
			boundaryStateValid ? "valid" : boundaryFailMsg,
			rigidStateStable ? "stable" : "changed");
		return false;
	}

	return true;
}

bool DistSimulationMgr::ValidateRankLocalParticleMetadata() const
{
	if (!distParticleBackend_) { return false; }

	const Sync::FluidConstView particlesF =
		distParticleBackend_->GetFluidParticleView();
	const Sync::BoundaryConstView particlesB =
		distParticleBackend_->GetBoundaryParticleView();

	std::string fluidFailMsg{};
	const bool fluidMetadataValid =
		particleStateF_.IsValidLocalParticleState(
			particlesF.id,
			N_expectedGlobalF_,
			&fluidFailMsg);

	std::string boundaryFailMsg{};
	const bool boundaryMetadataValid =
		particleStateB_.IsValidLocalParticleState(
			particlesB.id,
			N_expectedGlobalB_,
			&boundaryFailMsg);

	if (!fluidMetadataValid || !boundaryMetadataValid)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Invalid rank-local particle metadata "
			"(fluid={}, boundary={}).",
			mpiContext_.Rank(), mpiContext_.Size(),
			fluidMetadataValid ? "valid" : fluidFailMsg,
			boundaryMetadataValid ? "valid" : boundaryFailMsg);
		return false;
	}

	return true;
}

bool DistSimulationMgr::ApplyLastBackendParticleReOrder()
{
	if (!simBackend_) { return false; }

	std::span<const uint32_t> oldFluidLocalIdx_FromNewLocalIdx;
	std::span<const uint32_t> oldBoundaryLocalIdx_FromNewLocalIdx;

	const bool hasParticleReOrder =
		simBackend_->TryGetLastParticleReOrder(
			oldFluidLocalIdx_FromNewLocalIdx,
			oldBoundaryLocalIdx_FromNewLocalIdx);

	if (!hasParticleReOrder) { return true; }

	bool localOk = true;

	if (!ApplyLocalParticleReOrder(
		particleStateF_,
		oldFluidLocalIdx_FromNewLocalIdx,
		&fluidParticleReOrderWorkBuffers_))
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to apply backend fluid particle re-order.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	else
	{
		std::string failMsg{};
		if (!particleStateF_.IsValidLocalParticleLayout(
			static_cast<uint32_t>(oldFluidLocalIdx_FromNewLocalIdx.size()), &failMsg))
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Backend fluid particle re-order broke compact layout: {}",
				mpiContext_.Rank(), mpiContext_.Size(), failMsg);
			localOk = false;
		}
	}

	if (!ApplyLocalParticleReOrder(
		particleStateB_,
		oldBoundaryLocalIdx_FromNewLocalIdx,
		&boundaryParticleReOrderWorkBuffers_))
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to apply backend boundary particle re-order.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}
	else
	{
		std::string failMsg{};
		if (!particleStateB_.IsValidLocalParticleLayout(
			static_cast<uint32_t>(oldBoundaryLocalIdx_FromNewLocalIdx.size()), &failMsg))
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Backend boundary particle re-order broke compact layout: {}",
				mpiContext_.Rank(), mpiContext_.Size(), failMsg);
			localOk = false;
		}
	}

	return localOk;
}

bool DistSimulationMgr::TryReadProfileJson(
	const std::string& profile,
	std::string& jsonText)
{
	const std::filesystem::path paramsPath =
		std::filesystem::path(TSP_CONFIG_DIR) / Base::ParamsPathForProfile(profile);

	std::ifstream file(paramsPath, std::ios::in | std::ios::binary);
	if (!file) { return false; }

	jsonText.assign(
		std::istreambuf_iterator<char>(file),
		std::istreambuf_iterator<char>());
	return true;
}
