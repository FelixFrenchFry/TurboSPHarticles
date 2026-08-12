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
#include <span>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

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
	// checksum over the global indices owned by this rank, summed across all ranks it has
	// to stay at the value the initial partition produced (wrap-around sums are intended)
	uint64_t OwnedGlobalIdxChecksum(const DistParticleState& particleState)
	{
		uint64_t checksum = 0;

		for (const uint32_t localIdx : particleState.particleLocalIdx_Owned)
		{
			if (localIdx >= particleState.particleGlobalIdx_FromLocalIdx.size()) { continue; }

			// using SplitMix64-style mix
			uint64_t x = static_cast<uint64_t>(
				particleState.particleGlobalIdx_FromLocalIdx[localIdx]) + 0x9E3779B97F4A7C15ull;
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
		// the export directory is written by the root rank only
		loggingOptions.writeRunMetaToExportDir = mpiContext.IsRoot();
		loggingOptions.mpiRank = mpiContext.Rank();
		loggingOptions.N_mpiRanks = mpiContext.Size();
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

		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.idF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.xPosF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.yPosF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.zPosF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.velMagF));
		Mem::Add(usage, Mem::MemUsageOf_Vector(snap.rankF));
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

	Mem::MemUsage MemoryUsageOfExportSnapshotGatherBuffer(
		const DistExportSnapshotBuffer& gatherBuffer)
	{
		Mem::MemUsage usage{};

		Mem::Add(usage, Mem::MemUsageOf_Vector(gatherBuffer.fluidParticles));
		Mem::Add(usage, Mem::MemUsageOf_Vector(gatherBuffer.boundaryParticles));

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

	// the scenario builder stores the global scenario index in id, so a re-order that reached the
	// backend particle data but not the localIdx metadata (or the other way around) shows up here
	// (the rank-local scenario build slice is not addressable by global index, so the particle
	// payload itself can no longer be compared against it)
	bool FluidParticleIdsMatchLocalIdxMap(
		const Sync::UniversalSimState& backendState,
		const DistParticleState& particleState)
	{
		const auto& backendFluid = backendState.fluidParticles;

		if (particleState.particleGlobalIdx_FromLocalIdx.size() != backendFluid.N_)
		{
			return false;
		}
		if (backendFluid.id.size() != backendFluid.N_)
		{
			return false;
		}

		// TODO: parallelize this with TBB, or leave it because only called during init?
		for (uint32_t localIdx = 0; localIdx < backendFluid.N_; localIdx++)
		{
			if (backendFluid.id[localIdx] !=
				particleState.particleGlobalIdx_FromLocalIdx[localIdx])
			{
				return false;
			}
		}

		return true;
	}

	bool BoundaryParticleIdsMatchLocalIdxMap(
		const Sync::UniversalSimState& backendState,
		const DistParticleState& particleState)
	{
		const auto& backendBoundary = backendState.boundaryParticles;

		if (particleState.particleGlobalIdx_FromLocalIdx.size() != backendBoundary.N_)
		{
			return false;
		}
		if (backendBoundary.id.size() != backendBoundary.N_)
		{
			return false;
		}

		// TODO: parallelize this with TBB, or leave it because only called during init?
		for (uint32_t localIdx = 0; localIdx < backendBoundary.N_; localIdx++)
		{
			if (backendBoundary.id[localIdx] !=
				particleState.particleGlobalIdx_FromLocalIdx[localIdx])
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
	// every rank parses the same broadcast parameters, so all ranks pick the same strategy
	SelectDomainDecomp(simParams_.domainDecompMethod);
	if (!BuildDomainDecomp()) { return false; }
	if (!ConfigureLoadBalancing()) { return false; }

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

	distParticleBackend_ = simBackend_->GetDistributedParticleBackend();
	if (!distParticleBackend_)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Sim backend ({}) does not provide distributed "
			"particle storage access. CPU01 is currently the only distributed backend.",
			mpiContext_.Rank(), mpiContext_.Size(), static_cast<int>(version));
		simBackend_.reset();
		return;
	}

	simBackend_->GetBenchOps(curBenchOps_);
	const Bench::BenchOpList distBenchOps =
	{
		Bench::BenchOp::Dist_ParticleRedistribution,
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

	// use initial partition to validate if per-step ownership checksums match across ranks
	expectedOwnedIdxChecksumF_ =
		mpiContext_.AllReduceSumUint64(OwnedGlobalIdxChecksum(fluidParticleState_));
	expectedOwnedIdxChecksumB_ =
		mpiContext_.AllReduceSumUint64(OwnedGlobalIdxChecksum(boundaryParticleState_));

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
	report.backendVersion = Base::ToCString(simParams_.backendVersion);
	report.scenarioType = Base::ToCString(simParams_.scenarioType);
	report.neighborSearchMethod = Base::ToCString(simParams_.neighborSearchMethod);
	report.neighborListStorage = Base::ToEffectiveNeighborListStorageCString(
		simParams_.backendVersion,
		simParams_.neighborListMethod);

	report.N_globalF = static_cast<uint32_t>(N_expectedGlobalF_);
	report.N_globalB = static_cast<uint32_t>(N_expectedGlobalB_);
	report.N_globalR = globalSimState_.rigidParticles.N_;
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

	report.N_ownedF = fluidParticleState_.N_owned;
	report.N_ownedB = boundaryParticleState_.N_owned;
	report.N_replicatedR = localSimState_.rigidParticles.N_;

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
	particleRedistributionTimingReport_ = {};
	BeginRunLogging(paramsInit);
	DistDiagnosticsMgr::ResetProgress();
	bool runOk = true;

	// reported once the loop below is left, so every exit states why the run ended
	const char* stopReason = "unknown reason";

	// pack and serialize export snapshot of initial state
	if (paramsInit.exportEnabled)
	{
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

		SyncParameters(true);

		// local copy for step-local consistency
		auto params = simParams_;

		// the run length is fixed at launch, so progress reports match the step limit below
		params.maxSteps = paramsInit.maxSteps;
		const bool doSimStep = (params.simRunning || params.simRunNextStep);

		// let any rank stop the whole MPI loop cleanly
		const bool keepRunning = mpiContext_.AllTrue(
			!stopRequested_.load(std::memory_order_acquire));
		if (!keepRunning) { stopReason = "stop requested"; break; }

		// explicit maximum number of steps reached
		if (simStep_ >= static_cast<uint64_t>(paramsInit.maxSteps) &&
			paramsInit.maxSteps > 0)
		{
			stopReason = "maximum number of steps reached";
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
			if (!WriteExportSnapshot(params)) { runOk = false; }
		}

		if (runOk && !FillDistributedStatisticsRecord(params))
		{
			runOk = false;
		}

		// evaluate the benchmarks and memory usage for this step
		FinalizeBenchMemMeasurements(params);
		if (runOk && !UpdateLoadBalancing()) { runOk = false; }
		AppendLatestMemoryMetricsLog();
		AppendLatestBenchmarkMetricsLog();

		SyncParameters();

		DisplayMemoryUsageStats(params);

		// the progress block redraw would overwrite the error a failed stage just reported
		if (runOk) { DisplayDistributedProgress(params, exportedStep); }
	}

	auto paramsFinal = simParams_;
	paramsFinal.maxSteps = paramsInit.maxSteps;
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
						"Failed to load '/{}/params.json'. Falling back to '/default/params.json'.",
						mpiContext_, profile);
				}
				profile = "default";
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
				// the initial layout is uniform, so every rank holds the same subdomain extent
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
	scenarioMgr_.EndBuildRegion();

	// the generated particle set does not depend on the build region, so these counts are
	// the full scenario counts and every rank arrives at the same values
	N_expectedGlobalF_ = scenarioMgr_.GetBuiltGlobalFluidParticleCount();
	N_expectedGlobalB_ = scenarioMgr_.GetBuiltGlobalBoundaryParticleCount();

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
	pressureAccFluidParticleLocalIdxs_.clear();
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
		simParams_.distHaloRadius,
		static_cast<uint32_t>(N_expectedGlobalF_),
		static_cast<uint32_t>(N_expectedGlobalB_));

	localSimState_.ShrinkToFitAll();

	// the replicated scenario payload was only a build-time source, so
	// only the replicated rigid state is kept past the rank-local build
	globalSimState_.fluidParticles = {};
	globalSimState_.boundaryParticles = {};

	return true;
}

bool DistSimulationMgr::ValidateRankLocalSimState(
	const bool includeRigidState) const
{
	bool localOk = true;

	const uint32_t N_localF = localSimState_.fluidParticles.N_;
	const uint32_t N_localB = localSimState_.boundaryParticles.N_;

	std::string failMsg{};
	if (!fluidParticleState_.IsValidLocalLayout(N_localF, &failMsg))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid rank-local fluid particle layout: {}",
			mpiContext_.Rank(), mpiContext_.Size(), failMsg);
		localOk = false;
	}

	if (!boundaryParticleState_.IsValidLocalLayout(N_localB, &failMsg))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid rank-local boundary particle layout: {}",
			mpiContext_.Rank(), mpiContext_.Size(), failMsg);
		localOk = false;
	}

	if (fluidParticleState_.particleGlobalIdx_FromLocalIdx.size() !=
		static_cast<uint64_t>(N_localF))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid fluid local-to-global index map.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	if (fluidParticleState_.activeGlobalIdxFlags.size() != N_expectedGlobalF_)
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid fluid active global index flags.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	if (boundaryParticleState_.particleGlobalIdx_FromLocalIdx.size() !=
		static_cast<uint64_t>(N_localB))
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid boundary local-to-global index map.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	if (boundaryParticleState_.activeGlobalIdxFlags.size() != N_expectedGlobalB_)
	{
		SPDLOG_ERROR("[rank {}/{}] Invalid boundary active global index flags.",
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

bool DistSimulationMgr::ValidateGlobalOwnedParticleCounts() const
{
	// rank-local owned particle counts and global index checksums, summed across all ranks
	std::array<uint64_t, 4> N_globalOwned
	{
		fluidParticleState_.N_owned,
		boundaryParticleState_.N_owned,
		OwnedGlobalIdxChecksum(fluidParticleState_),
		OwnedGlobalIdxChecksum(boundaryParticleState_)
	};
	mpiContext_.AllReduceSumUint64s(N_globalOwned);

	const bool countsValid =
		N_globalOwned[0] == N_expectedGlobalF_ &&
		N_globalOwned[1] == N_expectedGlobalB_;
	const bool checksumsValid =
		N_globalOwned[2] == expectedOwnedIdxChecksumF_ &&
		N_globalOwned[3] == expectedOwnedIdxChecksumB_;

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
			"[rank {}/{}] Global owned particle indices are not owned by exactly one rank "
			"(fluid checksum={}/{}, boundary checksum={}/{}).",
			mpiContext_.Rank(), mpiContext_.Size(),
			N_globalOwned[2],
			expectedOwnedIdxChecksumF_,
			N_globalOwned[3],
			expectedOwnedIdxChecksumB_);
	}

	#if defined(_DEBUG)
	// [ rank-reduced ] names the first global index not owned by exactly one rank, detected by the checksums
	const auto ReportOwnedIdxConflicts = [&](
		const DistParticleState& particleState,
		const uint64_t N_global,
		const char* particleTypeName) -> bool
	{
		std::vector<uint32_t> ownerRankCounts(N_global, 0u);

		for (const uint32_t localIdx : particleState.particleLocalIdx_Owned)
		{
			if (localIdx >= particleState.particleGlobalIdx_FromLocalIdx.size()) { continue; }

			const uint32_t globalIdx = particleState.particleGlobalIdx_FromLocalIdx[localIdx];
			if (globalIdx < ownerRankCounts.size()) { ownerRankCounts[globalIdx] = 1; }
		}
		mpiContext_.AllReduceSumUint32s(ownerRankCounts);

		const auto invalidOwnerIt = std::find_if(
			ownerRankCounts.begin(),
			ownerRankCounts.end(),
			[](const uint32_t N_ranks) { return N_ranks != 1; });
		if (invalidOwnerIt != ownerRankCounts.end())
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Global {} particle {} has {} owner ranks (expected exactly one).",
				mpiContext_.Rank(), mpiContext_.Size(),
				particleTypeName,
				static_cast<uint32_t>(std::distance(ownerRankCounts.begin(), invalidOwnerIt)),
				*invalidOwnerIt);
			return false;
		}

		return true;
	};

	// both run unconditionally, because each one is a collective
	const bool fluidIdxsValid =
		ReportOwnedIdxConflicts(fluidParticleState_, N_expectedGlobalF_, "fluid");
	const bool boundaryIdxsValid =
		ReportOwnedIdxConflicts(boundaryParticleState_, N_expectedGlobalB_, "boundary");
	if (!fluidIdxsValid || !boundaryIdxsValid) { return false; }
	#endif

	return countsValid && checksumsValid;
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

			if (simBackend_->GetNumFluidParticles() != fluidParticleState_.N_owned +
				fluidParticleState_.N_halo)
			{
				SPDLOG_ERROR("[rank {}/{}] Rank-local backend fluid count mismatch.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
			}

			if (simBackend_->GetNumBoundaryParticles() != boundaryParticleState_.N_owned +
				boundaryParticleState_.N_halo)
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
			simBackend_->PushToUniversalSimState(backendSimState, false);
			localSimState_ = std::move(backendSimState);
			localSimState_.step = simStep_;
			localSimState_.time = simTime_;

			if (localOk && !ValidateRankLocalBackendParticleOrder()) { localOk = false; }

			// fluid and boundary payloads remain backend-resident after startup
			localSimState_.fluidParticles = {};
			localSimState_.boundaryParticles = {};
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

bool DistSimulationMgr::WriteExportSnapshot(const Sim::SimParams& params)
{
	benchmarksTracker_->StartBench(Bench::BenchOp::Serialization);
	if (!MaterializeRankLocalState())
	{
		benchmarksTracker_->StopBench(Bench::BenchOp::Serialization);
		return false;
	}

	DistSnapshotGatherStats stats{};

	auto& exportFluidParticleLocalIdxs = exportFluidParticleLocalIdxsScratch_;
	if (!BuildExportFluidParticleLocalIdxs(exportFluidParticleLocalIdxs))
	{
		localSimState_.fluidParticles = {};
		localSimState_.boundaryParticles = {};
		benchmarksTracker_->StopBench(Bench::BenchOp::Serialization);
		return false;
	}

	// fill export snapshot
	auto& expSnap = exportSnapScratch_;
	const bool gatherOk = GatherExportSnapshotToRoot(
		expSnap,
		exportSnapGatherScratch_,
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
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to gather the distributed export snapshot on the root rank.",
			mpiContext_.Rank(), mpiContext_.Size());
		localSimState_.fluidParticles = {};
		localSimState_.boundaryParticles = {};
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
	localSimState_.fluidParticles = {};
	localSimState_.boundaryParticles = {};

	return mpiContext_.AllTrue(writeOk);
}

bool DistSimulationMgr::BuildExportFluidParticleLocalIdxs(
	std::vector<uint32_t>& localIdxs) const
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
	globalNearSurfaceFlags.assign(N_expectedGlobalF_, 0);

	if (localOk)
	{
		for (const uint32_t localIdx : fluidParticleState_.particleLocalIdx_Owned)
		{
			if (localIdx >= localSimState_.fluidParticles.N_ ||
				!fluidParticleState_.HasMetadataAtLocalIdx(localIdx))
			{
				SPDLOG_ERROR(
					"[rank {}/{}] Invalid owned fluid particle metadata "
					"during distributed surface export.",
					mpiContext_.Rank(), mpiContext_.Size());
				localOk = false;
				break;
			}

			const uint32_t globalIdx =
				fluidParticleState_.particleGlobalIdx_FromLocalIdx[localIdx];
			if (globalIdx >= N_expectedGlobalF_)
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
		for (uint32_t localIdx = 0; localIdx < localSimState_.fluidParticles.N_; localIdx++)
		{
			if (!fluidParticleState_.HasMetadataAtLocalIdx(localIdx))
			{
				localOk = false;
				break;
			}

			const uint32_t globalIdx = fluidParticleState_.particleGlobalIdx_FromLocalIdx[localIdx];
			if (globalIdx >= N_expectedGlobalF_)
			{
				localOk = false;
				break;
			}

			localNearSurfaceFlagsScratch_[localIdx] = globalNearSurfaceFlags[globalIdx];
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
		configProfile_, paramsFile_, scenariosFile_, params,
		simParamsMgr_.CliOverrides(), mpiContext_.IsRoot());
}

void DistSimulationMgr::EndRunLogging()
{
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
	Mem::Add(particleMetadataUsage, GetParticleMemoryUsage(fluidParticleState_));
	Mem::Add(particleMetadataUsage, GetParticleMemoryUsage(boundaryParticleState_));
	Mem::Set(record, Mem::MemSpace::RAM, Mem::MemBucket::Dist_ParticleMetadata, particleMetadataUsage);

	Mem::MemUsage exchangeBufferUsage{};
	Mem::Add(exchangeBufferUsage, particleRedistributionMgr_.GetExchangeBufferMemoryUsage());
	Mem::Add(exchangeBufferUsage, fluidSolverSyncMgr_.GetExchangeBufferMemoryUsage());
	Mem::Set(record, Mem::MemSpace::RAM, Mem::MemBucket::Dist_ExchangeBuffers, exchangeBufferUsage);

	Mem::MemUsage miscUsage{};
	Mem::Add(miscUsage, particleRedistributionMgr_.GetScratchMemoryUsage());
	Mem::Add(miscUsage, fluidSolverSyncMgr_.GetPlanMemoryUsage());
	Mem::Add(miscUsage, GetParticleMemoryUsage(fluidParticleReOrderWorkBuffers_));
	Mem::Add(miscUsage, GetParticleMemoryUsage(boundaryParticleReOrderWorkBuffers_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(pressureAccFluidParticleLocalIdxs_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(activeRigidParticleLocalIdxs_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(activeRigidParticleRankCountScratch_));
	Mem::Add(miscUsage, MemoryUsageOfRigidBodyForceTorque(rigidBodyForceTorque_));
	Mem::Add(miscUsage, MemoryUsageOfRigidBodySyncBuffers(rigidBodySyncBuffers_));
	Mem::Add(miscUsage, MemoryUsageOfStatisticsPartial(statisticsPartialScratch_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(localNearSurfaceFlagsScratch_));
	Mem::Add(miscUsage, Mem::MemUsageOf_Vector(globalNearSurfaceFlagsScratch_));
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

	partial.N_B = boundaryParticleState_.N_owned;
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

	report.N_ownedF = fluidParticleState_.N_owned;
	report.N_innerF = static_cast<uint32_t>(fluidParticleState_.particleLocalIdx_Inner.size());
	report.N_outerF = static_cast<uint32_t>(fluidParticleState_.particleLocalIdx_Outer.size());
	report.N_haloF = fluidParticleState_.N_halo;
	report.N_ownedB = boundaryParticleState_.N_owned;
	report.N_innerB = static_cast<uint32_t>(boundaryParticleState_.particleLocalIdx_Inner.size());
	report.N_outerB = static_cast<uint32_t>(boundaryParticleState_.particleLocalIdx_Outer.size());
	report.N_haloB = boundaryParticleState_.N_halo;

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
		const Mem::MemSpace memSpace = PrimaryMemSpace(params.backendVersion);
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
		BuildRankBenchReport(),
		particleRedistributionTimingReport_);
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
		.N_ownedF = fluidParticleState_.N_owned,
		.N_ownedB = boundaryParticleState_.N_owned,
		.N_haloF = fluidParticleState_.N_halo,
		.N_haloB = boundaryParticleState_.N_halo,
		.rankBackendStepTimeNs =
			benchmarksTracker_->GetBench_Latest(Bench::BenchOp::Dist_RankBackendStep)
	};
	const Sync::FluidConstView fluidParticles =
		distParticleBackend_->GetFluidParticleView();
	const Sync::BoundaryConstView boundaryParticles =
		distParticleBackend_->GetBoundaryParticleView();
	const DomainDecompRepartitionParticleInput localParticleInput
	{
		.weightModel = simParams_.loadBalanceWeightModel,
		.rank = mpiContext_.Rank(),
		.fluid =
		{
			.x = fluidParticles.xPos,
			.y = fluidParticles.yPos,
			.z = fluidParticles.zPos,
			.ownedLocalIdx = fluidParticleState_.particleLocalIdx_Owned
		},
		.boundary =
		{
			.x = boundaryParticles.xPos,
			.y = boundaryParticles.yPos,
			.z = boundaryParticles.zPos,
			.ownedLocalIdx = boundaryParticleState_.particleLocalIdx_Owned
		}
	};

	std::string failMsg{};
	const bool ok = loadBalanceMgr_.EvaluateLoadAndRebalance(
		localSample,
		localParticleInput,
		*domainDecomp_,
		mpiContext_,
		&failMsg);
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
		if (!mpiContext_.AllTrue(localOk)) { return false; }

		// ensure all ranks have the same parameters before redistribution
		SyncParameters();

		BeginParticleRedistributionTimingReport(particleRedistributionTimingReport_, mpiContext_.Rank());
		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_ParticleRedistribution);
		const bool redistributionOk = RedistributeRankLocalParticles(params);
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_ParticleRedistribution);
		if (!redistributionOk) { return false; }

		if (!ValidateGlobalOwnedParticleCounts())
		{
			return false;
		}

		if (!PrepareRankLocalBackendStep()) { return false; }

		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_RankBackendStep);
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
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_RankBackendStep);
		if (!mpiContext_.AllTrue(localOk)) { return false; }

		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_StateSync);
		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_StateHandoff);
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
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to apply backend particle re-order: {}",
				mpiContext_.Rank(), mpiContext_.Size(), ex.what());
			localOk = false;
		}
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_StateHandoff);
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_StateSync);
		if (!mpiContext_.AllTrue(localOk)) { return false; }

		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_StateSync);
		benchmarksTracker_->StartBench(Bench::BenchOp::Dist_StateHandoff);
		const bool finishOk = FinishRankLocalBackendStep(params);
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_StateHandoff);
		benchmarksTracker_->StopBench(Bench::BenchOp::Dist_StateSync);
		if (!finishOk) { return false; }

		if (!ValidateRankLocalParticleMetadata()) { localOk = false; }
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR("[rank {}/{}] Failed distributed step: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
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
		if (localOk && !particleRedistributionMgr_.RedistributeParticles(
			*distParticleBackend_,
			fluidParticleState_,
			boundaryParticleState_,
			*domainDecomp_,
			mpiContext_,
			params,
			static_cast<uint32_t>(N_expectedGlobalF_),
			static_cast<uint32_t>(N_expectedGlobalB_),
			&failMsg,
			&particleRedistributionTimingReport_))
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to redistribute particles: {}",
				mpiContext_.Rank(), mpiContext_.Size(), failMsg);
			localOk = false;
		}
	}
	catch (const std::exception& ex)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to redistribute particles: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}

	return AllRanksTrue(localOk);
}

bool DistSimulationMgr::MaterializeRankLocalState()
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
			simBackend_->PushToUniversalSimState(backendSimState, false);
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
	// derived from the replicated global topology, so every rank agrees on whether the
	// rigid body collectives below run at all
	const bool hasRigidBodies =
		globalSimState_.rigidParticles.N_ > 0 ||
		globalSimState_.rigidBodies.N_ > 0;

	bool localOk = true;
	const bool computeFirstHaloLayerPressureAcc =
		simParams_.distHaloMode == Base::DistHaloMode::DoubleHalo;
	if (computeFirstHaloLayerPressureAcc)
	{
		// upper bound for the owned indices plus the appended first halo layer
		pressureAccFluidParticleLocalIdxs_.reserve(
			fluidParticleState_.N_owned + fluidParticleState_.N_halo);
	}
	pressureAccFluidParticleLocalIdxs_ =
		fluidParticleState_.particleLocalIdx_Owned;
	activeRigidParticleLocalIdxs_.clear();

	try
	{
		// computes pressure acceleration for the first halo layer locally
		if (computeFirstHaloLayerPressureAcc)
		{
			const Sync::FluidConstView fluidParticles =
				distParticleBackend_->GetFluidParticleView();
			const AABB singleHaloBounds =
				domainDecomp_->OwnedExtentAABB(mpiContext_.Rank())
					.Expanded(simParams_.kernelRadius);

			for (const uint32_t localIdx : fluidParticleState_.particleLocalIdx_Halo)
			{
				const glm::vec3 position{
					fluidParticles.xPos[localIdx],
					fluidParticles.yPos[localIdx],
					fluidParticles.zPos[localIdx]
				};
				if (singleHaloBounds.Contains(position))
				{
					pressureAccFluidParticleLocalIdxs_.push_back(localIdx);
				}
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
			activeRankCounts.assign(localSimState_.rigidParticles.N_, 0);
			for (const uint32_t localIdx : activeRigidParticleLocalIdxs_)
			{
				activeRankCounts[localIdx] = 1;
			}
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
		SPDLOG_ERROR(
			"[rank {}/{}] Failed to classify rank-local distributed work: {}",
			mpiContext_.Rank(), mpiContext_.Size(), ex.what());
		localOk = false;
	}
	if (!mpiContext_.AllTrue(localOk)) { return false; }

	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_StateSync);
	benchmarksTracker_->StartBench(Bench::BenchOp::Dist_StateHandoff);
	try
	{
		std::string failMsg{};
		if (!fluidSolverSyncMgr_.Rebuild(
			distParticleBackend_->GetFluidParticleView(),
			fluidParticleState_,
			*domainDecomp_,
			simParams_.kernelRadius,
			simParams_.distHaloMode,
			N_expectedGlobalF_,
			mpiContext_,
			*benchmarksTracker_,
			&failMsg))
		{
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to prepare distributed fluid solver synchronization: {}",
				mpiContext_.Rank(), mpiContext_.Size(), failMsg);
			localOk = false;
		}

		bool configured = false;
		if (localOk)
		{
			configured = simBackend_->ConfigureDistributedStep(
				fluidParticleState_.particleLocalIdx_Owned,
				pressureAccFluidParticleLocalIdxs_,
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
	benchmarksTracker_->StopBench(Bench::BenchOp::Dist_StateHandoff);
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

	if (hasRigidBodies)
	{
		try
		{
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
			SPDLOG_ERROR(
				"[rank {}/{}] Failed to synchronize rigid body state: {}",
				mpiContext_.Rank(), mpiContext_.Size(), ex.what());
			localOk = false;
		}
	}

	// every earlier exit is already reduced, so reaching this point means all ranks succeeded
	return mpiContext_.AllTrue(localOk);
}

bool DistSimulationMgr::ValidateRankLocalBackendParticleOrder() const
{
	const bool fluidOrderStable =
		FluidParticleIdsMatchLocalIdxMap(localSimState_, fluidParticleState_);
	const bool boundaryOrderStable =
		BoundaryParticleIdsMatchLocalIdxMap(localSimState_, boundaryParticleState_);
	const bool rigidStateStable =
		RigidStateMatchesGlobalState(localSimState_, globalSimState_);

	if (!fluidOrderStable || !boundaryOrderStable || !rigidStateStable)
	{
		SPDLOG_ERROR(
			"[rank {}/{}] Rank-local backend particle order changed "
			"(fluid={}, boundary={}, rigid={}) and DistParticleState localIdx metadata "
			"or payload does not match the backend state.",
			mpiContext_.Rank(), mpiContext_.Size(),
			fluidOrderStable ? "stable" : "changed",
			boundaryOrderStable ? "stable" : "changed",
			rigidStateStable ? "stable" : "changed");
		return false;
	}

	return true;
}

bool DistSimulationMgr::ValidateRankLocalParticleMetadata() const
{
	if (!distParticleBackend_) { return false; }

	const uint32_t N_localF = distParticleBackend_->GetFluidParticleView().N_;
	const uint32_t N_localB = distParticleBackend_->GetBoundaryParticleView().N_;

	std::string fluidFailMsg{};
	const bool fluidMetadataValid =
		fluidParticleState_.IsValidLocalParticleState(
			N_localF,
			static_cast<uint32_t>(N_expectedGlobalF_),
			&fluidFailMsg);

	std::string boundaryFailMsg{};
	const bool boundaryMetadataValid =
		boundaryParticleState_.IsValidLocalParticleState(
			N_localB,
			static_cast<uint32_t>(N_expectedGlobalB_),
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
		fluidParticleState_,
		oldFluidLocalIdx_FromNewLocalIdx,
		static_cast<uint32_t>(N_expectedGlobalF_),
		&fluidParticleReOrderWorkBuffers_))
	{
		SPDLOG_ERROR("[rank {}/{}] Failed to apply backend fluid particle re-order.",
			mpiContext_.Rank(), mpiContext_.Size());
		localOk = false;
	}

	if (!ApplyLocalParticleReOrder(
		boundaryParticleState_,
		oldBoundaryLocalIdx_FromNewLocalIdx,
		static_cast<uint32_t>(N_expectedGlobalB_),
		&boundaryParticleReOrderWorkBuffers_))
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
