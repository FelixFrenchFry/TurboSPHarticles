// simulation.cpp

// external
#include <chrono>
#include <mutex>
#include <spdlog/spdlog.h>
#include <stdexcept>

// internal
#include "source/eval/memory/memory.h"
#include "source/eval/serial/serialization.h"
#include "source/simulation/core/diagnostics.h"
#include "source/simulation/CPU/backend/cpu_backend.h"
#ifdef TURBOSPHARTICLES_WITH_CUDA
	#include "source/simulation/GPU/backend/gpu_backend.h"
#endif
#include "simulation.h"



using namespace Sim;

SimulationMgr::SimulationMgr(Sim::SimParamsMgr& simParamsMgr,
							 Scene::ScenarioMgr& scenarioMgr,
							 Bench::BenchmarksTracker& benchmarksTracker,
							 Mem::MemoryTracker& memoryTracker,
							 Stats::StatisticsTracker& statisticsTracker,
							 Export::SerializationMgr& serializationMgr)
	: simParamsMgr_{ simParamsMgr }
	, scenarioMgr_{ scenarioMgr }
	, benchmarksTracker_{ benchmarksTracker }
	, memoryTracker_{ memoryTracker }
	, statisticsTracker_{ statisticsTracker }
	, serializationMgr_{ serializationMgr }
{
	// nothing to do
}

void SimulationMgr::Init(const std::string& restartCheckpoint)
{
	initTime_ = std::chrono::steady_clock::now();

	// initial snapshot
	SyncParameters(true);

	// default sim state (empty)
	universalSimState_ = Sync::UniversalSimState{};

	SPDLOG_INFO("Init simulation state...");
	if (restartCheckpoint.empty())
	{
		SelectSimScenario(simParams_.scenarioType, configProfile_);
		ResetAndRebuildSimState();
	}
	else // restart from a simulation state checkpoint
	{
		LoadCheckpoint(restartCheckpoint);
		simParamsMgr_.Set_SimRunning(false);
		simParamsMgr_.Set_SimRunNextStep(false);
		SyncParameters();
	}
	SPDLOG_INFO("Init backend data structures...");
	CreateAndInitializeSimBackend(simParams_.backendType);

	// updated snapshot
	initialized_.store(true, std::memory_order_release);
	SyncParameters(true);

	const double initializationTimeS = std::chrono::duration<double>(
		std::chrono::steady_clock::now() - initTime_).count();
	SPDLOG_INFO("Initialization finished after {:.2f}s", initializationTimeS);

	DiagnosticsMgr::PrintSimParams(simParams_);
	DiagnosticsMgr::PrintSimSetup(*this, simParams_);

}

void SimulationMgr::Shutdown()
{
	RequestStop();
	JoinThreads();
	EndRunLogging();

	// TODO: free memory occupied by particles, grids, etc...

	initialized_.store(false, std::memory_order_release);
	simBackend_.reset();
}

void SimulationMgr::SyncParameters(
	bool force)
{
	uint64_t latestSimParamsVersion = simParamsMgr_.StableVersion();
	if (latestSimParamsVersion != simParamsVersion_ || force)
	{
		auto snapshot = simParamsMgr_.SnapshotParams();

		simParams_ = std::get<0>(snapshot);
		simParamsVersion_ = std::get<1>(snapshot);
	}
}

void SimulationMgr::SelectSimScenario(
	Base::ScenarioType type,
	const std::string& profile)
{
	if (running_.load(std::memory_order_acquire))
	{
		SPDLOG_ERROR("Stop the simulation before switching scenario.");
		return;
	}

	// create new scenario
	simScenario_ = Scene::MakeSimScenario(scenarioMgr_, type, profile);
	if (!simScenario_)
	{
		throw std::runtime_error("Failed to create simulation scenario " +
			std::to_string(static_cast<int>(type)) + ".");
	}
}

void SimulationMgr::CreateAndInitializeSimBackend(
	Base::BackendType backendType)
{
	if (running_.load(std::memory_order_acquire))
	{
		SPDLOG_ERROR("Stop the simulation before switching backend.");
		return;
	}

	if (simBackend_)
	{
		// TODO: implement?
		SPDLOG_ERROR("Simulation backend switching is currently disabled.");
		return;
	}

	// create new backend and initialize it with the universal state
	simBackend_ = MakeSimBackend(backendType, benchmarksTracker_, simParams_);
	if (!simBackend_)
	{
		throw std::runtime_error("Failed to create simulation backend " +
			std::to_string(static_cast<int>(backendType)) + ".");
	}

	{
		std::scoped_lock lock(stateMutex_);

		ApplySimDomainBoundsToBackend();

		// transfer universal state into backend-specific data structures
		simBackend_->PullFromUniversalSimState(universalSimState_, simParams_);
	}

	// clear benchmarks from previous backend
	benchmarksTracker_.ClearRecordHistory();

	// get and validate the new backend's benchmark operation order
	simBackend_->GetBenchOps(curBenchOps_);
	if (!Bench::ValidateOpList(curBenchOps_))
	{
		throw std::runtime_error("Invalid benchmark operation list for simulation backend " +
			std::to_string(static_cast<int>(backendType)) + ".");
	}
}

void SimulationMgr::ResetAndRebuildSimState()
{
	// stop simulation
	simParamsMgr_.Set_SimRunning(false);
	simParamsMgr_.Set_SimRunNextStep(false);

	// lock state during re-build
	std::unique_lock lock(stateMutex_);

	// reset metadata
	simStep_ = 0;
	simTime_ = 0.0;
	lastVisStep_.reset();
	if (visualSnapMailbox_) { visualSnapMailbox_->RequestSnapshot(); }
	serializationMgr_.ResetExportDirState();
	exportSnapshotDir_.clear();

	// reset scenario and all its data
	universalSimState_ = {};
	universalSimState_.step = 0;
	universalSimState_.time = 0.0;

	SyncParameters();

	// checkpoint startup does not generate a scene, but a reset uses the current profile
	if (!simScenario_)
	{
		simScenario_ = Scene::MakeSimScenario(scenarioMgr_, simParams_.scenarioType, configProfile_);
		if (!simScenario_) { throw std::runtime_error("Failed to create simulation scenario for reset."); }
	}

	// re-build initial scenario state by transferring into universal state
	scenarioMgr_.ResetBoundaryMeshParticleIdBegin();
	simScenario_->Build(universalSimState_, simParams_);
	meshIdBeginB_ = scenarioMgr_.GetBoundaryMeshParticleIdBegin();
	boundaryBoxes_ = simScenario_->GetBoundaryBoxes();
	DomainBounds bounds{};
	domainBounds_.reset();
	if (Scene::ComputeSimDomainBounds(boundaryBoxes_, simParams_, bounds.min, bounds.max))
	{
		domainBounds_ = bounds;
	}
	scenarioMgr_.FinalizeGeneratedParticleCounts(universalSimState_);
	universalSimState_.particlesF.ensureParticleIdsPresent();
	universalSimState_.particlesB.ensureParticleIdsPresent();
	universalSimState_.particlesR.ensureParticleIdsPresent();
	universalSimState_.particlesB.ensureBoundaryCoefficientsPresent(
		simParams_.massB, simParams_.restDensity);
	universalSimState_.particlesR.ensureRigidCoefficientsPresent(
		simParams_.massB, simParams_.restDensity);
	universalSimState_.ValidateRigidAttachments();
	Sync::CollectBoundaryMeshParticles(
		universalSimState_.particlesB.constView(), meshIdBeginB_, meshPosB_, meshVolB_);

	// compact staged particle vectors once after scenario construction
	universalSimState_.ShrinkToFitAll();

	if (simBackend_)
	{
		ApplySimDomainBoundsToBackend();

		// transfer universal state into backend-specific data structures
		simBackend_->PullFromUniversalSimState(universalSimState_, simParams_);
		return;
	}
}

void SimulationMgr::ApplySimDomainBoundsToBackend()
{
	if (!simBackend_) { return; }
	if (domainBounds_)
	{
		simBackend_->SetSimDomainBounds(domainBounds_->min, domainBounds_->max);
	}
	else
	{
		// invalid extents clear any bounds retained from a previous scene
		simBackend_->SetSimDomainBounds(glm::vec3(0.0f), glm::vec3(0.0f));
	}
}

void SimulationMgr::Launch_Async()
{
	if (!initialized_.load(std::memory_order_acquire))
	{
		SPDLOG_ERROR("Failed to launch async SimulationMgr (not initialized).");
		return;
	}
	if (!simBackend_)
	{
		CreateAndInitializeSimBackend(simParams_.backendType);
	}
	if (!simBackend_)
	{
		SPDLOG_ERROR("Failed to launch async SimulationMgr (no backend selected).");
		return;
	}
	if (running_.exchange(true))
	{
		SPDLOG_ERROR("Failed to launch async SimulationMgr (already running).");
		return;
	}

	stopRequested_.store(false, std::memory_order_release);

	// launch the controller thread
	controllerThread_ = std::jthread([this](const std::stop_token& stopToken)
		{
			try
			{
				RunSim(stopToken, true);
			}
			catch (const std::exception& ex)
			{
				SPDLOG_ERROR("Simulation thread failed: {}", ex.what());
				simParamsMgr_.Set_SimRunning(false);
				EndRunLogging();
			}
			catch (...)
			{
				SPDLOG_ERROR("Simulation thread failed: Unknown exception.");
				simParamsMgr_.Set_SimRunning(false);
				EndRunLogging();
			}

			running_.store(false, std::memory_order_release);
		});
}

void SimulationMgr::Launch_Blocking()
{
	if (!initialized_.load(std::memory_order_acquire))
	{
		SPDLOG_ERROR("Failed to launch blocking SimulationMgr (not initialized).");
		return;
	}
	if (!simBackend_)
	{
		CreateAndInitializeSimBackend(simParams_.backendType);
	}
	if (!simBackend_)
	{
		SPDLOG_ERROR("Failed to launch blocking SimulationMgr (no backend selected).");
		return;
	}
	if (running_.exchange(true))
	{
		SPDLOG_ERROR("Failed to launch blocking SimulationMgr (already running).");
		return;
	}

	stopRequested_.store(false, std::memory_order_release);

	// run on the calling thread and leave exceptions for the app to report
	try
	{
		RunSim(std::stop_token{}, false);
	}
	catch (...)
	{
		running_.store(false, std::memory_order_release);
		EndRunLogging();
		throw;
	}
	running_.store(false, std::memory_order_release);
}

void SimulationMgr::RunSim(const std::stop_token& stopToken, const bool publishVis)
{
	printf("\n");
	SPDLOG_INFO("Simulation started running...");

	// --- [ initialize before entering the simulation loop ] ----------------------------------------

	std::unique_lock stateLock(stateMutex_);

	// local copy for thread safety
	const auto paramsInit = simParams_;
	const uint64_t startStep = simStep_;
	BeginRunLogging(paramsInit);
	DiagnosticsMgr::ResetProgress(startStep);

	// initial neighbor search for correct statistics
	simBackend_->ExecuteNeighborSearch(simParams_);

	if (publishVis)
	{
		lastVisStep_.reset();
		PublishVisualSnapshot(true);
	}

	// pack and serialize the initial export snapshot
	if (paramsInit.exportEnabled && simStep_ == 0)
	{
		WriteExportSnapshot(paramsInit);
	}

	AppendInitialMemoryMetricsLog();
	const char* stopReason = "unknown reason";
	bool checkpointDue = false;

	// --- [ enter the simulation loop ] -------------------------------------------------------------

	stateLock.unlock();
	while (true)
	{
		std::unique_lock stepLock(stateMutex_);
		checkpointDue = paramsInit.checkpointTime > 0.0 &&
			std::chrono::duration<double>(std::chrono::steady_clock::now() - initTime_).count() >=
			paramsInit.checkpointTime;
		if (checkpointDue)
		{
			stopReason = "checkpoint time reached";
			SPDLOG_INFO("Checkpoint time reached, saving checkpoint and stopping...");
			break;
		}

		if (stopToken.stop_requested() ||
			stopRequested_.load(std::memory_order_acquire))
		{
			stopReason = "stop requested";
			break;
		}

		SyncParameters(true);

		// local copy for thread safety
		auto params = simParams_;

		// the run length is fixed at launch, so progress reports match the step limit below
		params.maxSteps = paramsInit.maxSteps;
		const bool doSimStep = (params.simRunning || params.simRunNextStep);
		DiagnosticsMgr::SetProgressRunning(doSimStep);

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
			// service requests arriving after a single step
			if (publishVis) { PublishVisualSnapshot(); }
			stepLock.unlock();
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		// reset data records for the new step
		benchmarksTracker_.BeginStep();
		memoryTracker_.BeginStep();
		statisticsTracker_.BeginStep();

		// starting point of total benchmark
		benchmarksTracker_.StartBench(Bench::BenchOp::Total);

		// perform the actual simulation step
		simBackend_->ExecuteFullSimStep(params);

		// update metadata
		simStep_ += 1;
		simTime_ += params.timeStep;

		// reset single-step flag
		if (simParams_.simRunNextStep) { simParamsMgr_.Set_SimRunNextStep(false); }

		if (publishVis)
		{
			const auto start = std::chrono::steady_clock::now();
			if (PublishVisualSnapshot())
			{
				const auto elapsed = std::chrono::steady_clock::now() - start;
				benchmarksTracker_.AddTimingSample(Bench::BenchOp::Visualization,
					static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
			}
		}

		// pack and serialize export snapshot at given frequency
		if (simStep_ % params.exportInterval == 0 &&
			params.exportEnabled)
		{
			benchmarksTracker_.StartBench(Bench::BenchOp::Serialization);
			WriteExportSnapshot(params);
			benchmarksTracker_.StopBench(Bench::BenchOp::Serialization);
		}

		// evaluate the benchmarks, memory usage, and statistics for this step
		FinalizeBenchMemStatMeasurements();
		AppendLatestMemoryMetricsLog();
		AppendLatestBenchmarkMetricsLog();

		// pack and send benchmark and statistics snapshots
		HandleBenchMemStatSnapshots();

		SyncParameters();

		DisplayMemoryUsageStats(params, simStep_ == startStep + 1);
		DisplayPerformanceStats(params);

		// TODO: check if scenario or backend need to be switched?
	}

	// --- [ finalize after exiting the simulation loop ] --------------------------------------------

	stateLock.lock();

	// local copy for thread safety
	const auto paramsFinal = simParams_;

	// retain all particles so visibility can change after finalization releases the backend
	if (publishVis) { PublishVisualSnapshot(true, true); }

	FinalizeRunState(paramsFinal, checkpointDue);

	DisplayMemoryUsageStats(paramsFinal, true);
	const bool processMemoryChanged = memoryTracker_.SampleProcessMemory();
	if (processMemoryChanged) { AppendLatestMemoryMetricsLog(); }

	EndRunLogging();

	printf("\n");
	SPDLOG_INFO("Simulation stopped running ({})", stopReason);
	DiagnosticsMgr::PrintFinalProgressPeakRam(
		memoryTracker_.GetCurrentRecord().process.N_residentBytes,
		memoryTracker_.GetCurrentRecord().process.N_peakResidentBytes,
		memoryTracker_.GetCurrentRecord().device.N_residentBytes,
		memoryTracker_.GetCurrentRecord().device.N_peakResidentBytes);
}

// requests the controller thread to stop cleanly
void SimulationMgr::RequestStop()
{
	stopRequested_.store(true, std::memory_order_release);

	if (controllerThread_.joinable())
	{
		controllerThread_.request_stop(); // non-blocking
	}
}

// blocks until the controller thread has stopped
void SimulationMgr::JoinThreads()
{
	if (controllerThread_.joinable())
	{
		controllerThread_.join();
		running_.store(false, std::memory_order_release);
	}
}

// sets the configuration profile and updates derived file paths
void SimulationMgr::SetConfigProfile(std::string profile)
{
	configProfile_ = std::move(profile);
	paramsFile_ = Base::ParamsPathForProfile(configProfile_);
	scenariosFile_ = Base::ScenariosPathForProfile(configProfile_);
}

void SimulationMgr::WriteExportSnapshot(const SimParams& params)
{
	// reserve the export directory on the first snapshot of this session
	if (exportSnapshotDir_.empty() || exportSnapshotBaseDir_ != params.exportDir)
	{
		exportSnapshotBaseDir_ = params.exportDir;
		exportSnapshotDir_ = serializationMgr_.CreateExportDirectory(params.exportDir);
		loggingMgr_.CopyRunMetaToExportDir(exportSnapshotDir_);
		exportBoundaryWritten_ = false;
		SPDLOG_INFO("Snapshots: {}", exportSnapshotDir_);
	}

	// fill export snapshot
	auto& expSnap = exportSnapScratch_;
	simBackend_->PackExportSnapshot(expSnap, simParams_, !exportBoundaryWritten_, meshIdBeginB_);

	// the non-distributed path owns every fluid particle, so the ranks are not exported
	expSnap.rankF.clear();

	// write export snapshot to files
	serializationMgr_.SaveExportSnapshot(expSnap, exportSnapshotDir_, !exportBoundaryWritten_);
	serializationMgr_.SaveExportSnapshotXdmfViews(expSnap, exportSnapshotDir_);
	exportBoundaryWritten_ = true;
}

void SimulationMgr::FinalizeRunState(const SimParams& params, const bool checkpointDue)
{
	if (!params.saveFinalState && !checkpointDue)
	{
		simBackend_.reset();
		return;
	}

	SaveCheckpoint(params);
}

void SimulationMgr::SetVisualParticleVisibility(const bool showParticlesF, const bool showParticlesR)
{
	const uint8_t particleMask = (showParticlesF ? visualParticleMaskF_ : 0) |
		(showParticlesR ? visualParticleMaskR_ : 0);
	visualParticleMask_.store(particleMask, std::memory_order_release);
}

bool SimulationMgr::PublishVisualSnapshot(const bool force, const bool includeAllParticles)
{
	const auto scalarMode = visualScalarMode_.load(std::memory_order_acquire);
	const uint8_t particleMask = includeAllParticles
		? visualParticleMaskF_ | visualParticleMaskR_
		: visualParticleMask_.load(std::memory_order_acquire);
	if (lastVisStep_ == simStep_ && lastVisScalarMode_ == scalarMode &&
		lastVisParticleMask_ == particleMask) { return false; }

	// preserve requests arriving during packing
	const bool requested = visualSnapMailbox_->TakeSnapshotRequest();
	if (!force && !requested) { return false; }
	const bool includeF = (particleMask & visualParticleMaskF_) != 0;
	const bool includeR = (particleMask & visualParticleMaskR_) != 0;

	// create visual snapshot and push pack misc data that is not handled by the backend
	// (if CUDA path is taken, the particle data buffers are not filled)
	auto visSnap = std::make_shared<Sync::VisualSnapshot>();
	visSnap->simStep = simStep_;
	visSnap->boxesB = boundaryBoxes_;
	visSnap->meshPosB = meshPosB_;
	visSnap->meshVolB = meshVolB_;
	visSnap->scalarModeF = scalarMode;

	// optionally pack particle data into CUDA-device buffers for OpenGL without going through the host
#ifdef TURBOSPHARTICLES_WITH_CUDA
	auto& visualFrameMailboxCUDA = visualSnapMailbox_->GetFrameMailboxCUDA();
	const auto result = simBackend_->TryPackVisualSnapshotCUDA(visualFrameMailboxCUDA, visSnap, includeF, includeR);
	if (result == Sync::VisualPackResultCUDA::Skipped && !force)
	{
		visualSnapMailbox_->RequestSnapshot();
		return false;
	}
	if (result == Sync::VisualPackResultCUDA::Published)
	{
		// discard stale host fallback
		visualSnapMailbox_->StoreSnapshot(nullptr);
	}
	if (result != Sync::VisualPackResultCUDA::Published)
#endif
	{
		// fallback to host snapshot packing if CUDA path is unavailable or unsupported
		simBackend_->PackVisualSnapshot(*visSnap, includeF, includeR);
		visualSnapMailbox_->StoreSnapshot(std::move(visSnap));
	}
	lastVisStep_ = simStep_;
	lastVisScalarMode_ = scalarMode;
	lastVisParticleMask_ = particleMask;
	// a visibility change may have arrived before this snapshot consumed its request
	if (!includeAllParticles && visualParticleMask_.load(std::memory_order_acquire) != particleMask)
	{
		visualSnapMailbox_->RequestSnapshot();
	}
	return true;
}

void SimulationMgr::HandleBenchMemStatSnapshots() const
{
	// create bench snapshot and push to its mailbox
	auto benchSnap = std::make_shared<Sync::BenchSnapshot>();
	benchmarksTracker_.PackBenchSnapshot(*benchSnap);
	benchSnapMailbox_->StoreSnapshot(std::move(benchSnap));

	// create mem snapshot and push to its mailbox
	auto memSnap = std::make_shared<Sync::MemSnapshot>();
	memoryTracker_.PackMemSnapshot(*memSnap);
	memSnapMailbox_->StoreSnapshot(std::move(memSnap));

	// create stat snapshot and push to its mailbox
	auto statSnap = std::make_shared<Sync::StatSnapshot>();
	statisticsTracker_.PackStatSnapshot(*statSnap);
	statSnapMailbox_->StoreSnapshot(std::move(statSnap));
}

void SimulationMgr::AppendInitialMemoryMetricsLog() const
{
	if (!loggingMgr_.IsActive()) { return; }

	Mem::MemRecordStep record{};
	record.simStep = simStep_;
	record.simTime = simTime_;

	simBackend_->FillMemoryRecord(record);
	memoryTracker_.SampleProcessMemory();
	record.process = memoryTracker_.GetCurrentRecord().process;
	record.device = memoryTracker_.GetCurrentRecord().device;
	loggingMgr_.AppendMemoryRecord(record);
}

void SimulationMgr::AppendLatestMemoryMetricsLog() const
{
	if (!loggingMgr_.IsActive()) { return; }

	loggingMgr_.AppendMemoryRecord(memoryTracker_.GetCurrentRecord());
}

void SimulationMgr::AppendLatestBenchmarkMetricsLog() const
{
	if (!loggingMgr_.IsActive()) { return; }

	loggingMgr_.AppendBenchmarkRecord(benchmarksTracker_.GetCurrentRecord());
}

void SimulationMgr::FinalizeBenchMemStatMeasurements() const
{
	// record statistics, archive data and update aggregates
	simBackend_->FillStatisticsRecord(statisticsTracker_.GetCurrentRecord(), simParams_);
	statisticsTracker_.EndStep(simStep_, simTime_);

	// record memory usage, archive data and update aggregates
	simBackend_->FillMemoryRecord(memoryTracker_.GetCurrentRecord());
	memoryTracker_.EndStep(simStep_, simTime_);

	// ending point of total benchmark, archive data and update aggregates
	benchmarksTracker_.StopBench(Bench::BenchOp::Total);
	benchmarksTracker_.EndStep(simStep_, simTime_);
}

void SimulationMgr::BeginRunLogging(const SimParams& params)
{
	// start a fresh export session for this run
	exportSnapshotDir_.clear();

	loggingMgr_.BeginRun(configProfile_, paramsFile_, scenariosFile_, params,
		simParamsMgr_.CliOverrides());
}

void SimulationMgr::EndRunLogging()
{
	loggingMgr_.EndRun();
}

void SimulationMgr::DisplayPerformanceStats(
	const SimParams& params) const
{
	DiagnosticsMgr::PrintLatestPerformanceSummaryAndProgress(
		benchmarksTracker_, curBenchOps_, params.timeUnit,
		params.progressReportMode, simStep_, params.maxSteps,
		memoryTracker_.GetCurrentRecord().process.N_residentBytes,
		memoryTracker_.GetCurrentRecord().process.N_peakResidentBytes,
		memoryTracker_.GetCurrentRecord().device.N_residentBytes,
		memoryTracker_.GetCurrentRecord().device.N_peakResidentBytes);
}

void SimulationMgr::DisplayMemoryUsageStats(
	const SimParams& params, const bool force) const
{
	if (!force && simStep_ != 1) { return; }

	DiagnosticsMgr::PrintMemorySummary_Standalone(
		memoryTracker_, params.byteUnit, params.backendType);
}
