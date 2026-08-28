// simulation.cpp

// external
#include <chrono>
#include <filesystem>
#include <iostream>
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

void SimulationMgr::Init()
{
	const auto initializationStart = std::chrono::steady_clock::now();

	// initial snapshot
	SyncParameters(true);

	// default sim state (empty)
	universalSimState_ = Sync::UniversalSimState{};

	// TODO: load initial sim state from a file?

	SPDLOG_INFO("Init simulation state...");
	SelectSimScenario(simParams_.scenarioType, configProfile_);
	SelectSimBackend(simParams_.backendType);

	SPDLOG_INFO("Init backend data structures...");
	ReBuildInitialSimState();

	// updated snapshot
	initialized_.store(true, std::memory_order_release);
	SyncParameters(true);

	const double initializationTimeS = std::chrono::duration<double>(
		std::chrono::steady_clock::now() - initializationStart).count();
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

void SimulationMgr::SelectSimBackend(
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
	simBackend_ = MakeSimBackend(backendType, simParamsMgr_, benchmarksTracker_, simParams_);
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

void SimulationMgr::ReBuildInitialSimState()
{
	if (!simScenario_)
	{
		throw std::runtime_error(
			"Failed to re-build initial simulation state without a selected scenario.");
	}

	// stop simulation
	simParamsMgr_.Set_SimRunning(false);
	simParamsMgr_.Set_SimRunNextStep(false);

	// lock state during re-build
	std::unique_lock lock(stateMutex_);

	// reset metadata
	simStep_ = 0;
	simTime_ = 0.0;

	// reset scenario and all its data
	universalSimState_ = {};
	universalSimState_.step = 0;
	universalSimState_.time = 0.0;

	SyncParameters();

	// re-build initial scenario state by transferring into universal state
	simScenario_->Build(universalSimState_, simParams_);
	universalSimState_.particlesF.ensureParticleIdsPresent();
	universalSimState_.particlesB.ensureParticleIdsPresent();
	universalSimState_.particlesR.ensureParticleIdsPresent();
	universalSimState_.particlesB.ensureBoundaryCoefficientsPresent(
		simParams_.massB, simParams_.restDensity);
	universalSimState_.particlesR.ensureRigidCoefficientsPresent(
		simParams_.massB, simParams_.restDensity);
	universalSimState_.ValidateRigidAttachments();

	// compact staged particle vectors once after scenario construction
	universalSimState_.ShrinkToFitAll();

	if (simBackend_)
	{
		ApplySimDomainBoundsToBackend();

		// transfer universal state into backend-specific data structures
		simBackend_->PullFromUniversalSimState(universalSimState_, simParams_);
		return;
	}

	lock.unlock();
	SelectSimBackend(simParams_.backendType);
}

void SimulationMgr::ApplySimDomainBoundsToBackend()
{
	if (!simScenario_ || !simBackend_) { return; }

	glm::vec3 domainMin{};
	glm::vec3 domainMax{};
	if (!Scene::ComputeSimDomainBounds(
		simScenario_->GetBoundaryBoxes(), simParams_, domainMin, domainMax))
	{
		return;
	}

	simBackend_->SetSimDomainBounds(domainMin, domainMax);
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
		SelectSimBackend(simParams_.backendType);
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
				RunSim_Async(stopToken);
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
		SelectSimBackend(simParams_.backendType);
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

	// launch threads
	RunSim_Blocking();
}

void SimulationMgr::RunSim_Async(const std::stop_token& stopToken)
{
	printf("\n");
	SPDLOG_INFO("Simulation started running...");

	// --- [ initialize before entering the simulation loop ] ----------------------------------------

	// local copy for thread safety
	const auto paramsInit = simParams_;
	BeginRunLogging(paramsInit);
	DiagnosticsMgr::ResetProgress();

	// initial neighbor search for correct statistics
	simBackend_->ExecuteNeighborSearch(simParams_);

	PublishVisualSnapshot();

	// vis throttle starting point
	lastVisWall_ = std::chrono::steady_clock::now();

	// pack and serialize export snapshot of initial state
	if (paramsInit.exportEnabled)
	{
		WriteExportSnapshot(paramsInit);
	}

	AppendInitialMemoryMetricsLog(paramsInit);

	// --- [ enter the simulation loop ] -------------------------------------------------------------

	while (!stopToken.stop_requested() &&
		   !stopRequested_.load(std::memory_order_acquire))
	{
		SyncParameters(true);

		// local copy for thread safety
		auto params = simParams_;

		// the run length is fixed at launch, so progress reports match the step limit below
		params.maxSteps = paramsInit.maxSteps;
		const bool doSimStep = (params.simRunning || params.simRunNextStep);

		// explicit maximum number of steps reached
		if (simStep_ >= static_cast<uint64_t>(paramsInit.maxSteps) &&
			paramsInit.maxSteps > 0) { break; }

		// small sleep to avoid busy spin when nothing to do yet
		if (!doSimStep)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		// execute one simulation step
		if (doSimStep)
		{
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
		}

		// TODO: adjust update frequency with throttle
		// TODO: add upper bound via a "new frame" flag from the visualizer
		// TODO: add lower bound via ...
		if (doSimStep)
		{
			const auto now = std::chrono::steady_clock::now();

			// compute minimum interval from the target rate
			const auto minInterval = std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::duration<double>(1.0 / fmaxf(1e-7f, visMaxHz_)));

			if (now - lastVisWall_ >= minInterval)
			{
				benchmarksTracker_.StartBench(Bench::BenchOp::Visualization);

				PublishVisualSnapshot();

				benchmarksTracker_.StopBench(Bench::BenchOp::Visualization);

				lastVisWall_ = now;
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
		if (doSimStep) { FinalizeBenchMemStatMeasurements(); }
		if (doSimStep) { AppendLatestMemoryMetricsLog(); }
		if (doSimStep) { AppendLatestBenchmarkMetricsLog(); }

		// pack and send benchmark and statistics snapshots
		if (doSimStep) { HandleBenchMemStatSnapshots(); }

		SyncParameters();

		DisplayMemoryUsageStats(params);
		DisplayPerformanceStats(params);

		// TODO: check if scenario or backend need to be switched?
	}

	// --- [ finalize after exiting the simulation loop ] --------------------------------------------

	// local copy for thread safety
	const auto paramsFinal = simParams_;

	FinalizeRunState(paramsFinal);

	DisplayMemoryUsageStats(paramsFinal, true);
	const bool processMemoryChanged = memoryTracker_.SampleProcessMemory();
	if (processMemoryChanged) { AppendLatestMemoryMetricsLog(); }

	EndRunLogging();

	printf("\n");
	SPDLOG_INFO("Simulation stopped running");
	DiagnosticsMgr::PrintFinalProgressPeakRam(
		memoryTracker_.GetCurrentRecord().process.N_residentBytes,
		memoryTracker_.GetCurrentRecord().process.N_peakResidentBytes);
}

void SimulationMgr::RunSim_Blocking()
{
	printf("\n");
	SPDLOG_INFO("Simulation started running...");

	// --- [ initialize before entering the simulation loop ] ----------------------------------------

	// local copy for thread safety
	const auto paramsInit = simParams_;
	BeginRunLogging(paramsInit);
	DiagnosticsMgr::ResetProgress();

	// initial neighbor search for correct statistics
	simBackend_->ExecuteNeighborSearch(simParams_);

	// pack and serialize export snapshot of initial state
	if (paramsInit.exportEnabled)
	{
		WriteExportSnapshot(paramsInit);
	}

	AppendInitialMemoryMetricsLog(paramsInit);

	// --- [ enter the simulation loop ] -------------------------------------------------------------

	while (!stopRequested_.load(std::memory_order_acquire))
	{
		SyncParameters(true);

		// local copy for thread safety
		auto params = simParams_;

		// the run length is fixed at launch, so progress reports match the step limit below
		params.maxSteps = paramsInit.maxSteps;
		const bool doSimStep = (params.simRunning || params.simRunNextStep);

		// explicit maximum number of steps reached
		if (simStep_ >= static_cast<uint64_t>(paramsInit.maxSteps) &&
			paramsInit.maxSteps > 0) { break; }

		// small sleep to avoid busy spin when nothing to do yet
		if (!doSimStep)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		// execute one simulation step
		if (doSimStep)
		{
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
		if (doSimStep) { FinalizeBenchMemStatMeasurements(); }
		if (doSimStep) { AppendLatestMemoryMetricsLog(); }
		if (doSimStep) { AppendLatestBenchmarkMetricsLog(); }

		// pack and send benchmark and statistics snapshots
		if (doSimStep) { HandleBenchMemStatSnapshots(); }

		SyncParameters();

		DisplayMemoryUsageStats(params);
		DisplayPerformanceStats(params);

		// TODO: check if scenario or backend need to be switched?
	}

	// --- [ finalize after exiting the simulation loop ] --------------------------------------------

	// local copy for thread safety
	const auto paramsFinal = simParams_;

	FinalizeRunState(paramsFinal);

	DisplayMemoryUsageStats(paramsFinal, true);
	const bool processMemoryChanged = memoryTracker_.SampleProcessMemory();
	if (processMemoryChanged) { AppendLatestMemoryMetricsLog(); }

	running_.store(false, std::memory_order_release);
	EndRunLogging();

	printf("\n");
	SPDLOG_INFO("Simulation stopped running");
	DiagnosticsMgr::PrintFinalProgressPeakRam(
		memoryTracker_.GetCurrentRecord().process.N_residentBytes,
		memoryTracker_.GetCurrentRecord().process.N_peakResidentBytes);
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

void SimulationMgr::WriteExportSnapshot(const SimParams& params) const
{
	// fill export snapshot
	auto& expSnap = exportSnapScratch_;
	simBackend_->PackExportSnapshot(expSnap, simParams_);

	// in the non-distributed path, root rank owns every fluid particle
	expSnap.rankF.assign(expSnap.N_F, 0);

	// write export snapshot to files
	serializationMgr_.SaveExportSnapshot(expSnap, params.exportDir);
	serializationMgr_.SaveExportSnapshotXdmfViews(expSnap, params.exportDir);
}

void SimulationMgr::FinalizeRunState(const SimParams& params)
{
	if (!params.saveFinalState)
	{
		simBackend_.reset();
		return;
	}

	{
		// transfer last sim state into the universal state to save it
		std::scoped_lock lock(stateMutex_);
		simBackend_->PushToUniversalSimState(universalSimState_, true);
		simBackend_.reset();
	}

	serializationMgr_.SaveUniversalSimState(
		universalSimState_,
		(std::filesystem::path(params.exportDir) /
			"simulation_state_final.bin").string());
}

void SimulationMgr::PublishVisualSnapshot()
{
	// create visual snapshot and push pack misc data that is not handled by the backend
	// (if CUDA path is taken, the particle data buffers are not filled)
	auto visSnap = std::make_shared<Sync::VisualSnapshot>();
	visSnap->simStep = simStep_;
	visSnap->boxesB = simScenario_->GetBoundaryBoxes();
	visSnap->scalarModeF = visualScalarMode_.load(std::memory_order_acquire);

	// optionally pack particle data into CUDA-device buffers for OpenGL without going through the host
#ifdef TURBOSPHARTICLES_WITH_CUDA
	auto& visualFrameMailboxCUDA = visualSnapMailbox_->GetFrameMailboxCUDA();
	const auto result = simBackend_->TryPackVisualSnapshotCUDA(visualFrameMailboxCUDA, visSnap);
	if (result != Sync::VisualPackResultCUDA::Unsupported) { return; } // skip host snapshot if successful
#endif

	// fallback to host snapshot packing if CUDA path is unavailable or unsupported
	simBackend_->PackVisualSnapshot(*visSnap);
	visualSnapMailbox_->StoreSnapshot(std::move(visSnap));
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

void SimulationMgr::AppendInitialMemoryMetricsLog(const SimParams& params) const
{
	if (!loggingMgr_.IsActive()) { return; }

	Mem::MemRecordStep record{};
	record.simStep = simStep_;
	record.simTime = simTime_;

	simBackend_->FillMemoryRecord(record, params);
	memoryTracker_.SampleProcessMemory();
	record.process = memoryTracker_.GetCurrentRecord().process;
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
	simBackend_->FillMemoryRecord(memoryTracker_.GetCurrentRecord(), simParams_);
	memoryTracker_.EndStep(simStep_, simTime_);

	// ending point of total benchmark, archive data and update aggregates
	benchmarksTracker_.StopBench(Bench::BenchOp::Total);
	benchmarksTracker_.EndStep(simStep_, simTime_);
}

void SimulationMgr::BeginRunLogging(const SimParams& params)
{
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
		memoryTracker_.GetCurrentRecord().process.N_peakResidentBytes);
}

void SimulationMgr::DisplayMemoryUsageStats(
	const SimParams& params, const bool force) const
{
	if (!force && simStep_ != 1) { return; }

	DiagnosticsMgr::PrintMemorySummary_Standalone(
		memoryTracker_, params.byteUnit, params.backendType);
}
