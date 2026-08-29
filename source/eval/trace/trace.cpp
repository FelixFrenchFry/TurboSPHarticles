// trace.cpp

// external
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <utility>

// internal
#include "trace.h"



namespace Trace
{
	bool TraceRecorder::BeginRun(
		const std::filesystem::path& runDir,
		const int mpiRank)
	{
		EndRun();
		mpiRank_ = mpiRank;

		if (runDir.empty())
		{
			SPDLOG_ERROR(
				"[rank {}] Failed to start trace recorder: run directory is empty.",
				mpiRank_);
			return false;
		}

		tracePath_ = runDir / "trace.log";
		try
		{
			auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(tracePath_.string(), true);
			auto logger = std::make_shared<spdlog::logger>("trace", std::move(sink));
			logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] %v");
			logger->set_level(spdlog::level::info);
			logger->flush_on(spdlog::level::info);

			logger_ = std::move(logger);
			sequence_ = 0;
			simStep_ = 0;
			simTime_ = 0.0;
			stepStart_ = Clock::now();
			active_ = true;
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR("[rank {}] Failed to open trace file '{}': {}.",
				mpiRank_, tracePath_.string(), ex.what());
			Disable();
			return false;
		}
		catch (...)
		{
			SPDLOG_ERROR("[rank {}] Failed to open trace file '{}': unknown error.",
				mpiRank_, tracePath_.string());
			Disable();
			return false;
		}

		return true;
	}

	void TraceRecorder::EndRun()
	{
		if (logger_)
		{
			try
			{
				logger_->flush();
			}
			catch (const std::exception& ex)
			{
				SPDLOG_ERROR("[rank {}] Failed to flush trace file '{}': {}.",
					mpiRank_, tracePath_.string(), ex.what());
			}
			catch (...)
			{
				SPDLOG_ERROR("[rank {}] Failed to flush trace file '{}': unknown error.",
					mpiRank_, tracePath_.string());
			}
		}

		Disable();
	}

	void TraceRecorder::BeginStep(
		const uint64_t simStep,
		const double simTime)
	{
		if (!active_ || !logger_) { return; }

		simStep_ = simStep;
		simTime_ = simTime;
		stepStart_ = Clock::now();
		WriteMarker(TraceStage::StepBegin);
	}

	void TraceRecorder::Mark(const TraceStage stage)
	{
		WriteMarker(stage);
	}

	void TraceRecorder::Disable()
	{
		logger_.reset();
		tracePath_.clear();
		mpiRank_ = 0;
		sequence_ = 0;
		simStep_ = 0;
		simTime_ = 0.0;
		stepStart_ = {};
		active_ = false;
	}

	void TraceRecorder::WriteMarker(const TraceStage stage)
	{
		if (!active_ || !logger_) { return; }

		const uint8_t stageIdx = GetIdx(stage);
		if (stageIdx >= GetIdx(TraceStage::Count))
		{
			SPDLOG_ERROR(
				"[rank {}] Failed to write trace marker: invalid stage index {}.",
				mpiRank_, static_cast<unsigned int>(stageIdx));
			Disable();
			return;
		}

		const uint64_t sequence = ++sequence_;
		const auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
			Clock::now() - stepStart_).count();

		try
		{
			logger_->info(
				"seq={} rank={} step={} simTime={:.17g} elapsedUs={} stage={}",
				sequence,
				mpiRank_,
				simStep_,
				simTime_,
				elapsedUs,
				Definitions[stageIdx].alias);
			logger_->flush();
		}
		catch (const std::exception& ex)
		{
			SPDLOG_ERROR("[rank {}] Failed to write trace file '{}': {}.",
				mpiRank_, tracePath_.string(), ex.what());
			Disable();
		}
		catch (...)
		{
			SPDLOG_ERROR("[rank {}] Failed to write trace file '{}': unknown error.",
				mpiRank_, tracePath_.string());
			Disable();
		}
	}

} // namespace Trace
