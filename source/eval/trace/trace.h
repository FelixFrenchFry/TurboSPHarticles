// trace.h
//
// TODO: comment

#pragma once

// external
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>

// internal
#include "stages.h"



namespace spdlog
{
	class logger;
}

namespace Trace
{
	class TraceRecorder final
	{
	public:

		bool BeginRun(
			const std::filesystem::path& runDir,
			int mpiRank = 0);
		void EndRun();

		bool IsActive() const { return active_; }
		void BeginStep(uint64_t simStep, double simTime);
		void Mark(TraceStage stage);

	private:

		using Clock = std::chrono::steady_clock;

		std::shared_ptr<spdlog::logger> logger_{};
		std::filesystem::path tracePath_{};

		int mpiRank_ = 0;
		uint64_t sequence_ = 0;
		uint64_t simStep_ = 0;
		double simTime_ = 0.0;
		Clock::time_point stepStart_{};
		bool active_ = false;

		void Disable();
		void WriteMarker(TraceStage stage);
	};

} // namespace Trace
