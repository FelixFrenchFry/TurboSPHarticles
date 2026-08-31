// trace.h
//
// TODO: comment

#pragma once

// external
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>

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
			int mpiRank = 0,
			bool enableFluidSynchronization = false);
		void EndRun();

		bool IsActive() const { return active_; }
		bool IsFluidSynchronizationActive() const
		{
			return active_ && fluidSynchronizationEnabled_;
		}
		void BeginStep(uint64_t simStep, double simTime);
		void Mark(TraceStage stage);
		void MarkFluidSynchronization(
			std::string_view phase,
			std::string_view event,
			uint32_t iteration,
			uint64_t epoch,
			uint32_t N_peers,
			uint64_t N_sendBytes,
			uint64_t N_recvBytes);

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
		bool fluidSynchronizationEnabled_ = false;

		void Disable();
		void WriteMarker(TraceStage stage);
	};

} // namespace Trace
