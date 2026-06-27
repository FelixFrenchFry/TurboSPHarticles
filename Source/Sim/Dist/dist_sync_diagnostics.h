// dist_sync_diagnostics.h

#pragma once

// external
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

// internal
#include "Source/Base/helpers.h"



namespace Dist
{
	enum class DistSyncTimingOp : uint8_t
	{
		Pull,
		Reorder,
		Push,
		ActiveIdx,

		Count
	};

	constexpr uint8_t GetIdx(const DistSyncTimingOp op)
	{
		return static_cast<uint8_t>(op);
	}

	// temporary, standalone timing report for profiling distributed backend sync work
	struct DistSyncTimingReport
	{
		int rank = 0;
		uint64_t sampleCount = 0;
		std::array<uint64_t, GetIdx(DistSyncTimingOp::Count)> totalNs{};
	};

	// temporary chrono scope
	class DistSyncTimingScope final
	{
	public:

		DistSyncTimingScope(
			DistSyncTimingReport* report,
			DistSyncTimingOp op);
		~DistSyncTimingScope();

		DistSyncTimingScope(const DistSyncTimingScope&) = delete;
		DistSyncTimingScope& operator=(const DistSyncTimingScope&) = delete;

	private:

		using Clock = std::chrono::steady_clock;

		DistSyncTimingReport* report_ = nullptr;
		DistSyncTimingOp op_ = DistSyncTimingOp::Count;
		Clock::time_point t0_{};
		bool active_ = false;
	};

	void BeginDistSyncTimingReport(
		DistSyncTimingReport& report,
		int rank);

	std::vector<std::string> BuildDistSyncTimingLines(
		const std::vector<DistSyncTimingReport>& reports,
		Bench::TimeUnit unit);

} // namespace Dist
