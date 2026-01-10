// benchmarks.h
//
// TODO: comment

#pragma once

// external
#include <chrono>
#include <deque>
#include <initializer_list>

// internal
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Sync/snapshots.h"
#include "operations.h"



namespace Bench
{
	class BenchmarksTracker
	{
	public: // -----------------------------------------------------------------

		BenchmarksTracker(const Sim::SimParams& simParams);

		// resets the latest bench data before pulling in a new record
		void BeginStep();

		// pushes the record to history and aggregates step's bench data
		void EndStep(uint32_t simStep, double simTime);

		// timing start/stop for groups of single or multiple operations
		void StartBench(std::initializer_list<Bench::BenchOp> ops);
		void StopBench(std::initializer_list<Bench::BenchOp> ops);

		// accumulate one timing sample into the current step's bench data
		void AddTimingSample(BenchOp op, uint64_t ns);

		void AddGroupTimingSample(std::initializer_list<BenchOp> ops, uint64_t ns);

		void PackBenchSnapshot(Sync::BenchSnapshot& snap) const;

		// per-operation queries
		uint64_t GetBench_Latest(BenchOp op) const;
		uint64_t GetBench_Avg_OverAll(BenchOp op) const;
		uint64_t GetBench_Avg_WhenUsed(BenchOp op) const;
		uint64_t GetBench_Min(BenchOp op) const;
		uint64_t GetBench_Max(BenchOp op) const;
		uint64_t GetBench_NumOcc(BenchOp op) const;

		// helpers
		void ClearRecordHistory();
		BenchRecordStep& GetCurrentRecord() { return recLatest_; }
		const BenchRecordStep& GetCurrentRecord() const { return recLatest_; }
		const std::deque<BenchRecordStep>& GetHistory() const { return recHistory_; }

	private: // ----------------------------------------------------------------

		// start timestamps
		using Clock = std::chrono::steady_clock;
		std::array<Clock::time_point, GetIdx(BenchOp::Count)> t0_single_{};

		// max and current number of steps for which timings are kept
		uint32_t maxRecCount_;
		uint32_t curRecCount_;

		// per-step per-operation timing (in nanoseconds) and occurrence flags
		std::array<uint64_t, GetIdx(BenchOp::Count)> curStepNs_{};
		std::array<uint8_t, GetIdx(BenchOp::Count)> curStepOcc_{};

		BenchRecordStep recLatest_;

		// rolling buffer for the timings for the last maxCount_ steps
		std::deque<BenchRecordStep> recHistory_;

		// all-time per-operation aggregates (in nanoseconds)
		std::array<uint64_t, GetIdx(BenchOp::Count)> benchSum_allTime_{};
		std::array<uint64_t, GetIdx(BenchOp::Count)> benchMin_allTime_{};
		std::array<uint64_t, GetIdx(BenchOp::Count)> benchMax_allTime_{};
		std::array<uint64_t, GetIdx(BenchOp::Count)> numOcc_allTime_{}; // number of occurrences

		// helpers
		void ClearAllTime();
	};

} // namespace Bench
