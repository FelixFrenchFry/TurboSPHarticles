// benchmarks.cpp

// external
#include <algorithm>
#include <limits>

// internal
#include "benchmarks.h"



using namespace Bench;

BenchmarksTracker::BenchmarksTracker(const Sim::SimParams& simParams)
	: N_maxRecords_(simParams.maxBenchRecs)
	, N_records_(0u)
{
	recLatest_ = {};
	std::ranges::fill(recLatest_.time, 0ull);
	ClearAllTime();
}

void BenchmarksTracker::BeginStep()
{
	std::ranges::fill(curStepNs_, 0);
	std::ranges::fill(curStepOcc_, 0);
}

void BenchmarksTracker::EndStep(const uint64_t simStep, const double simTime)
{
	// fill latest record
	recLatest_.simStep = simStep;
	recLatest_.simTime = simTime;
	recLatest_.time = curStepNs_;

	// push latest record to history and check excess
	recHistory_.push_back(recLatest_);
	if (recHistory_.size() > N_maxRecords_)
	{
		recHistory_.pop_front();
	}

	for (uint8_t i = 0; i < GetIdx(BenchOp::Count); i++)
	{
		// update all-time sum
		benchSum_allTime_[i] += recLatest_.time[i];

		// update all-time min (only if the operation occured)
		if (curStepOcc_[i] && recLatest_.time[i] < benchMin_allTime_[i])
		{
			benchMin_allTime_[i] = recLatest_.time[i];
		}

		// update all-time max
		benchMax_allTime_[i] = std::max(recLatest_.time[i],
										benchMax_allTime_[i]);

		// update occurrence count
		numOcc_allTime_[i] += curStepOcc_[i] ? 1 : 0;
	}

	N_records_ += 1;
}

void BenchmarksTracker::StartBench(const BenchOp op)
{
	const uint8_t i = GetIdx(op);

	t0_single_[i] = Clock::now();
	activeTimers_[i] = 1;
}

void BenchmarksTracker::StopBench(const BenchOp op)
{
	const uint8_t i = GetIdx(op);

	const auto t1 = Clock::now();
	const uint64_t ns = static_cast<uint64_t>(std::chrono::duration_cast
		<std::chrono::nanoseconds>(t1 - t0_single_[i]).count());

	AddTimingSample(op, ns);
	activeTimers_[i] = 0;
}

void BenchmarksTracker::PauseActiveTimers()
{
	if (timersPaused_) { return; }

	pauseStart_ = Clock::now();
	timersPaused_ = true;
}

void BenchmarksTracker::ResumeActiveTimers()
{
	if (!timersPaused_) { return; }

	const auto pauseDuration = Clock::now() - pauseStart_;
	for (uint8_t i = 0; i < GetIdx(BenchOp::Count); i++)
	{
		if (activeTimers_[i]) { t0_single_[i] += pauseDuration; }
	}

	timersPaused_ = false;
}

void BenchmarksTracker::AddTimingSample(const BenchOp op, const uint64_t ns)
{
	const uint8_t i = GetIdx(op);

	curStepNs_[i] += ns;
	curStepOcc_[i] = 1;
}

void BenchmarksTracker::PackBenchSnapshot(Sync::BenchSnapshot& snap) const
{
	// metadata
	snap.simStep = recLatest_.simStep;
	snap.simTime = recLatest_.simTime;

	// fill in latest, average over all, average when used, min, max benches
	for (uint8_t i = 0; i < GetIdx(BenchOp::Count); i++)
	{
		snap.latest[i] = GetBench_Latest(static_cast<BenchOp>(i));
		snap.avgAll[i] = GetBench_Avg_OverAll(static_cast<BenchOp>(i));
		snap.avgUsed[i] = GetBench_Avg_WhenUsed(static_cast<BenchOp>(i));
		snap.min[i] = GetBench_Min(static_cast<BenchOp>(i));
		snap.max[i] = GetBench_Max(static_cast<BenchOp>(i));
	}
}

uint64_t BenchmarksTracker::GetBench_Latest(const BenchOp op) const
{
	if (N_records_ == 0) { return 0; }

	return recLatest_.time[GetIdx(op)];
}

uint64_t BenchmarksTracker::GetBench_Avg_OverAll(const BenchOp op) const
{
	const uint64_t sum = benchSum_allTime_[GetIdx(op)];
	if (N_records_ == 0 || sum == 0) { return 0; }

	// integer division with ceil
	return ((sum - 1) / N_records_) + 1;
}

uint64_t BenchmarksTracker::GetBench_Avg_WhenUsed(const BenchOp op) const
{
	const uint64_t sum = benchSum_allTime_[GetIdx(op)];
	const uint64_t N_occurrences = numOcc_allTime_[GetIdx(op)];
	if (N_records_ == 0 || sum == 0 || N_occurrences == 0) { return 0; }

	// integer division with ceil
	return ((sum - 1) / N_occurrences) + 1;
}

uint64_t BenchmarksTracker::GetBench_Min(const BenchOp op) const
{
	if (N_records_ == 0) { return 0; }

	const uint64_t minNs = benchMin_allTime_[GetIdx(op)];
	return minNs == std::numeric_limits<uint64_t>::max() ? 0 : minNs;
}

uint64_t BenchmarksTracker::GetBench_Max(const BenchOp op) const
{
	if (N_records_ == 0) { return 0; }

	return benchMax_allTime_[GetIdx(op)];
}

void BenchmarksTracker::ClearRecordHistory()
{
	recHistory_.clear();
	N_records_ = 0;
	recLatest_ = {};
	std::ranges::fill(recLatest_.time, 0);
	std::ranges::fill(curStepNs_, 0);
	std::ranges::fill(curStepOcc_, 0);
	ClearAllTime();
}

void BenchmarksTracker::ClearAllTime()
{
	std::ranges::fill(benchSum_allTime_, 0);
	std::ranges::fill(benchMin_allTime_, std::numeric_limits<uint64_t>::max());
	std::ranges::fill(benchMax_allTime_, 0);
	std::ranges::fill(numOcc_allTime_, 0);
}
