// benchmarks.cpp

// external
#include <algorithm>
#include <limits>

// internal
#include "benchmarks.h"



using namespace Bench;

BenchmarksMgr::BenchmarksMgr(const Sim::SimParams& simParams)
	: maxRecCount_(simParams.maxBenchRecs)
	, curRecCount_(0u)
{
	recLatest_ = {};
	recLatest_.time.fill(0ull);
	ClearAllTime();
}

void BenchmarksMgr::BeginStep()
{
	curStepNs_.fill(0);
	curStepOcc_.fill(0);
}

void BenchmarksMgr::EndStep(const uint32_t simStep, const double simTime)
{
	// fill latest record
	recLatest_.simStep = simStep;
	recLatest_.simTime = simTime;
	recLatest_.time = curStepNs_;

	// push latest record to history and check excess
	recHistory_.push_back(recLatest_);
	if (recHistory_.size() > maxRecCount_)
	{
		recHistory_.pop_front();
	}

	if (curRecCount_ == 0) // first entry
	{
		benchSum_allTime_ = recLatest_.time;
		benchMax_allTime_ = recLatest_.time;

		for (uint8_t i = 0; i < GetIdx(BenchOp::Count); i++)
		{
			// initialize min only for operations that actually occurred
			benchMin_allTime_[i] = curStepOcc_[i] ? recLatest_.time[i]
								 : std::numeric_limits<uint64_t>::max();

			// convert from 8 to 64 bits, because simple array copy is not allowed
			numOcc_allTime_[i] = static_cast<uint64_t>(curStepOcc_[i]);
		}
	}
	else
	{
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
	}

	curRecCount_ += 1;
}

void BenchmarksMgr::StartBench(const std::initializer_list<BenchOp> ops)
{
	if (ops.size() == 0) { return; }

	const BenchOp leader = *ops.begin();
	const uint8_t i = GetIdx(leader);

	t0_single_[i] = Clock::now();
}

void BenchmarksMgr::StopBench(const std::initializer_list<BenchOp> ops)
{
	if (ops.size() == 0) { return; }

	const BenchOp leader = *ops.begin();
	const uint8_t i = GetIdx(leader);

	const auto t1 = Clock::now();
	const uint64_t ns = static_cast<uint64_t>(std::chrono::duration_cast
		<std::chrono::nanoseconds>(t1 - t0_single_[i]).count());

	// credit to all ops in the group
	AddGroupTimingSample(ops, ns);
}

void BenchmarksMgr::AddTimingSample(const BenchOp op, const uint64_t ns)
{
	const uint8_t i = GetIdx(op);

	curStepNs_[i] += ns;
	curStepOcc_[i] = 1;
}

void BenchmarksMgr::AddGroupTimingSample(const std::initializer_list<BenchOp> ops, const uint64_t ns)
{
	for (auto op : ops)
	{
		const uint8_t i = GetIdx(op);
		curStepNs_[i] += ns;
		curStepOcc_[i] = 1;
	}
}

void BenchmarksMgr::PackBenchSnapshot(Sync::BenchSnapshot& snap) const
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

uint64_t BenchmarksMgr::GetBench_Latest(const BenchOp op) const
{
	if (curRecCount_ == 0) { return 0; }

	return recLatest_.time[GetIdx(op)];
}

uint64_t BenchmarksMgr::GetBench_Avg_OverAll(const BenchOp op) const
{
	const uint64_t sum = benchSum_allTime_[GetIdx(op)];
	if (curRecCount_ == 0 || sum == 0) { return 0; }

	// integer division with ceil
	return ((sum - 1) / curRecCount_) + 1;
}

uint64_t BenchmarksMgr::GetBench_Avg_WhenUsed(const BenchOp op) const
{
	const uint64_t sum = benchSum_allTime_[GetIdx(op)];
	const uint64_t numOcc = numOcc_allTime_[GetIdx(op)];
	if (curRecCount_ == 0 || sum == 0 || numOcc == 0) { return 0; }

	// integer division with ceil
	return ((sum - 1) / numOcc) + 1;
}

uint64_t BenchmarksMgr::GetBench_Min(const BenchOp op) const
{
	if (curRecCount_ == 0) { return 0; }

	return benchMin_allTime_[GetIdx(op)];
}

uint64_t BenchmarksMgr::GetBench_Max(const BenchOp op) const
{
	if (curRecCount_ == 0) { return 0; }

	return benchMax_allTime_[GetIdx(op)];
}

uint64_t BenchmarksMgr::GetBench_NumOcc(const BenchOp op) const
{
	if (curRecCount_ == 0) { return 0; }

	return numOcc_allTime_[GetIdx(op)];
}

void BenchmarksMgr::ClearRecordHistory()
{
	recHistory_.clear();
	curRecCount_ = 0;
	recLatest_ = {};
	recLatest_.time.fill(0);
	curStepNs_.fill(0);
	curStepOcc_.fill(0);
	ClearAllTime();
}

void BenchmarksMgr::ClearAllTime()
{
	benchSum_allTime_.fill(0);
	benchMin_allTime_.fill(std::numeric_limits<uint64_t>::max());
	benchMax_allTime_.fill(0);
	numOcc_allTime_.fill(0);
}
