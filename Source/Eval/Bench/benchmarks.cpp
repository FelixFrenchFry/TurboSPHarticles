// benchmarks.cpp

// external
#include <algorithm>
#include <limits>
#include <omp.h>
#include <spdlog/spdlog.h>

// internal
#include "benchmarks.h"



using namespace Bench;

BenchmarksMgr::BenchmarksMgr(const Sim::SimParams& simParams)
	: maxRecCount_(simParams.maxBenchRecCount)
	, curRecCount_(0)
{
	recLatest_ = {};
	recLatest_.time.fill(0);
	ClearAllTime();
}

void BenchmarksMgr::BeginStep()
{
	curStepNs_.fill(0);
	curStepOcc_.fill(0);
}

void BenchmarksMgr::EndStep(int64_t simStep, double simTime)
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

		for (uint16_t i = 0; i < GetIdx(BenchOp::Count); i++)
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
		for (uint16_t i = 0; i < GetIdx(BenchOp::Count); i++)
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

void BenchmarksMgr::StartBench(std::initializer_list<Bench::BenchOp> ops)
{
	if (ops.size() == 0) { return; }

	const BenchOp leader = *ops.begin();
	const uint16_t i = GetIdx(leader);

	if (!omp_in_parallel()) // NOT inside an OMP parallel region
	{
		t0_single_[i] = Clock::now();
		return;
	}

	// inside an OMP parallel region, all threads must reach the barrier
	#pragma omp barrier
	#pragma omp single
	{
		t0_team_[i] = omp_get_wtime();
	}
	#pragma omp barrier
}

void BenchmarksMgr::StopBench(std::initializer_list<Bench::BenchOp> ops)
{
	if (ops.size() == 0) { return; }

	const BenchOp leader = *ops.begin();
	const uint16_t i = GetIdx(leader);

	if (!omp_in_parallel()) // NOT inside an OMP parallel region
	{
		const auto t1 = Clock::now();
		const uint64_t ns = static_cast<uint64_t>(std::chrono::duration_cast
			<std::chrono::nanoseconds>(t1 - t0_single_[i]).count());
		AddGroupTimingSample(ops, ns); // credit to all ops in the list
		return;
	}

	// inside an OMP parallel region, all threads must reach the barrier
	#pragma omp barrier
	#pragma omp single
	{
		const double t1 = omp_get_wtime();
		const uint64_t ns = static_cast<uint64_t>((t1 - t0_team_[i]) * 1e9); // s -> ns
		AddGroupTimingSample(ops, ns); // credit to all ops in the list
	}
	#pragma omp barrier
}

void BenchmarksMgr::AddTimingSample(BenchOp op, uint64_t ns)
{
	const uint16_t i = GetIdx(op);

	curStepNs_[i] += ns;
	curStepOcc_[i] = 1;
}

void BenchmarksMgr::AddGroupTimingSample(std::initializer_list<BenchOp> ops, uint64_t ns)
{
	for (auto op : ops)
	{
		const uint16_t i = GetIdx(op);
		curStepNs_[i] += ns;
		curStepOcc_[i] = 1;
	}
}

uint64_t BenchmarksMgr::GetBench_Latest(BenchOp op) const
{
	if (curRecCount_ == 0) { return 0; }

	return recLatest_.time[GetIdx(op)];
}

uint64_t BenchmarksMgr::GetBench_Avg_OverAll(BenchOp op) const
{
	const uint64_t sum = benchSum_allTime_[GetIdx(op)];
	if (curRecCount_ == 0 || sum == 0) { return 0; }

	// integer division with ceil
	return ((sum - 1) / curRecCount_) + 1;
}

uint64_t BenchmarksMgr::GetBench_Avg_WhenUsed(BenchOp op) const
{
	const uint64_t sum = benchSum_allTime_[GetIdx(op)];
	const uint64_t numOcc = numOcc_allTime_[GetIdx(op)];
	if (curRecCount_ == 0 || sum == 0 || numOcc == 0) { return 0; }

	// integer division with ceil
	return ((sum - 1) / numOcc) + 1;
}

uint64_t BenchmarksMgr::GetBench_Min(BenchOp op) const
{
	if (curRecCount_ == 0) { return 0; }

	return benchMin_allTime_[GetIdx(op)];
}

uint64_t BenchmarksMgr::GetBench_Max(BenchOp op) const
{
	if (curRecCount_ == 0) { return 0; }

	return benchMax_allTime_[GetIdx(op)];
}

uint64_t BenchmarksMgr::GetBench_NumOcc(BenchOp op) const
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
