// memory.cpp

// external
#include <algorithm>

// internal
#include "memory.h"



using namespace Mem;

MemoryTracker::MemoryTracker(const Sim::SimParams& simParams)
	: maxRecCount_(simParams.maxMemRecs)
	, curRecCount_(0u)
{
	recLatest_ = {};
	recLatest_.mem.fill(MemUsage{ 0ull, 0ull });
	ClearAllTime();
}

void MemoryTracker::BeginStep()
{
	recLatest_ = {};
	recLatest_.mem.fill(MemUsage{ 0ull, 0ull });
}

void MemoryTracker::EndStep(uint32_t simStep, double simTime)
{
	// fill latest record
	recLatest_.simStep = simStep;
	recLatest_.simTime = simTime;

	// push latest record to history and check excess
	recHistory_.push_back(recLatest_);
	if (recHistory_.size() > maxRecCount_)
	{
		recHistory_.pop_front();
	}

	if (curRecCount_ == 0) // first entry
	{
		memMax_allTime_ = recLatest_.mem;
	}
	else
	{
		for (uint8_t i = 0; i < GetIdx(MemBucket::Count); i++)
		{
			// update all-time max of used and allocated memory
			memMax_allTime_[i].usedBytes =
				std::max(recLatest_.mem[i].usedBytes,
						 memMax_allTime_[i].usedBytes);
			memMax_allTime_[i].allocBytes =
				std::max(recLatest_.mem[i].allocBytes,
						 memMax_allTime_[i].allocBytes);
		}
	}

	curRecCount_ += 1;
}

void MemoryTracker::PackMemSnapshot(Sync::MemSnapshot& snap) const
{
	// metadata
	snap.simStep = recLatest_.simStep;
	snap.simTime = recLatest_.simTime;

	// fill in the latest and max memory usage (used and allocated)
	for (uint8_t i = 0; i < GetIdx(MemBucket::Count); i++)
	{
		snap.latest[i] = recLatest_.mem[i];
		snap.max[i] = memMax_allTime_[i];
	}
}

uint64_t MemoryTracker::GetMem_Latest_Used(const MemBucket bucket) const
{
	if (curRecCount_ == 0) { return 0ull; }

	return recLatest_.mem[GetIdx(bucket)].usedBytes;
}

uint64_t MemoryTracker::GetMem_Latest_Alloc(MemBucket bucket) const
{
	if (curRecCount_ == 0) { return 0ull; }

	return recLatest_.mem[GetIdx(bucket)].allocBytes;
}

uint64_t MemoryTracker::GetMem_Max_Used(MemBucket bucket) const
{
	if (curRecCount_ == 0) { return 0ull; }

	return memMax_allTime_[GetIdx(bucket)].usedBytes;
}

uint64_t MemoryTracker::GetMem_Max_Alloc(MemBucket bucket) const
{
	if (curRecCount_ == 0) { return 0ull; }

	return memMax_allTime_[GetIdx(bucket)].allocBytes;
}

void MemoryTracker::ClearAllTime()
{
	memMax_allTime_.fill(MemUsage{ 0ull, 0ull });
}
