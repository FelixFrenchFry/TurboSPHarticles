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
	Clear(recLatest_);
	ClearAllTime();
}

void MemoryTracker::BeginStep()
{
	recLatest_ = {};
	Clear(recLatest_);
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
		for (uint8_t s = 0; s < GetIdx(MemSpace::Count); s++)
		{
			for (uint8_t i = 0; i < GetIdx(MemBucket::Count); i++)
			{
				// update all-time max of used and allocated memory
				memMax_allTime_[s][i].usedBytes =
					std::max(recLatest_.mem[s][i].usedBytes,
							 memMax_allTime_[s][i].usedBytes);
				memMax_allTime_[s][i].allocBytes =
					std::max(recLatest_.mem[s][i].allocBytes,
							 memMax_allTime_[s][i].allocBytes);
				memMax_allTime_[s][i].occupiedBytes =
					std::max(recLatest_.mem[s][i].occupiedBytes,
							 memMax_allTime_[s][i].occupiedBytes);
			}
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
	for (uint8_t s = 0; s < GetIdx(MemSpace::Count); s++)
	{
		for (uint8_t i = 0; i < GetIdx(MemBucket::Count); i++)
		{
			snap.latest[s][i] = recLatest_.mem[s][i];
			snap.max[s][i] = memMax_allTime_[s][i];
		}
	}
}

uint64_t MemoryTracker::GetMem_Latest_Used(
	const MemSpace space,
	const MemBucket bucket) const
{
	if (curRecCount_ == 0) { return 0ull; }

	return recLatest_.mem[GetIdx(space)][GetIdx(bucket)].usedBytes;
}

uint64_t MemoryTracker::GetMem_Latest_Alloc(
	const MemSpace space,
	const MemBucket bucket) const
{
	if (curRecCount_ == 0) { return 0ull; }

	return recLatest_.mem[GetIdx(space)][GetIdx(bucket)].allocBytes;
}

uint64_t MemoryTracker::GetMem_Latest_Occupied(
	const MemSpace space,
	const MemBucket bucket) const
{
	if (curRecCount_ == 0) { return 0ull; }

	return recLatest_.mem[GetIdx(space)][GetIdx(bucket)].occupiedBytes;
}

uint64_t MemoryTracker::GetMem_Max_Used(
	const MemSpace space,
	const MemBucket bucket) const
{
	if (curRecCount_ == 0) { return 0ull; }

	return memMax_allTime_[GetIdx(space)][GetIdx(bucket)].usedBytes;
}

uint64_t MemoryTracker::GetMem_Max_Alloc(
	const MemSpace space,
	const MemBucket bucket) const
{
	if (curRecCount_ == 0) { return 0ull; }

	return memMax_allTime_[GetIdx(space)][GetIdx(bucket)].allocBytes;
}

uint64_t MemoryTracker::GetMem_Max_Occupied(
	const MemSpace space,
	const MemBucket bucket) const
{
	if (curRecCount_ == 0) { return 0ull; }

	return memMax_allTime_[GetIdx(space)][GetIdx(bucket)].occupiedBytes;
}

uint64_t MemoryTracker::GetMem_Latest_Used(const MemBucket bucket) const
{
	return GetMem_Latest_Used(MemSpace::RAM, bucket);
}

uint64_t MemoryTracker::GetMem_Latest_Alloc(MemBucket bucket) const
{
	return GetMem_Latest_Alloc(MemSpace::RAM, bucket);
}

uint64_t MemoryTracker::GetMem_Latest_Occupied(MemBucket bucket) const
{
	return GetMem_Latest_Occupied(MemSpace::RAM, bucket);
}

uint64_t MemoryTracker::GetMem_Max_Used(MemBucket bucket) const
{
	return GetMem_Max_Used(MemSpace::RAM, bucket);
}

uint64_t MemoryTracker::GetMem_Max_Alloc(MemBucket bucket) const
{
	return GetMem_Max_Alloc(MemSpace::RAM, bucket);
}

uint64_t MemoryTracker::GetMem_Max_Occupied(MemBucket bucket) const
{
	return GetMem_Max_Occupied(MemSpace::RAM, bucket);
}

void MemoryTracker::ClearAllTime()
{
	Clear(memMax_allTime_);
}
