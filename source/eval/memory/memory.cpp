// memory.cpp

// external
#include <algorithm>
#if defined(_WIN32)
	#include <windows.h>
	#include <psapi.h>
#elif defined(__linux__)
	#include <sys/resource.h>
#endif

// internal
#include "memory.h"



namespace
{
	Mem::ProcessMemUsage QueryProcessMemoryUsage() noexcept
	{
#if defined(_WIN32)
		PROCESS_MEMORY_COUNTERS counters{};
		counters.cb = sizeof(counters);
		if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
		{
			return {};
		}

		return { static_cast<uint64_t>(counters.PeakWorkingSetSize) };
#elif defined(__linux__)
		rusage usage{};
		if (getrusage(RUSAGE_SELF, &usage) != 0)
		{
			return {};
		}

		return { static_cast<uint64_t>(usage.ru_maxrss) * 1024ull };
#else
		return {};
#endif
	}

} // namespace


using namespace Mem;

MemoryTracker::MemoryTracker(const Sim::SimParams& simParams)
	: N_maxRecords_(simParams.maxMemRecs)
	, N_records_(0u)
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

bool MemoryTracker::SampleProcessMemory()
{
	const ProcessMemUsage process = QueryProcessMemoryUsage();
	const uint64_t N_previousPeakResidentBytes =
		recLatest_.process.N_peakResidentBytes;
	const uint64_t N_peakResidentBytes = std::max(
		N_previousPeakResidentBytes,
		process.N_peakResidentBytes);
	const bool changed = N_peakResidentBytes > N_previousPeakResidentBytes;
	recLatest_.process.N_peakResidentBytes = N_peakResidentBytes;
	processMax_allTime_.N_peakResidentBytes = std::max(
		processMax_allTime_.N_peakResidentBytes,
		N_peakResidentBytes);
	return changed;
}

void MemoryTracker::EndStep(uint64_t simStep, double simTime)
{
	// fill latest record
	recLatest_.simStep = simStep;
	recLatest_.simTime = simTime;
	SampleProcessMemory();

	// push latest record to history and check excess
	recHistory_.push_back(recLatest_);
	if (recHistory_.size() > N_maxRecords_)
	{
		recHistory_.pop_front();
	}

	if (N_records_ == 0) // first entry
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
				memMax_allTime_[s][i].N_usedBytes =
					std::max(recLatest_.mem[s][i].N_usedBytes,
							 memMax_allTime_[s][i].N_usedBytes);
				memMax_allTime_[s][i].N_allocBytes =
					std::max(recLatest_.mem[s][i].N_allocBytes,
							 memMax_allTime_[s][i].N_allocBytes);
			}
		}
	}

	N_records_ += 1;
}

void MemoryTracker::PackMemSnapshot(Sync::MemSnapshot& snap) const
{
	// metadata
	snap.simStep = recLatest_.simStep;
	snap.simTime = recLatest_.simTime;
	snap.latestProcess = recLatest_.process;
	snap.maxProcess = processMax_allTime_;

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
	if (N_records_ == 0) { return 0ull; }

	return recLatest_.mem[GetIdx(space)][GetIdx(bucket)].N_usedBytes;
}

uint64_t MemoryTracker::GetMem_Latest_Alloc(
	const MemSpace space,
	const MemBucket bucket) const
{
	if (N_records_ == 0) { return 0ull; }

	return recLatest_.mem[GetIdx(space)][GetIdx(bucket)].N_allocBytes;
}

uint64_t MemoryTracker::GetMem_Max_Used(
	const MemSpace space,
	const MemBucket bucket) const
{
	if (N_records_ == 0) { return 0ull; }

	return memMax_allTime_[GetIdx(space)][GetIdx(bucket)].N_usedBytes;
}

uint64_t MemoryTracker::GetMem_Max_Alloc(
	const MemSpace space,
	const MemBucket bucket) const
{
	if (N_records_ == 0) { return 0ull; }

	return memMax_allTime_[GetIdx(space)][GetIdx(bucket)].N_allocBytes;
}

uint64_t MemoryTracker::GetMem_Latest_Used(const MemBucket bucket) const
{
	return GetMem_Latest_Used(MemSpace::RAM, bucket);
}

uint64_t MemoryTracker::GetMem_Latest_Alloc(MemBucket bucket) const
{
	return GetMem_Latest_Alloc(MemSpace::RAM, bucket);
}

uint64_t MemoryTracker::GetMem_Max_Used(MemBucket bucket) const
{
	return GetMem_Max_Used(MemSpace::RAM, bucket);
}

uint64_t MemoryTracker::GetMem_Max_Alloc(MemBucket bucket) const
{
	return GetMem_Max_Alloc(MemSpace::RAM, bucket);
}

void MemoryTracker::ClearAllTime()
{
	Clear(memMax_allTime_);
	processMax_allTime_ = {};
}
