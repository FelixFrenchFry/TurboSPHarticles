// memory.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>
#include <deque>

// internal
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/sync/snapshots.h"
#include "categories.h"



namespace Mem
{
	class MemoryTracker
	{
	public:

		MemoryTracker(const Sim::SimParams& simParams);

		// resets the latest memory usage before pulling in a new record
		void BeginStep();

		// pushes the record to history and aggregates step's memory usage
		void EndStep(uint64_t simStep, double simTime);

		// samples OS-reported process memory without adding a simulation-step record
		bool SampleProcessMemory();

		void PackMemSnapshot(Sync::MemSnapshot& snap) const;

		// per-metric queries
		uint64_t GetMem_Latest_Used(MemSpace space, MemBucket bucket) const;
		uint64_t GetMem_Latest_Alloc(MemSpace space, MemBucket bucket) const;
		uint64_t GetMem_Max_Used(MemSpace space, MemBucket bucket) const;
		uint64_t GetMem_Max_Alloc(MemSpace space, MemBucket bucket) const;
		uint64_t GetMem_Latest_Used(MemBucket bucket) const;
		uint64_t GetMem_Latest_Alloc(MemBucket bucket) const;
		uint64_t GetMem_Max_Used(MemBucket bucket) const;
		uint64_t GetMem_Max_Alloc(MemBucket bucket) const;

		// helpers
		MemRecordStep& GetCurrentRecord() { return recLatest_; }
		const MemRecordStep& GetCurrentRecord() const { return recLatest_; }
		const std::deque<MemRecordStep>& GetHistory() const { return recHistory_; }

	private:

		// max and current number of steps for which memory usage is kept
		uint32_t N_maxRecords_;
		uint32_t N_records_;

		// device memory is only sampled for the GPU backend
		Base::BackendType backendType_;

		MemRecordStep recLatest_;

		// rolling buffer for the memory usage for the last N_maxRecords_ steps
		std::deque<MemRecordStep> recHistory_;

		// all-time tracked memory aggregates
		MemSpaceRecord memMax_allTime_{};

		// all-time process peak
		ProcessMemUsage processMax_allTime_{};

		// all-time device peaks, the device reports none of its own
		uint64_t N_peakDeviceResidentBytes_allTime_ = 0;
		uint64_t N_peakDeviceAllocBytes_allTime_ = 0;

		// samples OS-reported and backend-tracked device memory
		bool SampleDeviceMemory();

		// helpers
		void ClearAllTime();
	};

} // namespace Mem
