// memory.h
//
// TODO: comment

#pragma once

// external
#include <deque>

// internal
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Sync/snapshots.h"
#include "metrics_mem.h"



namespace Mem
{
	enum class ByteUnit : uint8_t
	{
		GiB, MiB, KiB, B
	};

	class MemoryTracker
	{
	public: // -----------------------------------------------------------------

		MemoryTracker(const Sim::SimParams& simParams);

		// resets the latest memory usage before pulling in a new record
		void BeginStep();

		// pushes the record to history and aggregates step's memory usage
		void EndStep(uint32_t simStep, double simTime);

		void PackMemSnapshot(Sync::MemSnapshot& snap) const;

		// per-metric queries
		uint64_t GetMem_Latest_Used(MemBucket bucket) const;
		uint64_t GetMem_Latest_Alloc(MemBucket bucket) const;
		uint64_t GetMem_Max_Used(MemBucket bucket) const;
		uint64_t GetMem_Max_Alloc(MemBucket bucket) const;

		// helpers
		MemRecordStep& GetCurrentRecord() { return recLatest_; }
		const MemRecordStep& GetCurrentRecord() const { return recLatest_; }
		const std::deque<MemRecordStep>& GetHistory() const { return recHistory_; }

		static inline double ConvertFromBytes(ByteUnit unit, uint64_t bytes)
		{
			switch (unit)
			{
			case ByteUnit::GiB: { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }
			case ByteUnit::MiB: { return static_cast<double>(bytes) / (1024.0 * 1024.0); }
			case ByteUnit::KiB: { return static_cast<double>(bytes) / 1024.0; }
			case ByteUnit::B: { return static_cast<double>(bytes); }
			}
			return static_cast<double>(bytes);
		}

		static inline const char* UnitLabel(ByteUnit u)
		{
			switch (u)
			{
			case ByteUnit::GiB: { return "GiB"; }
			case ByteUnit::MiB: { return "MiB"; }
			case ByteUnit::KiB: { return "KiB"; }
			case ByteUnit::B: { return "B"; }
			}
			return "";
		}

	private: // ----------------------------------------------------------------

		// max and current number of steps for which memory usage is kept
		uint32_t maxRecCount_;
		uint32_t curRecCount_;

		MemRecordStep recLatest_;

		// rolling buffer for the memory usage for the last maxCount_ steps
		std::deque<MemRecordStep> recHistory_;

		// all-time per-bucket aggregates
		std::array<MemUsage, GetIdx(MemBucket::Count)> memMax_allTime_{};

		// helpers
		void ClearAllTime();
	};

} // namespace Mem
