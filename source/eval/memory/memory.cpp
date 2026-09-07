// memory.cpp

// external
#include <algorithm>
#include <atomic>
#include <spdlog/spdlog.h>
#include <vector>
#if defined(_WIN32)
	#include <windows.h>
	#include <psapi.h>
#elif defined(__linux__)
	#include <cstdio>
	#include <sys/resource.h>
	#include <unistd.h>
#endif
#ifdef TURBOSPHARTICLES_WITH_CUDA
	#include <cuda_runtime.h>
	#if defined(_WIN32)
		#include <cstring>
		#include <dxgi1_4.h>
	#elif defined(TSP_HAS_NVML)
		#include <nvml.h>
	#endif
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

		return {
			static_cast<uint64_t>(counters.WorkingSetSize),
			static_cast<uint64_t>(counters.PeakWorkingSetSize) };
#elif defined(__linux__)
		rusage usage{};
		if (getrusage(RUSAGE_SELF, &usage) != 0)
		{
			return {};
		}

		// getrusage only reports the high-water mark, the current one comes from procfs
		uint64_t N_residentBytes = 0;
		if (FILE* statm = std::fopen("/proc/self/statm", "r"))
		{
			unsigned long N_residentPages = 0;
			if (std::fscanf(statm, "%*u %lu", &N_residentPages) == 1)
			{
				N_residentBytes = static_cast<uint64_t>(N_residentPages)
					* static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
			}
			std::fclose(statm);
		}

		return {
			N_residentBytes,
			static_cast<uint64_t>(usage.ru_maxrss) * 1024ull };
#else
		return {};
#endif
	}

#ifdef TURBOSPHARTICLES_WITH_CUDA
#if defined(_WIN32)
	// the DXGI adapter backing the active CUDA device, matched by its LUID
	IDXGIAdapter3* ResolveCudaDxgiAdapter() noexcept
	{
		int device = 0;
		if (cudaGetDevice(&device) != cudaSuccess) { return nullptr; }

		cudaDeviceProp props{};
		if (cudaGetDeviceProperties(&props, device) != cudaSuccess) { return nullptr; }

		IDXGIFactory1* factory = nullptr;
		if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
			reinterpret_cast<void**>(&factory))))
		{
			return nullptr;
		}

		IDXGIAdapter3* match = nullptr;
		IDXGIAdapter1* adapter = nullptr;
		for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; i++)
		{
			DXGI_ADAPTER_DESC1 desc{};
			if (SUCCEEDED(adapter->GetDesc1(&desc))
				&& std::memcmp(&desc.AdapterLuid, props.luid, sizeof(desc.AdapterLuid)) == 0)
			{
				adapter->QueryInterface(__uuidof(IDXGIAdapter3),
					reinterpret_cast<void**>(&match));
				adapter->Release();
				break;
			}
			adapter->Release();
		}

		factory->Release();
		return match;
	}
#elif defined(TSP_HAS_NVML)
	// the NVML device backing the active CUDA device, matched by its PCI bus id
	nvmlDevice_t ResolveCudaNvmlDevice() noexcept
	{
		if (nvmlInit() != NVML_SUCCESS) { return nullptr; }

		int device = 0;
		if (cudaGetDevice(&device) != cudaSuccess)
		{
			nvmlShutdown();
			return nullptr;
		}

		char pciBusId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = {};
		if (cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device) != cudaSuccess)
		{
			nvmlShutdown();
			return nullptr;
		}

		nvmlDevice_t nvmlDevice{};
		if (nvmlDeviceGetHandleByPciBusId(pciBusId, &nvmlDevice) != NVML_SUCCESS)
		{
			nvmlShutdown();
			return nullptr;
		}

		return nvmlDevice;
	}
#endif

	// OS-reported device memory of this process, 0 when unavailable on the platform
	uint64_t QueryProcessDeviceMemoryUsage() noexcept
	{
#if defined(_WIN32)
		// the LUID match only has to be resolved once, and the adapter is never released
		static IDXGIAdapter3* const adapter = ResolveCudaDxgiAdapter();
		if (!adapter) { return 0ull; }

		DXGI_QUERY_VIDEO_MEMORY_INFO info{};
		if (FAILED(adapter->QueryVideoMemoryInfo(
			0u, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
		{
			return 0ull;
		}

		return static_cast<uint64_t>(info.CurrentUsage);
#elif defined(TSP_HAS_NVML)
		// the device handle only has to be resolved once
		static nvmlDevice_t nvmlDevice = ResolveCudaNvmlDevice();
		if (!nvmlDevice) { return 0ull; }

		// query the process list twice, since the first call only reports the required capacity
		// (a device without compute processes reports success instead, and has nothing to attribute)
		unsigned int N_procs = 0;
		if (nvmlDeviceGetComputeRunningProcesses(nvmlDevice, &N_procs, nullptr)
			!= NVML_ERROR_INSUFFICIENT_SIZE)
		{
			return 0ull;
		}

		// reused across steps, with headroom for processes that start between the two calls
		static std::vector<nvmlProcessInfo_t> procs{};
		procs.resize(N_procs + 8);
		unsigned int N_entries = static_cast<unsigned int>(procs.size());
		if (nvmlDeviceGetComputeRunningProcesses(nvmlDevice, &N_entries, procs.data())
			!= NVML_SUCCESS)
		{
			return 0ull;
		}

		const unsigned int pid = static_cast<unsigned int>(getpid());
		for (unsigned int i = 0; i < N_entries; i++)
		{
			if (procs[i].pid != pid) { continue; }
			if (procs[i].usedGpuMemory == NVML_VALUE_NOT_AVAILABLE) { break; }

			return static_cast<uint64_t>(procs[i].usedGpuMemory);
		}

		return 0ull;
#else
		// no per-process source, fall back to the device-wide occupancy
		size_t N_freeBytes = 0;
		size_t N_totalBytes = 0;
		if (cudaMemGetInfo(&N_freeBytes, &N_totalBytes) != cudaSuccess) { return 0ull; }

		return static_cast<uint64_t>(N_totalBytes - N_freeBytes);
#endif
	}
#endif

	// whether the platform can attribute device memory to this process
#if defined(TURBOSPHARTICLES_WITH_CUDA) && (defined(_WIN32) || defined(TSP_HAS_NVML))
	constexpr bool DEVICE_MEM_IS_PER_PROCESS = true;
#else
	constexpr bool DEVICE_MEM_IS_PER_PROCESS = false;
#endif

	// running totals of the raw device allocations made through the device buffers
	std::atomic<uint64_t> N_deviceAllocBytes{ 0 };
	std::atomic<uint64_t> N_peakDeviceAllocBytes{ 0 };

} // namespace


using namespace Mem;

void Mem::TrackDeviceAlloc(const uint64_t N_bytes) noexcept
{
	const uint64_t N_currentBytes =
		N_deviceAllocBytes.fetch_add(N_bytes, std::memory_order_relaxed) + N_bytes;

	uint64_t N_peakBytes = N_peakDeviceAllocBytes.load(std::memory_order_relaxed);
	while (N_currentBytes > N_peakBytes
		&& !N_peakDeviceAllocBytes.compare_exchange_weak(N_peakBytes, N_currentBytes,
			std::memory_order_relaxed))
	{
	}
}

void Mem::TrackDeviceFree(const uint64_t N_bytes) noexcept
{
	N_deviceAllocBytes.fetch_sub(N_bytes, std::memory_order_relaxed);
}

DeviceMemUsage Mem::GetTrackedDeviceAllocations() noexcept
{
	DeviceMemUsage device{};
	device.N_allocBytes = N_deviceAllocBytes.load(std::memory_order_relaxed);
	device.N_peakAllocBytes = N_peakDeviceAllocBytes.load(std::memory_order_relaxed);

	return device;
}

void Mem::ResetTrackedDeviceAllocationPeak() noexcept
{
	// the buffers still alive stay counted, only the peak restarts from them
	N_peakDeviceAllocBytes.store(
		N_deviceAllocBytes.load(std::memory_order_relaxed),
		std::memory_order_relaxed);
}

MemoryTracker::MemoryTracker(const Sim::SimParams& simParams)
	: N_maxRecords_(simParams.maxMemRecs)
	, N_records_(0u)
	, backendType_(simParams.backendType)
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
	recLatest_.process.N_residentBytes = process.N_residentBytes;
	recLatest_.process.N_peakResidentBytes = N_peakResidentBytes;
	processMax_allTime_.N_residentBytes = std::max(
		processMax_allTime_.N_residentBytes,
		process.N_residentBytes);
	processMax_allTime_.N_peakResidentBytes = std::max(
		processMax_allTime_.N_peakResidentBytes,
		N_peakResidentBytes);
	const bool deviceChanged = SampleDeviceMemory();

	return changed || deviceChanged;
}

bool MemoryTracker::SampleDeviceMemory()
{
	if (backendType_ != Base::BackendType::GPU) { return false; }

	DeviceMemUsage device = GetTrackedDeviceAllocations();
#ifdef TURBOSPHARTICLES_WITH_CUDA
	device.N_residentBytes = QueryProcessDeviceMemoryUsage();
#endif

	// a GPU run never occupies zero device memory, so zero means no source could be queried
	if (device.N_residentBytes == 0ull)
	{
		static bool reportedUnavailable = false;
		if (!reportedUnavailable)
		{
			reportedUnavailable = true;
			SPDLOG_WARN("Device memory reporting is unavailable, omitting VRSS and VHWM.");
		}
	}
	else if constexpr (!DEVICE_MEM_IS_PER_PROCESS)
	{
		static bool reportedDeviceWide = false;
		if (!reportedDeviceWide)
		{
			reportedDeviceWide = true;
			SPDLOG_WARN("No per-process device memory source, VRSS and VHWM cover the whole "
				"device and include other processes.");
		}
	}

	// the device reports no high-water mark, it is accumulated here
	// (unlike the OS-reported process memory, which has a high-water mark)
	const uint64_t N_previousPeakResidentBytes = N_peakDeviceResidentBytes_allTime_;
	N_peakDeviceResidentBytes_allTime_ = std::max(
		N_previousPeakResidentBytes,
		device.N_residentBytes);
	N_peakDeviceAllocBytes_allTime_ = std::max(
		N_peakDeviceAllocBytes_allTime_,
		device.N_peakAllocBytes);

	recLatest_.device.N_residentBytes = device.N_residentBytes;
	recLatest_.device.N_peakResidentBytes = N_peakDeviceResidentBytes_allTime_;
	recLatest_.device.N_allocBytes = device.N_allocBytes;
	recLatest_.device.N_peakAllocBytes = N_peakDeviceAllocBytes_allTime_;

	return N_peakDeviceResidentBytes_allTime_ > N_previousPeakResidentBytes;
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
	snap.latestDevice = recLatest_.device;

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
	N_peakDeviceResidentBytes_allTime_ = 0;
	N_peakDeviceAllocBytes_allTime_ = 0;
	ResetTrackedDeviceAllocationPeak();
}
