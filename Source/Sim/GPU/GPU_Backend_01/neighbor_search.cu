// neighbor_search.cu

// external
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/transform_reduce.h>

// internal
#include "common.cuh"
#include "neighbor_search.cuh"
#include "neighbor_search_kernels.cuh"
#include "sph_kernels.cuh"
#include "Source/Base/helpers.h"
#include "Source/Sim/GPU/gpu_utils.h"



using namespace Sim;

namespace
{
	static constexpr uint32_t N_SellWidth = 32u;

	struct ToU64
	{
		__host__ __device__ uint64_t operator()(const uint16_t v) const
		{
			return static_cast<uint64_t>(v);
		}
	};

	struct SellChunkPitchIfFormat
	{
		explicit SellChunkPitchIfFormat(const uint8_t targetFormat)
			: targetFormat_{ targetFormat } {}

		__host__ __device__ uint64_t operator()(
			const thrust::tuple<uint16_t, uint8_t>& t) const
		{
			if (thrust::get<1>(t) != targetFormat_) { return 0u; }

			return static_cast<uint64_t>(thrust::get<0>(t)) * N_SellWidth;
		}

		uint8_t targetFormat_;
	};

	__host__ __device__ inline uint32_t SellChunkNumTotal(const uint32_t N_F) noexcept
	{
		return (N_F == 0u) ? 0u : ((N_F - 1u) / N_SellWidth) + 1u;
	}

	inline uint32_t QueryCellSlotCount(const uint32_t cellQueryReach)
	{
		const uint64_t width = 2ull * static_cast<uint64_t>(cellQueryReach) + 1ull;
		constexpr uint64_t maxSlots =
			static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
		if (width > maxSlots / width || width * width > maxSlots / width) [[unlikely]]
		{
			throw std::runtime_error("GPU compact CLL query-cell cache slot count overflow.");
		}

		const uint64_t slots = width * width * width;
		return static_cast<uint32_t>(slots);
	}

	inline size_t AlignUpBytes(const size_t offset, const size_t align) noexcept
	{
		return (offset + align - 1u) & ~(align - 1u);
	}

	template <typename T>
	inline size_t ReserveScratchElems(size_t& cursor, const size_t count) noexcept
	{
		cursor = AlignUpBytes(cursor, alignof(T));
		const size_t offset = cursor;
		cursor += count * sizeof(T);

		return offset;
	}

	struct NS_FinalizeScratchLayout
	{
		size_t chunkMinRefFF = 0;
		size_t chunkMinRefFB = 0;
		size_t chunkMaxRefFF = 0;
		size_t chunkMaxRefFB = 0;
		size_t chunkOffsetsScratch = 0;
		size_t totalBytes = 0;
	};

	struct NS_CompactBuildScratchLayout
	{
		size_t cellIdF = 0;
		size_t cellIdB = 0;
		size_t sortedParticleIdxB = 0;
		size_t totalBytes = 0;
	};

	struct NS_CompactSharedScratchLayout
	{
		NS_FinalizeScratchLayout finalize;
		NS_CompactBuildScratchLayout build;
		size_t phaseScratchBytes = 0;
		size_t queryLoAnyByCellF = 0;
		size_t queryHiAnyByCellF = 0;
		size_t totalBytes = 0;
	};

	inline NS_FinalizeScratchLayout BuildFinalizeScratchLayout(const uint32_t N_F)
	{
		NS_FinalizeScratchLayout layout{};
		size_t cursor = 0;
		const uint32_t N_chunks = SellChunkNumTotal(N_F);

		layout.chunkMinRefFF = ReserveScratchElems<uint32_t>(cursor, N_chunks);
		layout.chunkMinRefFB = ReserveScratchElems<uint32_t>(cursor, N_chunks);
		layout.chunkMaxRefFF = ReserveScratchElems<uint32_t>(cursor, N_chunks);
		layout.chunkMaxRefFB = ReserveScratchElems<uint32_t>(cursor, N_chunks);
		layout.chunkOffsetsScratch = ReserveScratchElems<uint64_t>(
			cursor, static_cast<size_t>(N_chunks) + 1ull);
		layout.totalBytes = cursor;

		return layout;
	}

	inline NS_CompactBuildScratchLayout BuildCompactBuildScratchLayout(
		const uint32_t N_F, const uint32_t N_B)
	{
		NS_CompactBuildScratchLayout layout{};
		size_t cursor = 0;

		layout.cellIdF = ReserveScratchElems<uint64_t>(cursor, N_F);
		layout.cellIdB = ReserveScratchElems<uint64_t>(cursor, N_B);
		layout.sortedParticleIdxB = ReserveScratchElems<uint32_t>(cursor, N_B);
		layout.totalBytes = cursor;

		return layout;
	}

	inline NS_CompactSharedScratchLayout BuildCompactSharedScratchLayout(
		const uint32_t N_F, const uint32_t N_B)
	{
		NS_CompactSharedScratchLayout layout{};
		layout.finalize = BuildFinalizeScratchLayout(N_F);
		layout.build = BuildCompactBuildScratchLayout(N_F, N_B);
		layout.phaseScratchBytes = std::max(layout.finalize.totalBytes, layout.build.totalBytes);

		size_t cursor = layout.phaseScratchBytes;
		layout.queryLoAnyByCellF = ReserveScratchElems<uint32_t>(cursor, N_F);
		layout.queryHiAnyByCellF = ReserveScratchElems<uint32_t>(cursor, N_F);
		layout.totalBytes = cursor;

		return layout;
	}

	template <typename T>
	inline T* ScratchPtr(DeviceBuffer<uint8_t>& arena, const size_t offsetBytes) noexcept
	{
		if (arena.Size() == 0) { return nullptr; }
		return reinterpret_cast<T*>(arena.Data() + offsetBytes);
	}

	template <typename T>
	inline const T* ScratchPtr(const DeviceBuffer<uint8_t>& arena, const size_t offsetBytes) noexcept
	{
		if (arena.Size() == 0) { return nullptr; }
		return reinterpret_cast<const T*>(arena.Data() + offsetBytes);
	}

	#if defined(_DEBUG)
	inline void CheckNbrListErrorFlag(
		const DeviceBuffer<uint32_t>& d_errorFlag,
		cudaStream_t stream,
		const char* context)
	{
		uint32_t errorFlag = 0u;
		GPU::Copy_DeviceToHost(&errorFlag, d_errorFlag.Data(), 1, stream);
		GPU::StreamSync(stream);

		if (errorFlag == 0u) { return; }

		std::string msg = std::string(context)
			+ " failed while building compressed SELL32 neighbor lists";
		if (errorFlag & GPU01::NbrListError_FillOverflow)
		{
			msg += "; fill count exceeded chunk pitch";
		}
		if (errorFlag & GPU01::NbrListError_InvalidChunkFormat)
		{
			msg += "; invalid compressed SELL32 chunk format";
		}
		if (errorFlag & GPU01::NbrListError_DeltaRangeOverflow)
		{
			msg += "; neighbor index does not fit selected delta format";
		}

		throw std::runtime_error(msg + ".");
	}
	#endif

	__global__ void K_BuildSellChunkMaxCount(
		const uint16_t* __restrict__ nbrCount,
		const uint32_t N_F,
		uint16_t* __restrict__ chunkMaxCount)
	{
		const uint32_t chunk = blockIdx.x * blockDim.x + threadIdx.x;
		const uint32_t N_chunks = SellChunkNumTotal(N_F);
		if (chunk >= N_chunks) { return; }

		const uint32_t begin = chunk * N_SellWidth;
		const uint32_t end = (begin + N_SellWidth < N_F)
						   ? (begin + N_SellWidth)
						   : N_F;

		uint16_t maxCount = 0;
		for (uint32_t i = begin; i < end; i++)
		{
			if (nbrCount[i] > maxCount) { maxCount = nbrCount[i]; }
		}

		chunkMaxCount[chunk] = maxCount;
	}

	__global__ void K_BuildSellChunkCompressionMeta(
		const uint16_t* __restrict__ chunkMaxCount,
		const uint32_t* __restrict__ chunkMinRef,
		const uint32_t* __restrict__ chunkMaxRef,
		const uint32_t N_chunks,
		// destinations
		uint32_t* __restrict__ chunkBaseIdx,
		uint8_t* __restrict__ chunkFormat)
	{
		const uint32_t chunk = blockIdx.x * blockDim.x + threadIdx.x;
		if (chunk >= N_chunks) { return; }

		const uint16_t pitch = chunkMaxCount[chunk];
		if (pitch == 0u)
		{
			chunkBaseIdx[chunk] = 0u;
			chunkFormat[chunk] = static_cast<uint8_t>(GPU01::NbrChunkFormat::Abs32);
			return;
		}

		const uint32_t minRef = chunkMinRef[chunk];
		const uint32_t maxRef = chunkMaxRef[chunk];
		const uint32_t span = (maxRef >= minRef) ? (maxRef - minRef) : 0xffffffffu;
		const bool canUseDelta16 =
			(maxRef >= minRef) && (span <= 0xffffu);
		const bool canUseDelta8 =
			(maxRef >= minRef) && (span <= 0xffu);

		chunkBaseIdx[chunk] = minRef;
		chunkFormat[chunk] = canUseDelta8
			? static_cast<uint8_t>(GPU01::NbrChunkFormat::Delta8)
			: (canUseDelta16
				? static_cast<uint8_t>(GPU01::NbrChunkFormat::Delta16)
				: static_cast<uint8_t>(GPU01::NbrChunkFormat::Abs32));
	}

	__global__ void K_SelectChunkOffsetsByFormat(
		const uint64_t* __restrict__ offsetsByFormat,
		const uint8_t* __restrict__ chunkFormat,
		const uint8_t targetFormat,
		const uint32_t N_chunks,
		// destinations
		uint64_t* __restrict__ chunkOffsets)
	{
		const uint32_t chunk = blockIdx.x * blockDim.x + threadIdx.x;
		if (chunk >= N_chunks) { return; }

		if (chunkFormat[chunk] == targetFormat)
		{
			chunkOffsets[chunk] = offsetsByFormat[chunk];
		}
	}

	template <typename T>
	inline Mem::MemUsage MemUsageOf_DeviceBuffer(
		const DeviceBuffer<T>& src) noexcept
	{
		const uint64_t bytes = static_cast<uint64_t>(src.Size()) * sizeof(T);

		return Mem::MemUsage{ bytes, bytes };
	}

	template <typename T>
	inline void ApplyPermutation_DeviceBuffer(
		const uint32_t* d_perm, const uint32_t N,
		DeviceBuffer<T>& d_src, DeviceBuffer<T>& d_tmp,
		cudaStream_t stream)
	{
		if (N == 0) { return; }

		thrust::gather(
			thrust::cuda::par.on(stream),
			thrust::device_pointer_cast(d_perm),
			thrust::device_pointer_cast(d_perm + N),
			thrust::device_pointer_cast(d_src.Data()),
			thrust::device_pointer_cast(d_tmp.Data()));

		thrust::copy(
			thrust::cuda::par.on(stream),
			thrust::device_pointer_cast(d_tmp.Data()),
			thrust::device_pointer_cast(d_tmp.Data() + N),
			thrust::device_pointer_cast(d_src.Data()));
	}

} // anonymous

void SimBackend_GPU01::FinalizeNeighborLists_Count()
{
	if (N_F_ == 0) // nothing to do for empty sim
	{
		N_totalFF_ = 0;
		N_totalFB_ = 0;
		N_packedFF_ = 0;
		N_packedFB_ = 0;
		N_delta8FF_ = 0;
		N_delta8FB_ = 0;
		N_delta16FF_ = 0;
		N_delta16FB_ = 0;
		N_abs32FF_ = 0;
		N_abs32FB_ = 0;

		if (d_nbrChunkOffsetsFF_.Size() != 1) { d_nbrChunkOffsetsFF_.Resize(1); }
		if (d_nbrChunkOffsetsFB_.Size() != 1) { d_nbrChunkOffsetsFB_.Resize(1); }

		const uint64_t zero = 0;
		GPU::Copy_HostToDevice(d_nbrChunkOffsetsFF_.Data(), &zero, 1, stream_);
		GPU::Copy_HostToDevice(d_nbrChunkOffsetsFB_.Data(), &zero, 1, stream_);
		GPU::StreamSync(stream_);

		return;
	}

	const NS_FinalizeScratchLayout scratchLayout = BuildFinalizeScratchLayout(N_F_);
	EnsureNsSharedScratchCapacity(scratchLayout.totalBytes);
	uint32_t* const d_nbrChunkMinRefFF = ScratchPtr<uint32_t>(d_nsSharedScratch_, scratchLayout.chunkMinRefFF);
	uint32_t* const d_nbrChunkMinRefFB = ScratchPtr<uint32_t>(d_nsSharedScratch_, scratchLayout.chunkMinRefFB);
	uint32_t* const d_nbrChunkMaxRefFF = ScratchPtr<uint32_t>(d_nsSharedScratch_, scratchLayout.chunkMaxRefFF);
	uint32_t* const d_nbrChunkMaxRefFB = ScratchPtr<uint32_t>(d_nsSharedScratch_, scratchLayout.chunkMaxRefFB);
	uint64_t* const d_nbrChunkOffsetsScratch = ScratchPtr<uint64_t>(
		d_nsSharedScratch_, scratchLayout.chunkOffsetsScratch);

	const uint32_t N_chunks = SellChunkNumTotal(N_F_);
	const uint32_t N_BLOCKSIZE = 256;
	const uint32_t N_GRIDSIZE = (N_chunks + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

	K_BuildSellChunkMaxCount<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream_>>>(
		d_nbrCountFF_.Data(), N_F_, d_nbrChunkMaxCountFF_.Data());
	GPU::CheckLastKernel("K_BuildSellChunkMaxCount(FF)");

	K_BuildSellChunkMaxCount<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream_>>>(
		d_nbrCountFB_.Data(), N_F_, d_nbrChunkMaxCountFB_.Data());
	GPU::CheckLastKernel("K_BuildSellChunkMaxCount(FB)");

	GPU::MemsetZero(d_nbrChunkOffsetsFF_.Data(), static_cast<size_t>(N_chunks) + 1ull, stream_);
	GPU::MemsetZero(d_nbrChunkOffsetsFB_.Data(), static_cast<size_t>(N_chunks) + 1ull, stream_);
	GPU::MemsetZero(d_nbrChunkOffsetsScratch, static_cast<size_t>(N_chunks) + 1ull, stream_);

	K_BuildSellChunkCompressionMeta<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream_>>>(
		d_nbrChunkMaxCountFF_.Data(),
		d_nbrChunkMinRefFF,
		d_nbrChunkMaxRefFF,
		N_chunks,
		d_nbrChunkBaseIdxFF_.Data(),
		d_nbrChunkFormatFF_.Data());
	GPU::CheckLastKernel("K_BuildSellChunkCompressionMeta(FF)");

	auto scanAndSelectOffsets =
		[&](const uint16_t* d_chunkMaxCount,
			const uint8_t* d_chunkFormat,
			uint64_t* d_chunkOffsets,
			uint64_t& h_totalDelta8,
			uint64_t& h_totalDelta16,
			uint64_t& h_totalAbs32)
	{
		const uint64_t zero = 0;
		GPU::Copy_HostToDevice(d_nbrChunkOffsetsScratch, &zero, 1, stream_);

		thrust::transform_inclusive_scan(
			thrust::cuda::par.on(stream_),
			thrust::make_zip_iterator(thrust::make_tuple(
				thrust::device_pointer_cast(d_chunkMaxCount),
				thrust::device_pointer_cast(d_chunkFormat))),
			thrust::make_zip_iterator(thrust::make_tuple(
				thrust::device_pointer_cast(d_chunkMaxCount + N_chunks),
				thrust::device_pointer_cast(d_chunkFormat + N_chunks))),
			thrust::device_pointer_cast(d_nbrChunkOffsetsScratch + 1),
			SellChunkPitchIfFormat(static_cast<uint8_t>(GPU01::NbrChunkFormat::Delta8)),
			thrust::plus<uint64_t>{}
		);
		K_SelectChunkOffsetsByFormat<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream_>>>(
			d_nbrChunkOffsetsScratch,
			d_chunkFormat,
			static_cast<uint8_t>(GPU01::NbrChunkFormat::Delta8),
			N_chunks,
			d_chunkOffsets);
		GPU::CheckLastKernel("K_SelectChunkOffsetsByFormat(Delta8)");
		GPU::Copy_DeviceToHost(&h_totalDelta8,
			d_nbrChunkOffsetsScratch + N_chunks, 1, stream_);

		GPU::Copy_HostToDevice(d_nbrChunkOffsetsScratch, &zero, 1, stream_);
		thrust::transform_inclusive_scan(
			thrust::cuda::par.on(stream_),
			thrust::make_zip_iterator(thrust::make_tuple(
				thrust::device_pointer_cast(d_chunkMaxCount),
				thrust::device_pointer_cast(d_chunkFormat))),
			thrust::make_zip_iterator(thrust::make_tuple(
				thrust::device_pointer_cast(d_chunkMaxCount + N_chunks),
				thrust::device_pointer_cast(d_chunkFormat + N_chunks))),
			thrust::device_pointer_cast(d_nbrChunkOffsetsScratch + 1),
			SellChunkPitchIfFormat(static_cast<uint8_t>(GPU01::NbrChunkFormat::Delta16)),
			thrust::plus<uint64_t>{}
		);
		K_SelectChunkOffsetsByFormat<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream_>>>(
			d_nbrChunkOffsetsScratch,
			d_chunkFormat,
			static_cast<uint8_t>(GPU01::NbrChunkFormat::Delta16),
			N_chunks,
			d_chunkOffsets);
		GPU::CheckLastKernel("K_SelectChunkOffsetsByFormat(Delta16)");
		GPU::Copy_DeviceToHost(&h_totalDelta16,
			d_nbrChunkOffsetsScratch + N_chunks, 1, stream_);

		GPU::Copy_HostToDevice(d_nbrChunkOffsetsScratch, &zero, 1, stream_);
		thrust::transform_inclusive_scan(
			thrust::cuda::par.on(stream_),
			thrust::make_zip_iterator(thrust::make_tuple(
				thrust::device_pointer_cast(d_chunkMaxCount),
				thrust::device_pointer_cast(d_chunkFormat))),
			thrust::make_zip_iterator(thrust::make_tuple(
				thrust::device_pointer_cast(d_chunkMaxCount + N_chunks),
				thrust::device_pointer_cast(d_chunkFormat + N_chunks))),
			thrust::device_pointer_cast(d_nbrChunkOffsetsScratch + 1),
			SellChunkPitchIfFormat(static_cast<uint8_t>(GPU01::NbrChunkFormat::Abs32)),
			thrust::plus<uint64_t>{}
		);
		K_SelectChunkOffsetsByFormat<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream_>>>(
			d_nbrChunkOffsetsScratch,
			d_chunkFormat,
			static_cast<uint8_t>(GPU01::NbrChunkFormat::Abs32),
			N_chunks,
			d_chunkOffsets);
		GPU::CheckLastKernel("K_SelectChunkOffsetsByFormat(Abs32)");
		GPU::Copy_DeviceToHost(&h_totalAbs32,
			d_nbrChunkOffsetsScratch + N_chunks, 1, stream_);
	};

	uint64_t h_delta8FF = 0;
	uint64_t h_delta16FF = 0;
	uint64_t h_abs32FF = 0;
	uint64_t h_delta8FB = 0;
	uint64_t h_delta16FB = 0;
	uint64_t h_abs32FB = 0;

	scanAndSelectOffsets(
		d_nbrChunkMaxCountFF_.Data(),
		d_nbrChunkFormatFF_.Data(),
		d_nbrChunkOffsetsFF_.Data(),
		h_delta8FF, h_delta16FF, h_abs32FF);

	K_BuildSellChunkCompressionMeta<<<N_GRIDSIZE, N_BLOCKSIZE, 0, stream_>>>(
		d_nbrChunkMaxCountFB_.Data(),
		d_nbrChunkMinRefFB,
		d_nbrChunkMaxRefFB,
		N_chunks,
		d_nbrChunkBaseIdxFB_.Data(),
		d_nbrChunkFormatFB_.Data());
	GPU::CheckLastKernel("K_BuildSellChunkCompressionMeta(FB)");

	scanAndSelectOffsets(
		d_nbrChunkMaxCountFB_.Data(),
		d_nbrChunkFormatFB_.Data(),
		d_nbrChunkOffsetsFB_.Data(),
		h_delta8FB, h_delta16FB, h_abs32FB);

	// actual total neighbor counts remain the sum of per-particle counts
	N_totalFF_ = thrust::transform_reduce(
		thrust::cuda::par.on(stream_),
		thrust::device_pointer_cast(d_nbrCountFF_.Data()),
		thrust::device_pointer_cast(d_nbrCountFF_.Data() + N_F_),
		ToU64{},
		uint64_t{ 0 },
		thrust::plus<uint64_t>{}
	);

	N_totalFB_ = thrust::transform_reduce(
		thrust::cuda::par.on(stream_),
		thrust::device_pointer_cast(d_nbrCountFB_.Data()),
		thrust::device_pointer_cast(d_nbrCountFB_.Data() + N_F_),
		ToU64{},
		uint64_t{ 0 },
		thrust::plus<uint64_t>{}
	);

	GPU::StreamSync(stream_);

	N_delta8FF_ = h_delta8FF;
	N_delta8FB_ = h_delta8FB;
	N_delta16FF_ = h_delta16FF;
	N_delta16FB_ = h_delta16FB;
	N_abs32FF_ = h_abs32FF;
	N_abs32FB_ = h_abs32FB;

	N_packedFF_ = N_delta8FF_ * sizeof(uint8_t)
				+ N_delta16FF_ * sizeof(uint16_t)
				+ N_abs32FF_ * sizeof(uint32_t);
	N_packedFB_ = N_delta8FB_ * sizeof(uint8_t)
				+ N_delta16FB_ * sizeof(uint16_t)
				+ N_abs32FB_ * sizeof(uint32_t);

	if (N_delta8FF_ > d_nbrListDelta8FF_.Size())
	{
		const uint64_t capTarget = N_delta8FF_
								 + Base::ExtraPercent(N_delta8FF_, extraAllocCapacityPct_);
		d_nbrListDelta8FF_.Resize(capTarget);
	}
	if (N_delta16FF_ > d_nbrListDelta16FF_.Size())
	{
		const uint64_t capTarget = N_delta16FF_
								 + Base::ExtraPercent(N_delta16FF_, extraAllocCapacityPct_);
		d_nbrListDelta16FF_.Resize(capTarget);
	}
	if (N_abs32FF_ > d_nbrListAbs32FF_.Size())
	{
		const uint64_t capTarget = N_abs32FF_
								 + Base::ExtraPercent(N_abs32FF_, extraAllocCapacityPct_);
		d_nbrListAbs32FF_.Resize(capTarget);
	}
	if (N_delta8FB_ > d_nbrListDelta8FB_.Size())
	{
		const uint64_t capTarget = N_delta8FB_
								 + Base::ExtraPercent(N_delta8FB_, extraAllocCapacityPct_);
		d_nbrListDelta8FB_.Resize(capTarget);
	}
	if (N_delta16FB_ > d_nbrListDelta16FB_.Size())
	{
		const uint64_t capTarget = N_delta16FB_
								 + Base::ExtraPercent(N_delta16FB_, extraAllocCapacityPct_);
		d_nbrListDelta16FB_.Resize(capTarget);
	}
	if (N_abs32FB_ > d_nbrListAbs32FB_.Size())
	{
		const uint64_t capTarget = N_abs32FB_
								 + Base::ExtraPercent(N_abs32FB_, extraAllocCapacityPct_);
		d_nbrListAbs32FB_.Resize(capTarget);
	}

	GPU::StreamSync(stream_);
}

// -------------------------------------------------------------------------------------------------
// --- [ brute force implementation ] --------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_BruteForce_GPU01::Init(
	const SimParams& params)
{
	(void)params;

	// nothing to do
}

void NS_BruteForce_GPU01::Construct(
	const SimParams& params)
{
	GPU01::NS_ReloadConstants(
		be_.N_F_, be_.N_B_, params.kernelRadius, 0.0f, 1u,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, be_.stream_);

	GPU::StreamSync(be_.stream_);
}

void NS_BruteForce_GPU01::Query_Count(
	const SimParams& params)
{
	(void)params;

	const NS_FinalizeScratchLayout scratchLayout = BuildFinalizeScratchLayout(be_.N_F_);
	be_.EnsureNsSharedScratchCapacity(scratchLayout.totalBytes);
	uint32_t* const d_nbrChunkMinRefFF = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.chunkMinRefFF);
	uint32_t* const d_nbrChunkMinRefFB = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.chunkMinRefFB);
	uint32_t* const d_nbrChunkMaxRefFF = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.chunkMaxRefFF);
	uint32_t* const d_nbrChunkMaxRefFB = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.chunkMaxRefFB);

	const uint32_t N_chunks = SellChunkNumTotal(be_.N_F_);
	GPU::MemsetZero(be_.d_nbrCountFF_.Data(), be_.N_F_, be_.stream_);
	GPU::MemsetZero(be_.d_nbrCountFB_.Data(), be_.N_F_, be_.stream_);
	GPU::MemsetBytes(d_nbrChunkMinRefFF, 0xff, N_chunks, be_.stream_);
	GPU::MemsetBytes(d_nbrChunkMinRefFB, 0xff, N_chunks, be_.stream_);
	GPU::MemsetZero(d_nbrChunkMaxRefFF, N_chunks, be_.stream_);
	GPU::MemsetZero(d_nbrChunkMaxRefFB, N_chunks, be_.stream_);

	GPU01::Launch_BruteForce_Query_Count(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		be_.N_F_, be_.N_B_, be_.stream_,
		be_.d_nbrCountFF_.Data(), be_.d_nbrCountFB_.Data(),
		d_nbrChunkMinRefFF, d_nbrChunkMaxRefFF,
		d_nbrChunkMinRefFB, d_nbrChunkMaxRefFB);

	GPU::StreamSync(be_.stream_);
}

void NS_BruteForce_GPU01::Query_Fill(
	const SimParams& params)
{
	// reset overflow before filling packed SELL32 lists
	GPU::MemsetZero(be_.d_nbrOverflowFlag_.Data(), 1, be_.stream_);

	GPU01::Launch_BruteForce_Query_Fill(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		be_.N_F_, be_.N_B_, be_.stream_,
		be_.d_nbrChunkMaxCountFF_.Data(), be_.d_nbrChunkMaxCountFB_.Data(),
		be_.d_nbrChunkOffsetsFF_.Data(), be_.d_nbrChunkOffsetsFB_.Data(),
		be_.d_nbrChunkBaseIdxFF_.Data(), be_.d_nbrChunkBaseIdxFB_.Data(),
		be_.d_nbrChunkFormatFF_.Data(), be_.d_nbrChunkFormatFB_.Data(),
		be_.d_nbrListDelta8FF_.Data(),
		be_.d_nbrListDelta8FB_.Data(),
		be_.d_nbrListDelta16FF_.Data(),
		be_.d_nbrListDelta16FB_.Data(),
		be_.d_nbrListAbs32FF_.Data(),
		be_.d_nbrListAbs32FB_.Data(),
		be_.d_nbrOverflowFlag_.Data());

	#if defined(_DEBUG)
	CheckNbrListErrorFlag(be_.d_nbrOverflowFlag_, be_.stream_,
		"GPU brute force SELL32 fill-query");
	#else
	GPU::StreamSync(be_.stream_);
	#endif
}

void NS_BruteForce_GPU01::DataReOrder(
	const SimParams& params)
{
	(void)params;

	// no re-ordering in brute force implementation
}

std::pair<Mem::MemUsage, Mem::MemUsage> NS_BruteForce_GPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

	Mem::Add(usageF, MemUsageOf_DeviceBuffer(be_.d_nsSharedScratch_));

	return { usageF, usageB };
}

// -------------------------------------------------------------------------------------------------
// --- [ cell linked list compact implementation ] -------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_CellLinkedList_Compact_GPU01::Init(const SimParams& params)
{
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;
	cellQueryReach_ = 1;
	N_NonEmptyCellsF_ = 0;
	N_NonEmptyCellsB_ = 0;
	N_NonEmptyCellsAny_ = 0;
	N_queryCellSlots_ = 0;
	particlesMatchCurrentCellOrder_ = false;
	cellIdsAvailableForReuse_ = false;

	d_uniqueCellIdF_.Release();
	d_cellStartIdxF_.Release();
	d_sortedParticleIdxF_.Release();

	d_uniqueCellIdB_.Release();
	d_cellStartIdxB_.Release();

	d_uniqueCellIdAny_.Release();
	d_cellRefFByAny_.Release();
	d_cellRefBByAny_.Release();
	d_queryCellRefAnyByCellF_.Release();
}

void NS_CellLinkedList_Compact_GPU01::Construct(const SimParams& params)
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;

	if (N_F == 0 && N_B == 0)
	{
		std::cerr << "Warning: Cannot construct neighbor search for an empty simulation.\n";

		N_NonEmptyCellsF_ = 0;
		N_NonEmptyCellsB_ = 0;
		N_NonEmptyCellsAny_ = 0;
		N_queryCellSlots_ = 0;
		particlesMatchCurrentCellOrder_ = false;
		cellIdsAvailableForReuse_ = false;

		d_uniqueCellIdF_.Release();
		d_cellStartIdxF_.Release();
		d_sortedParticleIdxF_.Release();

		d_uniqueCellIdB_.Release();
		d_cellStartIdxB_.Release();

		d_uniqueCellIdAny_.Release();
		d_cellRefFByAny_.Release();
		d_cellRefBByAny_.Release();
		d_queryCellRefAnyByCellF_.Release();

		return;
	}

	cellSize_ = params.kernelRadius * params.gridCellSizeRatio;
	cellSizeInv_ = Base::ApproxEqu(cellSize_, 0.0f)
				 ? 0.0f : (1.0f / cellSize_);
	const float querySpanCells = Base::ApproxEqu(cellSize_, 0.0f)
		? 0.0f : (params.kernelRadius / cellSize_);
	cellQueryReach_ = Base::ApproxEqu(cellSize_, 0.0f)
		? 1u : std::max<uint32_t>(1u,
			static_cast<uint32_t>(std::ceil(std::max(0.0f, querySpanCells - 1e-6f))));

	EnsureStorageAllocation();
	const NS_CompactSharedScratchLayout scratchLayout =
		BuildCompactSharedScratchLayout(N_F, N_B);
	uint64_t* const d_cellIdF = ScratchPtr<uint64_t>(be_.d_nsSharedScratch_, scratchLayout.build.cellIdF);
	uint64_t* const d_cellIdB = ScratchPtr<uint64_t>(be_.d_nsSharedScratch_, scratchLayout.build.cellIdB);
	uint32_t* const d_sortedParticleIdxB = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.build.sortedParticleIdxB);

	GPU01::NS_ReloadConstants(
		be_.N_F_, be_.N_B_, params.kernelRadius, cellSize_, cellQueryReach_,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, be_.stream_);

	// if DataReOrder() already computed and permuted the per-particle cell IDs for this step,
	// then they can be reused directly here and no re-computation is needed
	const bool canReuseCellIds =
		particlesMatchCurrentCellOrder_ && cellIdsAvailableForReuse_;

	if (!canReuseCellIds)
	{
		cellIdsAvailableForReuse_ = false;

		// --- [ cell coordinates and indices for fluid and boundary particles ] ---------------------

		if (N_F > 0) // fluid
		{
			GPU01::Launch_FillCellIdAndInitParticleIdx(
				be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
				N_F, be_.stream_, d_cellIdF, d_sortedParticleIdxF_.Data());
		}

		if (N_B > 0) // boundary
		{
			GPU01::Launch_FillCellIdAndInitParticleIdx(
				be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
				N_B, be_.stream_, d_cellIdB, d_sortedParticleIdxB);
		}

		// --- [ sort particle indices by cell IDs ] -------------------------------------------------

		if (N_F > 0) // fluid
		{
			thrust::device_ptr<uint64_t> keysF(d_cellIdF);
			thrust::device_ptr<uint32_t> valsF(d_sortedParticleIdxF_.Data());

			// using radix sort from thrust
			thrust::sort_by_key(thrust::cuda::par.on(be_.stream_),
				keysF, keysF + N_F,valsF);
		}

		if (N_B > 0) // boundary
		{
			thrust::device_ptr<uint64_t> keysB(d_cellIdB);
			thrust::device_ptr<uint32_t> valsB(d_sortedParticleIdxB);

			// using radix sort from thrust
			thrust::sort_by_key(thrust::cuda::par.on(be_.stream_),
				keysB, keysB + N_B, valsB);
		}
	}

	// --- [ build compact cell list and fill in cell start indices ] --------------------------------

	if (N_F > 0) // fluid
	{
		BuildCompactCellList(d_uniqueCellIdF_, d_cellStartIdxF_, d_sortedParticleIdxF_.Data(),
			d_cellIdF,
			N_F, N_NonEmptyCellsF_);
	}
	else
	{
		N_NonEmptyCellsF_ = 0;
	}

	if (N_B > 0) // boundary
	{
		BuildCompactCellList(d_uniqueCellIdB_, d_cellStartIdxB_, d_sortedParticleIdxB,
			d_cellIdB,
			N_B, N_NonEmptyCellsB_);
	}
	else
	{
		N_NonEmptyCellsB_ = 0;
	}

	BuildMergedCompactCellLookup();
	BuildQueryLookupBounds();
	BuildQueryCellRefCache();

	// the compact cell IDs live in shared scratch and are no longer considered reusable
	// after construction completes, since later query/finalize phases alias that storage
	cellIdsAvailableForReuse_ = false;
}

void NS_CellLinkedList_Compact_GPU01::Query_Count(const SimParams& params)
{
	(void)params;

	if (!particlesMatchCurrentCellOrder_)
	{
		throw std::runtime_error(
			"GPU compact CLL query requires particle arrays to match the compact cell ordering");
	}

	const NS_CompactSharedScratchLayout scratchLayout =
		BuildCompactSharedScratchLayout(be_.N_F_, be_.N_B_);
	uint32_t* const d_nbrChunkMinRefFF = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.finalize.chunkMinRefFF);
	uint32_t* const d_nbrChunkMinRefFB = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.finalize.chunkMinRefFB);
	uint32_t* const d_nbrChunkMaxRefFF = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.finalize.chunkMaxRefFF);
	uint32_t* const d_nbrChunkMaxRefFB = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.finalize.chunkMaxRefFB);

	// reset per-particle counts before count-query
	const uint32_t N_chunks = SellChunkNumTotal(be_.N_F_);
	GPU::MemsetZero(be_.d_nbrCountFF_.Data(), be_.N_F_, be_.stream_);
	GPU::MemsetZero(be_.d_nbrCountFB_.Data(), be_.N_F_, be_.stream_);
	GPU::MemsetBytes(d_nbrChunkMinRefFF, 0xff, N_chunks, be_.stream_);
	GPU::MemsetBytes(d_nbrChunkMinRefFB, 0xff, N_chunks, be_.stream_);
	GPU::MemsetZero(d_nbrChunkMaxRefFF, N_chunks, be_.stream_);
	GPU::MemsetZero(d_nbrChunkMaxRefFB, N_chunks, be_.stream_);

	GPU01::Launch_CellLinkedList_Compact_Query_Count(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		d_cellRefFByAny_.Data(), d_cellRefBByAny_.Data(),
		d_cellStartIdxF_.Data(), d_cellStartIdxB_.Data(),
		d_sortedParticleIdxF_.Data(),
		d_queryCellRefAnyByCellF_.Data(), N_queryCellSlots_,
		be_.N_F_, be_.N_B_, be_.stream_,
		be_.d_nbrCountFF_.Data(), be_.d_nbrCountFB_.Data(),
		d_nbrChunkMinRefFF, d_nbrChunkMaxRefFF,
		d_nbrChunkMinRefFB, d_nbrChunkMaxRefFB);

	GPU::StreamSync(be_.stream_);
}

void NS_CellLinkedList_Compact_GPU01::Query_Fill(const SimParams& params)
{
	(void)params;

	if (!particlesMatchCurrentCellOrder_)
	{
		throw std::runtime_error(
			"GPU compact CLL query requires particle arrays to match the compact cell ordering");
	}

	// reset overflow before filling packed SELL32 lists
	GPU::MemsetZero(be_.d_nbrOverflowFlag_.Data(), 1, be_.stream_);

	GPU01::Launch_CellLinkedList_Compact_Query_Fill(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		d_cellRefFByAny_.Data(), d_cellRefBByAny_.Data(),
		d_cellStartIdxF_.Data(), d_cellStartIdxB_.Data(),
		d_sortedParticleIdxF_.Data(),
		d_queryCellRefAnyByCellF_.Data(), N_queryCellSlots_,
		be_.N_F_, be_.N_B_, be_.stream_,
		be_.d_nbrChunkMaxCountFF_.Data(), be_.d_nbrChunkMaxCountFB_.Data(),
		be_.d_nbrChunkOffsetsFF_.Data(), be_.d_nbrChunkOffsetsFB_.Data(),
		be_.d_nbrChunkBaseIdxFF_.Data(), be_.d_nbrChunkBaseIdxFB_.Data(),
		be_.d_nbrChunkFormatFF_.Data(), be_.d_nbrChunkFormatFB_.Data(),
		be_.d_nbrListDelta8FF_.Data(),
		be_.d_nbrListDelta8FB_.Data(),
		be_.d_nbrListDelta16FF_.Data(),
		be_.d_nbrListDelta16FB_.Data(),
		be_.d_nbrListAbs32FF_.Data(),
		be_.d_nbrListAbs32FB_.Data(),
		be_.d_nbrOverflowFlag_.Data());

	#if defined(_DEBUG)
	CheckNbrListErrorFlag(be_.d_nbrOverflowFlag_, be_.stream_,
		"GPU compact CLL SELL32 fill-query");
	#else
	GPU::StreamSync(be_.stream_);
	#endif
}

void NS_CellLinkedList_Compact_GPU01::DataReOrder(const SimParams& params)
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const uint32_t N_MAX = std::max(N_F, N_B);

	if (N_MAX == 0)
	{
		particlesMatchCurrentCellOrder_ = false;
		cellIdsAvailableForReuse_ = false;
		N_queryCellSlots_ = 0;
		return;
	}

	particlesMatchCurrentCellOrder_ = false;
	cellIdsAvailableForReuse_ = false;

	cellSize_ = params.kernelRadius * params.gridCellSizeRatio;
	cellSizeInv_ = Base::ApproxEqu(cellSize_, 0.0f) ? 0.0f : (1.0f / cellSize_);
	const float querySpanCells = Base::ApproxEqu(cellSize_, 0.0f)
		? 0.0f : (params.kernelRadius / cellSize_);
	cellQueryReach_ = Base::ApproxEqu(cellSize_, 0.0f)
		? 1u : std::max<uint32_t>(1u,
		static_cast<uint32_t>(std::ceil(std::max(0.0f, querySpanCells - 1e-6f))));

	EnsureStorageAllocation();
	const NS_CompactSharedScratchLayout scratchLayout =
		BuildCompactSharedScratchLayout(N_F, N_B);
	uint64_t* const d_cellIdF = ScratchPtr<uint64_t>(be_.d_nsSharedScratch_, scratchLayout.build.cellIdF);
	uint64_t* const d_cellIdB = ScratchPtr<uint64_t>(be_.d_nsSharedScratch_, scratchLayout.build.cellIdB);
	uint32_t* const d_sortedParticleIdxB = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.build.sortedParticleIdxB);

	GPU01::NS_ReloadConstants(
		be_.N_F_, be_.N_B_, params.kernelRadius, cellSize_, cellQueryReach_,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, be_.stream_);

	// build the exact compact-CLL cell IDs and particle permutation for fluid particles
	if (N_F > 0)
	{
		GPU01::Launch_FillCellIdAndInitParticleIdx(
			be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
			N_F, be_.stream_, d_cellIdF, d_sortedParticleIdxF_.Data());

		thrust::device_ptr<uint64_t> keysF(d_cellIdF);
		thrust::device_ptr<uint32_t> valsF(d_sortedParticleIdxF_.Data());

		// using radix sort from thrust
		thrust::sort_by_key(thrust::cuda::par.on(be_.stream_),
			keysF, keysF + N_F, valsF);
	}

	// build the exact compact-CLL cell IDs and particle permutation for boundary particles
	if (N_B > 0)
	{
		GPU01::Launch_FillCellIdAndInitParticleIdx(
			be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
			N_B, be_.stream_, d_cellIdB, d_sortedParticleIdxB);

		thrust::device_ptr<uint64_t> keysB(d_cellIdB);
		thrust::device_ptr<uint32_t> valsB(d_sortedParticleIdxB);

		// using radix sort from thrust
		thrust::sort_by_key(thrust::cuda::par.on(be_.stream_),
			keysB, keysB + N_B, valsB);
	}

	// density array as temp scratch storage (only needed after the particle arrays are re-ordered)
	DeviceBuffer<float>& d_reorderScratch = be_.d_densityF_;

	// re-order fluid SoA arrays according to sorted fluid indices
	if (N_F > 0)
	{
		const uint32_t* d_permF = d_sortedParticleIdxF_.Data();

		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_idF_, be_.d_reorderIdScratch_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_xPosF_, d_reorderScratch, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_yPosF_, d_reorderScratch, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_zPosF_, d_reorderScratch, be_.stream_);

		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_xVelF_, d_reorderScratch, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_yVelF_, d_reorderScratch, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_zVelF_, d_reorderScratch, be_.stream_);

		// remaining particle arrays do not need to be re-ordered, since they are re-initialized completely
		// (d_xyzVelPredF_, d_xyzAccPressF_, d_sourceTermF_, d_diagElemF_, d_pressureF_, d_densityF_)
		if (false) // TODO: execute only if no full sim step is performed, e.g. neighbor search 0 before sim run?
		{
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_xAccPressF_, d_reorderScratch, be_.stream_);
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_yAccPressF_, d_reorderScratch, be_.stream_);
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_zAccPressF_, d_reorderScratch, be_.stream_);

			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_xVelPredF_, d_reorderScratch, be_.stream_);
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_yVelPredF_, d_reorderScratch, be_.stream_);
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_zVelPredF_, d_reorderScratch, be_.stream_);

			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_sourceTermF_, d_reorderScratch, be_.stream_);
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_diagElemF_, d_reorderScratch, be_.stream_);
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_pressureF_, d_reorderScratch, be_.stream_);
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_densityF_, d_reorderScratch, be_.stream_);
		}

		// fluid arrays are now in sorted order, so sorted particle index becomes identity
		thrust::sequence(
			thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_sortedParticleIdxF_.Data()),
			thrust::device_pointer_cast(d_sortedParticleIdxF_.Data() + N_F),
			0u);
	}

	// re-order boundary SoA arrays according to sorted boundary indices
	if (N_B > 0)
	{
		const uint32_t* d_permB = d_sortedParticleIdxB;

		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_idB_, be_.d_reorderIdScratch_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_bodyIdB_, be_.d_reorderIdScratch_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_rigidBoundaryParticleIdxB_, be_.d_reorderIdScratch_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_xPosB_, d_reorderScratch, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_yPosB_, d_reorderScratch, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_zPosB_, d_reorderScratch, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_xVelB_, d_reorderScratch, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_yVelB_, d_reorderScratch, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_zVelB_, d_reorderScratch, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_volB_, d_reorderScratch, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_psiB_, d_reorderScratch, be_.stream_);

		// boundary arrays are now in sorted order, so sorted particle index becomes identity
		thrust::sequence(
			thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_sortedParticleIdxB),
			thrust::device_pointer_cast(d_sortedParticleIdxB + N_B),
			0u);
	}

	// keep the exact compact-CLL cell IDs in sync with the re-ordered particle arrays,
	// so Construct() can reuse them instead of re-computing them right away
	particlesMatchCurrentCellOrder_ = true;
	cellIdsAvailableForReuse_ = true;

	GPU::StreamSync(be_.stream_);
}

std::pair<Mem::MemUsage, Mem::MemUsage> NS_CellLinkedList_Compact_GPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_uniqueCellIdF_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_cellStartIdxF_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_sortedParticleIdxF_));

	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_uniqueCellIdB_));
	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_cellStartIdxB_));

	// shared merged lookup metadata is accounted in the fluid-side bucket to avoid double-counting
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_uniqueCellIdAny_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_cellRefFByAny_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_cellRefBByAny_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_queryCellRefAnyByCellF_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(be_.d_nsSharedScratch_));

	// shared scan scratch is accounted in the fluid-side bucket to avoid double-counting

	return { usageF, usageB };
}

void NS_CellLinkedList_Compact_GPU01::EnsureStorageAllocation()
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;

	if (N_F > 0)
	{
		// keep capacity once allocated and only grow when needed
		// (to avoid repeated device alloc/free between steps)
		// TODO: use heuristic instead of allocating worst-case size?
		if (d_uniqueCellIdF_.Size() < N_F) { d_uniqueCellIdF_.Resize(N_F); }
		if (d_cellStartIdxF_.Size() < N_F + 1) { d_cellStartIdxF_.Resize(N_F + 1); }
		if (d_sortedParticleIdxF_.Size() < N_F + 1) { d_sortedParticleIdxF_.Resize(N_F + 1); }
	}

	if (N_B > 0)
	{
		// keep capacity once allocated and only grow when needed
		// (to avoid repeated device alloc/free between steps)
		// TODO: use heuristic instead of allocating worst-case size?
		if (d_uniqueCellIdB_.Size() < N_B) { d_uniqueCellIdB_.Resize(N_B); }
		if (d_cellStartIdxB_.Size() < N_B + 1) { d_cellStartIdxB_.Resize(N_B + 1); }
	}

	const uint32_t N_ANY_MAX = N_F + N_B;
	if (N_ANY_MAX > 0)
	{
		// keep capacity once allocated and only grow when needed
		// (to avoid repeated device alloc/free between steps)
		if (d_uniqueCellIdAny_.Size() < N_ANY_MAX) { d_uniqueCellIdAny_.Resize(N_ANY_MAX); }
		if (d_cellRefFByAny_.Size() < N_ANY_MAX) { d_cellRefFByAny_.Resize(N_ANY_MAX); }
		if (d_cellRefBByAny_.Size() < N_ANY_MAX) { d_cellRefBByAny_.Resize(N_ANY_MAX); }
	}

	be_.EnsureNsSharedScratchCapacity(BuildCompactSharedScratchLayout(N_F, N_B).totalBytes);
}

void NS_CellLinkedList_Compact_GPU01::BuildCompactCellList(
	DeviceBuffer<uint64_t>& d_uniqueCellId,
	DeviceBuffer<uint32_t>& d_cellStartIdx,
	uint32_t* d_markerScanScratch,
	const uint64_t* d_cellIdSorted,
	uint32_t N, uint32_t& N_NonEmptyCells)
{
	if (N == 0)
	{
		N_NonEmptyCells = 0;
		return;
	}

	// mark cell starts in the caller-provided marker/scan scratch buffer
	GPU01::Launch_MarkCellStarts(
		d_cellIdSorted, N, be_.stream_, d_markerScanScratch);

	// inclusive scan in-place to get cell indices for each particle
	{
		thrust::device_ptr<uint32_t> scanBegin(d_markerScanScratch);
		thrust::device_ptr<uint32_t> scanEnd(d_markerScanScratch + N);

		thrust::inclusive_scan(thrust::cuda::par.on(be_.stream_),
			scanBegin, scanEnd, scanBegin);
	}

	// get last scan value to determine number of unique cells
	uint32_t lastScan = 0;
	GPU::Check(cudaMemcpyAsync(&lastScan,
		d_markerScanScratch + (N - 1),
		sizeof(uint32_t),
		cudaMemcpyDeviceToHost,
		be_.stream_));
	GPU::StreamSync(be_.stream_);
	N_NonEmptyCells = lastScan;

	// fill in cell-related data
	// (write only first N_NonEmptyCells entries and one sentinel entry)
	GPU01::Launch_FillUniqueCellIdAndCellStartIdx(
		d_cellIdSorted, d_markerScanScratch,
		N, N_NonEmptyCells, be_.stream_,
		d_uniqueCellId.Data(), d_cellStartIdx.Data());
}

void NS_CellLinkedList_Compact_GPU01::BuildMergedCompactCellLookup()
{
	N_NonEmptyCellsAny_ = 0;

	if (N_NonEmptyCellsF_ == 0 && N_NonEmptyCellsB_ == 0) { return; }

	const auto beginAny = thrust::device_pointer_cast(d_uniqueCellIdAny_.Data());
	auto endAny = beginAny;

	if (N_NonEmptyCellsF_ == 0)
	{
		endAny = thrust::copy(
			thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_uniqueCellIdB_.Data()),
			thrust::device_pointer_cast(d_uniqueCellIdB_.Data() + N_NonEmptyCellsB_),
			beginAny);
	}
	else if (N_NonEmptyCellsB_ == 0)
	{
		endAny = thrust::copy(
			thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_uniqueCellIdF_.Data()),
			thrust::device_pointer_cast(d_uniqueCellIdF_.Data() + N_NonEmptyCellsF_),
			beginAny);
	}
	else
	{
		endAny = thrust::set_union(
			thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_uniqueCellIdF_.Data()),
			thrust::device_pointer_cast(d_uniqueCellIdF_.Data() + N_NonEmptyCellsF_),
			thrust::device_pointer_cast(d_uniqueCellIdB_.Data()),
			thrust::device_pointer_cast(d_uniqueCellIdB_.Data() + N_NonEmptyCellsB_),
			beginAny);
	}

	N_NonEmptyCellsAny_ = static_cast<uint32_t>(endAny - beginAny);

	if (N_NonEmptyCellsAny_ == 0) { return; }

	GPU01::Launch_FillMergedCompactCellRefs(
		d_uniqueCellIdAny_.Data(), N_NonEmptyCellsAny_,
		d_uniqueCellIdF_.Data(), d_uniqueCellIdB_.Data(),
		N_NonEmptyCellsF_, N_NonEmptyCellsB_, be_.stream_,
		d_cellRefFByAny_.Data(), d_cellRefBByAny_.Data());
}

void NS_CellLinkedList_Compact_GPU01::BuildQueryLookupBounds()
{
	if (N_NonEmptyCellsF_ == 0 || N_NonEmptyCellsAny_ == 0)
	{
		return;
	}

	const NS_CompactSharedScratchLayout scratchLayout =
		BuildCompactSharedScratchLayout(be_.N_F_, be_.N_B_);
	uint32_t* const d_queryLoAnyByCellF = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.queryLoAnyByCellF);
	uint32_t* const d_queryHiAnyByCellF = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.queryHiAnyByCellF);

	GPU01::Launch_FillQueryLookupBoundsF(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		d_cellStartIdxF_.Data(), d_uniqueCellIdAny_.Data(),
		N_NonEmptyCellsF_, N_NonEmptyCellsAny_, cellQueryReach_, be_.stream_,
		d_sortedParticleIdxF_.Data(),
		d_queryLoAnyByCellF, d_queryHiAnyByCellF);
}

void NS_CellLinkedList_Compact_GPU01::BuildQueryCellRefCache()
{
	N_queryCellSlots_ = QueryCellSlotCount(cellQueryReach_);

	if (N_NonEmptyCellsF_ == 0 || N_NonEmptyCellsAny_ == 0)
	{
		return;
	}

	const uint64_t N_queryCellRefs = static_cast<uint64_t>(N_NonEmptyCellsF_) * N_queryCellSlots_;
	if (N_queryCellRefs > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) [[unlikely]]
	{
		throw std::runtime_error("GPU compact CLL query-cell cache too large.");
	}
	if (d_queryCellRefAnyByCellF_.Size() < static_cast<size_t>(N_queryCellRefs))
	{
		if (be_.extraAllocCapacityPct_ != 0)
		{
			const uint64_t maxCountWithExtra = (std::numeric_limits<uint64_t>::max() - 99u) / be_.extraAllocCapacityPct_;
			if (N_queryCellRefs > maxCountWithExtra) [[unlikely]]
			{
				throw std::runtime_error("GPU compact CLL query-cell cache too large.");
			}
		}

		const uint64_t capTarget = N_queryCellRefs
								 + Base::ExtraPercent(N_queryCellRefs, be_.extraAllocCapacityPct_);
		if (capTarget > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) [[unlikely]]
		{
			throw std::runtime_error("GPU compact CLL query-cell cache too large.");
		}

		d_queryCellRefAnyByCellF_.Resize(static_cast<size_t>(capTarget));
	}

	const NS_CompactSharedScratchLayout scratchLayout =
		BuildCompactSharedScratchLayout(be_.N_F_, be_.N_B_);
	const uint32_t* const d_queryLoAnyByCellF = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.queryLoAnyByCellF);
	const uint32_t* const d_queryHiAnyByCellF = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.queryHiAnyByCellF);

	GPU01::Launch_FillQueryCellRefsByCellF(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		d_cellStartIdxF_.Data(), d_uniqueCellIdAny_.Data(),
		d_queryLoAnyByCellF, d_queryHiAnyByCellF,
		N_NonEmptyCellsF_, N_queryCellSlots_, be_.stream_,
		d_queryCellRefAnyByCellF_.Data());
}
