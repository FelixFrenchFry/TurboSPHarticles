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
#include "source/base/helpers.h"
#include "source/simulation/GPU/gpu_utils.h"



using namespace Sim;

namespace
{
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

			return static_cast<uint64_t>(thrust::get<0>(t)) * GPU::N_SELL_WIDTH;
		}

		uint8_t targetFormat_;
	};

	__host__ __device__ inline uint32_t SellChunkNumTotal(const uint32_t N_F) noexcept
	{
		return (N_F == 0u) ? 0u : ((N_F - 1u) / GPU::N_SELL_WIDTH) + 1u;
	}

	inline uint32_t QueryCellSlotCount(const uint32_t cellQueryReach)
	{
		const uint64_t N_queryWidth =
			2ull * static_cast<uint64_t>(cellQueryReach) + 1ull;
		constexpr uint64_t N_maxSlots =
			static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
		if (N_queryWidth > N_maxSlots / N_queryWidth ||
			N_queryWidth * N_queryWidth > N_maxSlots / N_queryWidth) [[unlikely]]
		{
			throw std::runtime_error("GPU compact CLL query-cell cache slot count overflow.");
		}

		const uint64_t N_slots = N_queryWidth * N_queryWidth * N_queryWidth;
		return static_cast<uint32_t>(N_slots);
	}

	inline size_t AlignUpBytes(const size_t offset, const size_t align) noexcept
	{
		return (offset + align - 1u) & ~(align - 1u);
	}

	template <typename T>
	inline size_t ReserveScratchElems(size_t& cursor, const size_t N_elements) noexcept
	{
		cursor = AlignUpBytes(cursor, alignof(T));
		const size_t offset = cursor;
		cursor += N_elements * sizeof(T);

		return offset;
	}

	struct NS_FinalizeScratchLayout
	{
		size_t chunkOffsetsScratch = 0;
		size_t N_totalBytes = 0;
	};

	struct NS_CompactBuildScratchLayout
	{
		size_t cellIdF = 0;
		size_t cellIdB = 0;
		size_t sortedParticleIdxB = 0;
		size_t N_totalBytes = 0;
	};

	struct NS_CompactSharedScratchLayout
	{
		NS_FinalizeScratchLayout finalize;
		NS_CompactBuildScratchLayout build;
		size_t N_phaseScratchBytes = 0;
		size_t queryLoAnyByCellF = 0;
		size_t queryHiAnyByCellF = 0;
		size_t N_totalBytes = 0;
	};

	inline NS_FinalizeScratchLayout BuildFinalizeScratchLayout(const uint32_t N_F)
	{
		NS_FinalizeScratchLayout layout{};
		size_t cursor = 0;
		const uint32_t N_chunks = SellChunkNumTotal(N_F);

		layout.chunkOffsetsScratch = ReserveScratchElems<uint64_t>(
			cursor, static_cast<size_t>(N_chunks) + 1ull);
		layout.N_totalBytes = cursor;

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
		layout.N_totalBytes = cursor;

		return layout;
	}

	inline NS_CompactSharedScratchLayout BuildCompactSharedScratchLayout(
		const uint32_t N_F, const uint32_t N_B)
	{
		NS_CompactSharedScratchLayout layout{};
		layout.finalize = BuildFinalizeScratchLayout(N_F);
		layout.build = BuildCompactBuildScratchLayout(N_F, N_B);
		layout.N_phaseScratchBytes =
			std::max(layout.finalize.N_totalBytes, layout.build.N_totalBytes);

		size_t cursor = layout.N_phaseScratchBytes;
		layout.queryLoAnyByCellF = ReserveScratchElems<uint32_t>(cursor, N_F);
		layout.queryHiAnyByCellF = ReserveScratchElems<uint32_t>(cursor, N_F);
		layout.N_totalBytes = cursor;

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
		if (errorFlag & GPU::NbrListError_FillOverflow)
		{
			msg += "; fill count exceeded chunk pitch";
		}
		if (errorFlag & GPU::NbrListError_InvalidChunkFormat)
		{
			msg += "; invalid compressed SELL32 chunk format";
		}
		if (errorFlag & GPU::NbrListError_DeltaRangeOverflow)
		{
			msg += "; neighbor index does not fit selected delta format";
		}

		throw std::runtime_error(msg + ".");
	}
	#endif

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
		const uint64_t N_bytes = static_cast<uint64_t>(src.Size()) * sizeof(T);

		return Mem::MemUsage{ N_bytes, N_bytes };
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

		// equal-sized buffers exchange their storage instead of copying the gathered data back
		// (the re-order scratch is sized for the largest particle class, so smaller classes copy)
		if (d_src.Size() == d_tmp.Size())
		{
			d_src.Swap(d_tmp);
		}
		else
		{
			thrust::copy(
				thrust::cuda::par.on(stream),
				thrust::device_pointer_cast(d_tmp.Data()),
				thrust::device_pointer_cast(d_tmp.Data() + N),
				thrust::device_pointer_cast(d_src.Data()));
		}
	}

} // namespace anonymous

void SimBackend_GPU::FinalizeNeighborLists_Count()
{
	const uint32_t N_chunks = SellChunkNumTotal(N_F_);
	const NS_FinalizeScratchLayout scratchLayout = BuildFinalizeScratchLayout(N_F_);
	EnsureNsSharedScratchCapacity(scratchLayout.N_totalBytes);
	uint64_t* const d_nbrChunkOffsetsScratch = ScratchPtr<uint64_t>(d_nsSharedScratch_, scratchLayout.chunkOffsetsScratch);

	const uint64_t zero = 0;
	if (N_F_ == 0)
	{
		N_totalFF_ = N_totalFB_ = N_totalFR_ = 0;
		N_packedFF_ = N_packedFB_ = N_packedFR_ = 0;
		N_delta8FF_ = N_delta8FB_ = N_delta8FR_ = 0;
		N_delta16FF_ = N_delta16FB_ = N_delta16FR_ = 0;
		N_abs32FF_ = N_abs32FB_ = N_abs32FR_ = 0;
		GPU::Copy_HostToDevice(d_nbrChunkOffsetsFF_.Data(), &zero, 1, stream_);
		GPU::Copy_HostToDevice(d_nbrChunkOffsetsFB_.Data(), &zero, 1, stream_);
		GPU::Copy_HostToDevice(d_nbrChunkOffsetsFR_.Data(), &zero, 1, stream_);
	}
	else
	{
		const uint32_t N_BLOCK_SIZE = 256;
		const uint32_t N_GRID_SIZE = (N_chunks + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;

		auto finalizeSell = [&](
			const char* label,
			const auto& d_nbrCount,
			auto& d_nbrChunkMaxCount,
			auto& d_nbrChunkFormat,
			auto& d_nbrChunkOffsets,
			auto& d_nbrListDelta8,
			auto& d_nbrListDelta16,
			auto& d_nbrListAbs32,
			uint64_t& N_total,
			uint64_t& N_packed,
			uint64_t& N_delta8,
			uint64_t& N_delta16,
			uint64_t& N_abs32)
		{
			uint64_t totals[3]{};
			for (uint8_t format = 0; format < 3; format++)
			{
				GPU::Copy_HostToDevice(d_nbrChunkOffsetsScratch, &zero, 1, stream_);
				thrust::transform_inclusive_scan(
					thrust::cuda::par.on(stream_),
					thrust::make_zip_iterator(thrust::make_tuple(
						thrust::device_pointer_cast(d_nbrChunkMaxCount.Data()),
						thrust::device_pointer_cast(d_nbrChunkFormat.Data()))),
					thrust::make_zip_iterator(thrust::make_tuple(
						thrust::device_pointer_cast(d_nbrChunkMaxCount.Data() + N_chunks),
						thrust::device_pointer_cast(d_nbrChunkFormat.Data() + N_chunks))),
					thrust::device_pointer_cast(d_nbrChunkOffsetsScratch + 1),
					SellChunkPitchIfFormat(format), thrust::plus<uint64_t>{});
				K_SelectChunkOffsetsByFormat<<<N_GRID_SIZE, N_BLOCK_SIZE, 0, stream_>>>(
					d_nbrChunkOffsetsScratch, d_nbrChunkFormat.Data(), format,
					N_chunks, d_nbrChunkOffsets.Data());
				GPU::CheckLastKernel(label);
				GPU::Copy_DeviceToHost(&totals[format],
					d_nbrChunkOffsetsScratch + N_chunks, 1, stream_);
			}

			N_total = thrust::transform_reduce(
				thrust::cuda::par.on(stream_),
				thrust::device_pointer_cast(d_nbrCount.Data()),
				thrust::device_pointer_cast(d_nbrCount.Data() + N_F_),
				ToU64{}, uint64_t{ 0 }, thrust::plus<uint64_t>{});
			GPU::StreamSync(stream_);

			N_delta8 = totals[static_cast<uint8_t>(GPU::NbrChunkFormat::Delta8)];
			N_delta16 = totals[static_cast<uint8_t>(GPU::NbrChunkFormat::Delta16)];
			N_abs32 = totals[static_cast<uint8_t>(GPU::NbrChunkFormat::Abs32)];
			N_packed = N_delta8 * sizeof(uint8_t)
				+ N_delta16 * sizeof(uint16_t) + N_abs32 * sizeof(uint32_t);

			auto grow = [&](auto& buffer, const uint64_t required)
			{
				if (required <= buffer.Size()) { return; }
				buffer.Resize(required + Base::ExtraPercent(required, extraAllocCapacityPct_));
			};
			grow(d_nbrListDelta8, N_delta8);
			grow(d_nbrListDelta16, N_delta16);
			grow(d_nbrListAbs32, N_abs32);
		};

		finalizeSell("Finalize SELL32 FF", d_nbrCountFF_, d_nbrChunkMaxCountFF_,
			d_nbrChunkFormatFF_, d_nbrChunkOffsetsFF_,
			d_nbrListDelta8FF_, d_nbrListDelta16FF_, d_nbrListAbs32FF_,
			N_totalFF_, N_packedFF_, N_delta8FF_, N_delta16FF_, N_abs32FF_);
		finalizeSell("Finalize SELL32 FB", d_nbrCountFB_, d_nbrChunkMaxCountFB_,
			d_nbrChunkFormatFB_, d_nbrChunkOffsetsFB_,
			d_nbrListDelta8FB_, d_nbrListDelta16FB_, d_nbrListAbs32FB_,
			N_totalFB_, N_packedFB_, N_delta8FB_, N_delta16FB_, N_abs32FB_);
		finalizeSell("Finalize SELL32 FR", d_nbrCountFR_, d_nbrChunkMaxCountFR_,
			d_nbrChunkFormatFR_, d_nbrChunkOffsetsFR_,
			d_nbrListDelta8FR_, d_nbrListDelta16FR_, d_nbrListAbs32FR_,
			N_totalFR_, N_packedFR_, N_delta8FR_, N_delta16FR_, N_abs32FR_);
	}

	// build CSR offsets and storage for rigid source neighbors
	const auto finalizeCSR = [&](auto& d_nbrCount, auto& d_nbrOffsets,
		auto& d_nbrList, uint64_t& N_total)
		{
			GPU::Copy_HostToDevice(d_nbrOffsets.Data(), &zero, 1, stream_);
			if (N_R_ == 0)
			{
				N_total = 0;
				return;
			}

			thrust::transform_inclusive_scan(
				thrust::cuda::par.on(stream_),
				thrust::device_pointer_cast(d_nbrCount.Data()),
				thrust::device_pointer_cast(d_nbrCount.Data() + N_R_),
				thrust::device_pointer_cast(d_nbrOffsets.Data() + 1),
				ToU64{}, thrust::plus<uint64_t>{});
			GPU::Copy_DeviceToHost(&N_total, d_nbrOffsets.Data() + N_R_, 1, stream_);
			GPU::StreamSync(stream_);
			if (N_total > d_nbrList.Size())
			{
				d_nbrList.Resize(
					N_total + Base::ExtraPercent(N_total, extraAllocCapacityPct_));
			}
		};

	finalizeCSR(d_nbrCountRF_, d_nbrOffsetsRF_, d_nbrListRF_, N_totalRF_);
	finalizeCSR(d_nbrCountRB_, d_nbrOffsetsRB_, d_nbrListRB_, N_totalRB_);
	finalizeCSR(d_nbrCountRR_, d_nbrOffsetsRR_, d_nbrListRR_, N_totalRR_);
	GPU::StreamSync(stream_);
}

// -------------------------------------------------------------------------------------------------
// --- [ brute force implementation ] --------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_BruteForce_GPU::Init(
	const SimParams& params)
{
	(void)params;

	// nothing to do
}

void NS_BruteForce_GPU::Construct(
	const SimParams& params)
{
	GPU::NS_ReloadConstants(
		be_.N_F_, be_.N_B_, be_.N_R_, params.kernelRadius, 0.0f, 1u,
		0.0f, 0.0f, 0.0f,
		Sim::CellCoordFrame{}, 0u, 0u, be_.stream_);

	GPU::StreamSync(be_.stream_);
}

void NS_BruteForce_GPU::Query_Count(
	const SimParams& params)
{

	// (no count reset needed, query overwrites every per-particle count)

	if (be_.N_F_ > 0)
	{
		GPU::Launch_BruteForce_Query_Count(
			be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
			be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
			be_.d_xPosR_.Data(), be_.d_yPosR_.Data(), be_.d_zPosR_.Data(),
			be_.N_F_, be_.N_B_, be_.N_R_, be_.stream_,
			be_.d_nbrCountFF_.Data(), be_.d_nbrCountFB_.Data(), be_.d_nbrCountFR_.Data(),
			be_.d_nbrChunkMaxCountFF_.Data(),
			be_.d_nbrChunkMaxCountFB_.Data(),
			be_.d_nbrChunkMaxCountFR_.Data(),
			be_.d_nbrChunkBaseIdxFF_.Data(),
			be_.d_nbrChunkBaseIdxFB_.Data(),
			be_.d_nbrChunkBaseIdxFR_.Data(),
			be_.d_nbrChunkFormatFF_.Data(),
			be_.d_nbrChunkFormatFB_.Data(),
			be_.d_nbrChunkFormatFR_.Data());
	}

	GPU::Launch_BruteForce_Query_Count_RF_RB_RR(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		be_.d_xPosR_.Data(), be_.d_yPosR_.Data(), be_.d_zPosR_.Data(),
		be_.d_bodyIdR_.Data(),
		be_.N_F_, be_.N_B_, be_.N_R_, be_.stream_,
		be_.d_nbrCountRF_.Data(), be_.d_nbrCountRB_.Data(), be_.d_nbrCountRR_.Data());

	GPU::StreamSync(be_.stream_);
}

void NS_BruteForce_GPU::Query_Fill(
	const SimParams& params)
{
	(void)params;

	// reset overflow before filling packed SELL32 lists
	GPU::MemsetZero(be_.d_nbrOverflowFlag_.Data(), 1, be_.stream_);

	if (be_.N_F_ > 0)
	{
		GPU::Launch_BruteForce_Query_Fill(
			be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
			be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
			be_.d_xPosR_.Data(), be_.d_yPosR_.Data(), be_.d_zPosR_.Data(),
			be_.N_F_, be_.N_B_, be_.N_R_, be_.stream_,
			be_.d_nbrChunkMaxCountFF_.Data(),
			be_.d_nbrChunkMaxCountFB_.Data(),
			be_.d_nbrChunkMaxCountFR_.Data(),
			be_.d_nbrChunkOffsetsFF_.Data(),
			be_.d_nbrChunkOffsetsFB_.Data(),
			be_.d_nbrChunkOffsetsFR_.Data(),
			be_.d_nbrChunkBaseIdxFF_.Data(),
			be_.d_nbrChunkBaseIdxFB_.Data(),
			be_.d_nbrChunkBaseIdxFR_.Data(),
			be_.d_nbrChunkFormatFF_.Data(),
			be_.d_nbrChunkFormatFB_.Data(),
			be_.d_nbrChunkFormatFR_.Data(),
			be_.d_nbrListDelta8FF_.Data(),
			be_.d_nbrListDelta8FB_.Data(),
			be_.d_nbrListDelta8FR_.Data(),
			be_.d_nbrListDelta16FF_.Data(),
			be_.d_nbrListDelta16FB_.Data(),
			be_.d_nbrListDelta16FR_.Data(),
			be_.d_nbrListAbs32FF_.Data(),
			be_.d_nbrListAbs32FB_.Data(),
			be_.d_nbrListAbs32FR_.Data(),
			be_.d_nbrOverflowFlag_.Data());
	}

	GPU::Launch_BruteForce_Query_Fill_RF_RB_RR(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		be_.d_xPosR_.Data(), be_.d_yPosR_.Data(), be_.d_zPosR_.Data(),
		be_.d_bodyIdR_.Data(),
		be_.N_F_, be_.N_B_, be_.N_R_, be_.stream_,
		be_.d_nbrOffsetsRF_.Data(), be_.d_nbrOffsetsRB_.Data(), be_.d_nbrOffsetsRR_.Data(),
		be_.d_nbrListRF_.Data(), be_.d_nbrListRB_.Data(), be_.d_nbrListRR_.Data());

	#if defined(_DEBUG)
	CheckNbrListErrorFlag(be_.d_nbrOverflowFlag_, be_.stream_,
		"GPU brute force SELL32 fill-query");
	#else
	GPU::StreamSync(be_.stream_);
	#endif
}

void NS_BruteForce_GPU::DataReOrder(
	const SimParams& params)
{
	(void)params;

	// no re-ordering in brute force implementation
}

Mem::ParticleClassMemUsage NS_BruteForce_GPU::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageF{};

	Mem::Add(usageF, MemUsageOf_DeviceBuffer(be_.d_nsSharedScratch_));

	return { usageF, {}, {} };
}

// -------------------------------------------------------------------------------------------------
// --- [ cell linked list compact implementation ] -------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_CellLinkedList_Compact_GPU::Init(const SimParams& params)
{
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;
	cellQueryReach_ = 1;
	cellFrame_ = {};
	N_nonEmptyCellsF_ = 0;
	N_nonEmptyCellsB_ = 0;
	N_nonEmptyCellsR_ = 0;
	N_nonEmptyCellsAny_ = 0;
	N_queryCellSlots_ = 0;
	particlesMatchCurrentCellOrder_ = false;
	cellIdsAvailableForReuse_ = false;

	d_uniqueCellIdF_.Release();
	d_cellStartIdxF_.Release();
	d_sortedParticleIdxF_.Release();
	d_uniqueCellIdB_.Release();
	d_cellStartIdxB_.Release();
	d_uniqueCellIdR_.Release();
	d_cellStartIdxR_.Release();
	d_sortedParticleIdxR_.Release();
	d_cellIdR_.Release();
	d_cellMarkerR_.Release();
	d_uniqueCellIdAny_.Release();
	d_cellRefFByAny_.Release();
	d_cellRefBByAny_.Release();
	d_queryCellRefAnyByCellF_.Release();
}

void NS_CellLinkedList_Compact_GPU::UpdateCellFrame()
{
	cellFrame_ = {};

	if (!be_.simDomainBoundsValid_ || Base::ApproxEqu(cellSize_, 0.0f))
	{
		throw std::runtime_error(
			"GPU compact CLL requires valid simulation-domain bounds and cell size.");
	}

	const int64_t cxMin = static_cast<int64_t>(
		std::floor(be_.simDomainMin_.x * cellSizeInv_));
	const int64_t cyMin = static_cast<int64_t>(
		std::floor(be_.simDomainMin_.y * cellSizeInv_));
	const int64_t czMin = static_cast<int64_t>(
		std::floor(be_.simDomainMin_.z * cellSizeInv_));
	const int64_t cxMax = static_cast<int64_t>(
		std::floor(be_.simDomainMax_.x * cellSizeInv_));
	const int64_t cyMax = static_cast<int64_t>(
		std::floor(be_.simDomainMax_.y * cellSizeInv_));
	const int64_t czMax = static_cast<int64_t>(
		std::floor(be_.simDomainMax_.z * cellSizeInv_));

	// include the query reach to make every stencil cell stays inside the frame
	const int64_t frameMargin = static_cast<int64_t>(cellQueryReach_);
	const int64_t frameOrigX = cxMin - frameMargin;
	const int64_t frameOrigY = cyMin - frameMargin;
	const int64_t frameOrigZ = czMin - frameMargin;
	const int64_t frameMaxX = cxMax + frameMargin;
	const int64_t frameMaxY = cyMax + frameMargin;
	const int64_t frameMaxZ = czMax + frameMargin;

	CellCoordFrame nextFrame{};
	if (!TryBuildCellCoordFrame(
		frameOrigX, frameOrigY, frameOrigZ,
		frameMaxX, frameMaxY, frameMaxZ,
		nextFrame))
	{
		throw std::runtime_error(
			"GPU compact CLL simulation-domain cell frame exceeds the supported range.");
	}

	cellFrame_ = nextFrame;
}

void NS_CellLinkedList_Compact_GPU::Construct(const SimParams& params)
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const uint32_t N_R = be_.N_R_;

	if (N_F == 0 && N_B == 0 && N_R == 0)
	{
		std::cerr << "Warning: Cannot construct neighbor search for an empty simulation.\n";

		N_nonEmptyCellsF_ = 0;
		N_nonEmptyCellsB_ = 0;
		N_nonEmptyCellsR_ = 0;
		N_nonEmptyCellsAny_ = 0;
		N_queryCellSlots_ = 0;
		particlesMatchCurrentCellOrder_ = false;
		cellIdsAvailableForReuse_ = false;

		d_uniqueCellIdF_.Release();
		d_cellStartIdxF_.Release();
		d_sortedParticleIdxF_.Release();
		d_uniqueCellIdB_.Release();
		d_cellStartIdxB_.Release();
		d_uniqueCellIdR_.Release();
		d_cellStartIdxR_.Release();
		d_sortedParticleIdxR_.Release();
		d_cellIdR_.Release();
		d_cellMarkerR_.Release();
		d_uniqueCellIdAny_.Release();
		d_cellRefFByAny_.Release();
		d_cellRefBByAny_.Release();
		d_queryCellRefAnyByCellF_.Release();

		return;
	}

	cellSize_ = params.kernelRadius * params.gridCellSizeRatio;
	cellSizeInv_ = Base::ApproxEqu(cellSize_, 0.0f) ? 0.0f : (1.0f / cellSize_);
	const float querySpanCells = Base::ApproxEqu(cellSize_, 0.0f)
		? 0.0f : (params.kernelRadius / cellSize_);
	cellQueryReach_ = Base::ApproxEqu(cellSize_, 0.0f)
		? 1u : std::max<uint32_t>(1u,
			static_cast<uint32_t>(std::ceil(std::max(0.0f, querySpanCells - 1e-6f))));
	UpdateCellFrame();

	EnsureStorageAllocation();
	const NS_CompactSharedScratchLayout scratchLayout = BuildCompactSharedScratchLayout(N_F, N_B);
	uint64_t* const d_cellIdF = ScratchPtr<uint64_t>(be_.d_nsSharedScratch_, scratchLayout.build.cellIdF);
	uint64_t* const d_cellIdB = ScratchPtr<uint64_t>(be_.d_nsSharedScratch_, scratchLayout.build.cellIdB);
	uint32_t* const d_sortedParticleIdxB = ScratchPtr<uint32_t>(be_.d_nsSharedScratch_, scratchLayout.build.sortedParticleIdxB);

	GPU::NS_ReloadConstants(
		be_.N_F_, be_.N_B_, be_.N_R_, params.kernelRadius, cellSize_, cellQueryReach_,
		0.0f, 0.0f, 0.0f,
		cellFrame_, 0u, 0u,
		be_.stream_);

	// if DataReOrder() already computed and permuted the per-particle cell IDs for this step,
	// then they can be reused directly here and no re-computation is needed
	const bool canReuseCellIds = particlesMatchCurrentCellOrder_ && cellIdsAvailableForReuse_;

	if (!canReuseCellIds)
	{
		cellIdsAvailableForReuse_ = false;

		// --- [ cell coordinates and indices for fluid and boundary particles ] ---------------------

		if (N_F > 0) // fluid
		{
			GPU::Launch_FillCellIdAndInitParticleIdx(
				be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
				N_F, be_.stream_, d_cellIdF, d_sortedParticleIdxF_.Data());
		}

		if (N_B > 0) // boundary
		{
			GPU::Launch_FillCellIdAndInitParticleIdx(
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

	// build rigid particle cell keys without changing body-contiguous particle storage
	if (N_R > 0)
	{
		GPU::Launch_FillCellIdAndInitParticleIdx(
			be_.d_xPosR_.Data(), be_.d_yPosR_.Data(), be_.d_zPosR_.Data(),
			N_R, be_.stream_, d_cellIdR_.Data(), d_sortedParticleIdxR_.Data());
		thrust::sort_by_key(
			thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_cellIdR_.Data()),
			thrust::device_pointer_cast(d_cellIdR_.Data() + N_R),
			thrust::device_pointer_cast(d_sortedParticleIdxR_.Data()));
	}

	// --- [ build compact cell list and fill in cell start indices ] --------------------------------

	if (N_F > 0) // fluid
	{
		BuildCompactCellList(d_uniqueCellIdF_, d_cellStartIdxF_, d_sortedParticleIdxF_.Data(),
			d_cellIdF, N_F, N_nonEmptyCellsF_);
	}
	else
	{
		N_nonEmptyCellsF_ = 0;
	}

	if (N_B > 0) // boundary
	{
		BuildCompactCellList(d_uniqueCellIdB_, d_cellStartIdxB_, d_sortedParticleIdxB,
			d_cellIdB, N_B, N_nonEmptyCellsB_);
	}
	else
	{
		N_nonEmptyCellsB_ = 0;
	}

	if (N_R > 0) // rigid
	{
		BuildCompactCellList(d_uniqueCellIdR_, d_cellStartIdxR_, d_cellMarkerR_.Data(),
			d_cellIdR_.Data(), N_R, N_nonEmptyCellsR_);
	}
	else
	{
		N_nonEmptyCellsR_ = 0;
	}

	BuildMergedCompactCellLookup();
	BuildQueryLookupBounds();
	BuildQueryCellRefCache();

	// the compact cell IDs live in shared scratch and are no longer considered reusable
	// after construction completes, since later query/finalize phases alias that storage
	cellIdsAvailableForReuse_ = false;
}

void NS_CellLinkedList_Compact_GPU::Query_Count(const SimParams& params)
{
	(void)params;

	if (!particlesMatchCurrentCellOrder_)
	{
		throw std::runtime_error(
			"GPU compact CLL query requires particle arrays to match the compact cell ordering");
	}

	// (no count reset needed, query overwrites every per-particle count)

	if (be_.N_F_ > 0)
	{
		GPU::Launch_CellLinkedList_Compact_Query_Count(
			be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
			be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
			be_.d_xPosR_.Data(), be_.d_yPosR_.Data(), be_.d_zPosR_.Data(),
			d_cellRefFByAny_.Data(), d_cellRefBByAny_.Data(),
			d_cellStartIdxF_.Data(), d_cellStartIdxB_.Data(),
			d_sortedParticleIdxF_.Data(),
			d_queryCellRefAnyByCellF_.Data(),
			d_uniqueCellIdR_.Data(), d_cellStartIdxR_.Data(), d_sortedParticleIdxR_.Data(),
			N_nonEmptyCellsR_,
			N_queryCellSlots_, be_.N_F_, be_.N_B_, be_.N_R_, be_.stream_,
			be_.d_nbrCountFF_.Data(), be_.d_nbrCountFB_.Data(), be_.d_nbrCountFR_.Data(),
			be_.d_nbrChunkMaxCountFF_.Data(),
			be_.d_nbrChunkMaxCountFB_.Data(),
			be_.d_nbrChunkMaxCountFR_.Data(),
			be_.d_nbrChunkBaseIdxFF_.Data(),
			be_.d_nbrChunkBaseIdxFB_.Data(),
			be_.d_nbrChunkBaseIdxFR_.Data(),
			be_.d_nbrChunkFormatFF_.Data(),
			be_.d_nbrChunkFormatFB_.Data(),
			be_.d_nbrChunkFormatFR_.Data());
	}

	GPU::Launch_CellLinkedList_Compact_Query_Count_RF_RB_RR(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		be_.d_xPosR_.Data(), be_.d_yPosR_.Data(), be_.d_zPosR_.Data(),
		be_.d_bodyIdR_.Data(),
		d_uniqueCellIdF_.Data(), d_cellStartIdxF_.Data(),
		d_uniqueCellIdB_.Data(), d_cellStartIdxB_.Data(),
		d_uniqueCellIdR_.Data(), d_cellStartIdxR_.Data(), d_sortedParticleIdxR_.Data(),
		N_nonEmptyCellsF_, N_nonEmptyCellsB_, N_nonEmptyCellsR_,
		be_.N_F_, be_.N_B_, be_.N_R_, be_.stream_,
		be_.d_nbrCountRF_.Data(), be_.d_nbrCountRB_.Data(), be_.d_nbrCountRR_.Data());

	GPU::StreamSync(be_.stream_);
}

void NS_CellLinkedList_Compact_GPU::Query_Fill(const SimParams& params)
{
	(void)params;

	if (!particlesMatchCurrentCellOrder_)
	{
		throw std::runtime_error(
			"GPU compact CLL query requires particle arrays to match the compact cell ordering");
	}

	// reset overflow before filling packed SELL32 lists
	GPU::MemsetZero(be_.d_nbrOverflowFlag_.Data(), 1, be_.stream_);

	if (be_.N_F_ > 0)
	{
		GPU::Launch_CellLinkedList_Compact_Query_Fill(
			be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
			be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
			be_.d_xPosR_.Data(), be_.d_yPosR_.Data(), be_.d_zPosR_.Data(),
			d_cellRefFByAny_.Data(), d_cellRefBByAny_.Data(),
			d_cellStartIdxF_.Data(), d_cellStartIdxB_.Data(),
			d_sortedParticleIdxF_.Data(),
			d_queryCellRefAnyByCellF_.Data(),
			d_uniqueCellIdR_.Data(), d_cellStartIdxR_.Data(), d_sortedParticleIdxR_.Data(),
			N_nonEmptyCellsR_,
			N_queryCellSlots_, be_.N_F_, be_.N_B_, be_.N_R_, be_.stream_,
			be_.d_nbrChunkMaxCountFF_.Data(), be_.d_nbrChunkMaxCountFB_.Data(),
			be_.d_nbrChunkMaxCountFR_.Data(),
			be_.d_nbrChunkOffsetsFF_.Data(), be_.d_nbrChunkOffsetsFB_.Data(),
			be_.d_nbrChunkOffsetsFR_.Data(),
			be_.d_nbrChunkBaseIdxFF_.Data(), be_.d_nbrChunkBaseIdxFB_.Data(),
			be_.d_nbrChunkBaseIdxFR_.Data(),
			be_.d_nbrChunkFormatFF_.Data(), be_.d_nbrChunkFormatFB_.Data(),
			be_.d_nbrChunkFormatFR_.Data(),
			be_.d_nbrListDelta8FF_.Data(), be_.d_nbrListDelta8FB_.Data(),
			be_.d_nbrListDelta8FR_.Data(),
			be_.d_nbrListDelta16FF_.Data(), be_.d_nbrListDelta16FB_.Data(),
			be_.d_nbrListDelta16FR_.Data(),
			be_.d_nbrListAbs32FF_.Data(), be_.d_nbrListAbs32FB_.Data(),
			be_.d_nbrListAbs32FR_.Data(),
			be_.d_nbrOverflowFlag_.Data());
	}

	GPU::Launch_CellLinkedList_Compact_Query_Fill_RF_RB_RR(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		be_.d_xPosR_.Data(), be_.d_yPosR_.Data(), be_.d_zPosR_.Data(),
		be_.d_bodyIdR_.Data(),
		d_uniqueCellIdF_.Data(), d_cellStartIdxF_.Data(),
		d_uniqueCellIdB_.Data(), d_cellStartIdxB_.Data(),
		d_uniqueCellIdR_.Data(), d_cellStartIdxR_.Data(), d_sortedParticleIdxR_.Data(),
		N_nonEmptyCellsF_, N_nonEmptyCellsB_, N_nonEmptyCellsR_,
		be_.N_F_, be_.N_B_, be_.N_R_, be_.stream_,
		be_.d_nbrOffsetsRF_.Data(), be_.d_nbrOffsetsRB_.Data(), be_.d_nbrOffsetsRR_.Data(),
		be_.d_nbrListRF_.Data(), be_.d_nbrListRB_.Data(), be_.d_nbrListRR_.Data());

	#if defined(_DEBUG)
	CheckNbrListErrorFlag(be_.d_nbrOverflowFlag_, be_.stream_,
		"GPU compact CLL SELL32 fill-query");
	#else
	GPU::StreamSync(be_.stream_);
	#endif
}

void NS_CellLinkedList_Compact_GPU::DataReOrder(const SimParams& params)
{
	ReOrderParticleData(params, true);
}

void NS_CellLinkedList_Compact_GPU::DataReOrder_SolverStep(const SimParams& params)
{
	ReOrderParticleData(params, false);
}

void NS_CellLinkedList_Compact_GPU::ReOrderParticleData(
	const SimParams& params,
	const bool permuteSolverFields)
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
	UpdateCellFrame();

	EnsureStorageAllocation();
	const NS_CompactSharedScratchLayout scratchLayout = BuildCompactSharedScratchLayout(N_F, N_B);
	uint64_t* const d_cellIdF = ScratchPtr<uint64_t>(be_.d_nsSharedScratch_, scratchLayout.build.cellIdF);
	uint64_t* const d_cellIdB = ScratchPtr<uint64_t>(be_.d_nsSharedScratch_, scratchLayout.build.cellIdB);
	uint32_t* const d_sortedParticleIdxB = ScratchPtr<uint32_t>(be_.d_nsSharedScratch_, scratchLayout.build.sortedParticleIdxB);

	GPU::NS_ReloadConstants(
		be_.N_F_, be_.N_B_, be_.N_R_, params.kernelRadius, cellSize_, cellQueryReach_,
		0.0f, 0.0f, 0.0f,
		cellFrame_, 0u, 0u,
		be_.stream_);

	// build the exact compact-CLL cell IDs and particle permutation for fluid particles
	if (N_F > 0)
	{
		GPU::Launch_FillCellIdAndInitParticleIdx(
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
		GPU::Launch_FillCellIdAndInitParticleIdx(
			be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
			N_B, be_.stream_, d_cellIdB, d_sortedParticleIdxB);

		thrust::device_ptr<uint64_t> keysB(d_cellIdB);
		thrust::device_ptr<uint32_t> valsB(d_sortedParticleIdxB);

		// using radix sort from thrust
		thrust::sort_by_key(thrust::cuda::par.on(be_.stream_),
			keysB, keysB + N_B, valsB);
	}

	// density array as temp scratch storage (only needed after the particle arrays are re-ordered),
	// unless it is permuted itself below and needs a scratch buffer of its own
	DeviceBuffer<float> d_solverFieldScratch;
	if (permuteSolverFields) { d_solverFieldScratch.Resize(N_MAX); }
	DeviceBuffer<float>& d_reorderScratch =
		permuteSolverFields ? d_solverFieldScratch : be_.d_densityF_;

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

		// the remaining fluid particle fields only need to be permuted after backend-only neighbor searches,
		// because a following full solver step re-initializes them (TODO: update if solver step behavior changes)
		if (permuteSolverFields)
		{
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_xAccPressF_, d_reorderScratch, be_.stream_);
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_yAccPressF_, d_reorderScratch, be_.stream_);
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_zAccPressF_, d_reorderScratch, be_.stream_);

			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_xVelPredF_, d_reorderScratch, be_.stream_);
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_yVelPredF_, d_reorderScratch, be_.stream_);
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_zVelPredF_, d_reorderScratch, be_.stream_);

			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_sourceTermF_, d_reorderScratch, be_.stream_);
			ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_diagElemF_, d_reorderScratch, be_.stream_);
			// without a warm start, pressure is re-initialized by the source-term kernel
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
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_xPosB_, d_reorderScratch, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_yPosB_, d_reorderScratch, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_zPosB_, d_reorderScratch, be_.stream_);
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

Mem::ParticleClassMemUsage NS_CellLinkedList_Compact_GPU::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};
	Mem::MemUsage usageR{};

	// fluid
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_uniqueCellIdF_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_cellStartIdxF_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_sortedParticleIdxF_));

	// boundary
	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_uniqueCellIdB_));
	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_cellStartIdxB_));
	Mem::Add(usageR, MemUsageOf_DeviceBuffer(d_uniqueCellIdR_));

	// rigid
	Mem::Add(usageR, MemUsageOf_DeviceBuffer(d_cellStartIdxR_));
	Mem::Add(usageR, MemUsageOf_DeviceBuffer(d_sortedParticleIdxR_));
	Mem::Add(usageR, MemUsageOf_DeviceBuffer(d_cellIdR_));
	Mem::Add(usageR, MemUsageOf_DeviceBuffer(d_cellMarkerR_));

	// shared merged lookup metadata is accounted in the fluid-side bucket to avoid double-counting
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_uniqueCellIdAny_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_cellRefFByAny_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_cellRefBByAny_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_queryCellRefAnyByCellF_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(be_.d_nsSharedScratch_));

	// shared scan scratch is accounted in the fluid-side bucket to avoid double-counting

	return { usageF, usageB, usageR };
}

void NS_CellLinkedList_Compact_GPU::EnsureStorageAllocation()
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const uint32_t N_R = be_.N_R_;

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

	if (N_R > 0)
	{
		if (d_uniqueCellIdR_.Size() < N_R) { d_uniqueCellIdR_.Resize(N_R); }
		if (d_cellStartIdxR_.Size() < N_R + 1) { d_cellStartIdxR_.Resize(N_R + 1); }
		if (d_sortedParticleIdxR_.Size() < N_R) { d_sortedParticleIdxR_.Resize(N_R); }
		if (d_cellIdR_.Size() < N_R) { d_cellIdR_.Resize(N_R); }
		if (d_cellMarkerR_.Size() < N_R) { d_cellMarkerR_.Resize(N_R); }
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

	be_.EnsureNsSharedScratchCapacity(
		BuildCompactSharedScratchLayout(N_F, N_B).N_totalBytes);
}

void NS_CellLinkedList_Compact_GPU::BuildCompactCellList(
	DeviceBuffer<uint64_t>& d_uniqueCellId,
	DeviceBuffer<uint32_t>& d_cellStartIdx,
	uint32_t* d_markerScanScratch,
	const uint64_t* d_cellIdSorted,
	uint32_t N, uint32_t& N_nonEmptyCells)
{
	if (N == 0)
	{
		N_nonEmptyCells = 0;
		return;
	}

	// mark cell starts in the caller-provided marker/scan scratch buffer
	GPU::Launch_MarkCellStarts(
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
	N_nonEmptyCells = lastScan;

	// fill in cell-related data
	// (write only first N_NonEmptyCells entries and one sentinel entry)
	GPU::Launch_FillUniqueCellIdAndCellStartIdx(
		d_cellIdSorted, d_markerScanScratch,
		N, N_nonEmptyCells, be_.stream_,
		d_uniqueCellId.Data(), d_cellStartIdx.Data());
}

void NS_CellLinkedList_Compact_GPU::BuildMergedCompactCellLookup()
{
	N_nonEmptyCellsAny_ = 0;

	if (N_nonEmptyCellsF_ == 0 && N_nonEmptyCellsB_ == 0) { return; }

	const auto beginAny = thrust::device_pointer_cast(d_uniqueCellIdAny_.Data());
	auto endAny = beginAny;

	if (N_nonEmptyCellsF_ == 0)
	{
		endAny = thrust::copy(
			thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_uniqueCellIdB_.Data()),
			thrust::device_pointer_cast(d_uniqueCellIdB_.Data() + N_nonEmptyCellsB_),
			beginAny);
	}
	else if (N_nonEmptyCellsB_ == 0)
	{
		endAny = thrust::copy(
			thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_uniqueCellIdF_.Data()),
			thrust::device_pointer_cast(d_uniqueCellIdF_.Data() + N_nonEmptyCellsF_),
			beginAny);
	}
	else
	{
		endAny = thrust::set_union(
			thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_uniqueCellIdF_.Data()),
			thrust::device_pointer_cast(d_uniqueCellIdF_.Data() + N_nonEmptyCellsF_),
			thrust::device_pointer_cast(d_uniqueCellIdB_.Data()),
			thrust::device_pointer_cast(d_uniqueCellIdB_.Data() + N_nonEmptyCellsB_),
			beginAny);
	}

	N_nonEmptyCellsAny_ = static_cast<uint32_t>(endAny - beginAny);

	if (N_nonEmptyCellsAny_ == 0) { return; }

	GPU::Launch_FillMergedCompactCellRefs(
		d_uniqueCellIdAny_.Data(), N_nonEmptyCellsAny_,
		d_uniqueCellIdF_.Data(), d_uniqueCellIdB_.Data(),
		N_nonEmptyCellsF_, N_nonEmptyCellsB_, be_.stream_,
		d_cellRefFByAny_.Data(), d_cellRefBByAny_.Data());
}

void NS_CellLinkedList_Compact_GPU::BuildQueryLookupBounds()
{
	if (N_nonEmptyCellsF_ == 0 || N_nonEmptyCellsAny_ == 0)
	{
		return;
	}

	const NS_CompactSharedScratchLayout scratchLayout =
		BuildCompactSharedScratchLayout(be_.N_F_, be_.N_B_);
	uint32_t* const d_queryLoAnyByCellF = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.queryLoAnyByCellF);
	uint32_t* const d_queryHiAnyByCellF = ScratchPtr<uint32_t>(
		be_.d_nsSharedScratch_, scratchLayout.queryHiAnyByCellF);

	GPU::Launch_FillQueryLookupBoundsF(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		d_cellStartIdxF_.Data(), d_uniqueCellIdAny_.Data(),
		N_nonEmptyCellsF_, N_nonEmptyCellsAny_, cellQueryReach_, be_.stream_,
		d_sortedParticleIdxF_.Data(),
		d_queryLoAnyByCellF, d_queryHiAnyByCellF);
}

void NS_CellLinkedList_Compact_GPU::BuildQueryCellRefCache()
{
	N_queryCellSlots_ = QueryCellSlotCount(cellQueryReach_);

	if (N_nonEmptyCellsF_ == 0 || N_nonEmptyCellsAny_ == 0)
	{
		return;
	}

	const uint64_t N_queryCellRefs = static_cast<uint64_t>(N_nonEmptyCellsF_) * N_queryCellSlots_;
	if (N_queryCellRefs > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) [[unlikely]]
	{
		throw std::runtime_error("GPU compact CLL query-cell cache too large.");
	}
	if (d_queryCellRefAnyByCellF_.Size() < static_cast<size_t>(N_queryCellRefs))
	{
		const uint64_t extraCapacity = Base::ExtraPercent(N_queryCellRefs, be_.extraAllocCapacityPct_);
		if (extraCapacity > std::numeric_limits<uint64_t>::max() - N_queryCellRefs) [[unlikely]]
		{
			throw std::runtime_error("GPU compact CLL query-cell cache too large.");
		}

		const uint64_t capTarget = N_queryCellRefs + extraCapacity;
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

	GPU::Launch_FillQueryCellRefsByCellF(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		d_cellStartIdxF_.Data(), d_uniqueCellIdAny_.Data(),
		d_queryLoAnyByCellF, d_queryHiAnyByCellF,
		N_nonEmptyCellsF_, N_queryCellSlots_, be_.stream_,
		d_queryCellRefAnyByCellF_.Data());
}
