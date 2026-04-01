// neighbor_search.cu

// external
#include <cmath>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
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
	struct ToU64
	{
		__host__ __device__ uint64_t operator()(const uint32_t v) const
		{
			return static_cast<uint64_t>(v);
		}
	};

	template <typename T>
	inline Mem::MemUsage MemUsageOf_DeviceBuffer(
		const DeviceBuffer<T>& src) noexcept
	{
		const uint64_t bytes = static_cast<uint64_t>(src.Size()) * sizeof(T);

		return Mem::MemUsage{ bytes, bytes, bytes };
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
}

void SimBackend_GPU01::FinalizeNeighborLists_Count()
{
	if (N_F_ == 0) // nothing to do for empty sim
	{
		N_totalFF_ = 0;
		N_totalFB_ = 0;

		if (d_nbrOffsetsFF_.Size() != 1) { d_nbrOffsetsFF_.Resize(1); }
		if (d_nbrOffsetsFB_.Size() != 1) { d_nbrOffsetsFB_.Resize(1); }

		const uint64_t zero = 0;
		GPU::Copy_HostToDevice(d_nbrOffsetsFF_.Data(), &zero, 1, stream_);
		GPU::Copy_HostToDevice(d_nbrOffsetsFB_.Data(), &zero, 1, stream_);
		GPU::StreamSync(stream_);

		return;
	}

	const uint64_t N_offsets = static_cast<uint64_t>(N_F_) + 1;
	if (d_nbrOffsetsFF_.Size() != N_offsets) { d_nbrOffsetsFF_.Resize(N_offsets); }
	if (d_nbrOffsetsFB_.Size() != N_offsets) { d_nbrOffsetsFB_.Resize(N_offsets); }

	// turn temporary counts into actual offsets for FF
	thrust::transform_exclusive_scan(
		thrust::cuda::par.on(stream_),
		thrust::device_pointer_cast(d_nbrCountFF_.Data()),
		thrust::device_pointer_cast(d_nbrCountFF_.Data() + N_F_),
		thrust::device_pointer_cast(d_nbrOffsetsFF_.Data()),
		ToU64{},
		uint64_t{ 0 },
		thrust::plus<uint64_t>{}
	);

	// turn temporary counts into actual offsets for FB
	thrust::transform_exclusive_scan(
		thrust::cuda::par.on(stream_),
		thrust::device_pointer_cast(d_nbrCountFB_.Data()),
		thrust::device_pointer_cast(d_nbrCountFB_.Data() + N_F_),
		thrust::device_pointer_cast(d_nbrOffsetsFB_.Data()),
		ToU64{},
		uint64_t{ 0 },
		thrust::plus<uint64_t>{}
	);

	uint64_t h_lastOffsetFF = 0;
	uint64_t h_lastOffsetFB = 0;
	uint32_t h_lastCountFF = 0;
	uint32_t h_lastCountFB = 0;

	GPU::Copy_DeviceToHost(&h_lastOffsetFF,
		d_nbrOffsetsFF_.Data() + (N_F_ - 1), 1, stream_);
	GPU::Copy_DeviceToHost(&h_lastOffsetFB,
		d_nbrOffsetsFB_.Data() + (N_F_ - 1), 1, stream_);
	GPU::Copy_DeviceToHost(&h_lastCountFF,
		d_nbrCountFF_.Data() + (N_F_ - 1), 1, stream_);
	GPU::Copy_DeviceToHost(&h_lastCountFB,
		d_nbrCountFB_.Data() + (N_F_ - 1), 1, stream_);

	GPU::StreamSync(stream_);

	// finalize total fluid and boundary neighbor counts
	N_totalFF_ = h_lastOffsetFF + static_cast<uint64_t>(h_lastCountFF);
	N_totalFB_ = h_lastOffsetFB + static_cast<uint64_t>(h_lastCountFB);

	// store sentinel offsets at i = N_F_
	GPU::Copy_HostToDevice(
		d_nbrOffsetsFF_.Data() + N_F_, &N_totalFF_, 1, stream_);
	GPU::Copy_HostToDevice(
		d_nbrOffsetsFB_.Data() + N_F_, &N_totalFB_, 1, stream_);

	// ensure capacity of packed CSR storage for FF
	if (N_totalFF_ > d_nbrListFF_.Size())
	{
		const uint64_t capTarget = N_totalFF_
								 + Base::ExtraPercent(N_totalFF_, extraAllocCapacityPct_);
		d_nbrListFF_.Resize(capTarget);
	}

	// ensure capacity of packed CSR storage for FB
	if (N_totalFB_ > d_nbrListFB_.Size())
	{
		const uint64_t capTarget = N_totalFB_
								 + Base::ExtraPercent(N_totalFB_, extraAllocCapacityPct_);
		d_nbrListFB_.Resize(capTarget);
	}

	GPU::StreamSync(stream_);
}

// -------------------------------------------------------------------------------------------------
// --- [ brute force implementation ] --------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_BruteForce_GPU01::Init(const SimParams& params)
{
	// nothing
}

void NS_BruteForce_GPU01::Construct(const SimParams& params)
{
	GPU01::NS_ReloadConstants(
		be_.N_F_, be_.N_B_, params.kernelRadius, 0.0f, 1u,
		static_cast<uint32_t>(Base::GridCellIndexing::Morton), 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, be_.stream_);

	GPU::StreamSync(be_.stream_);
}

void NS_BruteForce_GPU01::Query_Count(const SimParams& params)
{
	GPU::MemsetZero(be_.d_nbrCountFF_.Data(), be_.N_F_, be_.stream_);
	GPU::MemsetZero(be_.d_nbrCountFB_.Data(), be_.N_F_, be_.stream_);

	GPU01::Launch_BruteForce_Query_Count(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		be_.N_F_, be_.N_B_, be_.stream_,
		be_.d_nbrCountFF_.Data(), be_.d_nbrCountFB_.Data());

	GPU::StreamSync(be_.stream_);
}

void NS_BruteForce_GPU01::Query_Fill(const SimParams& params)
{
	// (count-query and fill-query MUST find the same number of neighbors)
	// reset overflow before filling packed CSR lists
	GPU::MemsetZero(be_.d_nbrOverflowFlag_.Data(), 1, be_.stream_);

	GPU01::Launch_BruteForce_Query_Fill(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		be_.N_F_, be_.N_B_, be_.stream_,
		be_.d_nbrOffsetsFF_.Data(), be_.d_nbrOffsetsFB_.Data(),
		be_.d_nbrListFF_.Data(), be_.d_nbrListFB_.Data(),
		be_.d_nbrOverflowFlag_.Data());

	GPU::StreamSync(be_.stream_);
}

NeighborSpan NS_BruteForce_GPU01::QueryParticleFF(const uint32_t i) const
{
	(void)i;
	return NeighborSpan{};
}

NeighborSpan NS_BruteForce_GPU01::QueryParticleFB(const uint32_t i) const
{
	(void)i;
	return NeighborSpan{};
}

void NS_BruteForce_GPU01::DataReOrder(const SimParams& params)
{
	// TODO: implement
}

std::pair<Mem::MemUsage, Mem::MemUsage> NS_BruteForce_GPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

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

	d_uniqueCellIdF_.Release();
	d_cellStartIdxF_.Release();
	d_cellIdF_.Release();
	d_sortedParticleIdxF_.Release();

	d_uniqueCellIdB_.Release();
	d_cellStartIdxB_.Release();
	d_cellIdB_.Release();
	d_sortedParticleIdxB_.Release();

	d_cellStartMarkerScan_.Release();
}

void NS_CellLinkedList_Compact_GPU01::Construct(const SimParams& params)
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;

	if (N_F == 0 && N_B == 0)
	{
		std::cerr << "Warning: No construction possible (empty simulation)\n";

		N_NonEmptyCellsF_ = 0;
		N_NonEmptyCellsB_ = 0;

		d_uniqueCellIdF_.Release();
		d_cellStartIdxF_.Release();
		d_cellIdF_.Release();
		d_sortedParticleIdxF_.Release();

		d_uniqueCellIdB_.Release();
		d_cellStartIdxB_.Release();
		d_cellIdB_.Release();
		d_sortedParticleIdxB_.Release();

		d_cellStartMarkerScan_.Release();

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

	GPU01::NS_ReloadConstants(
		be_.N_F_, be_.N_B_, params.kernelRadius, cellSize_, cellQueryReach_,
		static_cast<uint32_t>(params.gridCellIndexing), 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, be_.stream_);

	// --- [ cell coordinates and indices for fluid and boundary particles ] -------------------------

	if (N_F > 0) // fluid
	{
		GPU01::Launch_FillCellIdAndInitParticleIdx(
			be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
			N_F, be_.stream_, d_cellIdF_.Data(), d_sortedParticleIdxF_.Data());
	}

	if (N_B > 0) // boundary
	{
		GPU01::Launch_FillCellIdAndInitParticleIdx(
			be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
			N_B, be_.stream_, d_cellIdB_.Data(), d_sortedParticleIdxB_.Data());
	}

	// --- [ sort particle indices by cell IDs ] -----------------------------------------------------

	if (N_F > 0) // fluid
	{
		thrust::device_ptr<uint64_t> keysF(d_cellIdF_.Data());
		thrust::device_ptr<uint32_t> valsF(d_sortedParticleIdxF_.Data());

		// using radix sort from thrust
		thrust::sort_by_key(thrust::cuda::par.on(be_.stream_),
			keysF, keysF + N_F,valsF);
	}

	if (N_B > 0) // boundary
	{
		thrust::device_ptr<uint64_t> keysB(d_cellIdB_.Data());
		thrust::device_ptr<uint32_t> valsB(d_sortedParticleIdxB_.Data());

		// using radix sort from thrust
		thrust::sort_by_key(thrust::cuda::par.on(be_.stream_),
			keysB, keysB + N_B, valsB);
	}

	// --- [ build compact cell list and fill in cell start indices ] --------------------------------

	if (N_F > 0) // fluid
	{
		BuildCompactCellList(d_uniqueCellIdF_, d_cellStartIdxF_,
			d_cellIdF_.Data(),
			N_F, N_NonEmptyCellsF_);
	}
	else
	{
		N_NonEmptyCellsF_ = 0;
	}

	if (N_B > 0) // boundary
	{
		BuildCompactCellList(d_uniqueCellIdB_, d_cellStartIdxB_,
			d_cellIdB_.Data(),
			N_B, N_NonEmptyCellsB_);
	}
	else
	{
		N_NonEmptyCellsB_ = 0;
	}
}

void NS_CellLinkedList_Compact_GPU01::Query_Count(const SimParams& params)
{
	// reset per-particle counts before count-query
	GPU::MemsetZero(be_.d_nbrCountFF_.Data(), be_.N_F_, be_.stream_);
	GPU::MemsetZero(be_.d_nbrCountFB_.Data(), be_.N_F_, be_.stream_);

	GPU01::Launch_CellLinkedList_Compact_Query_Count(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		d_uniqueCellIdF_.Data(), d_uniqueCellIdB_.Data(),
		d_cellStartIdxF_.Data(), d_cellStartIdxB_.Data(),
		d_sortedParticleIdxF_.Data(), d_sortedParticleIdxB_.Data(),
		N_NonEmptyCellsF_, N_NonEmptyCellsB_, be_.N_F_, be_.N_B_, be_.stream_,
		be_.d_nbrCountFF_.Data(), be_.d_nbrCountFB_.Data());

	GPU::StreamSync(be_.stream_);
}

void NS_CellLinkedList_Compact_GPU01::Query_Fill(const SimParams& params)
{
	// (count-query and fill-query MUST find the same number of neighbors)
	// reset overflow before filling packed CSR lists
	GPU::MemsetZero(be_.d_nbrOverflowFlag_.Data(), 1, be_.stream_);

	GPU01::Launch_CellLinkedList_Compact_Query_Fill(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		d_uniqueCellIdF_.Data(), d_uniqueCellIdB_.Data(),
		d_cellStartIdxF_.Data(), d_cellStartIdxB_.Data(),
		d_sortedParticleIdxF_.Data(), d_sortedParticleIdxB_.Data(),
		N_NonEmptyCellsF_, N_NonEmptyCellsB_, be_.N_F_, be_.N_B_, be_.stream_,
		be_.d_nbrOffsetsFF_.Data(), be_.d_nbrOffsetsFB_.Data(),
		be_.d_nbrListFF_.Data(), be_.d_nbrListFB_.Data(),
		be_.d_nbrOverflowFlag_.Data());

	// TODO: implement smart version of this, iterating over cells instead of particles

	GPU::StreamSync(be_.stream_);
}

NeighborSpan NS_CellLinkedList_Compact_GPU01::QueryParticleFF(const uint32_t i) const
{
	(void)i;
	return NeighborSpan{};
}

NeighborSpan NS_CellLinkedList_Compact_GPU01::QueryParticleFB(const uint32_t i) const
{
	(void)i;
	return NeighborSpan{};
}

void NS_CellLinkedList_Compact_GPU01::DataReOrder(const SimParams& params)
{
	(void)params;

	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const uint32_t N_MAX = std::max(N_F, N_B);

	if (N_MAX == 0) { return; }

	// shared scratch buffers for permutation gather
	if (be_.d_workBuffer_.Size() < N_MAX) { be_.d_workBuffer_.Resize(N_MAX); }

	// re-order fluid SoA arrays according to sorted fluid indices
	if (N_F > 0)
	{
		const uint32_t* d_permF = d_sortedParticleIdxF_.Data();

		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_xPosF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_yPosF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_zPosF_, be_.d_workBuffer_, be_.stream_);

		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_xVelF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_yVelF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_zVelF_, be_.d_workBuffer_, be_.stream_);

		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_xVelPredF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_yVelPredF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_zVelPredF_, be_.d_workBuffer_, be_.stream_);

		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_xAccPressF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_yAccPressF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_zAccPressF_, be_.d_workBuffer_, be_.stream_);

		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_sourceTermF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_diagElemF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_pressureF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_densityF_, be_.d_workBuffer_, be_.stream_);

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
		const uint32_t* d_permB = d_sortedParticleIdxB_.Data();

		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_xPosB_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_yPosB_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permB, N_B, be_.d_zPosB_, be_.d_workBuffer_, be_.stream_);

		// boundary arrays are now in sorted order, so sorted particle index becomes identity
		thrust::sequence(
			thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_sortedParticleIdxB_.Data()),
			thrust::device_pointer_cast(d_sortedParticleIdxB_.Data() + N_B),
			0u);
	}

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
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_cellIdF_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_sortedParticleIdxF_));

	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_uniqueCellIdB_));
	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_cellStartIdxB_));
	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_cellIdB_));
	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_sortedParticleIdxB_));

	// shared scan scratch is accounted in the fluid-side bucket
	// to avoid double-counting.
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_cellStartMarkerScan_));

	return { usageF, usageB };
}

void NS_CellLinkedList_Compact_GPU01::EnsureStorageAllocation()
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const uint32_t N_MAX = std::max(N_F, N_B);

	if (N_F > 0)
	{
		// keep capacity once allocated and only grow when needed
		// (to avoid repeated device alloc/free between steps)
		if (d_uniqueCellIdF_.Size() < N_F) { d_uniqueCellIdF_.Resize(N_F); }
		if (d_cellStartIdxF_.Size() < N_F + 1) { d_cellStartIdxF_.Resize(N_F + 1); }
		if (d_cellIdF_.Size() < N_F) { d_cellIdF_.Resize(N_F); }
		if (d_sortedParticleIdxF_.Size() < N_F + 1) { d_sortedParticleIdxF_.Resize(N_F + 1); }
	}

	if (N_B > 0)
	{
		// keep capacity once allocated and only grow when needed
		// (to avoid repeated device alloc/free between steps)
		if (d_uniqueCellIdB_.Size() < N_B) { d_uniqueCellIdB_.Resize(N_B); }
		if (d_cellStartIdxB_.Size() < N_B + 1) { d_cellStartIdxB_.Resize(N_B + 1); }
		if (d_cellIdB_.Size() < N_B) { d_cellIdB_.Resize(N_B); }
		if (d_sortedParticleIdxB_.Size() < N_B + 1) { d_sortedParticleIdxB_.Resize(N_B + 1); }
	}

	if (N_MAX > 0)
	{
		// keep capacity once allocated and only grow when needed
		// (to avoid repeated device alloc/free between steps)
		if (d_cellStartMarkerScan_.Size() < N_MAX) { d_cellStartMarkerScan_.Resize(N_MAX); }
	}
}

void NS_CellLinkedList_Compact_GPU01::BuildCompactCellList(
	DeviceBuffer<uint64_t>& d_uniqueCellId,
	DeviceBuffer<uint32_t>& d_cellStartIdx,
	const uint64_t* d_cellIdSorted,
	uint32_t N, uint32_t& N_NonEmptyCells)
{
	if (N == 0)
	{
		N_NonEmptyCells = 0;
		return;
	}

	// mark cell starts in the shared marker/scan scratch buffer
	GPU01::Launch_MarkCellStarts(
		d_cellIdSorted, N, be_.stream_, d_cellStartMarkerScan_.Data());

	// inclusive scan in-place to get cell indices for each particle
	{
		thrust::device_ptr<uint32_t> scanBegin(d_cellStartMarkerScan_.Data());
		thrust::device_ptr<uint32_t> scanEnd(d_cellStartMarkerScan_.Data() + N);

		thrust::inclusive_scan(thrust::cuda::par.on(be_.stream_),
			scanBegin, scanEnd, scanBegin);
	}

	// get last scan value to determine number of unique cells
	uint32_t lastScan = 0;
	GPU::Check(cudaMemcpyAsync(&lastScan,
		d_cellStartMarkerScan_.Data() + (N - 1),
		sizeof(uint32_t),
		cudaMemcpyDeviceToHost,
		be_.stream_));
	GPU::StreamSync(be_.stream_);
	N_NonEmptyCells = lastScan;

	// fill in cell-related data
	// (write only first N_NonEmptyCells entries and one sentinel entry)
	GPU01::Launch_FillUniqueCellIdAndCellStartIdx(
		d_cellIdSorted, d_cellStartMarkerScan_.Data(),
		N, N_NonEmptyCells, be_.stream_,
		d_uniqueCellId.Data(), d_cellStartIdx.Data());
}
