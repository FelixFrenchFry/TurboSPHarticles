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

// ------------------------------------------------------------------------------------------------
// --- [ brute force implementation ] -------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

void NS_BruteForce_GPU01::Init(const SimParams& params)
{
	// nothing
}

void NS_BruteForce_GPU01::Construct(const SimParams& params)
{
	GPU01::NS_ReloadConstants(
		be_.N_F_, be_.N_B_, params.kernelRadius, 0.0f, 0.0f, 0.0f, 0.0f,
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

// ------------------------------------------------------------------------------------------------
// --- [ cell linked list basic implementation ] --------------------------------------------------
// ------------------------------------------------------------------------------------------------

void NS_CellLinkedList_Basic_GPU01::Init(const SimParams& params)
{
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;

	xBoxOrig_ = 0.0f;
	yBoxOrig_ = 0.0f;
	zBoxOrig_ = 0.0f;

	N_DimX_ = 0;
	N_DimY_ = 0;
	N_DimZ_ = 0;
	N_Cells_ = 0;

	d_cellStartIdxF_.Release();
	d_cellEndIdxF_.Release();
	d_cellIdF_.Release();
	d_sortedParticleIdxF_.Release();

	d_cellStartIdxB_.Release();
	d_cellEndIdxB_.Release();
	d_cellIdB_.Release();
	d_sortedParticleIdxB_.Release();
}

void NS_CellLinkedList_Basic_GPU01::Construct(const SimParams& params)
{
	if (be_.N_F_ == 0 && be_.N_B_ == 0)
	{
		// SPDLOG_WARN("No construction possible (empty simulation)");
		std::cerr << "Warning: No construction possible (empty simulation)\n";

		N_DimX_ = 0;
		N_DimY_ = 0;
		N_DimZ_ = 0;
		N_Cells_ = 0;

		return;
	}

	cellSize_ = params.kernelRadius;
	cellSizeInv_ = Base::ApproxEqu(params.kernelRadius, 0.0f)
				 ? 0.0f : (1.0f / params.kernelRadius);

	// --- [ combined AABB over fluid and boundary particles ] ---------------------------------------

	// accumulators
	float initMin = std::numeric_limits<float>::max();
	float initMax = std::numeric_limits<float>::lowest();
	h_aabbF_ =
	{
		.xMin = initMin, .yMin = initMin, .zMin = initMin,
		.xMax = initMax, .yMax = initMax, .zMax = initMax
	};
	h_aabbB_ =
	{
		.xMin = initMin, .yMin = initMin, .zMin = initMin,
		.xMax = initMax, .yMax = initMax, .zMax = initMax
	};

	const bool hasF = (be_.N_F_ > 0);
	const bool hasB = (be_.N_B_ > 0);

	if (hasF)
	{
		auto X = thrust::device_pointer_cast(be_.d_xPosF_.Data());
		auto Y = thrust::device_pointer_cast(be_.d_yPosF_.Data());
		auto Z = thrust::device_pointer_cast(be_.d_zPosF_.Data());
		auto first = thrust::make_zip_iterator(thrust::make_tuple(X, Y, Z));
		auto last = first + be_.N_F_;

		h_aabbF_ = thrust::transform_reduce(
			thrust::cuda::par.on(be_.stream_),
			first, last,
			GPU01::ToMinMax3{},
			h_aabbF_,
			GPU01::ReduceMinMax3{});
	}

	if (hasB)
	{
		auto X = thrust::device_pointer_cast(be_.d_xPosB_.Data());
		auto Y = thrust::device_pointer_cast(be_.d_yPosB_.Data());
		auto Z = thrust::device_pointer_cast(be_.d_zPosB_.Data());
		auto first = thrust::make_zip_iterator(thrust::make_tuple(X, Y, Z));
		auto last = first + be_.N_B_;

		h_aabbB_ = thrust::transform_reduce(
			thrust::cuda::par.on(be_.stream_),
			first, last,
			GPU01::ToMinMax3{},
			h_aabbB_,
			GPU01::ReduceMinMax3{});
	}

	MinMax3 box;
	if (hasF && hasB)
	{
		box.xMin = std::min(h_aabbF_.xMin, h_aabbB_.xMin);
		box.yMin = std::min(h_aabbF_.yMin, h_aabbB_.yMin);
		box.zMin = std::min(h_aabbF_.zMin, h_aabbB_.zMin);
		box.xMax = std::max(h_aabbF_.xMax, h_aabbB_.xMax);
		box.yMax = std::max(h_aabbF_.yMax, h_aabbB_.yMax);
		box.zMax = std::max(h_aabbF_.zMax, h_aabbB_.zMax);
	}
	else if (hasF)
	{
		box = { h_aabbF_.xMin, h_aabbF_.yMin, h_aabbF_.zMin,
				  h_aabbF_.xMax, h_aabbF_.yMax, h_aabbF_.zMax };
	}
	else
	{
		box = { h_aabbB_.xMin, h_aabbB_.yMin, h_aabbB_.zMin,
				  h_aabbB_.xMax, h_aabbB_.yMax, h_aabbB_.zMax };
	}

	// padding to avoid edge clamp issues
	const float pad = 1e-7f;
	box.xMin -= pad; box.yMin -= pad; box.zMin -= pad;
	box.xMax += pad; box.yMax += pad; box.zMax += pad;

	xBoxOrig_ = box.xMin;
	yBoxOrig_ = box.yMin;
	zBoxOrig_ = box.zMin;

	const float xLen = fmaxf(0.0f, box.xMax - box.xMin);
	const float yLen = fmaxf(0.0f, box.yMax - box.yMin);
	const float zLen = fmaxf(0.0f, box.zMax - box.zMin);

	// compute grid dimensions
	N_DimX_ = std::max<uint32_t>(1, 
		static_cast<uint32_t>(std::ceil(xLen * cellSizeInv_)));
	N_DimY_ = std::max<uint32_t>(1, 
		static_cast<uint32_t>(std::ceil(yLen * cellSizeInv_)));
	N_DimZ_ = std::max<uint32_t>(1, 
		static_cast<uint32_t>(std::ceil(zLen * cellSizeInv_)));

	// overflow protection
	const uint64_t N_DimX = static_cast<uint64_t>(N_DimX_);
	const uint64_t N_DimY = static_cast<uint64_t>(N_DimY_);
	const uint64_t N_DimZ = static_cast<uint64_t>(N_DimZ_);
	const uint64_t N_Cells = N_DimX * N_DimY * N_DimZ;
	if (N_Cells > std::numeric_limits<uint32_t>::max())
	{
		// SPDLOG_ERROR("Too many grid cells for uint32_t ({})", N_Cells);
		std::cerr << "Error: Too many grid cells for uint32_t (" << N_Cells << ")\n";
	}
	else
	{
		N_Cells_ = static_cast<uint32_t>(N_Cells);
	}

	// --- [ preparation data structures ] -----------------------------------------------------------

	EnsureCapacity(be_.N_F_, be_.N_B_, N_Cells_);

	GPU01::NS_ReloadConstants(
		be_.N_F_, be_.N_B_, params.kernelRadius, cellSize_, xBoxOrig_, yBoxOrig_, zBoxOrig_,
		N_DimX_, N_DimY_, N_DimZ_, N_Cells_, EMPTY, be_.stream_);

	// --- [ compute cell IDs and init sorted indices ] ----------------------------------------------

	if (hasF)
	{
		GPU01::Launch_FillCellIdAndIdxF(
			be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
			be_.N_F_, be_.stream_, d_cellIdF_.Data());

		// sortedParticleIdxF_[i] = i
		thrust::sequence(
			thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_sortedParticleIdxF_.Data()),
			thrust::device_pointer_cast(d_sortedParticleIdxF_.Data() + be_.N_F_),
			0u);
	}

	if (hasB)
	{
		GPU01::Launch_FillCellIdAndIdxB(
			be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
			be_.N_B_, be_.stream_, d_cellIdB_.Data());

		// sortedParticleIdxB_[i] = i
		thrust::sequence(
			thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_sortedParticleIdxB_.Data()),
			thrust::device_pointer_cast(d_sortedParticleIdxB_.Data() + be_.N_B_), 
			0u);
	}

	// --- [ sort particle indices by cell IDs ] -----------------------------------------------------

	if (hasF)
	{
		// using radix sort from thrust
		thrust::sort_by_key(thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_cellIdF_.Data()),
			thrust::device_pointer_cast(d_cellIdF_.Data() + be_.N_F_),
			thrust::device_pointer_cast(d_sortedParticleIdxF_.Data()));
	}

	if (hasB)
	{
		// using radix sort from thrust
		thrust::sort_by_key(thrust::cuda::par.on(be_.stream_),
			thrust::device_pointer_cast(d_cellIdB_.Data()),
			thrust::device_pointer_cast(d_cellIdB_.Data() + be_.N_B_),
			thrust::device_pointer_cast(d_sortedParticleIdxB_.Data()));
	}

	// --- [ fill in cell start/end indices ] --------------------------------------------------------

	if (N_Cells_ > 0)
	{
		if (hasF)
		{
			// clear to EMPTY
			GPU01::Launch_ClearRanges(
				N_Cells_, be_.stream_,
				d_cellStartIdxF_.Data(), d_cellEndIdxF_.Data());
		}

		if (hasB)
		{
			// clear to EMPTY
			GPU01::Launch_ClearRanges(
				N_Cells_, be_.stream_,
				d_cellStartIdxB_.Data(), d_cellEndIdxB_.Data());
		}
	}

	if (hasF)
	{
		GPU01::Launch_BuildRangesFromSortedKeys(
			d_cellIdF_.Data(), be_.N_F_, be_.stream_,
			d_cellStartIdxF_.Data(), d_cellEndIdxF_.Data());
	}

	if (hasB)
	{
		GPU01::Launch_BuildRangesFromSortedKeys(
			d_cellIdB_.Data(), be_.N_B_, be_.stream_,
			d_cellStartIdxB_.Data(), d_cellEndIdxB_.Data());
	}
}

void NS_CellLinkedList_Basic_GPU01::Query_Count(const SimParams& params)
{
	// reset per-particle counts before count-query
	GPU::MemsetZero(be_.d_nbrCountFF_.Data(), be_.N_F_, be_.stream_);
	GPU::MemsetZero(be_.d_nbrCountFB_.Data(), be_.N_F_, be_.stream_);

	GPU01::Launch_CellLinkedList_Basic_Query_Count(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		d_cellStartIdxF_.Data(), d_cellEndIdxF_.Data(),
		d_cellStartIdxB_.Data(), d_cellEndIdxB_.Data(),
		d_sortedParticleIdxF_.Data(), d_sortedParticleIdxB_.Data(),
		be_.N_F_, be_.N_B_, be_.stream_,
		be_.d_nbrCountFF_.Data(), be_.d_nbrCountFB_.Data());

	GPU::StreamSync(be_.stream_);
}

void NS_CellLinkedList_Basic_GPU01::Query_Fill(const SimParams& params)
{
	// (count-query and fill-query MUST find the same number of neighbors)
	// reset overflow before filling packed CSR lists
	GPU::MemsetZero(be_.d_nbrOverflowFlag_.Data(), 1, be_.stream_);

	GPU01::Launch_CellLinkedList_Basic_Query_Fill(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		d_cellStartIdxF_.Data(), d_cellEndIdxF_.Data(),
		d_cellStartIdxB_.Data(), d_cellEndIdxB_.Data(),
		d_sortedParticleIdxF_.Data(), d_sortedParticleIdxB_.Data(),
		be_.N_F_, be_.N_B_, be_.stream_,
		be_.d_nbrOffsetsFF_.Data(), be_.d_nbrOffsetsFB_.Data(),
		be_.d_nbrListFF_.Data(), be_.d_nbrListFB_.Data(),
		be_.d_nbrOverflowFlag_.Data());

	GPU::StreamSync(be_.stream_);
}

NeighborSpan NS_CellLinkedList_Basic_GPU01::QueryParticleFF(const uint32_t i) const
{
	(void)i;
	return NeighborSpan{};
}

NeighborSpan NS_CellLinkedList_Basic_GPU01::QueryParticleFB(const uint32_t i) const
{
	(void)i;
	return NeighborSpan{};
}

void NS_CellLinkedList_Basic_GPU01::DataReOrder(const SimParams& params)
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

		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_xAccNonPressF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_yAccNonPressF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_zAccNonPressF_, be_.d_workBuffer_, be_.stream_);

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

std::pair<Mem::MemUsage, Mem::MemUsage> NS_CellLinkedList_Basic_GPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_cellStartIdxF_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_cellEndIdxF_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_cellIdF_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_sortedParticleIdxF_));

	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_cellStartIdxB_));
	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_cellEndIdxB_));
	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_cellIdB_));
	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_sortedParticleIdxB_));

	return { usageF, usageB };
}

// ------------------------------------------------------------------------------------------------
// --- [ cell linked list compact implementation ] ------------------------------------------------
// ------------------------------------------------------------------------------------------------

void NS_CellLinkedList_Compact_GPU01::Init(const SimParams& params)
{
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;
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

	cellSize_ = params.kernelRadius;
	cellSizeInv_ = Base::ApproxEqu(params.kernelRadius, 0.0f)
				 ? 0.0f : (1.0f / params.kernelRadius);

	EnsureStorageAllocation();

	GPU01::NS_ReloadConstants(
		be_.N_F_, be_.N_B_, params.kernelRadius, cellSize_, 0.0f, 0.0f, 0.0f,
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

		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_xAccNonPressF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_yAccNonPressF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_zAccNonPressF_, be_.d_workBuffer_, be_.stream_);

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

// ------------------------------------------------------------------------------------------------
// --- [ LBVH implementation ] --------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

void NS_BoundVolHier_Linear_LBVH_GPU01::Init(const SimParams& params)
{
	(void)params;

	d_mortonF_.Release();
	d_sortedParticleIdxF_.Release();
	d_nodesF_.Release();

	d_mortonB_.Release();
	d_sortedParticleIdxB_.Release();
	d_nodesB_.Release();

	d_leafBoxScratch_.Release();
	d_sortedLeafBoxScratch_.Release();
	d_parentOfScratch_.Release();
	d_visitCountScratch_.Release();

	boundaryBVHValid_ = false;
	cachedBoundaryCount_ = 0;
	cachedBoundaryXPosPtr_ = nullptr;
	cachedBoundaryYPosPtr_ = nullptr;
	cachedBoundaryZPosPtr_ = nullptr;
}

void NS_BoundVolHier_Linear_LBVH_GPU01::Construct(const SimParams& params)
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;

	if (N_F == 0 && N_B == 0)
	{
		std::cerr << "Warning: No construction possible (empty simulation)\n";

		d_mortonF_.Release();
		d_sortedParticleIdxF_.Release();
		d_nodesF_.Release();

		d_mortonB_.Release();
		d_sortedParticleIdxB_.Release();
		d_nodesB_.Release();

		d_leafBoxScratch_.Release();
		d_sortedLeafBoxScratch_.Release();
		d_parentOfScratch_.Release();
		d_visitCountScratch_.Release();

		boundaryBVHValid_ = false;
		cachedBoundaryCount_ = 0;
		cachedBoundaryXPosPtr_ = nullptr;
		cachedBoundaryYPosPtr_ = nullptr;
		cachedBoundaryZPosPtr_ = nullptr;

		return;
	}

	GPU01::NS_ReloadConstants(
		be_.N_F_, be_.N_B_, params.kernelRadius, 0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, be_.stream_);

	// build fluid BVH
	if (N_F > 0)
	{
		BuildBVH(
			be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
			N_F, d_mortonF_, d_sortedParticleIdxF_, d_nodesF_);
	}
	else
	{
		d_mortonF_.Release();
		d_sortedParticleIdxF_.Release();
		d_nodesF_.Release();
	}

	// build boundary BVH
	if (N_B > 0)
	{
		const float* xPosB = be_.d_xPosB_.Data();
		const float* yPosB = be_.d_yPosB_.Data();
		const float* zPosB = be_.d_zPosB_.Data();

		// only rebuild boundary BVH if storage/layout changed
		const bool boundaryLayoutChanged =
			(cachedBoundaryCount_ != N_B) ||
			(cachedBoundaryXPosPtr_ != xPosB) ||
			(cachedBoundaryYPosPtr_ != yPosB) ||
			(cachedBoundaryZPosPtr_ != zPosB);

		if (!boundaryBVHValid_ || boundaryLayoutChanged)
		{
			BuildBVH(
				xPosB, yPosB, zPosB,
				N_B, d_mortonB_, d_sortedParticleIdxB_, d_nodesB_);

			boundaryBVHValid_ = true;
			cachedBoundaryCount_ = N_B;
			cachedBoundaryXPosPtr_ = xPosB;
			cachedBoundaryYPosPtr_ = yPosB;
			cachedBoundaryZPosPtr_ = zPosB;
		}
	}
	else
	{
		d_mortonB_.Release();
		d_sortedParticleIdxB_.Release();
		d_nodesB_.Release();

		boundaryBVHValid_ = false;
		cachedBoundaryCount_ = 0;
		cachedBoundaryXPosPtr_ = nullptr;
		cachedBoundaryYPosPtr_ = nullptr;
		cachedBoundaryZPosPtr_ = nullptr;
	}
}

void NS_BoundVolHier_Linear_LBVH_GPU01::BuildBVH(
	const float* xPos, const float* yPos, const float* zPos,
	uint32_t N,
	DeviceBuffer<uint32_t>& d_morton,
	DeviceBuffer<uint32_t>& d_sortedParticleIdx,
	DeviceBuffer<GPU01::LBVH_InternalNode>& d_nodes)
{
	if (N == 0)
	{
		d_morton.Release();
		d_sortedParticleIdx.Release();
		d_nodes.Release();
		return;
	}

	if (d_morton.Size() != N) { d_morton.Resize(N); }
	if (d_sortedParticleIdx.Size() != N) { d_sortedParticleIdx.Resize(N); }
	if (d_leafBoxScratch_.Size() != N) { d_leafBoxScratch_.Resize(N); }
	if (d_sortedLeafBoxScratch_.Size() != N) { d_sortedLeafBoxScratch_.Resize(N); }

	// --- [ compute scene AABB for Morton code normalization ] ------------------------------------

	const float initMin = std::numeric_limits<float>::max();
	const float initMax = std::numeric_limits<float>::lowest();
	MinMax3 sceneAABB =
	{
		.xMin = initMin, .yMin = initMin, .zMin = initMin,
		.xMax = initMax, .yMax = initMax, .zMax = initMax
	};

	auto X = thrust::device_pointer_cast(xPos);
	auto Y = thrust::device_pointer_cast(yPos);
	auto Z = thrust::device_pointer_cast(zPos);
	auto first = thrust::make_zip_iterator(thrust::make_tuple(X, Y, Z));
	auto last = first + N;

	sceneAABB = thrust::transform_reduce(
		thrust::cuda::par.on(be_.stream_),
		first, last,
		GPU01::ToMinMax3{},
		sceneAABB,
		GPU01::ReduceMinMax3{});

	// add small epsilon to avoid division by zero on degenerate axes
	constexpr float eps = 1e-6f;
	const float extX = std::max(eps, sceneAABB.xMax - sceneAABB.xMin);
	const float extY = std::max(eps, sceneAABB.yMax - sceneAABB.yMin);
	const float extZ = std::max(eps, sceneAABB.zMax - sceneAABB.zMin);
	const float invExtX = 1.0f / extX;
	const float invExtY = 1.0f / extY;
	const float invExtZ = 1.0f / extZ;

	// --- [ compute Morton codes and build leaf AABBs ] -------------------------------------------

	GPU01::Launch_LBVH_FillMortonAndAABB(
		xPos, yPos, zPos, N,
		sceneAABB.xMin, sceneAABB.yMin, sceneAABB.zMin,
		invExtX, invExtY, invExtZ,
		be_.stream_,
		d_morton.Data(), d_sortedParticleIdx.Data(), d_leafBoxScratch_.Data());

	// --- [ sort particles by Morton code ] -------------------------------------------------------

	thrust::stable_sort_by_key(
		thrust::cuda::par.on(be_.stream_),
		thrust::device_pointer_cast(d_morton.Data()),
		thrust::device_pointer_cast(d_morton.Data() + N),
		thrust::device_pointer_cast(d_sortedParticleIdx.Data()));

	// reorder leaf boxes to sorted order
	thrust::gather(
		thrust::cuda::par.on(be_.stream_),
		thrust::device_pointer_cast(d_sortedParticleIdx.Data()),
		thrust::device_pointer_cast(d_sortedParticleIdx.Data() + N),
		thrust::device_pointer_cast(d_leafBoxScratch_.Data()),
		thrust::device_pointer_cast(d_sortedLeafBoxScratch_.Data()));

	// handle trivial case: single particle
	if (N <= 1)
	{
		d_nodes.Release();
		GPU::StreamSync(be_.stream_);
		return;
	}

	// --- [ build radix tree (Karras 2012) ] ------------------------------------------------------

	const uint32_t N_internal = N - 1;
	if (d_nodes.Size() != N_internal) { d_nodes.Resize(N_internal); }
	if (d_parentOfScratch_.Size() != N_internal + N) { d_parentOfScratch_.Resize(N_internal + N); }
	if (d_visitCountScratch_.Size() != N_internal) { d_visitCountScratch_.Resize(N_internal); }

	GPU01::Launch_LBVH_BuildInternalNodes(
		d_morton.Data(), N, be_.stream_, d_nodes.Data());

	// parent lookup is initialized to invalid parent
	GPU::MemsetBytes(d_parentOfScratch_.Data(), 0xFF, N_internal + N, be_.stream_);

	GPU01::Launch_LBVH_BuildParentLookupAndResetVisits(
		d_nodes.Data(), N_internal, N, be_.stream_,
		d_parentOfScratch_.Data(), d_visitCountScratch_.Data());

	// --- [ bottom-up AABB propagation ] ----------------------------------------------------------

	GPU01::Launch_LBVH_PropagateAABBs(
		d_sortedLeafBoxScratch_.Data(),
		d_parentOfScratch_.Data(),
		N, N_internal, be_.stream_,
		d_nodes.Data(),
		d_visitCountScratch_.Data());

	GPU::StreamSync(be_.stream_);
}

void NS_BoundVolHier_Linear_LBVH_GPU01::Query_Count(const SimParams& params)
{
	(void)params;

	// reset per-particle counts before count-query
	GPU::MemsetZero(be_.d_nbrCountFF_.Data(), be_.N_F_, be_.stream_);
	GPU::MemsetZero(be_.d_nbrCountFB_.Data(), be_.N_F_, be_.stream_);

	GPU01::Launch_LBVH_Query_Count(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		d_sortedParticleIdxF_.Data(), d_sortedParticleIdxB_.Data(),
		d_nodesF_.Data(), d_nodesB_.Data(),
		be_.N_F_, be_.N_B_, be_.stream_,
		be_.d_nbrCountFF_.Data(), be_.d_nbrCountFB_.Data());

	GPU::StreamSync(be_.stream_);
}

void NS_BoundVolHier_Linear_LBVH_GPU01::Query_Fill(const SimParams& params)
{
	(void)params;

	// (count-query and fill-query MUST find the same number of neighbors)
	// reset overflow before filling packed CSR lists
	GPU::MemsetZero(be_.d_nbrOverflowFlag_.Data(), 1, be_.stream_);

	GPU01::Launch_LBVH_Query_Fill(
		be_.d_xPosF_.Data(), be_.d_yPosF_.Data(), be_.d_zPosF_.Data(),
		be_.d_xPosB_.Data(), be_.d_yPosB_.Data(), be_.d_zPosB_.Data(),
		d_sortedParticleIdxF_.Data(), d_sortedParticleIdxB_.Data(),
		d_nodesF_.Data(), d_nodesB_.Data(),
		be_.N_F_, be_.N_B_, be_.stream_,
		be_.d_nbrOffsetsFF_.Data(), be_.d_nbrOffsetsFB_.Data(),
		be_.d_nbrListFF_.Data(), be_.d_nbrListFB_.Data(),
		be_.d_nbrOverflowFlag_.Data());

	GPU::StreamSync(be_.stream_);
}

NeighborSpan NS_BoundVolHier_Linear_LBVH_GPU01::QueryParticleFF(const uint32_t i) const
{
	(void)i;
	return NeighborSpan{};
}

NeighborSpan NS_BoundVolHier_Linear_LBVH_GPU01::QueryParticleFB(const uint32_t i) const
{
	(void)i;
	return NeighborSpan{};
}

void NS_BoundVolHier_Linear_LBVH_GPU01::DataReOrder(const SimParams& params)
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

		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_xAccNonPressF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_yAccNonPressF_, be_.d_workBuffer_, be_.stream_);
		ApplyPermutation_DeviceBuffer(d_permF, N_F, be_.d_zAccNonPressF_, be_.d_workBuffer_, be_.stream_);

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

	// particle arrays are permuted; cached boundary tree/index mapping is no longer valid
	boundaryBVHValid_ = false;

	GPU::StreamSync(be_.stream_);
}

std::pair<Mem::MemUsage, Mem::MemUsage> NS_BoundVolHier_Linear_LBVH_GPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_mortonF_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_sortedParticleIdxF_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_nodesF_));

	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_mortonB_));
	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_sortedParticleIdxB_));
	Mem::Add(usageB, MemUsageOf_DeviceBuffer(d_nodesB_));

	// shared construction scratch is accounted in the fluid-side bucket
	// to avoid double-counting.
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_leafBoxScratch_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_sortedLeafBoxScratch_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_parentOfScratch_));
	Mem::Add(usageF, MemUsageOf_DeviceBuffer(d_visitCountScratch_));

	return { usageF, usageB };
}
