// neighbor_search.cpp

// external
#include <cmath>
#include <spdlog/spdlog.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

// internal
#include "Source/Sim/GPU/gpu_utils.h"
#include "neighbor_search.h"
#include "neighbor_search_kernels.cuh"



using namespace Sim;

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
		SPDLOG_WARN("No construction possible (empty simulation)");

		N_DimX_ = 0;
		N_DimY_ = 0;
		N_DimZ_ = 0;
		N_Cells_ = 0;

		return;
	}

	cellSize_ = params.kernelRadius;
	cellSizeInv_ = 1.0f / params.kernelRadius;

	// --- [ combined AABB over fluid and boundary particles ] -----------------

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
			ToMinMax3{},
			h_aabbF_,
			ReduceMinMax3{});
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
			ToMinMax3{},
			h_aabbB_,
			ReduceMinMax3{});
	}

	MinMax3 box;
	if (hasF && hasB)
	{
		box.xMin = fminf(h_aabbF_.xMin, h_aabbB_.xMin);
		box.yMin = fminf(h_aabbF_.yMin, h_aabbB_.yMin);
		box.zMin = fminf(h_aabbF_.zMin, h_aabbB_.zMin);
		box.xMax = fmaxf(h_aabbF_.xMax, h_aabbB_.xMax);
		box.yMax = fmaxf(h_aabbF_.yMax, h_aabbB_.yMax);
		box.zMax = fmaxf(h_aabbF_.zMax, h_aabbB_.zMax);
	}

	// padding to avoid edge clamp issues
	const float pad = 1e-6f;
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
		SPDLOG_ERROR("Too many grid cells for uint32_t ({})", N_Cells);
	}
	else
	{
		N_Cells_ = static_cast<uint32_t>(N_Cells);
	}

	// --- [ preparation data structures ] -------------------------------------

	EnsureCapacity(be_.N_F_, be_.N_B_, N_Cells_);

	ReloadConstants(
		cellSize_, xBoxOrig_, yBoxOrig_, zBoxOrig_, 
		N_DimX_, N_DimY_, N_DimZ_, N_Cells_, be_.stream_);

	// --- [ compute cell IDs and init sorted indices ] ------------------------

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

	// --- [ sort particle indices by cell IDs ] -------------------------------

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

	// --- [ fill in cell start/end indices ] ----------------------------------

	if (N_Cells_ > 0)
	{
		if (hasF)
		{
			// clear to EMPTY
			GPU01::Launch_ClearRanges(
				EMPTY, N_Cells_, be_.stream_,
				d_cellStartIdxF_.Data(), d_cellEndIdxF_.Data());
		}

		if (hasB)
		{
			// clear to EMPTY
			GPU01::Launch_ClearRanges(
				EMPTY, N_Cells_, be_.stream_,
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
