// neighbor_search.cpp

// external
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <spdlog/spdlog.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/partitioner.h>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// internal
#include "cpu_backend_01.h"
#include "neighbor_lists.h"
#include "neighbor_search.h"
#include "Source/Base/helpers.h"

#if defined(_MSC_VER)
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif



using namespace Sim;

namespace
{
	// TODO: rework this?
	template<typename TIdx, typename TData>
	void ApplyPermutation(
		const std::vector<TIdx>& sortedKeys,
		std::vector<TData>& data,
		std::vector<TData>& scratch,
		bool preserveDataCapacity = false)
	{
		// (capped at uint32_t to correspond to max particle count)
		const uint32_t N = static_cast<uint32_t>(sortedKeys.size());
		if (N == 0) { return; }

		assert(data.size() == N && "Permutation and data vector size mismatch");
		scratch.resize(N);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const uint32_t srcIdx = static_cast<uint32_t>(sortedKeys[i]);
					scratch[i] = data[srcIdx];
				}
			},
			tbb::auto_partitioner{}
		);

		if (preserveDataCapacity)
		{
			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0, N),
				[&](const tbb::blocked_range<uint32_t>& r)
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						data[i] = scratch[i];
					}
				},
				tbb::auto_partitioner{}
			);
		}
		else
		{
			data.swap(scratch);
		}
	}

	inline int FloorToInt(float v) noexcept
	{
		int i = static_cast<int>(v);

		return (v < static_cast<float>(i)) ? (i - 1) : i;
	}

	struct SceneAABB
	{
		float minX = std::numeric_limits<float>::max();
		float minY = std::numeric_limits<float>::max();
		float minZ = std::numeric_limits<float>::max();
		float maxX = std::numeric_limits<float>::lowest();
		float maxY = std::numeric_limits<float>::lowest();
		float maxZ = std::numeric_limits<float>::lowest();
	};

	struct CellCoordFrame
	{
		int cellOrigX = 0;
		int cellOrigY = 0;
		int cellOrigZ = 0;
		uint32_t N_DimX = 0;
		uint32_t N_DimY = 0;
		uint32_t N_DimZ = 0;
	};

	// TODO: parallelize?
	inline SceneAABB ComputeSceneAABB_Combined(
		const float* xPosF, const float* yPosF, const float* zPosF, uint32_t N_F,
		const float* xPosB, const float* yPosB, const float* zPosB, uint32_t N_B)
	{
		SceneAABB aabb{};

		for (uint32_t i = 0; i < N_F; i++)
		{
			const float x = xPosF[i];
			const float y = yPosF[i];
			const float z = zPosF[i];

			aabb.minX = std::min(aabb.minX, x);
			aabb.minY = std::min(aabb.minY, y);
			aabb.minZ = std::min(aabb.minZ, z);
			aabb.maxX = std::max(aabb.maxX, x);
			aabb.maxY = std::max(aabb.maxY, y);
			aabb.maxZ = std::max(aabb.maxZ, z);
		}

		for (uint32_t i = 0; i < N_B; i++)
		{
			const float x = xPosB[i];
			const float y = yPosB[i];
			const float z = zPosB[i];

			aabb.minX = std::min(aabb.minX, x);
			aabb.minY = std::min(aabb.minY, y);
			aabb.minZ = std::min(aabb.minZ, z);
			aabb.maxX = std::max(aabb.maxX, x);
			aabb.maxY = std::max(aabb.maxY, y);
			aabb.maxZ = std::max(aabb.maxZ, z);
		}

		return aabb;
	}

	inline void ExpandSceneAABB(
		SceneAABB& aabb,
		float pad) noexcept
	{
		aabb.minX -= pad;
		aabb.minY -= pad;
		aabb.minZ -= pad;
		aabb.maxX += pad;
		aabb.maxY += pad;
		aabb.maxZ += pad;
	}

	inline CellCoordFrame ComputeCellCoordFrame_FromAABB(
		const SceneAABB& aabb,
		float cellSizeInv)
	{
		CellCoordFrame frame{};

		const int cxMin = FloorToInt(aabb.minX * cellSizeInv);
		const int cyMin = FloorToInt(aabb.minY * cellSizeInv);
		const int czMin = FloorToInt(aabb.minZ * cellSizeInv);
		const int cxMax = FloorToInt(aabb.maxX * cellSizeInv);
		const int cyMax = FloorToInt(aabb.maxY * cellSizeInv);
		const int czMax = FloorToInt(aabb.maxZ * cellSizeInv);

		frame.cellOrigX = cxMin;
		frame.cellOrigY = cyMin;
		frame.cellOrigZ = czMin;
		frame.N_DimX = static_cast<uint32_t>(cxMax - cxMin + 1);
		frame.N_DimY = static_cast<uint32_t>(cyMax - cyMin + 1);
		frame.N_DimZ = static_cast<uint32_t>(czMax - czMin + 1);

		return frame;
	}

	// TODO: parallelize this?
	void BuildCompactCellTable(
		const std::vector<uint64_t>& cellId,
		std::vector<uint64_t>& uniqueCellId,
		std::vector<uint32_t>& cellStartIdx)
	{
		const uint32_t N = static_cast<uint32_t>(cellId.size());
		uniqueCellId.clear();
		cellStartIdx.clear();
		if (N == 0) { return; }

		// first count the number of occupied cells so both output arrays can be sized exactly
		uint32_t N_Cells = 1;
		for (uint32_t i = 1; i < N; i++)
		{
			if (cellId[i] != cellId[i - 1]) { N_Cells++; }
		}

		uniqueCellId.resize(N_Cells);
		cellStartIdx.resize(N_Cells + 1); // +1 for sentinel

		uint32_t c = 0;
		uniqueCellId[c] = cellId[0];
		cellStartIdx[c] = 0;

		for (uint32_t i = 1; i < N; i++)
		{
			const uint64_t cid = cellId[i];

			if (cid != cellId[i - 1]) // new cell encountered
			{
				c++;
				uniqueCellId[c] = cid;
				cellStartIdx[c] = i;
			}
		}

		// end of the last cell's range as sentinel
		cellStartIdx[N_Cells] = N;
	}

} // unnamed

// -------------------------------------------------------------------------------------------------
// --- [ brute force implementation ] --------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_BruteForce_CPU01::Init(const SimParams& params)
{
	kernelRadiusSqu_ = params.kernelRadiusSqu;
}

void NS_BruteForce_CPU01::Construct(const SimParams& params)
{
	kernelRadiusSqu_ = params.kernelRadiusSqu;
}

void NS_BruteForce_CPU01::Query_Count(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				uint32_t cntFF = 0;
				uint32_t cntFB = 0;

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				// check all fluid particles j
				for (uint32_t j = 0; j < N_F; j++)
				{
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// squared distance between particles i and j
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						cntFF++;
					}
				}

				// check all boundary particles j
				for (uint32_t j = 0; j < N_B; j++)
				{
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// squared distance between particles i and j
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						cntFB++;
					}
				}

				// commit final neighbor counts to backend storage
				be_.neighborList_->SetNbrCountFF(i, cntFF);
				be_.neighborList_->SetNbrCountFB(i, cntFB);
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_BruteForce_CPU01::Query_Fill(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			// per-thread neighbor vectors for local accumulation and then bulk add
			// (reserve preparation in case of inline capacity not being enough)
			auto& scratchFF = scratchFF_.local();
			auto& scratchFB = scratchFB_.local();
			if (scratchFF.capacity() < params.expctdNbrCountHardMax_FF)
			{
				scratchFF.reserve(params.expctdNbrCountHardMax_FF);
			}
			if (scratchFB.capacity() < params.expctdNbrCountHardMax_FB)
			{
				scratchFB.reserve(params.expctdNbrCountHardMax_FB);
			}

			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// reset neighbor list for particle i
				// (delete contents, but keep capacity)
				scratchFF.clear();
				scratchFB.clear();

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				// check all fluid particles j
				for (uint32_t j = 0; j < N_F; j++)
				{
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// squared distance between particles i and j
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						// accumulate actual neighbors in scratch list
						scratchFF.push_back(j);
					}
				}

				// check all boundary particles j
				for (uint32_t j = 0; j < N_B; j++)
				{
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// squared distance between particles i and j
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						// accumulate actual neighbors in scratch list
						scratchFB.push_back(j);
					}
				}

				// commit final neighbor lists from scratch to backend storage
				be_.neighborList_->SetNbrListFF(i, scratchFF.data(),
					static_cast<uint32_t>(scratchFF.size()));
				be_.neighborList_->SetNbrListFB(i, scratchFB.data(),
					static_cast<uint32_t>(scratchFB.size()));
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_BruteForce_CPU01::DataReOrder(
	const SimParams& params)
{
	(void)params;

	// no re-ordering in brute force implementation
}

std::pair<Mem::MemUsage, Mem::MemUsage> NS_BruteForce_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

	return { usageF, usageB };
}

// -------------------------------------------------------------------------------------------------
// --- [ cell linked list compact implementation ] -------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_CellLinkedList_Compact_CPU01::Init(const SimParams& params)
{
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;
	kernelRadiusSqu_ = params.kernelRadiusSqu;
	cellQueryReach_ = 1;
	cellOrigX_ = 0;
	cellOrigY_ = 0;
	cellOrigZ_ = 0;
	N_DimX_ = 0;
	N_DimY_ = 0;
	N_DimZ_ = 0;
	cellFrameValid_ = false;
	cellIdsMatchCurrentParticleOrder_ = false;

	uniqueCellIdF_.clear();
	cellStartIdxF_.clear();
	cellIdF_.clear();
	particleOrderF_.clear();

	uniqueCellIdB_.clear();
	cellStartIdxB_.clear();
	cellIdB_.clear();
	particleOrderB_.clear();
}

void NS_CellLinkedList_Compact_CPU01::Construct(const SimParams& params)
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;

	if (N_F == 0 && N_B == 0)
	{
		SPDLOG_WARN("No construction possible (empty simulation)");
		cellOrigX_ = 0;
		cellOrigY_ = 0;
		cellOrigZ_ = 0;
		N_DimX_ = 0;
		N_DimY_ = 0;
		N_DimZ_ = 0;
		cellFrameValid_ = false;
		cellIdsMatchCurrentParticleOrder_ = false;

		uniqueCellIdF_.clear();
		cellStartIdxF_.clear();
		cellIdF_.clear();
		particleOrderF_.clear();

		uniqueCellIdB_.clear();
		cellStartIdxB_.clear();
		cellIdB_.clear();
		particleOrderB_.clear();

		return;
	}

	const float cellSize = params.kernelRadius * params.gridCellSizeRatio;
	const float querySpanCells = Base::ApproxEqu(cellSize, 0.0f)
		? 0.0f : (params.kernelRadius / cellSize);
	const uint32_t cellQueryReach = Base::ApproxEqu(cellSize, 0.0f)
		? 1u : std::max<uint32_t>(1u,
			static_cast<uint32_t>(std::ceil(std::max(0.0f, querySpanCells - 1e-6f))));

	const bool cellSizeChanged = !Base::ApproxEqu(cellSize_, cellSize);
	const bool cellQueryReachChanged = (cellQueryReach_ != cellQueryReach);

	kernelRadiusSqu_ = params.kernelRadiusSqu;
	cellSize_ = cellSize;
	cellSizeInv_ = Base::ApproxEqu(cellSize, 0.0f)
		? 0.0f : (1.0f / cellSize);
	cellQueryReach_ = cellQueryReach;

	if (cellSizeChanged || cellQueryReachChanged) { cellFrameValid_ = false; }
	UpdateCellFrame();

	// --- [ cell coordinates and indices for fluid and boundary particles ] --------------------------

	// if DataReOrder() already computed and permuted the per-particle cell IDs for
	// this step, then they can be reused directly here and no re-computation is needed
	const bool canReuseCellIds =
		cellIdsMatchCurrentParticleOrder_
		&& cellIdF_.size() == N_F
		&& cellIdB_.size() == N_B;

	// fluid
	if (!canReuseCellIds)
	{
		cellIdF_.resize(N_F);
		cellIdB_.resize(N_B);

		if (N_F > 0)
		{
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N_F),
				[&](const tbb::blocked_range<uint32_t>& r)
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						// compute cell coordinates
						const int cx = FloorToInt(be_.xPosF_[i] * cellSizeInv_);
						const int cy = FloorToInt(be_.yPosF_[i] * cellSizeInv_);
						const int cz = FloorToInt(be_.zPosF_[i] * cellSizeInv_);

						cellIdF_[i] = MapToCellId(cx, cy, cz);
					}
				},
				tbb::auto_partitioner{}
			);
		}

		if (N_B > 0)
		{
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N_B),
				[&](const tbb::blocked_range<uint32_t>& r)
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						// compute cell coordinates
						const int cx = FloorToInt(be_.xPosB_[i] * cellSizeInv_);
						const int cy = FloorToInt(be_.yPosB_[i] * cellSizeInv_);
						const int cz = FloorToInt(be_.zPosB_[i] * cellSizeInv_);

						cellIdB_[i] = MapToCellId(cx, cy, cz);
					}
				},
				tbb::auto_partitioner{}
			);
		}

		// cell IDs now correspond to the current particle order again
		cellIdsMatchCurrentParticleOrder_ = true;
	}

	// --- [ build compact cell list and fill in cell start indices ] ---------------------------------

	// fluid
	BuildCompactCellTable(cellIdF_, uniqueCellIdF_, cellStartIdxF_);

	// boundary
	BuildCompactCellTable(cellIdB_, uniqueCellIdB_, cellStartIdxB_);
}

void NS_CellLinkedList_Compact_CPU01::Query_Count(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadiusSqu;
	const int cellQueryReach = static_cast<int>(cellQueryReach_);
	const uint32_t N_NbrCells = NeighborCellCount();

	// local references for fewer pointer indirections
	float* RESTRICT xPosF = be_.xPosF_.data();
	float* RESTRICT yPosF = be_.yPosF_.data();
	float* RESTRICT zPosF = be_.zPosF_.data();
	float* RESTRICT xPosB = be_.xPosB_.data();
	float* RESTRICT yPosB = be_.yPosB_.data();
	float* RESTRICT zPosB = be_.zPosB_.data();

	// loop over all non-empty grid cells
	const uint64_t N_Cells = static_cast<uint64_t>(uniqueCellIdF_.size());
	tbb::parallel_for(
		tbb::blocked_range<uint64_t>(0, N_Cells),
		[&](const tbb::blocked_range<uint64_t>& range)
		{
			static constexpr uint32_t LookupCacheSize = 256;
			static_assert((LookupCacheSize & (LookupCacheSize - 1)) == 0);

			std::array<uint64_t, LookupCacheSize> lookupCacheKeyF{};
			std::array<uint32_t, LookupCacheSize> lookupCacheValF{};
			std::array<uint64_t, LookupCacheSize> lookupCacheKeyB{};
			std::array<uint32_t, LookupCacheSize> lookupCacheValB{};
			lookupCacheKeyF.fill(EMPTY_64);
			lookupCacheKeyB.fill(EMPTY_64);
			lookupCacheValF.fill(EMPTY_32);
			lookupCacheValB.fill(EMPTY_32);

			auto& queryMetaScratch = queryMetaScratch_.local();
			auto& nbrCellId = queryMetaScratch.nbrCellId;
			auto& cellOrder = queryMetaScratch.cellOrder;
			auto& fluidRanges = queryMetaScratch.fluidRanges;
			auto& boundaryRanges = queryMetaScratch.boundaryRanges;
			nbrCellId.resize(N_NbrCells);
			cellOrder.resize(N_NbrCells);
			if (fluidRanges.capacity() < N_NbrCells) { fluidRanges.reserve(N_NbrCells); }
			if (boundaryRanges.capacity() < N_NbrCells) { boundaryRanges.reserve(N_NbrCells); }

			const auto findCellIdxTernary = [](const std::vector<uint64_t>& sortedCellIds,
				uint64_t key,
				uint32_t lo,
				uint32_t hi) noexcept
				{
					while (hi > lo + 24u)
					{
						const uint32_t span = hi - lo;
						const uint32_t third = span / 3u;
						const uint32_t m1 = lo + third;
						const uint32_t m2 = hi - third - 1u;

						const uint64_t v1 = sortedCellIds[m1];
						const uint64_t v2 = sortedCellIds[m2];

						if (key < v1)
						{
							hi = m1;
							continue;
						}
						if (key > v2)
						{
							lo = m2 + 1u;
							continue;
						}
						if (key == v1) { return m1; }
						if (key == v2) { return m2; }

						lo = m1 + 1u;
						hi = m2;
					}

					while (lo < hi)
					{
						const uint32_t mid = lo + ((hi - lo) >> 1u);
						if (sortedCellIds[mid] < key)
						{
							lo = mid + 1u;
						}
						else
						{
							hi = mid;
						}
					}

					if (lo < sortedCellIds.size() && sortedCellIds[lo] == key)
					{
						return lo;
					}

					return EMPTY_32;
				};

			for (uint64_t c = range.begin(); c != range.end(); c++)
			{
				// range of all fluid particles in this cell
				const uint32_t sF = cellStartIdxF_[c]; // start
				const uint32_t eF = cellStartIdxF_[c + 1]; // end
				if (sF >= eF) { continue; }

				// determine a representative particle i0
				const uint32_t i0 = sF;

				const float xi0 = xPosF[i0];
				const float yi0 = yPosF[i0];
				const float zi0 = zPosF[i0];

				// cell coordinates of all particles in this cell
				const int cx = FloorToInt(xi0 * cellSizeInv_);
				const int cy = FloorToInt(yi0 * cellSizeInv_);
				const int cz = FloorToInt(zi0 * cellSizeInv_);

				// BigMin/LitMax range of the queried neighborhood in cell ID space
				uint64_t litMaxCellId = std::numeric_limits<uint64_t>::max();
				uint64_t bigMinCellId = 0;

				for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
				{
					const int oz = cz + rz; // target cell coordinate in z-direction

					for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
					{
						const int oy = cy + ry; // target cell coordinate in y-direction

						for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
						{
							const int ox = cx + rx; // target cell coordinate in x-direction

							// index of this neighbor cell in the queried neighborhood
							const int idx_nbr = NeighborOffset(rx, ry, rz);

							// cell ID of this neighbor cell
							const uint64_t cid_nbr = MapToCellId(ox, oy, oz);
							nbrCellId[idx_nbr] = cid_nbr;
							litMaxCellId = std::min(litMaxCellId, cid_nbr);
							bigMinCellId = std::max(bigMinCellId, cid_nbr);
						}
					}
				}

				// fluid lower/upper bounds over the full neighborhood range
				uint32_t loF = 0;
				uint32_t hiF = 0;
				if (!uniqueCellIdF_.empty())
				{
					loF = static_cast<uint32_t>(
						std::ranges::lower_bound(uniqueCellIdF_, litMaxCellId) - uniqueCellIdF_.begin());
					hiF = static_cast<uint32_t>(
						std::ranges::upper_bound(uniqueCellIdF_, bigMinCellId) - uniqueCellIdF_.begin());
					if (hiF < loF) { hiF = loF; }
				}

				// boundary lower/upper bounds over the full neighborhood range
				uint32_t loB = 0;
				uint32_t hiB = 0;
				if (!uniqueCellIdB_.empty())
				{
					loB = static_cast<uint32_t>(
						std::ranges::lower_bound(uniqueCellIdB_, litMaxCellId) - uniqueCellIdB_.begin());
					hiB = static_cast<uint32_t>(
						std::ranges::upper_bound(uniqueCellIdB_, bigMinCellId) - uniqueCellIdB_.begin());
					if (hiB < loB) { hiB = loB; }
				}

				// sort queried cells by cell ID so that queried cells can be traversed in Z-order
				std::iota(cellOrder.begin(), cellOrder.end(), 0u);
				std::sort(cellOrder.begin(), cellOrder.end(),
					[&](const uint32_t a, const uint32_t b)
					{
						return nbrCellId[a] < nbrCellId[b];
					});

				fluidRanges.clear();
				boundaryRanges.clear();

				// check existence and location of all queried grid cells
				for (uint32_t q = 0; q < N_NbrCells; q++)
				{
					const uint32_t idx_nbr = cellOrder[q];
					const uint64_t cid_nbr = nbrCellId[idx_nbr];

					// fluid scope
					{
						const uint32_t slot = static_cast<uint32_t>(cid_nbr) & (LookupCacheSize - 1u);
						uint32_t cF_nbr = EMPTY_32;

						if (lookupCacheKeyF[slot] == cid_nbr)
						{
							cF_nbr = lookupCacheValF[slot];
						}
						else
						{
							if (loF < hiF)
							{
								cF_nbr = findCellIdxTernary(uniqueCellIdF_, cid_nbr, loF, hiF);
							}
							lookupCacheKeyF[slot] = cid_nbr;
							lookupCacheValF[slot] = cF_nbr;
						}

						if (cF_nbr != EMPTY_32)
						{
							const uint32_t sF_nbr = cellStartIdxF_[cF_nbr];
							const uint32_t eF_nbr = cellStartIdxF_[cF_nbr + 1];

							if (sF_nbr < eF_nbr)
							{
								fluidRanges.push_back({ sF_nbr, eF_nbr });
							}
						}
					}

					// boundary scope
					{
						const uint32_t slot = static_cast<uint32_t>(cid_nbr) & (LookupCacheSize - 1u);
						uint32_t cB_nbr = EMPTY_32;

						if (lookupCacheKeyB[slot] == cid_nbr)
						{
							cB_nbr = lookupCacheValB[slot];
						}
						else
						{
							if (loB < hiB)
							{
								cB_nbr = findCellIdxTernary(uniqueCellIdB_, cid_nbr, loB, hiB);
							}
							lookupCacheKeyB[slot] = cid_nbr;
							lookupCacheValB[slot] = cB_nbr;
						}

						if (cB_nbr != EMPTY_32)
						{
							const uint32_t sB_nbr = cellStartIdxB_[cB_nbr];
							const uint32_t eB_nbr = cellStartIdxB_[cB_nbr + 1];

							if (sB_nbr < eB_nbr)
							{
								boundaryRanges.push_back({ sB_nbr, eB_nbr });
							}
						}
					}
				}

				// check all particles j in the queried neighboring grid cells
				for (uint32_t k = sF; k < eF; k++)
				{
					uint32_t cntFF = 0;
					uint32_t cntFB = 0;

					const uint32_t i = k;

					const float xi = xPosF[i];
					const float yi = yPosF[i];
					const float zi = zPosF[i];

					// fluid neighbors
					for (const ActiveCellRange& cellRange : fluidRanges)
					{
						#pragma omp simd
						for (uint32_t l = cellRange.begin; l < cellRange.end; l++)
						{
							const uint32_t j = l;

							// position difference
							const float dx = xi - xPosF[j];
							const float dy = yi - yPosF[j];
							const float dz = zi - zPosF[j];

							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= kernelRadiusSqu) [[unlikely]]
							{
								cntFF++;
							}
						}
					}

					// boundary neighbors
					for (const ActiveCellRange& cellRange : boundaryRanges)
					{
						#pragma omp simd
						for (uint32_t l = cellRange.begin; l < cellRange.end; l++)
						{
							const uint32_t j = l;

							// position difference
							const float dx = xi - xPosB[j];
							const float dy = yi - yPosB[j];
							const float dz = zi - zPosB[j];

							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= kernelRadiusSqu) [[unlikely]]
							{
								cntFB++;
							}
						}
					}

					// commit final neighbor counts to backend storage
					be_.neighborList_->SetNbrCountFF(i, cntFF);
					be_.neighborList_->SetNbrCountFB(i, cntFB);
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_CellLinkedList_Compact_CPU01::Query_Fill(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadiusSqu;
	const int cellQueryReach = static_cast<int>(cellQueryReach_);
	const uint32_t N_NbrCells = NeighborCellCount();

	// local references for fewer pointer indirections
	float* RESTRICT xPosF = be_.xPosF_.data();
	float* RESTRICT yPosF = be_.yPosF_.data();
	float* RESTRICT zPosF = be_.zPosF_.data();
	float* RESTRICT xPosB = be_.xPosB_.data();
	float* RESTRICT yPosB = be_.yPosB_.data();
	float* RESTRICT zPosB = be_.zPosB_.data();

	// loop over all non-empty grid cells
	const uint64_t N_Cells = static_cast<uint64_t>(uniqueCellIdF_.size());
	std::atomic<uint64_t> candCountFF{ 0 };
	std::atomic<uint64_t> posCountFF{ 0 };
	std::atomic<uint64_t> candCountFB{ 0 };
	std::atomic<uint64_t> posCountFB{ 0 };
	tbb::parallel_for(
		tbb::blocked_range<uint64_t>(0, N_Cells),
		[&](const tbb::blocked_range<uint64_t>& range)
		{
			static constexpr uint32_t LookupCacheSize = 256;
			static_assert((LookupCacheSize & (LookupCacheSize - 1)) == 0);

			std::array<uint64_t, LookupCacheSize> lookupCacheKeyF{};
			std::array<uint32_t, LookupCacheSize> lookupCacheValF{};
			std::array<uint64_t, LookupCacheSize> lookupCacheKeyB{};
			std::array<uint32_t, LookupCacheSize> lookupCacheValB{};
			lookupCacheKeyF.fill(EMPTY_64);
			lookupCacheKeyB.fill(EMPTY_64);
			lookupCacheValF.fill(EMPTY_32);
			lookupCacheValB.fill(EMPTY_32);

			auto& queryMetaScratch = queryMetaScratch_.local();
			auto& nbrCellId = queryMetaScratch.nbrCellId;
			auto& cellOrder = queryMetaScratch.cellOrder;
			auto& fluidRanges = queryMetaScratch.fluidRanges;
			auto& boundaryRanges = queryMetaScratch.boundaryRanges;
			nbrCellId.resize(N_NbrCells);
			cellOrder.resize(N_NbrCells);
			if (fluidRanges.capacity() < N_NbrCells) { fluidRanges.reserve(N_NbrCells); }
			if (boundaryRanges.capacity() < N_NbrCells) { boundaryRanges.reserve(N_NbrCells); }
			uint64_t candCountFF_local = 0;
			uint64_t posCountFF_local = 0;
			uint64_t candCountFB_local = 0;
			uint64_t posCountFB_local = 0;

			const auto findCellIdxTernary = [](const std::vector<uint64_t>& sortedCellIds,
				uint64_t key,
				uint32_t lo,
				uint32_t hi) noexcept
				{
					while (hi > lo + 24u)
					{
						const uint32_t span = hi - lo;
						const uint32_t third = span / 3u;
						const uint32_t m1 = lo + third;
						const uint32_t m2 = hi - third - 1u;

						const uint64_t v1 = sortedCellIds[m1];
						const uint64_t v2 = sortedCellIds[m2];

						if (key < v1)
						{
							hi = m1;
							continue;
						}
						if (key > v2)
						{
							lo = m2 + 1u;
							continue;
						}
						if (key == v1) { return m1; }
						if (key == v2) { return m2; }

						lo = m1 + 1u;
						hi = m2;
					}

					while (lo < hi)
					{
						const uint32_t mid = lo + ((hi - lo) >> 1u);
						if (sortedCellIds[mid] < key)
						{
							lo = mid + 1u;
						}
						else
						{
							hi = mid;
						}
					}

					if (lo < sortedCellIds.size() && sortedCellIds[lo] == key)
					{
						return lo;
					}

					return EMPTY_32;
				};

			// per-thread neighbor vectors for local accumulation and then bulk add
			// (reserve preparation in case of inline capacity not being enough)
			auto& scratchFF = scratchFF_.local();
			auto& scratchFB = scratchFB_.local();
			if (scratchFF.capacity() < params.expctdNbrCountHardMax_FF)
			{
				scratchFF.reserve(params.expctdNbrCountHardMax_FF);
			}
			if (scratchFB.capacity() < params.expctdNbrCountHardMax_FB)
			{
				scratchFB.reserve(params.expctdNbrCountHardMax_FB);
			}

			for (uint64_t c = range.begin(); c != range.end(); c++)
			{
				// range of all fluid particles in this cell
				const uint32_t sF = cellStartIdxF_[c]; // start
				const uint32_t eF = cellStartIdxF_[c + 1]; // end
				if (sF >= eF) { continue; }

				// determine a representative particle i0
				const uint32_t i0 = sF;

				const float xi0 = xPosF[i0];
				const float yi0 = yPosF[i0];
				const float zi0 = zPosF[i0];

				// cell coordinates of all particles in this cell
				const int cx = FloorToInt(xi0 * cellSizeInv_);
				const int cy = FloorToInt(yi0 * cellSizeInv_);
				const int cz = FloorToInt(zi0 * cellSizeInv_);

				// BigMin/LitMax range of the queried neighborhood in cell ID space
				uint64_t litMaxCellId = std::numeric_limits<uint64_t>::max();
				uint64_t bigMinCellId = 0;

				for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
				{
					const int oz = cz + rz; // target cell coordinate in z-direction

					for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
					{
						const int oy = cy + ry; // target cell coordinate in y-direction

						for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
						{
							const int ox = cx + rx; // target cell coordinate in x-direction

							// index of this neighbor cell in the queried neighborhood
							const int idx_nbr = NeighborOffset(rx, ry, rz);

							// cell ID of this neighbor cell
							const uint64_t cid_nbr = MapToCellId(ox, oy, oz);
							nbrCellId[idx_nbr] = cid_nbr;
							litMaxCellId = std::min(litMaxCellId, cid_nbr);
							bigMinCellId = std::max(bigMinCellId, cid_nbr);
						}
					}
				}

				// fluid lower/upper bounds over the full neighborhood range
				uint32_t loF = 0;
				uint32_t hiF = 0;
				if (!uniqueCellIdF_.empty())
				{
					loF = static_cast<uint32_t>(
						std::ranges::lower_bound(uniqueCellIdF_, litMaxCellId) - uniqueCellIdF_.begin());
					hiF = static_cast<uint32_t>(
						std::ranges::upper_bound(uniqueCellIdF_, bigMinCellId) - uniqueCellIdF_.begin());
					if (hiF < loF) { hiF = loF; }
				}

				// boundary lower/upper bounds over the full neighborhood range
				uint32_t loB = 0;
				uint32_t hiB = 0;
				if (!uniqueCellIdB_.empty())
				{
					loB = static_cast<uint32_t>(
						std::ranges::lower_bound(uniqueCellIdB_, litMaxCellId) - uniqueCellIdB_.begin());
					hiB = static_cast<uint32_t>(
						std::ranges::upper_bound(uniqueCellIdB_, bigMinCellId) - uniqueCellIdB_.begin());
					if (hiB < loB) { hiB = loB; }
				}

				// sort queried cells by cell ID so that queried cells can be traversed in Z-order
				std::iota(cellOrder.begin(), cellOrder.end(), 0u);
				std::sort(cellOrder.begin(), cellOrder.end(),
					[&](const uint32_t a, const uint32_t b)
					{
						return nbrCellId[a] < nbrCellId[b];
					});

				fluidRanges.clear();
				boundaryRanges.clear();

				// check existence and location of all queried grid cells
				for (uint32_t q = 0; q < N_NbrCells; q++)
				{
					const uint32_t idx_nbr = cellOrder[q];
					const uint64_t cid_nbr = nbrCellId[idx_nbr];

					// fluid scope
					{
						const uint32_t slot = static_cast<uint32_t>(cid_nbr) & (LookupCacheSize - 1u);
						uint32_t cF_nbr = EMPTY_32;

						if (lookupCacheKeyF[slot] == cid_nbr)
						{
							cF_nbr = lookupCacheValF[slot];
						}
						else
						{
							if (loF < hiF)
							{
								cF_nbr = findCellIdxTernary(uniqueCellIdF_, cid_nbr, loF, hiF);
							}
							lookupCacheKeyF[slot] = cid_nbr;
							lookupCacheValF[slot] = cF_nbr;
						}

						if (cF_nbr != EMPTY_32)
						{
							const uint32_t sF_nbr = cellStartIdxF_[cF_nbr];
							const uint32_t eF_nbr = cellStartIdxF_[cF_nbr + 1];

							if (sF_nbr < eF_nbr)
							{
								fluidRanges.push_back({ sF_nbr, eF_nbr });
							}
						}
					}

					// boundary scope
					{
						const uint32_t slot = static_cast<uint32_t>(cid_nbr) & (LookupCacheSize - 1u);
						uint32_t cB_nbr = EMPTY_32;

						if (lookupCacheKeyB[slot] == cid_nbr)
						{
							cB_nbr = lookupCacheValB[slot];
						}
						else
						{
							if (loB < hiB)
							{
								cB_nbr = findCellIdxTernary(uniqueCellIdB_, cid_nbr, loB, hiB);
							}
							lookupCacheKeyB[slot] = cid_nbr;
							lookupCacheValB[slot] = cB_nbr;
						}

						if (cB_nbr != EMPTY_32)
						{
							const uint32_t sB_nbr = cellStartIdxB_[cB_nbr];
							const uint32_t eB_nbr = cellStartIdxB_[cB_nbr + 1];

							if (sB_nbr < eB_nbr)
							{
								boundaryRanges.push_back({ sB_nbr, eB_nbr });
							}
						}
					}
				}

				// check all particles j in the queried neighboring grid cells
				for (uint32_t k = sF; k < eF; k++)
				{
					// reset scratch neighbor lists
					scratchFF.clear();
					scratchFB.clear();

					const uint32_t i = k;

					const float xi = xPosF[i];
					const float yi = yPosF[i];
					const float zi = zPosF[i];

					// fluid neighbors
					for (const ActiveCellRange& cellRange : fluidRanges)
					{
						#pragma omp simd
						for (uint32_t l = cellRange.begin; l < cellRange.end; l++)
						{
							const uint32_t j = l;
							candCountFF_local++;

							// position difference
							const float dx = xi - xPosF[j];
							const float dy = yi - yPosF[j];
							const float dz = zi - zPosF[j];

							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= kernelRadiusSqu) [[unlikely]]
							{
								// accumulate actual neighbors in scratch list
								posCountFF_local++;
								scratchFF.push_back(j);
							}
						}
					}

					// boundary neighbors
					for (const ActiveCellRange& cellRange : boundaryRanges)
					{
						#pragma omp simd
						for (uint32_t l = cellRange.begin; l < cellRange.end; l++)
						{
							const uint32_t j = l;
							candCountFB_local++;

							// position difference
							const float dx = xi - xPosB[j];
							const float dy = yi - yPosB[j];
							const float dz = zi - zPosB[j];

							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= kernelRadiusSqu) [[unlikely]]
							{
								// accumulate actual neighbors in scratch list
								posCountFB_local++;
								scratchFB.push_back(j);
							}
						}
					}

					// commit final neighbor lists from scratch to backend storage
					be_.neighborList_->SetNbrListFF(i, scratchFF.data(),
						static_cast<uint32_t>(scratchFF.size()));
					be_.neighborList_->SetNbrListFB(i, scratchFB.data(),
						static_cast<uint32_t>(scratchFB.size()));
				}
			}

			candCountFF.fetch_add(candCountFF_local, std::memory_order_relaxed);
			posCountFF.fetch_add(posCountFF_local, std::memory_order_relaxed);
			candCountFB.fetch_add(candCountFB_local, std::memory_order_relaxed);
			posCountFB.fetch_add(posCountFB_local, std::memory_order_relaxed);
		},
		tbb::auto_partitioner{}
	);
}

void NS_CellLinkedList_Compact_CPU01::DataReOrder(const SimParams& params)
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0 && N_B == 0)
	{
		cellIdsMatchCurrentParticleOrder_ = false;
		return;
	}

	cellIdsMatchCurrentParticleOrder_ = false;

	const float cellSize = params.kernelRadius * params.gridCellSizeRatio;
	const float querySpanCells = Base::ApproxEqu(cellSize, 0.0f)
		? 0.0f : (params.kernelRadius / cellSize);
	const uint32_t cellQueryReach = Base::ApproxEqu(cellSize, 0.0f)
		? 1u : std::max<uint32_t>(1u,
			static_cast<uint32_t>(std::ceil(std::max(0.0f, querySpanCells - 1e-6f))));

	const bool cellSizeChanged = !Base::ApproxEqu(cellSize_, cellSize);
	const bool cellQueryReachChanged = (cellQueryReach_ != cellQueryReach);

	kernelRadiusSqu_ = params.kernelRadiusSqu;
	cellSize_ = cellSize;
	cellSizeInv_ = Base::ApproxEqu(cellSize, 0.0f)
		? 0.0f : (1.0f / cellSize);
	cellQueryReach_ = cellQueryReach;

	if (cellSizeChanged || cellQueryReachChanged) { cellFrameValid_ = false; }
	UpdateCellFrame();

	// build the exact compact-CLL cell IDs and particle permutation for fluid particles
	cellIdF_.resize(N_F);
	particleOrderF_.resize(N_F);
	if (N_F > 0)
	{
		tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N_F),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const int cx = FloorToInt(be_.xPosF_[i] * cellSizeInv_);
					const int cy = FloorToInt(be_.yPosF_[i] * cellSizeInv_);
					const int cz = FloorToInt(be_.zPosF_[i] * cellSizeInv_);

					cellIdF_[i] = MapToCellId(cx, cy, cz);
					particleOrderF_[i] = i;
				}
			},
			tbb::auto_partitioner{}
		);

		tbb::parallel_sort(particleOrderF_.begin(), particleOrderF_.end(),
			[&](const uint32_t a, const uint32_t b)
			{
				if (cellIdF_[a] != cellIdF_[b]) { return cellIdF_[a] < cellIdF_[b]; }
				return a < b; // deterministic tie-break within the same cell
			});

		// keep the exact compact-CLL cell IDs in sync with the re-ordered particle arrays,
		// so Construct() can reuse them instead of re-computing them right away
		std::vector<uint64_t> cellIdScratchF;
		ApplyPermutation(particleOrderF_, cellIdF_, cellIdScratchF);
		ApplyPermutation(particleOrderF_, be_.xPosF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.yPosF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.zPosF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.xVelF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.yVelF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.zVelF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.xVelPredF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.yVelPredF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.zVelPredF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.xAccPressF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.yAccPressF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.zAccPressF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.sourceTermF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.diagElemF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.pressureF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.densityF_, be_.workBufferF_);
	}

	// build the exact compact-CLL cell IDs and particle permutation for boundary particles
	cellIdB_.resize(N_B);
	particleOrderB_.resize(N_B);
	if (N_B > 0)
	{
		tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N_B),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const int cx = FloorToInt(be_.xPosB_[i] * cellSizeInv_);
					const int cy = FloorToInt(be_.yPosB_[i] * cellSizeInv_);
					const int cz = FloorToInt(be_.zPosB_[i] * cellSizeInv_);

					cellIdB_[i] = MapToCellId(cx, cy, cz);
					particleOrderB_[i] = i;
				}
			},
			tbb::auto_partitioner{}
		);

		tbb::parallel_sort(particleOrderB_.begin(), particleOrderB_.end(),
			[&](const uint32_t a, const uint32_t b)
			{
				if (cellIdB_[a] != cellIdB_[b]) { return cellIdB_[a] < cellIdB_[b]; }
				return a < b; // deterministic tie-break within the same cell
			});

		// keep the exact compact-CLL cell IDs in sync with the re-ordered particle arrays,
		// so Construct() can reuse them instead of re-computing them right away
		std::vector<uint64_t> cellIdScratchB;
		ApplyPermutation(particleOrderB_, cellIdB_, cellIdScratchB);
		ApplyPermutation(particleOrderB_, be_.xPosB_, be_.workBufferB_);
		ApplyPermutation(particleOrderB_, be_.yPosB_, be_.workBufferB_);
		ApplyPermutation(particleOrderB_, be_.zPosB_, be_.workBufferB_);
	}

	cellIdsMatchCurrentParticleOrder_ = true;
}

std::pair<Mem::MemUsage, Mem::MemUsage> NS_CellLinkedList_Compact_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

	Mem::Add(usageF, Mem::MemUsageOf_Vector(uniqueCellIdF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellStartIdxF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellIdF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(particleOrderF_));

	Mem::Add(usageB, Mem::MemUsageOf_Vector(uniqueCellIdB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellStartIdxB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellIdB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(particleOrderB_));

	return { usageF, usageB };
}

void NS_CellLinkedList_Compact_CPU01::UpdateCellFrame()
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0 && N_B == 0)
	{
		cellOrigX_ = 0;
		cellOrigY_ = 0;
		cellOrigZ_ = 0;
		N_DimX_ = 0;
		N_DimY_ = 0;
		N_DimZ_ = 0;
		cellFrameValid_ = false;

		return;
	}

	bool rebuildFrame = !cellFrameValid_;

	// check whether the current occupied cell box still fits into the
	// reusable frame with the current query pad on all sides
	if (!rebuildFrame)
	{
		const int margin = static_cast<int>(cellQueryReach_);
		const int xMaxFrame = cellOrigX_ + static_cast<int>(N_DimX_) - 1;
		const int yMaxFrame = cellOrigY_ + static_cast<int>(N_DimY_) - 1;
		const int zMaxFrame = cellOrigZ_ + static_cast<int>(N_DimZ_) - 1;

		const int xMinInner = cellOrigX_ + margin;
		const int yMinInner = cellOrigY_ + margin;
		const int zMinInner = cellOrigZ_ + margin;
		const int xMaxInner = xMaxFrame - margin;
		const int yMaxInner = yMaxFrame - margin;
		const int zMaxInner = zMaxFrame - margin;

		if (xMinInner > xMaxInner || yMinInner > yMaxInner || zMinInner > zMaxInner)
		{
			rebuildFrame = true;
		}
		else
		{
			const bool fluidExceeded = tbb::parallel_reduce(
				tbb::blocked_range<uint32_t>(0, N_F),
				false,
				[&](const tbb::blocked_range<uint32_t>& r, bool localExceeded) -> bool
				{
					if (localExceeded) { return true; }

					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						const int cx = FloorToInt(be_.xPosF_[i] * cellSizeInv_);
						const int cy = FloorToInt(be_.yPosF_[i] * cellSizeInv_);
						const int cz = FloorToInt(be_.zPosF_[i] * cellSizeInv_);

						if (cx < xMinInner || cx > xMaxInner ||
							cy < yMinInner || cy > yMaxInner ||
							cz < zMinInner || cz > zMaxInner)
						{
							localExceeded = true;
							break;
						}
					}

					return localExceeded;
				},
				[](bool a, bool b) -> bool { return a || b; },
				tbb::auto_partitioner{}
			);

			if (fluidExceeded)
			{
				rebuildFrame = true;
			}
			else
			{
				const bool boundaryExceeded = tbb::parallel_reduce(
					tbb::blocked_range<uint32_t>(0, N_B),
					false,
					[&](const tbb::blocked_range<uint32_t>& r, bool localExceeded) -> bool
					{
						if (localExceeded) { return true; }

						for (uint32_t i = r.begin(); i != r.end(); i++)
						{
							const int cx = FloorToInt(be_.xPosB_[i] * cellSizeInv_);
							const int cy = FloorToInt(be_.yPosB_[i] * cellSizeInv_);
							const int cz = FloorToInt(be_.zPosB_[i] * cellSizeInv_);

							if (cx < xMinInner || cx > xMaxInner ||
								cy < yMinInner || cy > yMaxInner ||
								cz < zMinInner || cz > zMaxInner)
							{
								localExceeded = true;
								break;
							}
						}

						return localExceeded;
					},
					[](bool a, bool b) -> bool { return a || b; },
					tbb::auto_partitioner{}
				);

				rebuildFrame = boundaryExceeded;
			}
		}
	}

	if (!rebuildFrame) { return; }

	// rebuild cell frame from current occupied cell AABB
	const SceneAABB sceneAABB = ComputeSceneAABB_Combined(
		be_.xPosF_.data(), be_.yPosF_.data(), be_.zPosF_.data(), N_F,
		be_.xPosB_.data(), be_.yPosB_.data(), be_.zPosB_.data(), N_B);
	const CellCoordFrame cellFrame = ComputeCellCoordFrame_FromAABB(sceneAABB, cellSizeInv_);

	const int margin = static_cast<int>(cellQueryReach_ + N_CellFrameRebuildMargin_);
	cellOrigX_ = cellFrame.cellOrigX - margin;
	cellOrigY_ = cellFrame.cellOrigY - margin;
	cellOrigZ_ = cellFrame.cellOrigZ - margin;

	N_DimX_ = cellFrame.N_DimX + static_cast<uint32_t>(2 * margin);
	N_DimY_ = cellFrame.N_DimY + static_cast<uint32_t>(2 * margin);
	N_DimZ_ = cellFrame.N_DimZ + static_cast<uint32_t>(2 * margin);

	static constexpr uint32_t MaxAxisCells = (1u << 21);
	if (N_DimX_ > MaxAxisCells || N_DimY_ > MaxAxisCells || N_DimZ_ > MaxAxisCells)
	{
		SPDLOG_ERROR("Compact CLL cell frame exceeds 21-bit axis limit [{}, {}, {}]",
			N_DimX_, N_DimY_, N_DimZ_);
		assert(false);
		cellFrameValid_ = false;

		return;
	}

	cellFrameValid_ = true;
}

inline uint32_t NS_CellLinkedList_Compact_CPU01::NeighborCellCount() const noexcept
{
	const uint32_t N_NbrDim = 2u * cellQueryReach_ + 1u;
	return N_NbrDim * N_NbrDim * N_NbrDim;
}

inline uint64_t NS_CellLinkedList_Compact_CPU01::MapToCellId(
	int cx, int cy, int cz) const noexcept
{
	return MapToCellId_Morton(cx, cy, cz);
}

inline uint64_t NS_CellLinkedList_Compact_CPU01::MapToCellId_Morton(
	int cx, int cy, int cz) const noexcept
{
	// shift to local cell frame to avoid zig-zag preprocessing
	const int lx = cx - cellOrigX_;
	const int ly = cy - cellOrigY_;
	const int lz = cz - cellOrigZ_;
	assert(lx >= 0 && lx < static_cast<int>(N_DimX_));
	assert(ly >= 0 && ly < static_cast<int>(N_DimY_));
	assert(lz >= 0 && lz < static_cast<int>(N_DimZ_));

	const uint32_t ux = static_cast<uint32_t>(lx);
	const uint32_t uy = static_cast<uint32_t>(ly);
	const uint32_t uz = static_cast<uint32_t>(lz);
	assert(ux < (1u << 21));
	assert(uy < (1u << 21));
	assert(uz < (1u << 21));

	const uint64_t xx = SpreadBitsWithGaps_21to63(ux); // bits at pos 0,3,6,...
	const uint64_t yy = SpreadBitsWithGaps_21to63(uy) << 1; // bits at pos 1,4,7,...
	const uint64_t zz = SpreadBitsWithGaps_21to63(uz) << 2; // bits at pos 2,5,8,...

	return xx | yy | zz; // interleaved 3D Morton/Z-order code
}

// spread out lower 21 bits of v into lower 63 bits and fill with gaps zeros like 000x00x00x00x00x...
inline uint64_t NS_CellLinkedList_Compact_CPU01::SpreadBitsWithGaps_21to63(uint32_t v) noexcept
{
	uint64_t x = static_cast<uint64_t>(v & 0x1fffffU); // lower 21 bits

	x = (x | (x << 32)) & 0x1f00000000ffffULL;
	x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
	x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
	x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
	x = (x | (x << 2)) & 0x1249249249249249ULL;

	return x;
}

inline int NS_CellLinkedList_Compact_CPU01::NeighborOffset(int rx, int ry, int rz) const noexcept
{
	const int cellQueryReach = static_cast<int>(cellQueryReach_);
	const int N_NbrDim = 2 * cellQueryReach + 1;
	return (rz + cellQueryReach) * N_NbrDim * N_NbrDim
		 + (ry + cellQueryReach) * N_NbrDim
		 + (rx + cellQueryReach);
}

// -------------------------------------------------------------------------------------------------
// --- [ bounding volume hierarchy linear (LBVH) implementation ] ----------------------------------
// -------------------------------------------------------------------------------------------------

// TODO: implement
