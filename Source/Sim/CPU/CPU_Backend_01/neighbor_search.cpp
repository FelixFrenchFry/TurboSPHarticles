// neighbor_search.cpp

// external
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
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
#include "Source/Sim/CPU/simd.h"

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

	inline uint32_t ValidLaneMask_SIMD(
		const uint32_t begin,
		const uint32_t end) noexcept
	{
		uint32_t mask = 0u;
		for (uint32_t i = begin; i < end; i++)
		{
			mask |= (1u << (i - begin));
		}

		return mask;
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
		SceneAABB sceneAABB{};

		for (uint32_t i = 0; i < N_F; i++)
		{
			const float x = xPosF[i];
			const float y = yPosF[i];
			const float z = zPosF[i];

			sceneAABB.minX = std::min(sceneAABB.minX, x);
			sceneAABB.minY = std::min(sceneAABB.minY, y);
			sceneAABB.minZ = std::min(sceneAABB.minZ, z);
			sceneAABB.maxX = std::max(sceneAABB.maxX, x);
			sceneAABB.maxY = std::max(sceneAABB.maxY, y);
			sceneAABB.maxZ = std::max(sceneAABB.maxZ, z);
		}

		for (uint32_t i = 0; i < N_B; i++)
		{
			const float x = xPosB[i];
			const float y = yPosB[i];
			const float z = zPosB[i];

			sceneAABB.minX = std::min(sceneAABB.minX, x);
			sceneAABB.minY = std::min(sceneAABB.minY, y);
			sceneAABB.minZ = std::min(sceneAABB.minZ, z);
			sceneAABB.maxX = std::max(sceneAABB.maxX, x);
			sceneAABB.maxY = std::max(sceneAABB.maxY, y);
			sceneAABB.maxZ = std::max(sceneAABB.maxZ, z);
		}

		return sceneAABB;
	}

	inline CellCoordFrame ComputeCellCoordFrame_FromAABB(
		const SceneAABB& sceneAABB,
		float cellSizeInv)
	{
		CellCoordFrame frame{};

		const int cxMin = FloorToInt(sceneAABB.minX * cellSizeInv);
		const int cyMin = FloorToInt(sceneAABB.minY * cellSizeInv);
		const int czMin = FloorToInt(sceneAABB.minZ * cellSizeInv);
		const int cxMax = FloorToInt(sceneAABB.maxX * cellSizeInv);
		const int cyMax = FloorToInt(sceneAABB.maxY * cellSizeInv);
		const int czMax = FloorToInt(sceneAABB.maxZ * cellSizeInv);

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

void NS_BruteForce_CPU01::Init(
	const SimParams& params)
{
	kernelRadiusSqu_ = params.kernelRadiusSqu;
}

void NS_BruteForce_CPU01::Construct(
	const SimParams& params)
{
	kernelRadiusSqu_ = params.kernelRadiusSqu;
}

void NS_BruteForce_CPU01::Query_Count(
	const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadiusSqu;

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0) { return; }

	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	const uint32_t numChunks = be_.neighborList_->ChunkNumTotal();
	const bool trackChunkNeighborIdxRange =
		be_.neighborList_->NeedsChunkNeighborIdxRange();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, numChunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
				const uint32_t end = std::min(begin + CPU::N_SIMD_WIDTH, N_F);
				uint32_t cntFF[CPU::N_SIMD_WIDTH]{};
				uint32_t cntFB[CPU::N_SIMD_WIDTH]{};
				uint32_t minRefFF = std::numeric_limits<uint32_t>::max();
				uint32_t minRefFB = std::numeric_limits<uint32_t>::max();
				uint32_t maxRefFF = 0;
				uint32_t maxRefFB = 0;

				for (uint32_t i = begin; i < end; i++)
				{
					const uint32_t lane = i - begin;
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
							cntFF[lane]++;

							if (trackChunkNeighborIdxRange)
							{
								minRefFF = std::min(minRefFF, j);
								maxRefFF = std::max(maxRefFF, j);
							}
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
							cntFB[lane]++;

							if (trackChunkNeighborIdxRange)
							{
								minRefFB = std::min(minRefFB, j);
								maxRefFB = std::max(maxRefFB, j);
							}
						}
					}
				}

				// commit final neighbor counts to backend storage
				for (uint32_t lane = 0; lane < end - begin; lane++)
				{
					const uint32_t i = begin + lane;
					be_.neighborList_->SetNbrCountFF(i, cntFF[lane]);
					be_.neighborList_->SetNbrCountFB(i, cntFB[lane]);
				}

				if (trackChunkNeighborIdxRange)
				{
					be_.neighborList_->SetChunkNeighborIdxRangeFF(
						chunk, minRefFF, maxRefFF);
					be_.neighborList_->SetChunkNeighborIdxRangeFB(
						chunk, minRefFB, maxRefFB);
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_BruteForce_CPU01::Query_Fill(
	const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadiusSqu;

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0) { return; }

	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	const uint32_t numChunks = be_.neighborList_->ChunkNumTotal();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, numChunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
				const uint32_t end = std::min(begin + CPU::N_SIMD_WIDTH, N_F);
				uint32_t writtenFF[CPU::N_SIMD_WIDTH]{};
				uint32_t writtenFB[CPU::N_SIMD_WIDTH]{};
				ChunkedSELLNeighborSpan_Write chunkFF =
					be_.neighborList_->FFChunkWrite(chunk);
				ChunkedSELLNeighborSpan_Write chunkFB =
					be_.neighborList_->FBChunkWrite(chunk);

				for (uint32_t i = begin; i < end; i++)
				{
					const uint32_t lane = i - begin;
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
							const uint32_t k = writtenFF[lane]++;

							#if defined(_DEBUG)
							if (k >= chunkFF.pitch) [[unlikely]]
							{
								throw std::runtime_error(
									"Brute-force SELL FF fill exceeded chunk pitch.");
							}
							#endif

							chunkFF.StoreNeighbor(lane, k, j);
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
							const uint32_t k = writtenFB[lane]++;

							#if defined(_DEBUG)
							if (k >= chunkFB.pitch) [[unlikely]]
							{
								throw std::runtime_error(
									"Brute-force SELL FB fill exceeded chunk pitch.");
							}
							#endif

							chunkFB.StoreNeighbor(lane, k, j);
						}
					}
				}

				be_.neighborList_->FinishFFChunkWrite(chunk, writtenFF);
				be_.neighborList_->FinishFBChunkWrite(chunk, writtenFB);
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
	(void)params;

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
	cellTableF_.clear();

	uniqueCellIdB_.clear();
	cellStartIdxB_.clear();
	cellIdB_.clear();
	particleOrderB_.clear();
	cellTableB_.clear();
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
		cellTableF_.clear();

		uniqueCellIdB_.clear();
		cellStartIdxB_.clear();
		cellIdB_.clear();
		particleOrderB_.clear();
		cellTableB_.clear();

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

	BuildDenseCellTables();
}

void NS_CellLinkedList_Compact_CPU01::BuildDenseCellTables()
{
	const uint64_t tableSize = static_cast<uint64_t>(N_DimX_) * N_DimY_ * N_DimZ_;
	if (tableSize > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) [[unlikely]]
	{
		throw std::runtime_error("CPU01 dense compact CLL cell table too large.");
	}

	cellTableF_.assign(static_cast<size_t>(tableSize), EMPTY_32);
	cellTableB_.assign(static_cast<size_t>(tableSize), EMPTY_32);

	const auto buildTable = [&](const std::vector<uint32_t>& cellStartIdx,
		const std::vector<float>& xPos,
		const std::vector<float>& yPos,
		const std::vector<float>& zPos,
		std::vector<uint32_t>& cellTable)
		{
			const uint32_t N_Cells =
				(cellStartIdx.empty()) ? 0u : static_cast<uint32_t>(cellStartIdx.size() - 1u);

			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0, N_Cells),
				[&](const tbb::blocked_range<uint32_t>& r)
				{
					for (uint32_t c = r.begin(); c != r.end(); c++)
					{
						const uint32_t begin = cellStartIdx[c];
						const uint32_t end = cellStartIdx[c + 1];
						if (begin >= end) { continue; }

						const int cx = FloorToInt(xPos[begin] * cellSizeInv_);
						const int cy = FloorToInt(yPos[begin] * cellSizeInv_);
						const int cz = FloorToInt(zPos[begin] * cellSizeInv_);
						const uint64_t idx = MapToDenseCellIdx(cx, cy, cz);
						cellTable[idx] = c;
					}
				},
				tbb::auto_partitioner{}
			);
		};

	buildTable(cellStartIdxF_, be_.xPosF_, be_.yPosF_, be_.zPosF_, cellTableF_);
	buildTable(cellStartIdxB_, be_.xPosB_, be_.yPosB_, be_.zPosB_, cellTableB_);
}

void NS_CellLinkedList_Compact_CPU01::GatherCellQueryRanges(
	const int cx,
	const int cy,
	const int cz,
	boost::container::small_vector<ActiveCellRange, N_QueryCellInline>& fluidRanges,
	boost::container::small_vector<ActiveCellRange, N_QueryCellInline>& boundaryRanges) const
{
	const int cellQueryReach = static_cast<int>(cellQueryReach_);

	const auto appendUnique = [](auto& ranges, const ActiveCellRange range)
		{
			for (const ActiveCellRange& existing : ranges)
			{
				if (existing.begin == range.begin && existing.end == range.end)
				{
					return;
				}
			}

			ranges.push_back(range);
		};

	// direct local-frame lookup for the compact-cell index of every queried cell
	for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
	{
		const int oz = cz + rz; // target cell coordinate in z-direction

		for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
		{
			const int oy = cy + ry; // target cell coordinate in y-direction

			for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
			{
				const int ox = cx + rx; // target cell coordinate in x-direction
				const uint64_t tableIdx = MapToDenseCellIdx(ox, oy, oz);

				const uint32_t cF_nbr = cellTableF_[tableIdx];
				if (cF_nbr != EMPTY_32)
				{
					const uint32_t sF_nbr = cellStartIdxF_[cF_nbr];
					const uint32_t eF_nbr = cellStartIdxF_[cF_nbr + 1];
					if (sF_nbr < eF_nbr)
					{
						appendUnique(fluidRanges, ActiveCellRange{ sF_nbr, eF_nbr });
					}
				}

				const uint32_t cB_nbr = cellTableB_[tableIdx];
				if (cB_nbr != EMPTY_32)
				{
					const uint32_t sB_nbr = cellStartIdxB_[cB_nbr];
					const uint32_t eB_nbr = cellStartIdxB_[cB_nbr + 1];
					if (sB_nbr < eB_nbr)
					{
						appendUnique(boundaryRanges, ActiveCellRange{ sB_nbr, eB_nbr });
					}
				}
			}
		}
	}
}

void NS_CellLinkedList_Compact_CPU01::GatherChunkQueryRanges(
	const uint32_t begin,
	const uint32_t end,
	boost::container::small_vector<ActiveCellRange, N_QueryCellInline>& fluidRanges,
	boost::container::small_vector<ActiveCellRange, N_QueryCellInline>& boundaryRanges) const
{
	fluidRanges.clear();
	boundaryRanges.clear();

	int cellX[CPU::N_SIMD_WIDTH]{};
	int cellY[CPU::N_SIMD_WIDTH]{};
	int cellZ[CPU::N_SIMD_WIDTH]{};

	for (uint32_t i = begin; i < end; i++)
	{
		const uint32_t lane = i - begin;
		cellX[lane] = FloorToInt(be_.xPosF_[i] * cellSizeInv_);
		cellY[lane] = FloorToInt(be_.yPosF_[i] * cellSizeInv_);
		cellZ[lane] = FloorToInt(be_.zPosF_[i] * cellSizeInv_);
	}

	for (uint32_t i = begin; i < end; i++)
	{
		const uint32_t lane = i - begin;
		const int cx = cellX[lane];
		const int cy = cellY[lane];
		const int cz = cellZ[lane];

		bool alreadyVisited = false;
		for (uint32_t j = begin; j < i; j++)
		{
			const uint32_t prevLane = j - begin;
			if (cx == cellX[prevLane] &&
				cy == cellY[prevLane] &&
				cz == cellZ[prevLane])
			{
				alreadyVisited = true;
				break;
			}
		}

		if (alreadyVisited) { continue; }

		GatherCellQueryRanges(cx, cy, cz, fluidRanges, boundaryRanges);
	}
}

void NS_CellLinkedList_Compact_CPU01::Query_Count(const SimParams& params)
{
	{
		const float kernelRadiusSqu = params.kernelRadiusSqu;
		const uint32_t N_F = be_.N_F_;
		if (N_F == 0) { return; }

		// local references for fewer pointer indirections
		const float* RESTRICT xPosF = be_.xPosF_.data();
		const float* RESTRICT yPosF = be_.yPosF_.data();
		const float* RESTRICT zPosF = be_.zPosF_.data();
		const float* RESTRICT xPosB = be_.xPosB_.data();
		const float* RESTRICT yPosB = be_.yPosB_.data();
		const float* RESTRICT zPosB = be_.zPosB_.data();

		const uint32_t numChunks = ((N_F - 1u) / CPU::N_SIMD_WIDTH) + 1u;
		const CPU::Float32xSIMD kernelRadiusSquV = CPU::SetFloat32xSIMD(kernelRadiusSqu);
		auto* chunkedSELLNeighborList = be_.neighborList_.get();
		const bool trackChunkNeighborIdxRange =
			chunkedSELLNeighborList->NeedsChunkNeighborIdxRange();

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, numChunks),
			[&](const tbb::blocked_range<uint32_t>& range)
			{
				auto& queryMetaScratch = queryMetaScratch_.local();
				auto& fluidRanges = queryMetaScratch.fluidRanges;
				auto& boundaryRanges = queryMetaScratch.boundaryRanges;

				for (uint32_t chunk = range.begin(); chunk != range.end(); chunk++)
				{
					const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
					const uint32_t end = std::min(begin + CPU::N_SIMD_WIDTH, N_F);
					const uint32_t validMask = ValidLaneMask_SIMD(begin, end);
					const CPU::UInt32xSIMD idxI = CPU::ParticleIdx32xSIMD(begin, end);
					const CPU::Float32xSIMD xi = CPU::GatherFloat32xSIMD(xPosF, idxI);
					const CPU::Float32xSIMD yi = CPU::GatherFloat32xSIMD(yPosF, idxI);
					const CPU::Float32xSIMD zi = CPU::GatherFloat32xSIMD(zPosF, idxI);
					uint32_t cntFF[CPU::N_SIMD_WIDTH]{};
					uint32_t cntFB[CPU::N_SIMD_WIDTH]{};
					uint32_t minRefFF = std::numeric_limits<uint32_t>::max();
					uint32_t minRefFB = std::numeric_limits<uint32_t>::max();
					uint32_t maxRefFF = 0;
					uint32_t maxRefFB = 0;

					GatherChunkQueryRanges(begin, end, fluidRanges, boundaryRanges);

					// fluid neighbors
					for (const ActiveCellRange& cellRange : fluidRanges)
					{
						for (uint32_t l = cellRange.begin; l < cellRange.end; l++)
						{
							const CPU::Float32xSIMD xj = CPU::SetFloat32xSIMD(xPosF[l]);
							const CPU::Float32xSIMD yj = CPU::SetFloat32xSIMD(yPosF[l]);
							const CPU::Float32xSIMD zj = CPU::SetFloat32xSIMD(zPosF[l]);
							const CPU::Float32xSIMD dx = xi - xj;
							const CPU::Float32xSIMD dy = yi - yj;
							const CPU::Float32xSIMD dz = zi - zj;
							const CPU::Float32xSIMD distSqu = xsimd::fma(dx, dx,
								xsimd::fma(dy, dy, dz * dz));
							const uint32_t mask =
								CPU::MaskBits32xSIMD(distSqu <= kernelRadiusSquV) & validMask;

							if (trackChunkNeighborIdxRange && mask != 0u)
							{
								minRefFF = std::min(minRefFF, l);
								maxRefFF = std::max(maxRefFF, l);
							}

							for (uint32_t lane = 0; lane < end - begin; lane++)
							{
								if ((mask & (1u << lane)) != 0u) { cntFF[lane]++; }
							}
						}
					}

					// boundary neighbors
					for (const ActiveCellRange& cellRange : boundaryRanges)
					{
						for (uint32_t l = cellRange.begin; l < cellRange.end; l++)
						{
							const CPU::Float32xSIMD xj = CPU::SetFloat32xSIMD(xPosB[l]);
							const CPU::Float32xSIMD yj = CPU::SetFloat32xSIMD(yPosB[l]);
							const CPU::Float32xSIMD zj = CPU::SetFloat32xSIMD(zPosB[l]);
							const CPU::Float32xSIMD dx = xi - xj;
							const CPU::Float32xSIMD dy = yi - yj;
							const CPU::Float32xSIMD dz = zi - zj;
							const CPU::Float32xSIMD distSqu = xsimd::fma(dx, dx,
								xsimd::fma(dy, dy, dz * dz));
							const uint32_t mask =
								CPU::MaskBits32xSIMD(distSqu <= kernelRadiusSquV) & validMask;

							if (trackChunkNeighborIdxRange && mask != 0u)
							{
								minRefFB = std::min(minRefFB, l);
								maxRefFB = std::max(maxRefFB, l);
							}

							for (uint32_t lane = 0; lane < end - begin; lane++)
							{
								if ((mask & (1u << lane)) != 0u) { cntFB[lane]++; }
							}
						}
					}

					// commit final neighbor counts to backend storage
					for (uint32_t lane = 0; lane < end - begin; lane++)
					{
						const uint32_t i = begin + lane;
						be_.neighborList_->SetNbrCountFF(i, cntFF[lane]);
						be_.neighborList_->SetNbrCountFB(i, cntFB[lane]);
					}

					if (trackChunkNeighborIdxRange)
					{
						chunkedSELLNeighborList->SetChunkNeighborIdxRangeFF(
							chunk, minRefFF, maxRefFF);
						chunkedSELLNeighborList->SetChunkNeighborIdxRangeFB(
							chunk, minRefFB, maxRefFB);
					}
				}
			},
			tbb::auto_partitioner{}
		);
	}

}

void NS_CellLinkedList_Compact_CPU01::Query_Fill(const SimParams& params)
{
	{
		const float kernelRadiusSqu = params.kernelRadiusSqu;
		const uint32_t N_F = be_.N_F_;
		if (N_F == 0) { return; }

		auto* chunkedSELLNeighborList = be_.neighborList_.get();

		// local references for fewer pointer indirections
		const float* RESTRICT xPosF = be_.xPosF_.data();
		const float* RESTRICT yPosF = be_.yPosF_.data();
		const float* RESTRICT zPosF = be_.zPosF_.data();
		const float* RESTRICT xPosB = be_.xPosB_.data();
		const float* RESTRICT yPosB = be_.yPosB_.data();
		const float* RESTRICT zPosB = be_.zPosB_.data();

		const uint32_t numChunks = ((N_F - 1u) / CPU::N_SIMD_WIDTH) + 1u;
		const CPU::Float32xSIMD kernelRadiusSquV = CPU::SetFloat32xSIMD(kernelRadiusSqu);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, numChunks),
			[&](const tbb::blocked_range<uint32_t>& range)
			{
				auto& queryMetaScratch = queryMetaScratch_.local();
				auto& fluidRanges = queryMetaScratch.fluidRanges;
				auto& boundaryRanges = queryMetaScratch.boundaryRanges;

				for (uint32_t chunk = range.begin(); chunk != range.end(); chunk++)
				{
					const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
					const uint32_t end = std::min(begin + CPU::N_SIMD_WIDTH, N_F);
					const uint32_t validMask = ValidLaneMask_SIMD(begin, end);
					const CPU::UInt32xSIMD idxI = CPU::ParticleIdx32xSIMD(begin, end);
					const CPU::Float32xSIMD xi = CPU::GatherFloat32xSIMD(xPosF, idxI);
					const CPU::Float32xSIMD yi = CPU::GatherFloat32xSIMD(yPosF, idxI);
					const CPU::Float32xSIMD zi = CPU::GatherFloat32xSIMD(zPosF, idxI);
					uint32_t writtenFF[CPU::N_SIMD_WIDTH]{};
					uint32_t writtenFB[CPU::N_SIMD_WIDTH]{};
					ChunkedSELLNeighborSpan_Write chunkFF =
						chunkedSELLNeighborList->FFChunkWrite(chunk);
					ChunkedSELLNeighborSpan_Write chunkFB =
						chunkedSELLNeighborList->FBChunkWrite(chunk);

					GatherChunkQueryRanges(begin, end, fluidRanges, boundaryRanges);

					// fluid neighbors
					for (const ActiveCellRange& cellRange : fluidRanges)
					{
						for (uint32_t l = cellRange.begin; l < cellRange.end; l++)
						{
							const CPU::Float32xSIMD xj = CPU::SetFloat32xSIMD(xPosF[l]);
							const CPU::Float32xSIMD yj = CPU::SetFloat32xSIMD(yPosF[l]);
							const CPU::Float32xSIMD zj = CPU::SetFloat32xSIMD(zPosF[l]);
							const CPU::Float32xSIMD dx = xi - xj;
							const CPU::Float32xSIMD dy = yi - yj;
							const CPU::Float32xSIMD dz = zi - zj;
							const CPU::Float32xSIMD distSqu = xsimd::fma(dx, dx,
								xsimd::fma(dy, dy, dz * dz));
							const uint32_t mask =
								CPU::MaskBits32xSIMD(distSqu <= kernelRadiusSquV) & validMask;

							for (uint32_t lane = 0; lane < end - begin; lane++)
							{
								if ((mask & (1u << lane)) == 0u) { continue; }
								if (writtenFF[lane] >= chunkFF.pitch) [[unlikely]]
								{
									throw std::runtime_error(
										"CPU01 ChunkedSELL FF direct fill exceeded chunk pitch.");
								}

								chunkFF.StoreNeighbor(lane, writtenFF[lane], l);
								writtenFF[lane]++;
							}
						}
					}

					// boundary neighbors
					for (const ActiveCellRange& cellRange : boundaryRanges)
					{
						for (uint32_t l = cellRange.begin; l < cellRange.end; l++)
						{
							const CPU::Float32xSIMD xj = CPU::SetFloat32xSIMD(xPosB[l]);
							const CPU::Float32xSIMD yj = CPU::SetFloat32xSIMD(yPosB[l]);
							const CPU::Float32xSIMD zj = CPU::SetFloat32xSIMD(zPosB[l]);
							const CPU::Float32xSIMD dx = xi - xj;
							const CPU::Float32xSIMD dy = yi - yj;
							const CPU::Float32xSIMD dz = zi - zj;
							const CPU::Float32xSIMD distSqu = xsimd::fma(dx, dx,
								xsimd::fma(dy, dy, dz * dz));
							const uint32_t mask =
								CPU::MaskBits32xSIMD(distSqu <= kernelRadiusSquV) & validMask;

							for (uint32_t lane = 0; lane < end - begin; lane++)
							{
								if ((mask & (1u << lane)) == 0u) { continue; }
								if (writtenFB[lane] >= chunkFB.pitch) [[unlikely]]
								{
									throw std::runtime_error(
										"CPU01 ChunkedSELL FB direct fill exceeded chunk pitch.");
								}

								chunkFB.StoreNeighbor(lane, writtenFB[lane], l);
								writtenFB[lane]++;
							}
						}
					}

					chunkedSELLNeighborList->FinishFFChunkWrite(chunk, writtenFF);
					chunkedSELLNeighborList->FinishFBChunkWrite(chunk, writtenFB);
				}
			},
			tbb::auto_partitioner{}
		);
	}

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
		ApplyPermutation(particleOrderF_, be_.idF_, be_.workBufferIdF_);
		ApplyPermutation(particleOrderF_, be_.xPosF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.yPosF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.zPosF_, be_.workBufferF_);

		ApplyPermutation(particleOrderF_, be_.xVelF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.yVelF_, be_.workBufferF_);
		ApplyPermutation(particleOrderF_, be_.zVelF_, be_.workBufferF_);

		// keep all per-fluid-particle fields aligned after backend-only neighbor searches
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
		ApplyPermutation(particleOrderB_, be_.idB_, be_.workBufferIdB_);
		ApplyPermutation(particleOrderB_, be_.bodyIdB_, be_.workBufferIdB_);
		ApplyPermutation(particleOrderB_, be_.rigidBoundaryParticleIdxB_, be_.workBufferIdB_);
		ApplyPermutation(particleOrderB_, be_.xPosB_, be_.workBufferB_);
		ApplyPermutation(particleOrderB_, be_.yPosB_, be_.workBufferB_);
		ApplyPermutation(particleOrderB_, be_.zPosB_, be_.workBufferB_);
		ApplyPermutation(particleOrderB_, be_.xVelB_, be_.workBufferB_);
		ApplyPermutation(particleOrderB_, be_.yVelB_, be_.workBufferB_);
		ApplyPermutation(particleOrderB_, be_.zVelB_, be_.workBufferB_);
		ApplyPermutation(particleOrderB_, be_.volB_, be_.workBufferB_);
		ApplyPermutation(particleOrderB_, be_.psiB_, be_.workBufferB_);
	}

	cellIdsMatchCurrentParticleOrder_ = true;
}

bool NS_CellLinkedList_Compact_CPU01::TryGetLastParticleReOrder(
	std::vector<uint32_t>& oldFluidLocalIdx_FromNewLocalIdx,
	std::vector<uint32_t>& oldBoundaryLocalIdx_FromNewLocalIdx) const
{
	if (!cellIdsMatchCurrentParticleOrder_) { return false; }

	oldFluidLocalIdx_FromNewLocalIdx = particleOrderF_;
	oldBoundaryLocalIdx_FromNewLocalIdx = particleOrderB_;
	return true;
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
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellTableF_));

	Mem::Add(usageB, Mem::MemUsageOf_Vector(uniqueCellIdB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellStartIdxB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellIdB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(particleOrderB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellTableB_));

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

inline uint64_t NS_CellLinkedList_Compact_CPU01::MapToDenseCellIdx(
	int cx, int cy, int cz) const noexcept
{
	const int lx = cx - cellOrigX_;
	const int ly = cy - cellOrigY_;
	const int lz = cz - cellOrigZ_;
	assert(lx >= 0 && lx < static_cast<int>(N_DimX_));
	assert(ly >= 0 && ly < static_cast<int>(N_DimY_));
	assert(lz >= 0 && lz < static_cast<int>(N_DimZ_));

	return (static_cast<uint64_t>(lz) * N_DimY_ + static_cast<uint32_t>(ly))
		* N_DimX_ + static_cast<uint32_t>(lx);
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

