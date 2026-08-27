// neighbor_search.cpp

// external
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include <span>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/partitioner.h>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// internal
#include "cpu_backend.h"
#include "neighbor_lists.h"
#include "neighbor_search.h"
#include "source/base/helpers.h"
#include "source/simulation/CPU/simd.h"

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

	// gathers the three components of a vector field in one pass over the permutation,
	// so the sorted keys are streamed once instead of once per component
	template<typename TIdx, typename TData>
	void ApplyPermutation_Vec3(
		const std::vector<TIdx>& sortedKeys,
		std::vector<TData>& xData,
		std::vector<TData>& yData,
		std::vector<TData>& zData,
		std::vector<TData>& xScratch,
		std::vector<TData>& yScratch,
		std::vector<TData>& zScratch)
	{
		// (capped at uint32_t to correspond to max particle count)
		const uint32_t N = static_cast<uint32_t>(sortedKeys.size());
		if (N == 0) { return; }

		assert(xData.size() == N && "Permutation and data vector size mismatch");
		assert(yData.size() == N && "Permutation and data vector size mismatch");
		assert(zData.size() == N && "Permutation and data vector size mismatch");
		xScratch.resize(N);
		yScratch.resize(N);
		zScratch.resize(N);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const uint32_t srcIdx = static_cast<uint32_t>(sortedKeys[i]);
					xScratch[i] = xData[srcIdx];
					yScratch[i] = yData[srcIdx];
					zScratch[i] = zData[srcIdx];
				}
			},
			tbb::auto_partitioner{}
		);

		xData.swap(xScratch);
		yData.swap(yScratch);
		zData.swap(zScratch);
	}

	// checks whether the per-particle cell IDs are already in ascending order
	bool IsNonDecreasing(const std::span<const uint64_t> cellIds)
	{
		const uint32_t N = static_cast<uint32_t>(cellIds.size());
		if (N < 2) { return true; }

		return tbb::parallel_reduce(
			tbb::blocked_range<uint32_t>(1, N), true,
			[&](const tbb::blocked_range<uint32_t>& r, bool sorted) -> bool
			{
				for (uint32_t i = r.begin(); i != r.end() && sorted; i++)
				{
					sorted = cellIds[i - 1] <= cellIds[i];
				}
				return sorted;
			},
			[](const bool a, const bool b) -> bool { return a && b; },
			tbb::auto_partitioner{}
		);
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

	// drops the duplicates that overlapping query stencils leave behind
	// (distinct cells own distinct particle ranges, begin index identifies a range)
	template<typename TCellRanges>
	void DropDuplicateCellRanges(TCellRanges& ranges)
	{
		std::sort(ranges.begin(), ranges.end(),
			[](const auto& a, const auto& b) { return a.begin < b.begin; });

		ranges.erase(
			std::unique(ranges.begin(), ranges.end(),
				[](const auto& a, const auto& b) { return a.begin == b.begin; }),
			ranges.end());
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

	// merges one AABB into another (default bounds are the identity, so an empty one is a no-op)
	inline void MergeSceneAABB(SceneAABB& dst, const SceneAABB& src)
	{
		dst.minX = std::min(dst.minX, src.minX);
		dst.minY = std::min(dst.minY, src.minY);
		dst.minZ = std::min(dst.minZ, src.minZ);
		dst.maxX = std::max(dst.maxX, src.maxX);
		dst.maxY = std::max(dst.maxY, src.maxY);
		dst.maxZ = std::max(dst.maxZ, src.maxZ);
	}

	// reduces the positions of one particle class into its occupied AABB
	inline SceneAABB ComputeSceneAABB_ParticleClass(
		const float* xPos, const float* yPos, const float* zPos, uint32_t N)
	{
		return tbb::parallel_reduce(
			tbb::blocked_range<uint32_t>(0, N),
			SceneAABB{},
			[&](const tbb::blocked_range<uint32_t>& r, SceneAABB classAABB) -> SceneAABB
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const float x = xPos[i];
					const float y = yPos[i];
					const float z = zPos[i];

					classAABB.minX = std::min(classAABB.minX, x);
					classAABB.minY = std::min(classAABB.minY, y);
					classAABB.minZ = std::min(classAABB.minZ, z);
					classAABB.maxX = std::max(classAABB.maxX, x);
					classAABB.maxY = std::max(classAABB.maxY, y);
					classAABB.maxZ = std::max(classAABB.maxZ, z);
				}

				return classAABB;
			},
			[](SceneAABB a, const SceneAABB& b) -> SceneAABB
			{
				MergeSceneAABB(a, b);
				return a;
			},
			tbb::auto_partitioner{}
		);
	}

	inline SceneAABB ComputeSceneAABB_Combined(
		const float* xPosF, const float* yPosF, const float* zPosF, uint32_t N_F,
		const float* xPosB, const float* yPosB, const float* zPosB, uint32_t N_B,
		const float* xPosR, const float* yPosR, const float* zPosR, uint32_t N_R)
	{
		SceneAABB sceneAABB = ComputeSceneAABB_ParticleClass(xPosF, yPosF, zPosF, N_F);
		MergeSceneAABB(sceneAABB, ComputeSceneAABB_ParticleClass(xPosB, yPosB, zPosB, N_B));
		MergeSceneAABB(sceneAABB, ComputeSceneAABB_ParticleClass(xPosR, yPosR, zPosR, N_R));

		return sceneAABB;
	}

	// clips the scene AABB to the simulation domain (an empty domain extent clips nothing)
	inline void ClipSceneAABB_ToSimDomain(
		SceneAABB& sceneAABB,
		const glm::vec3& simDomainMin,
		const glm::vec3& simDomainMax)
	{
		if (simDomainMax.x <= simDomainMin.x ||
			simDomainMax.y <= simDomainMin.y ||
			simDomainMax.z <= simDomainMin.z) { return; }

		// clamping every bound keeps the AABB well formed, even with all particles on one side
		sceneAABB.minX = std::clamp(sceneAABB.minX, simDomainMin.x, simDomainMax.x);
		sceneAABB.minY = std::clamp(sceneAABB.minY, simDomainMin.y, simDomainMax.y);
		sceneAABB.minZ = std::clamp(sceneAABB.minZ, simDomainMin.z, simDomainMax.z);
		sceneAABB.maxX = std::clamp(sceneAABB.maxX, simDomainMin.x, simDomainMax.x);
		sceneAABB.maxY = std::clamp(sceneAABB.maxY, simDomainMin.y, simDomainMax.y);
		sceneAABB.maxZ = std::clamp(sceneAABB.maxZ, simDomainMin.z, simDomainMax.z);
	}

	constexpr uint64_t N_CELL_TABLE_CHUNK = 4096;

	// finds the start of every cell run in the sorted per-particle cell ID array
	// using a "parallel count -> prefix sum -> parallel fill" pipeline
	// (runStartCounts_ByChunk is caller-owned scratch and is overwritten on every call)
	void BuildCompactCellTable(
		const std::span<const uint64_t> cellId,
		const uint32_t particleLocalIdxOffset,
		std::vector<uint32_t>& cellStartIdx,
		std::vector<uint32_t>& runStartCounts_ByChunk)
	{
		// (particle counts are capped at uint32_t, so cell counts and start indices are too)
		const uint64_t N = cellId.size();
		if (N == 0)
		{
			cellStartIdx.clear();
			return;
		}

		const uint64_t N_chunks = (N + N_CELL_TABLE_CHUNK - 1) / N_CELL_TABLE_CHUNK;
		runStartCounts_ByChunk.assign(N_chunks, 0);

		// count cell run starts per chunk without shared counters
		tbb::parallel_for(
			tbb::blocked_range<uint64_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint64_t>& r)
			{
				for (uint64_t chunkIdx = r.begin(); chunkIdx != r.end(); chunkIdx++)
				{
					const uint64_t beginIdx = chunkIdx * N_CELL_TABLE_CHUNK;
					const uint64_t endIdx = std::min(beginIdx + N_CELL_TABLE_CHUNK, N);

					// the very first particle always starts the first cell run
					uint32_t N_runStarts = (beginIdx == 0) ? 1u : 0u;

					// reaching back into the previous chunk is a read-only access into
					// immutable input, so chunks stay independent
					const uint64_t firstIdx = (beginIdx == 0) ? 1u : beginIdx;
					for (uint64_t i = firstIdx; i != endIdx; i++)
					{
						if (cellId[i] != cellId[i - 1]) { N_runStarts++; }
					}

					runStartCounts_ByChunk[chunkIdx] = N_runStarts;
				}
			},
			tbb::auto_partitioner{}
		);

		// prefix chunk-local run start counts so each worker fills a disjoint output range
		uint32_t N_cells = 0;
		for (uint64_t chunkIdx = 0; chunkIdx != N_chunks; chunkIdx++)
		{
			const uint32_t N_runStarts = runStartCounts_ByChunk[chunkIdx];
			runStartCounts_ByChunk[chunkIdx] = N_cells; // now the chunk's write offset
			N_cells += N_runStarts;
		}

		// (not cleared beforehand, so a stable cell count avoids re-initializing the storage)
		cellStartIdx.resize(static_cast<uint64_t>(N_cells) + 1ull); // +1 for sentinel

		// fill the compact cell array while preserving the sorted cell run order
		tbb::parallel_for(
			tbb::blocked_range<uint64_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint64_t>& r)
			{
				for (uint64_t chunkIdx = r.begin(); chunkIdx != r.end(); chunkIdx++)
				{
					const uint64_t beginIdx = chunkIdx * N_CELL_TABLE_CHUNK;
					const uint64_t endIdx = std::min(beginIdx + N_CELL_TABLE_CHUNK, N);
					uint32_t writeIdx = runStartCounts_ByChunk[chunkIdx];

					// must mirror the counting pass exactly,
					// or the chunk write offsets do not line up
					if (beginIdx == 0)
					{
						cellStartIdx[writeIdx] = particleLocalIdxOffset;
						writeIdx++;
					}

					// TODO: use uint32_t instead?
					const uint64_t firstIdx = (beginIdx == 0) ? 1u : beginIdx;
					for (uint64_t i = firstIdx; i != endIdx; i++)
					{
						// still inside the same cell
						if (cellId[i] == cellId[i - 1]) { continue; }

						cellStartIdx[writeIdx] = particleLocalIdxOffset
											   + static_cast<uint32_t>(i);
						writeIdx++;
					}
				}
			},
			tbb::auto_partitioner{}
		);

		// end of the last cell's range as sentinel
		cellStartIdx[N_cells] = particleLocalIdxOffset + static_cast<uint32_t>(N);
	}

	// (larger than the cell table chunk, so the digit-count matrix stays compact)
	constexpr uint64_t N_RADIX_SORT_CHUNK = 65536;
	constexpr uint32_t N_RADIX_SORT_BITS = 8;
	constexpr uint32_t N_RADIX_SORT_BINS = 1u << N_RADIX_SORT_BITS;
	using RadixDigitCounts = std::array<uint32_t, N_RADIX_SORT_BINS>;

	// number of significant bits in the Morton cell IDs of a cell frame,
	// plus one bit that keeps the empty cell ID above every valid one
	uint32_t CellIdKeyBits(
		const uint32_t N_dimX,
		const uint32_t N_dimY,
		const uint32_t N_dimZ)
	{
		const uint32_t N_dimMax = std::max(N_dimX, std::max(N_dimY, N_dimZ));
		const uint32_t N_axisBits = (N_dimMax < 2)
			? 1u : static_cast<uint32_t>(std::bit_width(N_dimMax - 1u));

		return 3u * N_axisBits + 1u; // three interleaved axes plus the empty key bit
	}

	// sorts the particle permutation by cell ID using a stable LSD radix sort
	// with a "parallel count -> prefix sum -> parallel scatter" pipeline per digit
	// (the cell IDs are carried along, so every pass reads them sequentially instead of
	// gathering them through the permutation, and both scratches are overwritten)
	void SortParticleOrderByCellId(
		const std::span<uint64_t> cellId,
		const std::span<uint32_t> particleOrder,
		std::vector<uint64_t>& cellIdScratch,
		std::vector<uint32_t>& particleOrderScratch,
		std::vector<uint32_t>& digitCounts_ByChunk,
		const uint32_t N_keyBits)
	{
		// (particle counts are capped at uint32_t, so digit counts and write offsets are too)
		const uint64_t N = cellId.size();
		if (N < 2) { return; }

		assert(particleOrder.size() == N && "Permutation and cell ID vector size mismatch");
		Base::ResizeGrowScratchVector(cellIdScratch, N);
		Base::ResizeGrowScratchVector(particleOrderScratch, N);

		const uint64_t N_chunks = (N + N_RADIX_SORT_CHUNK - 1) / N_RADIX_SORT_CHUNK;

		// empty cell IDs sort behind every valid one without sorting all 64 bits
		const uint64_t emptyKey = 1ull << (N_keyBits - 1u);

		std::span<uint64_t> srcCellId = cellId;
		std::span<uint32_t> srcOrder = particleOrder;
		std::span<uint64_t> dstCellId = std::span<uint64_t>(cellIdScratch).first(N);
		std::span<uint32_t> dstOrder = std::span<uint32_t>(particleOrderScratch).first(N);

		for (uint32_t shift = 0; shift < N_keyBits; shift += N_RADIX_SORT_BITS)
		{
			const uint64_t N_digitCounts = N_chunks * N_RADIX_SORT_BINS;
			Base::ResizeGrowScratchVector(digitCounts_ByChunk, N_digitCounts);

			// count digits per chunk without shared counters
			tbb::parallel_for(
				tbb::blocked_range<uint64_t>(0, N_chunks),
				[&](const tbb::blocked_range<uint64_t>& r)
				{
					for (uint64_t chunkIdx = r.begin(); chunkIdx != r.end(); chunkIdx++)
					{
						const uint64_t beginIdx = chunkIdx * N_RADIX_SORT_CHUNK;
						const uint64_t endIdx = std::min(beginIdx + N_RADIX_SORT_CHUNK, N);
						uint32_t* counts = &digitCounts_ByChunk[chunkIdx * N_RADIX_SORT_BINS];
						std::fill(counts, counts + N_RADIX_SORT_BINS, uint32_t{ 0 });

						for (uint64_t i = beginIdx; i != endIdx; i++)
						{
							const uint64_t key = std::min(srcCellId[i], emptyKey);
							counts[(key >> shift) & (N_RADIX_SORT_BINS - 1)]++;
						}
					}
				},
				tbb::auto_partitioner{}
			);

			// prefix each digit across chunks, so particles with an equal
			// cell ID keep their relative order and the sort stays stable
			const RadixDigitCounts digitTotals = tbb::parallel_scan(
				tbb::blocked_range<uint64_t>(0, N_chunks),
				RadixDigitCounts{},
				[&](const tbb::blocked_range<uint64_t>& r,
					RadixDigitCounts prefix,
					const bool isFinalScan)
				{
					for (uint64_t chunkIdx = r.begin(); chunkIdx != r.end(); chunkIdx++)
					{
						uint32_t* counts = &digitCounts_ByChunk[chunkIdx * N_RADIX_SORT_BINS];
						for (uint32_t digit = 0; digit != N_RADIX_SORT_BINS; digit++)
						{
							const uint32_t N_digit = counts[digit];
							if (isFinalScan) { counts[digit] = prefix[digit]; }
							prefix[digit] += N_digit;
						}
					}

					return prefix;
				},
				[](RadixDigitCounts lhs, const RadixDigitCounts& rhs)
				{
					for (uint32_t digit = 0; digit != N_RADIX_SORT_BINS; digit++)
					{
						lhs[digit] += rhs[digit];
					}
					return lhs;
				},
				tbb::auto_partitioner{}
			);

			RadixDigitCounts digitWriteBase{};
			uint32_t writeIdx = 0;
			for (uint32_t digit = 0; digit != N_RADIX_SORT_BINS; digit++)
			{
				digitWriteBase[digit] = writeIdx;
				writeIdx += digitTotals[digit];
			}

			// scatter cell IDs and particle indices into their digit buckets
			tbb::parallel_for(
				tbb::blocked_range<uint64_t>(0, N_chunks),
				[&](const tbb::blocked_range<uint64_t>& r)
				{
					for (uint64_t chunkIdx = r.begin(); chunkIdx != r.end(); chunkIdx++)
					{
						const uint64_t beginIdx = chunkIdx * N_RADIX_SORT_CHUNK;
						const uint64_t endIdx = std::min(beginIdx + N_RADIX_SORT_CHUNK, N);
						uint32_t* offsets = &digitCounts_ByChunk[chunkIdx * N_RADIX_SORT_BINS];

						// must mirror the counting pass exactly,
						// or the chunk write offsets do not line up
						for (uint64_t i = beginIdx; i != endIdx; i++)
						{
							const uint64_t cellIdValue = srcCellId[i];
							const uint64_t key = std::min(cellIdValue, emptyKey);
							const uint32_t digit = static_cast<uint32_t>(
								(key >> shift) & (N_RADIX_SORT_BINS - 1));
							uint32_t& offsetInDigit = offsets[digit];
							const uint32_t outputIdx = digitWriteBase[digit] + offsetInDigit;

							dstCellId[outputIdx] = cellIdValue;
							dstOrder[outputIdx] = srcOrder[i];
							offsetInDigit++;
						}
					}
				},
				tbb::auto_partitioner{}
			);

			std::swap(srcCellId, dstCellId);
			std::swap(srcOrder, dstOrder);
		}

		// an odd pass count leaves the sorted result in the scratch buffers
		if (srcCellId.data() != cellId.data())
		{
			tbb::parallel_for(
				tbb::blocked_range<uint64_t>(0, N),
				[&](const tbb::blocked_range<uint64_t>& r)
				{
					for (uint64_t i = r.begin(); i != r.end(); i++)
					{
						cellId[i] = srcCellId[i];
						particleOrder[i] = srcOrder[i];
					}
				},
				tbb::auto_partitioner{}
			);
		}
	}

} // namespace anonymous

// -------------------------------------------------------------------------------------------------
// --- [ brute force implementation ] --------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_BruteForce_CPU::Init(
	const SimParams& params)
{
	kernelRadiusSqu_ = params.kernelRadiusSqu;
}

void NS_BruteForce_CPU::Construct(
	const SimParams& params)
{
	kernelRadiusSqu_ = params.kernelRadiusSqu;
}

void NS_BruteForce_CPU::Query_Count(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadiusSqu;
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const uint32_t N_R = be_.N_R_;

	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();
	const float* RESTRICT xPosR = be_.xPosR_.data();
	const float* RESTRICT yPosR = be_.yPosR_.data();
	const float* RESTRICT zPosR = be_.zPosR_.data();
	const uint32_t* RESTRICT bodyIdR = be_.bodyIdR_.data();

	// count FF, FB, and FR neighbors of each fluid particle
	const bool useSelectedFluidCenters = be_.useDistributedStepWork_;
	const uint32_t N_chunks = useSelectedFluidCenters
		? static_cast<uint32_t>(be_.pressureAccChunkIdxs_.size())
		: be_.neighborList_->ChunkNumTotal();
	const bool trackChunkNeighborIdxRange = be_.neighborList_->NeedsChunkNeighborIdxRange();
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_chunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t workIdx = r.begin(); workIdx != r.end(); workIdx++)
			{
				const uint32_t chunk = useSelectedFluidCenters
					? be_.pressureAccChunkIdxs_[workIdx] : workIdx;
				const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
				const uint32_t end = std::min(begin + CPU::N_SIMD_WIDTH, N_F);
				uint32_t cntFF[CPU::N_SIMD_WIDTH]{};
				uint32_t cntFB[CPU::N_SIMD_WIDTH]{};
				uint32_t cntFR[CPU::N_SIMD_WIDTH]{};
				uint32_t minRefFF = std::numeric_limits<uint32_t>::max();
				uint32_t minRefFB = std::numeric_limits<uint32_t>::max();
				uint32_t minRefFR = std::numeric_limits<uint32_t>::max();
				uint32_t maxRefFF = 0;
				uint32_t maxRefFB = 0;
				uint32_t maxRefFR = 0;

				for (uint32_t i = begin; i < end; i++)
				{
					if (useSelectedFluidCenters && be_.pressureAccFluidParticleFlags_[i] == 0)
						{ continue; }

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

					// check all rigid particles j
					for (uint32_t j = 0; j < N_R; j++)
					{
						const float dx = xi - xPosR[j];
						const float dy = yi - yPosR[j];
						const float dz = zi - zPosR[j];
						const float distSqu = dx * dx + dy * dy + dz * dz;

						if (distSqu <= kernelRadiusSqu) [[unlikely]]
						{
							cntFR[lane]++;
							if (trackChunkNeighborIdxRange)
							{
								minRefFR = std::min(minRefFR, j);
								maxRefFR = std::max(maxRefFR, j);
							}
						}
					}
				}

				for (uint32_t lane = 0; lane < end - begin; lane++)
				{
					const uint32_t i = begin + lane;
					be_.neighborList_->SetNbrCountFF(i, cntFF[lane]);
					be_.neighborList_->SetNbrCountFB(i, cntFB[lane]);
					be_.neighborList_->SetNbrCountFR(i, cntFR[lane]);
				}
				if (trackChunkNeighborIdxRange)
				{
					be_.neighborList_->SetChunkNeighborIdxRangeFF(chunk, minRefFF, maxRefFF);
					be_.neighborList_->SetChunkNeighborIdxRangeFB(chunk, minRefFB, maxRefFB);
					be_.neighborList_->SetChunkNeighborIdxRangeFR(chunk, minRefFR, maxRefFR);
				}
			}
		},
		tbb::auto_partitioner{}
	);

	// count RF, RB, and RR neighbors of each rigid particle
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_R),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				uint32_t countRF = 0;
				uint32_t countRB = 0;
				uint32_t countRR = 0;

				// check all fluid particles j
				for (uint32_t j = 0; j < N_F; j++)
				{
					const float dx = xPosR[i] - xPosF[j];
					const float dy = yPosR[i] - yPosF[j];
					const float dz = zPosR[i] - zPosF[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						countRF++;
					}
				}

				// check all boundary particles j
				for (uint32_t j = 0; j < N_B; j++)
				{
					const float dx = xPosR[i] - xPosB[j];
					const float dy = yPosR[i] - yPosB[j];
					const float dz = zPosR[i] - zPosB[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						countRB++;
					}
				}

				// check all rigid particles j
				for (uint32_t j = 0; j < N_R; j++)
				{
					// skip particles of the same rigid body
					if (j == i || bodyIdR[j] == bodyIdR[i]) { continue; }

					const float dx = xPosR[i] - xPosR[j];
					const float dy = yPosR[i] - yPosR[j];
					const float dz = zPosR[i] - zPosR[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						countRR++;
					}
				}

				be_.neighborList_->SetNbrCountRF(i, countRF);
				be_.neighborList_->SetNbrCountRB(i, countRB);
				be_.neighborList_->SetNbrCountRR(i, countRR);
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_BruteForce_CPU::Query_Fill(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadiusSqu;
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const uint32_t N_R = be_.N_R_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();
	const float* RESTRICT xPosR = be_.xPosR_.data();
	const float* RESTRICT yPosR = be_.yPosR_.data();
	const float* RESTRICT zPosR = be_.zPosR_.data();
	const uint32_t* RESTRICT bodyIdR = be_.bodyIdR_.data();

	// fill FF, FB, and FR neighbors of each fluid particle
	const bool useSelectedFluidCenters = be_.useDistributedStepWork_;
	const uint32_t N_chunks = useSelectedFluidCenters
		? static_cast<uint32_t>(be_.pressureAccChunkIdxs_.size())
		: be_.neighborList_->ChunkNumTotal();
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_chunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t workIdx = r.begin(); workIdx != r.end(); workIdx++)
			{
				const uint32_t chunk = useSelectedFluidCenters
					? be_.pressureAccChunkIdxs_[workIdx] : workIdx;
				const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
				const uint32_t end = std::min(begin + CPU::N_SIMD_WIDTH, N_F);
				uint32_t writtenFF[CPU::N_SIMD_WIDTH]{};
				uint32_t writtenFB[CPU::N_SIMD_WIDTH]{};
				uint32_t writtenFR[CPU::N_SIMD_WIDTH]{};
				ChunkedSELLNeighborSpan_Write chunkFF = be_.neighborList_->FFChunkWrite(chunk);
				ChunkedSELLNeighborSpan_Write chunkFB = be_.neighborList_->FBChunkWrite(chunk);
				ChunkedSELLNeighborSpan_Write chunkFR = be_.neighborList_->FRChunkWrite(chunk);

				for (uint32_t i = begin; i < end; i++)
				{
					if (useSelectedFluidCenters && be_.pressureAccFluidParticleFlags_[i] == 0)
						{ continue; }

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

					// check all rigid particles j
					for (uint32_t j = 0; j < N_R; j++)
					{
						const float dx = xi - xPosR[j];
						const float dy = yi - yPosR[j];
						const float dz = zi - zPosR[j];
						const float distSqu = dx * dx + dy * dy + dz * dz;

						if (distSqu <= kernelRadiusSqu) [[unlikely]]
						{
							const uint32_t k = writtenFR[lane]++;

							#if defined(_DEBUG)
							if (k >= chunkFR.pitch) [[unlikely]]
							{
								throw std::runtime_error(
									"Brute-force SELL FR fill exceeded chunk pitch.");
							}
							#endif

							chunkFR.StoreNeighbor(lane, k, j);
						}
					}
				}

				be_.neighborList_->FinishFFChunkWrite(chunk, writtenFF);
				be_.neighborList_->FinishFBChunkWrite(chunk, writtenFB);
				be_.neighborList_->FinishFRChunkWrite(chunk, writtenFR);
			}
		},
		tbb::auto_partitioner{}
	);

	// fill RF, RB, and RR neighbors of each rigid particle
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_R),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			std::vector<uint32_t> neighbors;
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// check all fluid particles j
				neighbors.clear();
				for (uint32_t j = 0; j < N_F; j++)
				{
					const float dx = xPosR[i] - xPosF[j];
					const float dy = yPosR[i] - yPosF[j];
					const float dz = zPosR[i] - zPosF[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						neighbors.push_back(j);
					}
				}
				be_.neighborList_->SetNbrListRF(
					i, neighbors.data(), static_cast<uint32_t>(neighbors.size()));

				// check all boundary particles j
				neighbors.clear();
				for (uint32_t j = 0; j < N_B; j++)
				{
					const float dx = xPosR[i] - xPosB[j];
					const float dy = yPosR[i] - yPosB[j];
					const float dz = zPosR[i] - zPosB[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						neighbors.push_back(j);
					}
				}
				be_.neighborList_->SetNbrListRB(
					i, neighbors.data(), static_cast<uint32_t>(neighbors.size()));

				// check all rigid particles j
				neighbors.clear();
				for (uint32_t j = 0; j < N_R; j++)
				{
					// skip particles of the same rigid body
					if (j == i || bodyIdR[j] == bodyIdR[i]) { continue; }

					const float dx = xPosR[i] - xPosR[j];
					const float dy = yPosR[i] - yPosR[j];
					const float dz = zPosR[i] - zPosR[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						neighbors.push_back(j);
					}
				}
				be_.neighborList_->SetNbrListRR(
					i, neighbors.data(), static_cast<uint32_t>(neighbors.size()));
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_BruteForce_CPU::DataReOrder(
	const SimParams& params)
{
	(void)params;

	// no re-ordering in brute force implementation
}

Mem::ParticleClassMemUsage NS_BruteForce_CPU::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	return {};
}

// -------------------------------------------------------------------------------------------------
// --- [ cell linked list compact implementation ] -------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_CellLinkedList_Compact_CPU::Init(const SimParams& params)
{
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;
	kernelRadiusSqu_ = params.kernelRadiusSqu;
	cellQueryReach_ = 1;
	cellFrame_ = {};
	cellFrameValid_ = false;
	cellIdsMatchCurrentParticleOrder_ = false;

	cellStartIdxF_.clear();
	cellIdF_.clear();
	cellIdScratchF_.clear();
	particleOrderF_.clear();
	particleOrderScratchF_.clear();
	cellTableF_.clear();
	cellStartIdxHaloF_.clear();
	cellTableHaloF_.clear();

	cellStartIdxB_.clear();
	cellIdB_.clear();
	cellIdScratchB_.clear();
	particleOrderB_.clear();
	particleOrderScratchB_.clear();
	cellTableB_.clear();
	cellStartIdxHaloB_.clear();
	cellTableHaloB_.clear();

	cellStartIdxR_.clear();
	cellIdR_.clear();
	cellIdScratchR_.clear();
	particleOrderR_.clear();
	particleOrderScratchR_.clear();
	cellTableR_.clear();

	cellRunStartCounts_ByChunk_.clear();
	radixDigitCounts_ByChunk_.clear();
}

void NS_CellLinkedList_Compact_CPU::Construct(const SimParams& params)
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const uint32_t N_R = be_.N_R_;

	if (N_F == 0 && N_B == 0 && N_R == 0)
	{
		SPDLOG_WARN("Cannot construct neighbor search for an empty simulation.");
		cellFrame_ = {};
		cellFrameValid_ = false;
		cellIdsMatchCurrentParticleOrder_ = false;

		cellStartIdxF_.clear();
		cellIdF_.clear();
		particleOrderF_.clear();
		particleOrderScratchF_.clear();
		cellTableF_.clear();
		cellStartIdxHaloF_.clear();
		cellTableHaloF_.clear();

		cellStartIdxB_.clear();
		cellIdB_.clear();
		particleOrderB_.clear();
		particleOrderScratchB_.clear();
		cellTableB_.clear();
		cellStartIdxHaloB_.clear();
		cellTableHaloB_.clear();

		cellStartIdxR_.clear();
		cellIdR_.clear();
		particleOrderR_.clear();
		particleOrderScratchR_.clear();
		cellTableR_.clear();

		cellRunStartCounts_ByChunk_.clear();
		radixDigitCounts_ByChunk_.clear();

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

	const bool cellFrameInvalidated = cellSizeChanged || cellQueryReachChanged;
	if (cellFrameInvalidated) { cellFrameValid_ = false; }

	// if DataReOrder() already computed and permuted the per-particle cell IDs for
	// this step, then they can be reused directly here and no re-computation is needed
	const bool canReuseCellIds =
		!cellFrameInvalidated
		&& cellIdsMatchCurrentParticleOrder_
		&& cellIdF_.size() == N_F
		&& cellIdB_.size() == N_B;

	// re-usable cell IDs imply a current cell frame, because DataReOrder() built them under it
	if (!canReuseCellIds) { UpdateCellFrame(); }

	// --- [ cell coordinates and indices for fluid and boundary particles ] --------------------------

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

						// particles outside the frame left the domain and stay out of the grid
						cellIdF_[i] = IsInsideCellFrame(cx, cy, cz)
							? MapToCellId(cx, cy, cz) : EMPTY_64;
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

						cellIdB_[i] = IsInsideCellFrame(cx, cy, cz)
							? MapToCellId(cx, cy, cz) : EMPTY_64;
					}
				},
				tbb::auto_partitioner{}
			);
		}

		// cell IDs now correspond to the current particle order again
		cellIdsMatchCurrentParticleOrder_ = true;
	}

	// rigid particles keep their body-contiguous order, so only the cell keys are sorted
	cellIdR_.resize(N_R);
	particleOrderR_.resize(N_R);
	if (N_R > 0)
	{
		tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N_R),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const int cx = FloorToInt(be_.xPosR_[i] * cellSizeInv_);
					const int cy = FloorToInt(be_.yPosR_[i] * cellSizeInv_);
					const int cz = FloorToInt(be_.zPosR_[i] * cellSizeInv_);
					cellIdR_[i] = IsInsideCellFrame(cx, cy, cz)
						? MapToCellId(cx, cy, cz) : EMPTY_64;
					particleOrderR_[i] = i;
				}
			},
			tbb::auto_partitioner{}
		);
		SortParticleOrderByCellId(
			cellIdR_, particleOrderR_, cellIdScratchR_, particleOrderScratchR_,
			radixDigitCounts_ByChunk_, CellIdKeyBits(
				cellFrame_.N_dimX, cellFrame_.N_dimY, cellFrame_.N_dimZ));
	}

	// --- [ build compact cell list and fill in cell start indices ] ---------------------------------

	const uint32_t N_ownedF = be_.useDistributedParticleLayout_ ? be_.N_ownedF_ : N_F;
	const uint32_t N_ownedB = be_.useDistributedParticleLayout_ ? be_.N_ownedB_ : N_B;
	const std::span<const uint64_t> cellIdsF = cellIdF_;
	const std::span<const uint64_t> cellIdsB = cellIdB_;

	// sort owned and halo particles separately to preserve [ owned ][ halo ] layout
	BuildCompactCellTable(
		cellIdsF.first(N_ownedF), 0, cellStartIdxF_, cellRunStartCounts_ByChunk_);
	BuildCompactCellTable(
		cellIdsF.subspan(N_ownedF), N_ownedF,
		cellStartIdxHaloF_, cellRunStartCounts_ByChunk_);
	BuildCompactCellTable(
		cellIdsB.first(N_ownedB), 0, cellStartIdxB_, cellRunStartCounts_ByChunk_);
	BuildCompactCellTable(
		cellIdsB.subspan(N_ownedB), N_ownedB,
		cellStartIdxHaloB_, cellRunStartCounts_ByChunk_);

	// rigid
	BuildCompactCellTable(cellIdR_, 0, cellStartIdxR_, cellRunStartCounts_ByChunk_);

	BuildDenseCellTables();
}

void NS_CellLinkedList_Compact_CPU::BuildDenseCellTables()
{
	const uint64_t N_tableCells = static_cast<uint64_t>(cellFrame_.N_dimX)
		* cellFrame_.N_dimY * cellFrame_.N_dimZ;
	if (N_tableCells > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) [[unlikely]]
	{
		throw std::runtime_error("CPU dense compact CLL cell table too large.");
	}

	// resets one dense cell table in parallel and leaves empty particle classes without a table
	const auto clearTable = [&](std::vector<uint32_t>& cellTable, const uint32_t N_particles)
		{
			const uint64_t N_entries = (N_particles == 0) ? 0ull : N_tableCells;

			// a changed entry count re-initializes the storage anyway
			if (cellTable.size() != N_entries)
			{
				cellTable.assign(static_cast<size_t>(N_entries), CELL_TABLE_EMPTY);
				return;
			}

			tbb::parallel_for(
				tbb::blocked_range<uint64_t>(0, N_entries),
				[&](const tbb::blocked_range<uint64_t>& r)
				{
					std::fill_n(cellTable.data() + r.begin(), r.size(), CELL_TABLE_EMPTY);
				},
				tbb::auto_partitioner{}
			);
		};

	const uint32_t N_ownedF = be_.useDistributedParticleLayout_ ? be_.N_ownedF_ : be_.N_F_;
	const uint32_t N_ownedB = be_.useDistributedParticleLayout_ ? be_.N_ownedB_ : be_.N_B_;
	clearTable(cellTableF_, N_ownedF);
	clearTable(cellTableHaloF_, be_.N_F_ - N_ownedF);
	clearTable(cellTableB_, N_ownedB);
	clearTable(cellTableHaloB_, be_.N_B_ - N_ownedB);
	clearTable(cellTableR_, be_.N_R_);

	const auto buildTable = [&](const std::vector<uint32_t>& cellStartIdx,
		const std::vector<float>& xPos,
		const std::vector<float>& yPos,
		const std::vector<float>& zPos,
		std::vector<uint32_t>& cellTable)
		{
			const uint32_t N_cells =
				(cellStartIdx.empty()) ? 0u : static_cast<uint32_t>(cellStartIdx.size() - 1u);

			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0, N_cells),
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

						// the sentinel cell of the particles that left the domain has no slot
						if (!IsInsideCellFrame(cx, cy, cz)) [[unlikely]] { continue; }

						const uint64_t idx = MapToDenseCellIdx(cx, cy, cz);
						cellTable[idx] = c + 1; // biased, so an empty slot stays zero
					}
				},
				tbb::auto_partitioner{}
			);
		};

	buildTable(cellStartIdxF_, be_.xPosF_, be_.yPosF_, be_.zPosF_, cellTableF_);
	buildTable(cellStartIdxHaloF_, be_.xPosF_, be_.yPosF_, be_.zPosF_, cellTableHaloF_);
	buildTable(cellStartIdxB_, be_.xPosB_, be_.yPosB_, be_.zPosB_, cellTableB_);
	buildTable(cellStartIdxHaloB_, be_.xPosB_, be_.yPosB_, be_.zPosB_, cellTableHaloB_);

	// rigid ranges index the sorted permutation instead of reordering rigid particle data
	const uint32_t N_cellsR = cellStartIdxR_.empty()
		? 0u : static_cast<uint32_t>(cellStartIdxR_.size() - 1u);
	for (uint32_t c = 0; c < N_cellsR; c++)
	{
		const uint32_t begin = cellStartIdxR_[c];
		const uint32_t rigidParticleIdx = particleOrderR_[begin];
		const int cx = FloorToInt(be_.xPosR_[rigidParticleIdx] * cellSizeInv_);
		const int cy = FloorToInt(be_.yPosR_[rigidParticleIdx] * cellSizeInv_);
		const int cz = FloorToInt(be_.zPosR_[rigidParticleIdx] * cellSizeInv_);

		// the sentinel cell of the particles that left the domain has no slot
		if (!IsInsideCellFrame(cx, cy, cz)) [[unlikely]] { continue; }

		cellTableR_[MapToDenseCellIdx(cx, cy, cz)] = c + 1; // biased, so an empty slot stays zero
	}
}

void NS_CellLinkedList_Compact_CPU::GatherBoxQueryRanges(
	const int cxBegin, const int cxEnd,
	const int cyBegin, const int cyEnd,
	const int czBegin, const int czEnd,
	boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& fluidRanges,
	boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& boundaryRanges,
	boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& rigidRanges) const
{
	// keeps the dense table lookups in bounds for boxes reaching past the clipped frame
	const int cxLo = std::max(cxBegin, static_cast<int>(cellFrame_.cellOrigX));
	const int cyLo = std::max(cyBegin, static_cast<int>(cellFrame_.cellOrigY));
	const int czLo = std::max(czBegin, static_cast<int>(cellFrame_.cellOrigZ));
	const int cxHi = std::min(cxEnd,
		static_cast<int>(cellFrame_.cellOrigX) + static_cast<int>(cellFrame_.N_dimX) - 1);
	const int cyHi = std::min(cyEnd,
		static_cast<int>(cellFrame_.cellOrigY) + static_cast<int>(cellFrame_.N_dimY) - 1);
	const int czHi = std::min(czEnd,
		static_cast<int>(cellFrame_.cellOrigZ) + static_cast<int>(cellFrame_.N_dimZ) - 1);

	const auto appendCellRange = [](
		const std::vector<uint32_t>& cellTable,
		const std::vector<uint32_t>& cellStartIdxs,
		const uint64_t tableIdx,
		auto& ranges)
	{
		// an empty particle class has no table to look its cells up in
		const uint32_t compactCellIdx = cellTable.empty()
			? CELL_TABLE_EMPTY : cellTable[tableIdx];
		if (compactCellIdx == CELL_TABLE_EMPTY) { return; }

		const uint32_t begin = cellStartIdxs[compactCellIdx - 1];
		const uint32_t end = cellStartIdxs[compactCellIdx];
		if (begin < end) { ranges.push_back(ActiveCellRange{ begin, end }); }
	};

	// direct local-frame lookup for the compact-cell index of every queried cell
	// (every cell coordinate is visited once, so the appended ranges are distinct)
	for (int oz = czLo; oz <= czHi; oz++)
	{
		for (int oy = cyLo; oy <= cyHi; oy++)
		{
			for (int ox = cxLo; ox <= cxHi; ox++)
			{
				const uint64_t tableIdx = MapToDenseCellIdx(ox, oy, oz);

				appendCellRange(cellTableF_, cellStartIdxF_, tableIdx, fluidRanges);
				appendCellRange(cellTableHaloF_, cellStartIdxHaloF_, tableIdx, fluidRanges);
				appendCellRange(cellTableB_, cellStartIdxB_, tableIdx, boundaryRanges);
				appendCellRange(cellTableHaloB_, cellStartIdxHaloB_, tableIdx, boundaryRanges);
				appendCellRange(cellTableR_, cellStartIdxR_, tableIdx, rigidRanges);
			}
		}
	}
}

void NS_CellLinkedList_Compact_CPU::GatherCellQueryRanges(
	const int cx,
	const int cy,
	const int cz,
	boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& fluidRanges,
	boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& boundaryRanges,
	boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& rigidRanges) const
{
	const int cellQueryReach = static_cast<int>(cellQueryReach_);

	GatherBoxQueryRanges(
		cx - cellQueryReach, cx + cellQueryReach,
		cy - cellQueryReach, cy + cellQueryReach,
		cz - cellQueryReach, cz + cellQueryReach,
		fluidRanges, boundaryRanges, rigidRanges);
}

void NS_CellLinkedList_Compact_CPU::GatherChunkQueryRanges(
	const uint32_t begin,
	const uint32_t end,
	boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& fluidRanges,
	boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& boundaryRanges,
	boost::container::small_vector<ActiveCellRange, N_QUERY_CELL_INLINE>& rigidRanges) const
{
	fluidRanges.clear();
	boundaryRanges.clear();
	rigidRanges.clear();

	// collect the distinct cells the chunk's particles sit in
	int cellX[CPU::N_SIMD_WIDTH]{};
	int cellY[CPU::N_SIMD_WIDTH]{};
	int cellZ[CPU::N_SIMD_WIDTH]{};
	uint32_t N_cells = 0;

	// TODO: parallelize with TBB?
	for (uint32_t i = begin; i < end; i++)
	{
		const int cx = FloorToInt(be_.xPosF_[i] * cellSizeInv_);
		const int cy = FloorToInt(be_.yPosF_[i] * cellSizeInv_);
		const int cz = FloorToInt(be_.zPosF_[i] * cellSizeInv_);

		// particles that left the domain query no cells, the frame cannot address them
		if (!IsInsideCellFrame(cx, cy, cz)) [[unlikely]] { continue; }

		bool alreadyVisited = false;
		for (uint32_t cell = 0; cell < N_cells; cell++)
		{
			if (cx == cellX[cell] &&
				cy == cellY[cell] &&
				cz == cellZ[cell])
			{
				alreadyVisited = true;
				break;
			}
		}

		if (alreadyVisited) { continue; }

		cellX[N_cells] = cx;
		cellY[N_cells] = cy;
		cellZ[N_cells] = cz;
		N_cells++;
	}

	if (N_cells == 0) { return; }

	// cell-coordinate box enclosing all of them
	int cxMin = cellX[0]; int cxMax = cellX[0];
	int cyMin = cellY[0]; int cyMax = cellY[0];
	int czMin = cellZ[0]; int czMax = cellZ[0];

	for (uint32_t cell = 1; cell < N_cells; cell++)
	{
		cxMin = std::min(cxMin, cellX[cell]); cxMax = std::max(cxMax, cellX[cell]);
		cyMin = std::min(cyMin, cellY[cell]); cyMax = std::max(cyMax, cellY[cell]);
		czMin = std::min(czMin, cellZ[cell]); czMax = std::max(czMax, cellZ[cell]);
	}

	const int cellQueryReach = static_cast<int>(cellQueryReach_);
	const uint64_t N_stencilSpan = 2ull * cellQueryReach_ + 1ull;
	const uint64_t N_stencilCells = N_stencilSpan * N_stencilSpan * N_stencilSpan;
	const uint64_t N_boxCells =
		(static_cast<uint64_t>(cxMax - cxMin) + N_stencilSpan) *
		(static_cast<uint64_t>(cyMax - cyMin) + N_stencilSpan) *
		(static_cast<uint64_t>(czMax - czMin) + N_stencilSpan);

	// one box sweep visits every cell once and needs no duplicate filtering, but Z-ordered cell IDs
	// can spread a chunk out, so only use if it is smaller than the per-cell stencils it replaces
	if (N_boxCells <= static_cast<uint64_t>(N_cells) * N_stencilCells)
	{
		GatherBoxQueryRanges(
			cxMin - cellQueryReach, cxMax + cellQueryReach,
			cyMin - cellQueryReach, cyMax + cellQueryReach,
			czMin - cellQueryReach, czMax + cellQueryReach,
			fluidRanges, boundaryRanges, rigidRanges);
		return;
	}

	// spread-out cells keep their own stencils, whose overlaps have to be dropped again
	for (uint32_t cell = 0; cell < N_cells; cell++)
	{
		GatherCellQueryRanges(
			cellX[cell], cellY[cell], cellZ[cell],
			fluidRanges, boundaryRanges, rigidRanges);
	}

	DropDuplicateCellRanges(fluidRanges);
	DropDuplicateCellRanges(boundaryRanges);
	DropDuplicateCellRanges(rigidRanges);
}

void NS_CellLinkedList_Compact_CPU::Query_Count(const SimParams& params)
{
	{
		const float kernelRadiusSqu = params.kernelRadiusSqu;
		const uint32_t N_F = be_.N_F_;
		const uint32_t N_R = be_.N_R_;
		if (N_F == 0 && N_R == 0) { return; }

		// local references for fewer pointer indirections
		const float* RESTRICT xPosF = be_.xPosF_.data();
		const float* RESTRICT yPosF = be_.yPosF_.data();
		const float* RESTRICT zPosF = be_.zPosF_.data();
		const float* RESTRICT xPosB = be_.xPosB_.data();
		const float* RESTRICT yPosB = be_.yPosB_.data();
		const float* RESTRICT zPosB = be_.zPosB_.data();
		const float* RESTRICT xPosR = be_.xPosR_.data();
		const float* RESTRICT yPosR = be_.yPosR_.data();
		const float* RESTRICT zPosR = be_.zPosR_.data();
		const uint32_t* RESTRICT bodyIdR = be_.bodyIdR_.data();

		const uint32_t N_totalChunks = (N_F == 0u)
			? 0u : ((N_F - 1u) / CPU::N_SIMD_WIDTH) + 1u;
		const bool useSelectedFluidCenters = be_.useDistributedStepWork_;
		const uint32_t N_chunks = useSelectedFluidCenters
			? static_cast<uint32_t>(be_.pressureAccChunkIdxs_.size())
			: N_totalChunks;
		const CPU::Float32xSIMD kernelRadiusSquV = CPU::SetFloat32xSIMD(kernelRadiusSqu);
		auto* chunkedSELLNeighborList = be_.neighborList_.get();
		const bool trackChunkNeighborIdxRange =
			chunkedSELLNeighborList->NeedsChunkNeighborIdxRange();

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& range)
			{
				auto& queryMetaScratch = queryMetaScratch_.local();
				auto& fluidRanges = queryMetaScratch.fluidRanges;
				auto& boundaryRanges = queryMetaScratch.boundaryRanges;
				auto& rigidRanges = queryMetaScratch.rigidRanges;

				for (uint32_t workIdx = range.begin(); workIdx != range.end(); workIdx++)
				{
					const uint32_t chunk = useSelectedFluidCenters
						? be_.pressureAccChunkIdxs_[workIdx] : workIdx;
					const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
					const uint32_t end = std::min(begin + CPU::N_SIMD_WIDTH, N_F);
					const CPU::Mask32xSIMD validLanes = CPU::ParticleFlagsMask32xSIMD(
						useSelectedFluidCenters
							? be_.pressureAccFluidParticleFlags_.data() : nullptr,
						begin, end);
					const CPU::UInt32xSIMD idxI = CPU::ParticleIdx32xSIMD(begin, end);
					const CPU::Float32xSIMD xi = CPU::GatherFloat32xSIMD(xPosF, idxI);
					const CPU::Float32xSIMD yi = CPU::GatherFloat32xSIMD(yPosF, idxI);
					const CPU::Float32xSIMD zi = CPU::GatherFloat32xSIMD(zPosF, idxI);
					CPU::UInt32xSIMD cntFF = CPU::SetUInt32xSIMD(0u);
					CPU::UInt32xSIMD cntFB = CPU::SetUInt32xSIMD(0u);
					CPU::UInt32xSIMD cntFR = CPU::SetUInt32xSIMD(0u);
					uint32_t minRefFF = std::numeric_limits<uint32_t>::max();
					uint32_t minRefFB = std::numeric_limits<uint32_t>::max();
					uint32_t minRefFR = std::numeric_limits<uint32_t>::max();
					uint32_t maxRefFF = 0;
					uint32_t maxRefFB = 0;
					uint32_t maxRefFR = 0;

					GatherChunkQueryRanges(begin, end, fluidRanges, boundaryRanges, rigidRanges);

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
							const CPU::Mask32xSIMD hit =
								(distSqu <= kernelRadiusSquV) && validLanes;

							if (trackChunkNeighborIdxRange && CPU::MaskBits32xSIMD(hit) != 0u)
							{
								minRefFF = std::min(minRefFF, l);
								maxRefFF = std::max(maxRefFF, l);
							}

							cntFF = CPU::AddMaskedOneUInt32xSIMD(cntFF, hit);
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
							const CPU::Mask32xSIMD hit =
								(distSqu <= kernelRadiusSquV) && validLanes;

							if (trackChunkNeighborIdxRange && CPU::MaskBits32xSIMD(hit) != 0u)
							{
								minRefFB = std::min(minRefFB, l);
								maxRefFB = std::max(maxRefFB, l);
							}

							cntFB = CPU::AddMaskedOneUInt32xSIMD(cntFB, hit);
						}
					}

					// rigid neighbors
					for (const ActiveCellRange& cellRange : rigidRanges)
					{
						for (uint32_t sortedIdx = cellRange.begin; sortedIdx < cellRange.end; sortedIdx++)
						{
							const uint32_t l = particleOrderR_[sortedIdx];
							const CPU::Float32xSIMD xj = CPU::SetFloat32xSIMD(xPosR[l]);
							const CPU::Float32xSIMD yj = CPU::SetFloat32xSIMD(yPosR[l]);
							const CPU::Float32xSIMD zj = CPU::SetFloat32xSIMD(zPosR[l]);
							const CPU::Float32xSIMD dx = xi - xj;
							const CPU::Float32xSIMD dy = yi - yj;
							const CPU::Float32xSIMD dz = zi - zj;
							const CPU::Float32xSIMD distSqu = xsimd::fma(dx, dx,
								xsimd::fma(dy, dy, dz * dz));
							const CPU::Mask32xSIMD hit =
								(distSqu <= kernelRadiusSquV) && validLanes;
							if (trackChunkNeighborIdxRange && CPU::MaskBits32xSIMD(hit) != 0u)
							{
								minRefFR = std::min(minRefFR, l);
								maxRefFR = std::max(maxRefFR, l);
							}
							cntFR = CPU::AddMaskedOneUInt32xSIMD(cntFR, hit);
						}
					}

					// commit final neighbor counts to backend storage
					alignas(CPU::N_SIMD_ALIGNMENT) uint32_t cntLanesFF[CPU::N_SIMD_WIDTH]{};
					alignas(CPU::N_SIMD_ALIGNMENT) uint32_t cntLanesFB[CPU::N_SIMD_WIDTH]{};
					alignas(CPU::N_SIMD_ALIGNMENT) uint32_t cntLanesFR[CPU::N_SIMD_WIDTH]{};
					cntFF.store_aligned(cntLanesFF);
					cntFB.store_aligned(cntLanesFB);
					cntFR.store_aligned(cntLanesFR);

					for (uint32_t lane = 0; lane < end - begin; lane++)
					{
						const uint32_t i = begin + lane;
						be_.neighborList_->SetNbrCountFF(i, cntLanesFF[lane]);
						be_.neighborList_->SetNbrCountFB(i, cntLanesFB[lane]);
						be_.neighborList_->SetNbrCountFR(i, cntLanesFR[lane]);
					}

					if (trackChunkNeighborIdxRange)
					{
						chunkedSELLNeighborList->SetChunkNeighborIdxRangeFF(
							chunk, minRefFF, maxRefFF);
						chunkedSELLNeighborList->SetChunkNeighborIdxRangeFB(
							chunk, minRefFB, maxRefFB);
						chunkedSELLNeighborList->SetChunkNeighborIdxRangeFR(
							chunk, minRefFR, maxRefFR);
					}
				}
			},
			tbb::auto_partitioner{}
		);

		// rigid source queries preserve rigid particle order
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_R),
			[&](const tbb::blocked_range<uint32_t>& range)
			{
				auto& queryMetaScratch = queryMetaScratch_.local();
				for (uint32_t i = range.begin(); i != range.end(); i++)
				{
					auto& fluidRanges = queryMetaScratch.fluidRanges;
					auto& boundaryRanges = queryMetaScratch.boundaryRanges;
					auto& rigidRanges = queryMetaScratch.rigidRanges;
					fluidRanges.clear(); boundaryRanges.clear(); rigidRanges.clear();
					const int cx = FloorToInt(xPosR[i] * cellSizeInv_);
					const int cy = FloorToInt(yPosR[i] * cellSizeInv_);
					const int cz = FloorToInt(zPosR[i] * cellSizeInv_);
					// particles that left the domain query no cells, the frame cannot address them
					if (IsInsideCellFrame(cx, cy, cz)) [[likely]]
					{
						GatherCellQueryRanges(cx, cy, cz, fluidRanges, boundaryRanges, rigidRanges);
					}
					uint32_t countRF = 0;
					uint32_t countRB = 0;
					uint32_t countRR = 0;

					for (const ActiveCellRange& cellRange : fluidRanges)
					{
						for (uint32_t j = cellRange.begin; j < cellRange.end; j++)
						{
							const float dx = xPosR[i] - xPosF[j];
							const float dy = yPosR[i] - yPosF[j];
							const float dz = zPosR[i] - zPosF[j];
							if (dx * dx + dy * dy + dz * dz <= kernelRadiusSqu) { countRF++; }
						}
					}

					for (const ActiveCellRange& cellRange : boundaryRanges)
					{
						for (uint32_t j = cellRange.begin; j < cellRange.end; j++)
						{
							const float dx = xPosR[i] - xPosB[j];
							const float dy = yPosR[i] - yPosB[j];
							const float dz = zPosR[i] - zPosB[j];
							if (dx * dx + dy * dy + dz * dz <= kernelRadiusSqu) { countRB++; }
						}
					}

					for (const ActiveCellRange& cellRange : rigidRanges)
					{
						for (uint32_t sortedIdx = cellRange.begin; sortedIdx < cellRange.end; sortedIdx++)
						{
							const uint32_t j = particleOrderR_[sortedIdx];
							if (j == i || bodyIdR[j] == bodyIdR[i]) { continue; }
							const float dx = xPosR[i] - xPosR[j];
							const float dy = yPosR[i] - yPosR[j];
							const float dz = zPosR[i] - zPosR[j];
							if (dx * dx + dy * dy + dz * dz <= kernelRadiusSqu) { countRR++; }
						}
					}

					be_.neighborList_->SetNbrCountRF(i, countRF);
					be_.neighborList_->SetNbrCountRB(i, countRB);
					be_.neighborList_->SetNbrCountRR(i, countRR);
				}
			},
			tbb::auto_partitioner{}
		);
	}
}

void NS_CellLinkedList_Compact_CPU::Query_Fill(const SimParams& params)
{
	{
		const float kernelRadiusSqu = params.kernelRadiusSqu;
		const uint32_t N_F = be_.N_F_;
		const uint32_t N_R = be_.N_R_;
		if (N_F == 0 && N_R == 0) { return; }

		auto* chunkedSELLNeighborList = be_.neighborList_.get();

		// local references for fewer pointer indirections
		const float* RESTRICT xPosF = be_.xPosF_.data();
		const float* RESTRICT yPosF = be_.yPosF_.data();
		const float* RESTRICT zPosF = be_.zPosF_.data();
		const float* RESTRICT xPosB = be_.xPosB_.data();
		const float* RESTRICT yPosB = be_.yPosB_.data();
		const float* RESTRICT zPosB = be_.zPosB_.data();
		const float* RESTRICT xPosR = be_.xPosR_.data();
		const float* RESTRICT yPosR = be_.yPosR_.data();
		const float* RESTRICT zPosR = be_.zPosR_.data();
		const uint32_t* RESTRICT bodyIdR = be_.bodyIdR_.data();

		const uint32_t N_totalChunks = (N_F == 0u)
			? 0u : ((N_F - 1u) / CPU::N_SIMD_WIDTH) + 1u;
		const bool useSelectedFluidCenters = be_.useDistributedStepWork_;
		const uint32_t N_chunks = useSelectedFluidCenters
			? static_cast<uint32_t>(be_.pressureAccChunkIdxs_.size())
			: N_totalChunks;
		const CPU::Float32xSIMD kernelRadiusSquV = CPU::SetFloat32xSIMD(kernelRadiusSqu);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& range)
			{
				auto& queryMetaScratch = queryMetaScratch_.local();
				auto& fluidRanges = queryMetaScratch.fluidRanges;
				auto& boundaryRanges = queryMetaScratch.boundaryRanges;
				auto& rigidRanges = queryMetaScratch.rigidRanges;

				for (uint32_t workIdx = range.begin(); workIdx != range.end(); workIdx++)
				{
					const uint32_t chunk = useSelectedFluidCenters
						? be_.pressureAccChunkIdxs_[workIdx] : workIdx;
					const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
					const uint32_t end = std::min(begin + CPU::N_SIMD_WIDTH, N_F);
					uint32_t validMask = ValidLaneMask_SIMD(begin, end);
					if (useSelectedFluidCenters)
					{
						validMask = 0;
						for (uint32_t lane = 0; lane < end - begin; lane++)
						{
							if (be_.pressureAccFluidParticleFlags_[begin + lane] != 0)
							{
								validMask |= 1u << lane;
							}
						}
					}
					const CPU::UInt32xSIMD idxI = CPU::ParticleIdx32xSIMD(begin, end);
					const CPU::Float32xSIMD xi = CPU::GatherFloat32xSIMD(xPosF, idxI);
					const CPU::Float32xSIMD yi = CPU::GatherFloat32xSIMD(yPosF, idxI);
					const CPU::Float32xSIMD zi = CPU::GatherFloat32xSIMD(zPosF, idxI);
					uint32_t writtenFF[CPU::N_SIMD_WIDTH]{};
					uint32_t writtenFB[CPU::N_SIMD_WIDTH]{};
					uint32_t writtenFR[CPU::N_SIMD_WIDTH]{};
					ChunkedSELLNeighborSpan_Write chunkFF =
						chunkedSELLNeighborList->FFChunkWrite(chunk);
					ChunkedSELLNeighborSpan_Write chunkFB =
						chunkedSELLNeighborList->FBChunkWrite(chunk);
					ChunkedSELLNeighborSpan_Write chunkFR =
						chunkedSELLNeighborList->FRChunkWrite(chunk);

					GatherChunkQueryRanges(begin, end, fluidRanges, boundaryRanges, rigidRanges);

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

							// padding lanes are already cleared, so only the matching lanes are visited
							for (uint32_t hitLanes = mask; hitLanes != 0u; hitLanes &= hitLanes - 1u)
							{
								const uint32_t lane = static_cast<uint32_t>(std::countr_zero(hitLanes));
								if (writtenFF[lane] >= chunkFF.pitch) [[unlikely]]
								{
									throw std::runtime_error(
										"CPU ChunkedSELL FF direct fill exceeded chunk pitch.");
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

							// padding lanes are already cleared, so only the matching lanes are visited
							for (uint32_t hitLanes = mask; hitLanes != 0u; hitLanes &= hitLanes - 1u)
							{
								const uint32_t lane = static_cast<uint32_t>(std::countr_zero(hitLanes));
								if (writtenFB[lane] >= chunkFB.pitch) [[unlikely]]
								{
									throw std::runtime_error(
										"CPU ChunkedSELL FB direct fill exceeded chunk pitch.");
								}

								chunkFB.StoreNeighbor(lane, writtenFB[lane], l);
								writtenFB[lane]++;
							}
						}
					}

					// rigid neighbors
					for (const ActiveCellRange& cellRange : rigidRanges)
					{
						for (uint32_t sortedIdx = cellRange.begin; sortedIdx < cellRange.end; sortedIdx++)
						{
							const uint32_t l = particleOrderR_[sortedIdx];
							const CPU::Float32xSIMD xj = CPU::SetFloat32xSIMD(xPosR[l]);
							const CPU::Float32xSIMD yj = CPU::SetFloat32xSIMD(yPosR[l]);
							const CPU::Float32xSIMD zj = CPU::SetFloat32xSIMD(zPosR[l]);
							const CPU::Float32xSIMD dx = xi - xj;
							const CPU::Float32xSIMD dy = yi - yj;
							const CPU::Float32xSIMD dz = zi - zj;
							const CPU::Float32xSIMD distSqu = xsimd::fma(dx, dx,
								xsimd::fma(dy, dy, dz * dz));
							const uint32_t mask =
								CPU::MaskBits32xSIMD(distSqu <= kernelRadiusSquV) & validMask;
							// padding lanes are already cleared, so only the matching lanes are visited
							for (uint32_t hitLanes = mask; hitLanes != 0u; hitLanes &= hitLanes - 1u)
							{
								const uint32_t lane = static_cast<uint32_t>(std::countr_zero(hitLanes));
								chunkFR.StoreNeighbor(lane, writtenFR[lane], l);
								writtenFR[lane]++;
							}
						}
					}

					chunkedSELLNeighborList->FinishFFChunkWrite(chunk, writtenFF);
					chunkedSELLNeighborList->FinishFBChunkWrite(chunk, writtenFB);
					chunkedSELLNeighborList->FinishFRChunkWrite(chunk, writtenFR);
				}
			},
			tbb::auto_partitioner{}
		);

		// fill rigid source neighbors without changing rigid particle order
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_R),
			[&](const tbb::blocked_range<uint32_t>& range)
			{
				auto& queryMetaScratch = queryMetaScratch_.local();
				std::vector<uint32_t> neighbors;
				for (uint32_t i = range.begin(); i != range.end(); i++)
				{
					auto& fluidRanges = queryMetaScratch.fluidRanges;
					auto& boundaryRanges = queryMetaScratch.boundaryRanges;
					auto& rigidRanges = queryMetaScratch.rigidRanges;
					fluidRanges.clear(); boundaryRanges.clear(); rigidRanges.clear();
					const int cx = FloorToInt(xPosR[i] * cellSizeInv_);
					const int cy = FloorToInt(yPosR[i] * cellSizeInv_);
					const int cz = FloorToInt(zPosR[i] * cellSizeInv_);
					// particles that left the domain query no cells, the frame cannot address them
					if (IsInsideCellFrame(cx, cy, cz)) [[likely]]
					{
						GatherCellQueryRanges(cx, cy, cz, fluidRanges, boundaryRanges, rigidRanges);
					}

					neighbors.clear();
					for (const ActiveCellRange& cellRange : fluidRanges)
					{
						for (uint32_t j = cellRange.begin; j < cellRange.end; j++)
						{
							const float dx = xPosR[i] - xPosF[j];
							const float dy = yPosR[i] - yPosF[j];
							const float dz = zPosR[i] - zPosF[j];
							if (dx * dx + dy * dy + dz * dz <= kernelRadiusSqu) { neighbors.push_back(j); }
						}
					}
					be_.neighborList_->SetNbrListRF(
						i, neighbors.data(), static_cast<uint32_t>(neighbors.size()));

					neighbors.clear();
					for (const ActiveCellRange& cellRange : boundaryRanges)
					{
						for (uint32_t j = cellRange.begin; j < cellRange.end; j++)
						{
							const float dx = xPosR[i] - xPosB[j];
							const float dy = yPosR[i] - yPosB[j];
							const float dz = zPosR[i] - zPosB[j];
							if (dx * dx + dy * dy + dz * dz <= kernelRadiusSqu) { neighbors.push_back(j); }
						}
					}
					be_.neighborList_->SetNbrListRB(
						i, neighbors.data(), static_cast<uint32_t>(neighbors.size()));

					neighbors.clear();
					for (const ActiveCellRange& cellRange : rigidRanges)
					{
						for (uint32_t sortedIdx = cellRange.begin; sortedIdx < cellRange.end; sortedIdx++)
						{
							const uint32_t j = particleOrderR_[sortedIdx];
							if (j == i || bodyIdR[j] == bodyIdR[i]) { continue; }
							const float dx = xPosR[i] - xPosR[j];
							const float dy = yPosR[i] - yPosR[j];
							const float dz = zPosR[i] - zPosR[j];
							if (dx * dx + dy * dy + dz * dz <= kernelRadiusSqu) { neighbors.push_back(j); }
						}
					}
					be_.neighborList_->SetNbrListRR(
						i, neighbors.data(), static_cast<uint32_t>(neighbors.size()));
				}
			},
			tbb::auto_partitioner{}
		);
	}
}

void NS_CellLinkedList_Compact_CPU::DataReOrder(const SimParams& params)
{
	ReOrderParticleData(params, true);
}

void NS_CellLinkedList_Compact_CPU::DataReOrder_SolverStep(const SimParams& params)
{
	ReOrderParticleData(params, false);
}

void NS_CellLinkedList_Compact_CPU::ReOrderParticleData(
	const SimParams& params,
	const bool permuteSolverFields)
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const uint32_t N_ownedF = be_.useDistributedParticleLayout_ ? be_.N_ownedF_ : N_F;
	const uint32_t N_ownedB = be_.useDistributedParticleLayout_ ? be_.N_ownedB_ : N_B;
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
	cellSizeInv_ = Base::ApproxEqu(cellSize, 0.0f) ? 0.0f : (1.0f / cellSize);
	cellQueryReach_ = cellQueryReach;

	if (cellSizeChanged || cellQueryReachChanged) { cellFrameValid_ = false; }
	UpdateCellFrame();

	const uint32_t N_keyBits = CellIdKeyBits(
		cellFrame_.N_dimX, cellFrame_.N_dimY, cellFrame_.N_dimZ);

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

					cellIdF_[i] = IsInsideCellFrame(cx, cy, cz)
						? MapToCellId(cx, cy, cz) : EMPTY_64;
					particleOrderF_[i] = i;
				}
			},
			tbb::auto_partitioner{}
		);

		// sort owned and halo particles separately to preserve [ owned ][ halo ] layout
		std::span<uint64_t> cellIdsF = cellIdF_;
		std::span<uint32_t> particleOrderF = particleOrderF_;
		SortParticleOrderByCellId(
			cellIdsF.first(N_ownedF), particleOrderF.first(N_ownedF),
			cellIdScratchF_, particleOrderScratchF_,
			radixDigitCounts_ByChunk_, N_keyBits);
		SortParticleOrderByCellId(
			cellIdsF.subspan(N_ownedF), particleOrderF.subspan(N_ownedF),
			cellIdScratchF_, particleOrderScratchF_,
			radixDigitCounts_ByChunk_, N_keyBits);

		ApplyPermutation(particleOrderF_, be_.idF_, be_.workBufferIdF_);
		ApplyPermutation_Vec3(particleOrderF_,
			be_.xPosF_, be_.yPosF_, be_.zPosF_,
			be_.workBufferF_, be_.workBufferF2_, be_.workBufferF3_);

		ApplyPermutation_Vec3(particleOrderF_,
			be_.xVelF_, be_.yVelF_, be_.zVelF_,
			be_.workBufferF_, be_.workBufferF2_, be_.workBufferF3_);

		// the remaining fluid particle fields only need to be permuted after backend-only neighbor searches,
		// because a following full solver step re-initializes them (TODO: update if solver step behavior changes)
		if (permuteSolverFields)
		{
			ApplyPermutation_Vec3(particleOrderF_,
				be_.xVelPredF_, be_.yVelPredF_, be_.zVelPredF_,
				be_.workBufferF_, be_.workBufferF2_, be_.workBufferF3_);

			ApplyPermutation_Vec3(particleOrderF_,
				be_.xAccPressF_, be_.yAccPressF_, be_.zAccPressF_,
				be_.workBufferF_, be_.workBufferF2_, be_.workBufferF3_);

			ApplyPermutation(particleOrderF_, be_.sourceTermF_, be_.workBufferF_);
			ApplyPermutation(particleOrderF_, be_.diagElemF_, be_.workBufferF_);
			// without a warm start, pressure is re-initialized by the source-term kernel
			ApplyPermutation(particleOrderF_, be_.pressureF_, be_.workBufferF_);
			ApplyPermutation(particleOrderF_, be_.densityF_, be_.workBufferF_);
		}
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

					cellIdB_[i] = IsInsideCellFrame(cx, cy, cz)
						? MapToCellId(cx, cy, cz) : EMPTY_64;
					particleOrderB_[i] = i;
				}
			},
			tbb::auto_partitioner{}
		);

		std::span<uint64_t> cellIdsB = cellIdB_;
		std::span<uint32_t> particleOrderB = particleOrderB_;
		const bool reOrderOwned = !IsNonDecreasing(cellIdsB.first(N_ownedB));
		const bool reOrderHalo = !IsNonDecreasing(cellIdsB.subspan(N_ownedB));
		if (reOrderOwned)
		{
			SortParticleOrderByCellId(
				cellIdsB.first(N_ownedB), particleOrderB.first(N_ownedB),
				cellIdScratchB_, particleOrderScratchB_,
				radixDigitCounts_ByChunk_, N_keyBits);
		}
		if (reOrderHalo)
		{
			SortParticleOrderByCellId(
				cellIdsB.subspan(N_ownedB), particleOrderB.subspan(N_ownedB),
				cellIdScratchB_, particleOrderScratchB_,
				radixDigitCounts_ByChunk_, N_keyBits);
		}

		if (reOrderOwned || reOrderHalo)
		{
			// keep the exact compact-CLL cell IDs in sync with the re-ordered particle arrays,
			// so Construct() can reuse them instead of re-computing them right away
			ApplyPermutation(particleOrderB_, be_.idB_, be_.workBufferIdB_);
			ApplyPermutation_Vec3(particleOrderB_,
				be_.xPosB_, be_.yPosB_, be_.zPosB_,
				be_.workBufferB_, be_.workBufferB2_, be_.workBufferB3_);
			ApplyPermutation(particleOrderB_, be_.volB_, be_.workBufferB_);
			ApplyPermutation(particleOrderB_, be_.psiB_, be_.workBufferB_);
		}
	}

	cellIdsMatchCurrentParticleOrder_ = true;
}

bool NS_CellLinkedList_Compact_CPU::TryGetLastParticleReOrder(
	std::span<const uint32_t>& oldFluidLocalIdx_FromNewLocalIdx,
	std::span<const uint32_t>& oldBoundaryLocalIdx_FromNewLocalIdx) const
{
	if (!cellIdsMatchCurrentParticleOrder_) { return false; }

	// the permutations stay until the next re-order rebuilds them, so a view is enough
	oldFluidLocalIdx_FromNewLocalIdx = particleOrderF_;
	oldBoundaryLocalIdx_FromNewLocalIdx = particleOrderB_;
	return true;
}

Mem::ParticleClassMemUsage NS_CellLinkedList_Compact_CPU::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};
	Mem::MemUsage usageR{};

	// F
	// (the shared cell run start and digit count scratches are booked here,
	// because the fluid particles dominate)
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellStartIdxF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellIdF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellIdScratchF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(particleOrderF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(particleOrderScratchF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellTableF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellStartIdxHaloF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellTableHaloF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellRunStartCounts_ByChunk_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(radixDigitCounts_ByChunk_));

	// B
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellStartIdxB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellIdB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellIdScratchB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(particleOrderB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(particleOrderScratchB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellTableB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellStartIdxHaloB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellTableHaloB_));

	// R
	Mem::Add(usageR, Mem::MemUsageOf_Vector(cellStartIdxR_));
	Mem::Add(usageR, Mem::MemUsageOf_Vector(cellIdR_));
	Mem::Add(usageR, Mem::MemUsageOf_Vector(cellIdScratchR_));
	Mem::Add(usageR, Mem::MemUsageOf_Vector(particleOrderR_));
	Mem::Add(usageR, Mem::MemUsageOf_Vector(particleOrderScratchR_));
	Mem::Add(usageR, Mem::MemUsageOf_Vector(cellTableR_));

	return { usageF, usageB, usageR };
}

void NS_CellLinkedList_Compact_CPU::UpdateCellFrame()
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const uint32_t N_R = be_.N_R_;
	if (N_F == 0 && N_B == 0 && N_R == 0)
	{
		cellFrame_ = {};
		cellFrameValid_ = false;
		return;
	}

	bool rebuildFrame = !cellFrameValid_;
	if (!rebuildFrame)
	{
		const int margin = static_cast<int>(cellQueryReach_);
		const int xMinInner = static_cast<int>(cellFrame_.cellOrigX) + margin;
		const int yMinInner = static_cast<int>(cellFrame_.cellOrigY) + margin;
		const int zMinInner = static_cast<int>(cellFrame_.cellOrigZ) + margin;
		const int xMaxInner = static_cast<int>(cellFrame_.cellOrigX)
			+ static_cast<int>(cellFrame_.N_dimX) - 1 - margin;
		const int yMaxInner = static_cast<int>(cellFrame_.cellOrigY)
			+ static_cast<int>(cellFrame_.N_dimY) - 1 - margin;
		const int zMaxInner = static_cast<int>(cellFrame_.cellOrigZ)
			+ static_cast<int>(cellFrame_.N_dimZ) - 1 - margin;

		const auto particlesExceeded = [&](const std::vector<float>& xPos,
			const std::vector<float>& yPos, const std::vector<float>& zPos,
			const uint32_t N) -> bool
			{
				return tbb::parallel_reduce(
					tbb::blocked_range<uint32_t>(0, N), false,
					[&](const tbb::blocked_range<uint32_t>& r, bool exceeded) -> bool
					{
						for (uint32_t i = r.begin(); i != r.end() && !exceeded; i++)
						{
							// particles that left the domain sit outside the clipped frame
							// permanently, so they must not keep triggering rebuilds
							if (!IsInsideSimDomain(xPos[i], yPos[i], zPos[i])) { continue; }

							const int cx = FloorToInt(xPos[i] * cellSizeInv_);
							const int cy = FloorToInt(yPos[i] * cellSizeInv_);
							const int cz = FloorToInt(zPos[i] * cellSizeInv_);
							exceeded = cx < xMinInner || cx > xMaxInner ||
								cy < yMinInner || cy > yMaxInner ||
								cz < zMinInner || cz > zMaxInner;
						}
						return exceeded;
					},
					[](const bool a, const bool b) -> bool { return a || b; },
					tbb::auto_partitioner{}
				);
			};

		rebuildFrame = xMinInner > xMaxInner || yMinInner > yMaxInner || zMinInner > zMaxInner ||
			particlesExceeded(be_.xPosF_, be_.yPosF_, be_.zPosF_, N_F) ||
			particlesExceeded(be_.xPosB_, be_.yPosB_, be_.zPosB_, N_B) ||
			particlesExceeded(be_.xPosR_, be_.yPosR_, be_.zPosR_, N_R);
	}

	if (!rebuildFrame) { return; }

	// rebuild cell frame from current occupied cell AABB
	SceneAABB sceneAABB = ComputeSceneAABB_Combined(
		be_.xPosF_.data(), be_.yPosF_.data(), be_.zPosF_.data(), N_F,
		be_.xPosB_.data(), be_.yPosB_.data(), be_.zPosB_.data(), N_B,
		be_.xPosR_.data(), be_.yPosR_.data(), be_.zPosR_.data(), N_R);

	// particles that left the simulation domain must not stretch the frame, because the dense
	// cell tables scale with its volume (they are dropped from the grid by the cell ID build)
	ClipSceneAABB_ToSimDomain(sceneAABB, be_.simDomainMin_, be_.simDomainMax_);

	const int64_t margin = static_cast<int64_t>(cellQueryReach_)
		+ N_CELL_FRAME_REBUILD_MARGIN;
	const int64_t frameOrigX = static_cast<int64_t>(
		FloorToInt(sceneAABB.minX * cellSizeInv_)) - margin;
	const int64_t frameOrigY = static_cast<int64_t>(
		FloorToInt(sceneAABB.minY * cellSizeInv_)) - margin;
	const int64_t frameOrigZ = static_cast<int64_t>(
		FloorToInt(sceneAABB.minZ * cellSizeInv_)) - margin;
	const int64_t frameMaxX = static_cast<int64_t>(
		FloorToInt(sceneAABB.maxX * cellSizeInv_)) + margin;
	const int64_t frameMaxY = static_cast<int64_t>(
		FloorToInt(sceneAABB.maxY * cellSizeInv_)) + margin;
	const int64_t frameMaxZ = static_cast<int64_t>(
		FloorToInt(sceneAABB.maxZ * cellSizeInv_)) + margin;
	const int64_t N_dimX = frameMaxX - frameOrigX + 1;
	const int64_t N_dimY = frameMaxY - frameOrigY + 1;
	const int64_t N_dimZ = frameMaxZ - frameOrigZ + 1;

	CellCoordFrame nextFrame{};
	if (!TryBuildCellCoordFrame(
		frameOrigX, frameOrigY, frameOrigZ,
		frameMaxX, frameMaxY, frameMaxZ,
		nextFrame))
	{
		SPDLOG_ERROR("Compact CLL cell frame exceeds the supported range [{}, {}, {}].",
			N_dimX, N_dimY, N_dimZ);
		assert(false);
		return;
	}

	cellFrame_ = nextFrame;
	cellFrameValid_ = true;
}

inline bool NS_CellLinkedList_Compact_CPU::IsInsideSimDomain(
	const float x, const float y, const float z) const noexcept
{
	const glm::vec3& min = be_.simDomainMin_;
	const glm::vec3& max = be_.simDomainMax_;
	if (max.x <= min.x || max.y <= min.y || max.z <= min.z) { return true; }

	return x >= min.x && x <= max.x
		&& y >= min.y && y <= max.y
		&& z >= min.z && z <= max.z;
}

inline bool NS_CellLinkedList_Compact_CPU::IsInsideCellFrame(
	int cx, int cy, int cz) const noexcept
{
	const int lx = cx - static_cast<int>(cellFrame_.cellOrigX);
	const int ly = cy - static_cast<int>(cellFrame_.cellOrigY);
	const int lz = cz - static_cast<int>(cellFrame_.cellOrigZ);

	return lx >= 0 && lx < static_cast<int>(cellFrame_.N_dimX)
		&& ly >= 0 && ly < static_cast<int>(cellFrame_.N_dimY)
		&& lz >= 0 && lz < static_cast<int>(cellFrame_.N_dimZ);
}

inline uint64_t NS_CellLinkedList_Compact_CPU::MapToCellId(
	int cx, int cy, int cz) const noexcept
{
	return MapToCellId_Morton(cx, cy, cz);
}

inline uint64_t NS_CellLinkedList_Compact_CPU::MapToCellId_Morton(
	int cx, int cy, int cz) const noexcept
{
	// shift to local cell frame to avoid zig-zag preprocessing
	const int lx = cx - static_cast<int>(cellFrame_.cellOrigX);
	const int ly = cy - static_cast<int>(cellFrame_.cellOrigY);
	const int lz = cz - static_cast<int>(cellFrame_.cellOrigZ);
	assert(lx >= 0 && lx < static_cast<int>(cellFrame_.N_dimX));
	assert(ly >= 0 && ly < static_cast<int>(cellFrame_.N_dimY));
	assert(lz >= 0 && lz < static_cast<int>(cellFrame_.N_dimZ));

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

inline uint64_t NS_CellLinkedList_Compact_CPU::MapToDenseCellIdx(
	int cx, int cy, int cz) const noexcept
{
	const int lx = cx - static_cast<int>(cellFrame_.cellOrigX);
	const int ly = cy - static_cast<int>(cellFrame_.cellOrigY);
	const int lz = cz - static_cast<int>(cellFrame_.cellOrigZ);
	assert(lx >= 0 && lx < static_cast<int>(cellFrame_.N_dimX));
	assert(ly >= 0 && ly < static_cast<int>(cellFrame_.N_dimY));
	assert(lz >= 0 && lz < static_cast<int>(cellFrame_.N_dimZ));

	return (static_cast<uint64_t>(lz) * cellFrame_.N_dimY + static_cast<uint32_t>(ly))
		* cellFrame_.N_dimX + static_cast<uint32_t>(lx);
}

// spread out lower 21 bits of v into lower 63 bits and fill with gaps zeros like 000x00x00x00x00x...
inline uint64_t NS_CellLinkedList_Compact_CPU::SpreadBitsWithGaps_21to63(uint32_t v) noexcept
{
	uint64_t x = static_cast<uint64_t>(v & 0x1fffffU); // lower 21 bits

	x = (x | (x << 32)) & 0x1f00000000ffffULL;
	x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
	x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
	x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
	x = (x | (x << 2)) & 0x1249249249249249ULL;

	return x;
}
