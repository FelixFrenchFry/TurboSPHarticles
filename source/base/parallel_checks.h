// parallel_checks.h
#pragma once

// external
#include <atomic>
#include <cstdint>
#include <span>
#include <vector>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/partitioner.h>



namespace Check
{
	// element count below which a serial check is used instead of the parallel check
	inline constexpr uint32_t PARALLEL_CHECK_MIN_ELEMENTS = 4096;

	// counts elements in [ firstIdx, lastIdx_Excl ) for which the predicate reports a violation
	// (the predicate must be read-only, since it runs concurrently and in unspecified order)
	template <typename Predicate>
	uint32_t CountViolations(
		const uint32_t firstIdx,
		const uint32_t lastIdx_Excl,
		Predicate isViolation)
	{
		if (lastIdx_Excl <= firstIdx) { return 0; }

		if (lastIdx_Excl - firstIdx < PARALLEL_CHECK_MIN_ELEMENTS)
		{
			uint32_t N_violations = 0;
			for (uint32_t idx = firstIdx; idx != lastIdx_Excl; idx++)
			{
				N_violations += isViolation(idx) ? 1u : 0u;
			}

			return N_violations;
		}

		return tbb::parallel_reduce(
			tbb::blocked_range<uint32_t>(firstIdx, lastIdx_Excl), 0u,
			[&](const tbb::blocked_range<uint32_t>& range, uint32_t N_violations) -> uint32_t
			{
				for (uint32_t idx = range.begin(); idx != range.end(); idx++)
				{
					N_violations += isViolation(idx) ? 1u : 0u;
				}

				return N_violations;
			},
			[](const uint32_t a, const uint32_t b) -> uint32_t { return a + b; },
			tbb::auto_partitioner{}
		);
	}

	// counts adjacent pairs in [ firstIdx, lastIdx_Excl ) for which the predicate reports a violation
	// (the predicate receives the preceding and current index and must be read-only)
	template <typename Predicate>
	uint32_t CountAdjacentViolations(
		const uint32_t firstIdx,
		const uint32_t lastIdx_Excl,
		Predicate isViolation)
	{
		if (lastIdx_Excl <= firstIdx ||
			lastIdx_Excl - firstIdx < 2) { return 0; }

		return CountViolations(
			firstIdx + 1,
			lastIdx_Excl,
			[&](const uint32_t currentIdx)
			{
				return isViolation(currentIdx - 1, currentIdx);
			});
	}

	// finds the first element in [ firstIdx, lastIdx_Excl ) for which the predicate
	// reports a violation, or returns lastIdx_Excl if there is none
	template <typename Predicate>
	uint32_t FindFirstViolation(
		const uint32_t firstIdx,
		const uint32_t lastIdx_Excl,
		Predicate isViolation)
	{
		if (lastIdx_Excl <= firstIdx) { return lastIdx_Excl; }

		for (uint32_t idx = firstIdx; idx != lastIdx_Excl; idx++)
		{
			if (isViolation(idx)) { return idx; }
		}

		return lastIdx_Excl;
	}

	// claims one bit per index to find the lowest duplicate position, or the index count if none
	// (expects every index to be range checked against N_range first, and reports either
	// position of a duplicate pair, because the losing claim depends on the worker order)
	inline uint32_t FindFirstDuplicateIdx(
		const std::span<const uint32_t> indices,
		const uint32_t N_range,
		std::vector<uint32_t>& claimBitsScratch)
	{
		const uint32_t N_indices = static_cast<uint32_t>(indices.size());
		claimBitsScratch.assign((N_range + 31u) / 32u, 0u);
		std::atomic<uint32_t> firstDuplicateIdx{ N_indices };

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_indices),
			[&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t idx = range.begin(); idx != range.end(); idx++)
			{
				const uint32_t claimedIdx = indices[idx];
				const uint32_t bitMask = 1u << (claimedIdx & 31u);
				std::atomic_ref<uint32_t> bitWord{ claimBitsScratch[claimedIdx >> 5] };
				if ((bitWord.fetch_or(bitMask, std::memory_order_relaxed) & bitMask) == 0)
				{
					continue;
				}

				uint32_t observedIdx = firstDuplicateIdx.load(std::memory_order_relaxed);
				while (idx < observedIdx &&
					!firstDuplicateIdx.compare_exchange_weak(
						observedIdx, idx, std::memory_order_relaxed))
				{
				}
			}
		});

		return firstDuplicateIdx.load(std::memory_order_relaxed);
	}

} // namespace Check
