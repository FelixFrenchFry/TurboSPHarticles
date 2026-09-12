// particle_types.h
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>



namespace Sim
{
	inline constexpr uint64_t INVALID_PARTICLE_ID =
		std::numeric_limits<uint64_t>::max(); // 18,446,744,073,709,551,615
	inline constexpr uint32_t MaxLocalIndexCount =
		std::numeric_limits<uint32_t>::max(); // 4,294,967,295
	inline constexpr uint32_t MaxLocalParticleCount =
		static_cast<uint32_t>(std::numeric_limits<int32_t>::max()); // 2,147,483,647

	inline uint64_t CheckedParticleAdd(
		const uint64_t a,
		const uint64_t b,
		const char* const context)
	{
		if (b > std::numeric_limits<uint64_t>::max() - a)
		{
			throw std::overflow_error(std::string(context) + " particle count addition overflow.");
		}

		return a + b;
	}

	inline uint64_t CheckedParticleMultiply(
		const uint64_t a,
		const uint64_t b,
		const char* const context)
	{
		if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
		{
			throw std::overflow_error(std::string(context) + " particle count multiplication overflow.");
		}

		return a * b;
	}

	inline uint64_t CheckedParticleId(
		const uint64_t base,
		const uint64_t offset,
		const char* const context)
	{
		const uint64_t id = CheckedParticleAdd(base, offset, context);
		if (id == INVALID_PARTICLE_ID)
		{
			throw std::overflow_error(std::string(context) + " produced the invalid particle ID sentinel.");
		}

		return id;
	}

	inline uint32_t CheckedLocalParticleCount(
		const uint64_t count,
		const char* const context)
	{
		if (count > MaxLocalParticleCount)
		{
			throw std::length_error(std::string(context) + " exceeds the local particle limit.");
		}

		return static_cast<uint32_t>(count);
	}

	inline uint32_t CheckedLocalIndexCount(
		const uint64_t count,
		const char* const context)
	{
		if (count > MaxLocalIndexCount)
		{
			throw std::length_error(std::string(context) + " exceeds local index storage.");
		}

		return static_cast<uint32_t>(count);
	}

	inline uint32_t CheckedPaddedLocalCount(
		const uint32_t count,
		const uint32_t alignment,
		const char* const context)
	{
		if (alignment == 0)
		{
			throw std::invalid_argument(std::string(context) + " has zero alignment.");
		}

		const uint64_t N_chunks = count == 0
			? 0
			: 1ull + (static_cast<uint64_t>(count) - 1ull) / alignment;
		return CheckedLocalIndexCount(
			CheckedParticleMultiply(N_chunks, alignment, context),
			context);
	}

	inline uint64_t CheckedCapacityWithSlack(
		const uint64_t required,
		const uint64_t extra,
		const uint64_t maxCapacity,
		const char* const context)
	{
		if (required > maxCapacity)
		{
			throw std::length_error(std::string(context) + " exceeds storage capacity.");
		}

		return extra > maxCapacity - required
			? maxCapacity
			: required + extra;
	}

	inline uint32_t CheckedCombinedLocalParticleCount(
		const uint64_t a,
		const uint64_t b,
		const char* const context)
	{
		return CheckedLocalParticleCount(CheckedParticleAdd(a, b, context), context);
	}

} // namespace Sim
