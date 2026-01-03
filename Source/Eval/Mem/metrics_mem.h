// metrics_mem.h
//
// TODO: comment

#pragma once

// external
#include <ankerl/unordered_dense.h>
#include <array>
#include <boost/container/small_vector.hpp>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>



namespace Mem
{
	struct MemUsage
	{
		uint64_t usedBytes = 0;
		uint64_t allocBytes = 0;
	};

	// "buckets" of data structures used by the sim that allocate memory
	enum class MemBucket : uint8_t
	{
		ParticlePropertiesF,			// buffers storing fluid particle properties (position, velocity, pressure, ...)
		ParticlePropertiesB,			// buffers storing boundary particle properties (position, ...)

		Neighbors_SearchDataStructF,	// data structures used for the neighbor search query of fluid particles (testing for neighborhood with fluid particles)
		Neighbors_SearchDataStructB,	// data structures used for the neighbor search query of fluid particles (testing for neighborhood with boundary particles)
		Neighbors_ListStorageF,			// data structures used for the storage of references to fluid neighbors of fluid particles
		Neighbors_ListStorageB,			// data structures used for the storage of references to boundary neighbors of fluid particles

		ScratchBuffers,

		Misc,

		Test,
		Count
	};

	struct MemBucketDef
	{
		MemBucket bucket;
		std::string_view name; // display name
		std::string_view tip;  // tooltip description
	};

	constexpr inline std::array<MemBucketDef, static_cast<uint16_t>(MemBucket::Count)> Definitions =
	{
		{
			{
				.bucket = MemBucket::ParticlePropertiesF,
				.name = "fluid particle properties",
				.tip =
				"Memory used/allocated by buffers that store fluid particle properties, namely:\n"
				"position, velocity, predicted velocity, acceleration, pressure acceleration,\n"
				"viscosity acceleration, external acceleration, source term, diagonal element,\n"
				"system vector element, pressure, density"
			},
			{
				.bucket = MemBucket::ParticlePropertiesB,
				.name = "bound particle properties",
				.tip =
				"Memory used/allocated by buffers that store boundary particle properties, namely:\n"
				"position, velocity"
			},
			{
				.bucket = MemBucket::Neighbors_SearchDataStructF,
				.name = "fluid neighbor search data struct",
				.tip =
				"Memory used/allocated by data structures used for the neighbor search query\n"
				"of fluid particles (testing for neighborhood with fluid particles)"
			},
			{
				.bucket = MemBucket::Neighbors_SearchDataStructB,
				.name = "bound neighbor search data struct",
				.tip =
				"Memory used/allocated by data structures used for the neighbor search query\n"
				"of boundary particles (testing for neighborhood with boundary particles)"
			},
			{
				.bucket = MemBucket::Neighbors_ListStorageF,
				.name = "fluid neighbor list storage",
				.tip =
				"Memory used/allocated by data structures used for storing references to the\n"
				"fluid particle neighbors of fluid particles"
			},
			{
				.bucket = MemBucket::Neighbors_ListStorageB,
				.name = "bound neighbor list storage",
				.tip =
				"Memory used/allocated by data structures used for storing references to the\n"
				"boundary particle neighbors of fluid particles"
			},
			{
				.bucket = MemBucket::ScratchBuffers,
				.name = "scratch buffers",
				.tip =
				"TODO"
			},
			{
				.bucket = MemBucket::Misc,
				.name = "Misc",
				.tip =
				"TODO"
			},
			{
				.bucket = MemBucket::Test,
				.name = "Test",
				.tip =
				"Memory used/allocated by custom test data structures"
			}
		}
	};
	static_assert(Definitions.size() == static_cast<uint8_t>(MemBucket::Count),
		"Mem::Definitions must list all MemBucket entries");

	constexpr inline uint8_t GetIdx(MemBucket bucket) { return static_cast<uint8_t>(bucket); }

	// record of all per-metric memory usage data for one particular sim step
	struct MemRecordStep
	{
		// metadata
		uint32_t simStep = 0;
		double simTime = 0.0;

		// stat data
		std::array<MemUsage, GetIdx(MemBucket::Count)> mem{};
	};

	inline MemUsage Get(const MemRecordStep& record, MemBucket bucket)
		{ return record.mem[GetIdx(bucket)]; }
	inline void Set(MemRecordStep& record, MemBucket bucket, MemUsage xy)
		{ record.mem[GetIdx(bucket)] = xy; }
	inline void Add(MemUsage& a, const MemUsage& b)
		{ a.usedBytes += b.usedBytes; a.allocBytes += b.allocBytes; }

	// helper for computing memory usage of array storing arbitrary type T
	template<class T, size_t N>
	inline MemUsage MemUsageOf_Array(const std::array<T, N>& src)
	{
		const uint64_t used = static_cast<uint64_t>(N) * sizeof(T);
		const uint64_t alloc = static_cast<uint64_t>(N) * sizeof(T);

		return MemUsage{ used, alloc };
	}

	// helper for computing memory usage of vector storing arbitrary type T
	template<class T>
	inline MemUsage MemUsageOf_Vector(const std::vector<T>& src)
	{
		const uint64_t used = static_cast<uint64_t>(src.size()) * sizeof(T);
		const uint64_t alloc = static_cast<uint64_t>(src.capacity()) * sizeof(T);

		return MemUsage{ used, alloc };
	}

	// helper for computing memory usage of a small vector storing arbitrary type T
	template<class T, std::size_t N, class Alloc>
	inline MemUsage MemUsageOf_SmallVector(const boost::container::small_vector<T, N, Alloc>& src)
	{
		const uint64_t used = static_cast<uint64_t>(src.size()) * sizeof(T);
		uint64_t alloc = static_cast<uint64_t>(N) * sizeof(T);

		// if inlined size is exceeded, use size of the spill into heap allocation
		if (src.capacity() > N)
		{
			// TODO: add instead of overwriting to get actual footprint?
			alloc = static_cast<uint64_t>(src.capacity()) * sizeof(T);
		}

		return MemUsage{ used, alloc };
	}

	// helper for computing memory usage of an unordered dense map with std::vector<uint32_t> for values
	template<class K, class... Rest>
	inline MemUsage MemUsageOf_UnorderedDenseMap(
		const ankerl::unordered_dense::map<K, std::vector<uint32_t>, Rest...>& src)
	{
		uint64_t used = 0;
		uint64_t alloc = 0;

		// container that holds the hash map keys
		const auto& keyStorage = src.keys();
		used += static_cast<uint64_t>(keyStorage.size()) * sizeof(K);
		alloc += static_cast<uint64_t>(keyStorage.capacity()) * sizeof(K);

		// container that holds the hash map values
		const auto& valueStorage = src.values(); // stores std::vector<uint32_t> objects, not the data inside
		used += static_cast<uint64_t>(valueStorage.size()) * sizeof(std::vector<uint32_t>);
		alloc += static_cast<uint64_t>(valueStorage.capacity()) * sizeof(std::vector<uint32_t>);

		// container that holds the hash map bucket metadata (heuristic)
		constexpr uint64_t bucketOverheadBytes = 8ull; // in this case of ankerl::unordered_dense::map<...>
		used += static_cast<uint64_t>(src.bucket_count()) * bucketOverheadBytes;
		alloc += static_cast<uint64_t>(src.bucket_count()) * bucketOverheadBytes;

		// nested allocations in the values (in this case of std::vector<uint32_t>)
		for (auto const& bucketVec : valueStorage)
		{
			used += static_cast<uint64_t>(bucketVec.size()) * sizeof(uint32_t);
			alloc += static_cast<uint64_t>(bucketVec.capacity()) * sizeof(uint32_t);
		}

		return MemUsage{ used, alloc };
	}

} // namespace Mem
