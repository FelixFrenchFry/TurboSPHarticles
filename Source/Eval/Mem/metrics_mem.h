// metrics_mem.h
//
// TODO: comment

#pragma once

// external
#include <ankerl/unordered_dense.h>
#include <array>
#include <boost/container/small_vector.hpp>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>



namespace Mem
{
	enum class MemUsageType : uint8_t
	{
		Used,
		Alloc,
		Occupied
	};

	struct MemUsage
	{
		uint64_t usedBytes = 0;
		uint64_t allocBytes = 0;
		uint64_t occupiedBytes = 0;
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

		Misc,

		Count
	};

	struct MemBucketDef
	{
		MemBucket bucket;
		Base::ParticleType type;
		std::string_view name; // display name
		std::string_view tip;  // tooltip description
	};

	constexpr inline std::array<MemBucketDef, static_cast<uint16_t>(MemBucket::Count)> Definitions =
	{
		{
			{
				.bucket = MemBucket::ParticlePropertiesF,
				.type = Base::ParticleType::Fluid,
				.name = "fluid particle properties",
				.tip =
				"Memory used/allocated by buffers that store fluid particle properties, namely:\n"
				"position, velocity, predicted velocity, acceleration, pressure acceleration,\n"
				"viscosity acceleration, external acceleration, source term, diagonal element,\n"
				"system vector element, pressure, density"
			},
			{
				.bucket = MemBucket::ParticlePropertiesB,
				.type = Base::ParticleType::Boundary,
				.name = "bound particle properties",
				.tip =
				"Memory used/allocated by buffers that store boundary particle properties, namely:\n"
				"position, velocity"
			},
			{
				.bucket = MemBucket::Neighbors_SearchDataStructF,
				.type = Base::ParticleType::Fluid,
				.name = "fluid neighbor search data struct",
				.tip =
				"Memory used/allocated by data structures used for the neighbor search query\n"
				"of fluid particles (testing for neighborhood with fluid particles)"
			},
			{
				.bucket = MemBucket::Neighbors_SearchDataStructB,
				.type = Base::ParticleType::Boundary,
				.name = "bound neighbor search data struct",
				.tip =
				"Memory used/allocated by data structures used for the neighbor search query\n"
				"of boundary particles (testing for neighborhood with boundary particles)"
			},
			{
				.bucket = MemBucket::Neighbors_ListStorageF,
				.type = Base::ParticleType::Fluid,
				.name = "fluid neighbor list storage",
				.tip =
				"Memory used/allocated by data structures used for storing references to the\n"
				"fluid particle neighbors of fluid particles"
			},
			{
				.bucket = MemBucket::Neighbors_ListStorageB,
				.type = Base::ParticleType::Boundary,
				.name = "bound neighbor list storage",
				.tip =
				"Memory used/allocated by data structures used for storing references to the\n"
				"boundary particle neighbors of fluid particles"
			},
			{
				.bucket = MemBucket::Misc,
				.type = Base::ParticleType::Fluid,
				.name = "misc",
				.tip =
				"TODO"
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
	inline void Set(MemRecordStep& record, MemBucket bucket, const MemUsage& mem)
		{ record.mem[GetIdx(bucket)] = mem; }
	inline void Add(MemUsage& a, const MemUsage& b)
	{
		a.usedBytes += b.usedBytes;
		a.allocBytes += b.allocBytes;
		a.occupiedBytes += b.occupiedBytes;
	}

	// helpers for computing memory usage of common data structures
	template<class T, size_t N>
	inline MemUsage MemUsageOf_Array([[maybe_unused]] const std::array<T, N>& src)
	{
		const uint64_t used = static_cast<uint64_t>(N) * sizeof(T);
		const uint64_t alloc = static_cast<uint64_t>(N) * sizeof(T);
		const uint64_t occupied = sizeof(src);

		return MemUsage{ used, alloc, occupied };
	}

	template<class T, class Alloc>
	inline MemUsage MemUsageOf_Vector(const std::vector<T, Alloc>& src)
	{
		const uint64_t used = static_cast<uint64_t>(src.size()) * sizeof(T);
		const uint64_t alloc = static_cast<uint64_t>(src.capacity()) * sizeof(T);
		const uint64_t occupied = sizeof(src) + alloc;

		return MemUsage{ used, alloc, occupied };
	}

	template<class T, class Alloc>
	inline MemUsage MemUsageOf_Vector_HeapOnly(const std::vector<T, Alloc>& src)
	{
		const uint64_t used = static_cast<uint64_t>(src.size()) * sizeof(T);
		const uint64_t alloc = static_cast<uint64_t>(src.capacity()) * sizeof(T);
		// only count heap allocation as occupied memory
		// (to prevent double-counting if nested in another data struct)
		const uint64_t occupied = alloc;

		return MemUsage{ used, alloc, occupied };
	}

	template<class T, std::size_t N, class Alloc>
	inline MemUsage MemUsageOf_SmallVector(const boost::container::small_vector<T, N, Alloc>& src)
	{
		const uint64_t used = static_cast<uint64_t>(src.size()) * sizeof(T);
		const uint64_t alloc = static_cast<uint64_t>(src.capacity()) * sizeof(T);

		// if inlined size is exceeded, add size of the spill into heap to the occupied amount
		const bool spilled = src.capacity() > N;
		const uint64_t heap = spilled ? static_cast<uint64_t>(src.capacity()) * sizeof(T) : 0ull;
		const uint64_t occupied = heap + sizeof(src); // (inlined mem is included in sizeof(src))

		return MemUsage{ used, alloc, occupied };
	}

	template<class T, std::size_t N, class Alloc>
	inline MemUsage MemUsageOf_SmallVector_HeapOnly(const boost::container::small_vector<T, N, Alloc>& src)
	{
		if (src.capacity() <= N) // no spill -> no heap allocation
		{
			return MemUsage{ 0ull, 0ull, 0ull };
		}

		const uint64_t used = static_cast<uint64_t>(src.size()) * sizeof(T);
		const uint64_t alloc = static_cast<uint64_t>(src.capacity()) * sizeof(T);
		// only count heap allocation as occupied memory
		// (to prevent double-counting if nested in another data struct)
		const uint64_t occupied = alloc;

		return MemUsage{ used, alloc, occupied };
	}

	template<class T, class AllocInner, class AllocOuter>
	inline MemUsage MemUsageOf_VectorOfVector(
		const std::vector<std::vector<T, AllocInner>, AllocOuter>& src)
	{
		// outer object + outer heap block containing inner vector objects
		MemUsage mem = MemUsageOf_Vector(src);

		// add only the heap blocks of inner vectors (no sizeof(inner vector))
		for (auto const& v : src)
		{
			Add(mem, MemUsageOf_Vector_HeapOnly(v));
		}

		return mem;
	}

	template<class T, std::size_t N, class Alloc>
	inline MemUsage MemUsageOf_VectorOfSmallVector(
		const std::vector<boost::container::small_vector<T, N, Alloc>>& src)
	{
		// outer object + outer heap block containing inner vector objects
		MemUsage total = MemUsageOf_Vector(src);

		// add only the heap blocks of inner small_vectors (no sizeof(inner small_vector))
		for (auto const& sv : src)
		{
			Add(total, MemUsageOf_SmallVector_HeapOnly(sv)); // only spill heap
		}

		return total;
	}

	// TODO: implement this using a custom allocator to track the memory usage and avoid non-compiling templates
	/*
	template<class MapT>
	inline MemUsage MemUsageOf_UnorderedDenseMap_TableOnly(const MapT& src)
	{
		using BucketT = typename MapT::bucket_type;

		MemUsage total{};

		// values storage (dense array of value_type, typically std::pair<K, V>)
		auto const& values = src.values();
		using ValueT = typename std::decay_t<decltype(values)>::value_type;

		const uint64_t valuesUsed = static_cast<uint64_t>(values.size()) * sizeof(ValueT);
		const uint64_t valuesAlloc = static_cast<uint64_t>(values.capacity()) * sizeof(ValueT);

		// heap-only for the values array
		Add(total, MemUsage{ valuesUsed, valuesAlloc, valuesAlloc });

		// buckets storage
		const size_t bucketCount = src.bucket_count();
		const uint64_t bucketsAlloc = static_cast<uint64_t>(bucketCount) * sizeof(BucketT);

		// bucket array is fully present -> used == alloc (simple and consistent)
		Add(total, MemUsage{ bucketsAlloc, bucketsAlloc, bucketsAlloc });

		// root object storage
		total.occupiedBytes += sizeof(src);

		return total;
	}

	template<class K, class V, class... Rest>
	inline MemUsage MemUsageOf_UnorderedDenseMap(
		const ankerl::unordered_dense::map<K, V, Rest...>& src)
	{
		using MapT = ankerl::unordered_dense::map<K, V, Rest...>;
		return MemUsageOf_UnorderedDenseMap_TableOnly<MapT>(src);
	}

	template<class K, class T, class AllocV, class... Rest>
	inline MemUsage MemUsageOf_UnorderedDenseMap(
		const ankerl::unordered_dense::map<K, std::vector<T, AllocV>, Rest...>& src)
	{
		// generic table cost (entries + buckets + sizeof(map))
		using MapT = ankerl::unordered_dense::map<K, std::vector<T, AllocV>, Rest...>;
		MemUsage total = MemUsageOf_UnorderedDenseMap_TableOnly<MapT>(src);

		// add only the heap blocks of inner vectors (prevents double-counting sizeof(vector))
		for (auto const& kv : src.values())
		{
			Add(total, MemUsageOf_Vector_HeapOnly(kv.second));
		}

		return total;
	}
	*/

} // namespace Mem
