// metrics_mem.h

#pragma once

// external
#include <array>
#include <boost/container/small_vector.hpp>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

// internal
#include "Source/Base/enum_hub.h"



namespace Mem
{
	enum class MemSpace : uint8_t
	{
		RAM,
		VRAM,

		Count
	};

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

		Misc,

		Dist_GlobalState,               // additional full scenario state held by the distributed simulation manager
		Dist_LocalState,                // backend-agnostic rank-local owned + halo particle state
		Dist_ParticleMetadata,          // distributed ownership/classification/index-map metadata
		Dist_ExchangeBuffers,           // persistent distributed send/receive/aggregation buffers
		Dist_Misc,                      // reusable distributed rebuild/export scratches and other misc buffers

		Count
	};

	struct MemBucketDef
	{
		MemBucket bucket;
		Base::ParticleType type;
		std::string_view name; // display name
		std::string_view tip;  // tooltip description
	};

	struct MemSpaceDef
	{
		MemSpace space;
		std::string_view name;	// display name
		std::string_view key;	// CSV key
	};

	constexpr inline std::array<MemSpaceDef, static_cast<uint16_t>(MemSpace::Count)> SpaceDefinitions =
	{
		{
			{
				.space = MemSpace::RAM,
				.name = "RAM",
				.key = "ram"
			},
			{
				.space = MemSpace::VRAM,
				.name = "VRAM",
				.key = "vram"
			}
		}
	};
	static_assert(SpaceDefinitions.size() == static_cast<uint8_t>(MemSpace::Count),
		"Mem::SpaceDefinitions must list all MemSpace entries");

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
			},
			{
				.bucket = MemBucket::Dist_GlobalState,
				.type = Base::ParticleType::Fluid,
				.name = "dist global state",
				.tip =
				"Memory used/allocated by the full scenario state retained by the\n"
				"distributed simulation manager"
			},
			{
				.bucket = MemBucket::Dist_LocalState,
				.type = Base::ParticleType::Fluid,
				.name = "dist local state",
				.tip =
				"Memory used/allocated by backend-agnostic rank-local owned and\n"
				"halo particle arrays"
			},
			{
				.bucket = MemBucket::Dist_ParticleMetadata,
				.type = Base::ParticleType::Fluid,
				.name = "dist particle metadata",
				.tip =
				"Memory used/allocated by distributed particle class lists and\n"
				"local/global index maps"
			},
			{
				.bucket = MemBucket::Dist_ExchangeBuffers,
				.type = Base::ParticleType::Fluid,
				.name = "dist exchange buffers",
				.tip =
				"Memory used/allocated by persistent distributed send, receive,\n"
				"and aggregation buffers"
			},
			{
				.bucket = MemBucket::Dist_Misc,
				.type = Base::ParticleType::Fluid,
				.name = "dist misc",
				.tip =
				"Memory used/allocated by reusable distributed rebuild and export\n"
				"scratches and other misc buffers"
			}
		}
	};
	static_assert(Definitions.size() == static_cast<uint8_t>(MemBucket::Count),
		"Mem::Definitions must list all MemBucket entries");

	constexpr inline uint8_t GetIdx(MemSpace space) { return static_cast<uint8_t>(space); }
	constexpr inline uint8_t GetIdx(MemBucket bucket) { return static_cast<uint8_t>(bucket); }

	using MemBucketRecord =
		std::array<MemUsage, GetIdx(MemBucket::Count)>;
	using MemSpaceRecord =
		std::array<MemBucketRecord, GetIdx(MemSpace::Count)>;

	// record of all per-metric memory usage data for one particular sim step
	struct MemRecordStep
	{
		// metadata
		uint32_t simStep = 0;
		double simTime = 0.0;

		// stat data
		MemSpaceRecord mem{};
	};

	inline MemUsage Get(const MemRecordStep& record, MemSpace space, MemBucket bucket)
		{ return record.mem[GetIdx(space)][GetIdx(bucket)]; }
	inline MemUsage Get(const MemRecordStep& record, MemBucket bucket)
		{ return Get(record, MemSpace::RAM, bucket); }
	inline void Set(MemRecordStep& record, MemSpace space, MemBucket bucket, const MemUsage& mem)
		{ record.mem[GetIdx(space)][GetIdx(bucket)] = mem; }
	inline void Set(MemRecordStep& record, MemBucket bucket, const MemUsage& mem)
		{ Set(record, MemSpace::RAM, bucket, mem); }
	inline void Add(MemUsage& a, const MemUsage& b)
	{
		a.usedBytes += b.usedBytes;
		a.allocBytes += b.allocBytes;
	}
	inline void Clear(MemBucketRecord& record)
	{
		record.fill(MemUsage{});
	}
	inline void Clear(MemSpaceRecord& record)
	{
		for (auto& spaceRecord : record)
		{
			Clear(spaceRecord);
		}
	}
	inline void Clear(MemRecordStep& record)
	{
		record.simStep = 0u;
		record.simTime = 0.0;
		Clear(record.mem);
	}

	// helpers for computing memory usage of common data structures
	template<class T, size_t N>
	inline MemUsage MemUsageOf_Array([[maybe_unused]] const std::array<T, N>& src)
	{
		const uint64_t used = static_cast<uint64_t>(N) * sizeof(T);
		const uint64_t alloc = static_cast<uint64_t>(N) * sizeof(T);

		return MemUsage{ used, alloc };
	}

	template<class T, class Alloc>
	inline MemUsage MemUsageOf_Vector(const std::vector<T, Alloc>& src)
	{
		const uint64_t used = static_cast<uint64_t>(src.size()) * sizeof(T);
		const uint64_t alloc = static_cast<uint64_t>(src.capacity()) * sizeof(T);

		return MemUsage{ used, alloc };
	}

	template<class T, std::size_t N, class Alloc>
	inline MemUsage MemUsageOf_SmallVector(const boost::container::small_vector<T, N, Alloc>& src)
	{
		const uint64_t used = static_cast<uint64_t>(src.size()) * sizeof(T);
		const uint64_t alloc = static_cast<uint64_t>(src.capacity()) * sizeof(T);

		return MemUsage{ used, alloc };
	}

	template<class T, std::size_t N, class Alloc>
	inline MemUsage MemUsageOf_SmallVector_HeapOnly(const boost::container::small_vector<T, N, Alloc>& src)
	{
		if (src.capacity() <= N) // no spill -> no heap allocation
		{
			return MemUsage{ 0ull, 0ull };
		}

		const uint64_t used = static_cast<uint64_t>(src.size()) * sizeof(T);
		const uint64_t alloc = static_cast<uint64_t>(src.capacity()) * sizeof(T);
		// only count spill heap allocation to prevent double-counting
		// inline small_vector storage nested in another data struct

		return MemUsage{ used, alloc };
	}

	template<class T, class AllocInner, class AllocOuter>
	inline MemUsage MemUsageOf_VectorOfVector(
		const std::vector<std::vector<T, AllocInner>, AllocOuter>& src)
	{
		// outer heap block containing inner vector objects
		MemUsage mem = MemUsageOf_Vector(src);

		// add the heap blocks owned by each inner vector
		for (auto const& v : src)
		{
			Add(mem, MemUsageOf_Vector(v));
		}

		return mem;
	}

	template<class T, std::size_t N, class Alloc>
	inline MemUsage MemUsageOf_VectorOfSmallVector(
		const std::vector<boost::container::small_vector<T, N, Alloc>>& src)
	{
		// outer heap block containing inner small_vector objects
		MemUsage total = MemUsageOf_Vector(src);

		// add only spill heap blocks of inner small_vectors
		for (auto const& sv : src)
		{
			Add(total, MemUsageOf_SmallVector_HeapOnly(sv));
		}

		return total;
	}

} // namespace Mem
