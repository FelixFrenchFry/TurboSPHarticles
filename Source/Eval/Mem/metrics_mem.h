// metrics_mem.h

#pragma once

// external
#include <algorithm>
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
		uint64_t N_usedBytes = 0;
		uint64_t N_allocBytes = 0;
	};

	struct ProcessMemUsage
	{
		// high-water resident process memory report by OS, separate from tracked buckets
		uint64_t N_peakResidentBytes = 0;
	};

	struct ParticleClassMemUsage
	{
		MemUsage fluid{};
		MemUsage boundary{};
		MemUsage rigid{};
	};

	// "buckets" of data structures used by the sim that allocate memory
	enum class MemBucket : uint8_t
	{
		ParticlePropertiesF,			// buffers storing fluid particle properties (position, velocity, pressure, ...)
		ParticlePropertiesB,			// buffers storing boundary particle properties (position, ...)
		ParticlePropertiesR,			// buffers storing rigid particle properties (position, velocity, ...)

		Neighbors_SearchDataStructF,	// data structures used for the neighbor search query of fluid particles (testing for neighborhood with fluid particles)
		Neighbors_SearchDataStructB,	// data structures used for the neighbor search query of fluid particles (testing for neighborhood with boundary particles)
		Neighbors_SearchDataStructR,	// data structures used for the neighbor search query of rigid particles (testing for neighborhood with rigid particles)
		Neighbors_ListStorageF,         // data structures used for the storage of references involving fluid particles
		Neighbors_ListStorageB,         // data structures used for the storage of references involving boundary particles
		Neighbors_ListStorageR,         // data structures used for the storage of references involving rigid particles

		Dist_GlobalState,               // additional full scenario state held by the distributed simulation manager
		Dist_LocalState,                // backend-agnostic rank-local owned + halo particle state
		Dist_ParticleMetadata,          // distributed ownership/classification/index-map metadata
		Dist_ExchangeBuffers,           // persistent distributed send/receive/aggregation buffers
		Dist_Misc,                      // reusable distributed rebuild/export scratches and other misc buffers

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
				"id, position, velocity, predicted velocity, pressure acceleration,\n"
				"source term, diagonal element, pressure, density"
			},
			{
				.bucket = MemBucket::ParticlePropertiesB,
				.type = Base::ParticleType::Boundary,
				.name = "boundary particle properties",
				.tip =
				"Memory used/allocated by buffers that store boundary particle properties, namely:\n"
				"id, position, volB, psiB"
			},
			{
				.bucket = MemBucket::ParticlePropertiesR,
				.type = Base::ParticleType::Rigid,
				.name = "rigid particle properties",
				.tip =
				"Memory used/allocated by buffers that store rigid particle properties, namely:\n"
				"id, body attachment, relative/world position, velocity, force, volR, psiR"
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
				.name = "boundary neighbor search data struct",
				.tip =
				"Memory used/allocated by data structures used for the neighbor search query\n"
				"of boundary particles (testing for neighborhood with boundary particles)"
			},
			{
				.bucket = MemBucket::Neighbors_SearchDataStructR,
				.type = Base::ParticleType::Rigid,
				.name = "rigid neighbor search data struct",
				.tip =
				"Memory used/allocated by data structures used for the neighbor search query\n"
				"of fluid and rigid particles (testing for neighborhood with rigid particles)"
			},
			{
				.bucket = MemBucket::Neighbors_ListStorageF,
				.type = Base::ParticleType::Fluid,
				.name = "fluid neighbor list storage",
				.tip =
				"Memory used/allocated by data structures used for storing references\n"
				"involving fluid particles"
			},
			{
				.bucket = MemBucket::Neighbors_ListStorageB,
				.type = Base::ParticleType::Boundary,
				.name = "boundary neighbor list storage",
				.tip =
				"Memory used/allocated by data structures used for storing references\n"
				"involving boundary particles"
			},
			{
				.bucket = MemBucket::Neighbors_ListStorageR,
				.type = Base::ParticleType::Rigid,
				.name = "rigid neighbor list storage",
				.tip =
				"Memory used/allocated by data structures used for storing references\n"
				"involving rigid particles"
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
		uint64_t simStep = 0;
		double simTime = 0.0;

		// process and tracked memory data
		ProcessMemUsage process{};
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
		a.N_usedBytes += b.N_usedBytes;
		a.N_allocBytes += b.N_allocBytes;
	}
	inline void Clear(MemBucketRecord& record)
	{
		std::ranges::fill(record, MemUsage{});
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
		record.simStep = 0ull;
		record.simTime = 0.0;
		record.process = {};
		Clear(record.mem);
	}

	// helpers for computing memory usage of common data structures
	template<class T, size_t N>
	inline MemUsage MemUsageOf_Array([[maybe_unused]] const std::array<T, N>& src)
	{
		const uint64_t N_usedBytes = static_cast<uint64_t>(N) * sizeof(T);
		const uint64_t N_allocBytes = static_cast<uint64_t>(N) * sizeof(T);

		return MemUsage{ N_usedBytes, N_allocBytes };
	}

	template<class T, class Alloc>
	inline MemUsage MemUsageOf_Vector(const std::vector<T, Alloc>& src)
	{
		const uint64_t N_usedBytes = static_cast<uint64_t>(src.size()) * sizeof(T);
		const uint64_t N_allocBytes = static_cast<uint64_t>(src.capacity()) * sizeof(T);

		return MemUsage{ N_usedBytes, N_allocBytes };
	}

	template<class T, std::size_t N, class Alloc>
	inline MemUsage MemUsageOf_SmallVector(const boost::container::small_vector<T, N, Alloc>& src)
	{
		const uint64_t N_usedBytes = static_cast<uint64_t>(src.size()) * sizeof(T);
		const uint64_t N_allocBytes = static_cast<uint64_t>(src.capacity()) * sizeof(T);

		return MemUsage{ N_usedBytes, N_allocBytes };
	}

	template<class T, std::size_t N, class Alloc>
	inline MemUsage MemUsageOf_SmallVector_HeapOnly(const boost::container::small_vector<T, N, Alloc>& src)
	{
		if (src.capacity() <= N) // no spill -> no heap allocation
		{
			return MemUsage{ 0ull, 0ull };
		}

		const uint64_t N_usedBytes = static_cast<uint64_t>(src.size()) * sizeof(T);
		const uint64_t N_allocBytes = static_cast<uint64_t>(src.capacity()) * sizeof(T);
		// only count spill heap allocation to prevent double-counting
		// inline small_vector storage nested in another data struct

		return MemUsage{ N_usedBytes, N_allocBytes };
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
