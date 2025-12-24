// neighbor_lists.cpp

// external
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>

// internal
#include "neighbor_lists.h"

#include "spdlog/spdlog.h"


using namespace Sim;

// -----------------------------------------------------------------------------
// --- [ small buffers neighbor lists ] ----------------------------------------
// -----------------------------------------------------------------------------

void NeighborLists_SmallBuffers_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B)
{
    N_F_ = N_F;
    N_B_ = N_B;

    neighborsFF_.resize(N_F_);
    neighborsFB_.resize(N_F_);
}

void NeighborLists_SmallBuffers_CPU01::ClearOne(
	const uint32_t i)
{
    neighborsFF_[i].clear();
    neighborsFB_[i].clear();
}

void NeighborLists_SmallBuffers_CPU01::ClearAll()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
                neighborsFF_[i].clear();
                neighborsFB_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_SmallBuffers_CPU01::Finalize()
{
	// nothing to do
}

void NeighborLists_SmallBuffers_CPU01::AddNbrFF(
	const uint32_t i, const uint32_t j)
{
	neighborsFF_[i].push_back(j);
}

void NeighborLists_SmallBuffers_CPU01::AddNbrFB(
	const uint32_t i, const uint32_t j)
{
	neighborsFB_[i].push_back(j);
}

void NeighborLists_SmallBuffers_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0) { neighborsFF_[i].clear(); return; }

	// overwrite and resize combined
	neighborsFF_[i].assign(js, js + count);
}

void NeighborLists_SmallBuffers_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0) { neighborsFB_[i].clear(); return; }

	// overwrite and resize combined
	neighborsFB_[i].assign(js, js + count);
}

NeighborSpan NeighborLists_SmallBuffers_CPU01::FF(
	const uint32_t i) const
{
	const auto& nbrs = neighborsFF_[i];
	return NeighborSpan{ nbrs.data(), static_cast<uint32_t>(nbrs.size()) };
}

NeighborSpan NeighborLists_SmallBuffers_CPU01::FB(
	const uint32_t i) const
{
	const auto& nbrs = neighborsFB_[i];
	return NeighborSpan{ nbrs.data(), static_cast<uint32_t>(nbrs.size()) };
}

// -----------------------------------------------------------------------------
// --- [ fixed arrays neighbor lists ] -----------------------------------------
// -----------------------------------------------------------------------------

void NeighborLists_FixedArrays_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B)
{
	N_F_ = N_F;
	N_B_ = N_B;

	nbrCountFF_.assign(N_F_, 0);
	nbrCountFB_.assign(N_F_, 0);

	neighborsFF_.resize(static_cast<size_t>(N_F_) * N_MAX_FF_);
	neighborsFB_.resize(static_cast<size_t>(N_F_) * N_MAX_FB_);
}

void NeighborLists_FixedArrays_CPU01::ClearOne(
	const uint32_t i)
{
	nbrCountFF_[i] = 0;
	nbrCountFB_[i] = 0;
}

void NeighborLists_FixedArrays_CPU01::ClearAll()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				nbrCountFF_[i] = 0;
				nbrCountFB_[i] = 0;
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_FixedArrays_CPU01::Finalize()
{
	// nothing to do
}

void NeighborLists_FixedArrays_CPU01::AddNbrFF(
	const uint32_t i, const uint32_t j)
{
	uint32_t& count = nbrCountFF_[i];

	if (count >= N_MAX_FF_) // neighbor list full, ignore new neighbor
	{
		return;
	}

	const uint64_t base = static_cast<size_t>(i) * N_MAX_FF_;
	neighborsFF_[base + count] = j;
	count++;
}

void NeighborLists_FixedArrays_CPU01::AddNbrFB(
	const uint32_t i, const uint32_t j)
{
	uint32_t& count = nbrCountFB_[i];

	if (count >= N_MAX_FB_) // neighbor list full, ignore new neighbor
	{
		return;
	}

	const uint64_t base = static_cast<size_t>(i) * N_MAX_FB_;
	neighborsFB_[base + count] = j;
	count++;
}

void NeighborLists_FixedArrays_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, uint32_t count)
{
	// clamp to max size
	count = std::min(count, N_MAX_FF_);
	nbrCountFF_[i] = count;

	const uint64_t base = static_cast<uint64_t>(i) * N_MAX_FF_;
	if (count)
	{
		std::memcpy(neighborsFF_.data() + base, js, sizeof(uint32_t) * count);
	}
}

void NeighborLists_FixedArrays_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, uint32_t count)
{
	// clamp to max size
	count = std::min(count, N_MAX_FB_);
	nbrCountFB_[i] = count;

	const uint64_t base = static_cast<uint64_t>(i) * N_MAX_FB_;
	if (count)
	{
		std::memcpy(neighborsFB_.data() + base, js, sizeof(uint32_t) * count);
	}
}

NeighborSpan NeighborLists_FixedArrays_CPU01::FF(
	const uint32_t i) const
{
	const uint64_t base = static_cast<uint64_t>(i) * N_MAX_FF_;
	const uint32_t counter = nbrCountFF_[i];

	// data pointer to the start of the neighbor list for particle i and its size
	return NeighborSpan{ neighborsFF_.data() + base, counter};
}

NeighborSpan NeighborLists_FixedArrays_CPU01::FB(
	const uint32_t i) const
{
	const uint64_t base = static_cast<uint64_t>(i) * N_MAX_FB_;
	const uint32_t counter = nbrCountFB_[i];

	// data pointer to the start of the neighbor list for particle i and its size
	return NeighborSpan{ neighborsFB_.data() + base, counter };
}

// -----------------------------------------------------------------------------
// --- [ compact vectors neighbor lists ] --------------------------------------
// -----------------------------------------------------------------------------

void NeighborLists_CompactVectors_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B)
{
	N_F_ = N_F;
	N_B_ = N_B;

	// temporary 2D vector structure
	tmpFF_.clear();
	tmpFB_.clear();
	tmpFF_.resize(N_F_);
	tmpFB_.resize(N_F_);

	// compact vector structure
	offsetsFF_.resize(N_F_ + 1, 0);
	offsetsFB_.resize(N_F_ + 1, 0);
	neighborsFF_.clear();
	neighborsFB_.clear();
}

void NeighborLists_CompactVectors_CPU01::ClearOne(
	const uint32_t i)
{
	tmpFF_[i].clear();
	tmpFB_[i].clear();
}

void NeighborLists_CompactVectors_CPU01::ClearAll()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				tmpFF_[i].clear();
				tmpFB_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_CompactVectors_CPU01::Finalize()
{
	if (N_F_ == 0) // nothing to do for empty sim
	{
		offsetsFF_.assign(1, 0);
		offsetsFB_.assign(1, 0);
		neighborsFF_.clear();
		neighborsFB_.clear();
		
		return;
	}

	// --- [ build compact vector for FF ] -------------------------------------

	// build offsets via prefix sum
	offsetsFF_[0] = 0;
	uint64_t runningFF = 0;
	for (uint32_t i = 0; i < N_F_; i++)
	{
		offsetsFF_[i] = runningFF;
		runningFF += static_cast<uint64_t>(tmpFF_[i].size());
	}
	offsetsFF_[N_F_] = runningFF;

	// allocate flat neighbor storage
	neighborsFF_.resize(runningFF);

	// copy data from temp into final data structure
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const uint64_t base = offsetsFF_[i];
				const uint32_t count = static_cast<uint32_t>(tmpFF_[i].size());

				if (count)
				{
					std::memcpy(neighborsFF_.data() + base, tmpFF_[i].data(),
						count * sizeof(uint32_t));
				}
			}
		},
		tbb::auto_partitioner{}
	);

	// --- [ build compact vector for FB ] -------------------------------------

	// build offsets via prefix sum
	offsetsFB_[0] = 0;
	uint64_t runningFB = 0;
	for (uint32_t i = 0; i < N_F_; i++)
	{
		offsetsFB_[i] = runningFB;
		runningFB += static_cast<uint64_t>(tmpFB_[i].size());
	}
	offsetsFB_[N_F_] = runningFB;

	// allocate flat neighbor storage
	neighborsFB_.resize(runningFB);

	// copy data from temp into final data structure
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const uint64_t base = offsetsFB_[i];
				const uint32_t count = static_cast<uint32_t>(tmpFB_[i].size());

				if (count)
				{
					std::memcpy(neighborsFB_.data() + base, tmpFB_[i].data(),
						count * sizeof(uint32_t));
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_CompactVectors_CPU01::AddNbrFF(
	const uint32_t i, const uint32_t j)
{
	tmpFF_[i].push_back(j);
}

void NeighborLists_CompactVectors_CPU01::AddNbrFB(
	const uint32_t i, const uint32_t j)
{
	tmpFB_[i].push_back(j);
}

void NeighborLists_CompactVectors_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0) { tmpFF_[i].clear(); return; }

	// overwrite and resize combined
	tmpFF_[i].assign(js, js + count);
}

void NeighborLists_CompactVectors_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0) { tmpFB_[i].clear(); return; }

	// overwrite and resize combined
	tmpFB_[i].assign(js, js + count);
}

NeighborSpan NeighborLists_CompactVectors_CPU01::FF(
	const uint32_t i) const
{
	// start and end offsets of the neighbor list for fluid particle i
	const uint64_t s = offsetsFF_[i];
	const uint64_t e = offsetsFF_[i + 1];
	const uint32_t count = static_cast<uint32_t>(e - s);

	return NeighborSpan{ neighborsFF_.data() + s, count };
}

NeighborSpan NeighborLists_CompactVectors_CPU01::FB(
	const uint32_t i) const
{
	// start and end offsets of the neighbor list for boundary particle i
	const uint64_t s = offsetsFB_[i];
	const uint64_t e = offsetsFB_[i + 1];
	const uint32_t count = static_cast<uint32_t>(e - s);

	return NeighborSpan{ neighborsFB_.data() + s, count };
}

// -----------------------------------------------------------------------------
// --- [ compact vectors neighbor lists (ZigZag + LEB128 compressed) ] ---------
// -----------------------------------------------------------------------------

void NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B)
{
	N_F_ = N_F;
	N_B_ = N_B;

	// temporary 2D vector structure
	tmpCompressedFF_.resize(N_F_);
	tmpCompressedFB_.resize(N_F_);

	// reserve space for each temporary compressed neighbor list vector
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				if (tmpCompressedFF_[i].capacity() < N_COMPRESSED_NBR_SIZE_FF_)
				{
					tmpCompressedFF_[i].reserve(N_COMPRESSED_NBR_SIZE_FF_);
				}

				if (tmpCompressedFB_[i].capacity() < N_COMPRESSED_NBR_SIZE_FB_)
				{
					tmpCompressedFB_[i].reserve(N_COMPRESSED_NBR_SIZE_FB_);
				}

				// ensure lists are empty but keep capacity
				tmpCompressedFF_[i].clear();
				tmpCompressedFB_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);

	nbrCountFF_.assign(N_F_, 0);
	nbrCountFB_.assign(N_F_, 0);

	// compact vector structure (bytes)
	offsetsFF_.resize(N_F_ + 1, 0);
	offsetsFB_.resize(N_F_ + 1, 0);
	compressedNeighborsFF_.clear();
	compressedNeighborsFB_.clear();
}

void NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::ClearOne(
	const uint32_t i)
{
	nbrCountFF_[i] = 0;
	nbrCountFB_[i] = 0;

	tmpCompressedFF_[i].clear();
	tmpCompressedFB_[i].clear();
}

void NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::ClearAll()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				nbrCountFF_[i] = 0;
				nbrCountFB_[i] = 0;

				tmpCompressedFF_[i].clear();
				tmpCompressedFB_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::Finalize()
{
	if (N_F_ == 0) // nothing to do for empty sim
	{
		offsetsFF_.assign(1, 0);
		offsetsFB_.assign(1, 0);
		compressedNeighborsFF_.clear();
		compressedNeighborsFB_.clear();

		return;
	}

	// --- [ build compact vector for compressed FF ] --------------------------

	// build offsets via prefix sum of byte sizes
	offsetsFF_[0] = 0;
	uint64_t runningFF = 0;
	for (uint32_t i = 0; i < N_F_; i++)
	{
		offsetsFF_[i] = runningFF;
		runningFF += static_cast<uint64_t>(tmpCompressedFF_[i].size());
	}
	offsetsFF_[N_F_] = runningFF;

	// allocate flat neighbor storage
	compressedNeighborsFF_.resize(runningFF);

	// copy data from temp into final data structure
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const uint64_t base = offsetsFF_[i];
				const uint32_t byteCount = static_cast<uint32_t>(tmpCompressedFF_[i].size());

				if (byteCount)
				{
					std::memcpy(compressedNeighborsFF_.data() + base, 
						tmpCompressedFF_[i].data(), byteCount);
				}
			}
		},
		tbb::auto_partitioner{}
	);

	// --- [ build compact vector for compressed FB ] --------------------------

	// build offsets via prefix sum of byte sizes
	offsetsFB_[0] = 0;
	uint64_t runningFB = 0;
	for (uint32_t i = 0; i < N_F_; i++)
	{
		offsetsFB_[i] = runningFB;
		runningFB += static_cast<uint64_t>(tmpCompressedFB_[i].size());
	}
	offsetsFB_[N_F_] = runningFB;

	// allocate flat neighbor storage
	compressedNeighborsFB_.resize(runningFB);

	// copy data from temp into final data structure
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const uint64_t base = offsetsFB_[i];
				const uint32_t byteCount = static_cast<uint32_t>(tmpCompressedFB_[i].size());

				if (byteCount)
				{
					std::memcpy(compressedNeighborsFB_.data() + base, 
						tmpCompressedFB_[i].data(), byteCount);
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::AddNbrFF(
	const uint32_t i, const uint32_t j)
{
	SPDLOG_ERROR("AddNbrFF() -> not implemented for compressed neighbor lists");
}

void NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::AddNbrFB(
	const uint32_t i, const uint32_t j)
{
	SPDLOG_ERROR("AddNbrFB() -> not implemented for compressed neighbor lists");
}

void NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	nbrCountFF_[i] = count;
	if (count == 0) { tmpCompressedFF_[i].clear(); return; }

	EncodeNeighborList(js, count, tmpCompressedFF_[i]);
}

void NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	nbrCountFB_[i] = count;
	if (count == 0) { tmpCompressedFB_[i].clear(); return; }

	EncodeNeighborList(js, count, tmpCompressedFB_[i]);
}

NeighborSpan NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0) { return NeighborSpan{ nullptr, 0 }; }

	// start and end offsets of the compressed neighbor list for fluid particle i
	const uint64_t s = offsetsFF_[i];
	const uint64_t e = offsetsFF_[i + 1];
	const uint32_t byteCount = static_cast<uint32_t>(e - s);

	const uint8_t* dataPointer = compressedNeighborsFF_.data() + s;

	auto& scratch = scratchFF_.local();
	DecodeNeighborList(dataPointer, count, byteCount, scratch);

	// TODO: is it safe to use count or scratch.size() here?
	return NeighborSpan{ scratch.data(), count };
}

NeighborSpan NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0) { return NeighborSpan{ nullptr, 0 }; }

	// start and end offsets of the compressed neighbor list for boundary particle i
	const uint64_t s = offsetsFB_[i];
	const uint64_t e = offsetsFB_[i + 1];
	const uint32_t byteCount = static_cast<uint32_t>(e - s);

	const uint8_t* dataPointer = compressedNeighborsFB_.data() + s;

	auto& scratch = scratchFB_.local();
	DecodeNeighborList(dataPointer, count, byteCount, scratch);

	// TODO: is it safe to use count or scratch.size() here?
	return NeighborSpan{ scratch.data(), count };
}

inline uint32_t NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::ZigZag_Encode32(
	const int32_t v) noexcept
{
	// (arithmetic shift)
	return (static_cast<uint32_t>(v) << 1) ^ static_cast<uint32_t>(v >> 31);
}

inline int32_t NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::ZigZag_Decode32(
	const uint32_t v) noexcept
{
	// (arithmetic shift)
	return static_cast<int32_t>((v >> 1) ^ (0u - (v & 1u)));
}

inline void NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::PutVarU32(
	uint32_t v, uint8_t*& dst) noexcept
{
	// write 7 bits per byte, most significant bit of 1 means more bytes follow
	// (LEB128 encoding)
	while (v >= 0x80u)
	{
		*dst++ = static_cast<uint8_t>((v & 0x7Fu) | 0x80u);
		v >>= 7;
	}
	*dst++ = static_cast<uint8_t>(v);
}

inline uint32_t NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::GetVarU32(
	const uint8_t*& src, const uint8_t* end) noexcept
{
	uint32_t result = 0;
	uint32_t shift = 0;

	// read until continuation bit is 0
	while (src < end)
	{
		const uint32_t byte = *src++;
		result |= (byte & 0x7Fu) << shift;

		if ((byte & 0x80u) == 0) { break; }

		shift += 7;
	}
	return result;
}

inline uint32_t NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::VarU32_ByteCount(
	uint32_t v) noexcept
{
	// count how many bytes (1 to 5) are needed to encode v in LEB128
	uint32_t n = 1;
	while (v >= 0x80u) { v >>= 7; n++; }

	return n;
}

void NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::EncodeNeighborList(
	const uint32_t* js, uint32_t count, std::vector<uint8_t>& out)
{
	if (count == 0) { out.clear(); return; }

	// compute exact byte size of encoded neighbor list
	uint64_t tmp_totalBytes = 0;
	uint32_t prev = 0;
	for (uint32_t k = 0; k < count; k++)
	{
		const uint32_t curr = js[k];

		// delta can be negative if neighbor list js not sorted
		const int32_t delta = static_cast<int32_t>(curr) - static_cast<int32_t>(prev);
		const uint32_t zz = ZigZag_Encode32(delta);
		tmp_totalBytes += VarU32_ByteCount(zz);

		prev = curr;
	}

	// totalBytes fits in uint32 for realistic neighbor list sizes
	const uint32_t totalBytes = static_cast<uint32_t>(tmp_totalBytes);

	// encode neighbor list and write to output vector
	out.resize(totalBytes);
	uint8_t* dst = out.data();

	prev = 0;
	for (uint32_t k = 0; k < count; k++)
	{
		const uint32_t curr = js[k];
		const int32_t delta = static_cast<int32_t>(curr) - static_cast<int32_t>(prev);
		const uint32_t zz = ZigZag_Encode32(delta);

		PutVarU32(zz, dst);
		prev = curr;
	}

	#ifndef NDEBUG
	// dst should land exactly at end
	assert(static_cast<size_t>(dst - out.data()) == out.size());
	#endif
}

void NeighborLists_CompactVectors_Compressed_ZigZag_CPU01::DecodeNeighborList(
	const uint8_t* data, uint32_t count, uint32_t byteCount, std::vector<uint32_t>& out)
{
	out.resize(count);
	if (count == 0) { return; }

	const uint8_t* src = data;
	const uint8_t* end = data + byteCount;

	uint32_t prev = 0;
	for (uint32_t k = 0; k < count; k++)
	{
		const uint32_t zz = GetVarU32(src, end);
		const int32_t delta = ZigZag_Decode32(zz);

		// reconstruct current particle index
		const int64_t curr64 = static_cast<int64_t>(prev) + static_cast<int64_t>(delta);

		#ifndef NDEBUG
		// if this triggers, encoding assumptions (int32 delta) broke
		assert(curr64 >= 0 && curr64 <= static_cast<int64_t>(UINT32_MAX));
		#endif

		const uint32_t curr = static_cast<uint32_t>(curr64);
		out[k] = curr;
		prev = curr;
	}
}
