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
// --- [ compact vectors neighbor lists (ULEB128 compression of deltas) ] ------
// -----------------------------------------------------------------------------

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::Resize(
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

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::ClearOne(
	const uint32_t i)
{
	nbrCountFF_[i] = 0;
	nbrCountFB_[i] = 0;

	tmpCompressedFF_[i].clear();
	tmpCompressedFB_[i].clear();
}

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::ClearAll()
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

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::Finalize()
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

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::AddNbrFF(
	const uint32_t i, const uint32_t j)
{
	SPDLOG_ERROR("AddNbrFF() -> not implemented for compressed neighbor lists");
}

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::AddNbrFB(
	const uint32_t i, const uint32_t j)
{
	SPDLOG_ERROR("AddNbrFB() -> not implemented for compressed neighbor lists");
}

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0)
	{
		nbrCountFF_[i] = 0;
		tmpCompressedFF_[i].clear();
		return;
	}

	// copy to thread-local scratch and sort to avoid encoding of negative deltas
	auto& scratch = sortScratchFF_.local();
	if (scratch.capacity() < count) { scratch.reserve(count); }
	scratch.assign(js, js + count);

	std::sort(scratch.begin(), scratch.end());

	const uint32_t newCount = static_cast<uint32_t>(scratch.size());
	nbrCountFF_[i] = newCount;

	EncodeNeighborList_ULEB128(scratch.data(), newCount, tmpCompressedFF_[i]);
}

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0)
	{
		nbrCountFB_[i] = 0;
		tmpCompressedFB_[i].clear();
		return;
	}

	// copy to thread-local scratch and sort to avoid encoding of negative deltas
	auto& scratch = sortScratchFB_.local();
	if (scratch.capacity() < count) { scratch.reserve(count); }
	scratch.assign(js, js + count);

	std::sort(scratch.begin(), scratch.end());

	const uint32_t newCount = static_cast<uint32_t>(scratch.size());
	nbrCountFB_[i] = newCount;

	EncodeNeighborList_ULEB128(scratch.data(), newCount, tmpCompressedFB_[i]);
}

NeighborSpan NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0) { return NeighborSpan{ nullptr, 0 }; }

	// start and end offsets of the compressed neighbor list for fluid particle i
	const uint64_t s = offsetsFF_[i];
	const uint64_t e = offsetsFF_[i + 1];
	const uint32_t byteCount = static_cast<uint32_t>(e - s);

	// // decode into provided per-thread scratch vector
	auto& scratch = nbrScratchFF_.local();
	if (scratch.capacity() < N_expctMaxNbrCountF) // reserve "once" on first call
	{
		scratch.reserve(N_expctMaxNbrCountF);
	}
	const uint8_t* data = compressedNeighborsFF_.data() + s;
	DecodeNeighborList_ULEB128(data, count, byteCount, scratch);

	return NeighborSpan{ scratch.data(), count };
}

NeighborSpan NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0) { return NeighborSpan{ nullptr, 0 }; }

	// start and end offsets of the compressed neighbor list for boundary particle i
	const uint64_t s = offsetsFB_[i];
	const uint64_t e = offsetsFB_[i + 1];
	const uint32_t byteCount = static_cast<uint32_t>(e - s);

	// // decode into provided per-thread scratch vector
	auto& scratch = nbrScratchFB_.local();
	if (scratch.capacity() < N_expctMaxNbrCountB) // reserve "once" on first call
	{
		scratch.reserve(N_expctMaxNbrCountB);
	}
	const uint8_t* data = compressedNeighborsFB_.data() + s;
	DecodeNeighborList_ULEB128(data, count, byteCount, scratch);

	return NeighborSpan{ scratch.data(), count };
}

inline void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::PutU32(
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

inline uint32_t NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::GetU32(
	const uint8_t*& src, const uint8_t* end) noexcept
{
	uint32_t result = 0;
	uint32_t shift = 0;

	// read until continuation bit is 0 (or max 5 bytes)
	for (int i = 0; i < 5 && src < end; i++)
	{
		uint32_t byte = *src++;

		result |= (byte & 0x7Fu) << shift;
		if ((byte & 0x80u) == 0) break;

		shift += 7;
	}

	return result;
}

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::EncodeNeighborList_ULEB128(
	const uint32_t* js, const uint32_t count, std::vector<uint8_t>& out)
{
	if (count == 0) { out.clear(); return; }

	// max number of bytes for LEB128 encoding of uint32_t is 5
	const uint32_t maxBytes = count * 5u;

	out.resize(maxBytes);
	uint8_t* dst = out.data();

	// encode deltas with LEB128 to compress indices
	uint32_t prev = 0;
	for (uint32_t k = 0; k < count; k++)
	{
		const uint32_t curr = js[k];
		const uint32_t delta = curr - prev;
		PutU32(delta, dst);
		prev = curr;
	}

	// shrink to actual bytes written
	out.resize(static_cast<size_t>(dst - out.data()));
}

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::DecodeNeighborList_ULEB128(
	const uint8_t* data, const uint32_t count, uint32_t byteCount, std::vector<uint32_t>& out)
{
	out.resize(count);

	const uint8_t* src = data;
	const uint8_t* end = data + byteCount;

	// decode deltas with LEB128 to reconstruct original indices
	uint32_t prev = 0;
	for (uint32_t k = 0; k < count; k++)
	{
		const uint32_t delta = GetU32(src, end);
		const uint32_t curr = prev + delta;

		out[k] = curr;
		prev = curr;
	}
}

// -----------------------------------------------------------------------------
// --- [ compact vectors neighbor lists (Stream VByte delta compression) ] -----
// -----------------------------------------------------------------------------

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::Resize(
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

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::ClearOne(
	const uint32_t i)
{
	nbrCountFF_[i] = 0;
	nbrCountFB_[i] = 0;

	tmpCompressedFF_[i].clear();
	tmpCompressedFB_[i].clear();
}

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::ClearAll()
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

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::Finalize()
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

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::AddNbrFF(
	const uint32_t i, const uint32_t j)
{
	SPDLOG_ERROR("AddNbrFF() -> not implemented for compressed neighbor lists");
}

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::AddNbrFB(
	const uint32_t i, const uint32_t j)
{
	SPDLOG_ERROR("AddNbrFB() -> not implemented for compressed neighbor lists");
}

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0)
	{
		nbrCountFF_[i] = 0;
		tmpCompressedFF_[i].clear();
		return;
	}

	// copy to thread-local scratch and sort to avoid encoding of negative deltas
	auto& scratch = sortScratchFF_.local();
	if (scratch.capacity() < count) { scratch.reserve(count); }
	scratch.assign(js, js + count);

	std::sort(scratch.begin(), scratch.end());

	const uint32_t newCount = static_cast<uint32_t>(scratch.size());
	nbrCountFF_[i] = newCount;

	EncodeNeighborList_StreamVByteDelta(scratch.data(), newCount, 
		tmpCompressedFF_[i]);
}

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0)
	{
		nbrCountFB_[i] = 0;
		tmpCompressedFB_[i].clear();
		return;
	}

	// copy to thread-local scratch and sort to avoid encoding of negative deltas
	auto& scratch = sortScratchFB_.local();
	if (scratch.capacity() < count) { scratch.reserve(count); }
	scratch.assign(js, js + count);

	std::sort(scratch.begin(), scratch.end());

	const uint32_t newCount = static_cast<uint32_t>(scratch.size());
	nbrCountFB_[i] = newCount;

	EncodeNeighborList_StreamVByteDelta(scratch.data(), newCount, 
		tmpCompressedFB_[i]);
}

NeighborSpan NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0) { return NeighborSpan{ nullptr, 0 }; }

	// start and end offsets of the compressed neighbor list for fluid particle i
	const uint64_t s = offsetsFF_[i];
	const uint64_t e = offsetsFF_[i + 1];
	const uint32_t byteCount = static_cast<uint32_t>(e - s);

	// // decode into provided per-thread scratch vector
	auto& scratch = nbrScratchFF_.local();
	if (scratch.capacity() < N_expctMaxNbrCountF) // reserve "once" on first call
	{
		scratch.reserve(N_expctMaxNbrCountF);
	}
	const uint8_t* data = compressedNeighborsFF_.data() + s;
	DecodeNeighborList_StreamVByteDelta(data, count, byteCount, scratch);

	return NeighborSpan{ scratch.data(), count };
}

NeighborSpan NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0) { return NeighborSpan{ nullptr, 0 }; }

	// start and end offsets of the compressed neighbor list for boundary particle i
	const uint64_t s = offsetsFB_[i];
	const uint64_t e = offsetsFB_[i + 1];
	const uint32_t byteCount = static_cast<uint32_t>(e - s);

	// // decode into provided per-thread scratch vector
	auto& scratch = nbrScratchFB_.local();
	if (scratch.capacity() < N_expctMaxNbrCountB) // reserve "once" on first call
	{
		scratch.reserve(N_expctMaxNbrCountB);
	}
	const uint8_t* data = compressedNeighborsFB_.data() + s;
	DecodeNeighborList_StreamVByteDelta(data, count, byteCount, scratch);

	return NeighborSpan{ scratch.data(), count };
}

inline void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::PutU32(
	uint32_t v, uint8_t*& dst) noexcept
{
	*dst++ = static_cast<uint8_t>(v);
	*dst++ = static_cast<uint8_t>(v >> 8);
	*dst++ = static_cast<uint8_t>(v >> 16);
	*dst++ = static_cast<uint8_t>(v >> 24);
}

inline uint32_t NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::GetU32(
	const uint8_t*& src, const uint8_t* end) noexcept
{
	if ((end - src) < 4)
	{
		src = end;
		return 0;
	}

	const uint32_t b0 = src[0];
	const uint32_t b1 = src[1];
	const uint32_t b2 = src[2];
	const uint32_t b3 = src[3];
	src += 4;

	return (b0) | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static inline uint32_t ControlByteCount_ForNeighborCount(uint32_t count) noexcept
{
	// first neighbor is stored uncompressed, remaining (count-1) values use 2 control bits each
	if (count <= 1) return 0;
	const uint32_t deltas = count - 1;
	return (deltas + 3u) / 4u; // 4 deltas per control byte
}

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::EncodeNeighborList_StreamVByteDelta(
	const uint32_t* js, const uint32_t count, std::vector<uint8_t>& out)
{
	if (count == 0) { out.clear(); return; }

	const uint32_t ctrlBytes = ControlByteCount_ForNeighborCount(count);
	const uint32_t deltas = (count > 0) ? (count - 1) : 0;

	// worst case size with 4 bytes for n1, ctrlBytes for control bytes, 4 bytes per delta
	const uint32_t maxBytes = static_cast<uint32_t>(4ull + ctrlBytes + 4ull * deltas);

	out.resize(maxBytes);
	uint8_t* base = out.data();
	uint8_t* dst = base;

	// store first neighbor uncompressed
	PutU32(js[0], dst);

	// reserve and zero control byte stream
	uint8_t* ctrl = dst;
	if (ctrlBytes)
	{
		std::memset(ctrl, 0, ctrlBytes);
	}
	dst += ctrlBytes;

	// dst now points to the beginning of the data byte stream
	uint8_t* data = dst;

	// encode deltas as reduced deltas with control bits + data bytes
	uint32_t prev = js[0];
	for (uint32_t k = 1; k < count; k++)
	{
		const uint32_t curr = js[k];

		// reduced delta mapping: rDelta = curr - prev - 1 (>= 0)
		const uint32_t diff = curr - prev;
		const uint32_t rDelta = diff - 1u;

		// choose control code + emit bytes
		uint8_t code = 0;
		if (rDelta == 0u)
		{
			code = 0; // 0b00
		}
		else if (rDelta == 1u)
		{
			code = 1; // 0b01
		}
		else if (rDelta < 256u)
		{
			code = 2; // 0b10
			*data++ = static_cast<uint8_t>(rDelta);
		}
		else
		{
			code = 3; // 0b11
			PutU32(rDelta, data);
		}

		const uint32_t d = k - 1; // rDelta index [0..count-2]
		const uint32_t cIdx = d >> 2; // 4 rDeltas per control byte
		const uint32_t shift = (d & 3u) * 2u; // 2 bits per rDelta
		ctrl[cIdx] = static_cast<uint8_t>(ctrl[cIdx] | (code << shift));

		prev = curr;
	}

	// shrink to actual bytes written
	out.resize(static_cast<size_t>(data - base));
}

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::DecodeNeighborList_StreamVByteDelta(
	const uint8_t* data, const uint32_t count, uint32_t byteCount, std::vector<uint32_t>& out)
{
	out.resize(count);
	if (count == 0) { return; }

	const uint8_t* src = data;
	const uint8_t* end = data + byteCount;

	const uint32_t ctrlBytes = ControlByteCount_ForNeighborCount(count);

	// read first neighbor uncompressed
	const uint32_t n1 = GetU32(src, end);
	out[0] = n1;

	// src now points to the data byte stream
	const uint8_t* ctrl = src;
	src += ctrlBytes;

	uint32_t prev = n1;

	// decode reduced deltas from control stream + data stream
	for (uint32_t k = 1; k < count; k++)
	{
		const uint32_t d = k - 1; // rDelta index [0..count-2]
		const uint32_t cIdx = d >> 2; // 4 rDeltas per control byte
		const uint32_t shift = (d & 3u) * 2u; // 2 bits per rDelta
		const uint32_t code = (ctrlBytes ? ((ctrl[cIdx] >> shift) & 0x3u) : 0u);

		uint32_t rDelta = 0;
		if (code == 0u)
		{
			rDelta = 0u;
		}
		else if (code == 1u)
		{
			rDelta = 1u;
		}
		else if (code == 2u)
		{
			if (src >= end) { rDelta = 0u; }
			else { rDelta = static_cast<uint32_t>(*src++); }
		}
		else // code == 3
		{
			rDelta = GetU32(src, end);
		}

		// reconstruct particle index
		const uint32_t curr = prev + rDelta + 1u;

		out[k] = curr;
		prev = curr;
	}
}
