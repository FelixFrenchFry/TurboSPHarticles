// neighbor_lists.h
//
// TODO: comment

#pragma once

// external
#include <vector>
#include <boost/container/small_vector.hpp>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/scalable_allocator.h>

// internal
#include "Source/Sim/Core/backend_sim.h"



using uint8_tbb_alloc = tbb::scalable_allocator<uint8_t>;
using uint32_tbb_alloc = tbb::scalable_allocator<uint32_t>;

namespace Sim
{
	// -------------------------------------------------------------------------
	// --- [ individual vectors neighbor lists ] -------------------------------
	// -------------------------------------------------------------------------

	// TODO: implement alternative using boost::small_vector ?
	class NeighborLists_IndividualVectors_CPU01 final
		: public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearAll() override;

		void Finalize_Count() override {}

		void Finalize_Fill() override {}

		void SetNbrCountFF(uint32_t i, uint32_t count) override {}
		void SetNbrCountFB(uint32_t i, uint32_t count) override {}

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(
			const SimParams& params) const override;

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// neighbor storage
		std::vector<std::vector<uint32_t, uint32_tbb_alloc>> nbrListFF_;
		std::vector<std::vector<uint32_t, uint32_tbb_alloc>> nbrListFB_;
	};

	// -------------------------------------------------------------------------
	// --- [ CSR vector neighbor lists, final concat ] -------------------------
	// -------------------------------------------------------------------------

	// TODO: implement alternative using boost::small_vector ?
	class NeighborLists_CSRVector_FinalConcat_CPU01 final
		: public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearAll() override;

		void Finalize_Count() override {}

		void Finalize_Fill() override;

		void SetNbrCountFF(uint32_t i, uint32_t count) override {}
		void SetNbrCountFB(uint32_t i, uint32_t count) override {}

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(
			const SimParams& params) const override;

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// individual neighbor lists used for the construction, not the query
		std::vector<std::vector<uint32_t, uint32_tbb_alloc>> nbrTempFF_;
		std::vector<std::vector<uint32_t, uint32_tbb_alloc>> nbrTempFB_;

		// CSR neighbor storage
		std::vector<uint64_t> nbrOffsetsFF_;
		std::vector<uint64_t> nbrOffsetsFB_;
		std::vector<uint32_t> nbrListFF_;
		std::vector<uint32_t> nbrListFB_;

		// misc
		const uint32_t extraAllocCapacityPct_ = 5;
	};

	// -------------------------------------------------------------------------
	// --- [ CSR vector neighbor lists, double query ] -------------------------
	// -------------------------------------------------------------------------

	// TODO: implement alternative using boost::small_vector ?
	class NeighborLists_CSRVector_DoubleQuery_CPU01 final
		: public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearAll() override;

		void Finalize_Count() override;

		void Finalize_Fill() override {}

		void SetNbrCountFF(uint32_t i, uint32_t count) override;
		void SetNbrCountFB(uint32_t i, uint32_t count) override;

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(
			const SimParams& params) const override;

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// CSR neighbor storage
		// (offset vectors act as count vectors for the count-query)
		std::vector<uint64_t> nbrOffsetsFF_;
		std::vector<uint64_t> nbrOffsetsFB_;
		std::vector<uint32_t> nbrListFF_;
		std::vector<uint32_t> nbrListFB_;

		// misc
		const uint32_t extraAllocCapacityPct_ = 5;
	};

	// -------------------------------------------------------------------------
	// --- [ individual vectors neighbor lists, adaptive compression ] ---------
	// -------------------------------------------------------------------------

	class NeighborLists_IndividualVectors_AdaptCompress_CPU01 final
		: public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearAll() override;

		void Finalize_Count() override {}

		void Finalize_Fill() override {}

		void SetNbrCountFF(uint32_t i, uint32_t count) override {}
		void SetNbrCountFB(uint32_t i, uint32_t count) override {}

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(
			const SimParams& params) const override;

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// compressed neighbor storage
		std::vector<uint32_t> nbrCountFF_;
		std::vector<uint32_t> nbrCountFB_;
		std::vector<std::vector<uint8_t, uint8_tbb_alloc>> nbrListCompressedFF_;
		std::vector<std::vector<uint8_t, uint8_tbb_alloc>> nbrListCompressedFB_;

		// thread-local decode scratch for FF()/FB() to return neighbors as uint32_t
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> nbrScratchFF_;
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> nbrScratchFB_;

		// thread-local sorting scratch for EncodeNeighborList() to only have positive deltas
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> sortScratchFF_;
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> sortScratchFB_;

		// misc
		const uint32_t expctBytesPerNbrCeil_ = 2;

		// helpers
		static uint8_t ChooseWidthCode(uint32_t maxDelta) noexcept;
		static uint32_t GetNumWidthBytes(uint8_t widthCode) noexcept;
		static inline uint16_t ReadU16LE(const uint8_t* src) noexcept;
		static inline uint32_t ReadU32LE(const uint8_t* src) noexcept;
		static inline void WriteU16LE(uint8_t* dst, uint16_t v) noexcept;
		static inline void WriteU32LE(uint8_t* dst, uint32_t v) noexcept;
		static void DecodeNeighborList(const std::vector<uint8_t, uint8_tbb_alloc>& src,
			uint32_t count, std::vector<uint32_t>& out);
		static void EncodeNeighborList(std::vector<uint8_t, uint8_tbb_alloc>& dst,
			const uint32_t* js, uint32_t count, std::vector<uint32_t>& scratch);
	};

	// -------------------------------------------------------------------------
	// --- [ individual vectors neighbor lists, VByteDerived compression ] -----
	// -------------------------------------------------------------------------

	class NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01 final
		: public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearAll() override;

		void Finalize_Count() override {}

		void Finalize_Fill() override {}

		void SetNbrCountFF(uint32_t i, uint32_t count) override {}
		void SetNbrCountFB(uint32_t i, uint32_t count) override {}

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(
			const SimParams& params) const override;

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// compressed neighbor storage
		std::vector<uint32_t> nbrCountFF_;
		std::vector<uint32_t> nbrCountFB_;
		std::vector<std::vector<uint8_t, uint8_tbb_alloc>> nbrListCompressedFF_;
		std::vector<std::vector<uint8_t, uint8_tbb_alloc>> nbrListCompressedFB_;

		// thread-local decode scratch for FF()/FB() to return neighbors as uint32_t
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> nbrScratchFF_;
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> nbrScratchFB_;

		// thread-local sorting scratch for EncodeNeighborList() to only have positive deltas
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> sortScratchFF_;
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> sortScratchFB_;

		// misc
		const uint32_t expctBytesPerNbrCeil_ = 1;

		// helpers
		static uint32_t ControlBytesFor(uint32_t count) noexcept;
		static uint8_t DeltaMask(uint32_t delta) noexcept;
		static inline uint32_t ReadU32LE(const uint8_t* src) noexcept;
		static inline void WriteU32LE(uint8_t* dst, uint32_t v) noexcept;
		static void DecodeNeighborList(const std::vector<uint8_t, uint8_tbb_alloc>& src,
			uint32_t count, std::vector<uint32_t>& out);
		static void EncodeNeighborList(std::vector<uint8_t, uint8_tbb_alloc>& dst,
			const uint32_t* js, uint32_t count, std::vector<uint32_t>& scratch);
	};

	// -------------------------------------------------------------------------
	// --- [ CSR vector neighbor lists, final concat, ULEB128 compression ] ----
	// -------------------------------------------------------------------------

	/*
	class NeighborLists_CSRVector_FinalConcat_ULEB128Compress_CPU01 final
		: public I_NeighborLists
	{

	};

	// -------------------------------------------------------------------------
	// --- [ CSR vector neighbor lists, final concat, VByteDerived compression ] 
	// -------------------------------------------------------------------------

	class NeighborLists_CSRVector_FinalConcat_VByteDerivedCompress_CPU01 final
		: public I_NeighborLists
	{

	};

	// -------------------------------------------------------------------------
	// --- [ compact vectors neighbor lists (ULEB128 compression of deltas) ] --
	// -------------------------------------------------------------------------

	class NeighborLists_CompactVectors_Compressed_ULEB128_CPU01 final
		: public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearAll() override;

		void Finalize() override;

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

		uint32_t CountFF(uint32_t i) const override { return nbrCountFF_[i]; }
		uint32_t CountFB(uint32_t i) const override { return nbrCountFB_[i]; }

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// temp compressed neighbor lists during construction (bytes)
		std::vector<std::vector<uint8_t>> tmpCompressedFF_;
		std::vector<std::vector<uint8_t>> tmpCompressedFB_;

		// compressed neighbor storage
		std::vector<uint32_t> nbrCountFF_;
		std::vector<uint32_t> nbrCountFB_;
		std::vector<uint64_t> offsetsFF_; // byte offsets into compressed FF data
		std::vector<uint64_t> offsetsFB_; // byte offsets into compressed FF data
		// TODO: just keep them in the tmp vectors instead of concatenating?
		std::vector<uint8_t> compressedNeighborsFF_; // concatenated bytes
		std::vector<uint8_t> compressedNeighborsFB_; // concatenated bytes

		// thread-local decode scratch for FF()/FB() to return neighbors as uint32_t
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> nbrScratchFF_;
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> nbrScratchFB_;

		// thread-local sorting scratch for EncodeNeighborList() to only have positive deltas
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> sortScratchFF_;
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> sortScratchFB_;

		// helpers
		static inline void PutU32(uint32_t v, uint8_t*& dst) noexcept;
		static inline uint32_t GetU32(const uint8_t*& src, const uint8_t* end) noexcept;

		static void EncodeNeighborList_ULEB128(const uint32_t* js, uint32_t count,
			std::vector<uint8_t>& out);
		static void DecodeNeighborList_ULEB128(const uint8_t* data, uint32_t count,
			uint32_t byteCount, std::vector<uint32_t>& out);

		// size guess to reserve memory for temporary compressed neighbor lists
		static constexpr uint32_t N_COMPRESSED_NBR_SIZE_FF_ = N_expctNbrCountF * 2; // bytes
		static constexpr uint32_t N_COMPRESSED_NBR_SIZE_FB_ = N_expctNbrCountB * 2; // bytes
	};

	// -------------------------------------------------------------------------
	// --- [ compact vectors neighbor lists (Stream VByte delta compression) ] -
	// -------------------------------------------------------------------------

	class NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01 final
		: public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearAll() override;

		void Finalize() override;

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

		uint32_t CountFF(uint32_t i) const override { return nbrCountFF_[i]; }
		uint32_t CountFB(uint32_t i) const override { return nbrCountFB_[i]; }

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(const SimParams& params) const override;

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// temp compressed neighbor lists during construction (bytes)
		std::vector<std::vector<uint8_t>> tmpCompressedFF_;
		std::vector<std::vector<uint8_t>> tmpCompressedFB_;

		// compressed neighbor storage
		std::vector<uint32_t> nbrCountFF_;
		std::vector<uint32_t> nbrCountFB_;
		std::vector<uint64_t> offsetsFF_; // byte offsets into compressed FF data
		std::vector<uint64_t> offsetsFB_; // byte offsets into compressed FF data
		// TODO: just keep them in the tmp vectors instead of concatenating?
		std::vector<uint8_t> compressedNeighborsFF_; // concatenated bytes
		std::vector<uint8_t> compressedNeighborsFB_; // concatenated bytes

		// thread-local decode scratch for FF()/FB() to return neighbors as uint32_t
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> nbrScratchFF_;
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> nbrScratchFB_;

		// thread-local sorting scratch for EncodeNeighborList() to only have positive deltas
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> sortScratchFF_;
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> sortScratchFB_;

		// helpers
		static inline void PutU32(uint32_t v, uint8_t*& dst) noexcept;
		static inline uint32_t GetU32(const uint8_t*& src, const uint8_t* end) noexcept;

		static void EncodeNeighborList_StreamVByteDelta(const uint32_t* js, uint32_t count,
			std::vector<uint8_t>& out);
		static void DecodeNeighborList_StreamVByteDelta(const uint8_t* data, uint32_t count,
			uint32_t byteCount, std::vector<uint32_t>& out);

		// size guess to reserve memory for temporary compressed neighbor lists
		static constexpr uint32_t N_COMPRESSED_NBR_SIZE_FF_ = N_expctNbrCountF * 1; // bytes
		static constexpr uint32_t N_COMPRESSED_NBR_SIZE_FB_ = N_expctNbrCountB * 1; // bytes

		// control bits for reduced deltas between particle indices: rDelta = curr - prev - 1
		// (2 bits per rDelta, 4 deltas per control byte)
		// 00 -> rDelta = 0
		// 01 -> rDelta = 1
		// 10 -> rDelta in [2, 255]
		// 11 -> rDelta in [256, 2^32-1]
		static constexpr uint8_t cBits_d0 = 0b00;
		static constexpr uint8_t cBits_d1 = 0b01;
		static constexpr uint8_t cBits_u8 = 0b10;
		static constexpr uint8_t cBits_u32 = 0b11;
	};
	*/

} // namespace Sim
