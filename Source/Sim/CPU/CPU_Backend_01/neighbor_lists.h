// neighbor_lists.h
//
// TODO: comment

#pragma once

// external
#include <atomic>
#include <cstdint>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <tbb/enumerable_thread_specific.h>

// internal
#include "Source/Sim/Core/backend_sim.h"



namespace Sim
{
	// -------------------------------------------------------------------------
	// --- [ small buffers neighbor lists ] ------------------------------------
	// -------------------------------------------------------------------------

	class NeighborLists_SmallBuffers_CPU01 final : public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearOne(uint32_t i) override;

		void ClearAll() override;

		void Finalize() override;

		void AddNbrFF(uint32_t i, uint32_t j) override;
		void AddNbrFB(uint32_t i, uint32_t j) override;

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// neighbor storage
		std::vector<boost::container::small_vector<uint32_t, N_expctNbrCountF>> neighborsFF_;
		std::vector<boost::container::small_vector<uint32_t, N_expctNbrCountB>> neighborsFB_;
	};

	// -------------------------------------------------------------------------
	// --- [ fixed arrays neighbor lists ] -------------------------------------
	// -------------------------------------------------------------------------

	class NeighborLists_FixedArrays_CPU01 final : public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearOne(uint32_t i) override;

		void ClearAll() override;

		void Finalize() override;

		void AddNbrFF(uint32_t i, uint32_t j) override;
		void AddNbrFB(uint32_t i, uint32_t j) override;

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// max neighbor counts
		static constexpr uint32_t N_MAX_FF_ = 64;
		static constexpr uint32_t N_MAX_FB_ = 32;

		// neighbor storage
		std::vector<uint32_t> nbrCountFF_;
		std::vector<uint32_t> nbrCountFB_;
		std::vector<uint32_t> neighborsFF_; // N_F_ * N_MAX_FF_
		std::vector<uint32_t> neighborsFB_; // N_F_ * N_MAX_FB_
	};

	// -------------------------------------------------------------------------
	// --- [ compact vectors neighbor lists ] ----------------------------------
	// -------------------------------------------------------------------------

	class NeighborLists_CompactVectors_CPU01 final : public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearOne(uint32_t i) override;

		void ClearAll() override;

		void Finalize() override;

		void AddNbrFF(uint32_t i, uint32_t j) override;
		void AddNbrFB(uint32_t i, uint32_t j) override;

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// temp neighbor lists during construction
		std::vector<boost::container::small_vector<uint32_t, N_expctNbrCountF>> tmpFF_;
		std::vector<boost::container::small_vector<uint32_t, N_expctNbrCountB>> tmpFB_;

		// neighbor storage
		std::vector<uint64_t> offsetsFF_;
		std::vector<uint64_t> offsetsFB_;
		std::vector<uint32_t> neighborsFF_;
		std::vector<uint32_t> neighborsFB_;
	};

	// -------------------------------------------------------------------------
	// --- [ compact vectors neighbor lists (ULEB128 compression of deltas) ] --
	// -------------------------------------------------------------------------

	class NeighborLists_CompactVectors_Compressed_ZigZag_CPU01 final
		: public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearOne(uint32_t i) override;

		void ClearAll() override;

		void Finalize() override;

		void AddNbrFF(uint32_t i, uint32_t j) override;
		void AddNbrFB(uint32_t i, uint32_t j) override;

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

		uint32_t CountFF(uint32_t i) const override { return nbrCountFF_[i]; }
		uint32_t CountFB(uint32_t i) const override { return nbrCountFB_[i]; }

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// size guess to reserve memory for temporary compressed neighbor lists
		static constexpr uint32_t N_COMPRESSED_NBR_SIZE_FF_ = N_expctNbrCountF * 2; // bytes
		static constexpr uint32_t N_COMPRESSED_NBR_SIZE_FB_ = N_expctNbrCountB * 2; // bytes

		// temp compressed neighbor lists during construction (bytes)
		// TODO: reserve space for each vector?
		std::vector<std::vector<uint8_t>> tmpCompressedFF_;
		std::vector<std::vector<uint8_t>> tmpCompressedFB_;

		// compressed neighbor storage
		std::vector<uint32_t> nbrCountFF_;
		std::vector<uint32_t> nbrCountFB_;
		std::vector<uint64_t> offsetsFF_; // byte offsets into compressed FF data
		std::vector<uint64_t> offsetsFB_; // byte offsets into compressed FF data
		std::vector<uint8_t> compressedNeighborsFF_; // concatenated bytes
		std::vector<uint8_t> compressedNeighborsFB_; // concatenated bytes

		// thread-local decode scratch for FF()/FB() to return neighbors as uint32_t
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> nbrScratchFF_;
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> nbrScratchFB_;

		// thread-local sorting scratch for EncodeNeighborList() to only have positive deltas
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> sortScratchFF_;
		mutable tbb::enumerable_thread_specific<std::vector<uint32_t>> sortScratchFB_;

		// helpers
		static inline void PutVarU32(uint32_t v, uint8_t*& dst) noexcept;
		static inline uint32_t GetVarU32(const uint8_t*& src, const uint8_t* end) noexcept;

		static void EncodeNeighborList(const uint32_t* js, uint32_t count,
			std::vector<uint8_t>& out);
		static void DecodeNeighborList(const uint8_t* data, uint32_t count, uint32_t byteCount,
			std::vector<uint32_t>& out);
	};

} // namespace Sim
