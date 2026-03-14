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
	// -----------------------------------------------------------------------------------------------
	// --- [ on-demand neighbor lists ] --------------------------------------------------------------
	// -----------------------------------------------------------------------------------------------

	class NeighborLists_OnDemand_CPU01 final
		: public I_NeighborLists
	{
	public: // ---------------------------------------------------------------------------------------

		explicit NeighborLists_OnDemand_CPU01(const I_NeighborSearchStrat& neighborSearchStrat)
			: neighborSearchStrat_{ &neighborSearchStrat } {}

		void SetNeighborSearchStrat(const I_NeighborSearchStrat& neighborSearchStrat)
		{
			neighborSearchStrat_ = &neighborSearchStrat;
		}

		void Resize(uint32_t N_F, uint32_t N_B, const SimParams& params) override;

		void ClearAll() override {}

		void Finalize_Count() override {}

		void Finalize_Fill() override {}

		void SetNbrCountFF(uint32_t i, uint32_t count) override {}
		void SetNbrCountFB(uint32_t i, uint32_t count) override {}

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override {}
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override {}

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

		uint32_t CountFF(uint32_t i) const override;
		uint32_t CountFB(uint32_t i) const override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(
			const SimParams& params) const override;

	private: // --------------------------------------------------------------------------------------

		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;
		const I_NeighborSearchStrat* neighborSearchStrat_ = nullptr;
	};

	// -----------------------------------------------------------------------------------------------
	// --- [ individual vectors neighbor lists ] -----------------------------------------------------
	// -----------------------------------------------------------------------------------------------

	// TODO: implement alternative using boost::small_vector ?
	class NeighborLists_IndividualVectors_CPU01 final
		: public I_NeighborLists
	{
	public: // ---------------------------------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B, const SimParams& params) override;

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

	private: // --------------------------------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// neighbor storage
		std::vector<std::vector<uint32_t, uint32_tbb_alloc>> nbrListFF_;
		std::vector<std::vector<uint32_t, uint32_tbb_alloc>> nbrListFB_;
	};

	// -----------------------------------------------------------------------------------------------
	// --- [ CSR vector neighbor lists, final concat ] -----------------------------------------------
	// -----------------------------------------------------------------------------------------------

	// TODO: implement alternative using boost::small_vector ?
	class NeighborLists_CSRVector_FinalConcat_CPU01 final
		: public I_NeighborLists
	{
	public: // ---------------------------------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B, const SimParams& params) override;

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

	private: // --------------------------------------------------------------------------------------
		
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

	// -----------------------------------------------------------------------------------------------
	// --- [ CSR vector neighbor lists, double query ] -----------------------------------------------
	// -----------------------------------------------------------------------------------------------

	// TODO: implement alternative using boost::small_vector ?
	class NeighborLists_CSRVector_DoubleQuery_CPU01 final
		: public I_NeighborLists
	{
	public: // ---------------------------------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B, const SimParams& params) override;

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

	private: // --------------------------------------------------------------------------------------

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
		const uint32_t extraAllocCapacityPct_ = 4; // TODO: fine-tune
	};

	// -----------------------------------------------------------------------------------------------
	// --- [ individual vectors neighbor lists, adaptive compression ] -------------------------------
	// -----------------------------------------------------------------------------------------------

	class NeighborLists_IndividualVectors_AdaptCompress_CPU01 final
		: public I_NeighborLists
	{
	public: // ---------------------------------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B, const SimParams& params) override;

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

	private: // --------------------------------------------------------------------------------------

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

	// -----------------------------------------------------------------------------------------------
	// --- [ individual vectors neighbor lists, VByteDerived compression ] ---------------------------
	// -----------------------------------------------------------------------------------------------

	class NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01 final
		: public I_NeighborLists
	{
	public: // ---------------------------------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B, const SimParams& params) override;

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

	private: // --------------------------------------------------------------------------------------

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

} // namespace Sim
