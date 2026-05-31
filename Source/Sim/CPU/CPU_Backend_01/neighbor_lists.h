// neighbor_lists.h

#pragma once

// external
#include <cassert>
#include <boost/container/small_vector.hpp>
#include <vector>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/scalable_allocator.h>

// internal
#include "Source/Sim/Core/backend_sim.h"
#include "Source/Sim/CPU/simd8.h"



using uint32_tbb_alloc = tbb::scalable_allocator<uint32_t>;
// TODO: make these used the params based on heuristic expected values?
static constexpr uint32_t N_perScratchInlineFF = 64;
static constexpr uint32_t N_perScratchInlineFB = 48;

namespace Sim
{
	enum class SELL8ChunkFormat : uint8_t
	{
		Explicit32,
		Delta8,
		Delta16,
		Abs32
	};

	// read-view used in the neighbor retrieval (e.g. solver kernels)
	struct ChunkedSELL8NeighborSpan_Read
	{
		const uint32_t* data32 = nullptr;
		const uint16_t* data16 = nullptr;
		const uint8_t* data8 = nullptr;
		const uint32_t* counts = nullptr;
		uint32_t baseIdx = 0;
		uint32_t chunk = 0;
		uint32_t begin = 0;
		uint32_t end = 0;
		uint32_t pitch = 0;
		SELL8ChunkFormat format = SELL8ChunkFormat::Explicit32;

		uint32_t ReadNeighborIdx(uint32_t lane, uint32_t k) const noexcept
		{
			const uint64_t linearIdx = static_cast<uint64_t>(k) * CPU::N_SIMD_WIDTH + lane;

			if (format == SELL8ChunkFormat::Delta8)
			{
				return baseIdx + static_cast<uint32_t>(data8[linearIdx]);
			}
			if (format == SELL8ChunkFormat::Delta16)
			{
				return baseIdx + static_cast<uint32_t>(data16[linearIdx]);
			}

			return data32[linearIdx];
		}

		CPU::UInt32x8 LoadNeighborIdx32x8(uint32_t k) const noexcept
		{
			const uint64_t linearIdx = static_cast<uint64_t>(k) * CPU::N_SIMD_WIDTH;

			if (format == SELL8ChunkFormat::Delta8)
			{
				return CPU::LoadUInt8AsUInt32x8(data8 + linearIdx)
					+ CPU::SetUInt32x8(baseIdx);
			}
			if (format == SELL8ChunkFormat::Delta16)
			{
				return CPU::LoadUInt16AsUInt32x8(data16 + linearIdx)
					+ CPU::SetUInt32x8(baseIdx);
			}

			return CPU::LoadUInt32x8(data32 + linearIdx);
		}
	};

	// write-view used in the neighbor search (e.g. fill-queries)
	struct ChunkedSELL8NeighborSpan_Write
	{
		uint32_t* data32 = nullptr;
		uint16_t* data16 = nullptr;
		uint8_t* data8 = nullptr;
		const uint32_t* counts = nullptr;
		uint32_t baseIdx = 0;
		uint32_t chunk = 0;
		uint32_t begin = 0;
		uint32_t end = 0;
		uint32_t pitch = 0;
		SELL8ChunkFormat format = SELL8ChunkFormat::Explicit32;

		void StoreNeighbor(uint32_t lane, uint32_t k, uint32_t j) const noexcept
		{
			const uint64_t linearIdx = static_cast<uint64_t>(k) * CPU::N_SIMD_WIDTH + lane;

			if (format == SELL8ChunkFormat::Delta8)
			{
				assert(j >= baseIdx && j - baseIdx <= 0xffu);
				data8[linearIdx] = static_cast<uint8_t>(j - baseIdx);
			}
			else if (format == SELL8ChunkFormat::Delta16)
			{
				assert(j >= baseIdx && j - baseIdx <= 0xffffu);
				data16[linearIdx] = static_cast<uint16_t>(j - baseIdx);
			}
			else
			{
				data32[linearIdx] = j;
			}
		}
	};

	class I_NeighborListSELL8 : public I_NeighborList
	{
	public: // -------------------------------------------------------------------------------------

		virtual uint32_t ChunkNumTotal() const noexcept = 0;

		virtual ChunkedSELL8NeighborSpan_Read FFChunk(uint32_t chunk) const noexcept = 0;
		virtual ChunkedSELL8NeighborSpan_Read FBChunk(uint32_t chunk) const noexcept = 0;
		virtual ChunkedSELL8NeighborSpan_Write FFChunkWrite(uint32_t chunk) noexcept = 0;
		virtual ChunkedSELL8NeighborSpan_Write FBChunkWrite(uint32_t chunk) noexcept = 0;

		virtual void FinishFFChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) = 0;
		virtual void FinishFBChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) = 0;

		virtual bool NeedsChunkNeighborIdxRange() const noexcept { return false; }
		virtual void SetChunkNeighborIdxRangeFF(uint32_t chunk, uint32_t minRef, uint32_t maxRef)
		{
			(void)chunk;
			(void)minRef;
			(void)maxRef;
		}
		virtual void SetChunkNeighborIdxRangeFB(uint32_t chunk, uint32_t minRef, uint32_t maxRef)
		{
			(void)chunk;
			(void)minRef;
			(void)maxRef;
		}
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ individual vectors neighbor lists, explicit ] -----------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NeighborLists_IndividualVectors_Explicit_CPU01 final
		: public I_NeighborList
	{
	public: // -------------------------------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B, const SimParams& params) override;

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

		bool RunCountQuery() const override { return false; }
		bool RunFillQuery() const override { return true; }
		bool RunSecondFillQuery() const override { return false; }

	private: // ------------------------------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// neighbor storage
		std::vector<std::vector<uint32_t, uint32_tbb_alloc>> nbrListFF_;
		std::vector<std::vector<uint32_t, uint32_tbb_alloc>> nbrListFB_;
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ single CSR vector neighbor list, explicit ] -------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NeighborLists_SingleCSRVector_Explicit_CPU01 final
		: public I_NeighborList
	{
	public: // -------------------------------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B, const SimParams& params) override;

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

		bool RunCountQuery() const override { return true; }
		bool RunFillQuery() const override { return true; }
		bool RunSecondFillQuery() const override { return false; }

	private: // ------------------------------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// explicit CSR neighbor storage using CSR blocks
		// (32-bit local offset per-particle and 64-bit global block offset per-block)
		std::vector<uint32_t> nbrCountFF_; // TODO: make this uint16_t?
		std::vector<uint32_t> nbrCountFB_; // TODO: make this uint16_t?
		std::vector<uint32_t> nbrLocalOffsetsFF_; // size: N_F
		std::vector<uint32_t> nbrLocalOffsetsFB_; // size: N_F
		std::vector<uint64_t> nbrBlockBasesFF_; // size: N_F / offsetBlockParticleCount_ + 1
		std::vector<uint64_t> nbrBlockBasesFB_; // size: N_F / offsetBlockParticleCount_ + 1
		std::vector<uint32_t> nbrListFF_; // size: N_F * avg FF neighbor count
		std::vector<uint32_t> nbrListFB_; // size: N_F * avg FB neighbor count

		// misc
		static constexpr uint32_t offsetBlockParticleCount_ = 8192u;
		static constexpr uint32_t extraAllocCapacityPct_ = 4; // TODO: fine-tune

		// helpers
		static uint32_t OffsetBlockNumTotal(uint32_t N_F) noexcept;
		static uint32_t OffsetBlockOfParticle(uint32_t i) noexcept;
		static uint32_t OffsetBlockBegin(uint32_t block) noexcept;
		static uint32_t OffsetBlockEnd(uint32_t block, uint32_t N_F) noexcept;
		void FinalizeBlockCSR(
			const std::vector<uint32_t>& nbrCounts,
			std::vector<uint32_t>& nbrLocalOffsets,
			std::vector<uint64_t>& nbrBlockBases,
			std::vector<uint32_t>& nbrList,
			const char* offsetOverflowMsg) const;
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ single CSR vector neighbor list, adaptive compression ] -------------------------------
	// ---------------------------------------------------------------------------------------------

	class NeighborLists_SingleCSRVector_AdaptCompress_CPU01 final
		: public I_NeighborList
	{
	public: // -------------------------------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B, const SimParams& params) override;

		void Finalize_Count() override;

		void Finalize_Fill() override;

		void SetNbrCountFF(uint32_t i, uint32_t count) override {}
		void SetNbrCountFB(uint32_t i, uint32_t count) override {}

		void SetInputOrder(NeighborListInputOrder inputOrder) override
		{
			inputOrder_ = inputOrder;
		}

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

		uint32_t CountFF(uint32_t i) const override;
		uint32_t CountFB(uint32_t i) const override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(
			const SimParams& params) const override;

		bool RunCountQuery() const override { return false; }
		bool RunFillQuery() const override { return true; }
		bool RunSecondFillQuery() const override { return true; }

	private: // ------------------------------------------------------------------------------------

		enum class FillPhase : uint8_t
		{
			Measure,
			Encode,
			Ready
		};

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// compressed CSR neighbor storage using CSR blocks
		// (32-bit local offset per-particle and 64-bit global block offset per-block)
		// (local offset vectors temporarily store encoded neighbor list byte counts during
		// first fill-query, then become block-local byte offsets after finalization)
		std::vector<uint32_t> nbrCountFF_; // TODO: make this uint16_t?
		std::vector<uint32_t> nbrCountFB_; // TODO: make this uint16_t?
		std::vector<uint32_t> nbrLocalOffsetsFF_; // size: N_F
		std::vector<uint32_t> nbrLocalOffsetsFB_; // size: N_F
		std::vector<uint64_t> nbrBlockBasesFF_; // size: N_F / offsetBlockParticleCount_ + 1
		std::vector<uint64_t> nbrBlockBasesFB_; // size: N_F / offsetBlockParticleCount_ + 1
		std::vector<uint8_t> nbrListCompressedFF_; // size: N_F * avg FF neighbor count * avg bytes per neighbor
		std::vector<uint8_t> nbrListCompressedFB_; // size: N_F * avg FB neighbor count * avg bytes per neighbor
		std::vector<uint8_t> nbrWidthCodeFF_; // width code from measure phase, reused in encode
		std::vector<uint8_t> nbrWidthCodeFB_; // width code from measure phase, reused in encode

		// thread-local decode scratch for FF()/FB() to return neighbors as uint32_t
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> nbrScratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> nbrScratchFB_;

		// thread-local sorting scratch used if incoming list is not guaranteed to be sorted
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> sortScratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> sortScratchFB_;

		// misc
		NeighborListInputOrder inputOrder_ = NeighborListInputOrder::Unknown;
		FillPhase fillPhase_ = FillPhase::Measure;
		static constexpr uint32_t offsetBlockParticleCount_ = 8192u;
		static constexpr uint32_t extraAllocCapacityPct_ = 4; // TODO: fine-tune
		static constexpr uint32_t overheadBytesPerNeighborList = 5;
		static constexpr double expctdBytePerCompressedNeighbor = 1.6; // TODO: fine-tune

		// helpers
		static uint32_t OffsetBlockNumTotal(uint32_t N_F) noexcept;
		static uint32_t OffsetBlockOfParticle(uint32_t i) noexcept;
		static uint32_t OffsetBlockBegin(uint32_t block) noexcept;
		static uint32_t OffsetBlockEnd(uint32_t block, uint32_t N_F) noexcept;
		static uint64_t ParticleStorageBegin(uint32_t i,
			const std::vector<uint32_t>& nbrLocalOffsets,
			const std::vector<uint64_t>& nbrBlockBases) noexcept;
		static uint64_t ParticleStorageEnd(uint32_t i, uint32_t N_F,
			const std::vector<uint32_t>& nbrLocalOffsets,
			const std::vector<uint64_t>& nbrBlockBases) noexcept;
		void FinalizeBlockCSR(
			std::vector<uint32_t>& nbrLocalOffsets,
			std::vector<uint64_t>& nbrBlockBases,
			std::vector<uint8_t>& nbrList) const;
		static uint8_t ChooseWidthCode(uint32_t maxGap) noexcept;
		static uint32_t GetNumWidthBytes(uint8_t widthCode) noexcept;
		static uint64_t MeasureEncodedBytes(
			const uint32_t* js, uint32_t count, uint8_t& widthCode);
		static inline uint16_t ReadU16LE(const uint8_t* src) noexcept;
		static inline uint32_t ReadU32LE(const uint8_t* src) noexcept;
		static inline void WriteU16LE(uint8_t* dst, uint16_t v) noexcept;
		static inline void WriteU32LE(uint8_t* dst, uint32_t v) noexcept;
		template<typename T_Out>
		static void DecodeNeighborList(const uint8_t* src, uint64_t srcBytes,
			uint32_t count, T_Out& out);
		static void EncodeNeighborList(uint8_t* dst, uint64_t dstBytes,
			const uint32_t* js, uint32_t count, uint8_t widthCode);
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ single CSR vector neighbor lists, VByte compression ] ---------------------------------
	// ---------------------------------------------------------------------------------------------

	class NeighborLists_SingleCSRVector_VByteCompress_CPU01 final
		: public I_NeighborList
	{
	public: // -------------------------------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B, const SimParams& params) override;

		void Finalize_Count() override;

		void Finalize_Fill() override;

		void SetNbrCountFF(uint32_t i, uint32_t count) override {}
		void SetNbrCountFB(uint32_t i, uint32_t count) override {}

		void SetInputOrder(NeighborListInputOrder inputOrder) override
		{
			inputOrder_ = inputOrder;
		}

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

		uint32_t CountFF(uint32_t i) const override;
		uint32_t CountFB(uint32_t i) const override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(
			const SimParams& params) const override;

		bool RunCountQuery() const override { return false; }
		bool RunFillQuery() const override { return true; }
		bool RunSecondFillQuery() const override { return true; }

	private: // ------------------------------------------------------------------------------------

		enum class FillPhase : uint8_t
		{
			Measure,
			Encode,
			Ready
		};

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// compressed CSR neighbor storage using CSR blocks
		// (32-bit local offset per-particle and 64-bit global block offset per-block)
		// (local offset vectors temporarily store encoded neighbor list byte counts during
		// first fill-query, then become block-local byte offsets after finalization)
		std::vector<uint32_t> nbrCountFF_; // TODO: make this uint16_t?
		std::vector<uint32_t> nbrCountFB_; // TODO: make this uint16_t?
		std::vector<uint32_t> nbrLocalOffsetsFF_; // size: N_F
		std::vector<uint32_t> nbrLocalOffsetsFB_; // size: N_F
		std::vector<uint64_t> nbrBlockBasesFF_; // size: N_F / offsetBlockParticleCount_ + 1
		std::vector<uint64_t> nbrBlockBasesFB_; // size: N_F / offsetBlockParticleCount_ + 1
		std::vector<uint8_t> nbrListCompressedFF_; // size: N_F * avg FF neighbor count * avg bytes per neighbor
		std::vector<uint8_t> nbrListCompressedFB_; // size: N_F * avg FB neighbor count * avg bytes per neighbor

		// thread-local decode scratch for FF()/FB() to return neighbors as uint32_t
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> nbrScratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> nbrScratchFB_;

		// thread-local sorting scratch used if incoming list is not guaranteed to be sorted
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> sortScratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> sortScratchFB_;

		// misc
		NeighborListInputOrder inputOrder_ = NeighborListInputOrder::Unknown;
		FillPhase fillPhase_ = FillPhase::Measure;
		static constexpr uint32_t offsetBlockParticleCount_ = 8192u;
		static constexpr uint32_t extraAllocCapacityPct_ = 4; // TODO: fine-tune
		static constexpr uint32_t overheadBytesPerNeighborList = 4;
		static constexpr double expctdBytePerCompressedNeighbor = 0.8; // TODO: fine-tune

		// helpers
		static uint32_t OffsetBlockNumTotal(uint32_t N_F) noexcept;
		static uint32_t OffsetBlockOfParticle(uint32_t i) noexcept;
		static uint32_t OffsetBlockBegin(uint32_t block) noexcept;
		static uint32_t OffsetBlockEnd(uint32_t block, uint32_t N_F) noexcept;
		static uint64_t ParticleStorageBegin(uint32_t i,
			const std::vector<uint32_t>& nbrLocalOffsets,
			const std::vector<uint64_t>& nbrBlockBases) noexcept;
		static uint64_t ParticleStorageEnd(uint32_t i, uint32_t N_F,
			const std::vector<uint32_t>& nbrLocalOffsets,
			const std::vector<uint64_t>& nbrBlockBases) noexcept;
		void FinalizeBlockCSR(
			std::vector<uint32_t>& nbrLocalOffsets,
			std::vector<uint64_t>& nbrBlockBases,
			std::vector<uint8_t>& nbrList) const;
		static uint32_t ControlBytesFor(uint32_t count) noexcept;
		static uint8_t DeltaMask(uint32_t delta) noexcept;
		static uint64_t MeasureEncodedBytes(const uint32_t* js, uint32_t count);
		static inline uint32_t ReadU32LE(const uint8_t* src) noexcept;
		static inline void WriteU32LE(uint8_t* dst, uint32_t v) noexcept;
		template<typename T_Out>
		static void DecodeNeighborList(const uint8_t* src, uint64_t srcBytes,
			uint32_t count, T_Out& out);
		static void EncodeNeighborList(uint8_t* dst, uint64_t dstBytes,
			const uint32_t* js, uint32_t count);
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ chunked SELL8 neighbor list, explicit ] -----------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NeighborLists_ChunkedSELL8_Explicit_CPU01 final
		: public I_NeighborListSELL8
	{
	public: // -------------------------------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B, const SimParams& params) override;

		void Finalize_Count() override;

		void Finalize_Fill() override {}

		void SetNbrCountFF(uint32_t i, uint32_t count) override;
		void SetNbrCountFB(uint32_t i, uint32_t count) override;

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

		uint32_t CountFF(uint32_t i) const override;
		uint32_t CountFB(uint32_t i) const override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(
			const SimParams& params) const override;

		bool RunCountQuery() const override { return true; }
		bool RunFillQuery() const override { return true; }
		bool RunSecondFillQuery() const override { return false; }

		uint32_t ChunkNumTotal() const noexcept override;
		ChunkedSELL8NeighborSpan_Read FFChunk(uint32_t chunk) const noexcept override;
		ChunkedSELL8NeighborSpan_Read FBChunk(uint32_t chunk) const noexcept override;
		ChunkedSELL8NeighborSpan_Write FFChunkWrite(uint32_t chunk) noexcept override;
		ChunkedSELL8NeighborSpan_Write FBChunkWrite(uint32_t chunk) noexcept override;
		void FinishFFChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) override;
		void FinishFBChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) override;

	private: // ------------------------------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// explicit chunked SELL8 neighbor storage
		std::vector<uint32_t> nbrCountFF_; // per-particle neighbor count
		std::vector<uint32_t> nbrCountFB_;
		std::vector<uint32_t> nbrChunkMaxCountFF_; // per-chunk max neighbor count (pitch used for chunk length)
		std::vector<uint32_t> nbrChunkMaxCountFB_;
		std::vector<uint64_t> nbrChunkOffsetsFF_; // per-chunk starting index for neighbor stored in nbrListFF_
		std::vector<uint64_t> nbrChunkOffsetsFB_;
		std::vector<uint32_t> nbrListFF_; // actual neighbor indices, concatenated for all chunks
		std::vector<uint32_t> nbrListFB_;

		// thread-local neighbor scratch for contiguous access
		// TODO: still needed despite chunked SELL8 storage?
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> nbrScratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> nbrScratchFB_;

		// misc
		static constexpr uint32_t chunkWidth = 8u;
		static constexpr uint32_t extraAllocCapacityPct_ = 4; // TODO: fine-tune

		// helpers
		static uint32_t ChunkOfParticle(uint32_t i) noexcept;
		static uint32_t LaneOfParticle(uint32_t i) noexcept;
		static uint32_t ChunkBegin(uint32_t chunk) noexcept;
		static uint32_t ChunkEnd(uint32_t chunk, uint32_t N_F) noexcept;
		static uint32_t ChunkNumTotal(uint32_t N_F) noexcept;
		static uint64_t ParticleStorageIdx(
			uint32_t i, uint32_t k,
			const std::vector<uint64_t>& nbrChunkOffsets) noexcept;

		void BuildChunkMaxCount(
			const std::vector<uint32_t>& nbrCounts,
			std::vector<uint32_t>& nbrChunkMaxCount) const;

		void FinalizeChunkOffsets(
			const std::vector<uint32_t>& nbrChunkMaxCount,
			std::vector<uint64_t>& nbrChunkOffsets,
			std::vector<uint32_t>& nbrList) const;
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ chunked SELL8 neighbor list, adaptive compression ] -----------------------------------
	// ---------------------------------------------------------------------------------------------

	class NeighborLists_ChunkedSELL8_AdaptCompress_CPU01 final
		: public I_NeighborListSELL8
	{
	public: // -------------------------------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B, const SimParams& params) override;

		void Finalize_Count() override;

		void Finalize_Fill() override {}

		void SetNbrCountFF(uint32_t i, uint32_t count) override;
		void SetNbrCountFB(uint32_t i, uint32_t count) override;

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

		uint32_t CountFF(uint32_t i) const override;
		uint32_t CountFB(uint32_t i) const override;

		std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(
			const SimParams& params) const override;

		bool RunCountQuery() const override { return true; }
		bool RunFillQuery() const override { return true; }
		bool RunSecondFillQuery() const override { return false; }

		uint32_t ChunkNumTotal() const noexcept override;
		ChunkedSELL8NeighborSpan_Read FFChunk(uint32_t chunk) const noexcept override;
		ChunkedSELL8NeighborSpan_Read FBChunk(uint32_t chunk) const noexcept override;
		ChunkedSELL8NeighborSpan_Write FFChunkWrite(uint32_t chunk) noexcept override;
		ChunkedSELL8NeighborSpan_Write FBChunkWrite(uint32_t chunk) noexcept override;
		void FinishFFChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) override;
		void FinishFBChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) override;

		bool NeedsChunkNeighborIdxRange() const noexcept override { return true; }
		void SetChunkNeighborIdxRangeFF(uint32_t chunk, uint32_t minRef, uint32_t maxRef) override;
		void SetChunkNeighborIdxRangeFB(uint32_t chunk, uint32_t minRef, uint32_t maxRef) override;

	private: // ------------------------------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// compressed chunked SELL8 neighbor storage
		std::vector<uint32_t> nbrCountFF_; // per-particle neighbor count
		std::vector<uint32_t> nbrCountFB_;
		std::vector<uint32_t> nbrChunkMaxCountFF_; // per-chunk max neighbor count (pitch used for chunk length)
		std::vector<uint32_t> nbrChunkMaxCountFB_;
		std::vector<uint32_t> nbrChunkMinRefFF_; // per-chunk min neighbor index from the count-query
		std::vector<uint32_t> nbrChunkMinRefFB_;
		std::vector<uint32_t> nbrChunkMaxRefFF_; // per-chunk max neighbor index from the count-query
		std::vector<uint32_t> nbrChunkMaxRefFB_;
		std::vector<uint32_t> nbrChunkBaseIdxFF_; // per-chunk base index used for delta storage
		std::vector<uint32_t> nbrChunkBaseIdxFB_;
		std::vector<SELL8ChunkFormat> nbrChunkFormatFF_; // per-chunk storage format
		std::vector<SELL8ChunkFormat> nbrChunkFormatFB_;
		std::vector<uint64_t> nbrChunkOffsetsFF_; // per-chunk starting index into the selected format buffer
		std::vector<uint64_t> nbrChunkOffsetsFB_;
		std::vector<uint8_t> nbrListDelta8FF_;
		std::vector<uint8_t> nbrListDelta8FB_;
		std::vector<uint16_t> nbrListDelta16FF_;
		std::vector<uint16_t> nbrListDelta16FB_;
		std::vector<uint32_t> nbrListAbs32FF_;
		std::vector<uint32_t> nbrListAbs32FB_;

		// thread-local neighbor scratch for contiguous access
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFF>> nbrScratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_perScratchInlineFB>> nbrScratchFB_;

		// misc
		static constexpr uint32_t chunkWidth = 8u;
		static constexpr uint32_t extraAllocCapacityPct_ = 4; // TODO: fine-tune

		// helpers
		static uint32_t ChunkOfParticle(uint32_t i) noexcept;
		static uint32_t LaneOfParticle(uint32_t i) noexcept;
		static uint32_t ChunkBegin(uint32_t chunk) noexcept;
		static uint32_t ChunkEnd(uint32_t chunk, uint32_t N_F) noexcept;
		static uint32_t ChunkNumTotal(uint32_t N_F) noexcept;
		static SELL8ChunkFormat ChooseChunkFormat(
			uint32_t pitch, uint32_t minRef, uint32_t maxRef) noexcept;
		static uint64_t ChunkStorageSize(uint32_t pitch) noexcept;

		ChunkedSELL8NeighborSpan_Read MakeReadSpanFF(uint32_t chunk) const noexcept;
		ChunkedSELL8NeighborSpan_Read MakeReadSpanFB(uint32_t chunk) const noexcept;
		ChunkedSELL8NeighborSpan_Write MakeWriteSpanFF(uint32_t chunk) noexcept;
		ChunkedSELL8NeighborSpan_Write MakeWriteSpanFB(uint32_t chunk) noexcept;

		void BuildChunkMaxCount(
			const std::vector<uint32_t>& nbrCounts,
			std::vector<uint32_t>& nbrChunkMaxCount) const;

		void FinalizeChunkStorage(
			const std::vector<uint32_t>& nbrChunkMaxCount,
			const std::vector<uint32_t>& nbrChunkMinRef,
			const std::vector<uint32_t>& nbrChunkMaxRef,
			std::vector<uint32_t>& nbrChunkBaseIdx,
			std::vector<SELL8ChunkFormat>& nbrChunkFormat,
			std::vector<uint64_t>& nbrChunkOffsets,
			std::vector<uint8_t>& nbrListDelta8,
			std::vector<uint16_t>& nbrListDelta16,
			std::vector<uint32_t>& nbrListAbs32) const;
	};

} // namespace Sim
