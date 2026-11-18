// neighbor_lists.h
#pragma once

// external
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <tbb/enumerable_thread_specific.h>

// internal
#include "source/simulation/core/backend_sim.h"
#include "source/simulation/CPU/simd.h"



// TODO: make these used the params based on heuristic expected values?
static constexpr uint32_t N_PER_SCRATCH_INLINE_FF = 64;
static constexpr uint32_t N_PER_SCRATCH_INLINE_FB = 48;
static constexpr uint32_t N_PER_SCRATCH_INLINE_FR = 48;

namespace Sim
{
	enum class SELLChunkFormat : uint8_t
	{
		Delta16,
		Abs32
	};

	enum class SELLDictChunkFormat : uint8_t
	{
		DictDelta16x2, // uint16 offsets from the base index of a low or a high group
		DictAbs32
	};

	// TODO: use uint8_t instead?
	using SELLNeighborCount = uint16_t;

	// max distinct neighbors of a dict chunk
	static constexpr uint32_t N_SELL_DICT_SIZE_MAX = 256;

	// one bit per SIMD lane, set for the lanes that have a dict entry as neighbor
	using SELLLaneMask = std::conditional_t<CPU::N_SIMD_WIDTH == 16, uint16_t, uint8_t>;

	// byte size of one dict entry, without its lane mask
	inline uint32_t SELLDictEntryBytes(SELLDictChunkFormat dictFormat) noexcept
	{
		return (dictFormat == SELLDictChunkFormat::DictAbs32)
			? static_cast<uint32_t>(sizeof(uint32_t))
			: static_cast<uint32_t>(sizeof(uint16_t));
	}

	// byte size of a dict chunk's dictionary and lane masks, rounded up to whole uint32 for alignment
	inline uint32_t SELLDictBytes(SELLDictChunkFormat dictFormat, uint32_t N_entries) noexcept
	{
		const uint32_t N_entryBytes = SELLDictEntryBytes(dictFormat)
			+ static_cast<uint32_t>(sizeof(SELLLaneMask));
		return (N_entries * N_entryBytes + 3u) & ~3u;
	}

	// read-view used in the neighbor retrieval (e.g. solver kernels)
	struct ChunkedSELLNeighborSpan_Read
	{
		const uint32_t* data32 = nullptr;
		const uint16_t* data16 = nullptr;
		const SELLNeighborCount* counts = nullptr;
		uint32_t baseIdx = 0; // base index of the chunk, used for delta-encoded neighbors
		uint32_t baseIdx2 = 0; // base index of the high group of a DictDelta16x2 dictionary
		uint32_t N_dictLow = 0; // its low group fills the dictionary positions below this one
		uint32_t chunk = 0;
		uint32_t begin = 0;
		uint32_t end = 0;
		uint32_t pitch = 0;
		SELLChunkFormat format = SELLChunkFormat::Abs32;
		const uint8_t* dict = nullptr; // distinct neighbors of a dict chunk, nullptr for a normal chunk
		SELLDictChunkFormat dictFormat = SELLDictChunkFormat::DictAbs32;
		uint32_t N_dict = 0; // dictionary entries of a dict chunk
		const uint8_t* dictLaneMasks = nullptr; // SELLLaneMask of each dict entry, after the dictionary

		// lanes that have the neighbor at a position in the dictionary
		uint32_t ReadDictLaneMask(uint32_t dictIdx) const noexcept
		{
			SELLLaneMask laneMask = 0;
			std::memcpy(&laneMask, dictLaneMasks + sizeof(laneMask) * dictIdx, sizeof(laneMask));
			return laneMask;
		}

		// neighbor at a position in the dictionary of a dict chunk
		uint32_t ReadDictNeighborIdx(uint32_t dictIdx) const noexcept
		{
			if (dictFormat == SELLDictChunkFormat::DictDelta16x2)
			{
				uint16_t delta = 0;
				std::memcpy(&delta, dict + sizeof(delta) * dictIdx, sizeof(delta));
				return (dictIdx < N_dictLow ? baseIdx : baseIdx2) + static_cast<uint32_t>(delta);
			}

			uint32_t idx = 0;
			std::memcpy(&idx, dict + sizeof(idx) * dictIdx, sizeof(idx));
			return idx;
		}

		uint32_t ReadNeighborIdx(uint32_t lane, uint32_t k) const noexcept
		{
			assert(!dict);
			const uint64_t linearIdx = static_cast<uint64_t>(k) * CPU::N_SIMD_WIDTH + lane;

			if (format == SELLChunkFormat::Delta16)
			{
				return baseIdx + static_cast<uint32_t>(data16[linearIdx]);
			}

			return data32[linearIdx];
		}

		CPU::UInt32xSIMD LoadNeighborIdx32xSIMD(uint32_t k) const noexcept
		{
			assert(!dict);
			const uint64_t linearIdx = static_cast<uint64_t>(k) * CPU::N_SIMD_WIDTH;

			if (format == SELLChunkFormat::Delta16)
			{
				return CPU::LoadUInt16AsUInt32xSIMD(data16 + linearIdx)
					+ CPU::SetUInt32xSIMD(baseIdx);
			}

			return CPU::LoadUInt32xSIMD(data32 + linearIdx);
		}
	};

	// write-view used in the neighbor search (e.g. fill-queries)
	struct ChunkedSELLNeighborSpan_Write
	{
		uint32_t* data32 = nullptr;
		uint16_t* data16 = nullptr;
		const SELLNeighborCount* counts = nullptr;
		uint32_t baseIdx = 0; // base index of the chunk, used for delta-encoded neighbors
		uint32_t baseIdx2 = 0; // base index of the high group of a DictDelta16x2 dictionary
		uint32_t N_dictLow = 0; // those of the low group of a DictDelta16x2 dictionary
		uint32_t chunk = 0;
		uint32_t begin = 0;
		uint32_t end = 0;
		uint32_t pitch = 0;
		SELLChunkFormat format = SELLChunkFormat::Abs32;
		uint8_t* dict = nullptr; // distinct neighbors of a dict chunk, nullptr for a normal chunk
		SELLDictChunkFormat dictFormat = SELLDictChunkFormat::DictAbs32;
		uint32_t N_dictMax = 0; // dictionary size fixed by the count-query
		uint32_t N_dict = 0; // dictionary entries written so far
		uint8_t* dictLaneMasks = nullptr; // SELLLaneMask of each dict entry, after the dictionary

		// stores the lanes that have the neighbor at a position in the dictionary
		void StoreDictLaneMask(uint32_t dictIdx, uint32_t laneMask) const noexcept
		{
			const SELLLaneMask dictLaneMask = static_cast<SELLLaneMask>(laneMask);
			std::memcpy(dictLaneMasks + sizeof(dictLaneMask) * dictIdx, &dictLaneMask, sizeof(dictLaneMask));
		}

		// adds neighbor j to the dictionary of a dict chunk and returns its position for StoreDictLaneMask()
		uint32_t RegisterNeighbor(uint32_t j)
		{
			assert(dict);
			if (N_dict >= N_dictMax) [[unlikely]]
			{
				throw std::runtime_error(
					"ChunkedSELL dict fill exceeded the chunk neighbor dictionary.");
			}

			uint32_t dictIdx = N_dict;
			if (dictFormat == SELLDictChunkFormat::DictDelta16x2)
			{
				// low group fills the dictionary from the front, high group from the back
				const bool isLow = (j - baseIdx <= 0xffffu);
				assert(j >= baseIdx && (isLow || (j >= baseIdx2 && j - baseIdx2 <= 0xffffu)));
				dictIdx = isLow ? N_dictLow++ : N_dictMax - 1u - (N_dict - N_dictLow);
				const uint16_t delta = static_cast<uint16_t>(j - (isLow ? baseIdx : baseIdx2));
				std::memcpy(dict + sizeof(delta) * dictIdx, &delta, sizeof(delta));
			}
			else
			{
				std::memcpy(dict + sizeof(j) * dictIdx, &j, sizeof(j));
			}

			N_dict++;
			return dictIdx;
		}

		void StoreNeighbor(uint32_t lane, uint32_t k, uint32_t j) const noexcept
		{
			assert(!dict);
			const uint64_t linearIdx = static_cast<uint64_t>(k) * CPU::N_SIMD_WIDTH + lane;

			if (format == SELLChunkFormat::Delta16)
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

	struct ChunkedSELLNeighborCountResult
	{
		const uint32_t* counts;
		uint32_t minRef;
		uint32_t maxRef;
		uint32_t N_distinct = 0; // distinct neighbors of the chunk, 0 if the query does not track them
		const uint32_t* distinct = nullptr; // their indices, up to N_SELL_DICT_SIZE_MAX
		const uint32_t* distinctLaneMasks = nullptr; // the lanes of each of them, nullptr leaves the chunk a normal chunk
	};

	class I_NeighborListSELL : public I_NeighborList
	{
	public:

		virtual uint32_t ChunkNumTotal() const noexcept = 0;

		virtual ChunkedSELLNeighborSpan_Read FFChunk(uint32_t chunk) const noexcept = 0;
		virtual ChunkedSELLNeighborSpan_Read FBChunk(uint32_t chunk) const noexcept = 0;
		virtual ChunkedSELLNeighborSpan_Read FRChunk(uint32_t chunk) const noexcept = 0;
		virtual ChunkedSELLNeighborSpan_Write FFChunkWrite(uint32_t chunk) noexcept = 0;
		virtual ChunkedSELLNeighborSpan_Write FBChunkWrite(uint32_t chunk) noexcept = 0;
		virtual ChunkedSELLNeighborSpan_Write FRChunkWrite(uint32_t chunk) noexcept = 0;

		virtual void FinishFFChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) = 0;
		virtual void FinishFBChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) = 0;
		virtual void FinishFRChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) = 0;

		virtual void PrepareFluidCountWrite(bool allChunksWillBeWritten) = 0;
		virtual void CommitFluidChunkCounts(
			uint32_t chunk,
			const ChunkedSELLNeighborCountResult& countsFF,
			const ChunkedSELLNeighborCountResult& countsFB,
			const ChunkedSELLNeighborCountResult& countsFR) = 0;

		virtual bool NeedsChunkNeighborIdxRange() const noexcept { return false; }
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ chunked SELL neighbor list, explicit ] -----------------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NeighborLists_ChunkedSELL_Explicit_CPU final
		: public I_NeighborListSELL
	{
	public:

		void Resize(
			uint32_t N_F, uint32_t N_B, uint32_t N_R,
			const SimParams& params) override;

		void Finalize_Count() override;

		void Finalize_Fill() override {}

		void SetNbrCountFF(uint32_t i, uint32_t N_nbrs) override;
		void SetNbrCountFB(uint32_t i, uint32_t N_nbrs) override;
		void SetNbrCountFR(uint32_t i, uint32_t N_nbrs) override;
		void SetNbrCountRF(uint32_t i, uint32_t N_nbrs) override;
		void SetNbrCountRB(uint32_t i, uint32_t N_nbrs) override;
		void SetNbrCountRR(uint32_t i, uint32_t N_nbrs) override;

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t N_nbrs) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t N_nbrs) override;
		void SetNbrListFR(uint32_t i, const uint32_t* js, uint32_t N_nbrs) override;
		void SetNbrListRF(uint32_t i, const uint32_t* js, uint32_t N_nbrs) override;
		void SetNbrListRB(uint32_t i, const uint32_t* js, uint32_t N_nbrs) override;
		void SetNbrListRR(uint32_t i, const uint32_t* js, uint32_t N_nbrs) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;
		NeighborSpan FR(uint32_t i) const override;
		NeighborSpan RF(uint32_t i) const override;
		NeighborSpan RB(uint32_t i) const override;
		NeighborSpan RR(uint32_t i) const override;

		uint32_t CountFF(uint32_t i) const override;
		uint32_t CountFB(uint32_t i) const override;
		uint32_t CountFR(uint32_t i) const override;
		uint32_t CountRF(uint32_t i) const override;
		uint32_t CountRB(uint32_t i) const override;
		uint32_t CountRR(uint32_t i) const override;

		Mem::ParticleClassMemUsage GetMemUsage() const override;

		bool RunCountQuery() const override { return true; }
		bool RunFillQuery() const override { return true; }

		uint32_t ChunkNumTotal() const noexcept override;
		ChunkedSELLNeighborSpan_Read FFChunk(uint32_t chunk) const noexcept override;
		ChunkedSELLNeighborSpan_Read FBChunk(uint32_t chunk) const noexcept override;
		ChunkedSELLNeighborSpan_Read FRChunk(uint32_t chunk) const noexcept override;
		ChunkedSELLNeighborSpan_Write FFChunkWrite(uint32_t chunk) noexcept override;
		ChunkedSELLNeighborSpan_Write FBChunkWrite(uint32_t chunk) noexcept override;
		ChunkedSELLNeighborSpan_Write FRChunkWrite(uint32_t chunk) noexcept override;

		void FinishFFChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) override;
		void FinishFBChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) override;
		void FinishFRChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) override;

		void PrepareFluidCountWrite(bool allChunksWillBeWritten) override;
		void CommitFluidChunkCounts(
			uint32_t chunk,
			const ChunkedSELLNeighborCountResult& countsFF,
			const ChunkedSELLNeighborCountResult& countsFB,
			const ChunkedSELLNeighborCountResult& countsFR) override;

	private:

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;
		uint32_t N_R_ = 0;

		// explicit chunked SELL neighbor storage
		// (for fluid source particles)
		std::vector<SELLNeighborCount> nbrCountFF_; // per-particle neighbor count
		std::vector<SELLNeighborCount> nbrCountFB_;
		std::vector<SELLNeighborCount> nbrCountFR_;
		std::vector<SELLNeighborCount> nbrChunkMaxCountFF_; // per-chunk max neighbor count (pitch used for chunk length)
		std::vector<SELLNeighborCount> nbrChunkMaxCountFB_;
		std::vector<SELLNeighborCount> nbrChunkMaxCountFR_;
		std::vector<uint64_t> nbrChunkOffsetsFF_; // per-chunk starting index for neighbor stored in nbrListFF_
		std::vector<uint64_t> nbrChunkOffsetsFB_;
		std::vector<uint64_t> nbrChunkOffsetsFR_;
		std::vector<uint32_t> nbrListFF_; // actual neighbor indices, concatenated for all chunks
		std::vector<uint32_t> nbrListFB_;
		std::vector<uint32_t> nbrListFR_;

		// CSR neighbor storage for rigid source particles
		std::vector<SELLNeighborCount> nbrCountRF_;
		std::vector<SELLNeighborCount> nbrCountRB_;
		std::vector<SELLNeighborCount> nbrCountRR_;
		std::vector<uint64_t> nbrOffsetsRF_;
		std::vector<uint64_t> nbrOffsetsRB_;
		std::vector<uint64_t> nbrOffsetsRR_;
		std::vector<uint32_t> nbrListRF_;
		std::vector<uint32_t> nbrListRB_;
		std::vector<uint32_t> nbrListRR_;

		// thread-local neighbor scratch for contiguous access
		// TODO: still needed despite chunked SELL storage?
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_PER_SCRATCH_INLINE_FF>> nbrScratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_PER_SCRATCH_INLINE_FB>> nbrScratchFB_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_PER_SCRATCH_INLINE_FR>> nbrScratchFR_;

		// misc
		static constexpr uint32_t chunkWidth = CPU::N_SIMD_WIDTH;
		static constexpr float extraAllocCapacityPct_ = 4.0f; // TODO: fine-tune

		// helpers
		static uint32_t ChunkOfParticle(uint32_t i) noexcept;
		static uint32_t LaneOfParticle(uint32_t i) noexcept;
		static uint32_t ChunkBegin(uint32_t chunk) noexcept;
		static uint32_t ChunkEnd(uint32_t chunk, uint32_t N_F) noexcept;
		static uint32_t ChunkNumTotal(uint32_t N_F) noexcept;
		static uint64_t ParticleStorageIdx(
			uint32_t i, uint32_t k,
			const std::vector<uint64_t>& nbrChunkOffsets) noexcept;
		void FinalizeChunkOffsets(
			const std::vector<SELLNeighborCount>& nbrChunkMaxCount,
			std::vector<uint64_t>& nbrChunkOffsets,
			std::vector<uint32_t>& nbrList) const;
	};

	// ---------------------------------------------------------------------------------------------
	// --- [ chunked SELL neighbor list, adaptive compression ] ------------------------------------
	// ---------------------------------------------------------------------------------------------

	class NeighborLists_ChunkedSELL_AdaptCompress_CPU final
		: public I_NeighborListSELL
	{
	public:

		void Resize(
			uint32_t N_F, uint32_t N_B, uint32_t N_R,
			const SimParams& params) override;

		void Finalize_Count() override;

		void Finalize_Fill() override {}

		void SetNbrCountFF(uint32_t i, uint32_t N_nbrs) override;
		void SetNbrCountFB(uint32_t i, uint32_t N_nbrs) override;
		void SetNbrCountFR(uint32_t i, uint32_t N_nbrs) override;
		void SetNbrCountRF(uint32_t i, uint32_t N_nbrs) override;
		void SetNbrCountRB(uint32_t i, uint32_t N_nbrs) override;
		void SetNbrCountRR(uint32_t i, uint32_t N_nbrs) override;

		void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t N_nbrs) override;
		void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t N_nbrs) override;
		void SetNbrListFR(uint32_t i, const uint32_t* js, uint32_t N_nbrs) override;
		void SetNbrListRF(uint32_t i, const uint32_t* js, uint32_t N_nbrs) override;
		void SetNbrListRB(uint32_t i, const uint32_t* js, uint32_t N_nbrs) override;
		void SetNbrListRR(uint32_t i, const uint32_t* js, uint32_t N_nbrs) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;
		NeighborSpan FR(uint32_t i) const override;
		NeighborSpan RF(uint32_t i) const override;
		NeighborSpan RB(uint32_t i) const override;
		NeighborSpan RR(uint32_t i) const override;

		uint32_t CountFF(uint32_t i) const override;
		uint32_t CountFB(uint32_t i) const override;
		uint32_t CountFR(uint32_t i) const override;
		uint32_t CountRF(uint32_t i) const override;
		uint32_t CountRB(uint32_t i) const override;
		uint32_t CountRR(uint32_t i) const override;

		Mem::ParticleClassMemUsage GetMemUsage() const override;

		bool RunCountQuery() const override { return true; }
		bool RunFillQuery() const override { return true; }

		uint32_t ChunkNumTotal() const noexcept override;
		ChunkedSELLNeighborSpan_Read FFChunk(uint32_t chunk) const noexcept override;
		ChunkedSELLNeighborSpan_Read FBChunk(uint32_t chunk) const noexcept override;
		ChunkedSELLNeighborSpan_Read FRChunk(uint32_t chunk) const noexcept override;
		ChunkedSELLNeighborSpan_Write FFChunkWrite(uint32_t chunk) noexcept override;
		ChunkedSELLNeighborSpan_Write FBChunkWrite(uint32_t chunk) noexcept override;
		ChunkedSELLNeighborSpan_Write FRChunkWrite(uint32_t chunk) noexcept override;

		void FinishFFChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) override;
		void FinishFBChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) override;
		void FinishFRChunkWrite(uint32_t chunk, const uint32_t* writtenCounts) override;

		void PrepareFluidCountWrite(bool allChunksWillBeWritten) override;
		void CommitFluidChunkCounts(
			uint32_t chunk,
			const ChunkedSELLNeighborCountResult& countsFF,
			const ChunkedSELLNeighborCountResult& countsFB,
			const ChunkedSELLNeighborCountResult& countsFR) override;

		bool NeedsChunkNeighborIdxRange() const noexcept override { return true; }

	private:

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;
		uint32_t N_R_ = 0;

		// compressed chunked SELL neighbor storage
		// (for fluid source particles)
		std::vector<SELLNeighborCount> nbrCountFF_; // per-particle neighbor count
		std::vector<SELLNeighborCount> nbrCountFB_;
		std::vector<SELLNeighborCount> nbrCountFR_;
		std::vector<SELLNeighborCount> nbrChunkMaxCountFF_; // per-chunk max neighbor count (pitch used for chunk length)
		std::vector<SELLNeighborCount> nbrChunkMaxCountFB_;
		std::vector<SELLNeighborCount> nbrChunkMaxCountFR_;
		std::vector<uint32_t> nbrChunkBaseIdxFF_; // per-chunk base index used for delta storage
		std::vector<uint32_t> nbrChunkBaseIdxFB_;
		std::vector<uint32_t> nbrChunkBaseIdxFR_;
		std::vector<SELLChunkFormat> nbrChunkFormatFF_; // per-chunk storage format
		std::vector<SELLChunkFormat> nbrChunkFormatFB_;
		std::vector<SELLChunkFormat> nbrChunkFormatFR_;
		std::vector<uint64_t> nbrChunkOffsetsFF_; // per-chunk starting index into the selected format buffer
		std::vector<uint64_t> nbrChunkOffsetsFB_;
		std::vector<uint64_t> nbrChunkOffsetsFR_;
		std::vector<uint16_t> nbrChunkDictSizeFF_; // per-chunk number of distinct neighbors of dict chunks, 0 for normal chunks
		std::vector<SELLDictChunkFormat> nbrChunkDictFormatFF_; // per-chunk dictionary format of dict chunks
		std::vector<uint16_t> nbrChunkDictLowFF_; // per-chunk low group size of DictDelta16x2 dictionaries
		std::vector<uint32_t> nbrChunkBaseIdx2FF_; // per-chunk base index of their high group
		std::vector<uint8_t*> nbrChunkDictFF_; // per-chunk dictionary of dict chunks, in nbrDictPagesFF_
		std::vector<std::unique_ptr<uint8_t[]>> nbrDictPagesFF_; // dictionaries of the dict chunks, in write blocks of the count-query
		std::vector<uint16_t> nbrList16FF_;
		std::vector<uint16_t> nbrList16FB_;
		std::vector<uint16_t> nbrList16FR_;
		std::vector<uint32_t> nbrList32FF_;
		std::vector<uint32_t> nbrList32FB_;
		std::vector<uint32_t> nbrList32FR_;

		// CSR neighbor storage for rigid source particles
		std::vector<SELLNeighborCount> nbrCountRF_;
		std::vector<SELLNeighborCount> nbrCountRB_;
		std::vector<SELLNeighborCount> nbrCountRR_;
		std::vector<uint64_t> nbrOffsetsRF_;
		std::vector<uint64_t> nbrOffsetsRB_;
		std::vector<uint64_t> nbrOffsetsRR_;
		std::vector<uint32_t> nbrListRF_;
		std::vector<uint32_t> nbrListRB_;
		std::vector<uint32_t> nbrListRR_;

		// thread-local neighbor scratch for contiguous access
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_PER_SCRATCH_INLINE_FF>> nbrScratchFF_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_PER_SCRATCH_INLINE_FB>> nbrScratchFB_;
		mutable tbb::enumerable_thread_specific<boost::container::small_vector<
			uint32_t, N_PER_SCRATCH_INLINE_FR>> nbrScratchFR_;

		// per-thread write blocks of the count-query, taken from the front of the dict pages
		struct DictWriteBlock
		{
			uint8_t* cursor = nullptr;
			uint8_t* end = nullptr;
		};
		tbb::enumerable_thread_specific<DictWriteBlock> dictWriteBlocksFF_;
		std::mutex dictPagesMutexFF_; // guards nbrDictPagesFF_ and N_dictBlockBytesFF_ while the count-query takes blocks
		uint64_t N_dictBlockBytesFF_ = 0; // bytes of the write blocks taken so far
		uint64_t N_dictPagesUsedMaxFF_ = 0; // most dict pages a count-query used since the last freeing check
		uint32_t N_countQueriesSinceDictPageFreeFF_ = 0; // count-queries since the last freeing check

		// misc
		static constexpr uint32_t chunkWidth = CPU::N_SIMD_WIDTH;
		static constexpr float extraAllocCapacityPct_ = 4.0f; // TODO: fine-tune
		static constexpr uint64_t N_DICT_WRITE_BLOCK_BYTES = 256ull * 1024ull;
		static constexpr uint64_t N_DICT_PAGE_BYTES = 8ull * 1024ull * 1024ull;
		static constexpr uint32_t N_DICT_PAGE_FREE_INTERVAL = 64; // count-queries a dict page has to stay unused before it is freed

		// helpers
		static uint32_t ChunkOfParticle(uint32_t i) noexcept;
		static uint32_t LaneOfParticle(uint32_t i) noexcept;
		static uint32_t ChunkBegin(uint32_t chunk) noexcept;
		static uint32_t ChunkEnd(uint32_t chunk, uint32_t N_F) noexcept;
		static uint32_t ChunkNumTotal(uint32_t N_F) noexcept;
		static SELLChunkFormat ChooseChunkFormat(uint32_t pitch, uint32_t minRef, uint32_t maxRef) noexcept;
		static bool SplitDictDelta16x2(
			const ChunkedSELLNeighborCountResult& counts, uint32_t& N_dictLow, uint32_t& baseIdx2) noexcept;
		static uint64_t ChunkStorageSize(uint32_t pitch) noexcept;

		uint8_t* AllocateDictFF(uint32_t N_bytes);

		ChunkedSELLNeighborSpan_Read MakeReadSpanFF(uint32_t chunk) const noexcept;
		ChunkedSELLNeighborSpan_Read MakeReadSpanFB(uint32_t chunk) const noexcept;
		ChunkedSELLNeighborSpan_Read MakeReadSpanFR(uint32_t chunk) const noexcept;
		ChunkedSELLNeighborSpan_Write MakeWriteSpanFF(uint32_t chunk) noexcept;
		ChunkedSELLNeighborSpan_Write MakeWriteSpanFB(uint32_t chunk) noexcept;
		ChunkedSELLNeighborSpan_Write MakeWriteSpanFR(uint32_t chunk) noexcept;

		void FinalizeChunkStorage(
			const std::vector<SELLNeighborCount>& nbrChunkMaxCount,
			const std::vector<SELLChunkFormat>& nbrChunkFormat,
			const std::vector<uint16_t>* nbrChunkDictSize, // nullptr without dict chunks
			std::vector<uint64_t>& nbrChunkOffsets,
			std::vector<uint16_t>& nbrList16,
			std::vector<uint32_t>& nbrList32) const;
	};

} // namespace Sim
