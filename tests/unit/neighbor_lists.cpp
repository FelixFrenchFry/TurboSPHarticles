// neighbor_lists.cpp

// external
#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

// internal
#include "source/simulation/CPU/backend/neighbor_lists.h"



namespace
{
	// categories follow the public interface order: FF, FB, FR, RF, RB, RR
	using NeighborData = std::array<std::vector<std::vector<uint32_t>>, 6>;
	constexpr std::array setCounts{
		&Sim::I_NeighborList::SetNbrCountFF, &Sim::I_NeighborList::SetNbrCountFB,
		&Sim::I_NeighborList::SetNbrCountFR, &Sim::I_NeighborList::SetNbrCountRF,
		&Sim::I_NeighborList::SetNbrCountRB, &Sim::I_NeighborList::SetNbrCountRR };
	constexpr std::array setLists{
		&Sim::I_NeighborList::SetNbrListFF, &Sim::I_NeighborList::SetNbrListFB,
		&Sim::I_NeighborList::SetNbrListFR, &Sim::I_NeighborList::SetNbrListRF,
		&Sim::I_NeighborList::SetNbrListRB, &Sim::I_NeighborList::SetNbrListRR };
	constexpr std::array readCounts{
		&Sim::I_NeighborList::CountFF, &Sim::I_NeighborList::CountFB,
		&Sim::I_NeighborList::CountFR, &Sim::I_NeighborList::CountRF,
		&Sim::I_NeighborList::CountRB, &Sim::I_NeighborList::CountRR };
	constexpr std::array readLists{
		&Sim::I_NeighborList::FF, &Sim::I_NeighborList::FB, &Sim::I_NeighborList::FR,
		&Sim::I_NeighborList::RF, &Sim::I_NeighborList::RB, &Sim::I_NeighborList::RR };
	constexpr std::array readChunks{
		&Sim::I_NeighborListSELL::FFChunk, &Sim::I_NeighborListSELL::FBChunk,
		&Sim::I_NeighborListSELL::FRChunk };
	constexpr std::array writeChunks{
		&Sim::I_NeighborListSELL::FFChunkWrite, &Sim::I_NeighborListSELL::FBChunkWrite,
		&Sim::I_NeighborListSELL::FRChunkWrite };
	constexpr std::array finishChunks{
		&Sim::I_NeighborListSELL::FinishFFChunkWrite, &Sim::I_NeighborListSELL::FinishFBChunkWrite,
		&Sim::I_NeighborListSELL::FinishFRChunkWrite };

	NeighborData MakeNeighborData(const uint32_t N_F, const uint32_t N_R)
	{
		NeighborData data;
		for (uint32_t category = 0; category < data.size(); category++)
		{
			data[category].resize(category < 3 ? N_F : N_R);
		}
		return data;
	}

	NeighborData MakeMixedNeighbors(const uint32_t N_F)
	{
		auto data = MakeNeighborData(N_F, 3);
		data[0][0] = { 5, 1, 3 };
		data[0][1] = { 0 };
		data[0][N_F - 1] = { 2, 4 };
		data[1][0] = { 7, 2 };
		data[1][2] = { 1 };
		data[1][N_F - 1] = { 6, 0, 3 };
		data[2][0] = { 2, 0, 1 };
		data[2][1] = { 1 };
		data[2][N_F - 1] = { 0, 2 };
		data[3] = { { 5, 1 }, {}, { 0, 3, 2 } };
		data[4] = { { 6 }, { 2, 0 }, {} };
		data[5] = { { 2, 1 }, {}, { 0 } };
		return data;
	}

	struct FluidChunkCounts
	{
		std::array<std::array<uint32_t, CPU::N_SIMD_WIDTH>, 3> counts{};
		std::array<uint32_t, 3> minRef{
			std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max(),
			std::numeric_limits<uint32_t>::max() };
		std::array<uint32_t, 3> maxRef{};
	};

	FluidChunkCounts CountFluidChunk(const NeighborData& data, const uint32_t chunk)
	{
		FluidChunkCounts result;
		const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
		for (uint32_t category = 0; category < 3; category++)
		{
			for (uint32_t lane = 0; lane < CPU::N_SIMD_WIDTH && begin + lane < data[category].size(); lane++)
			{
				const auto& neighbors = data[category][begin + lane];
				result.counts[category][lane] = static_cast<uint32_t>(neighbors.size());
				for (const uint32_t j : neighbors)
				{
					result.minRef[category] = std::min(result.minRef[category], j);
					result.maxRef[category] = std::max(result.maxRef[category], j);
				}
			}
		}
		return result;
	}

	// distinct neighbors of a chunk in first-use order
	std::vector<uint32_t> DistinctChunkNeighbors(const NeighborData& data, const uint32_t category, const uint32_t chunk)
	{
		std::vector<uint32_t> distinct;
		const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
		for (uint32_t lane = 0; lane < CPU::N_SIMD_WIDTH && begin + lane < data[category].size(); lane++)
		{
			for (const uint32_t j : data[category][begin + lane])
			{
				if (std::find(distinct.begin(), distinct.end(), j) == distinct.end()) { distinct.push_back(j); }
			}
		}
		return distinct;
	}

	// lanes of a chunk that have neighbor j
	uint32_t ChunkLaneMask(const NeighborData& data, const uint32_t category, const uint32_t chunk, const uint32_t j)
	{
		uint32_t laneMask = 0;
		const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
		for (uint32_t lane = 0; lane < CPU::N_SIMD_WIDTH && begin + lane < data[category].size(); lane++)
		{
			const auto& neighbors = data[category][begin + lane];
			if (std::find(neighbors.begin(), neighbors.end(), j) != neighbors.end()) { laneMask |= 1u << lane; }
		}
		return laneMask;
	}

	// tracking the distinct FF neighbors and their lanes lets the compressed lists write FF chunks as dict chunks
	void CommitFluidCounts(Sim::I_NeighborListSELL& lists, const NeighborData& data, const uint32_t chunk,
		const bool trackDistinctFF = false)
	{
		const auto counts = CountFluidChunk(data, chunk);
		const auto distinctFF = trackDistinctFF ? DistinctChunkNeighbors(data, 0, chunk) : std::vector<uint32_t>{};
		std::vector<uint32_t> laneMasksFF;
		for (const uint32_t j : distinctFF) { laneMasksFF.push_back(ChunkLaneMask(data, 0, chunk, j)); }
		lists.CommitFluidChunkCounts(chunk,
			{ counts.counts[0].data(), counts.minRef[0], counts.maxRef[0],
				static_cast<uint32_t>(distinctFF.size()), distinctFF.data(), laneMasksFF.data() },
			{ counts.counts[1].data(), counts.minRef[1], counts.maxRef[1] },
			{ counts.counts[2].data(), counts.minRef[2], counts.maxRef[2] });
	}

	void PrepareStorage(Sim::I_NeighborListSELL& lists, const NeighborData& data, const uint32_t N_B,
		const bool trackDistinctFF = false)
	{
		Sim::SimParams params;
		params.expctdAvgNbrCountHardMax_FF = 0;
		params.expctdAvgNbrCountHardMax_FB = 0;
		lists.Resize(static_cast<uint32_t>(data[0].size()), N_B, static_cast<uint32_t>(data[3].size()), params);
		lists.PrepareFluidCountWrite(true);
		for (uint32_t chunk = 0; chunk < lists.ChunkNumTotal(); chunk++)
		{
			CommitFluidCounts(lists, data, chunk, trackDistinctFF);
		}
		for (uint32_t category = 3; category < data.size(); category++)
		{
			for (uint32_t i = 0; i < data[category].size(); i++)
			{
				(lists.*setCounts[category])(i, static_cast<uint32_t>(data[category][i].size()));
			}
		}
		lists.Finalize_Count();
	}

	void FillStorage(Sim::I_NeighborListSELL& lists, const NeighborData& data, const bool direct)
	{
		for (uint32_t chunk = 0; chunk < lists.ChunkNumTotal(); chunk++)
		{
			const auto counts = CountFluidChunk(data, chunk);
			for (uint32_t category = 0; category < 3; category++)
			{
				auto view = (lists.*writeChunks[category])(chunk);
				if (view.pitch == 0) { continue; }
				// (dict chunks are written during the count, like in the compact cell list query)
				if (direct && view.dict) { continue; }
				// poison payloads so unwritten padding cannot pass through zero-initialized storage
				const uint64_t N_entries = static_cast<uint64_t>(view.pitch) * CPU::N_SIMD_WIDTH;
				if (view.data16) { std::fill_n(view.data16, N_entries, std::numeric_limits<uint16_t>::max()); }
				if (view.data32) { std::fill_n(view.data32, N_entries, std::numeric_limits<uint32_t>::max()); }
				for (uint32_t i = view.begin; i < view.end; i++)
				{
					const auto& neighbors = data[category][i];
					if (direct)
					{
						for (uint32_t k = 0; k < neighbors.size(); k++)
						{
							view.StoreNeighbor(i - view.begin, k, neighbors[k]);
						}
					}
					else
					{
						(lists.*setLists[category])(i, neighbors.data(), static_cast<uint32_t>(neighbors.size()));
					}
				}
				if (direct) { (lists.*finishChunks[category])(chunk, counts.counts[category].data()); }
			}
		}
		for (uint32_t category = 3; category < data.size(); category++)
		{
			for (uint32_t i = 0; i < data[category].size(); i++)
			{
				const auto& neighbors = data[category][i];
				(lists.*setLists[category])(i, neighbors.data(), static_cast<uint32_t>(neighbors.size()));
			}
		}
		lists.Finalize_Fill();
	}

	void ExpectNeighbors(const Sim::I_NeighborListSELL& lists, const NeighborData& data)
	{
		for (uint32_t category = 0; category < data.size(); category++)
		{
			SCOPED_TRACE(category);
			for (uint32_t i = 0; i < data[category].size(); i++)
			{
				SCOPED_TRACE(i);
				const auto& expected = data[category][i];
				EXPECT_EQ((lists.*readCounts[category])(i), expected.size());
				// consume each span before another retrieval can overwrite its scratch storage
				const auto actual = (lists.*readLists[category])(i);
				ASSERT_EQ(actual.N, expected.size());
				if (expected.empty()) { EXPECT_EQ(actual.data, nullptr); continue; }
				ASSERT_NE(actual.data, nullptr);
				// (dict chunks list the neighbors in dictionary order)
				if (category == 0 && lists.FFChunk(i / CPU::N_SIMD_WIDTH).dict)
				{
					std::vector<uint32_t> actualNeighbors(actual.data, actual.data + actual.N);
					std::vector<uint32_t> expectedNeighbors = expected;
					std::sort(actualNeighbors.begin(), actualNeighbors.end());
					std::sort(expectedNeighbors.begin(), expectedNeighbors.end());
					EXPECT_EQ(actualNeighbors, expectedNeighbors);
					continue;
				}
				for (uint32_t k = 0; k < expected.size(); k++)
				{
					SCOPED_TRACE(k);
					EXPECT_EQ(actual.data[k], expected[k]);
				}
			}
		}
	}

	void ExpectFluidChunks(const Sim::I_NeighborListSELL& lists, const NeighborData& data,
		const uint32_t N_B, const bool includeTail)
	{
		const uint32_t N_F = static_cast<uint32_t>(data[0].size());
		const std::array<uint32_t, 3> targetCounts{ N_F, N_B, static_cast<uint32_t>(data[3].size()) };
		for (uint32_t chunk = 0; chunk < lists.ChunkNumTotal(); chunk++)
		{
			SCOPED_TRACE(chunk);
			for (uint32_t category = 0; category < 3; category++)
			{
				SCOPED_TRACE(category);
				const auto view = (lists.*readChunks[category])(chunk);
				const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
				const uint32_t end = std::min(begin + CPU::N_SIMD_WIDTH, N_F);
				EXPECT_EQ(view.chunk, chunk);
				EXPECT_EQ(view.begin, begin);
				EXPECT_EQ(view.end, end);
				ASSERT_NE(view.counts, nullptr);
				uint32_t expectedPitch = 0;
				for (uint32_t lane = 0; lane < CPU::N_SIMD_WIDTH; lane++)
				{
					const uint32_t count = begin + lane < end
						? static_cast<uint32_t>(data[category][begin + lane].size()) : 0;
					EXPECT_EQ(view.counts[begin + lane], count);
					expectedPitch = std::max(expectedPitch, count);
				}
				ASSERT_EQ(view.pitch, expectedPitch);
				if (view.pitch == 0) { continue; }
				if (view.dict)
				{
					// every entry of a dict chunk is a distinct neighbor, with a lane mask of exactly the lanes that have it
					ASSERT_EQ(view.N_dict, DistinctChunkNeighbors(data, category, chunk).size());
					for (uint32_t dictIdx = 0; dictIdx < view.N_dict; dictIdx++)
					{
						SCOPED_TRACE(dictIdx);
						const uint32_t j = view.ReadDictNeighborIdx(dictIdx);
						EXPECT_LT(j, targetCounts[category]);
						EXPECT_EQ(view.ReadDictLaneMask(dictIdx), ChunkLaneMask(data, category, chunk, j));
					}
					continue;
				}
				ASSERT_TRUE(view.data16 || view.data32);
				for (uint32_t k = 0; k < view.pitch; k++)
				{
					SCOPED_TRACE(k);
					std::array<uint32_t, CPU::N_SIMD_WIDTH> loaded{};
					view.LoadNeighborIdx32xSIMD(k).store_unaligned(loaded.data());
					for (uint32_t lane = 0; lane < CPU::N_SIMD_WIDTH; lane++)
					{
						SCOPED_TRACE(lane);
						if (!includeTail && begin + lane >= end) { continue; }
						EXPECT_EQ(view.ReadNeighborIdx(lane, k), loaded[lane]);
						if (begin + lane < end && k < data[category][begin + lane].size())
						{
							EXPECT_EQ(loaded[lane], data[category][begin + lane][k]);
						}
						else
						{
							EXPECT_LT(loaded[lane], targetCounts[category]);
						}
					}
				}
			}
		}
	}

	class NeighborListStorageTests : public testing::TestWithParam<bool>
	{
	protected:
		void SetUp() override
		{
			if (GetParam()) { lists_ = std::make_unique<Sim::NeighborLists_ChunkedSELL_AdaptCompress_CPU>(); }
			else { lists_ = std::make_unique<Sim::NeighborLists_ChunkedSELL_Explicit_CPU>(); }
		}

		std::unique_ptr<Sim::I_NeighborListSELL> lists_;
	};

	// per-particle writes preserve unsorted indices, empty rows, and safe padding in active lanes
	TEST_P(NeighborListStorageTests, PreservesPerParticleCountsAndNeighbors)
	{
		for (const uint32_t N_F : { CPU::N_SIMD_WIDTH + 1, CPU::N_SIMD_WIDTH + 2 })
		{
			SCOPED_TRACE(N_F);
			const auto data = MakeMixedNeighbors(N_F);
			PrepareStorage(*lists_, data, 8);
			FillStorage(*lists_, data, false);

			ExpectNeighbors(*lists_, data);
			ExpectFluidChunks(*lists_, data, 8, false);
		}
	}

	// direct chunk writes are readable through the per-particle, scalar, and SIMD interfaces
	TEST_P(NeighborListStorageTests, PreservesDirectChunkWritesAndSafePadding)
	{
		const auto data = MakeMixedNeighbors(CPU::N_SIMD_WIDTH + 2);
		PrepareStorage(*lists_, data, 8);
		FillStorage(*lists_, data, true);

		ExpectNeighbors(*lists_, data);
		ExpectFluidChunks(*lists_, data, 8, true);
	}

	// absent populations and particles without neighbors do not require payload storage
	TEST_P(NeighborListStorageTests, HandlesEmptyPopulationsAndLists)
	{
		const std::array<std::array<uint32_t, 3>, 3> populations{{
			{ 0, 0, 0 }, { 0, 8, 3 }, { CPU::N_SIMD_WIDTH + 1, 0, 0 }
		}};
		for (const auto& counts : populations)
		{
			SCOPED_TRACE(counts[0]);
			const auto data = MakeNeighborData(counts[0], counts[2]);
			PrepareStorage(*lists_, data, counts[1]);
			FillStorage(*lists_, data, true);

			EXPECT_EQ(lists_->ChunkNumTotal(), counts[0] == 0 ? 0u : 2u);
			ExpectNeighbors(*lists_, data);
			ExpectFluidChunks(*lists_, data, counts[1], true);
		}
	}

	// empty categories remain empty beside populated categories sharing the same source particles
	TEST_P(NeighborListStorageTests, KeepsEmptyCategoriesIndependent)
	{
		auto data = MakeNeighborData(CPU::N_SIMD_WIDTH + 1, 0);
		data[0][0] = { 2, 1 };
		data[1].back() = { 4, 1 };
		PrepareStorage(*lists_, data, 8);
		FillStorage(*lists_, data, true);

		ExpectNeighbors(*lists_, data);
		ExpectFluidChunks(*lists_, data, 8, true);
		EXPECT_EQ(lists_->FRChunk(0).pitch, 0u);
		EXPECT_EQ(lists_->FRChunk(1).pitch, 0u);
	}

	// chunk bounds and counts remain correct immediately below, at, and above a SIMD width
	TEST_P(NeighborListStorageTests, HandlesParticleCountsAroundChunkBoundary)
	{
		const std::array<uint32_t, 3> counts{ CPU::N_SIMD_WIDTH - 1, CPU::N_SIMD_WIDTH, CPU::N_SIMD_WIDTH + 1 };
		for (uint32_t i = 0; i < counts.size(); i++)
		{
			SCOPED_TRACE(counts[i]);
			const auto data = MakeMixedNeighbors(counts[i]);
			PrepareStorage(*lists_, data, 8);
			FillStorage(*lists_, data, true);

			EXPECT_EQ(lists_->ChunkNumTotal(), i == 2 ? 2u : 1u);
			ExpectNeighbors(*lists_, data);
			ExpectFluidChunks(*lists_, data, 8, true);
		}
	}

	// compression depends on the index span rather than the absolute index magnitude
	TEST_P(NeighborListStorageTests, PreservesIndicesAtCompressionBoundaries)
	{
		const std::array<uint32_t, 4> spans{ 255, 256, 65'535, 65'536 };
		const std::array<Sim::SELLChunkFormat, 4> formats{
			Sim::SELLChunkFormat::Delta16, Sim::SELLChunkFormat::Delta16,
			Sim::SELLChunkFormat::Delta16, Sim::SELLChunkFormat::Abs32 };
		for (uint32_t i = 0; i < spans.size(); i++)
		{
			SCOPED_TRACE(spans[i]);
			for (const bool direct : { false, true })
			{
				SCOPED_TRACE(direct);
				auto data = MakeNeighborData(2, 0);
				data[1][0] = { 70'000 + spans[i], 70'000, 70'001 };
				data[1][1] = { 70'001 };
				PrepareStorage(*lists_, data, 140'000);
				FillStorage(*lists_, data, direct);

				EXPECT_EQ(lists_->FBChunk(0).format, GetParam() ? formats[i] : Sim::SELLChunkFormat::Abs32);
				ExpectNeighbors(*lists_, data);
				ExpectFluidChunks(*lists_, data, 140'000, direct);
			}
		}
	}

	// separate offsets are required when multiple chunks reuse each compressed payload buffer
	TEST_P(NeighborListStorageTests, KeepsMixedFormatChunksIndependent)
	{
		auto data = MakeNeighborData(4 * CPU::N_SIMD_WIDTH + 1, 0);
		const std::array<uint32_t, 5> spans{ 255, 256, 65'536, 255, 65'535 };
		const std::array<Sim::SELLChunkFormat, 5> formats{
			Sim::SELLChunkFormat::Delta16, Sim::SELLChunkFormat::Delta16, Sim::SELLChunkFormat::Abs32,
			Sim::SELLChunkFormat::Delta16, Sim::SELLChunkFormat::Delta16 };
		for (uint32_t chunk = 0; chunk < spans.size(); chunk++)
		{
			const uint32_t base = 17 + 11 * chunk;
			data[1][chunk * CPU::N_SIMD_WIDTH] = { base + spans[chunk], base, base + 13 };
		}
		PrepareStorage(*lists_, data, 100'000);
		FillStorage(*lists_, data, true);

		ExpectNeighbors(*lists_, data);
		ExpectFluidChunks(*lists_, data, 100'000, true);
		for (uint32_t chunk = 0; chunk < spans.size(); chunk++)
		{
			SCOPED_TRACE(chunk);
			EXPECT_EQ(lists_->FBChunk(chunk).format, GetParam() ? formats[chunk] : Sim::SELLChunkFormat::Abs32);
		}
	}

	// resizing and rebuilding must replace old counts, payloads, and chunk metadata
	TEST_P(NeighborListStorageTests, RebuildsAfterGrowthShrinkageAndEmptyState)
	{
		for (const uint32_t N_F : { CPU::N_SIMD_WIDTH - 1, 2 * CPU::N_SIMD_WIDTH + 1,
			CPU::N_SIMD_WIDTH + 1, 0u, CPU::N_SIMD_WIDTH })
		{
			SCOPED_TRACE(N_F);
			const auto data = N_F == 0 ? MakeNeighborData(0, 0) : MakeMixedNeighbors(N_F);
			PrepareStorage(*lists_, data, 8);
			FillStorage(*lists_, data, true);

			EXPECT_EQ(lists_->ChunkNumTotal(), N_F == 0 ? 0u : (N_F - 1) / CPU::N_SIMD_WIDTH + 1);
			ExpectNeighbors(*lists_, data);
			ExpectFluidChunks(*lists_, data, 8, true);
		}
	}

	// a partial count pass clears chunks it does not visit, while rigid counts are rewritten explicitly
	// (with and without dict chunks)
	TEST_P(NeighborListStorageTests, ClearsUnvisitedChunksBeforePartialRebuild)
	{
		for (const bool trackDistinctFF : { false, true })
		{
			SCOPED_TRACE(trackDistinctFF);
			const uint32_t N_F = 2 * CPU::N_SIMD_WIDTH + 1;
			const auto original = MakeMixedNeighbors(N_F);
			PrepareStorage(*lists_, original, 8, trackDistinctFF);
			FillStorage(*lists_, original, true);
			auto data = MakeNeighborData(N_F, 3);
			data[0][CPU::N_SIMD_WIDTH] = { 1, 0 };
			data[1][CPU::N_SIMD_WIDTH] = { 2 };
			data[2][CPU::N_SIMD_WIDTH] = { 1 };

			lists_->PrepareFluidCountWrite(false);
			CommitFluidCounts(*lists_, data, 1, trackDistinctFF);
			for (uint32_t category = 3; category < data.size(); category++)
			{
				for (uint32_t i = 0; i < data[category].size(); i++) { (lists_.get()->*setCounts[category])(i, 0); }
			}
			lists_->Finalize_Count();
			FillStorage(*lists_, data, true);

			ExpectNeighbors(*lists_, data);
			ExpectFluidChunks(*lists_, data, 8, true);
			EXPECT_EQ(lists_->FFChunk(1).dict != nullptr, GetParam() && trackDistinctFF);
		}
	}

	// rebuilding the same particles can switch payload formats and then remove all neighbors
	TEST_P(NeighborListStorageTests, ReplacesCompressionFormatsAndClearsLists)
	{
		const std::array<uint32_t, 4> spans{ 65'536, 255, 65'535, 0 };
		const std::array<Sim::SELLChunkFormat, 4> formats{
			Sim::SELLChunkFormat::Abs32, Sim::SELLChunkFormat::Delta16,
			Sim::SELLChunkFormat::Delta16, Sim::SELLChunkFormat::Abs32 };
		for (uint32_t i = 0; i < spans.size(); i++)
		{
			const uint32_t span = spans[i];
			SCOPED_TRACE(span);
			auto data = MakeNeighborData(CPU::N_SIMD_WIDTH + 1, 0);
			if (span != 0) { data[1][0] = { 17 + span, 17 }; data[1].back() = { 18 }; }
			PrepareStorage(*lists_, data, 100'000);
			FillStorage(*lists_, data, true);

			EXPECT_EQ(lists_->FBChunk(0).format, GetParam() ? formats[i] : Sim::SELLChunkFormat::Abs32);
			ExpectNeighbors(*lists_, data);
			ExpectFluidChunks(*lists_, data, 100'000, true);
		}
	}

	// writes larger than the allocated pitch fail before touching the payload
	TEST_P(NeighborListStorageTests, RejectsWritesLargerThanChunkPitch)
	{
		auto data = MakeNeighborData(1, 1);
		for (uint32_t category = 0; category < 3; category++) { data[category][0] = { 0 }; }
		PrepareStorage(*lists_, data, 1);
		FillStorage(*lists_, data, true);
		const std::array<uint32_t, 2> neighbors{ 0, 0 };

		for (uint32_t category = 0; category < 3; category++)
		{
			SCOPED_TRACE(category);
			EXPECT_THROW((lists_.get()->*setLists[category])(0, neighbors.data(), 2), std::runtime_error);
		}
		ExpectNeighbors(*lists_, data);
	}

	// dict chunks are readable through the per-particle and dictionary interfaces, including tail lanes
	TEST_P(NeighborListStorageTests, PreservesDictChunkWrites)
	{
		const auto data = MakeMixedNeighbors(CPU::N_SIMD_WIDTH + 2);
		PrepareStorage(*lists_, data, 8, true);
		FillStorage(*lists_, data, true);

		ExpectNeighbors(*lists_, data);
		ExpectFluidChunks(*lists_, data, 8, true);
		for (uint32_t chunk = 0; chunk < lists_->ChunkNumTotal(); chunk++)
		{
			SCOPED_TRACE(chunk);
			EXPECT_EQ(lists_->FFChunk(chunk).dict != nullptr, GetParam());
		}
	}

	// dict entries take a second base for neighbors beyond uint16 reach of the lowest one,
	// so only spans that two uint16 windows cannot cover store absolute indices
	TEST_P(NeighborListStorageTests, PreservesDictChunkIndicesAtCompressionBoundaries)
	{
		const std::array<std::vector<uint32_t>, 3> neighbors{ {
			{ 17 + 65'535, 17, 18 },
			{ 17 + 65'536, 17, 18, 17 + 65'536 + 65'535 },
			{ 17 + 65'536, 17, 18, 17 + 65'536 + 65'536 } } };
		const std::array<Sim::SELLChunkFormat, 3> formats{
			Sim::SELLChunkFormat::Delta16, Sim::SELLChunkFormat::Abs32, Sim::SELLChunkFormat::Abs32 };
		const std::array<Sim::SELLDictChunkFormat, 3> dictFormats{
			Sim::SELLDictChunkFormat::DictDelta16x2, Sim::SELLDictChunkFormat::DictDelta16x2,
			Sim::SELLDictChunkFormat::DictAbs32 };
		for (uint32_t i = 0; i < neighbors.size(); i++)
		{
			SCOPED_TRACE(i);
			auto data = MakeNeighborData(140'000, 0);
			data[0][0] = neighbors[i];
			data[0][1] = { 18 };
			PrepareStorage(*lists_, data, 8, true);
			FillStorage(*lists_, data, true);

			ExpectNeighbors(*lists_, data);
			ExpectFluidChunks(*lists_, data, 8, true);
			const auto view = lists_->FFChunk(0);
			ASSERT_EQ(view.dict != nullptr, GetParam());
			if (GetParam())
			{
				EXPECT_EQ(view.format, formats[i]);
				EXPECT_EQ(view.dictFormat, dictFormats[i]);
			}
		}
	}

	// dict chunks list neighbors in candidate order, except the DictDelta16x2 high group (reversed)
	TEST_P(NeighborListStorageTests, ListsDictChunkNeighborsInDictionaryOrder)
	{
		auto data = MakeNeighborData(6, 0);
		data[0][0] = { 5, 1, 3 };
		data[0][1] = { 3, 0, 5 };
		PrepareStorage(*lists_, data, 8, true);
		FillStorage(*lists_, data, true);

		// (normal chunks keep each lane's own order)
		const std::vector<uint32_t> expected = GetParam()
			? std::vector<uint32_t>{ 5, 3, 0 } : data[0][1];
		const auto actual = lists_->FF(1);
		EXPECT_EQ(std::vector<uint32_t>(actual.data, actual.data + actual.N), expected);
	}

	// the count writes dict chunks in candidate order across write blocks, over growing and shrinking rebuilds
	TEST_P(NeighborListStorageTests, WritesDictChunksDuringCount)
	{
		const auto MakeDictNeighbors = [](const uint32_t N_nbrs)
		{
			auto data = MakeNeighborData(64'003, 0); // keep every neighbor index in one uint16 delta range
			const uint32_t N_F = static_cast<uint32_t>(data[0].size());
			for (uint32_t i = 0; i < N_F; i++)
			{
				for (uint32_t k = 0; k < N_nbrs; k++) { data[0][i].push_back((i + 3 * k) % N_F); }
			}
			data[1][0] = { 2, 5 };
			return data;
		};

		for (const uint32_t N_nbrs : { 3u, 6u, 3u })
		{
			SCOPED_TRACE(N_nbrs);
			const auto data = MakeDictNeighbors(N_nbrs);
			PrepareStorage(*lists_, data, 8, true);
			FillStorage(*lists_, data, true);

			ExpectNeighbors(*lists_, data);
			ExpectFluidChunks(*lists_, data, 8, true);
			for (uint32_t chunk = 0; chunk < lists_->ChunkNumTotal(); chunk++)
			{
				SCOPED_TRACE(chunk);
				const auto view = lists_->FFChunk(chunk);
				ASSERT_EQ(view.dict != nullptr, GetParam());
				const auto distinct = DistinctChunkNeighbors(data, 0, chunk);
				for (uint32_t dictIdx = 0; dictIdx < view.N_dict; dictIdx++)
				{
					EXPECT_EQ(view.ReadDictNeighborIdx(dictIdx), distinct[dictIdx]);
				}
			}
		}
	}

	// dict pages that the following counts leave unused are freed, here after one large count over 8 MB
	// (the pages are checked every 64 counts, so 127 small counts span a whole check window without it)
	TEST_P(NeighborListStorageTests, FreesDictPagesUnusedByLaterCounts)
	{
		if (!GetParam()) { return; }
		const uint32_t N_chunks = 12'000;
		const auto RebuildDictChunks = [&](const uint32_t N_dict)
		{
			Sim::SimParams params;
			params.expctdAvgNbrCountHardMax_FF = 0;
			params.expctdAvgNbrCountHardMax_FB = 0;
			lists_->Resize(N_chunks * CPU::N_SIMD_WIDTH, 0, 0, params);
			lists_->PrepareFluidCountWrite(true);
			// (the first lane of every chunk has the same N_dict neighbors)
			std::vector<uint32_t> distinct(N_dict);
			for (uint32_t n = 0; n < N_dict; n++) { distinct[n] = n; }
			const std::vector<uint32_t> laneMasks(N_dict, 1u);
			std::array<uint32_t, CPU::N_SIMD_WIDTH> countsFF{};
			countsFF[0] = N_dict;
			const std::array<uint32_t, CPU::N_SIMD_WIDTH> noCounts{};
			for (uint32_t chunk = 0; chunk < N_chunks; chunk++)
			{
				lists_->CommitFluidChunkCounts(chunk,
					{ countsFF.data(), 0, N_dict - 1, N_dict, distinct.data(), laneMasks.data() },
					{ noCounts.data(), std::numeric_limits<uint32_t>::max(), 0 },
					{ noCounts.data(), std::numeric_limits<uint32_t>::max(), 0 });
			}
			lists_->Finalize_Count();
		};

		RebuildDictChunks(Sim::N_SELL_DICT_SIZE_MAX);
		const uint64_t N_allocBytesLarge = lists_->GetMemUsage().fluid.N_allocBytes;
		for (uint32_t count = 0; count < 127; count++) { RebuildDictChunks(1); }
		EXPECT_LT(lists_->GetMemUsage().fluid.N_allocBytes, N_allocBytesLarge);
		EXPECT_EQ(lists_->FF(0).N, 1u);
	}

	// chunks with more distinct neighbors than a dict chunk holds stay normal chunks
	TEST_P(NeighborListStorageTests, LimitsDictChunksToDictSize)
	{
		for (const uint32_t N_distinct : { Sim::N_SELL_DICT_SIZE_MAX, Sim::N_SELL_DICT_SIZE_MAX + 1 })
		{
			SCOPED_TRACE(N_distinct);
			auto data = MakeNeighborData(N_distinct + 1, 0);
			for (uint32_t j = 1; j <= N_distinct; j++) { data[0][0].push_back(j); }
			// the second lane shares every neighbor, so it adds no dict entries
			data[0][1].assign(data[0][0].rbegin(), data[0][0].rend());
			PrepareStorage(*lists_, data, 8, true);
			FillStorage(*lists_, data, true);

			ExpectNeighbors(*lists_, data);
			ExpectFluidChunks(*lists_, data, 8, true);
			EXPECT_EQ(lists_->FFChunk(0).dict != nullptr, GetParam() && N_distinct <= Sim::N_SELL_DICT_SIZE_MAX);
		}
	}

	// per-particle writes reject dict chunks, which only the count writes
	TEST_P(NeighborListStorageTests, RejectsPerParticleWritesIntoDictChunks)
	{
		const auto data = MakeMixedNeighbors(CPU::N_SIMD_WIDTH + 1);
		PrepareStorage(*lists_, data, 8, true);
		const auto& neighbors = data[0][0];

		if (GetParam())
		{
			EXPECT_THROW(lists_->SetNbrListFF(0, neighbors.data(), static_cast<uint32_t>(neighbors.size())),
				std::runtime_error);
		}
		else
		{
			EXPECT_NO_THROW(lists_->SetNbrListFF(0, neighbors.data(), static_cast<uint32_t>(neighbors.size())));
		}
	}

	#if defined(_DEBUG)
	// debug guards reject counts that cannot fit in the uint16 neighbor-count storage
	TEST_P(NeighborListStorageTests, RejectsNeighborCountOverflow)
	{
		PrepareStorage(*lists_, MakeNeighborData(1, 1), 1);
		for (const auto setCount : setCounts)
		{
			EXPECT_THROW((lists_.get()->*setCount)(0, 65'536), std::runtime_error);
		}
		auto data = MakeNeighborData(1, 1);
		auto counts = CountFluidChunk(data, 0);
		counts.counts[0][0] = 65'536;
		EXPECT_THROW(lists_->CommitFluidChunkCounts(0,
			{ counts.counts[0].data(), 0, 0 }, { counts.counts[1].data(), 0, 0 },
			{ counts.counts[2].data(), 0, 0 }), std::runtime_error);
	}

	// debug guards catch underfilled lists through both per-particle and direct chunk APIs
	TEST_P(NeighborListStorageTests, RejectsCountAndFillMismatches)
	{
		const auto data = MakeMixedNeighbors(CPU::N_SIMD_WIDTH + 1);
		PrepareStorage(*lists_, data, 8);
		FillStorage(*lists_, data, true);
		for (const auto setList : setLists)
		{
			EXPECT_THROW((lists_.get()->*setList)(0, nullptr, 0), std::runtime_error);
		}
		const std::array<uint32_t, CPU::N_SIMD_WIDTH> writtenCounts{};
		for (const auto finishChunk : finishChunks)
		{
			EXPECT_THROW((lists_.get()->*finishChunk)(0, writtenCounts.data()), std::runtime_error);
		}
		ExpectNeighbors(*lists_, data);
	}
	#endif

	INSTANTIATE_TEST_SUITE_P(ExplicitAndCompressed, NeighborListStorageTests, testing::Bool());
}
