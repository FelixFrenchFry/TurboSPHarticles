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

	void CommitFluidCounts(Sim::I_NeighborListSELL& lists, const NeighborData& data, const uint32_t chunk)
	{
		const auto counts = CountFluidChunk(data, chunk);
		lists.CommitFluidChunkCounts(chunk,
			{ counts.counts[0].data(), counts.minRef[0], counts.maxRef[0] },
			{ counts.counts[1].data(), counts.minRef[1], counts.maxRef[1] },
			{ counts.counts[2].data(), counts.minRef[2], counts.maxRef[2] });
	}

	void PrepareStorage(Sim::I_NeighborListSELL& lists, const NeighborData& data, const uint32_t N_B)
	{
		Sim::SimParams params;
		params.expctdAvgNbrCountHardMax_FF = 0;
		params.expctdAvgNbrCountHardMax_FB = 0;
		lists.Resize(static_cast<uint32_t>(data[0].size()), N_B, static_cast<uint32_t>(data[3].size()), params);
		lists.PrepareFluidCountWrite(true);
		for (uint32_t chunk = 0; chunk < lists.ChunkNumTotal(); chunk++)
		{
			CommitFluidCounts(lists, data, chunk);
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
				const auto view = (lists.*writeChunks[category])(chunk);
				if (view.pitch == 0) { continue; }
				// poison payloads so unwritten padding cannot pass through zero-initialized storage
				const uint64_t N_entries = static_cast<uint64_t>(view.pitch) * CPU::N_SIMD_WIDTH;
				if (view.data8) { std::fill_n(view.data8, N_entries, std::numeric_limits<uint8_t>::max()); }
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
				ASSERT_TRUE(view.data8 || view.data16 || view.data32);
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
			Sim::SELLChunkFormat::Delta8, Sim::SELLChunkFormat::Delta16,
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

				EXPECT_EQ(lists_->FBChunk(0).format, GetParam() ? formats[i] : Sim::SELLChunkFormat::Explicit32);
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
			Sim::SELLChunkFormat::Delta8, Sim::SELLChunkFormat::Delta16, Sim::SELLChunkFormat::Abs32,
			Sim::SELLChunkFormat::Delta8, Sim::SELLChunkFormat::Delta16 };
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
			EXPECT_EQ(lists_->FBChunk(chunk).format, GetParam() ? formats[chunk] : Sim::SELLChunkFormat::Explicit32);
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
	TEST_P(NeighborListStorageTests, ClearsUnvisitedChunksBeforePartialRebuild)
	{
		const uint32_t N_F = 2 * CPU::N_SIMD_WIDTH + 1;
		const auto original = MakeMixedNeighbors(N_F);
		PrepareStorage(*lists_, original, 8);
		FillStorage(*lists_, original, true);
		auto data = MakeNeighborData(N_F, 3);
		data[0][CPU::N_SIMD_WIDTH] = { 1, 0 };
		data[1][CPU::N_SIMD_WIDTH] = { 2 };
		data[2][CPU::N_SIMD_WIDTH] = { 1 };

		lists_->PrepareFluidCountWrite(false);
		CommitFluidCounts(*lists_, data, 1);
		for (uint32_t category = 3; category < data.size(); category++)
		{
			for (uint32_t i = 0; i < data[category].size(); i++) { (lists_.get()->*setCounts[category])(i, 0); }
		}
		lists_->Finalize_Count();
		FillStorage(*lists_, data, true);

		ExpectNeighbors(*lists_, data);
		ExpectFluidChunks(*lists_, data, 8, true);
	}

	// rebuilding the same particles can switch payload formats and then remove all neighbors
	TEST_P(NeighborListStorageTests, ReplacesCompressionFormatsAndClearsLists)
	{
		const std::array<uint32_t, 4> spans{ 65'536, 255, 65'535, 0 };
		const std::array<Sim::SELLChunkFormat, 4> formats{
			Sim::SELLChunkFormat::Abs32, Sim::SELLChunkFormat::Delta8,
			Sim::SELLChunkFormat::Delta16, Sim::SELLChunkFormat::Abs32 };
		for (uint32_t i = 0; i < spans.size(); i++)
		{
			const uint32_t span = spans[i];
			SCOPED_TRACE(span);
			auto data = MakeNeighborData(CPU::N_SIMD_WIDTH + 1, 0);
			if (span != 0) { data[1][0] = { 17 + span, 17 }; data[1].back() = { 18 }; }
			PrepareStorage(*lists_, data, 100'000);
			FillStorage(*lists_, data, true);

			EXPECT_EQ(lists_->FBChunk(0).format, GetParam() ? formats[i] : Sim::SELLChunkFormat::Explicit32);
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
