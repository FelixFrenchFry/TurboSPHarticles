// scratch_storage.cpp

// external
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

// internal
#include "source/base/helpers.h"



namespace
{
	// rounds fractional allocation padding upward and ignores nonpositive percentages
	TEST(ScratchStorageTests, RoundsExtraCapacityUp)
	{
		EXPECT_EQ(Base::ExtraPercent(3, 25.0f), 1u);
		EXPECT_EQ(Base::ExtraPercent(8, 25.0f), 2u);
		EXPECT_EQ(Base::ExtraPercent(0, 25.0f), 0u);
		EXPECT_EQ(Base::ExtraPercent(8, 0.0f), 0u);
		EXPECT_EQ(Base::ExtraPercent(8, -25.0f), 0u);
	}

	// preserves contents and storage when the existing capacity satisfies a reservation
	TEST(ScratchStorageTests, ReusesSufficientCapacity)
	{
		std::vector<int> values{ 1, 2, 3 };
		values.reserve(16);
		const auto capacity = values.capacity();
		const auto* data = values.data();
		const std::vector<int> expected{ 1, 2, 3 };

		Base::ReserveScratchVector(values, capacity, 25.0f);

		EXPECT_EQ(values, expected);
		EXPECT_EQ(values.capacity(), capacity);
		EXPECT_EQ(values.data(), data);
	}

	// discards scratch contents on growth and reserves the requested percentage padding
	TEST(ScratchStorageTests, ClearsContentsWhenCapacityGrows)
	{
		std::vector<int> values{ 1, 2, 3 };
		const uint64_t N_requested = static_cast<uint64_t>(values.capacity()) + 1;
		const uint64_t N_expected = N_requested + (N_requested + 3) / 4;

		Base::ReserveScratchVector(values, N_requested, 25.0f);

		EXPECT_TRUE(values.empty());
		EXPECT_GE(values.capacity(), N_expected);
	}

	// rejects overflowing padded sizes before clearing existing scratch contents
	TEST(ScratchStorageTests, RejectsCapacityOverflow)
	{
		std::vector<int> values{ 1, 2, 3 };
		const std::vector<int> expected{ 1, 2, 3 };
		const auto capacity = values.capacity();

		EXPECT_THROW(Base::ReserveScratchVector(values,
			std::numeric_limits<uint64_t>::max(), 100.0f), std::length_error);

		EXPECT_EQ(values, expected);
		EXPECT_EQ(values.capacity(), capacity);
	}

	// rejects reservations above the container limit before discarding existing contents
	TEST(ScratchStorageTests, RejectsSizesAboveContainerLimit)
	{
		std::vector<int> values{ 1, 2, 3 };
		const std::vector<int> expected{ 1, 2, 3 };
		const uint64_t N_requested = static_cast<uint64_t>(values.max_size()) + 1;

		EXPECT_THROW(Base::ReserveScratchVector(values, N_requested), std::length_error);

		EXPECT_EQ(values, expected);
	}

	// resizing within capacity retains existing elements and initializes newly added elements
	TEST(ScratchStorageTests, ResizesWithoutReallocating)
	{
		std::vector<int> values{ 1, 2, 3 };
		values.reserve(16);
		const auto* data = values.data();
		const std::vector<int> expected{ 1, 2, 3, 0, 0 };

		Base::ResizeScratchVector(values, 5);

		EXPECT_EQ(values, expected);
		EXPECT_EQ(values.data(), data);
	}

	// a grow-only resize leaves a larger scratch buffer and its contents intact
	TEST(ScratchStorageTests, GrowOnlyResizeDoesNotShrink)
	{
		std::vector<int> values{ 1, 2, 3 };
		const std::vector<int> expected{ 1, 2, 3 };
		const auto* data = values.data();

		Base::ResizeGrowScratchVector(values, 2);

		EXPECT_EQ(values, expected);
		EXPECT_EQ(values.data(), data);
	}
}
