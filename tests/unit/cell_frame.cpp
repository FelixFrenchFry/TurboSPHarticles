// cell_frame.cpp

// external
#include <array>
#include <cstdint>
#include <gtest/gtest.h>

// internal
#include "source/simulation/core/cell_frame.h"



namespace
{
	// builds inclusive dimensions from signed bounds, including a single-cell axis
	TEST(CellCoordFrameTests, BuildsInclusiveSignedBounds)
	{
		Sim::CellCoordFrame frame;

		ASSERT_TRUE(Sim::TryBuildCellCoordFrame(-3, 5, -7, 1, 6, -7, frame));

		EXPECT_EQ(frame.cellOrigX, -3);
		EXPECT_EQ(frame.cellOrigY, 5);
		EXPECT_EQ(frame.cellOrigZ, -7);
		EXPECT_EQ(frame.N_dimX, 5u);
		EXPECT_EQ(frame.N_dimY, 2u);
		EXPECT_EQ(frame.N_dimZ, 1u);
	}

	// accepts the full 21-bit Morton range independently on each axis
	TEST(CellCoordFrameTests, AcceptsMaximumAxisExtent)
	{
		for (int axis = 0; axis < 3; axis++)
		{
			SCOPED_TRACE(axis);
			std::array<int64_t, 3> cellMax{ 0, 0, 0 };
			cellMax[axis] = 2'097'151;
			Sim::CellCoordFrame frame;

			ASSERT_TRUE(Sim::TryBuildCellCoordFrame(0, 0, 0,
				cellMax[0], cellMax[1], cellMax[2], frame));

			std::array<uint32_t, 3> expected{ 1, 1, 1 };
			expected[axis] = 2'097'152;
			EXPECT_EQ(frame.N_dimX, expected[0]);
			EXPECT_EQ(frame.N_dimY, expected[1]);
			EXPECT_EQ(frame.N_dimZ, expected[2]);
		}
	}

	using CellCoordBounds = std::array<int64_t, 6>;
	class CellCoordFrameInvalidBoundsTests : public testing::TestWithParam<CellCoordBounds> {};

	// rejects inverted, unrepresentable, or oversized bounds without replacing the frame
	TEST_P(CellCoordFrameInvalidBoundsTests, PreservesFrameOnInvalidBounds)
	{
		const auto& bounds = GetParam();
		Sim::CellCoordFrame frame{ 4, 5, 6, 7, 8, 9 };

		EXPECT_FALSE(Sim::TryBuildCellCoordFrame(bounds[0], bounds[1], bounds[2],
			bounds[3], bounds[4], bounds[5], frame));

		EXPECT_EQ(frame.cellOrigX, 4);
		EXPECT_EQ(frame.cellOrigY, 5);
		EXPECT_EQ(frame.cellOrigZ, 6);
		EXPECT_EQ(frame.N_dimX, 7u);
		EXPECT_EQ(frame.N_dimY, 8u);
		EXPECT_EQ(frame.N_dimZ, 9u);
	}

	INSTANTIATE_TEST_SUITE_P(InvalidBounds, CellCoordFrameInvalidBoundsTests,
		testing::Values(
			CellCoordBounds{ 1, 0, 0, 0, 0, 0 },
			CellCoordBounds{ 0, 1, 0, 0, 0, 0 },
			CellCoordBounds{ 0, 0, 1, 0, 0, 0 },
			CellCoordBounds{ -2'147'483'649ll, 0, 0, -2'147'483'649ll, 0, 0 },
			CellCoordBounds{ 0, -2'147'483'649ll, 0, 0, -2'147'483'649ll, 0 },
			CellCoordBounds{ 0, 0, -2'147'483'649ll, 0, 0, -2'147'483'649ll },
			CellCoordBounds{ 2'147'483'648ll, 0, 0, 2'147'483'648ll, 0, 0 },
			CellCoordBounds{ 0, 2'147'483'648ll, 0, 0, 2'147'483'648ll, 0 },
			CellCoordBounds{ 0, 0, 2'147'483'648ll, 0, 0, 2'147'483'648ll },
			CellCoordBounds{ 0, 0, 0, 2'097'152, 0, 0 },
			CellCoordBounds{ 0, 0, 0, 0, 2'097'152, 0 },
			CellCoordBounds{ 0, 0, 0, 0, 0, 2'097'152 }));
}
