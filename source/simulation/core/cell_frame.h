// cell_frame.h

#pragma once

// external
#include <cstdint>
#include <limits>



namespace Sim
{
	// local cell-frame metadata shared by compact neighbor-search implementations
	struct CellCoordFrame
	{
		int32_t cellOrigX = 0;
		int32_t cellOrigY = 0;
		int32_t cellOrigZ = 0;
		uint32_t N_dimX = 0;
		uint32_t N_dimY = 0;
		uint32_t N_dimZ = 0;
	};

	// markers for an absent cell ID, index or reference
	// (used by both the CPU and GPU backend)
	inline constexpr uint32_t EMPTY_32 = std::numeric_limits<uint32_t>::max();
	inline constexpr uint64_t EMPTY_64 = std::numeric_limits<uint64_t>::max();

	// the Morton encoder spreads 21 bits per axis into the 64-bit cell ID
	inline constexpr uint32_t CellCoordFrameMaxAxisCells = (1u << 21);

	// builds a frame from inclusive global cell-coordinate bounds
	inline bool TryBuildCellCoordFrame(
		const int64_t cellMinX, const int64_t cellMinY, const int64_t cellMinZ,
		const int64_t cellMaxX, const int64_t cellMaxY, const int64_t cellMaxZ,
		CellCoordFrame& frame) noexcept
	{
		const auto isAxisValid = [](const int64_t cellMin, const int64_t cellMax) noexcept
			{
				if (cellMin > cellMax) { return false; }
				if (cellMin < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
					cellMax > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
				{
					return false;
				}

				return cellMax - cellMin + 1 <=
					static_cast<int64_t>(CellCoordFrameMaxAxisCells);
			};

		if (!isAxisValid(cellMinX, cellMaxX) ||
			!isAxisValid(cellMinY, cellMaxY) ||
			!isAxisValid(cellMinZ, cellMaxZ))
		{
			return false;
		}

		frame = CellCoordFrame{
			static_cast<int32_t>(cellMinX),
			static_cast<int32_t>(cellMinY),
			static_cast<int32_t>(cellMinZ),
			static_cast<uint32_t>(cellMaxX - cellMinX + 1),
			static_cast<uint32_t>(cellMaxY - cellMinY + 1),
			static_cast<uint32_t>(cellMaxZ - cellMinZ + 1)
		};
		return true;
	}

} // namespace Sim
