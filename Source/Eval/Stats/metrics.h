// metrics.h
//
// TODO: comment

#pragma once

// external
#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>



namespace Stats
{
	enum class StatGroup : uint8_t
	{
		Particles,
		Density,
		Velocity,

		Count
	};

	enum class StatMetric : uint8_t
	{
		// particles and neighbors
		NumF,				// total number of simulated fluid particles
		NumB,				// total number of simulated boundary particles
		NumNbrFF_Avg,		// average number of fluid neighbors
		NumNbrFF_P95,		// top 5% of fluid neighbors
		NumNbrFF_P99,		// top 1% of fluid neighbors
		NumNbrFF_Min,		// min number of fluid neighbors
		NumNbrFF_Max,		// max number of fluid neighbors
		NumNbrFB_Avg,		// average number of boundary neighbors
		NumNbrFB_P95,		// top 5% of boundary neighbors
		NumNbrFB_P99,		// top 1% of boundary neighbors
		NumNbrFB_Min,		// min number of boundary neighbors
		NumNbrFB_Max,		// max number of boundary neighbors

		// density deviation
		// (relative error in %, aka deviation)
		DensRelError_Raw_Avg,	// average raw value relative density error
		// TODO: DensRelError_Raw_P95 // top 5% of raw relative density errors
		// TODO: DensRelError_Raw_P99 // top 1% of raw relative density errors
		// TODO: DensRelError_Raw_Min // min raw relative density error
		// TODO: DensRelError_Raw_Max // max raw relative density error
		DensRelError_Pos_Avg,	// average positive value relative density error
		// TODO: DensRelError_Pos_P95 // top 5% of positive relative density errors
		// TODO: DensRelError_Pos_P99 // top 1% of positive relative density errors
		// TODO: DensRelError_Pos_Min // min positive relative density error
		// TODO: DensRelError_Pos_Max // max positive relative density error
		DensRelError_Abs_Avg,	// average absolute value relative density error
		// TODO: DensRelError_Abs_P95 // top 5% of absolute relative density errors
		// TODO: DensRelError_Abs_P99 // top 1% of absolute relative density errors
		// TODO: DensRelError_Abs_Min // min absolute relative density error
		// TODO: DensRelError_Abs_Max // max absolute relative density error

		// pressure solver
		NumSolverIters,		// number of pressure solver iterations performed

		// velocity stuff
		VelMag_Avg,			// average velocity magnitude
		VelMag_Min,			// min velocity magnitude
		VelMag_Max,			// max velocity magnitude

		Count
	};

	struct StatMetricDef
	{
		StatMetric metric;
		std::string_view name; // display name
		std::string_view unit; // unit hint
		std::string_view var;  // variable type
		std::string_view tip;  // tooltip description
	};

	constexpr inline std::array<StatMetricDef, static_cast<uint16_t>(StatMetric::Count)> Definitions =
	{
		{
			{
				.metric = StatMetric::NumF,
				.name = "num fluid particles",
				.unit = "",
				.var = "int",
				.tip =
				"Total number of fluid particles currently simulated"
			},
			{
				.metric = StatMetric::NumB,
				.name = "num bound particles",
				.unit = "",
				.var = "int",
				.tip =
				"Total number of boundary particles currently simulated"
			},
			{
				.metric = StatMetric::NumNbrFF_Avg,
				.name = "num fluid neighbors (avg)",
				.unit = "",
				.var = "float",
				.tip = 
				"Average number of fluid particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFF_P95,
				.name = "num fluid neighbors (p95)",
				.unit = "",
				.var = "float",
				.tip =
				"Top 5%% of fluid neighbor counts among fluid particles"
			},
			{
				.metric = StatMetric::NumNbrFF_P99,
				.name = "num fluid neighbors (p99)",
				.unit = "",
				.var = "float",
				.tip =
				"Top 1%% of fluid neighbor counts among fluid particles"
			},
			{
				.metric = StatMetric::NumNbrFF_Min,
				.name = "num fluid neighbors (min)",
				.unit = "",
				.var = "int",
				.tip =
				"Minimum number of fluid particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFF_Max,
				.name = "num fluid neighbors (max)",
				.unit = "",
				.var = "int",
				.tip =
				"Maximum number of fluid particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFB_Avg,
				.name = "num bound neighbors (avg)",
				.unit = "",
				.var = "float",
				.tip =
				"Average number of boundary particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFB_P95,
				.name = "num bound neighbors (p95)",
				.unit = "",
				.var = "int",
				.tip =
				"Top 5%% of boundary neighbor counts among fluid particles"
			},
			{
				.metric = StatMetric::NumNbrFB_P99,
				.name = "num bound neighbors (p99)",
				.unit = "",
				.var = "int",
				.tip =
				"Top 1%% of boundary neighbor counts among fluid particles"
			},
			{
				.metric = StatMetric::NumNbrFB_Min,
				.name = "num bound neighbors (min)",
				.unit = "",
				.var = "int",
				.tip =
				"Minimum number of boundary particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFB_Max,
				.name = "num bound neighbors (max)",
				.unit = "",
				.var = "int",
				.tip =
				"Maximum number of boundary particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::DensRelError_Raw_Avg,
				.name = "dens rel raw err (avg)",
				.unit = "%",
				.var = "float",
				.tip =
				"Average of raw relative density error among fluid particles"
			},
			{
				.metric = StatMetric::DensRelError_Pos_Avg,
				.name = "dens rel pos err (avg)",
				.unit = "%",
				.var = "float",
				.tip =
				"Average of positive relative density error among fluid particles"
			},
			{
				.metric = StatMetric::DensRelError_Abs_Avg,
				.name = "dens rel abs err (avg)",
				.unit = "%",
				.var = "float",
				.tip =
				"Average of absolute relative density error among fluid particles"
			},
			{
				.metric = StatMetric::NumSolverIters,
				.name = "num solver iterations",
				.unit = "",
				.var = "int",
				.tip =
				"Number of pressure solver iterations performed"
			},
			{
				.metric = StatMetric::VelMag_Avg,
				.name = "velocity mag (avg)",
				.unit = "m/s",
				.var = "float",
				.tip =
				"Average velocity magnitude among fluid particles"
			},
			{
				.metric = StatMetric::VelMag_Min,
				.name = "velocity mag (min)",
				.unit = "m/s",
				.var = "float",
				.tip =
				"Minimum velocity magnitude among fluid particles"
			},
			{
				.metric = StatMetric::VelMag_Max,
				.name = "velocity mag (max)",
				.unit = "m/s",
				.var = "float",
				.tip =
				"Maximum velocity magnitude among fluid particles"
			}
		}
	};
	static_assert(Definitions.size() == static_cast<uint8_t>(StatMetric::Count),
		"Stats::Definitions must list all StatMetric entries");

	constexpr inline uint8_t GetIdx(StatGroup group) { return static_cast<uint8_t>(group); }
	constexpr inline uint8_t GetIdx(StatMetric metric) { return static_cast<uint8_t>(metric); }

	// record of all per-metric stat data for one particular sim step
	struct StatRecordStep
	{
		// metadata
		uint32_t simStep = 0;
		double simTime = 0.0;

		// stat data
		std::array<float, GetIdx(StatMetric::Count)> stats{};
	};

	inline float Get(const StatRecordStep& record, StatMetric metric) { return record.stats[GetIdx(metric)]; }
	inline void Set(StatRecordStep& record, StatMetric metric, float x) { record.stats[GetIdx(metric)] = x; }
	inline void Add(StatRecordStep& record, StatMetric metric, float x) { record.stats[GetIdx(metric)] += x; }
	inline void MaxWith(StatRecordStep& record, StatMetric metric, float x)
		{ record.stats[GetIdx(metric)] = std::max(x, record.stats[GetIdx(metric)]); }

} // namespace Stats
