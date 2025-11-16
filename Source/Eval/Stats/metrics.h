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
	enum class StatMetric : uint16_t
	{
		// particles and neighbors
		NumF,				// total number of simulated fluid particles
		NumB,				// total number of simulated boundary particles
		NumNeighFF_Avg,		// average number of fluid neighbors
		NumNeighFB_Avg,		// average number of boundary neighbors
		NumNeighFF_Min,		// min number of fluid neighbors
		NumNeighFB_Min,		// min number of boundary neighbors
		NumNeighFF_Max,		// max number of fluid neighbors
		NumNeighFB_Max,		// max number of boundary neighbors

		// density
		// (relative error in % aka deviation)
		DensRelError_Avg,	// average relative density error
		DensRelError_P95,	// top 5% of relative density errors
		DensRelError_P99,	// top 1% of relative density errors
		DensRelError_Min,	// min relative density error
		DensRelError_Max,	// max relative density error

		// velocity
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
				.metric = StatMetric::NumNeighFF_Avg,
				.name = "num fluid neighbors (avg)",
				.unit = "",
				.var = "float",
				.tip = 
				"Average number of fluid particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNeighFB_Avg,
				.name = "num bound neighbors (avg)",
				.unit = "",
				.var = "float",
				.tip =
				"Average number of boundary particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNeighFF_Min,
				.name = "num fluid neighbors (min)",
				.unit = "",
				.var = "int",
				.tip =
				"Minimum number of fluid particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNeighFB_Min,
				.name = "num bound neighbors (min)",
				.unit = "",
				.var = "int",
				.tip =
				"Minimum number of boundary particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNeighFF_Max,
				.name = "num fluid neighbors (max)",
				.unit = "",
				.var = "int",
				.tip =
				"Maximum number of fluid particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNeighFB_Max,
				.name = "num bound neighbors (max)",
				.unit = "",
				.var = "int",
				.tip =
				"Maximum number of boundary particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::DensRelError_Avg,
				.name = "density rel err (avg)",
				.unit = "%",
				.var = "float",
				.tip =
				"Average relative density error among fluid particles"
			},
			{
				.metric = StatMetric::DensRelError_P95,
				.name = "density rel err (p95)",
				.unit = "%",
				.var = "float",
				.tip =
				"Top 5%% of relative density errors among fluid particles"
			},
			{
				.metric = StatMetric::DensRelError_P99,
				.name = "density rel err (p99)",
				.unit = "%",
				.var = "float",
				.tip =
				"Top 1%% of relative density errors among fluid particles"
			},
			{
				.metric = StatMetric::DensRelError_Min,
				.name = "density rel err (min)",
				.unit = "%",
				.var = "float",
				.tip =
				"Minimum relative density error among fluid particles"
			},
			{
				.metric = StatMetric::DensRelError_Max,
				.name = "density rel err (max)",
				.unit = "%",
				.var = "float",
				.tip =
				"Maximum relative density error among fluid particles"
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
	static_assert(Definitions.size() == static_cast<uint16_t>(StatMetric::Count),
		"Stats::Definitions must list all StatMetric entries");

	constexpr inline uint16_t GetIdx(StatMetric metric) { return static_cast<uint16_t>(metric); }

	// record of all per-metric stat data for one particular sim step
	struct StatRecordStep
	{
		// metadata
		int64_t simStep = 0;
		double simTime = 0.0;

		// stat data
		std::array<float, GetIdx(StatMetric::Count)> stats{};
	};

	inline float Get(const StatRecordStep& record, StatMetric metric) { return record.stats[GetIdx(metric)]; }
	inline void Set(StatRecordStep& record, StatMetric metric, float x) { record.stats[GetIdx(metric)] = x; }
	inline void Add(StatRecordStep& record, StatMetric metric, float x) { record.stats[GetIdx(metric)] += x; }
	inline void MaxWith(StatRecordStep& record, StatMetric metric, float x)
		{ record.stats[GetIdx(metric)] = fmaxf(x, record.stats[GetIdx(metric)]); }

} // namespace Stats
