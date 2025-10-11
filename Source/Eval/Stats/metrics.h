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
		// particle counts
		NumF,
		NumB,
		NumNeighF_Avg,		// average number of fluid neighbors
		NumNeighB_Avg,		// average number of boundary neighbors
		NumNeighF_Max,		// max number of fluid neighbors
		NumNeighB_Max,		// max number of boundary neighbors

		// density
		// (relative error in % aka deviation)
		DensRelError_Avg,	// average relative density error
		DensRelError_P95,	// top 5% of relative density errors
		DensRelError_P99,	// top 1% of relative density errors
		DensRelError_Max,	// max relative density error

		// velocity
		VelMag_Avg,			// average velocity magnitude
		VelMag_Max,			// max velocity magnitude

		Count
	};

	struct StatMetricDef
	{
		StatMetric metric;
		std::string_view name; // display name
		std::string_view unit; // unit hint
	};

	constexpr inline std::array<StatMetricDef, static_cast<uint16_t>(StatMetric::Count)> Definitions =
	{
		{
			{
				.metric = StatMetric::NumF,
				.name = "num fluid particles",
				.unit = ""
			},
			{
				.metric = StatMetric::NumB,
				.name = "num bound particles",
				.unit = ""
			},
			{
				.metric = StatMetric::NumNeighF_Avg,
				.name = "num fluid neighbors (avg)",
				.unit = ""
			},
			{
				.metric = StatMetric::NumNeighB_Avg,
				.name = "num bound neighbors (avg)",
				.unit = ""
			},
			{
				.metric = StatMetric::NumNeighF_Max,
				.name = "num fluid neighbors (max)",
				.unit = ""
			},
			{
				.metric = StatMetric::NumNeighB_Max,
				.name = "num bound neighbors (max)",
				.unit = ""
			},
			{
				.metric = StatMetric::DensRelError_Avg,
				.name = "density rel err (avg)",
				.unit = "%"
			},
			{
				.metric = StatMetric::DensRelError_P95,
				.name = "density rel err (p95)",
				.unit = "%"
			},
			{
				.metric = StatMetric::DensRelError_P99,
				.name = "density rel err (p99)",
				.unit = "%"
			},
			{
				.metric = StatMetric::DensRelError_Max,
				.name = "density rel err (max)",
				.unit = "%"
			},
			{
				.metric = StatMetric::VelMag_Avg,
				.name = "velocity mag (avg)",
				.unit = "m/s"
			},
			{
				.metric = StatMetric::VelMag_Max,
				.name = "velocity mag (max)",
				.unit = "m/s"
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
		float simTime = 0.0f;

		// stat data
		std::array<float, GetIdx(StatMetric::Count)> stats{};
	};

	inline float Get(const StatRecordStep& record, StatMetric metric) { return record.stats[GetIdx(metric)]; }
	inline void Set(StatRecordStep& record, StatMetric metric, float x) { record.stats[GetIdx(metric)] = x; }
	inline void Add(StatRecordStep& record, StatMetric metric, float x) { record.stats[GetIdx(metric)] += x; }
	inline void MaxWith(StatRecordStep& record, StatMetric metric, float x)
		{ record.stats[GetIdx(metric)] = std::max(x, record.stats[GetIdx(metric)]); }

} // namespace Stats
