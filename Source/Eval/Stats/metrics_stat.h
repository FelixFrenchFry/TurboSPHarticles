// metrics_stat.h
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

		// misc
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
			{ StatMetric::NumF,				"num fluid particles",				"" },
			{ StatMetric::NumB,				"num bound particles",				"" },
			{ StatMetric::NumNeighF_Avg,		"num fluid neighbors (avg)",		"" },
			{ StatMetric::NumNeighB_Avg,		"num bound neighbors (avg)",		"" },
			{ StatMetric::NumNeighF_Max,		"num fluid neighbors (max)",		"" },
			{ StatMetric::NumNeighB_Max,		"num bound neighbors (max)",		"" },
			{ StatMetric::DensRelError_Avg,	"density rel err (avg)",			"%" },
			{ StatMetric::DensRelError_P95,	"density rel err (p95)",			"%" },
			{ StatMetric::DensRelError_P99,	"density rel err (p99)",			"%" },
			{ StatMetric::DensRelError_Max,   "density rel err (max)",			"%" },
			{ StatMetric::VelMag_Avg,			"velocity mag (avg)",				"m/s" },
			{ StatMetric::VelMag_Max,			"velocity mag (max)",				"m/s" },
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
