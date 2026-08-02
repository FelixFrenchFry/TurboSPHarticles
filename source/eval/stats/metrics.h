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
		NumR,				// total number of simulated rigid particles

		// neighbors of fluid particles
		NumNbrFF_Avg,		// average number of fluid neighbors
		NumNbrFF_Min,		// min number of fluid neighbors
		NumNbrFF_Max,		// max number of fluid neighbors
		NumNbrFF_P99,		// top 1% of fluid neighbors
		NumNbrFB_Avg,		// average number of boundary neighbors
		NumNbrFB_Min,		// min number of boundary neighbors
		NumNbrFB_Max,		// max number of boundary neighbors
		NumNbrFB_P99,		// top 1% of boundary neighbors
		NumNbrFR_Avg,		// average number of rigid neighbors of fluid particles
		NumNbrFR_Min,		// min number of rigid neighbors of fluid particles
		NumNbrFR_Max,		// max number of rigid neighbors of fluid particles
		NumNbrFR_P99,		// top 1% of rigid neighbors of fluid particles

		// neighbors of rigid particles
		NumNbrRF_Avg,		// average number of fluid neighbors of rigid particles
		NumNbrRF_Min,		// min number of fluid neighbors of rigid particles
		NumNbrRF_Max,		// max number of fluid neighbors of rigid particles
		NumNbrRF_P99,		// top 1% of fluid neighbors of rigid particles
		NumNbrRB_Avg,		// average number of boundary neighbors of rigid particles
		NumNbrRB_Min,		// min number of boundary neighbors of rigid particles
		NumNbrRB_Max,		// max number of boundary neighbors of rigid particles
		NumNbrRB_P99,		// top 1% of boundary neighbors of rigid particles
		NumNbrRR_Avg,		// average number of rigid neighbors of rigid particles
		NumNbrRR_Min,		// min number of rigid neighbors of rigid particles
		NumNbrRR_Max,		// max number of rigid neighbors of rigid particles
		NumNbrRR_P99,		// top 1% of rigid neighbors of rigid particles

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
		CFL_Number,			// CFL number given the max velocity magnitude
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
				.name = "num boundary particles",
				.unit = "",
				.var = "int",
				.tip =
				"Total number of boundary particles currently simulated"
			},
			{
				.metric = StatMetric::NumR,
				.name = "num rigid particles",
				.unit = "",
				.var = "int",
				.tip =
				"Total number of rigid particles currently simulated"
			},
			{
				.metric = StatMetric::NumNbrFF_Avg,
				.name = "num nbr FF (avg)",
				.unit = "",
				.var = "float",
				.tip = 
				"Average number of fluid particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFF_Min,
				.name = "num nbr FF (min)",
				.unit = "",
				.var = "int",
				.tip =
				"Minimum number of fluid particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFF_Max,
				.name = "num nbr FF (max)",
				.unit = "",
				.var = "int",
				.tip =
				"Maximum number of fluid particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFF_P99,
				.name = "num nbr FF (p99)",
				.unit = "",
				.var = "int",
				.tip =
				"Top 1%% of fluid neighbor counts among fluid particles"
			},
			{
				.metric = StatMetric::NumNbrFB_Avg,
				.name = "num nbr FB (avg)",
				.unit = "",
				.var = "float",
				.tip =
				"Average number of boundary particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFB_Min,
				.name = "num nbr FB (min)",
				.unit = "",
				.var = "int",
				.tip =
				"Minimum number of boundary particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFB_Max,
				.name = "num nbr FB (max)",
				.unit = "",
				.var = "int",
				.tip =
				"Maximum number of boundary particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFB_P99,
				.name = "num nbr FB (p99)",
				.unit = "",
				.var = "int",
				.tip =
				"Top 1%% of boundary neighbor counts among fluid particles"
			},
			{
				.metric = StatMetric::NumNbrFR_Avg,
				.name = "num nbr FR (avg)",
				.unit = "",
				.var = "float",
				.tip =
				"Average number of rigid particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFR_Min,
				.name = "num nbr FR (min)",
				.unit = "",
				.var = "int",
				.tip =
				"Minimum number of rigid particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFR_Max,
				.name = "num nbr FR (max)",
				.unit = "",
				.var = "int",
				.tip =
				"Maximum number of rigid particles neighboring a fluid particle"
			},
			{
				.metric = StatMetric::NumNbrFR_P99,
				.name = "num nbr FR (p99)",
				.unit = "",
				.var = "int",
				.tip =
				"Top 1%% of rigid neighbor counts among fluid particles"
			},
			{
				.metric = StatMetric::NumNbrRF_Avg,
				.name = "num nbr RF (avg)",
				.unit = "",
				.var = "float",
				.tip =
				"Average number of fluid particles neighboring a rigid particle"
			},
			{
				.metric = StatMetric::NumNbrRF_Min,
				.name = "num nbr RF (min)",
				.unit = "",
				.var = "int",
				.tip =
				"Minimum number of fluid particles neighboring a rigid particle"
			},
			{
				.metric = StatMetric::NumNbrRF_Max,
				.name = "num nbr RF (max)",
				.unit = "",
				.var = "int",
				.tip =
				"Maximum number of fluid particles neighboring a rigid particle"
			},
			{
				.metric = StatMetric::NumNbrRF_P99,
				.name = "num nbr RF (p99)",
				.unit = "",
				.var = "int",
				.tip =
				"Top 1%% of fluid neighbor counts among rigid particles"
			},
			{
				.metric = StatMetric::NumNbrRB_Avg,
				.name = "num nbr RB (avg)",
				.unit = "",
				.var = "float",
				.tip =
				"Average number of boundary particles neighboring a rigid particle"
			},
			{
				.metric = StatMetric::NumNbrRB_Min,
				.name = "num nbr RB (min)",
				.unit = "",
				.var = "int",
				.tip =
				"Minimum number of boundary particles neighboring a rigid particle"
			},
			{
				.metric = StatMetric::NumNbrRB_Max,
				.name = "num nbr RB (max)",
				.unit = "",
				.var = "int",
				.tip =
				"Maximum number of boundary particles neighboring a rigid particle"
			},
			{
				.metric = StatMetric::NumNbrRB_P99,
				.name = "num nbr RB (p99)",
				.unit = "",
				.var = "int",
				.tip =
				"Top 1%% of boundary neighbor counts among rigid particles"
			},
			{
				.metric = StatMetric::NumNbrRR_Avg,
				.name = "num nbr RR (avg)",
				.unit = "",
				.var = "float",
				.tip =
				"Average number of other-body rigid particles neighboring a rigid particle"
			},
			{
				.metric = StatMetric::NumNbrRR_Min,
				.name = "num nbr RR (min)",
				.unit = "",
				.var = "int",
				.tip =
				"Minimum number of other-body rigid particles neighboring a rigid particle"
			},
			{
				.metric = StatMetric::NumNbrRR_Max,
				.name = "num nbr RR (max)",
				.unit = "",
				.var = "int",
				.tip =
				"Maximum number of other-body rigid particles neighboring a rigid particle"
			},
			{
				.metric = StatMetric::NumNbrRR_P99,
				.name = "num nbr RR (p99)",
				.unit = "",
				.var = "int",
				.tip =
				"Top 1%% of other-body rigid neighbor counts among rigid particles"
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
				.metric = StatMetric::CFL_Number,
				.name = "CFL number",
				.unit = "",
				.var = "float",
				.tip =
				"Maximum distance covered by any particle, given as a fraction"
				"of its size using (max|v| * dt / h)"
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
		uint64_t simStep = 0;
		double simTime = 0.0;

		// stat data
		std::array<float, GetIdx(StatMetric::Count)> stats{};
	};

	inline float Get(const StatRecordStep& record, StatMetric metric)
		{ return record.stats[GetIdx(metric)]; }
	inline void Set(StatRecordStep& record, StatMetric metric, float x)
		{ record.stats[GetIdx(metric)] = x; }

} // namespace Stats
