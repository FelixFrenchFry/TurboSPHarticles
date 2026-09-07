// stages.h
//
// TODO: comment

#pragma once

// external
#include <array>
#include <cstdint>
#include <string_view>



namespace Trace
{
	enum class TraceStage : uint8_t
	{
		StepBegin,
		Dist_LoopParameterSync,
		Dist_InputValidation,
		Dist_ParameterSync,
		Dist_ParticleRedistribution,
		Dist_GlobalCountValidation,
		Dist_RankParticlePreparation,
		Dist_FluidSyncPlanRebuild,
		Dist_BackendConfiguration,
		Dist_RankBackendStep,
		Dist_ParticlePermutation,
		Dist_RigidBodySynchronization,
		Dist_RigidForceReduction,
		Dist_RigidForceIntegration,
		Dist_RigidStateBroadcast,
		Dist_ParticleMetadataValidation,
		Dist_ExportSnapshot,
		Dist_Statistics,
		Dist_LoadBalancing,
		Dist_Metrics,
		Dist_PostStepParameterSync,
		Dist_ProgressReport,
		StepComplete,
		Dist_FinalStateSerialization,
		Dist_RigidParticleAuthorityValidation,
		Dist_RigidForceCollection,
		Dist_ExportSnapshotGather,
		Dist_ExportSnapshotSerialization,
		Dist_RankBackendStepComplete,
		Count
	};

	struct TraceStageDef
	{
		TraceStage stage;
		std::string_view name;  // display name
		std::string_view alias; // stable machine-readable name
	};

	constexpr inline std::array<TraceStageDef,
		static_cast<uint16_t>(TraceStage::Count)> Definitions =
	{
		{
			{
				.stage = TraceStage::StepBegin,
				.name = "step begin",
				.alias = "step_begin"
			},
			{
				.stage = TraceStage::Dist_LoopParameterSync,
				.name = "distributed loop parameter synchronization",
				.alias = "dist_loop_parameter_sync"
			},
			{
				.stage = TraceStage::Dist_InputValidation,
				.name = "distributed input validation",
				.alias = "dist_input_validation"
			},
			{
				.stage = TraceStage::Dist_ParameterSync,
				.name = "distributed parameter synchronization",
				.alias = "dist_parameter_sync"
			},
			{
				.stage = TraceStage::Dist_ParticleRedistribution,
				.name = "distributed particle redistribution",
				.alias = "dist_particle_redistribution"
			},
			{
				.stage = TraceStage::Dist_GlobalCountValidation,
				.name = "distributed global count validation",
				.alias = "dist_global_count_validation"
			},
			{
				.stage = TraceStage::Dist_RankParticlePreparation,
				.name = "distributed rank-particle preparation",
				.alias = "dist_rank_particle_preparation"
			},
			{
				.stage = TraceStage::Dist_FluidSyncPlanRebuild,
				.name = "distributed fluid sync plan rebuild",
				.alias = "dist_fluid_sync_plan_rebuild"
			},
			{
				.stage = TraceStage::Dist_BackendConfiguration,
				.name = "distributed backend configuration",
				.alias = "dist_backend_configuration"
			},
			{
				.stage = TraceStage::Dist_RankBackendStep,
				.name = "distributed rank backend step",
				.alias = "dist_rank_backend_step"
			},
			{
				.stage = TraceStage::Dist_ParticlePermutation,
				.name = "distributed particle permutation",
				.alias = "dist_particle_permutation"
			},
			{
				.stage = TraceStage::Dist_RigidBodySynchronization,
				.name = "distributed rigid-body synchronization",
				.alias = "dist_rigid_body_synchronization"
			},
			{
				.stage = TraceStage::Dist_RigidForceReduction,
				.name = "distributed rigid force reduction",
				.alias = "dist_rigid_force_reduction"
			},
			{
				.stage = TraceStage::Dist_RigidForceIntegration,
				.name = "distributed rigid force integration",
				.alias = "dist_rigid_force_integration"
			},
			{
				.stage = TraceStage::Dist_RigidStateBroadcast,
				.name = "distributed rigid state broadcast",
				.alias = "dist_rigid_state_broadcast"
			},
			{
				.stage = TraceStage::Dist_ParticleMetadataValidation,
				.name = "distributed particle metadata validation",
				.alias = "dist_particle_metadata_validation"
			},
			{
				.stage = TraceStage::Dist_ExportSnapshot,
				.name = "distributed export snapshot",
				.alias = "dist_export_snapshot"
			},
			{
				.stage = TraceStage::Dist_Statistics,
				.name = "distributed statistics",
				.alias = "dist_statistics"
			},
			{
				.stage = TraceStage::Dist_LoadBalancing,
				.name = "distributed load balancing",
				.alias = "dist_load_balancing"
			},
			{
				.stage = TraceStage::Dist_Metrics,
				.name = "distributed metrics",
				.alias = "dist_metrics"
			},
			{
				.stage = TraceStage::Dist_PostStepParameterSync,
				.name = "distributed post-step parameter synchronization",
				.alias = "dist_post_step_parameter_sync"
			},
			{
				.stage = TraceStage::Dist_ProgressReport,
				.name = "distributed progress report",
				.alias = "dist_progress_report"
			},
			{
				.stage = TraceStage::StepComplete,
				.name = "step complete",
				.alias = "step_complete"
			},
			{
				.stage = TraceStage::Dist_FinalStateSerialization,
				.name = "distributed final-state serialization",
				.alias = "dist_final_state_serialization"
			},
			{
				.stage = TraceStage::Dist_RigidParticleAuthorityValidation,
				.name = "distributed rigid-particle authority validation",
				.alias = "dist_rigid_particle_authority_validation"
			},
			{
				.stage = TraceStage::Dist_RigidForceCollection,
				.name = "distributed rigid force collection",
				.alias = "dist_rigid_force_collection"
			},
			{
				.stage = TraceStage::Dist_ExportSnapshotGather,
				.name = "distributed export snapshot gather",
				.alias = "dist_export_snapshot_gather"
			},
			{
				.stage = TraceStage::Dist_ExportSnapshotSerialization,
				.name = "distributed export snapshot serialization",
				.alias = "dist_export_snapshot_serialization"
			},
			{
				.stage = TraceStage::Dist_RankBackendStepComplete,
				.name = "distributed rank backend step complete",
				.alias = "dist_rank_backend_step_complete"
			}
		}
	};
	static_assert(Definitions.size() == static_cast<uint8_t>(TraceStage::Count),
		"Trace::Definitions must list all TraceStage entries");

	constexpr inline bool TraceStageDefinitionsMatchEnum()
	{
		for (uint8_t i = 0; i < static_cast<uint8_t>(TraceStage::Count); i++)
		{
			if (Definitions[i].stage != static_cast<TraceStage>(i))
			{
				return false;
			}
		}

		return true;
	}
	static_assert(TraceStageDefinitionsMatchEnum(),
		"Trace::Definitions entries must match TraceStage ordering");

	constexpr inline uint8_t GetIdx(const TraceStage stage)
	{
		return static_cast<uint8_t>(stage);
	}

} // namespace Trace
