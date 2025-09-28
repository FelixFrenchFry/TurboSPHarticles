// backend_sim.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>
#include <memory>

// internal
#include "parameters_sim.h"
#include "Source/Base/enum_hub.h"
#include "Source/Sim/Sync/snapshots.h"
#include "Source/Sim/Sync/state_sim.h"



namespace Sim
{
	class ISimBackend
	{
	public: // ==============================

		virtual ~ISimBackend() = default;

		virtual const char* Name() const = 0;

		// copies & translates a universal simulation state into -> backend-specific data structures
		virtual void PullFromUniversalSimState(const Sync::UniversalSimState& src) = 0;

		// copies & translates backend-specific data structures into -> universal simulation state
		virtual void PushToUniversalSimState(Sync::UniversalSimState& dst) const = 0;

		// packs the current simulation state into a snapshot used for exporting or visualization
		// (implementation is backend-specific for better performance)
		virtual void PackExportSnapshot(Sync::ExportSnapshot& snap) const = 0;
		virtual void PackVisualSnapshot(Sync::VisualSnapshot& snap) const = 0;

		// fills in the scalar data based on given different color coding modes
		// (backend-specific such that possible transformations can be applied efficiently)
		virtual void FillScalarData(Sync::VisualSnapshot& snap, Base::ColorCodingMode mode) const = 0;

		// advances the simulation by N_steps
		virtual void SimStep(SimParams params, int64_t N_steps = 1) = 0;
	};

	// backend factory
	std::unique_ptr<ISimBackend> MakeSimBackend(Base::BackendVersion version, const SimParams& params);

} // namespace Sim
