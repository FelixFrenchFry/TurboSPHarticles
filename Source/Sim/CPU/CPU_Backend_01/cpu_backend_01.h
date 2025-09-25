// cpu_backend_01.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>
#include <vector>

// internal
#include "Source/Sim/Core/backend_sim.h"
#include "Source/Sim/Core/parameters_sim.h"



namespace Sim
{
	class SimBackendCPU_01 final : public ISimBackend
	{
	public: // ==============================

		SimBackendCPU_01() = default;

		const char* Name() const override { return "CPU_01"; }

		// copies & translates a universal simulation state into -> backend-specific data structures
		void PullFromUniversalSimState(const Sync::UniversalSimState& src) override;

		// copies & translates backend-specific data structures into -> universal simulation state
		void PushToUniversalSimState(Sync::UniversalSimState& dst) const override;

		// packs the current simulation state into a snapshot used for exporting or visualization
		// (implementation is backend-specific for better performance)
		void PackExportSnapshot(Sync::ExportSnapshot& snap) const override;
		void PackVisualSnapshot(Sync::VisualSnapshot& snap) const override;

		// fills in the scalar data based on given different color coding modes
		// (backend-specific such that possible transformations can be applied efficiently)
		void FillScalarData(Sync::VisualSnapshot& snap, Base::ColorCodingMode mode) const override;

		// advances the simulation by N_steps
		void SimStep(const SimParams& params, std::int64_t N_steps) override;

	private: // ==============================

		// misc
		std::int64_t step_ = 0;

		// owned simulation state stored in backend-specific data structures
		// fluid particles
		std::int64_t N_F_ = 0;
		std::vector<float> xPosF_, yPosF_, zPosF_;
		std::vector<float> xVelF_, yVelF_, zVelF_;
		std::vector<float> xAccF_, yAccF_, zAccF_;
		std::vector<float> xAccPressF_, yAccPressF_, zAccPressF_;
		std::vector<float> xAccViscF_, yAccViscF_, zAccViscF_;
		std::vector<float> xAccExtF_, yAccExtF_, zAccExtF_;
		std::vector<float> densityF_;
		std::vector<float> pressureF_;
		std::vector<float> massF_;
		std::vector<float> sizeF_;
		std::vector<std::uint32_t> idF_;
		// boundary particles
		std::int64_t N_B_ = 0;
		std::vector<float> xPosB_, yPosB_, zPosB_;
		std::vector<float> xVelB_, yVelB_, zVelB_;
		std::vector<float> densityB_;
		std::vector<float> pressureB_;
		std::vector<float> massB_;
		std::vector<float> sizeB_;
		std::vector<std::uint32_t> idB_;
	};

} // namespace Sim
