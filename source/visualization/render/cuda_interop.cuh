// cuda_interop.cuh
#pragma once

// external
#include <cstdint>
#include <cuda_runtime_api.h>



// forward declarations
namespace Sync
{
	struct VisualParticleSample;
} // namespace Sync

namespace Render
{
	struct InstanceData;

	void Launch_PackVisualInstancesCUDA(
		const Sync::VisualParticleSample* fluid,
		const Sync::VisualParticleSample* rigid,
		const float fluidRadius,
		const float particleSizeB,
		const float rigidRadiusScale,
		const uint32_t N_F, const uint32_t N_R,
		cudaStream_t stream,
		// destinations
		InstanceData* dst);

} // namespace Render
