// particle_radius.h

#pragma once

// external
#include <algorithm>
#include <cmath>



namespace Render
{
	inline float ComputeRigidBoundaryRenderRadius(
		const float volume,
		const float particleSizeB,
		const float radiusScale)
	{
		const float nonNegativeScale = std::max(radiusScale, 0.0f);
		const float fallbackRadius = particleSizeB * 0.5f * nonNegativeScale;
		if (!std::isfinite(volume) || volume <= 0.0f)
		{
			return fallbackRadius;
		}

		const float radius = 0.5f * std::cbrt(volume) * nonNegativeScale;
		return std::isfinite(radius) ? radius : fallbackRadius;
	}

} // namespace Render