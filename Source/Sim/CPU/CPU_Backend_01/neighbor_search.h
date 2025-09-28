// neighbor_search.h
//
// TODO: comment

#pragma once

// external

// internal



namespace Sim
{
	float KernelFunction_CubicSpline(float distNorm, float alpha)
	{
		const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
		const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

		return alpha * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);
	}

	glm::vec3 KernelGradient_CubicSpline(glm::vec3 diff, float distNorm, float size, float alpha)
	{
		const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
		const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

		return alpha * (diff / (distNorm * size * size)) * (-3.0f * t2 * t2 + 12 * t1 * t1);
	}

} // namespace Sim
