// solver_constants.h

#pragma once

// external
#include <numbers>



namespace Sim
{
	// host-computed coefficients shared by CPU and GPU backend
	struct SolverConstants
	{
		float sizeInvF;
		float sizeSquF;
		float cubicSplineAlpha;
		float restDensitySquInv;
	};

	inline constexpr SolverConstants BuildSolverConstants(
		const float particleSizeF, const float restDensity) noexcept
	{
		// derivation in FP64 for better accuracy, final coefficients in FP32
		// (alpha normalizes the hardcoded cubic spline with support radius of 2 * particleSizeF)
		const double sizeF = static_cast<double>(particleSizeF);
		const double sizeInvF = 1.0 / sizeF;
		const double density = static_cast<double>(restDensity);

		return SolverConstants{
			static_cast<float>(sizeInvF),
			static_cast<float>(sizeF * sizeF),
			static_cast<float>(sizeInvF * sizeInvF * sizeInvF
				/ (4.0 * std::numbers::pi_v<double>)),
			static_cast<float>(1.0 / (density * density))
		};
	}

} // namespace Sim
