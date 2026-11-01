// sph_kernels.cpp

// external
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/vec3.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>

// internal
#include "source/simulation/CPU/simd.h"
#include "source/simulation/core/solver_constants.h"



namespace
{
	struct KernelValues
	{
		double weight = 0.0;
		glm::dvec3 gradient{ 0.0 };
	};

	using KernelOffsets = std::array<glm::vec3, CPU::N_SIMD_WIDTH>;

	std::array<KernelValues, CPU::N_SIMD_WIDTH> EvaluateKernels(
		const KernelOffsets& offsets, const float particleSizeF = 1.0f)
	{
		std::array<float, CPU::N_SIMD_WIDTH> dx{}, dy{}, dz{};
		for (uint32_t lane = 0; lane < CPU::N_SIMD_WIDTH; lane++)
		{
			dx[lane] = offsets[lane].x;
			dy[lane] = offsets[lane].y;
			dz[lane] = offsets[lane].z;
		}

		const auto constants = Sim::BuildSolverConstants(particleSizeF, 1000.0f);
		const float kernelRadiusSqu = 4.0f * particleSizeF * particleSizeF;
		const auto x = CPU::LoadFloat32xSIMD(dx.data());
		const auto y = CPU::LoadFloat32xSIMD(dy.data());
		const auto z = CPU::LoadFloat32xSIMD(dz.data());
		const auto weight = CPU::KernelWeight_CubicSpline_SIMD(x, y, z,
			constants.sizeInvF, kernelRadiusSqu, constants.cubicSplineAlpha);
		CPU::Float32xSIMD xkg, ykg, zkg;
		CPU::KernelGradient_CubicSpline_SIMD(x, y, z,
			constants.sizeInvF, kernelRadiusSqu, constants.cubicSplineAlpha, xkg, ykg, zkg);

		std::array<float, CPU::N_SIMD_WIDTH> weights{}, xGradient{}, yGradient{}, zGradient{};
		weight.store_unaligned(weights.data());
		xkg.store_unaligned(xGradient.data());
		ykg.store_unaligned(yGradient.data());
		zkg.store_unaligned(zGradient.data());

		std::array<KernelValues, CPU::N_SIMD_WIDTH> result{};
		for (uint32_t lane = 0; lane < CPU::N_SIMD_WIDTH; lane++)
		{
			result[lane] = { weights[lane], { xGradient[lane], yGradient[lane], zGradient[lane] } };
		}
		return result;
	}

	KernelValues EvaluateKernel(const glm::vec3& offset, const float particleSizeF = 1.0f)
	{
		KernelOffsets offsets;
		offsets.fill(offset);
		return EvaluateKernels(offsets, particleSizeF)[0];
	}

	// independent piecewise polynomial and radial derivative in double precision
	KernelValues ReferenceKernel(const glm::vec3& offset, const double particleSizeF = 1.0)
	{
		const glm::dvec3 displacement(offset);
		const double distance = std::sqrt(displacement.x * displacement.x
			+ displacement.y * displacement.y + displacement.z * displacement.z);
		const double q = distance / particleSizeF;
		if (q >= 2.0) { return {}; }

		const double normalization = 1.0 / (std::numbers::pi_v<double>
			* particleSizeF * particleSizeF * particleSizeF);
		const double shape = q < 1.0
			? 1.0 - 1.5 * q * q + 0.75 * q * q * q
			: 0.25 * std::pow(2.0 - q, 3);
		const double derivative = q < 1.0
			? -3.0 * q + 2.25 * q * q
			: -0.75 * (2.0 - q) * (2.0 - q);

		KernelValues result{ normalization * shape, { 0.0, 0.0, 0.0 } };
		if (distance > 0.0)
		{
			result.gradient = displacement * (normalization * derivative / (particleSizeF * distance));
		}
		return result;
	}

	void ExpectKernelNear(const KernelValues& actual, const KernelValues& expected,
		const double particleSizeF = 1.0)
	{
		// compare dimensionless values so tolerances follow the h^-3 and h^-4 scaling
		const double weightScale = std::pow(particleSizeF, 3);
		const double gradientScale = weightScale * particleSizeF;
		const double expectedWeight = expected.weight * weightScale;
		EXPECT_NEAR(actual.weight * weightScale, expectedWeight,
			1e-7 + 2e-6 * std::abs(expectedWeight));
		for (int axis = 0; axis < 3; axis++)
		{
			SCOPED_TRACE(axis);
			const double expectedGradient = expected.gradient[axis] * gradientScale;
			EXPECT_NEAR(actual.gradient[axis] * gradientScale, expectedGradient,
				1e-7 + 2e-6 * std::abs(expectedGradient));
		}
	}

	// pins the normalization and both polynomial branches to hand-derived values
	TEST(SphKernelTests, ComputesKnownWeights)
	{
		const std::array<float, 6> distances{ 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f };
		const std::array<double, 6> expected{ 1.0, 23.0 / 32.0, 0.25, 1.0 / 32.0, 0.0, 0.0 };
		for (uint32_t i = 0; i < distances.size(); i++)
		{
			SCOPED_TRACE(distances[i]);
			const auto actual = EvaluateKernel({ distances[i], 0.0f, 0.0f });

			EXPECT_NEAR(actual.weight, expected[i] / std::numbers::pi_v<double>, 1e-7);
		}
	}

	// pins the derivative sign and magnitude without using the reference helper
	TEST(SphKernelTests, ComputesKnownGradients)
	{
		const std::array<float, 6> distances{ 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f };
		const std::array<double, 6> expected{ 0.0, -15.0 / 16.0, -0.75, -3.0 / 16.0, 0.0, 0.0 };
		for (uint32_t i = 0; i < distances.size(); i++)
		{
			SCOPED_TRACE(distances[i]);
			const auto actual = EvaluateKernel({ distances[i], 0.0f, 0.0f });

			EXPECT_NEAR(actual.gradient.x, expected[i] / std::numbers::pi_v<double>, 1e-7);
			EXPECT_EQ(actual.gradient.y, 0.0);
			EXPECT_EQ(actual.gradient.z, 0.0);
		}
	}

	// checks off-axis results against the analytical kernel at several particle sizes
	TEST(SphKernelTests, MatchesIndependentReferenceAcrossParticleSizes)
	{
		const std::array<glm::vec3, 8> offsets{{
			{ 0.125f, -0.25f, 0.375f }, { -0.5f, 0.25f, -0.125f },
			{ 0.0f, 0.6f, -0.8f }, { 0.75f, -1.0f, 0.5f },
			{ -1.0f, 0.5f, 1.25f }, { 1.5f, 1.5f, 0.25f },
			{ 0.3f, -0.4f, 0.25f }, { 0.9f, -1.2f, 0.0f }
		}};
		for (const float particleSizeF : { 0.25f, 0.7f, 1.0f, 4.0f })
		{
			SCOPED_TRACE(particleSizeF);
			for (uint32_t i = 0; i < offsets.size(); i++)
			{
				SCOPED_TRACE(i);
				const glm::vec3 offset = offsets[i] * particleSizeF;

				ExpectKernelNear(EvaluateKernel(offset, particleSizeF),
					ReferenceKernel(offset, particleSizeF), particleSizeF);
			}
		}
	}

	// neighboring floats on either side of q=1 meet the same weight and derivative
	TEST(SphKernelTests, RemainsContinuousAtPolynomialTransition)
	{
		for (const float distance : { std::nextafter(1.0f, 0.0f), 1.0f, std::nextafter(1.0f, 2.0f) })
		{
			SCOPED_TRACE(distance);
			const glm::vec3 offset{ distance, 0.0f, 0.0f };
			const auto actual = EvaluateKernel(offset);

			ExpectKernelNear(actual, ReferenceKernel(offset));
			EXPECT_NEAR(actual.weight, 0.25 / std::numbers::pi_v<double>, 1e-7);
			EXPECT_NEAR(actual.gradient.x, -0.75 / std::numbers::pi_v<double>, 1e-7);
		}
	}

	// support rejection applies to every axis and to off-axis distances
	TEST(SphKernelTests, ReturnsExactlyZeroAtAndBeyondSupport)
	{
		const std::array<glm::vec3, 7> offsets{{
			{ 2.0f, 0.0f, 0.0f }, { -2.0f, 0.0f, 0.0f }, { 0.0f, 2.0f, 0.0f },
			{ 0.0f, 0.0f, -2.0f }, { std::nextafter(2.0f, 3.0f), 0.0f, 0.0f },
			{ 2.5f, 0.0f, 0.0f }, { 1.5f, 1.5f, 0.0f }
		}};
		for (uint32_t i = 0; i < offsets.size(); i++)
		{
			SCOPED_TRACE(i);
			const auto actual = EvaluateKernel(offsets[i]);

			EXPECT_EQ(actual.weight, 0.0);
			EXPECT_EQ(actual.gradient.x, 0.0);
			EXPECT_EQ(actual.gradient.y, 0.0);
			EXPECT_EQ(actual.gradient.z, 0.0);
		}
	}

	// relative checks retain sensitivity to the tiny nonzero values next to the cutoff
	TEST(SphKernelTests, RetainsValuesImmediatelyInsideSupport)
	{
		const glm::vec3 offset{ std::nextafter(2.0f, 0.0f), 0.0f, 0.0f };
		const auto actual = EvaluateKernel(offset);
		const auto expected = ReferenceKernel(offset);

		EXPECT_GT(actual.weight, 0.0);
		EXPECT_LT(actual.gradient.x, 0.0);
		EXPECT_NEAR(actual.weight / expected.weight, 1.0, 2e-6);
		EXPECT_NEAR(actual.gradient.x / expected.gradient.x, 1.0, 2e-6);
		EXPECT_EQ(actual.gradient.y, 0.0);
		EXPECT_EQ(actual.gradient.z, 0.0);
	}

	// coincident and sub-threshold offsets retain a finite weight and exactly zero gradient
	TEST(SphKernelTests, SuppressesGradientsAtCoincidentAndTinyOffsets)
	{
		const float minDistance = std::sqrt(CPU::KERNEL_GRADIENT_MIN_DIST_SQU_F);
		for (const float distance : { 0.0f, 0.5f * minDistance, std::nextafter(minDistance, 0.0f) })
		{
			SCOPED_TRACE(distance);
			const auto actual = EvaluateKernel({ 0.0f, -distance, 0.0f });

			EXPECT_NEAR(actual.weight, 1.0 / std::numbers::pi_v<double>, 1e-7);
			EXPECT_EQ(actual.gradient.x, 0.0);
			EXPECT_EQ(actual.gradient.y, 0.0);
			EXPECT_EQ(actual.gradient.z, 0.0);
		}
	}

	// the first tested distance above the guard must not be swallowed by an absolute tolerance
	TEST(SphKernelTests, ComputesGradientImmediatelyAboveMinimumDistance)
	{
		const float minDistance = std::sqrt(CPU::KERNEL_GRADIENT_MIN_DIST_SQU_F);
		const float distance = std::nextafter(minDistance, std::numeric_limits<float>::infinity());
		ASSERT_GT(distance * distance, CPU::KERNEL_GRADIENT_MIN_DIST_SQU_F);
		const glm::vec3 offset{ 0.0f, 0.0f, distance };
		const auto actual = EvaluateKernel(offset);
		const auto expected = ReferenceKernel(offset);

		EXPECT_LT(actual.gradient.z, 0.0);
		EXPECT_NEAR(actual.gradient.z / expected.gradient.z, 1.0, 2e-6);
		EXPECT_EQ(actual.gradient.x, 0.0);
		EXPECT_EQ(actual.gradient.y, 0.0);
	}

	// exchanging a particle pair preserves weight and reverses every gradient component
	TEST(SphKernelTests, PreservesWeightAndReversesGradientForOppositeOffsets)
	{
		for (const glm::vec3 offset : { glm::vec3{ 0.25f, -0.5f, 0.125f }, glm::vec3{ 1.0f, -0.75f, 0.5f } })
		{
			const auto forward = EvaluateKernel(offset);
			const auto reverse = EvaluateKernel(-offset);

			EXPECT_EQ(forward.weight, reverse.weight);
			for (int axis = 0; axis < 3; axis++)
			{
				SCOPED_TRACE(axis);
				EXPECT_EQ(forward.gradient[axis], -reverse.gradient[axis]);
			}
		}
	}

	// the radial derivative points opposite the displacement in both polynomial branches
	TEST(SphKernelTests, ComputesRadialGradientForOffAxisOffsets)
	{
		for (const float scale : { 0.5f, 1.5f })
		{
			SCOPED_TRACE(scale);
			const glm::vec3 offset = glm::vec3{ 0.6f, -0.8f, 0.0f } * scale;
			const auto actual = EvaluateKernel(offset);

			ExpectKernelNear(actual, ReferenceKernel(offset));
			EXPECT_LT(actual.gradient.x, 0.0);
			EXPECT_GT(actual.gradient.y, 0.0);
			EXPECT_EQ(actual.gradient.z, 0.0);
			EXPECT_NEAR(actual.gradient.x * offset.y - actual.gradient.y * offset.x, 0.0, 1e-7);
		}
	}

	// different lane masks and branches must remain independent at either SIMD width
	TEST(SphKernelTests, EvaluatesMixedLanesIndependently)
	{
		const std::array<glm::vec3, 8> cases{{
			{ 0.0f, 0.0f, 0.0f }, { 1e-8f, 0.0f, 0.0f },
			{ 0.25f, -0.5f, 0.125f }, { 1.0f, 0.0f, 0.0f },
			{ 0.9f, -1.2f, 0.0f }, { 0.0f, 0.0f, 2.0f },
			{ -2.5f, 0.0f, 0.0f }, { -0.5f, 0.25f, -0.125f }
		}};
		KernelOffsets offsets;
		for (uint32_t lane = 0; lane < CPU::N_SIMD_WIDTH; lane++)
		{
			offsets[lane] = cases[lane % cases.size()];
		}
		const auto actual = EvaluateKernels(offsets);

		for (uint32_t lane = 0; lane < CPU::N_SIMD_WIDTH; lane++)
		{
			SCOPED_TRACE(lane);
			auto expected = ReferenceKernel(offsets[lane]);
			if (lane % cases.size() == 1) { expected.gradient = glm::dvec3{ 0.0 }; }
			ExpectKernelNear(actual[lane], expected);
			if (lane % cases.size() == 0 || lane % cases.size() == 1 ||
				lane % cases.size() == 5 || lane % cases.size() == 6)
			{
				EXPECT_EQ(actual[lane].gradient.x, 0.0);
				EXPECT_EQ(actual[lane].gradient.y, 0.0);
				EXPECT_EQ(actual[lane].gradient.z, 0.0);
			}
			if (lane % cases.size() == 5 || lane % cases.size() == 6)
			{
				EXPECT_EQ(actual[lane].weight, 0.0);
			}
		}
	}
}
