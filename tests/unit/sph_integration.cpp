// sph_integration.cpp

// external
#include <array>
#include <cstdint>
#include <glm/vec3.hpp>
#include <gtest/gtest.h>
#include <optional>
#include <tbb/partitioner.h>
#include <vector>

// internal
#include "source/simulation/CPU/backend/sph_solver.h"



namespace
{
	struct IntegrationParticle
	{
		glm::vec3 position;
		glm::vec3 predictedVelocity;
		glm::vec3 pressureAcceleration;
	};

	struct IntegratedParticle
	{
		glm::vec3 position;
		glm::vec3 velocity;
	};

	std::vector<IntegratedParticle> EvaluateIntegration(const std::vector<IntegrationParticle>& particles,
		const float timeStep, const std::optional<uint32_t> particleLimit = {})
	{
		constexpr float guard = -12345.0f;
		const uint32_t N_F = static_cast<uint32_t>(particles.size());
		std::array<std::vector<float>, 3> positions, velocities, predictedVelocities, pressureAccelerations;
		for (int axis = 0; axis < 3; axis++)
		{
			positions[axis].assign(N_F + 2, guard);
			velocities[axis].assign(N_F + 2, guard);
			predictedVelocities[axis].resize(N_F);
			pressureAccelerations[axis].resize(N_F);
			for (uint32_t i = 0; i < N_F; i++)
			{
				positions[axis][i + 1] = particles[i].position[axis];
				// old velocities differ from predictions and must be overwritten
				velocities[axis][i + 1] = static_cast<float>(100 + axis);
				predictedVelocities[axis][i] = particles[i].predictedVelocity[axis];
				pressureAccelerations[axis][i] = particles[i].pressureAcceleration[axis];
			}
		}
		const auto originalPredictedVelocities = predictedVelocities;
		const auto originalPressureAccelerations = pressureAccelerations;
		CPU::Integration(
			predictedVelocities[0].data(), predictedVelocities[1].data(), predictedVelocities[2].data(),
			pressureAccelerations[0].data(), pressureAccelerations[1].data(), pressureAccelerations[2].data(),
			timeStep, particleLimit.value_or(N_F),
			positions[0].data() + 1, positions[1].data() + 1, positions[2].data() + 1,
			velocities[0].data() + 1, velocities[1].data() + 1, velocities[2].data() + 1);

		EXPECT_EQ(predictedVelocities, originalPredictedVelocities);
		EXPECT_EQ(pressureAccelerations, originalPressureAccelerations);
		for (int axis = 0; axis < 3; axis++)
		{
			SCOPED_TRACE(axis);
			EXPECT_EQ(positions[axis].front(), guard);
			EXPECT_EQ(positions[axis].back(), guard);
			EXPECT_EQ(velocities[axis].front(), guard);
			EXPECT_EQ(velocities[axis].back(), guard);
		}
		std::vector<IntegratedParticle> result(N_F);
		for (uint32_t i = 0; i < N_F; i++)
		{
			for (int axis = 0; axis < 3; axis++)
			{
				result[i].position[axis] = positions[axis][i + 1];
				result[i].velocity[axis] = velocities[axis][i + 1];
			}
		}
		return result;
	}

	void ExpectIntegrationEqual(const std::vector<IntegratedParticle>& actual,
		const std::vector<IntegratedParticle>& expected)
	{
		ASSERT_EQ(actual.size(), expected.size());
		for (uint32_t i = 0; i < expected.size(); i++)
		{
			SCOPED_TRACE(i);
			for (int axis = 0; axis < 3; axis++)
			{
				SCOPED_TRACE(axis);
				EXPECT_FLOAT_EQ(actual[i].position[axis], expected[i].position[axis]);
				EXPECT_FLOAT_EQ(actual[i].velocity[axis], expected[i].velocity[axis]);
			}
		}
	}

	// pressure corrects predicted velocity before the updated velocity advances position
	TEST(SphIntegrationTests, ComputesKnownPressureCorrectionAndPosition)
	{
		const auto actual = EvaluateIntegration({
			{ { 1.0f, -4.0f, 10.0f }, { 3.0f, -2.0f, 0.5f }, { 4.0f, 8.0f, -6.0f } } }, 0.25f);

		ExpectIntegrationEqual(actual, { { { 2.0f, -4.0f, 9.75f }, { 4.0f, 0.0f, -1.0f } } });
	}

	// zero pressure acceleration gives constant-velocity motion, including stationary particles
	TEST(SphIntegrationTests, AdvancesAtConstantVelocityWithoutPressureAcceleration)
	{
		const auto actual = EvaluateIntegration({
			{ { 1.0f, -2.0f, 3.0f }, { 2.0f, -4.0f, 6.0f }, { 0.0f, 0.0f, 0.0f } },
			{ { -7.0f, 8.0f, -9.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } } }, 0.5f);

		ExpectIntegrationEqual(actual, {
			{ { 2.0f, -4.0f, 6.0f }, { 2.0f, -4.0f, 6.0f } },
			{ { -7.0f, 8.0f, -9.0f }, { 0.0f, 0.0f, 0.0f } } });
	}

	// a zero time step preserves positions while replacing old velocities with predictions
	TEST(SphIntegrationTests, PreservesPositionsAndCopiesPredictedVelocitiesAtZeroTimeStep)
	{
		const auto actual = EvaluateIntegration({
			{ { 1.0f, -4.0f, 10.0f }, { 3.0f, -2.0f, 0.5f }, { 4.0f, 8.0f, -6.0f } },
			{ { -2.0f, 3.0f, -4.0f }, { -1.0f, 4.0f, -3.0f }, { 8.0f, -4.0f, 12.0f } } }, 0.0f);

		ExpectIntegrationEqual(actual, {
			{ { 1.0f, -4.0f, 10.0f }, { 3.0f, -2.0f, 0.5f } },
			{ { -2.0f, 3.0f, -4.0f }, { -1.0f, 4.0f, -3.0f } } });
	}

	// hand-derived results at several time steps catch missing or repeated time factors
	TEST(SphIntegrationTests, AppliesTimeStepToVelocityAndPosition)
	{
		const std::array<float, 3> timeSteps{ 0.125f, 0.5f, 2.0f };
		const std::array<IntegratedParticle, 3> expected{{
			{ { 1.375f, -2.25f, 3.375f }, { 3.0f, -2.0f, 3.0f } },
			{ { 4.0f, 0.0f, 0.0f }, { 6.0f, 4.0f, -6.0f } },
			{ { 37.0f, 54.0f, -81.0f }, { 18.0f, 28.0f, -42.0f } }
		}};
		for (uint32_t i = 0; i < timeSteps.size(); i++)
		{
			SCOPED_TRACE(timeSteps[i]);
			const auto actual = EvaluateIntegration({
				{ { 1.0f, -2.0f, 3.0f }, { 2.0f, -4.0f, 6.0f }, { 8.0f, 16.0f, -24.0f } } }, timeSteps[i]);

			ExpectIntegrationEqual(actual, { expected[i] });
		}
	}

	// distinct neighboring particles retain independent updates across small and larger arrays
	TEST(SphIntegrationTests, IntegratesEveryParticleIndependently)
	{
		const std::array<IntegrationParticle, 3> particles{{
			{ { 1.0f, -4.0f, 10.0f }, { 3.0f, -2.0f, 0.5f }, { 4.0f, 8.0f, -6.0f } },
			{ { -2.0f, 3.0f, -4.0f }, { -1.0f, 4.0f, -3.0f }, { 8.0f, -4.0f, 12.0f } },
			{ { 8.0f, -6.0f, 2.0f }, { 0.0f, -2.0f, 8.0f }, { -4.0f, 0.0f, -8.0f } }
		}};
		const std::array<IntegratedParticle, 3> expected{{
			{ { 2.0f, -4.0f, 9.75f }, { 4.0f, 0.0f, -1.0f } },
			{ { -1.75f, 3.75f, -4.0f }, { 1.0f, 3.0f, 0.0f } },
			{ { 7.75f, -6.5f, 3.5f }, { -1.0f, -2.0f, 6.0f } }
		}};
		for (const uint32_t N_F : { 3u, 17u, 257u })
		{
			SCOPED_TRACE(N_F);
			std::vector<IntegrationParticle> input;
			std::vector<IntegratedParticle> output;
			for (uint32_t i = 0; i < N_F; i++)
			{
				const float index = static_cast<float>(i);
				auto particle = particles[i % particles.size()];
				particle.position += glm::vec3{ 8.0f * index, 16.0f * index, -32.0f * index };
				particle.predictedVelocity += glm::vec3{ 4.0f * index, -8.0f * index, 16.0f * index };
				input.push_back(particle);
				// at dt=1/4, the added velocity shifts position by (index, -2*index, 4*index)
				auto integrated = expected[i % expected.size()];
				integrated.position += glm::vec3{ 9.0f * index, 14.0f * index, -28.0f * index };
				integrated.velocity += glm::vec3{ 4.0f * index, -8.0f * index, 16.0f * index };
				output.push_back(integrated);
			}

			ExpectIntegrationEqual(EvaluateIntegration(input, 0.25f), output);
		}
	}

	// owned prefixes update independently while halo positions and old velocities remain unchanged
	TEST(SphIntegrationTests, IntegratesOwnedPrefixWithoutChangingHaloParticles)
	{
		const std::vector<IntegrationParticle> particles{
			{ { 1.0f, -4.0f, 10.0f }, { 3.0f, -2.0f, 0.5f }, { 4.0f, 8.0f, -6.0f } },
			{ { -2.0f, 3.0f, -4.0f }, { -1.0f, 4.0f, -3.0f }, { 8.0f, -4.0f, 12.0f } },
			{ { 8.0f, -6.0f, 2.0f }, { 0.0f, -2.0f, 8.0f }, { -4.0f, 0.0f, -8.0f } } };
		const std::vector<IntegratedParticle> integrated{
			{ { 2.0f, -4.0f, 9.75f }, { 4.0f, 0.0f, -1.0f } },
			{ { -1.75f, 3.75f, -4.0f }, { 1.0f, 3.0f, 0.0f } },
			{ { 7.75f, -6.5f, 3.5f }, { -1.0f, -2.0f, 6.0f } } };
		for (const uint32_t N_ownedF : { 0u, 1u, 2u })
		{
			SCOPED_TRACE(N_ownedF);
			const auto actual = EvaluateIntegration(particles, 0.25f, N_ownedF);
			auto expected = integrated;
			for (uint32_t i = N_ownedF; i < particles.size(); i++)
			{
				expected[i] = { particles[i].position, { 100.0f, 101.0f, 102.0f } };
				for (int axis = 0; axis < 3; axis++)
				{
					SCOPED_TRACE(i);
					SCOPED_TRACE(axis);
					EXPECT_EQ(actual[i].position[axis], expected[i].position[axis]);
					EXPECT_EQ(actual[i].velocity[axis], expected[i].velocity[axis]);
				}
			}
			ExpectIntegrationEqual(actual, expected);
		}
	}

	// an empty population does no work and leaves every output guard intact
	TEST(SphIntegrationTests, HandlesEmptyPopulations)
	{
		EXPECT_TRUE(EvaluateIntegration({}, 0.25f).empty());
	}
}
