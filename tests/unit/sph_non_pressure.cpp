// sph_non_pressure.cpp

// external
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/vec3.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <tbb/partitioner.h>
#include <vector>

// internal
#include "source/simulation/CPU/backend/sph_solver.h"
#include "source/simulation/core/solver_constants.h"



namespace
{
	struct NonPressureParticle
	{
		glm::vec3 position;
		glm::vec3 velocity{ 0.0f };
		float density = 1.0f;
		float volume = 1.0f;
	};

	// populations and neighbor categories follow fluid, boundary, rigid order
	using NonPressureScene = std::array<std::vector<NonPressureParticle>, 3>;
	using NonPressureNeighbors = std::array<std::vector<std::vector<uint32_t>>, 3>;
	constexpr float untouchedVelocity = -12345.0f;
	constexpr std::array writeChunks{
		&Sim::I_NeighborListSELL::FFChunkWrite, &Sim::I_NeighborListSELL::FBChunkWrite,
		&Sim::I_NeighborListSELL::FRChunkWrite };
	constexpr std::array finishChunks{
		&Sim::I_NeighborListSELL::FinishFFChunkWrite, &Sim::I_NeighborListSELL::FinishFBChunkWrite,
		&Sim::I_NeighborListSELL::FinishFRChunkWrite };

	Sim::SimParams MakeParams(const float particleSizeF = 1.0f)
	{
		Sim::SimParams params;
		params.timeStep = 0.125f;
		params.particleSizeF = particleSizeF;
		params.kernelRadiusSqu = 4.0f * particleSizeF * particleSizeF;
		params.massF = 2.0f;
		params.viscosityF = 0.1f;
		params.viscosityB = 0.2f;
		params.gravity = glm::vec3(0.0f);
		return params;
	}

	NonPressureNeighbors MakeNeighbors(const uint32_t N_F)
	{
		NonPressureNeighbors neighbors;
		for (auto& category : neighbors) { category.resize(N_F); }
		return neighbors;
	}

	NonPressureScene MakeMixedScene(const uint32_t N_F)
	{
		NonPressureScene scene;
		for (uint32_t i = 0; i < N_F; i++)
		{
			scene[0].push_back({ { 0.25f * (i % 7) - 0.5f, 0.125f * (i % 3), 0.2f * (i % 5) },
				{ 0.5f * (i % 3) - 1.0f, 0.25f * (i % 5), -0.125f * (i % 7) },
				static_cast<float>(2 + i % 5) });
		}
		scene[1] = {
			{ { 0.0f, 0.0f, 0.0f }, glm::vec3(0.0f), 1.0f, 0.25f },
			{ { 0.5f, -0.25f, 0.75f }, glm::vec3(0.0f), 1.0f, 0.5f },
			{ { -0.25f, 0.5f, 0.0f }, glm::vec3(0.0f), 1.0f, 0.0f },
			{ { 1.0f, 0.0f, -0.5f }, glm::vec3(0.0f), 1.0f, 0.75f } };
		scene[2] = {
			{ { 0.0f, 0.0f, 0.0f }, { -0.5f, 1.0f, 0.25f }, 1.0f, 0.5f },
			{ { -0.5f, 0.25f, 0.0f }, { 0.0f, -0.5f, 0.5f }, 1.0f, 0.25f },
			{ { 0.5f, 1.0f, 0.25f }, { 1.0f, 0.25f, -0.5f }, 1.0f, 0.0f },
			{ { 0.0f, -0.25f, 1.0f }, { -1.0f, 0.5f, 0.0f }, 1.0f, 0.75f } };
		return scene;
	}

	NonPressureNeighbors MakeMixedNeighbors(const NonPressureScene& scene)
	{
		auto neighbors = MakeNeighbors(static_cast<uint32_t>(scene[0].size()));
		for (uint32_t category = 0; category < neighbors.size(); category++)
		{
			const uint32_t N_target = static_cast<uint32_t>(scene[category].size());
			for (uint32_t i = 0; i < scene[0].size(); i++)
			{
				const uint32_t N_nbrs = std::min((i + 2 * category) % 4, N_target);
				for (uint32_t k = 0; k < N_nbrs; k++)
				{
					neighbors[category][i].push_back((3 * i + category + N_target - 1 - k) % N_target);
				}
			}
		}
		return neighbors;
	}

	// independent radial derivative and viscosity sum in double precision
	std::vector<glm::dvec3> ReferenceVelocities(const NonPressureScene& scene,
		const NonPressureNeighbors& neighbors, const Sim::SimParams& params)
	{
		const double h = params.particleSizeF;
		const double normalization = 1.0 / (std::numbers::pi_v<double> * h * h * h * h);
		std::vector<glm::dvec3> result(scene[0].size());
		for (uint32_t i = 0; i < result.size(); i++)
		{
			const glm::dvec3 position(scene[0][i].position);
			const glm::dvec3 velocity(scene[0][i].velocity);
			glm::dvec3 acceleration(params.gravity);
			for (uint32_t category = 0; category < neighbors.size(); category++)
			{
				for (const uint32_t j : neighbors[category][i])
				{
					const auto& target = scene[category][j];
					const glm::dvec3 displacement = position - glm::dvec3(target.position);
					const double distanceSqu = displacement.x * displacement.x
						+ displacement.y * displacement.y + displacement.z * displacement.z;
					if (distanceSqu <= 1e-14 || distanceSqu >= params.kernelRadiusSqu) { continue; }
					const double distance = std::sqrt(distanceSqu);
					const double q = distance / h;
					const double derivative = q < 1.0
						? -3.0 * q + 2.25 * q * q : -0.75 * (2.0 - q) * (2.0 - q);
					const glm::dvec3 gradient = displacement * (normalization * derivative / distance);
					const glm::dvec3 relativeVelocity = category == 1
						? velocity : velocity - glm::dvec3(target.velocity);
					const double dot = relativeVelocity.x * displacement.x
						+ relativeVelocity.y * displacement.y + relativeVelocity.z * displacement.z;
					const double weight = category == 0
						? params.massF / std::max(static_cast<double>(target.density), 1e-7) : target.volume;
					const double viscosity = category == 0 ? params.viscosityF : params.viscosityB;
					acceleration += gradient * (10.0 * viscosity * weight * dot / (distanceSqu + 0.01 * h * h));
				}
			}
			result[i] = velocity + static_cast<double>(params.timeStep) * acceleration;
		}
		return result;
	}

	void ExpectVelocitiesNear(const std::vector<glm::vec3>& actual, const std::vector<glm::dvec3>& expected)
	{
		ASSERT_EQ(actual.size(), expected.size());
		for (uint32_t i = 0; i < expected.size(); i++)
		{
			SCOPED_TRACE(i);
			for (int axis = 0; axis < 3; axis++)
			{
				SCOPED_TRACE(axis);
				EXPECT_NEAR(actual[i][axis], expected[i][axis], 2e-6 + 8e-6 * std::abs(expected[i][axis]));
			}
		}
	}

	class SphNonPressureTests : public testing::TestWithParam<bool>
	{
	protected:
		void SetUp() override
		{
			if (GetParam()) { lists_ = std::make_unique<Sim::NeighborLists_ChunkedSELL_AdaptCompress_CPU>(); }
			else { lists_ = std::make_unique<Sim::NeighborLists_ChunkedSELL_Explicit_CPU>(); }
		}

		void BuildLists(const NonPressureScene& scene, const NonPressureNeighbors& neighbors)
		{
			const uint32_t N_F = static_cast<uint32_t>(scene[0].size());
			const uint32_t N_B = static_cast<uint32_t>(scene[1].size());
			const uint32_t N_R = static_cast<uint32_t>(scene[2].size());
			Sim::SimParams params;
			params.expctdAvgNbrCountHardMax_FF = 0;
			params.expctdAvgNbrCountHardMax_FB = 0;
			lists_->Resize(N_F, N_B, N_R, params);
			lists_->PrepareFluidCountWrite(true);
			for (uint32_t chunk = 0; chunk < lists_->ChunkNumTotal(); chunk++)
			{
				std::array<std::array<uint32_t, CPU::N_SIMD_WIDTH>, 3> counts{};
				std::array<uint32_t, 3> minRef;
				minRef.fill(std::numeric_limits<uint32_t>::max());
				std::array<uint32_t, 3> maxRef{};
				const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
				const uint32_t end = std::min(begin + CPU::N_SIMD_WIDTH, N_F);
				for (uint32_t category = 0; category < neighbors.size(); category++)
				{
					for (uint32_t i = begin; i < end; i++)
					{
						counts[category][i - begin] = static_cast<uint32_t>(neighbors[category][i].size());
						for (const uint32_t j : neighbors[category][i])
						{
							minRef[category] = std::min(minRef[category], j);
							maxRef[category] = std::max(maxRef[category], j);
						}
					}
				}
				lists_->CommitFluidChunkCounts(chunk,
					{ counts[0].data(), minRef[0], maxRef[0] },
					{ counts[1].data(), minRef[1], maxRef[1] },
					{ counts[2].data(), minRef[2], maxRef[2] });
			}
			for (uint32_t i = 0; i < N_R; i++)
			{
				lists_->SetNbrCountRF(i, 0);
				lists_->SetNbrCountRB(i, 0);
				lists_->SetNbrCountRR(i, 0);
			}
			lists_->Finalize_Count();
			for (uint32_t chunk = 0; chunk < lists_->ChunkNumTotal(); chunk++)
			{
				for (uint32_t category = 0; category < neighbors.size(); category++)
				{
					const auto view = (lists_.get()->*writeChunks[category])(chunk);
					std::array<uint32_t, CPU::N_SIMD_WIDTH> counts{};
					for (uint32_t i = view.begin; i < view.end; i++)
					{
						const auto& row = neighbors[category][i];
						counts[i - view.begin] = static_cast<uint32_t>(row.size());
						for (uint32_t k = 0; k < row.size(); k++) { view.StoreNeighbor(i - view.begin, k, row[k]); }
					}
					(lists_.get()->*finishChunks[category])(chunk, counts.data());
				}
			}
			lists_->Finalize_Fill();
		}

		std::vector<glm::vec3> EvaluateNonPressure(const NonPressureScene& scene,
			const NonPressureNeighbors& neighbors, const Sim::SimParams& params = MakeParams(),
			const std::span<const uint32_t> selectedChunks = {}, const bool useSelectedChunks = false,
			const std::optional<uint32_t> particleLimit = {})
		{
			BuildLists(scene, neighbors);
			std::array<std::array<std::vector<float>, 3>, 3> positions, velocities;
			std::array<std::vector<float>, 3> volumes;
			std::vector<float> density;
			for (uint32_t category = 0; category < scene.size(); category++)
			{
				for (const auto& particle : scene[category])
				{
					for (int axis = 0; axis < 3; axis++)
					{
						positions[category][axis].push_back(particle.position[axis]);
						velocities[category][axis].push_back(particle.velocity[axis]);
					}
					if (category == 0) { density.push_back(particle.density); }
					else { volumes[category].push_back(particle.volume); }
				}
			}
			const auto originalPositions = positions;
			const auto originalVelocities = velocities;
			const auto originalVolumes = volumes;
			const auto originalDensity = density;
			const uint32_t N_F = static_cast<uint32_t>(scene[0].size());
			std::array<std::vector<float>, 3> predicted;
			for (auto& axis : predicted) { axis.assign(N_F + CPU::N_SIMD_WIDTH + 1, untouchedVelocity); }
			const auto constants = Sim::BuildSolverConstants(params.particleSizeF, params.restDensity);
			CPU::NonPressureAcceleration_ChunkedSELL(
				positions[0][0].data(), positions[0][1].data(), positions[0][2].data(),
				positions[1][0].data(), positions[1][1].data(), positions[1][2].data(),
				positions[2][0].data(), positions[2][1].data(), positions[2][2].data(),
				velocities[0][0].data(), velocities[0][1].data(), velocities[0][2].data(),
				velocities[2][0].data(), velocities[2][1].data(), velocities[2][2].data(),
				density.data(), volumes[1].data(), volumes[2].data(), *lists_,
				params.timeStep, constants.sizeInvF, constants.sizeSquF, params.kernelRadiusSqu,
				constants.cubicSplineAlpha, params.massF, params.viscosityF, params.viscosityB,
				params.gravity.x, params.gravity.y, params.gravity.z, particleLimit.value_or(N_F),
				predicted[0].data() + 1, predicted[1].data() + 1, predicted[2].data() + 1,
				selectedChunks, useSelectedChunks);

			EXPECT_EQ(positions, originalPositions);
			EXPECT_EQ(velocities, originalVelocities);
			EXPECT_EQ(volumes, originalVolumes);
			EXPECT_EQ(density, originalDensity);
			for (int axis = 0; axis < 3; axis++)
			{
				SCOPED_TRACE(axis);
				EXPECT_EQ(predicted[axis].front(), untouchedVelocity);
				for (uint32_t i = N_F + 1; i < predicted[axis].size(); i++)
				{
					EXPECT_EQ(predicted[axis][i], untouchedVelocity);
				}
			}
			std::vector<glm::vec3> result(N_F);
			for (uint32_t i = 0; i < N_F; i++)
			{
				for (int axis = 0; axis < 3; axis++) { result[i][axis] = predicted[axis][i + 1]; }
			}
			return result;
		}

		std::unique_ptr<Sim::I_NeighborListSELL> lists_;
	};

	// without neighbors, gravity alone updates each velocity component
	TEST_P(SphNonPressureTests, ComputesKnownGravityOnlyMotion)
	{
		NonPressureScene scene;
		scene[0] = { { { 0.0f, 0.0f, 0.0f }, { 1.0f, -2.0f, 3.0f } },
			{ { 2.0f, 3.0f, -1.0f }, { -3.0f, 4.0f, -5.0f } } };
		auto params = MakeParams();
		params.gravity = { 2.0f, -4.0f, 6.0f };

		ExpectVelocitiesNear(EvaluateNonPressure(scene, MakeNeighbors(2), params),
			{ { 1.25, -2.5, 3.75 }, { -2.75, 3.5, -4.25 } });
	}

	// a pair at one smoothing length pins the gradient, mass, and neighbor-density weighting
	TEST_P(SphNonPressureTests, ComputesKnownFluidViscosity)
	{
		for (const float massF : { 1.0f, 3.0f })
		{
			for (const float densityJ : { 2.0f, 5.0f })
			{
				SCOPED_TRACE(massF);
				SCOPED_TRACE(densityJ);
				NonPressureScene scene;
				scene[0] = { { { 0.0f, 0.0f, 0.0f }, { 1.0f, 2.0f, 3.0f }, 9.0f },
					{ { 1.0f, 0.0f, 0.0f }, { -1.0f, 2.0f, 3.0f }, densityJ } };
				auto neighbors = MakeNeighbors(2);
				neighbors[0][0] = { 1 };
				auto params = MakeParams();
				params.massF = massF;
				const double decrement = 3.0 * massF / (16.0 * std::numbers::pi_v<double> * densityJ * 1.01);

				ExpectVelocitiesNear(EvaluateNonPressure(scene, neighbors, params),
					{ { 1.0 - decrement, 2.0, 3.0 }, { -1.0, 2.0, 3.0 } });
			}
		}
	}

	// zero and sub-floor densities use the floor, while densities above it keep their own weighting
	TEST_P(SphNonPressureTests, ClampsSmallFluidNeighborDensities)
	{
		const std::array<float, 4> densities{ 0.0f, 1e-8f, 1e-7f, 2e-7f };
		const std::array<double, 4> effectiveDensities{ 1e-7, 1e-7, 1e-7, 2e-7 };
		for (uint32_t i = 0; i < densities.size(); i++)
		{
			SCOPED_TRACE(densities[i]);
			NonPressureScene scene;
			scene[0] = { { { 0.0f, 0.0f, 0.0f }, { 1.0f, 2.0f, 3.0f }, 9.0f },
				{ { 1.0f, 0.0f, 0.0f }, { -1.0f, 2.0f, 3.0f }, densities[i] } };
			auto neighbors = MakeNeighbors(2);
			neighbors[0][0] = { 1 };
			// at h=1 and dt=1/8, mass=2 gives this analytical velocity decrement
			const double decrement = 6.0 / (16.0 * std::numbers::pi_v<double> * effectiveDensities[i] * 1.01);
			const auto actual = EvaluateNonPressure(scene, neighbors);

			for (const auto& velocity : actual)
			{
				for (int axis = 0; axis < 3; axis++) { EXPECT_TRUE(std::isfinite(velocity[axis])); }
			}
			ExpectVelocitiesNear(actual, { { 1.0 - decrement, 2.0, 3.0 }, { -1.0, 2.0, 3.0 } });
		}
	}

	// stationary boundaries and moving rigid particles use volume and boundary viscosity
	TEST_P(SphNonPressureTests, ComputesKnownBoundaryAndRigidViscosity)
	{
		for (uint32_t category = 1; category < 3; category++)
		{
			for (const float volume : { 0.5f, 2.0f })
			{
				SCOPED_TRACE(category);
				SCOPED_TRACE(volume);
				NonPressureScene scene;
				scene[0] = { { { 0.0f, 0.0f, 0.0f }, { 1.0f, 2.0f, 3.0f }, 7.0f } };
				scene[category] = { { { 1.0f, 0.0f, 0.0f }, { -1.0f, 2.0f, 3.0f }, 11.0f, volume } };
				auto neighbors = MakeNeighbors(1);
				neighbors[category][0] = { 0 };
				auto params = MakeParams();
				params.massF = 9.0f;
				const double relativeSpeed = category == 1 ? 1.0 : 2.0;
				const double decrement = 3.0 * volume * relativeSpeed / (16.0 * std::numbers::pi_v<double> * 1.01);

				ExpectVelocitiesNear(EvaluateNonPressure(scene, neighbors, params), { { 1.0 - decrement, 2.0, 3.0 } });
			}
		}
	}

	// mixed populations and uneven rows must agree across full and partial SIMD chunks
	TEST_P(SphNonPressureTests, MatchesScalarReferenceAcrossSimdBoundaries)
	{
		auto params = MakeParams();
		params.gravity = { 0.5f, -1.0f, 0.25f };
		for (const uint32_t N_F : { 1u, CPU::N_SIMD_WIDTH - 1, CPU::N_SIMD_WIDTH,
			CPU::N_SIMD_WIDTH + 1, 2 * CPU::N_SIMD_WIDTH + 3 })
		{
			SCOPED_TRACE(N_F);
			const auto scene = MakeMixedScene(N_F);
			const auto neighbors = MakeMixedNeighbors(scene);

			ExpectVelocitiesNear(EvaluateNonPressure(scene, neighbors, params), ReferenceVelocities(scene, neighbors, params));
		}
	}

	// size regularization and both viscosity coefficients must retain their independent effects
	TEST_P(SphNonPressureTests, MatchesScalarReferenceAtDifferentScales)
	{
		for (const float h : { 0.5f, 2.0f })
		{
			SCOPED_TRACE(h);
			auto scene = MakeMixedScene(CPU::N_SIMD_WIDTH + 1);
			for (auto& population : scene)
			{
				for (auto& particle : population) { particle.position *= h; }
			}
			const auto neighbors = MakeMixedNeighbors(scene);
			for (const bool fluidViscosity : { false, true })
			{
				auto params = MakeParams(h);
				params.viscosityF = fluidViscosity ? 0.3f : 0.0f;
				params.viscosityB = fluidViscosity ? 0.0f : 0.4f;
				params.timeStep = 0.0625f;

				ExpectVelocitiesNear(EvaluateNonPressure(scene, neighbors, params), ReferenceVelocities(scene, neighbors, params));
			}
		}
	}

	// equal fluid and rigid velocities eliminate their viscosity contributions
	TEST_P(SphNonPressureTests, PreservesUniformVelocityWithoutGravity)
	{
		auto scene = MakeMixedScene(CPU::N_SIMD_WIDTH + 1);
		for (auto& population : scene)
		{
			for (auto& particle : population) { particle.velocity = { 1.0f, -2.0f, 3.0f }; }
		}
		auto neighbors = MakeMixedNeighbors(scene);
		for (auto& row : neighbors[1]) { row.clear(); }

		ExpectVelocitiesNear(EvaluateNonPressure(scene, neighbors),
			std::vector<glm::dvec3>(scene[0].size(), { 1.0, -2.0, 3.0 }));
	}

	// tangential relative motion has zero projection onto the particle separation
	TEST_P(SphNonPressureTests, IgnoresRelativeMotionPerpendicularToSeparation)
	{
		for (uint32_t category = 0; category < 3; category++)
		{
			for (int axis = 0; axis < 3; axis++)
			{
				SCOPED_TRACE(category);
				SCOPED_TRACE(axis);
				glm::vec3 position(0.0f), velocity(0.0f);
				position[axis] = 1.0f;
				velocity[(axis + 1) % 3] = 2.0f;
				NonPressureScene scene;
				scene[0] = { { glm::vec3(0.0f), velocity } };
				scene[category].push_back({ position });
				auto neighbors = MakeNeighbors(static_cast<uint32_t>(scene[0].size()));
				neighbors[category][0] = { category == 0 ? 1u : 0u };
				std::vector<glm::dvec3> expected(scene[0].size(), glm::dvec3(0.0));
				expected[0] = glm::dvec3(velocity);

				ExpectVelocitiesNear(EvaluateNonPressure(scene, neighbors), expected);
			}
		}
	}

	// zero viscosity leaves only gravity even when moving neighbors are present
	TEST_P(SphNonPressureTests, AppliesOnlyGravityWhenViscositiesAreZero)
	{
		const auto scene = MakeMixedScene(CPU::N_SIMD_WIDTH + 1);
		const auto neighbors = MakeMixedNeighbors(scene);
		auto params = MakeParams();
		params.viscosityF = 0.0f;
		params.viscosityB = 0.0f;
		params.gravity = { 2.0f, -4.0f, 6.0f };
		std::vector<glm::dvec3> expected;
		for (const auto& particle : scene[0])
		{
			expected.push_back(glm::dvec3(particle.velocity) + glm::dvec3(0.25, -0.5, 0.75));
		}

		ExpectVelocitiesNear(EvaluateNonPressure(scene, neighbors, params), expected);
	}

	// a zero time step copies current velocities despite nonzero gravity and viscosity
	TEST_P(SphNonPressureTests, CopiesCurrentVelocitiesAtZeroTimeStep)
	{
		const auto scene = MakeMixedScene(CPU::N_SIMD_WIDTH + 1);
		const auto neighbors = MakeMixedNeighbors(scene);
		auto params = MakeParams();
		params.timeStep = 0.0f;
		params.gravity = { 2.0f, -4.0f, 6.0f };
		std::vector<glm::dvec3> expected;
		for (const auto& particle : scene[0]) { expected.push_back(glm::dvec3(particle.velocity)); }

		ExpectVelocitiesNear(EvaluateNonPressure(scene, neighbors, params), expected);
	}

	// coincident particles and neighbors outside support contribute no viscosity
	TEST_P(SphNonPressureTests, HandlesZeroDistanceAndKernelCutoff)
	{
		for (const float distance : { 0.0f, 2.0f, 2.5f })
		{
			SCOPED_TRACE(distance);
			NonPressureScene scene;
			scene[0] = { { { 0.0f, 0.0f, 0.0f }, { 1.0f, -2.0f, 3.0f } },
				{ { distance, 0.0f, 0.0f }, { -3.0f, 4.0f, -5.0f } } };
			scene[1] = { { { 0.0f, distance, 0.0f } } };
			scene[2] = { { { 0.0f, 0.0f, distance }, { -1.0f, 1.0f, 2.0f } } };
			auto neighbors = MakeNeighbors(2);
			neighbors[0][0] = { 0, 1 };
			neighbors[1][0] = { 0 };
			neighbors[2][0] = { 0 };

			ExpectVelocitiesNear(EvaluateNonPressure(scene, neighbors), { { 1.0, -2.0, 3.0 }, { -3.0, 4.0, -5.0 } });
		}
	}

	// empty fluid populations leave output guards untouched with or without other populations
	TEST_P(SphNonPressureTests, HandlesEmptyFluidPopulations)
	{
		EXPECT_TRUE(EvaluateNonPressure({}, MakeNeighbors(0)).empty());
		EXPECT_TRUE(EvaluateNonPressure(MakeMixedScene(0), MakeNeighbors(0)).empty());
	}

	// noncontiguous selections update only the requested chunks, including a final partial chunk
	TEST_P(SphNonPressureTests, UpdatesOnlySelectedChunks)
	{
		const auto scene = MakeMixedScene(3 * CPU::N_SIMD_WIDTH + 1);
		const auto neighbors = MakeMixedNeighbors(scene);
		const auto params = MakeParams();
		const std::array<uint32_t, 2> selectedChunks{ 3, 1 };
		const auto actual = EvaluateNonPressure(scene, neighbors, params, selectedChunks, true);
		auto expected = ReferenceVelocities(scene, neighbors, params);
		for (uint32_t i = 0; i < expected.size(); i++)
		{
			if (i / CPU::N_SIMD_WIDTH == 1 || i / CPU::N_SIMD_WIDTH == 3) { continue; }
			expected[i] = glm::dvec3(untouchedVelocity);
			for (int axis = 0; axis < 3; axis++) { EXPECT_EQ(actual[i][axis], untouchedVelocity); }
		}

		ExpectVelocitiesNear(actual, expected);
	}

	// an empty selection does no work only when selected-chunk processing is enabled
	TEST_P(SphNonPressureTests, HonorsSelectedChunkMode)
	{
		const auto scene = MakeMixedScene(CPU::N_SIMD_WIDTH + 1);
		const auto neighbors = MakeMixedNeighbors(scene);
		const auto params = MakeParams();
		const auto actual = EvaluateNonPressure(scene, neighbors, params, {}, true);
		for (const auto& velocity : actual)
		{
			for (int axis = 0; axis < 3; axis++) { EXPECT_EQ(velocity[axis], untouchedVelocity); }
		}
		const std::array<uint32_t, 1> selectedChunks{ 1 };

		ExpectVelocitiesNear(EvaluateNonPressure(scene, neighbors, params, selectedChunks, false),
			ReferenceVelocities(scene, neighbors, params));
	}

	// an owned prefix can gather halo velocities and densities without updating halo destinations
	TEST_P(SphNonPressureTests, ComputesOwnedPrefixUsingHaloNeighbors)
	{
		const uint32_t N_F = 2 * CPU::N_SIMD_WIDTH + 3;
		const uint32_t N_ownedF = CPU::N_SIMD_WIDTH + 1;
		const auto scene = MakeMixedScene(N_F);
		auto neighbors = MakeMixedNeighbors(scene);
		neighbors[0][0] = { N_F - 1 };
		neighbors[0][N_ownedF - 1] = { N_F - 1 };
		const auto params = MakeParams();
		auto expected = ReferenceVelocities(scene, neighbors, params);
		for (uint32_t i = N_ownedF; i < N_F; i++) { expected[i] = glm::dvec3(untouchedVelocity); }
		const std::array<uint32_t, 2> selectedChunks{ 1, 0 };
		for (const bool useSelectedChunks : { false, true })
		{
			const auto actual = EvaluateNonPressure(scene, neighbors, params, selectedChunks, useSelectedChunks, N_ownedF);
			ExpectVelocitiesNear(actual, expected);
			for (uint32_t i = N_ownedF; i < N_F; i++)
			{
				for (int axis = 0; axis < 3; axis++) { EXPECT_EQ(actual[i][axis], untouchedVelocity); }
			}
		}
	}

	// wide index spans exercise velocity, density, and volume gathers in every neighbor category
	TEST_P(SphNonPressureTests, ComputesViscosityAcrossCompressedIndexFormats)
	{
		NonPressureScene scene;
		for (auto& population : scene)
		{
			population.assign(70'000, { glm::vec3(0.0f), glm::vec3(0.0f), 2.0f, 0.25f });
		}
		auto neighbors = MakeNeighbors(static_cast<uint32_t>(scene[0].size()));
		const std::array<uint32_t, 3> spans{ 255, 256, 65'536 };
		const std::array<Sim::SELLChunkFormat, 3> formats{
			Sim::SELLChunkFormat::Delta8, Sim::SELLChunkFormat::Delta16, Sim::SELLChunkFormat::Abs32 };
		for (uint32_t chunk = 0; chunk < spans.size(); chunk++)
		{
			const uint32_t source = chunk * CPU::N_SIMD_WIDTH;
			scene[0][source].velocity = { 1.0f, -2.0f, 0.5f };
			const uint32_t base = 17 + 3 * chunk;
			for (uint32_t category = 0; category < scene.size(); category++)
			{
				auto& low = scene[category][base];
				auto& high = scene[category][base + spans[chunk]];
				low.position = { 0.25f * (category + 1), 0.125f * (chunk + 1), 0.0f };
				high.position = { 0.0f, -0.25f * (category + 1), 0.125f * (chunk + 1) };
				low.velocity = { -0.5f * (category + 1), 0.25f * (chunk + 1), -1.0f };
				high.velocity = { 0.25f * (chunk + 1), 0.5f * (category + 1), -0.25f };
				low.density = static_cast<float>(2 + chunk);
				high.density = static_cast<float>(5 + 2 * chunk);
				low.volume = 0.125f * (1 + category + chunk);
				high.volume = 0.25f * (1 + category + chunk);
				neighbors[category][source] = { base + spans[chunk], base };
			}
		}
		const auto params = MakeParams();
		ExpectVelocitiesNear(EvaluateNonPressure(scene, neighbors, params), ReferenceVelocities(scene, neighbors, params));
		for (uint32_t chunk = 0; chunk < spans.size(); chunk++)
		{
			SCOPED_TRACE(chunk);
			const auto expected = GetParam() ? formats[chunk] : Sim::SELLChunkFormat::Explicit32;
			EXPECT_EQ(lists_->FFChunk(chunk).format, expected);
			EXPECT_EQ(lists_->FBChunk(chunk).format, expected);
			EXPECT_EQ(lists_->FRChunk(chunk).format, expected);
		}
	}

	INSTANTIATE_TEST_SUITE_P(ExplicitAndCompressed, SphNonPressureTests, testing::Bool());
}
