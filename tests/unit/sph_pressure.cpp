// sph_pressure.cpp

// external
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <tbb/partitioner.h>
#include <vector>

// internal
#include "source/eval/stats/statistics.h"
#include "source/simulation/CPU/backend/cpu_backend.h"
#include "source/simulation/CPU/backend/sph_solver.h"



namespace
{
	struct PressureParticle
	{
		glm::vec3 position;
		glm::vec3 predictedVelocity{ 0.0f };
		float density = 1.0f;
		float psi = 0.0f;
	};

	// populations and neighbor categories follow fluid, boundary, rigid order
	using PressureScene = std::array<std::vector<PressureParticle>, 3>;
	using PressureNeighbors = std::array<std::vector<std::vector<uint32_t>>, 3>;
	constexpr float untouchedOutput = -12345.0f;
	constexpr std::array writeChunks{
		&Sim::I_NeighborListSELL::FFChunkWrite, &Sim::I_NeighborListSELL::FBChunkWrite,
		&Sim::I_NeighborListSELL::FRChunkWrite };
	constexpr std::array finishChunks{
		&Sim::I_NeighborListSELL::FinishFFChunkWrite, &Sim::I_NeighborListSELL::FinishFBChunkWrite,
		&Sim::I_NeighborListSELL::FinishFRChunkWrite };

	Sim::SimParams MakeParams()
	{
		Sim::SimParams params;
		params.timeStep = 0.5f;
		params.particleSizeF = 1.0f;
		params.particleSizeInvF = 1.0f;
		params.kernelRadius = 2.0f;
		params.kernelRadiusSqu = 4.0f;
		params.restDensity = 1.0f;
		params.massF = 1.0f;
		params.pressGamma = 0.375f;
		params.jacobiOmega = 0.25f;
		params.gravity = glm::vec3(0.0f);
		return params;
	}

	PressureNeighbors MakeNeighbors(const uint32_t N_F)
	{
		PressureNeighbors neighbors;
		for (auto& category : neighbors) { category.resize(N_F); }
		return neighbors;
	}

	PressureScene MakeFluidPair(const float density = 9.0f / 8.0f)
	{
		PressureScene scene;
		scene[0] = {
			{ { -0.25f, 0.0f, 0.0f }, glm::vec3(0.0f), density },
			{ { 0.25f, 0.0f, 0.0f }, glm::vec3(0.0f), density } };
		return scene;
	}

	PressureNeighbors MakePairNeighbors()
	{
		auto neighbors = MakeNeighbors(2);
		neighbors[0] = { { 1 }, { 0 } };
		return neighbors;
	}

	PressureScene MakeMixedScene(const uint32_t N_F, const float restDensity)
	{
		PressureScene scene;
		for (uint32_t i = 0; i < N_F; i++)
		{
			scene[0].push_back({ { 0.0625f * i - 0.5f, 0.125f * (i % 3), -0.125f * (i % 2) },
				{ 0.125f * (i % 3), -0.25f * (i % 2), 0.0625f * i },
				restDensity * (i % 2 == 0 ? 0.75f : 1.25f) });
		}
		scene[1] = { { { -0.5f, 0.125f, 0.25f }, glm::vec3(0.0f), 1.0f, 0.75f } };
		scene[2] = { { { 0.25f, -0.625f, 0.5f }, { -0.25f, 0.375f, 0.5f }, 1.0f, 1.25f } };
		return scene;
	}

	PressureNeighbors MakeMixedNeighbors(const uint32_t N_F)
	{
		auto neighbors = MakeNeighbors(N_F);
		for (uint32_t i = 0; i < N_F; i++)
		{
			neighbors[0][i] = { (i + N_F - 1) % N_F, (i + 1) % N_F };
			if (i % 3 != 1) { neighbors[1][i] = { 0 }; }
			if (i % 3 != 2) { neighbors[2][i] = { 0 }; }
		}
		return neighbors;
	}

	// fluid neighbors of each fluid particle by distance, itself included like the neighbor search does
	std::vector<std::vector<uint32_t>> MakeSupportNeighborsF(const PressureScene& scene, const float kernelRadiusSqu)
	{
		std::vector<std::vector<uint32_t>> neighbors(scene[0].size());
		for (uint32_t i = 0; i < scene[0].size(); i++)
		{
			for (uint32_t j = 0; j < scene[0].size(); j++)
			{
				const glm::vec3 displacement = scene[0][i].position - scene[0][j].position;
				if (glm::dot(displacement, displacement) <= kernelRadiusSqu) { neighbors[i].push_back(j); }
			}
		}
		return neighbors;
	}

	// independent radial derivative in double precision with unit particle size and normalization
	glm::dvec3 ReferenceGradient(const glm::vec3& positionI, const glm::vec3& positionJ)
	{
		const glm::dvec3 displacement = glm::dvec3(positionI) - glm::dvec3(positionJ);
		const double distance = glm::length(displacement);
		if (distance == 0.0 || distance >= 2.0) { return glm::dvec3(0.0); }
		const double derivative = distance < 1.0
			? -3.0 * distance + 2.25 * distance * distance : -0.75 * (2.0 - distance) * (2.0 - distance);
		return displacement * (derivative / distance);
	}

	std::vector<glm::dvec3> ReferencePressureAccelerations(const PressureScene& scene,
		const PressureNeighbors& neighbors, const Sim::SimParams& params, const std::span<const double> pressure)
	{
		const double restDensity = params.restDensity;
		std::vector<glm::dvec3> acceleration(scene[0].size(), glm::dvec3(0.0));
		for (uint32_t i = 0; i < acceleration.size(); i++)
		{
			for (uint32_t category = 0; category < neighbors.size(); category++)
			{
				for (const uint32_t j : neighbors[category][i])
				{
					const auto& target = scene[category][j];
					const double weight = category == 0
						? params.massF * (pressure[i] + pressure[j])
						: 2.0 * params.pressGamma * target.psi * pressure[i];
					acceleration[i] -= ReferenceGradient(scene[0][i].position, target.position)
						* (weight / (restDensity * restDensity));
				}
			}
		}
		return acceleration;
	}

	double ReferenceDensityRate(const PressureScene& scene, const PressureNeighbors& neighbors,
		const Sim::SimParams& params, const uint32_t i, const std::span<const glm::dvec3> valuesF,
		const std::span<const glm::dvec3> valuesR = {})
	{
		double rate = 0.0;
		for (uint32_t category = 0; category < neighbors.size(); category++)
		{
			for (const uint32_t j : neighbors[category][i])
			{
				const auto& target = scene[category][j];
				const double weight = category == 0 ? params.massF : target.psi;
				glm::dvec3 relativeValue = valuesF[i];
				if (category == 0) { relativeValue -= valuesF[j]; }
				else if (category == 2 && !valuesR.empty()) { relativeValue -= valuesR[j]; }
				rate += weight * glm::dot(relativeValue, ReferenceGradient(scene[0][i].position, target.position));
			}
		}
		return rate;
	}

	void ExpectPressureNear(const double actual, const double expected)
	{
		EXPECT_NEAR(actual, expected, 1e-7 + 5e-6 * std::abs(expected));
	}

	struct InitializedPressure
	{
		std::vector<float> sourceTerm, diagElem, pressure;
	};

	class SphPressureTests : public testing::TestWithParam<bool>
	{
	protected:
		void SetUp() override
		{
			if (GetParam()) { lists_ = std::make_unique<Sim::NeighborLists_ChunkedSELL_AdaptCompress_CPU>(); }
			else { lists_ = std::make_unique<Sim::NeighborLists_ChunkedSELL_Explicit_CPU>(); }
		}

		void BuildLists(const PressureScene& scene, const PressureNeighbors& neighbors,
			const bool trackDistinctFF)
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
				std::vector<uint32_t> distinctFF;
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
							if (trackDistinctFF && category == 0 &&
								std::find(distinctFF.begin(), distinctFF.end(), j) == distinctFF.end())
							{
								distinctFF.push_back(j);
							}
						}
					}
				}
				// (dict chunks are written here, with the lanes of each distinct neighbor)
				std::vector<uint32_t> laneMasksFF;
				for (const uint32_t j : distinctFF)
				{
					uint32_t laneMask = 0;
					for (uint32_t i = begin; i < end; i++)
					{
						const auto& row = neighbors[0][i];
						if (std::find(row.begin(), row.end(), j) != row.end()) { laneMask |= 1u << (i - begin); }
					}
					laneMasksFF.push_back(laneMask);
				}
				lists_->CommitFluidChunkCounts(chunk,
					{ counts[0].data(), minRef[0], maxRef[0],
						static_cast<uint32_t>(distinctFF.size()), distinctFF.data(), laneMasksFF.data() },
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
					auto view = (lists_.get()->*writeChunks[category])(chunk);
					if (view.dict) { continue; }
					std::array<uint32_t, CPU::N_SIMD_WIDTH> counts{};
					for (uint32_t i = view.begin; i < view.end; i++)
					{
						const auto& row = neighbors[category][i];
						counts[i - view.begin] = static_cast<uint32_t>(row.size());
						for (uint32_t k = 0; k < row.size(); k++)
						{
							const uint32_t j = row[k];
							view.StoreNeighbor(i - view.begin, k, j);
						}
					}
					(lists_.get()->*finishChunks[category])(chunk, counts.data());
				}
			}
			lists_->Finalize_Fill();
		}

		void LoadScene(const PressureScene& scene, const PressureNeighbors& neighbors,
			const bool trackDistinctFF = false)
		{
			BuildLists(scene, neighbors, trackDistinctFF);
			positions_ = {};
			predictedVelocities_ = {};
			densityF_.clear();
			psi_ = {};
			for (uint32_t category = 0; category < scene.size(); category++)
			{
				for (const auto& particle : scene[category])
				{
					for (int axis = 0; axis < 3; axis++)
					{
						positions_[category][axis].push_back(particle.position[axis]);
						predictedVelocities_[category][axis].push_back(particle.predictedVelocity[axis]);
					}
					if (category == 0) { densityF_.push_back(particle.density); }
					else { psi_[category].push_back(particle.psi); }
				}
			}
		}

		InitializedPressure EvaluateInitialPressure(
			const std::optional<std::span<const uint32_t>> selectedChunks = std::nullopt)
		{
			const uint32_t N_F = static_cast<uint32_t>(densityF_.size());
			const float restDensitySquInv = 1.0f / (params_.restDensity * params_.restDensity);
			InitializedPressure result;
			result.sourceTerm.assign(N_F, untouchedOutput);
			result.diagElem.assign(N_F, untouchedOutput);
			result.pressure.assign(N_F, untouchedOutput);
			CPU::SourceTermAndDiagonalElementAndPressureJacobiInit_ChunkedSELL(
				positions_[0][0].data(), positions_[0][1].data(), positions_[0][2].data(),
				positions_[1][0].data(), positions_[1][1].data(), positions_[1][2].data(),
				positions_[2][0].data(), positions_[2][1].data(), positions_[2][2].data(),
				predictedVelocities_[0][0].data(), predictedVelocities_[0][1].data(), predictedVelocities_[0][2].data(),
				predictedVelocities_[2][0].data(), predictedVelocities_[2][1].data(), predictedVelocities_[2][2].data(),
				densityF_.data(), psi_[1].data(), psi_[2].data(), *lists_,
				params_.timeStep, params_.particleSizeInvF, cubicSplineAlpha, params_.restDensity, restDensitySquInv,
				params_.massF, params_.pressGamma, params_.jacobiOmega, N_F,
				result.sourceTerm.data(), result.diagElem.data(), result.pressure.data(),
				selectedChunks.value_or(std::span<const uint32_t>{}), selectedChunks.has_value());
			return result;
		}

		std::vector<glm::vec3> EvaluatePressureAcceleration(const std::span<const float> pressure,
			const std::optional<std::span<const uint32_t>> selectedChunks = std::nullopt,
			const std::span<const uint8_t> activeParticles = {})
		{
			const uint32_t N_F = static_cast<uint32_t>(densityF_.size());
			tbb::affinity_partitioner affinity;
			std::array<std::vector<float>, 3> acceleration;
			for (auto& axis : acceleration) { axis.assign(N_F, untouchedOutput); }
			CPU::PressureAcceleration_ChunkedSELL(
				positions_[0][0].data(), positions_[0][1].data(), positions_[0][2].data(),
				positions_[1][0].data(), positions_[1][1].data(), positions_[1][2].data(),
				positions_[2][0].data(), positions_[2][1].data(), positions_[2][2].data(),
				pressure.data(), psi_[1].data(), psi_[2].data(), *lists_,
				params_.particleSizeInvF, cubicSplineAlpha, params_.restDensity, params_.massF, params_.pressGamma,
				N_F, affinity, acceleration[0].data(), acceleration[1].data(), acceleration[2].data(),
				selectedChunks.value_or(std::span<const uint32_t>{}), selectedChunks.has_value(), activeParticles.data());

			std::vector<glm::vec3> result(N_F);
			for (uint32_t i = 0; i < N_F; i++)
			{
				result[i] = { acceleration[0][i], acceleration[1][i], acceleration[2][i] };
			}
			return result;
		}

		Sim::PressureSolveErrorSums EvaluateJacobiUpdate(const std::span<const glm::vec3> acceleration,
			const std::span<const float> sourceTerm, const std::span<const float> diagElem,
			const std::span<float> pressure,
			const std::optional<std::span<const uint32_t>> selectedChunks = std::nullopt)
		{
			std::array<std::vector<float>, 3> pressureAcceleration;
			for (const auto& particleAcceleration : acceleration)
			{
				for (int axis = 0; axis < 3; axis++) { pressureAcceleration[axis].push_back(particleAcceleration[axis]); }
			}
			Sim::PressureSolveErrorSums errors{};
			tbb::affinity_partitioner affinity;
			CPU::PressureRelaxedJacobiSolver_ChunkedSELL(
				positions_[0][0].data(), positions_[0][1].data(), positions_[0][2].data(),
				positions_[1][0].data(), positions_[1][1].data(), positions_[1][2].data(),
				positions_[2][0].data(), positions_[2][1].data(), positions_[2][2].data(),
				pressureAcceleration[0].data(), pressureAcceleration[1].data(), pressureAcceleration[2].data(),
				sourceTerm.data(), diagElem.data(), psi_[1].data(), psi_[2].data(), *lists_,
				params_.timeStep, params_.particleSizeInvF, cubicSplineAlpha, params_.restDensity,
				params_.massF, params_.jacobiOmega, static_cast<uint32_t>(densityF_.size()),
				affinity, pressure.data(), nullptr, errors,
				selectedChunks.value_or(std::span<const uint32_t>{}), selectedChunks.has_value());
			return errors;
		}

		// one quarter gives unit normalization for the kernel's scaled polynomial
		static constexpr float cubicSplineAlpha = 0.25f;
		Sim::SimParams params_ = MakeParams();
		std::unique_ptr<Sim::I_NeighborListSELL> lists_;
		std::array<std::array<std::vector<float>, 3>, 3> positions_, predictedVelocities_;
		std::array<std::vector<float>, 3> psi_;
		std::vector<float> densityF_;
	};

	// a stationary over-dense pair has a negative source term and positive initial pressure
	TEST_P(SphPressureTests, ComputesKnownSourceDiagonalAndInitialPressure)
	{
		LoadScene(MakeFluidPair(), MakePairNeighbors());
		const auto actual = EvaluateInitialPressure();
		for (uint32_t i = 0; i < 2; i++)
		{
			SCOPED_TRACE(i);
			ExpectPressureNear(actual.sourceTerm[i], -1.0 / 8.0);
			ExpectPressureNear(actual.diagElem[i], -225.0 / 512.0);
			ExpectPressureNear(actual.pressure[i], 16.0 / 225.0);
		}
	}

	// equal masses give opposite pressure accelerations along the line between the particles
	TEST_P(SphPressureTests, ComputesEqualAndOppositePairAcceleration)
	{
		LoadScene(MakeFluidPair(), MakePairNeighbors());
		const std::array pressure{ 16.0f / 225.0f, 16.0f / 225.0f };
		const auto actual = EvaluatePressureAcceleration(pressure);
		ExpectPressureNear(actual[0].x, -2.0 / 15.0);
		ExpectPressureNear(actual[1].x, 2.0 / 15.0);
		for (uint32_t i = 0; i < actual.size(); i++)
		{
			SCOPED_TRACE(i);
			EXPECT_EQ(actual[i].y, 0.0f);
			EXPECT_EQ(actual[i].z, 0.0f);
		}
	}

	// known accelerations give a hand-derived pressure correction and density error
	TEST_P(SphPressureTests, ComputesKnownRelaxedJacobiUpdate)
	{
		LoadScene(MakeFluidPair(), MakePairNeighbors());
		const std::array sourceTerm{ -1.0f / 8.0f, -1.0f / 8.0f };
		const std::array diagElem{ -225.0f / 512.0f, -225.0f / 512.0f };
		const std::array acceleration{
			glm::vec3{ -2.0f / 15.0f, 0.0f, 0.0f }, glm::vec3{ 2.0f / 15.0f, 0.0f, 0.0f } };
		std::array pressure{ 16.0f / 225.0f, 16.0f / 225.0f };
		const auto errors = EvaluateJacobiUpdate(acceleration, sourceTerm, diagElem, pressure);
		for (const float value : pressure) { ExpectPressureNear(value, 24.0 / 225.0); }
		ExpectPressureNear(errors.raw, 1.0 / 8.0);
		ExpectPressureNear(errors.pos, 1.0 / 8.0);
		ExpectPressureNear(errors.abs, 1.0 / 8.0);
	}

	// an isolated particle has a zero diagonal and contributes no error even when over-dense
	TEST_P(SphPressureTests, SkipsZeroDiagonalWhenUpdatingPressureAndErrors)
	{
		PressureScene scene;
		scene[0] = { { glm::vec3(0.0f), glm::vec3(0.0f), 9.0f / 8.0f } };
		LoadScene(scene, MakeNeighbors(1));
		auto actual = EvaluateInitialPressure();
		EXPECT_EQ(actual.sourceTerm[0], -1.0f / 8.0f);
		EXPECT_EQ(actual.diagElem[0], 0.0f);
		EXPECT_EQ(actual.pressure[0], 0.0f);

		actual.pressure[0] = 1.0f;
		const std::array acceleration{ glm::vec3(0.0f) };
		const auto errors = EvaluateJacobiUpdate(acceleration, actual.sourceTerm, actual.diagElem, actual.pressure);
		EXPECT_EQ(actual.pressure[0], 0.0f);
		EXPECT_EQ(errors.raw, 0.0);
		EXPECT_EQ(errors.pos, 0.0);
		EXPECT_EQ(errors.abs, 0.0);
	}

	// an under-dense pair with a valid diagonal clamps negative pressures to zero
	TEST_P(SphPressureTests, ClampsNegativePressure)
	{
		LoadScene(MakeFluidPair(7.0f / 8.0f), MakePairNeighbors());
		const auto actual = EvaluateInitialPressure();
		for (const float value : actual.pressure) { EXPECT_EQ(value, 0.0f); }

		const std::array sourceTerm{ 1.0f / 8.0f, 1.0f / 8.0f };
		const std::array diagElem{ -225.0f / 512.0f, -225.0f / 512.0f };
		const std::array acceleration{ glm::vec3(0.0f), glm::vec3(0.0f) };
		std::array pressure{ 1.0f / 32.0f, 1.0f / 32.0f };
		EvaluateJacobiUpdate(acceleration, sourceTerm, diagElem, pressure);
		for (const float value : pressure) { EXPECT_EQ(value, 0.0f); }
	}

	// distinct boundary and rigid weights and a moving rigid neighbor exercise all three sums
	TEST_P(SphPressureTests, MatchesScalarReferenceAcrossSimdBoundaries)
	{
		const uint32_t N_F = CPU::N_SIMD_WIDTH + 1;
		params_.restDensity = 2.0f;
		const auto scene = MakeMixedScene(N_F, params_.restDensity);
		auto neighbors = MakeMixedNeighbors(N_F);
		if (GetParam()) { neighbors[0] = MakeSupportNeighborsF(scene, params_.kernelRadiusSqu); }
		LoadScene(scene, neighbors, GetParam());
		for (uint32_t chunk = 0; chunk < lists_->ChunkNumTotal(); chunk++)
		{
			SCOPED_TRACE(chunk);
			ASSERT_EQ(lists_->FFChunk(chunk).dict != nullptr, GetParam());
		}
		const auto initial = EvaluateInitialPressure();
		std::vector<float> pressure(N_F);
		std::vector<glm::dvec3> predictedVelocitiesF(N_F);
		for (uint32_t i = 0; i < N_F; i++)
		{
			pressure[i] = 0.125f + 0.03125f * i;
			predictedVelocitiesF[i] = glm::dvec3(scene[0][i].predictedVelocity);
		}
		const std::array predictedVelocitiesR{ glm::dvec3(scene[2][0].predictedVelocity) };
		const std::vector<double> referencePressure(pressure.begin(), pressure.end());
		const auto referenceAcceleration = ReferencePressureAccelerations(scene, neighbors, params_, referencePressure);
		const auto acceleration = EvaluatePressureAcceleration(pressure);
		const auto errors = EvaluateJacobiUpdate(acceleration, initial.sourceTerm, initial.diagElem, pressure);

		Sim::PressureSolveErrorSums expectedErrors{};
		std::vector<double> unitPressure(N_F, 0.0);
		const double timeStep = params_.timeStep;
		for (uint32_t i = 0; i < N_F; i++)
		{
			SCOPED_TRACE(i);
			const double sourceTerm = params_.restDensity - scene[0][i].density - timeStep
				* ReferenceDensityRate(scene, neighbors, params_, i, predictedVelocitiesF, predictedVelocitiesR);

			// unit pressure at i gives the diagonal through the resulting density change
			unitPressure[i] = 1.0;
			const auto unitAcceleration = ReferencePressureAccelerations(scene, neighbors, params_, unitPressure);
			const double diagElem = timeStep * timeStep
				* ReferenceDensityRate(scene, neighbors, params_, i, unitAcceleration);
			unitPressure[i] = 0.0;
			ASSERT_LT(diagElem, -1e-6);

			const double systemValue = timeStep * timeStep
				* ReferenceDensityRate(scene, neighbors, params_, i, referenceAcceleration);
			const double residual = sourceTerm - systemValue;
			ExpectPressureNear(initial.sourceTerm[i], sourceTerm);
			ExpectPressureNear(initial.diagElem[i], diagElem);
			ExpectPressureNear(initial.pressure[i], std::max(0.0, params_.jacobiOmega * sourceTerm / diagElem));
			for (int axis = 0; axis < 3; axis++)
			{
				SCOPED_TRACE(axis);
				ExpectPressureNear(acceleration[i][axis], referenceAcceleration[i][axis]);
			}
			ExpectPressureNear(pressure[i],
				std::max(0.0, referencePressure[i] + params_.jacobiOmega * residual / diagElem));

			const double error = -residual / params_.restDensity;
			expectedErrors.raw += error;
			expectedErrors.pos += std::max(0.0, error);
			expectedErrors.abs += std::abs(error);
		}
		ExpectPressureNear(errors.raw, expectedErrors.raw);
		ExpectPressureNear(errors.pos, expectedErrors.pos);
		ExpectPressureNear(errors.abs, expectedErrors.abs);
	}

	// select the partial second chunk and mask out one of its two lanes for acceleration
	TEST_P(SphPressureTests, PreservesOutputsOutsideSelectedChunksAndActiveLanes)
	{
		const uint32_t begin = CPU::N_SIMD_WIDTH;
		const uint32_t N_F = begin + 2;
		auto scene = MakeFluidPair();
		scene[0].resize(N_F, { glm::vec3(0.0f) });
		scene[0][begin] = scene[0][0];
		scene[0][begin + 1] = scene[0][1];
		auto neighbors = MakeNeighbors(N_F);
		neighbors[0][0] = { 1 };
		neighbors[0][1] = { 0 };
		neighbors[0][begin] = { begin + 1 };
		neighbors[0][begin + 1] = { begin };
		LoadScene(scene, neighbors);

		const std::array selectedChunks{ 1u };
		const auto initial = EvaluateInitialPressure(selectedChunks);
		std::vector<uint8_t> activeParticles(N_F, 1);
		activeParticles[begin + 1] = 0;
		const std::vector<float> pairPressure(N_F, 16.0f / 225.0f);
		const auto maskedAcceleration = EvaluatePressureAcceleration(pairPressure, selectedChunks, activeParticles);
		for (uint32_t i = 0; i < N_F; i++)
		{
			SCOPED_TRACE(i);
			if (i < begin)
			{
				EXPECT_EQ(initial.sourceTerm[i], untouchedOutput);
				EXPECT_EQ(initial.diagElem[i], untouchedOutput);
				EXPECT_EQ(initial.pressure[i], untouchedOutput);
			}
			else
			{
				ExpectPressureNear(initial.sourceTerm[i], -1.0 / 8.0);
				ExpectPressureNear(initial.diagElem[i], -225.0 / 512.0);
				ExpectPressureNear(initial.pressure[i], 16.0 / 225.0);
			}
			if (i == begin)
			{
				ExpectPressureNear(maskedAcceleration[i].x, -2.0 / 15.0);
				EXPECT_EQ(maskedAcceleration[i].y, 0.0f);
				EXPECT_EQ(maskedAcceleration[i].z, 0.0f);
			}
			else { EXPECT_EQ(maskedAcceleration[i], glm::vec3(untouchedOutput)); }
		}

		const auto fullInitial = EvaluateInitialPressure();
		const auto acceleration = EvaluatePressureAcceleration(pairPressure);
		auto pressure = initial.pressure;
		const auto errors = EvaluateJacobiUpdate(
			acceleration, fullInitial.sourceTerm, fullInitial.diagElem, pressure, selectedChunks);
		for (uint32_t i = 0; i < begin; i++) { EXPECT_EQ(pressure[i], untouchedOutput) << i; }
		ExpectPressureNear(pressure[begin], 24.0 / 225.0);
		ExpectPressureNear(pressure[begin + 1], 24.0 / 225.0);
		ExpectPressureNear(errors.raw, 1.0 / 8.0);
		ExpectPressureNear(errors.pos, 1.0 / 8.0);
		ExpectPressureNear(errors.abs, 1.0 / 8.0);
	}

	TEST_P(SphPressureTests, LeavesOutputsUntouchedWithEmptyChunkSelection)
	{
		LoadScene(MakeFluidPair(), MakePairNeighbors());
		const std::span<const uint32_t> selectedChunks{};
		const auto initial = EvaluateInitialPressure(selectedChunks);
		const std::array pairPressure{ 16.0f / 225.0f, 16.0f / 225.0f };
		const auto maskedAcceleration = EvaluatePressureAcceleration(pairPressure, selectedChunks);
		const auto fullInitial = EvaluateInitialPressure();
		const auto acceleration = EvaluatePressureAcceleration(pairPressure);
		auto pressure = pairPressure;
		const auto errors = EvaluateJacobiUpdate(
			acceleration, fullInitial.sourceTerm, fullInitial.diagElem, pressure, selectedChunks);
		for (uint32_t i = 0; i < pressure.size(); i++)
		{
			SCOPED_TRACE(i);
			EXPECT_EQ(initial.sourceTerm[i], untouchedOutput);
			EXPECT_EQ(initial.diagElem[i], untouchedOutput);
			EXPECT_EQ(initial.pressure[i], untouchedOutput);
			EXPECT_EQ(maskedAcceleration[i], glm::vec3(untouchedOutput));
			EXPECT_EQ(pressure[i], pairPressure[i]);
		}
		EXPECT_EQ(errors.raw, 0.0);
		EXPECT_EQ(errors.pos, 0.0);
		EXPECT_EQ(errors.abs, 0.0);
	}

	INSTANTIATE_TEST_SUITE_P(ExplicitAndCompressed, SphPressureTests, testing::Bool());

	struct PressureIterationLimits
	{
		uint32_t fixedIters, minIters, maxIters, expectedIters;
		Base::SolverTerminationCond termination = Base::SolverTerminationCond::FixedIters;
	};

	class SphPressureIterationTests : public testing::TestWithParam<PressureIterationLimits> {};

	// initialization counts as iteration one, and density-based termination needs a measured error
	TEST_P(SphPressureIterationTests, HonorsIterationCountAndLimits)
	{
		const auto limits = GetParam();
		auto params = MakeParams();
		params.neighborSearchMethod = Base::NeighborSearchMethod::BruteForce;
		params.neighborListMethod = Base::NeighborListMethod::ChunkedSELLX_Explicit;
		params.solverTerminationCond = limits.termination;
		params.densityDevTarget = 1.0f;
		params.fixedIters = limits.fixedIters;
		params.minSolverIters = limits.minIters;
		params.maxSolverIters = limits.maxIters;
		params.massF = 2.0f; // the pair starts above rest density with the backend's kernel normalization

		Sync::UniversalSimState state;
		state.ResizeFluid(2);
		state.particlesF.id = { 1, 2 };
		state.particlesF.xPos = { -0.25f, 0.25f };
		Bench::BenchmarksTracker benchmarks(params);
		Sim::SimBackend_CPU backend(benchmarks, params);
		backend.PullFromUniversalSimState(state, params);
		backend.ExecuteFullSimStep(params, 1);

		Stats::StatisticsPartial statistics;
		ASSERT_TRUE(backend.FillStatisticsPartial(statistics, params));
		EXPECT_EQ(statistics.N_solverIters, limits.expectedIters);
		const auto fluid = backend.GetFluidParticleView();
		ASSERT_EQ(fluid.xPos.size(), 2u);
		EXPECT_GT(std::abs(fluid.xPos[1] - fluid.xPos[0]), 0.5f);
	}

	INSTANTIATE_TEST_SUITE_P(IterationLimits, SphPressureIterationTests,
		testing::Values(
			PressureIterationLimits{ 5, 2, 8, 5 },
			PressureIterationLimits{ 2, 4, 8, 4 },
			PressureIterationLimits{ 8, 2, 3, 3 },
			PressureIterationLimits{ 8, 1, 8, 2, Base::SolverTerminationCond::RelDensityError_Abs }));
}
