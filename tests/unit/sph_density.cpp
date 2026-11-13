// sph_density.cpp

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
	struct DensityParticles
	{
		std::vector<float> xPos, yPos, zPos;
		std::vector<float> psi;
	};

	// populations and neighbor categories follow fluid, boundary, rigid order
	using DensityScene = std::array<DensityParticles, 3>;
	using DensityNeighbors = std::array<std::vector<std::vector<uint32_t>>, 3>;
	constexpr float untouchedDensity = -12345.0f;
	constexpr std::array writeChunks{
		&Sim::I_NeighborListSELL::FFChunkWrite, &Sim::I_NeighborListSELL::FBChunkWrite,
		&Sim::I_NeighborListSELL::FRChunkWrite };
	constexpr std::array finishChunks{
		&Sim::I_NeighborListSELL::FinishFFChunkWrite, &Sim::I_NeighborListSELL::FinishFBChunkWrite,
		&Sim::I_NeighborListSELL::FinishFRChunkWrite };

	DensityParticles MakeParticles(const std::vector<glm::vec3>& positions, const std::vector<float>& psi = {})
	{
		DensityParticles particles;
		for (const auto& position : positions)
		{
			particles.xPos.push_back(position.x);
			particles.yPos.push_back(position.y);
			particles.zPos.push_back(position.z);
		}
		particles.psi = psi;
		return particles;
	}

	DensityNeighbors MakeNeighbors(const uint32_t N_F)
	{
		DensityNeighbors neighbors;
		for (auto& category : neighbors) { category.resize(N_F); }
		return neighbors;
	}

	DensityScene MakeMixedScene(const uint32_t N_F)
	{
		std::vector<glm::vec3> fluid;
		for (uint32_t i = 0; i < N_F; i++)
		{
			fluid.push_back({ 0.25f * (i % 7) - 0.5f, 0.125f * (i % 3), 0.2f * (i % 5) });
		}
		return { MakeParticles(fluid),
			MakeParticles({ { 0.0f, 0.0f, 0.0f }, { 0.5f, -0.25f, 0.75f },
				{ -0.25f, 0.5f, 0.0f }, { 1.0f, 0.0f, -0.5f }, { 0.0f, -0.5f, 0.25f } },
				{ 2.0f, 0.0f, 5.0f, 7.0f, 11.0f }),
			MakeParticles({ { 0.0f, 0.0f, 0.0f }, { -0.5f, 0.25f, 0.0f },
				{ 0.5f, 1.0f, 0.25f }, { 0.0f, -0.25f, 1.0f } }, { 13.0f, 17.0f, 0.0f, 23.0f }) };
	}

	DensityNeighbors MakeMixedNeighbors(const DensityScene& scene)
	{
		const uint32_t N_F = static_cast<uint32_t>(scene[0].xPos.size());
		auto neighbors = MakeNeighbors(N_F);
		for (uint32_t category = 0; category < neighbors.size(); category++)
		{
			const uint32_t N_target = static_cast<uint32_t>(scene[category].xPos.size());
			for (uint32_t i = 0; i < N_F; i++)
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

	// independent piecewise polynomial in double precision, without production kernels or list decoding
	std::vector<double> ReferenceDensity(const DensityScene& scene, const DensityNeighbors& neighbors,
		const double particleSizeF = 1.0, const double massF = 2.0)
	{
		const double normalization = 1.0 / (std::numbers::pi_v<double>
			* particleSizeF * particleSizeF * particleSizeF);
		std::vector<double> density(scene[0].xPos.size(), 0.0);
		for (uint32_t i = 0; i < density.size(); i++)
		{
			for (uint32_t category = 0; category < neighbors.size(); category++)
			{
				const auto& target = scene[category];
				for (const uint32_t j : neighbors[category][i])
				{
					const double dx = static_cast<double>(scene[0].xPos[i]) - target.xPos[j];
					const double dy = static_cast<double>(scene[0].yPos[i]) - target.yPos[j];
					const double dz = static_cast<double>(scene[0].zPos[i]) - target.zPos[j];
					const double q = std::sqrt(dx * dx + dy * dy + dz * dz) / particleSizeF;
					if (q >= 2.0) { continue; }
					const double shape = q < 1.0
						? 1.0 - 1.5 * q * q + 0.75 * q * q * q
						: 0.25 * std::pow(2.0 - q, 3);
					const double weight = category == 0 ? massF : target.psi[j];
					density[i] += weight * normalization * shape;
				}
			}
		}
		return density;
	}

	void ExpectDensityNear(const std::vector<float>& actual, const std::vector<double>& expected)
	{
		ASSERT_EQ(actual.size(), expected.size());
		for (uint32_t i = 0; i < expected.size(); i++)
		{
			SCOPED_TRACE(i);
			EXPECT_NEAR(actual[i], expected[i], 1e-7 + 3e-6 * std::abs(expected[i]));
		}
	}

	class SphDensityTests : public testing::TestWithParam<bool>
	{
	protected:
		void SetUp() override
		{
			if (GetParam()) { lists_ = std::make_unique<Sim::NeighborLists_ChunkedSELL_AdaptCompress_CPU>(); }
			else { lists_ = std::make_unique<Sim::NeighborLists_ChunkedSELL_Explicit_CPU>(); }
		}

		void BuildLists(const DensityScene& scene, const DensityNeighbors& neighbors)
		{
			const uint32_t N_F = static_cast<uint32_t>(scene[0].xPos.size());
			const uint32_t N_B = static_cast<uint32_t>(scene[1].xPos.size());
			const uint32_t N_R = static_cast<uint32_t>(scene[2].xPos.size());
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

		std::vector<float> EvaluateDensity(const DensityScene& scene, const DensityNeighbors& neighbors,
			const float particleSizeF = 1.0f, const float massF = 2.0f,
			const std::span<const uint32_t> selectedChunks = {}, const bool useSelectedChunks = false,
			const std::optional<uint32_t> particleLimit = {})
		{
			BuildLists(scene, neighbors);
			const uint32_t N_F = static_cast<uint32_t>(scene[0].xPos.size());
			const auto constants = Sim::BuildSolverConstants(particleSizeF, 1000.0f);
			// a full SIMD-width guard also catches writes beyond the last active lane
			std::vector<float> density(N_F + CPU::N_SIMD_WIDTH + 1, untouchedDensity);
			CPU::Density_ChunkedSELL(
				scene[0].xPos.data(), scene[0].yPos.data(), scene[0].zPos.data(),
				scene[1].xPos.data(), scene[1].yPos.data(), scene[1].zPos.data(),
				scene[2].xPos.data(), scene[2].yPos.data(), scene[2].zPos.data(),
				scene[1].psi.data(), scene[2].psi.data(), *lists_,
				constants.sizeInvF, 4.0f * particleSizeF * particleSizeF,
				constants.cubicSplineAlpha, massF, particleLimit.value_or(N_F),
				density.data() + 1, selectedChunks, useSelectedChunks);

			EXPECT_EQ(density.front(), untouchedDensity);
			for (uint32_t i = N_F + 1; i < density.size(); i++) { EXPECT_EQ(density[i], untouchedDensity); }
			return std::vector<float>(density.begin() + 1, density.begin() + 1 + N_F);
		}

		std::unique_ptr<Sim::I_NeighborListSELL> lists_;
	};

	// fluid mass multiplies the listed kernel weights, including an explicitly listed self-neighbor
	TEST_P(SphDensityTests, ComputesKnownFluidContributions)
	{
		DensityScene scene;
		scene[0] = MakeParticles({ { 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.5f, 0.0f } });
		auto neighbors = MakeNeighbors(3);
		neighbors[0] = { { 2, 0, 1 }, { 1 }, {} };
		const auto actual = EvaluateDensity(scene, neighbors);

		ExpectDensityNear(actual, { 10.25 / (4.0 * std::numbers::pi_v<double>),
			2.0 / std::numbers::pi_v<double>, 0.0 });
	}

	// boundary and rigid contributions use each neighbor's psi independently of the fluid mass
	TEST_P(SphDensityTests, ComputesKnownBoundaryAndRigidContributions)
	{
		for (uint32_t category = 1; category < 3; category++)
		{
			SCOPED_TRACE(category);
			DensityScene scene;
			scene[0] = MakeParticles({ { 0.0f, 0.0f, 0.0f } });
			scene[category] = MakeParticles({ { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.5f, 0.0f },
				{ 0.0f, 0.0f, 1.0f }, { -1.5f, 0.0f, 0.0f } }, { 2.0f, 3.0f, 5.0f, 7.0f });
			auto neighbors = MakeNeighbors(1);
			neighbors[category][0] = { 3, 1, 0, 2 };
			const auto actual = EvaluateDensity(scene, neighbors, 1.0f, 11.0f);

			ExpectDensityNear(actual, { 22.5 / (4.0 * std::numbers::pi_v<double>) });
		}
	}

	// uneven rows and nonzero padding targets must agree with an independent mixed-population sum
	TEST_P(SphDensityTests, MatchesScalarReferenceAcrossSimdBoundaries)
	{
		for (const uint32_t N_F : { 1u, CPU::N_SIMD_WIDTH - 1, CPU::N_SIMD_WIDTH,
			CPU::N_SIMD_WIDTH + 1, 2 * CPU::N_SIMD_WIDTH + 3 })
		{
			SCOPED_TRACE(N_F);
			const auto scene = MakeMixedScene(N_F);
			const auto neighbors = MakeMixedNeighbors(scene);

			ExpectDensityNear(EvaluateDensity(scene, neighbors), ReferenceDensity(scene, neighbors));
		}
	}

	// supplied neighbors at or beyond support contribute zero in every population
	TEST_P(SphDensityTests, ExcludesContributionsOutsideKernelSupport)
	{
		DensityScene scene;
		scene[0] = MakeParticles({ { 0.0f, 0.0f, 0.0f }, { 2.0f, 0.0f, 0.0f }, { 2.5f, 0.0f, 0.0f } });
		scene[1] = MakeParticles({ { 0.0f, 2.0f, 0.0f }, { 0.0f, 2.5f, 0.0f } }, { 5.0f, 7.0f });
		scene[2] = MakeParticles({ { 0.0f, 0.0f, 2.0f }, { 0.0f, 0.0f, 2.5f } }, { 11.0f, 13.0f });
		auto neighbors = MakeNeighbors(3);
		for (auto& category : neighbors) { category[0] = { 0, 1 }; }
		neighbors[0][0] = { 1, 2 };
		const auto actual = EvaluateDensity(scene, neighbors);

		EXPECT_EQ(actual, (std::vector<float>{ 0.0f, 0.0f, 0.0f }));
	}

	// changing particle size and mass must retain the correct normalization and population weighting
	TEST_P(SphDensityTests, MatchesScalarReferenceAtDifferentScales)
	{
		for (const float particleSizeF : { 0.25f, 1.0f, 2.0f })
		{
			SCOPED_TRACE(particleSizeF);
			auto scene = MakeMixedScene(CPU::N_SIMD_WIDTH + 1);
			for (auto& population : scene)
			{
				for (auto* positions : { &population.xPos, &population.yPos, &population.zPos })
				{
					for (auto& value : *positions) { value *= particleSizeF; }
				}
			}
			const auto neighbors = MakeMixedNeighbors(scene);
			for (const float massF : { 0.5f, 3.0f })
			{
				SCOPED_TRACE(massF);
				ExpectDensityNear(EvaluateDensity(scene, neighbors, particleSizeF, massF),
					ReferenceDensity(scene, neighbors, particleSizeF, massF));
			}
		}
	}

	// each spline weight fits in FP32, but adding them before applying mass would overflow
	TEST_P(SphDensityTests, KeepsFluidDensityFiniteWhenUnweightedSumWouldOverflow)
	{
		constexpr float particleSizeF = 1e-13f;
		constexpr float massF = 1000.0f * particleSizeF * particleSizeF * particleSizeF;
		DensityScene scene;
		scene[0] = MakeParticles({ { 0.0f, 0.0f, 0.0f }, { 0.25f * particleSizeF, 0.0f, 0.0f } });
		auto neighbors = MakeNeighbors(2);
		neighbors[0] = { { 0, 1 }, { 0, 1 } };

		ExpectDensityNear(EvaluateDensity(scene, neighbors, particleSizeF, massF),
			ReferenceDensity(scene, neighbors, particleSizeF, massF));
	}

	// particles without listed neighbors overwrite old density with zero
	TEST_P(SphDensityTests, ClearsDensityForEmptyNeighborLists)
	{
		const auto scene = MakeMixedScene(CPU::N_SIMD_WIDTH + 1);
		const auto actual = EvaluateDensity(scene, MakeNeighbors(CPU::N_SIMD_WIDTH + 1));

		EXPECT_EQ(actual, std::vector<float>(CPU::N_SIMD_WIDTH + 1, 0.0f));
	}

	// absent fluid particles leave output guards untouched with or without other populations
	TEST_P(SphDensityTests, HandlesEmptyFluidPopulations)
	{
		EXPECT_TRUE(EvaluateDensity({}, MakeNeighbors(0)).empty());
		EXPECT_TRUE(EvaluateDensity(MakeMixedScene(0), MakeNeighbors(0)).empty());
	}

	// noncontiguous chunks can be visited out of order, including the final partial chunk
	TEST_P(SphDensityTests, UpdatesOnlySelectedChunks)
	{
		const auto scene = MakeMixedScene(3 * CPU::N_SIMD_WIDTH + 1);
		const auto neighbors = MakeMixedNeighbors(scene);
		const std::array<uint32_t, 2> selectedChunks{ 3, 1 };
		const auto actual = EvaluateDensity(scene, neighbors, 1.0f, 2.0f, selectedChunks, true);
		auto expected = ReferenceDensity(scene, neighbors);
		for (uint32_t i = 0; i < expected.size(); i++)
		{
			const uint32_t chunk = i / CPU::N_SIMD_WIDTH;
			if (chunk != 1 && chunk != 3)
			{
				EXPECT_EQ(actual[i], untouchedDensity);
				expected[i] = untouchedDensity;
			}
		}

		ExpectDensityNear(actual, expected);
	}

	// an empty selection does no work only when selected-chunk processing is enabled
	TEST_P(SphDensityTests, HonorsSelectedChunkMode)
	{
		const auto scene = MakeMixedScene(CPU::N_SIMD_WIDTH + 1);
		const auto neighbors = MakeMixedNeighbors(scene);
		const auto actual = EvaluateDensity(scene, neighbors, 1.0f, 2.0f, {}, true);
		EXPECT_EQ(actual, std::vector<float>(CPU::N_SIMD_WIDTH + 1, untouchedDensity));
		const std::array<uint32_t, 1> selectedChunks{ 1 };

		ExpectDensityNear(EvaluateDensity(scene, neighbors, 1.0f, 2.0f, selectedChunks, false),
			ReferenceDensity(scene, neighbors));
	}

	// an owned prefix can end inside a stored chunk while still gathering neighbors from the halo
	TEST_P(SphDensityTests, ComputesOwnedPrefixUsingHaloNeighbors)
	{
		const uint32_t N_F = 2 * CPU::N_SIMD_WIDTH + 3;
		const uint32_t N_ownedF = CPU::N_SIMD_WIDTH + 1;
		const auto scene = MakeMixedScene(N_F);
		auto neighbors = MakeMixedNeighbors(scene);
		neighbors[0][0] = { N_F - 1 };
		neighbors[0][N_ownedF - 1] = { N_F - 1 };
		auto expected = ReferenceDensity(scene, neighbors);
		for (uint32_t i = N_ownedF; i < N_F; i++) { expected[i] = untouchedDensity; }
		const std::array<uint32_t, 2> selectedChunks{ 1, 0 };
		for (const bool useSelectedChunks : { false, true })
		{
			SCOPED_TRACE(useSelectedChunks);
			const auto actual = EvaluateDensity(scene, neighbors, 1.0f, 2.0f,
				selectedChunks, useSelectedChunks, N_ownedF);
			ExpectDensityNear(actual, expected);
			for (uint32_t i = N_ownedF; i < N_F; i++)
			{
				SCOPED_TRACE(i);
				EXPECT_EQ(actual[i], untouchedDensity);
			}
		}
	}

	// fluid gathers must decode every format and apply the shared fluid mass
	TEST_P(SphDensityTests, ComputesFluidDensityAcrossCompressedIndexFormats)
	{
		DensityScene scene;
		scene[0] = MakeParticles(std::vector<glm::vec3>(70'000, glm::vec3{ 0.0f }));
		auto neighbors = MakeNeighbors(static_cast<uint32_t>(scene[0].xPos.size()));
		const std::array<uint32_t, 3> spans{ 255, 256, 65'536 };
		const std::array<Sim::SELLChunkFormat, 3> formats{
			Sim::SELLChunkFormat::Delta8, Sim::SELLChunkFormat::Delta16, Sim::SELLChunkFormat::Abs32 };
		for (uint32_t chunk = 0; chunk < spans.size(); chunk++)
		{
			const uint32_t base = 17 + 3 * chunk;
			scene[0].xPos[base] = 0.25f * (chunk + 1);
			scene[0].zPos[base + spans[chunk]] = 0.5f * (chunk + 1);
			neighbors[0][chunk * CPU::N_SIMD_WIDTH] = { base + spans[chunk], base };
		}
		ExpectDensityNear(EvaluateDensity(scene, neighbors), ReferenceDensity(scene, neighbors));
		for (uint32_t chunk = 0; chunk < spans.size(); chunk++)
		{
			SCOPED_TRACE(chunk);
			EXPECT_EQ(lists_->FFChunk(chunk).format, GetParam() ? formats[chunk] : Sim::SELLChunkFormat::Explicit32);
		}
	}

	// density gathers must decode narrow deltas and absolute indices with distinct per-neighbor weights
	TEST_P(SphDensityTests, ComputesDensityAcrossCompressedIndexFormats)
	{
		auto scene = MakeMixedScene(2 * CPU::N_SIMD_WIDTH + 1);
		for (uint32_t category = 1; category < 3; category++)
		{
			scene[category] = MakeParticles(std::vector<glm::vec3>(70'000, glm::vec3(0.0f)),
				std::vector<float>(70'000, 0.0f));
		}
		auto neighbors = MakeNeighbors(static_cast<uint32_t>(scene[0].xPos.size()));
		const std::array<uint32_t, 3> spans{ 255, 256, 65'536 };
		const std::array<Sim::SELLChunkFormat, 3> formats{
			Sim::SELLChunkFormat::Delta8, Sim::SELLChunkFormat::Delta16, Sim::SELLChunkFormat::Abs32 };
		for (uint32_t chunk = 0; chunk < spans.size(); chunk++)
		{
			const uint32_t base = 17 + 3 * chunk;
			for (uint32_t category = 1; category < 3; category++)
			{
				auto& target = scene[category];
				target.xPos[base] = 0.25f * category;
				target.yPos[base + spans[chunk]] = -0.5f * category;
				target.psi[base] = static_cast<float>(2 + 5 * category + chunk);
				target.psi[base + spans[chunk]] = static_cast<float>(11 + 7 * category + chunk);
				neighbors[category][chunk * CPU::N_SIMD_WIDTH] = { base + spans[chunk], base };
			}
		}
		ExpectDensityNear(EvaluateDensity(scene, neighbors), ReferenceDensity(scene, neighbors));
		for (uint32_t chunk = 0; chunk < spans.size(); chunk++)
		{
			SCOPED_TRACE(chunk);
			EXPECT_EQ(lists_->FBChunk(chunk).format, GetParam() ? formats[chunk] : Sim::SELLChunkFormat::Explicit32);
			EXPECT_EQ(lists_->FRChunk(chunk).format, GetParam() ? formats[chunk] : Sim::SELLChunkFormat::Explicit32);
		}
	}

	INSTANTIATE_TEST_SUITE_P(ExplicitAndCompressed, SphDensityTests, testing::Bool());
}
