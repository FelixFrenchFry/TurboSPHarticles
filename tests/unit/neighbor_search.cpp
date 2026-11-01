// neighbor_search.cpp

// external
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/vec3.hpp>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>

// internal
#include "source/simulation/CPU/backend/cpu_backend.h"



namespace Sim
{
	// read-only test access to neighbor lists
	// (scene loading and search execution use the public backend API)
	class NeighborSearchTestAccess
	{
	public:
		static const I_NeighborListSELL& NeighborLists(const SimBackend_CPU& backend)
		{
			return *backend.neighborList_;
		}
	};
}

namespace
{
	struct SearchParticle
	{
		uint64_t id;
		glm::vec3 position;
		uint32_t bodyIdx = 0;
	};

	// populations follow the universal state order: fluid, boundary, rigid
	using SearchParticles = std::array<std::vector<SearchParticle>, 3>;
	using NeighborIdsByParticle = std::map<uint64_t, std::vector<uint64_t>>;
	using NeighborSearchConfig = std::tuple<Base::NeighborSearchMethod, Base::NeighborListMethod>;

	SearchParticles MakeParticles(const std::vector<glm::vec3>& fluid,
		const std::vector<glm::vec3>& boundary = {}, const std::vector<glm::vec3>& rigid = {},
		const std::vector<uint32_t>& bodyIds = {})
	{
		const std::array<std::span<const glm::vec3>, 3> positions{ fluid, boundary, rigid };
		SearchParticles particles;
		for (uint32_t population = 0; population < particles.size(); population++)
		{
			for (uint32_t i = 0; i < positions[population].size(); i++)
			{
				// noncontiguous, descending IDs distinguish identity from local index and sorting order
				const uint64_t id = 5'000'000'000ull * (population + 1) + 17 * (positions[population].size() - i);
				const uint32_t bodyIdx = population == 2 ? (bodyIds.empty() ? i : bodyIds[i]) : 0;
				particles[population].push_back({ id, positions[population][i], bodyIdx });
			}
		}
		return particles;
	}

	Sync::UniversalSimState MakeState(const SearchParticles& particles)
	{
		Sync::UniversalSimState state;
		state.ResizeFluid(particles[0].size());
		state.ResizeBoundary(particles[1].size());
		state.ResizeRigid(particles[2].size());
		for (uint32_t i = 0; i < particles[0].size(); i++)
		{
			const auto& particle = particles[0][i];
			state.particlesF.id[i] = particle.id;
			state.particlesF.xPos[i] = particle.position.x;
			state.particlesF.yPos[i] = particle.position.y;
			state.particlesF.zPos[i] = particle.position.z;
		}
		for (uint32_t i = 0; i < particles[1].size(); i++)
		{
			const auto& particle = particles[1][i];
			state.particlesB.id[i] = particle.id;
			state.particlesB.xPos[i] = particle.position.x;
			state.particlesB.yPos[i] = particle.position.y;
			state.particlesB.zPos[i] = particle.position.z;
			state.particlesB.volB[i] = 1.0f;
			state.particlesB.psiB[i] = 1000.0f;
		}
		uint32_t N_bodies = 0;
		for (const auto& particle : particles[2]) { N_bodies = std::max(N_bodies, particle.bodyIdx + 1); }
		state.ResizeRigidBodies(N_bodies);
		for (uint32_t bodyIdx = 0; bodyIdx < N_bodies; bodyIdx++)
		{
			state.rigidBodies.id[bodyIdx] = bodyIdx;
			state.rigidBodies.mass[bodyIdx] = 1.0f;
			state.rigidBodies.massInv[bodyIdx] = 1.0f;
			state.rigidBodies.iBodyInv00[bodyIdx] = 1.0f;
			state.rigidBodies.iBodyInv11[bodyIdx] = 1.0f;
			state.rigidBodies.iBodyInv22[bodyIdx] = 1.0f;
		}
		for (uint32_t i = 0; i < particles[2].size(); i++)
		{
			const auto& particle = particles[2][i];
			state.particlesR.id[i] = particle.id;
			state.particlesR.bodyId[i] = particle.bodyIdx;
			// identity orientation and zero center of mass make relative and world positions equal
			state.particlesR.xPosRel[i] = particle.position.x;
			state.particlesR.yPosRel[i] = particle.position.y;
			state.particlesR.zPosRel[i] = particle.position.z;
			state.particlesR.volR[i] = 1.0f;
			state.particlesR.psiR[i] = 1000.0f;
			state.rigidBodies.rigidParticleCount[particle.bodyIdx]++;
		}
		return state;
	}

	// independent all-pairs reference, including fluid self-neighbors and rigid body exclusions
	NeighborIdsByParticle ReferenceNeighbors(const SearchParticles& particles, const uint32_t source,
		const uint32_t target, const double radiusSqu)
	{
		NeighborIdsByParticle result;
		for (const auto& a : particles[source])
		{
			auto& neighbors = result[a.id];
			for (const auto& b : particles[target])
			{
				if (source == 2 && target == 2 && a.bodyIdx == b.bodyIdx) { continue; }
				const double dx = static_cast<double>(a.position.x) - b.position.x;
				const double dy = static_cast<double>(a.position.y) - b.position.y;
				const double dz = static_cast<double>(a.position.z) - b.position.z;
				if (dx * dx + dy * dy + dz * dz <= radiusSqu) { neighbors.push_back(b.id); }
			}
			std::sort(neighbors.begin(), neighbors.end());
		}
		return result;
	}

	void ExpectNeighbors(const Sim::SimBackend_CPU& backend, const SearchParticles& particles, const float radiusSqu)
	{
		const auto& lists = Sim::NeighborSearchTestAccess::NeighborLists(backend);
		const std::array<std::span<const uint64_t>, 3> ids{
			backend.GetFluidParticleView().id, backend.GetBoundaryParticleView().id, backend.GetRigidParticleView().id };
		constexpr std::array readLists{
			&Sim::I_NeighborList::FF, &Sim::I_NeighborList::FB, &Sim::I_NeighborList::FR,
			&Sim::I_NeighborList::RF, &Sim::I_NeighborList::RB, &Sim::I_NeighborList::RR };
		constexpr std::array<uint32_t, 6> sources{ 0, 0, 0, 2, 2, 2 };
		constexpr std::array<uint32_t, 6> targets{ 0, 1, 2, 0, 1, 2 };
		for (uint32_t population = 0; population < particles.size(); population++)
		{
			EXPECT_EQ(ids[population].size(), particles[population].size());
		}
		for (uint32_t category = 0; category < readLists.size(); category++)
		{
			SCOPED_TRACE(category);
			const uint32_t source = sources[category];
			const uint32_t target = targets[category];
			NeighborIdsByParticle actual;
			for (uint32_t i = 0; i < ids[source].size(); i++)
			{
				SCOPED_TRACE(ids[source][i]);
				const auto [entry, inserted] = actual.emplace(ids[source][i], std::vector<uint64_t>{});
				ASSERT_TRUE(inserted);
				// consume each span before the next retrieval can overwrite its scratch buffer
				const auto neighbors = (lists.*readLists[category])(i);
				if (neighbors.N > 0) { ASSERT_NE(neighbors.data, nullptr); }
				for (uint32_t k = 0; k < neighbors.N; k++)
				{
					ASSERT_LT(neighbors.data[k], ids[target].size());
					entry->second.push_back(ids[target][neighbors.data[k]]);
				}
				// preserve duplicates so repeated neighbors cannot disappear during comparison
				std::sort(entry->second.begin(), entry->second.end());
			}
			EXPECT_EQ(actual, ReferenceNeighbors(particles, source, target, radiusSqu));
		}
	}

	class NeighborSearchTests : public testing::TestWithParam<NeighborSearchConfig>
	{
	protected:
		void SetUp() override
		{
			params_.neighborSearchMethod = std::get<0>(GetParam());
			params_.neighborListMethod = std::get<1>(GetParam());
			params_.particleSizeF = 0.5f;
			params_.particleSizeB = 0.5f;
			params_.particleSizeInvF = 2.0f;
			params_.particleSizeInvB = 2.0f;
			params_.kernelRadius = 1.0f;
			params_.kernelRadiusSqu = 1.0f;
			params_.distHaloRadius = 1.0f;
			params_.massF = 125.0f;
			params_.massB = 125.0f;
			params_.expctdAvgNbrCountHardMax_FF = 0;
			params_.expctdAvgNbrCountHardMax_FB = 0;
			benchmarks_ = std::make_unique<Bench::BenchmarksTracker>(params_);
			backend_ = std::make_unique<Sim::SimBackend_CPU>(*benchmarks_, params_);
		}

		void SearchAndExpect(const SearchParticles& particles)
		{
			backend_->ExecuteNeighborSearch(params_);
			ExpectNeighbors(*backend_, particles, params_.kernelRadiusSqu);
		}

		void LoadAndSearch(const SearchParticles& particles)
		{
			auto state = MakeState(particles);
			backend_->PullFromUniversalSimState(state, params_);
			SearchAndExpect(particles);
		}

		Sim::SimParams params_;
		std::unique_ptr<Bench::BenchmarksTracker> benchmarks_;
		std::unique_ptr<Sim::SimBackend_CPU> backend_;
	};

	// shuffled input order and separate clusters exercise all six neighbor categories by stable ID
	TEST_P(NeighborSearchTests, FindsMixedPopulationNeighbors)
	{
		const auto particles = MakeParticles(
			{ { 3.0f, 0.0f, 0.0f }, { -0.25f, 0.0f, 0.0f }, { 0.5f, 0.25f, 0.0f },
				{ -3.0f, 1.0f, 0.0f }, { 0.25f, -0.5f, 0.25f } },
			{ { -3.0f, 0.5f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 3.5f, 0.25f, 0.0f }, { 6.0f, 6.0f, 6.0f } },
			{ { 0.25f, 0.25f, 0.25f }, { -0.5f, 0.25f, 0.0f }, { 3.25f, 0.0f, 0.0f }, { -3.0f, 1.5f, 0.0f } },
			{ 0, 0, 1, 2 });

		LoadAndSearch(particles);
	}

	// exactly representable axis-aligned cases separate the inclusive cutoff from its neighboring floats
	TEST_P(NeighborSearchTests, IncludesExactCutoffAndExcludesOutsidePairs)
	{
		const float inside = std::nextafter(1.0f, 0.0f);
		const float outside = std::nextafter(1.0f, 2.0f);
		const auto particles = MakeParticles(
			{ { outside, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { inside, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
			{ { -outside, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f }, { -inside, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } },
			{ { 1.0f, 0.0f, 0.0f }, { outside, 0.0f, 0.0f }, { inside, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f } });

		LoadAndSearch(particles);
	}

	// binary-fraction coordinates cross positive and negative cell faces, edges, and corners
	TEST_P(NeighborSearchTests, FindsNeighborsAcrossCellBoundaries)
	{
		const auto particles = MakeParticles(
			{ { -0.125f, -0.125f, -0.125f }, { 0.125f, -0.125f, -0.125f },
				{ 0.125f, 0.125f, -0.125f }, { 0.125f, 0.125f, 0.125f },
				{ -1.125f, -0.125f, 0.125f }, { -0.875f, 0.125f, -0.125f }, { 2.125f, 2.125f, 2.125f } },
			{ { 0.875f, -0.125f, 0.125f }, { 1.125f, 0.125f, -0.125f }, { -0.125f, 0.125f, 0.125f } },
			{ { -0.125f, 0.125f, -0.125f }, { 0.125f, -0.125f, 0.125f } });

		LoadAndSearch(particles);
	}

	// smaller cells require querying beyond adjacent cells, including after the cell size changes
	TEST_P(NeighborSearchTests, SearchesBeyondAdjacentCellsWhenCellsAreSmall)
	{
		const auto particles = MakeParticles(
			{ { 0.125f, 0.125f, 0.125f }, { 0.875f, 0.125f, 0.125f },
				{ -0.625f, 0.125f, 0.125f }, { 2.125f, 0.125f, 0.125f } },
			{ { 0.125f, 0.875f, 0.125f }, { 0.125f, -0.625f, 0.125f } },
			{ { 0.125f, 0.125f, 0.875f }, { 0.125f, 0.125f, -0.625f } });
		LoadAndSearch(particles);
		for (const float ratio : { 0.5f, 0.25f, 1.0f })
		{
			SCOPED_TRACE(ratio);
			params_.gridCellSizeRatio = ratio;
			SearchAndExpect(particles);
		}
	}

	// empty scenes and missing populations remain valid through repeated imports and searches
	TEST_P(NeighborSearchTests, HandlesEmptyAndSinglePopulationScenes)
	{
		const std::array<SearchParticles, 5> scenes{
			MakeParticles({}),
			MakeParticles({ { 2.0f, 0.0f, 0.0f }, { -2.0f, 0.0f, 0.0f } }),
			MakeParticles({}, { { 0.0f, 0.0f, 0.0f }, { 0.5f, 0.0f, 0.0f } }),
			MakeParticles({}, {}, { { 0.0f, 0.0f, 0.0f }, { 0.5f, 0.0f, 0.0f } }),
			MakeParticles({}) };
		for (uint32_t i = 0; i < scenes.size(); i++)
		{
			SCOPED_TRACE(i);
			LoadAndSearch(scenes[i]);
		}
	}

	// coincident fluid particles include themselves
	// (rigid particles exclude neighbors from their entire own body)
	TEST_P(NeighborSearchTests, HandlesCoincidentParticlesAndRigidBodyExclusions)
	{
		const auto particles = MakeParticles(
			{ { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } },
			{ { 0.0f, 0.0f, 0.0f } },
			{ { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } },
			{ 0, 0, 1, 1 });

		LoadAndSearch(particles);
	}

	// full and partial SIMD chunks must yield the same neighborhoods as scalar all-pairs queries
	TEST_P(NeighborSearchTests, HandlesParticleCountsAroundSimdWidth)
	{
		for (const uint32_t N_F : { CPU::N_SIMD_WIDTH - 1, CPU::N_SIMD_WIDTH, CPU::N_SIMD_WIDTH + 1 })
		{
			SCOPED_TRACE(N_F);
			std::vector<glm::vec3> fluid;
			for (uint32_t i = 0; i < N_F; i++)
			{
				fluid.push_back({ 0.25f * static_cast<float>(i % 5) - 0.5f,
					0.25f * static_cast<float>(i / 5), i % 2 == 0 ? 0.125f : -0.125f });
			}
			const auto particles = MakeParticles(fluid,
				{ { -0.25f, 0.25f, 0.0f }, { 0.5f, 0.5f, 0.25f } },
				{ { 0.25f, 0.25f, 0.25f }, { -0.5f, 0.5f, -0.25f } });

			LoadAndSearch(particles);
		}
	}

	// importing moved particles reuses the search object while retiring old cell ranges and neighbors
	TEST_P(NeighborSearchTests, RebuildsAfterParticlesMoveBetweenCells)
	{
		const auto original = MakeParticles(
			{ { 0.0f, 0.0f, 0.0f }, { 0.5f, 0.0f, 0.0f }, { -4.0f, 0.25f, 0.25f } },
			{ { 0.25f, 0.0f, 0.0f }, { -4.25f, 0.25f, 0.25f } },
			{ { 0.75f, 0.0f, 0.0f }, { -4.0f, 0.5f, 0.25f } });
		LoadAndSearch(original);
		auto moved = original;
		moved[0][0].position = { 32.0f, 0.0f, 0.0f };
		moved[0][1].position = { -4.5f, 0.25f, 0.25f };
		moved[1][0].position = { 32.25f, 0.0f, 0.0f };
		moved[2][0].position = { 32.75f, 0.0f, 0.0f };
		LoadAndSearch(moved);
		for (auto& population : moved)
		{
			for (auto& particle : population) { particle.position += glm::vec3{ 64.0f, -16.0f, 8.0f }; }
		}
		LoadAndSearch(moved);
		LoadAndSearch(original);
	}

	// body motion updates rigid world positions without reimporting the scene or replacing the search object
	TEST_P(NeighborSearchTests, RebuildsAfterRigidBodyMotion)
	{
		auto particles = MakeParticles({ { 0.0f, 0.0f, 0.0f } }, { { 0.5f, 0.0f, 0.0f } },
			{ { 0.25f, 0.0f, 0.0f }, { 0.75f, 0.0f, 0.0f } });
		LoadAndSearch(particles);
		Sync::RigidBodies bodies;
		backend_->CopyRigidBodyStateTo(bodies);
		bodies.xCM[0] = 32.0f;
		ASSERT_TRUE(backend_->ApplyRigidBodyStateFrom(bodies));
		particles[2][0].position.x = 32.25f;
		SearchAndExpect(particles);

		bodies.xCM[0] = 0.0f;
		ASSERT_TRUE(backend_->ApplyRigidBodyStateFrom(bodies));
		particles[2][0].position.x = 0.25f;
		SearchAndExpect(particles);
	}

	INSTANTIATE_TEST_SUITE_P(SearchAndStorageMethods, NeighborSearchTests,
		testing::Combine(
			testing::Values(Base::NeighborSearchMethod::BruteForce, Base::NeighborSearchMethod::CellLinkedList_Compact),
			testing::Values(Base::NeighborListMethod::ChunkedSELLX_Explicit, Base::NeighborListMethod::ChunkedSELLX_AdaptCompress)),
		[](const testing::TestParamInfo<NeighborSearchConfig>& info)
		{
			const char* search = std::get<0>(info.param) == Base::NeighborSearchMethod::BruteForce
				? "BruteForce" : "CompactCellList";
			const char* storage = std::get<1>(info.param) == Base::NeighborListMethod::ChunkedSELLX_Explicit
				? "Explicit" : "Compressed";
			return std::string(search) + "_" + storage;
		});
}
