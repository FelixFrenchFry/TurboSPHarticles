// fluid_halo_exchange.cpp

// external
#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <glm/vec3.hpp>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <utility>
#include <vector>

// internal
#include "source/simulation/CPU/backend/cpu_backend.h"
#include "source/simulation/distributed/fluid_solver_sync.h"
#include "source/simulation/distributed/particle_redistribution.h"



namespace
{
	constexpr int N_ranks = 3;
	const Dist::AABB slabBounds{ { 0.0f, 0.0f, 0.0f }, { 4.0f, 4.0f, 12.0f } };
	Dist::MpiContext* mpiContext = nullptr;

	struct HaloTestParticle
	{
		glm::vec3 position;
		int ownerRank;
		int singleHaloRank;
		int doubleHaloRank;
	};

	// three slabs along z, with one-way traffic from rank 2 in double-halo mode
	constexpr std::array<HaloTestParticle, 7> particles{ {
		{ { 2.25f, 1.0f, 5.50f }, 1, -1, 0 },
		{ { 2.25f, 1.0f, 3.75f }, 0, 1, 1 },
		{ { 2.25f, 1.0f, 4.75f }, 1, 0, 0 },
		{ { 0.25f, 1.0f, 3.25f }, 0, 1, 1 },
		{ { 0.25f, 1.0f, 4.25f }, 1, 0, 0 },
		{ { 0.25f, 1.0f, 9.75f }, 2, -1, 1 },
		{ { 0.25f, 1.0f, 10.25f }, 2, -1, -1 }
	} };

	Sim::SimParams MakeParams(const Base::DistHaloMode mode)
	{
		Sim::SimParams params;
		params.distHaloMode = mode;
		params.particleSizeF = 0.5f;
		params.particleSizeInvF = 2.0f;
		params.kernelRadius = 1.0f;
		params.kernelRadiusSqu = 1.0f;
		params.distHaloRadius = mode == Base::DistHaloMode::DoubleHalo ? 2.0f : 1.0f;
		params.massF = 125.0f;
		params.viscosityF = 0.0f;
		params.viscosityB = 0.0f;
		params.gravity = glm::vec3(0.0f);
		params.jacobiWarmStart = 1.0f;
		params.fixedIters = 2;
		return params;
	}

	glm::vec3 InitialVelocity(const uint64_t id)
	{
		const float value = static_cast<float>(id);
		return { value + 1.0f, value + 11.0f, -value - 21.0f };
	}

	Sync::UniversalSimState MakeState()
	{
		Sync::UniversalSimState state;
		state.ResizeFluid(particles.size());
		auto& fluid = state.particlesF;
		for (uint64_t id = 0; id < particles.size(); id++)
		{
			fluid.id[id] = id;
			fluid.xPos[id] = particles[id].position.x;
			fluid.yPos[id] = particles[id].position.y;
			fluid.zPos[id] = particles[id].position.z;
			const auto velocity = InitialVelocity(id);
			fluid.xVel[id] = velocity.x;
			fluid.yVel[id] = velocity.y;
			fluid.zVel[id] = velocity.z;
			fluid.pressure[id] = 100.0f + static_cast<float>(id);
		}
		return state;
	}

	void ExecuteDistributedStep(Sim::SimBackend_CPU& backend,
		Dist::FluidSolverSyncMgr& sync, Dist::DistParticleState& fluidState,
		const Dist::IDomainDecomp& decomp, const Sim::SimParams& params, const uint64_t N_global)
	{
		auto& mpi = *mpiContext;
		ASSERT_TRUE(mpi.AllTrue(backend.ConfigureDistributedParticleLayout(fluidState.N_owned, 0, 0)));
		const auto fluid = backend.GetFluidParticleView();
		std::vector<uint32_t> outerLocalIdxs;
		Dist::ClassifyOwnedFluidParticlesIntoInnerOuter(
			fluidState, fluid, decomp, mpi.Rank(), params.distHaloRadius, &outerLocalIdxs);

		std::string failMsg;
		ASSERT_TRUE(mpi.AllTrue(sync.Rebuild(fluid, fluidState, outerLocalIdxs, decomp,
			params.kernelRadius, params.distHaloMode, N_global, mpi, &failMsg))) << failMsg;
		backend.TakeFluidSolverSyncLocalIdxs(sync);
		ASSERT_TRUE(mpi.AllTrue(backend.ConfigureDistributedStep(outerLocalIdxs, sync)));
		backend.ExecuteFullSimStep(params, 1);
	}

	class FluidHaloExchangeTests : public testing::TestWithParam<Base::DistHaloMode> {};

	// shuffled local slots and distinct velocity components expose packing and remapping errors
	TEST_P(FluidHaloExchangeTests, PreservesHaloFieldsByParticleIdAfterReorder)
	{
		auto& mpi = *mpiContext;
		const int rank = mpi.Rank();
		SCOPED_TRACE(rank);
		const auto mode = GetParam();
		const auto params = MakeParams(mode);
		const bool doubleHalo = mode == Base::DistHaloMode::DoubleHalo;

		auto decomp = Dist::MakeDomainDecomp(Base::DomainDecompMethod::AABB_Slabs);
		decomp->Build({ slabBounds, params.distHaloRadius, N_ranks });
		Sync::UniversalSimState state;
		Dist::DistParticleState fluidState, boundaryState, rigidState;
		Dist::BuildLocalParticlesFromGlobalState(
			state, fluidState, boundaryState, rigidState, MakeState(), *decomp, rank);
		// distinguish a missed pressure update from a valid zero received from the owner
		std::fill_n(state.particlesF.pressure.begin() + fluidState.N_owned, fluidState.N_halo, -1.0f);

		Bench::BenchmarksTracker benchmarks(params);
		Sim::SimBackend_CPU backend(benchmarks);
		backend.SetSimDomainBounds(slabBounds.min, slabBounds.max);
		backend.PullFromUniversalSimState(state, params);
		const auto before = backend.GetFluidParticleView();
		const std::vector<uint64_t> idsBefore(before.id.begin(), before.id.end());

		Dist::FluidSolverSyncMgr sync;
		ASSERT_NO_FATAL_FAILURE(ExecuteDistributedStep(backend, sync, fluidState, *decomp, params, particles.size()));

		const auto after = backend.GetFluidParticleView();
		if (rank < 2) { EXPECT_FALSE(std::ranges::equal(idsBefore, after.id)); }
		ASSERT_TRUE(mpi.AllTrue(std::ranges::all_of(after.id,
			[](const uint64_t id) { return id < particles.size(); })));

		std::vector<float> ownerPressure(particles.size(), 0.0f);
		for (uint32_t i = 0; i < fluidState.N_owned; i++)
		{
			EXPECT_EQ(particles[after.id[i]].ownerRank, rank);
			ownerPressure[after.id[i]] = after.pressure[i];
		}
		mpi.AllReduceSumFloats(ownerPressure);

		std::vector<uint64_t> actualHaloIds;
		for (uint32_t i = fluidState.N_owned; i < after.N_; i++)
		{
			const uint64_t id = after.id[i];
			SCOPED_TRACE(id);
			actualHaloIds.push_back(id);
			// predicted velocities use the single halo in both modes
			if (particles[id].singleHaloRank == rank)
			{
				const auto expected = InitialVelocity(id);
				EXPECT_EQ(after.xVelPred[i], expected.x);
				EXPECT_EQ(after.yVelPred[i], expected.y);
				EXPECT_EQ(after.zVelPred[i], expected.z);
			}
			EXPECT_EQ(after.pressure[i], ownerPressure[id]);
		}
		std::ranges::sort(actualHaloIds);
		std::vector<uint64_t> expectedHaloIds;
		for (uint64_t id = 0; id < particles.size(); id++)
		{
			const auto& particle = particles[id];
			const int haloRank = doubleHalo ? particle.doubleHaloRank : particle.singleHaloRank;
			if (haloRank == rank) { expectedHaloIds.push_back(id); }
		}
		EXPECT_EQ(actualHaloIds, expectedHaloIds);
	}

	INSTANTIATE_TEST_SUITE_P(HaloModes, FluidHaloExchangeTests,
		testing::Values(Base::DistHaloMode::SingleHalo, Base::DistHaloMode::DoubleHalo),
		[](const testing::TestParamInfo<Base::DistHaloMode>& info)
		{
			return Base::ToCString(info.param);
		});

	// an over-dense pair straddles the rank boundary, so each pressure update uses the other's acceleration
	TEST(DoubleHaloPressureTests, MatchesSerialPressureAcrossRankBoundary)
	{
		const int rank = mpiContext->Rank();
		SCOPED_TRACE(rank);
		auto params = MakeParams(Base::DistHaloMode::DoubleHalo);
		params.restDensity = 500.0f;
		params.timeStep = 0.01f;
		params.jacobiWarmStart = 0.0f;

		Sync::UniversalSimState pair;
		pair.ResizeFluid(2);
		pair.particlesF.id = { 0, 1 };
		pair.particlesF.xPos = { 1.0f, 1.0f };
		pair.particlesF.yPos = { 1.0f, 1.0f };
		pair.particlesF.zPos = { 3.875f, 4.125f };

		auto serialState = pair;
		Bench::BenchmarksTracker serialBenchmarks(params);
		Sim::SimBackend_CPU serialBackend(serialBenchmarks);
		serialBackend.SetSimDomainBounds(slabBounds.min, slabBounds.max);
		serialBackend.PullFromUniversalSimState(serialState, params);
		serialBackend.ExecuteFullSimStep(params, 1);
		const auto expected = serialBackend.GetFluidParticleView();

		auto decomp = Dist::MakeDomainDecomp(Base::DomainDecompMethod::AABB_Slabs);
		decomp->Build({ slabBounds, params.distHaloRadius, N_ranks });
		Sync::UniversalSimState state;
		Dist::DistParticleState fluidState, boundaryState, rigidState;
		Dist::BuildLocalParticlesFromGlobalState(
			state, fluidState, boundaryState, rigidState, std::move(pair), *decomp, rank);

		Bench::BenchmarksTracker benchmarks(params);
		Sim::SimBackend_CPU backend(benchmarks);
		backend.SetSimDomainBounds(slabBounds.min, slabBounds.max);
		backend.PullFromUniversalSimState(state, params);
		Dist::FluidSolverSyncMgr sync;
		ASSERT_NO_FATAL_FAILURE(ExecuteDistributedStep(backend, sync, fluidState, *decomp, params, 2));

		const auto actual = backend.GetFluidParticleView();
		EXPECT_EQ(fluidState.N_owned, rank < 2 ? 1u : 0u);
		for (uint32_t i = 0; i < fluidState.N_owned; i++)
		{
			const uint64_t id = actual.id[i];
			SCOPED_TRACE(id);
			const auto expectedId = std::ranges::find(expected.id, id);
			ASSERT_NE(expectedId, expected.id.end());
			const float pressure = expected.pressure[expectedId - expected.id.begin()];
			EXPECT_GT(pressure, 0.0f);
			EXPECT_NEAR(actual.pressure[i], pressure, pressure * 1e-5f);
		}
	}

	// provides malformed backend views through the public redistribution entry point
	class RigidRedistributionTestBackend final : public Sim::I_DistributedParticleBackend
	{
	public:
		explicit RigidRedistributionTestBackend(const Sync::RigidConstView rigidView)
			: rigidView_(rigidView) {}

		Sync::FluidConstView GetFluidParticleView() const override { return {}; }
		Sync::BoundaryConstView GetBoundaryParticleView() const override { return {}; }
		Sync::RigidConstView GetRigidParticleView() const override { return rigidView_; }
		Sync::ParticlePositionConstView GetRigidParticleWorldPositionView() const override { return {}; }
		bool ApplyParticleRedistribution(
			uint32_t, uint32_t, uint32_t,
			std::span<const Sync::ParticleCompactionMove>,
			std::span<const Sync::ParticleCompactionMove>,
			std::span<const Sync::ParticleCompactionMove>,
			const Sync::FluidParticles&, const Sync::BoundaryParticles&, const Sync::RigidParticles&,
			uint32_t, uint32_t, uint32_t, bool, const Sim::SimParams&) override
		{
			ADD_FAILURE() << "Malformed rigid particle view reached backend mutation";
			return false;
		}
		void CopyRigidBodyStateTo(Sync::RigidBodies&) const override {}
		bool ApplyRigidBodyStateFrom(const Sync::RigidBodies&) override { return false; }
		bool ConfigureDistributedParticleLayout(uint32_t, uint32_t, uint32_t) override { return false; }
		void TakeFluidSolverSyncLocalIdxs(Sim::I_DistributedFluidSolverSync&) override {}
		bool ConfigureDistributedStep(
			const std::vector<uint32_t>&, Sim::I_DistributedFluidSolverSync&) override
		{
			return false;
		}
		bool BuildExportFluidParticleLocalIdxs(
			const Sim::SimParams&, Sim::I_DistributedFluidSolverSync&, std::vector<uint32_t>&) const override
		{
			return false;
		}
		bool GetRigidBodyForceTorque(Sim::RigidBodyForceTorque&) const override { return false; }
		bool ApplyRigidBodyForceTorque(const Sim::RigidBodyForceTorque&, float) override { return false; }
		bool FillStatisticsPartial(Stats::StatisticsPartial&, const Sim::SimParams&) const override { return false; }

	private:
		Sync::RigidConstView rigidView_;
	};

	enum class RigidViewFault : uint8_t
	{
		ShortBodyIds,
		InvalidBodyIndex
	};

	void ExpectRigidRedistributionFailure(const RigidViewFault fault)
	{
		auto& mpi = *mpiContext;
		const int rank = mpi.Rank();
		SCOPED_TRACE(rank);
		const auto params = MakeParams(Base::DistHaloMode::SingleHalo);
		auto decomp = Dist::MakeDomainDecomp(Base::DomainDecompMethod::AABB_Slabs);
		decomp->Build({ slabBounds, params.distHaloRadius, N_ranks });

		Sync::RigidBodies bodies;
		// back both indices with transforms so a missed bounds check reaches the halo exchange
		bodies.resizeAll(2);
		bodies.N_ = 1;
		bodies.rigidParticleCount = { 1 };
		bodies.zCM = { 3.75f, 3.75f };

		Sync::RigidParticles rigid;
		rigid.resizeAll(rank == 0 ? 1 : 0);
		if (rank == 0)
		{
			rigid.id[0] = 0;
			rigid.bodyId[0] = fault == RigidViewFault::InvalidBodyIndex ? bodies.N_ : 0;
			rigid.xPosRel[0] = 1.0f;
			rigid.yPosRel[0] = 1.0f;
		}
		auto rigidView = rigid.constView();
		if (rank == 0 && fault == RigidViewFault::ShortBodyIds)
		{
			rigidView.bodyId = rigidView.bodyId.first(0);
		}

		RigidRedistributionTestBackend backend(rigidView);
		Dist::ParticleRedistributionMgr redistribution;
		Dist::DistParticleState fluidState, boundaryState, rigidState;
		rigidState.N_owned = rigid.N_;
		std::vector<uint32_t> outerLocalIdxs;
		std::string failMsg;
		EXPECT_FALSE(redistribution.RedistributeParticles(
			backend, fluidState, boundaryState, rigidState, outerLocalIdxs, bodies, *decomp, mpi,
			params, false, 0, 0, 1, &failMsg));
		if (rank == 0) { EXPECT_FALSE(failMsg.empty()); }
		EXPECT_EQ(redistribution.GetStats().N_recvHaloR, 0u);
	}

	// a truncated body-ID span must fail before exchanging halos or mutating the backend
	TEST(RigidRedistributionTests, RejectsShortBodyIdViewBeforeHaloExchange)
	{
		ExpectRigidRedistributionFailure(RigidViewFault::ShortBodyIds);
	}

	// an out-of-range body index must fail before exchanging halos or mutating the backend
	TEST(RigidRedistributionTests, RejectsOutOfRangeBodyIndexBeforeHaloExchange)
	{
		ExpectRigidRedistributionFailure(RigidViewFault::InvalidBodyIndex);
	}
}

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	// let the MPI abort handler catch exceptions from the test body
	GTEST_FLAG_SET(catch_exceptions, false);
	Dist::MpiContext mpi(argc, argv);
	if (mpi.Size() != N_ranks) { mpi.Abort("Fluid halo exchange tests require exactly three MPI ranks."); }
	mpiContext = &mpi;
	try
	{
		const int result = RUN_ALL_TESTS();
		return mpi.AllTrue(result == 0) ? 0 : 1;
	}
	catch (const std::exception& ex)
	{
		mpi.Abort(ex.what());
	}
	catch (...)
	{
		mpi.Abort("Unknown exception in the MPI tests");
	}
}
