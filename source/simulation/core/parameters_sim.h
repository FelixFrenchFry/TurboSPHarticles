// parameters_sim.h

#pragma once

// external
#include <atomic>
#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// internal
#include "source/base/enum_hub.h"
#include "source/base/helpers.h"
#include "source/eval/stats/metrics.h"



namespace Sim
{
	// TODO: leave cubic spline kernels hardcoded for support ratio 2.0?
	inline constexpr float CubicSplineKernelSuppRatio = 2.0f;

	// snapshot of simulation parameters (passed to compute-heavy functions)
	struct SimParams
	{
		// modes
		Base::BackendVersion backendVersion =
			Base::BackendVersion::CPU01;
		Base::ScenarioType scenarioType	=
			Base::ScenarioType::BreakingDam;
		Base::NeighborSearchMethod neighborSearchMethod =
			Base::NeighborSearchMethod::CellLinkedList_Compact;
		Base::NeighborListMethod neighborListMethod =
			Base::NeighborListMethod::ChunkedSELLX_AdaptCompress;
		Base::DomainDecompMethod domainDecompMethod =
			Base::DomainDecompMethod::AABB_Uniform; // (distributed only)

		// simulation state
		bool simRunning         = false;
		bool simRunNextStep     = false;
		int64_t maxSteps        = -1;

		// time
		float timeStep          = 0.0001f;

		// particle sizes
		// (is treated as smoothing length and spacing "h")
		float particleSizeF     = 0.01f;
		float particleSizeB     = 0.01f;
		float particleSizeInvF  = 100.0f;
		float particleSizeInvB  = 100.0f;

		// kernel
		float kernelSuppRatio   = CubicSplineKernelSuppRatio;
		float kernelRadius      = 0.02f; // radius = size * supp ratio
		float kernelRadiusSqu   = 0.0004f;

		// neighbor count heuristics
		// TODO: use p99 values for heuristics?
		uint32_t expctdNbrCountSoftMax_FF = 40;
		uint32_t expctdNbrCountSoftMax_FB = 20;
		uint32_t expctdNbrCountHardMax_FF = 64;
		uint32_t expctdNbrCountHardMax_FB = 32;
		// (used for the initial reserve of large packed CSR neighbor list buffers)
		uint32_t expctdAvgNbrCountHardMax_FF = 36;
		uint32_t expctdAvgNbrCountHardMax_FB = 4;

		// density and mass
		float restDensity       = 1'000.0f;
		float massF             = 0.001f;
		float massB             = 0.001f;

		// non-pressure accelerations
		float viscosityF        = 0.000003f;
		float viscosityB        = 0.000003f;
		glm::vec3 gravity       = { 0.0f, -9.81f, 0.0f };

		// pressure solving
		float pressGamma        = 0.5f;
		float jacobiOmega       = 0.6f;
		float densityDevTarget  = 0.01f;
		uint32_t fixedIters     = 10;
		uint32_t minSolverIters = 5;
		uint32_t maxSolverIters = 50;
		Base::SolverTerminationCond solverTerminationCond =
			Base::SolverTerminationCond::FixedIters;

		// data structuring
		float gridCellSizeRatio = 1.0f;

		// fluid conditions
		float jitterFracF       = 0.0f; // % of particle size
		int64_t jitterSeedF     = 42; // RNG seed for jitter

		// boundary conditions
		float jitterFracB       = 0.0f; // % of particle size
		int64_t jitterSeedB     = 42; // RNG seed for jitter
		uint32_t boundaryLayers = 2;

		// surface sampling spacing
		float surfaceSamplingMinDistanceRatio = 0.8f; // minimum accepted sample spacing / particleSizeB
		float surfaceSamplingMaxDistanceRatio = 1.0f; // maximum surface-to-sample gap / particleSizeB
		float surfaceSamplingCoverageProbeSpacingRatio = 0.2f; // coverage probe spacing / particleSizeB

		// resolved surface sampling distances
		float surfaceSamplingMinDistance = 0.008f; // minimum accepted sample spacing
		float surfaceSamplingMaxDistance = 0.01f; // maximum surface-to-sample gap
		float surfaceSamplingCoverageProbeSpacing = 0.002f; // spacing of coverage probes

		// candidate generation
		uint32_t surfaceSamplingCandidateMultiplier = 8; // candidate pool size multiplier
		uint32_t surfaceSamplingMaxTrialsPerCell = 10; // max candidate trials per spatial cell
		uint64_t surfaceSamplingSeed = 42; // RNG seed

		// threading
		int64_t maxThreadsSim = -1; // -1 = use default budget

		// benchmarks
		uint32_t maxBenchRecs = 1'000'000;
		Bench::TimeUnit timeUnit = Bench::TimeUnit::ms;

		// memory
		uint32_t maxMemRecs = 1'000'000;
		Mem::ByteUnit byteUnit = Mem::ByteUnit::MB;

		// statistics
		uint32_t maxStatRecs = 1'000'000;
		std::array<bool, static_cast<uint8_t>(Stats::StatGroup::Count)> statGroupsEnabled =
		{
			false,	// particles
			false,	// density
			false	// velocity
		};

		// export
		bool exportEnabled      = true;
		std::string exportDir   = "A";
		uint32_t exportInterval = 10;

		// export surface classification
		float surfaceNbrCountRatio = 0.75f; // near-surface neighborhood fill ratio
		uint32_t surfaceNbrCountThreshold = 27; // fluid neighbor count derived from the ratio
	};

	class SimParamsMgr
	{
	public: // ---------------------------------------------------------------------------------------

		SimParamsMgr() = default;

		// returns snapshot and version of current simulation parameters
		std::pair<SimParams, uint64_t> SnapshotParams() const;

		// returns current stable version number to check for changes
		uint64_t StableVersion() const;

		bool LoadSimParamsFromJson(const nlohmann::json& j, bool verbose = false);
		bool LoadSimParamsFromProfile(const std::string& profile, bool verbose = false);
		bool LoadSimParamsFromCliOverrides(
			const std::vector<std::string>& overrides, bool verbose = false);

		// setters for individual parameters and their dependent values
		void Set_SimRunning(bool simRunning, bool verbose = false);
		void Set_SimRunNextStep(bool simRunNextStep, bool verbose = false);
		void Set_TimeStep(float timeStep, bool verbose = false);
		void Set_ParticleSize(float particleSize, Base::ParticleType type, bool verbose = false);
		void Set_KernelSuppRatio(float kernelSuppRatio, bool verbose = false);
		void Set_KernelRadius(float kernelRadius, bool verbose = false);
		void Set_RestDensity(float restDensity, bool verbose = false);
		void Set_Mass(float mass, Base::ParticleType type, bool verbose = false);
		void Set_Viscosity(float viscosity, Base::ParticleType type, bool verbose = false);
		void Set_Gravity(glm::vec3 gravity, bool verbose = false);
		void Set_PressGamma(float pressGamma, bool verbose = false);
		void Set_JacobiOmega(float jacobiOmega, bool verbose = false);
		void Set_DensityDevTarget(float densityDevTarget, bool verbose = false);
		void Set_MinSolverIters(uint32_t minSolverIters, bool verbose = false);
		void Set_MaxSolverIters(uint32_t maxSolverIters, bool verbose = false);
		void Set_BoundaryLayers(uint32_t boundaryLayers, bool verbose = false);
		void Set_SurfaceSamplingDistanceRatios(
			float minDistanceRatio,
			float maxDistanceRatio,
			float coverageProbeSpacingRatio,
			bool verbose = false);
		void Set_SurfaceNbrCountRatio(float surfaceNbrCountRatio, bool verbose = false);

		// helpers
		bool AssertParamDerivationCorrectness() const;
		uint32_t HeuristicExpectedNeighborCount_SoftMax(
			float kernelSuppRatio, Base::ParticleType type) const;
		uint32_t HeuristicExpectedNeighborCount_HardMax(
			float kernelSuppRatio, Base::ParticleType type) const;
		uint32_t HeuristicExpectedAvgNeighborCount_HardMax(
			float kernelSuppRatio, Base::ParticleType type) const;
		uint32_t HeuristicSurfaceNbrCountThreshold(
			float surfaceNbrCountRatio, uint32_t expctdAvgNbrCountHardMax_FF) const;

	private: // --------------------------------------------------------------------------------------

		alignas(64) SimParams params_{}; // current active copy of visualization parameters
		alignas(64) std::atomic<uint64_t> version_{ 0 }; // even = stable, odd = writing

		// keeps the dilated export surface selection from covering the whole fluid domain
		static constexpr float surfaceNbrCountRatioMax_ = 0.85f;

		void UpdateNeighborCountHeuristics_();
		void UpdateSurfaceSamplingDistances_();
		void UpdateSurfaceNbrCountThreshold_();
		static bool SurfaceSamplingDistanceRatiosValid_(
			float minDistanceRatio,
			float maxDistanceRatio,
			float coverageProbeSpacingRatio,
			float particleSizeF,
			float particleSizeB);
		static bool SurfaceNbrCountRatioValid_(float surfaceNbrCountRatio);
		inline void BeginWrite(); // mark beginning of write operation
		inline void EndWrite(); // mark end of write operation
	};

} // namespace Sim
