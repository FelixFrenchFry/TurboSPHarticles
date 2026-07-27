// dist_redistribution_diagnostics.h

#pragma once

// external
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

// internal
#include "Source/Base/helpers.h"



namespace Dist
{
	enum class ParticleRedistributionTimingOp : uint8_t
	{
		RoutePrep,
		RouteFluid,
		RouteBoundary,
		Pack,
		Exchange,
		FluidPrep,
		KeptFluid,
		MigratedFluid,
		HaloFluid,
		BoundaryPrep,
		KeptBoundary,
		HaloBoundary,
		Backend,
		SyncPack,
		SyncCounts,
		SyncRebuild,
		SyncBackend,

		Count
	};

	constexpr uint8_t GetIdx(const ParticleRedistributionTimingOp op)
	{
		return static_cast<uint8_t>(op);
	}

	// temporary, standalone timing report for profiling particle redistribution
	struct ParticleRedistributionTimingReport
	{
		int rank = 0;
		uint64_t sampleCount = 0;
		std::array<uint64_t, GetIdx(ParticleRedistributionTimingOp::Count)> totalNs{};
	};

	// temporary chrono scope
	class ParticleRedistributionTimingScope final
	{
	public:

		ParticleRedistributionTimingScope(
			ParticleRedistributionTimingReport* report,
			ParticleRedistributionTimingOp op);
		~ParticleRedistributionTimingScope();

		ParticleRedistributionTimingScope(
			const ParticleRedistributionTimingScope&) = delete;
		ParticleRedistributionTimingScope& operator=(
			const ParticleRedistributionTimingScope&) = delete;

	private:

		using Clock = std::chrono::steady_clock;

		ParticleRedistributionTimingReport* report_ = nullptr;
		ParticleRedistributionTimingOp op_ =
			ParticleRedistributionTimingOp::Count;
		Clock::time_point t0_{};
		bool active_ = false;
	};

	void BeginParticleRedistributionTimingReport(
		ParticleRedistributionTimingReport& report,
		int rank);

	std::vector<std::string> BuildParticleRedistributionTimingLines(
		const std::vector<ParticleRedistributionTimingReport>& reports,
		Bench::TimeUnit unit);

} // namespace Dist
