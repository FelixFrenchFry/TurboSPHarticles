// particle_migration_diagnostics.h

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
	enum class ParticleMigrationTimingOp : uint8_t
	{
		Pack,
		CountExchange,
		PayloadExchange,
		Aggregate,
		Rebuild,

		Count
	};

	constexpr uint8_t GetIdx(const ParticleMigrationTimingOp op)
	{
		return static_cast<uint8_t>(op);
	}

	// temporary, standalone timing report for profiling the current migration pipeline
	struct ParticleMigrationTimingReport
	{
		int rank = 0;
		uint64_t sampleCount = 0;
		std::array<uint64_t, GetIdx(ParticleMigrationTimingOp::Count)> totalNs{};
	};

	// temporary chrono scope
	class ParticleMigrationTimingScope final
	{
	public:

		ParticleMigrationTimingScope(
			ParticleMigrationTimingReport* report,
			ParticleMigrationTimingOp op);
		~ParticleMigrationTimingScope();

		ParticleMigrationTimingScope(const ParticleMigrationTimingScope&) = delete;
		ParticleMigrationTimingScope& operator=(const ParticleMigrationTimingScope&) = delete;

	private:

		using Clock = std::chrono::steady_clock;

		ParticleMigrationTimingReport* report_ = nullptr;
		ParticleMigrationTimingOp op_ = ParticleMigrationTimingOp::Count;
		Clock::time_point t0_{};
		bool active_ = false;
	};

	void BeginParticleMigrationTimingReport(
		ParticleMigrationTimingReport& report,
		int rank);

	std::vector<std::string> BuildParticleMigrationTimingLines(
		const std::vector<ParticleMigrationTimingReport>& reports,
		Bench::TimeUnit unit);

} // namespace Dist
