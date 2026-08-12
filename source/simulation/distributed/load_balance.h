// load_balance.h

#pragma once

// external
#include <cstdint>
#include <memory>
#include <span>
#include <string>

// internal
#include "source/base/enum_hub.h"
#include "domain_decomp.h"



namespace Sim
{
	struct SimParams;
}

namespace Dist
{
	class MpiContext;

	// rank-local load observations used to evaluate different load metrics
	struct RankLoadSample
	{
		uint64_t step = 0;

		uint32_t N_ownedF = 0;
		uint32_t N_ownedB = 0;
		uint32_t N_haloF = 0;
		uint32_t N_haloB = 0;

		uint64_t rankBackendStepTimeNs = 0;
	};

	// globally visible load state used by rebalance triggers
	struct LoadBalanceSnapshot
	{
		uint64_t step = 0;

		std::span<const double> loadByRank{};
		double meanLoad = 0.0;
		double maxLoad = 0.0;
		double relativeImbalance = 0.0;
	};

	class ILoadMetric
	{
	public:

		virtual ~ILoadMetric() = default;

		virtual const char* Name() const = 0;
		virtual double Evaluate(const RankLoadSample& sample) = 0;
	};

	// turns load sample into load value from owned fluid particle counts
	class RankOwnedParticleCountMetric final : public ILoadMetric
	{
	public:

		const char* Name() const override;
		double Evaluate(const RankLoadSample& sample) override;
	};

	// turns load sample into load value from the rank-local backend step duration
	class RankBackendStepTimeMetric final : public ILoadMetric
	{
	public:

		const char* Name() const override;
		double Evaluate(const RankLoadSample& sample) override;
	};

	class IRebalanceTrigger
	{
	public:

		virtual ~IRebalanceTrigger() = default;

		virtual const char* Name() const = 0;
		virtual bool ShouldRebalance(const LoadBalanceSnapshot& snapshot) = 0;
	};

	// decides to rebalance the load based on fixed step intervals
	class PeriodicTrigger final : public IRebalanceTrigger
	{
	public:

		explicit PeriodicTrigger(uint32_t intervalSteps);

		const char* Name() const override;
		bool ShouldRebalance(const LoadBalanceSnapshot& snapshot) override;

	private:

		uint32_t intervalSteps_ = 1;
	};

	// decides to rebalance the load based on consecutive imbalanced steps, followed by a cooldown
	class SustainedImbalanceTrigger final : public IRebalanceTrigger
	{
	public:

		SustainedImbalanceTrigger(
			double imbalanceThreshold,
			uint32_t sustainSteps,
			uint32_t cooldownSteps);

		const char* Name() const override;
		bool ShouldRebalance(const LoadBalanceSnapshot& snapshot) override;

	private:

		double imbalanceThreshold_ = 0.0;
		uint32_t sustainSteps_ = 1;
		uint32_t cooldownSteps_ = 0;
		uint32_t badStepCount_ = 0;
		uint32_t cooldownRemaining_ = 0;
	};

	std::unique_ptr<ILoadMetric> MakeLoadMetric(Base::LoadBalanceMetric method);

	std::unique_ptr<IRebalanceTrigger> MakeRebalanceTrigger(
		Base::LoadBalanceTrigger method,
		uint32_t intervalSteps,
		double imbalanceThreshold,
		uint32_t sustainSteps,
		uint32_t cooldownSteps);

	class LoadBalanceMgr final
	{
	public:

		bool Configure(
			const Sim::SimParams& params,
			std::string* failMsg = nullptr);

		// [ rank-reduced ] evaluates the local metric and broadcasts the trigger decision
		// then builds a global particle profile and repartitions the domain when requested
		bool EvaluateLoadAndRebalance(
			const RankLoadSample& localSample,
			const DomainDecompRepartitionParticleInput& localParticleInput,
			IDomainDecomp& domainDecomp,
			const MpiContext& mpiContext,
			std::string* failMsg = nullptr);

		bool Enabled() const { return loadMetric_ != nullptr; }

	private:

		std::unique_ptr<ILoadMetric> loadMetric_{ nullptr };
		std::unique_ptr<IRebalanceTrigger> rebalanceTrigger_{ nullptr };
		Base::LoadBalanceWeightModel weightModel_ =
			Base::LoadBalanceWeightModel::OwnedParticleCount;
	};

} // namespace Dist
