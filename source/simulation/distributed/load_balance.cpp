// load_balance.cpp

// external
#include <algorithm>
#include <cmath>
#include <numeric>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

// internal
#include "source/simulation/core/parameters_sim.h"
#include "load_balance.h"
#include "mpi_context.h"



using namespace Dist;

const char* RankOwnedParticleCountMetric::Name() const
{
	return "RankOwnedParticleCount";
}

double RankOwnedParticleCountMetric::Evaluate(const RankLoadSample& sample)
{
	return static_cast<double>(sample.N_ownedF) +
		static_cast<double>(sample.N_ownedB);
}

const char* RankBackendStepTimeMetric::Name() const
{
	return "RankBackendStepTime";
}

double RankBackendStepTimeMetric::Evaluate(const RankLoadSample& sample)
{
	return static_cast<double>(sample.rankBackendStepTimeNs);
}

PeriodicTrigger::PeriodicTrigger(const uint32_t intervalSteps)
	: intervalSteps_(intervalSteps)
{
	if (intervalSteps_ == 0)
	{
		throw std::runtime_error("Periodic load balance interval must be greater than zero.");
	}
}

const char* PeriodicTrigger::Name() const
{
	return "Periodic";
}

bool PeriodicTrigger::ShouldRebalance(const LoadBalanceSnapshot& snapshot)
{
	return snapshot.step > 0 && snapshot.step % intervalSteps_ == 0;
}

SustainedImbalanceTrigger::SustainedImbalanceTrigger(
	const double imbalanceThreshold,
	const uint32_t sustainSteps,
	const uint32_t cooldownSteps)
	: imbalanceThreshold_(imbalanceThreshold)
	, sustainSteps_(sustainSteps)
	, cooldownSteps_(cooldownSteps)
{
	if (!std::isfinite(imbalanceThreshold_) || imbalanceThreshold_ < 0.0)
	{
		throw std::runtime_error(
			"Sustained load balance threshold must be finite and non-negative.");
	}
	if (sustainSteps_ == 0)
	{
		throw std::runtime_error(
			"Sustained load balance step count must be greater than zero.");
	}
}

const char* SustainedImbalanceTrigger::Name() const
{
	return "SustainedImbalance";
}

bool SustainedImbalanceTrigger::ShouldRebalance(const LoadBalanceSnapshot& snapshot)
{
	// skip trigger evaluation during cooldown
	if (cooldownRemaining_ > 0)
	{
		cooldownRemaining_ -= 1;
		badStepCount_ = 0;
		return false;
	}

	// reset the streak when imbalance is within the threshold
	if (snapshot.relativeImbalance <= imbalanceThreshold_)
	{
		badStepCount_ = 0;
		return false;
	}

	badStepCount_ += 1;
	if (badStepCount_ < sustainSteps_) { return false; }

	// reset the streak and begin cooldown after triggering
	badStepCount_ = 0;
	cooldownRemaining_ = cooldownSteps_;
	return true;
}

std::unique_ptr<ILoadMetric> Dist::MakeLoadMetric(const Base::LoadBalanceMetric method)
{
	switch (method)
	{
	case Base::LoadBalanceMetric::Disabled:
		return nullptr;
	case Base::LoadBalanceMetric::RankOwnedParticleCount:
		return std::make_unique<RankOwnedParticleCountMetric>();
	case Base::LoadBalanceMetric::RankBackendStepTime:
		return std::make_unique<RankBackendStepTimeMetric>();
	}

	throw std::runtime_error("Unsupported load balance metric requested.");
}

std::unique_ptr<IRebalanceTrigger> Dist::MakeRebalanceTrigger(
	const Base::LoadBalanceTrigger method,
	const uint32_t intervalSteps,
	const double imbalanceThreshold,
	const uint32_t sustainSteps,
	const uint32_t cooldownSteps)
{
	switch (method)
	{
	case Base::LoadBalanceTrigger::Periodic:
		return std::make_unique<PeriodicTrigger>(intervalSteps);
	case Base::LoadBalanceTrigger::SustainedImbalance:
		return std::make_unique<SustainedImbalanceTrigger>(
			imbalanceThreshold,
			sustainSteps,
			cooldownSteps);
	}

	throw std::runtime_error("Unsupported load balance trigger requested.");
}

bool LoadBalanceMgr::Configure(
	const Sim::SimParams& params,
	const IDomainDecomp& domainDecomp,
	std::string* failMsg)
{
	if (failMsg) { failMsg->clear(); }

	// reset selected policies
	loadMetric_.reset();
	rebalanceTrigger_.reset();

	try
	{
		loadMetric_ = MakeLoadMetric(params.loadBalanceMetric);
		if (!loadMetric_) { return true; }

		// verify repartition support before creating the trigger
		if (!domainDecomp.SupportsRepartition())
		{
			if (failMsg)
			{
				*failMsg = "Selected domain decomposition does not support load balancing.";
			}
			loadMetric_.reset();
			return false;
		}

		// create trigger with the configured limits
		rebalanceTrigger_ = MakeRebalanceTrigger(
			params.loadBalanceTrigger,
			params.loadBalanceIntervalSteps,
			params.loadBalanceThreshold,
			params.loadBalanceSustainSteps,
			params.loadBalanceCooldownSteps);
	}
	catch (const std::exception& ex)
	{
		if (failMsg) { *failMsg = ex.what(); }
		loadMetric_.reset();
		rebalanceTrigger_.reset();
		return false;
	}

	return true;
}

bool LoadBalanceMgr::EvaluateLoadAndRebalance(
	const RankLoadSample& localSample,
	IDomainDecomp& domainDecomp,
	const MpiContext& mpiContext,
	std::string* failMsg)
{
	if (failMsg) { failMsg->clear(); }
	if (!Enabled()) { return true; }

	// evaluate and validate load of this rank
	bool localOk = rebalanceTrigger_ != nullptr && domainDecomp.SupportsRepartition();
	double localLoad = 0.0;
	if (localOk)
	{
		try
		{
			localLoad = loadMetric_->Evaluate(localSample);
			if (!std::isfinite(localLoad) || localLoad < 0.0)
			{
				throw std::runtime_error("Load balance metric produced an invalid rank load.");
			}
		}
		catch (const std::exception& ex)
		{
			if (failMsg) { *failMsg = ex.what(); }
			localOk = false;
		}
	}
	else if (failMsg)
	{
		*failMsg = "Load balance manager is not configured for the selected domain decomposition.";
	}
	if (!mpiContext.AllTrue(localOk)) { return false; }

	// gather loads in rank order
	std::vector<double> loadByRank{};
	mpiContext.AllGatherDouble(localLoad, loadByRank);

	// build trigger inputs from the gathered loads
	const double totalLoad = std::accumulate(loadByRank.begin(), loadByRank.end(), 0.0);
	const double meanLoad = totalLoad / static_cast<double>(loadByRank.size());
	const double maxLoad = *std::max_element(loadByRank.begin(), loadByRank.end());
	const double relativeImbalance = meanLoad > 0.0
		? maxLoad / meanLoad - 1.0
		: 0.0;

	const LoadBalanceSnapshot snapshot
	{
		.step = localSample.step,
		.loadByRank = loadByRank,
		.meanLoad = meanLoad,
		.maxLoad = maxLoad,
		.relativeImbalance = relativeImbalance
	};

	// evaluate the trigger on root and broadcast its decision
	int shouldRebalance = 0;
	if (mpiContext.IsRoot())
	{
		try
		{
			shouldRebalance = rebalanceTrigger_->ShouldRebalance(snapshot) ? 1 : 0;
		}
		catch (const std::exception& ex)
		{
			if (failMsg) { *failMsg = ex.what(); }
			localOk = false;
		}
	}
	if (!mpiContext.AllTrue(localOk)) { return false; }
	mpiContext.BroadcastInt(shouldRebalance, mpiContext.RootRank());
	if (shouldRebalance == 0) { return true; }

	// apply the same domain repartition input on every rank
	try
	{
		domainDecomp.Repartition({ .loadByRank = loadByRank });
		if (!domainDecomp.IsValidForRankCount(mpiContext.Size()))
		{
			throw std::runtime_error("Load balancing produced an invalid domain decomposition.");
		}
	}
	catch (const std::exception& ex)
	{
		if (failMsg) { *failMsg = ex.what(); }
		localOk = false;
	}

	// reduce the domain repartition result across ranks
	const bool globalOk = mpiContext.AllTrue(localOk);
	if (globalOk && mpiContext.IsRoot())
	{
		SPDLOG_INFO(
			"Repartitioned distributed domain at step {} using {} / {} "
			"(relative imbalance {:.3f}).",
			localSample.step,
			loadMetric_->Name(),
			rebalanceTrigger_->Name(),
			relativeImbalance);
	}
	return globalOk;
}
