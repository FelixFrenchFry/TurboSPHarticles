// dist_checkpoint.h
#pragma once

// external
#include <string>

// internal
#include "source/eval/serial/state_format.h"
#include "source/simulation/sync/state_sim.h"
#include "domain_decomp.h"



namespace Export
{
	class SerializationMgr;
}

namespace Dist
{
	class MpiContext;

	// builds this rank's startup profile from disjoint checkpoint position ranges
	void BuildCheckpointRepartitionProfile(
		const Export::SerializationMgr& serializationMgr,
		const std::string& path,
		const IDomainDecomp& domainDecomp,
		Base::LoadBalanceWeightModel weightModel,
		int rank,
		int N_ranks,
		DomainDecompRepartitionProfile& profile);

	// collective output of owned fluid/boundary particles and one copy of rigid state
	// closes the file on every rank before returning its header for publication
	Export::BinHeaderSim SaveDistributedCheckpointState(
		const Sync::UniversalSimState& state,
		const std::string& path,
		const MpiContext& mpiContext);
} // namespace Dist
