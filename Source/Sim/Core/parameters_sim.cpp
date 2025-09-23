// parameters_sim.cpp

// external
#include <spdlog/spdlog.h>

// internal
#include "Source/Base/helpers.h"
#include "parameters_sim.h"



using namespace Sim;

std::pair<SimParams, int64_t> SimParamsMgr::SnapshotParams() const
{
	while (true)
	{
		// check for ongoing write
		int64_t before = version_.load();
		if (before % 2 != 0) { continue; }

		// copy parameters
		SimParams res = params_;

		// make sure no write happened during copy
		int64_t after = version_.load();
		if (before == after) { return { res, after }; }
	}
}

int64_t SimParamsMgr::StableVersion() const
{
	while (true)
	{
		// make sure no write is ongoing
		uint64_t a = version_.load();
		if (a % 2 == 0)
		{
			uint64_t b = version_.load();
			if (a == b) { return a; }
		}
	}
}

/*
void SimParamsMgr::SetParam(TimeStepPatch patch)
{
	// detect if any parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(patch.timeStep, params_.time.timeStep, changed);
	Base::MarkIfChanged(patch.timeStepInv, params_.time.timeStepInv, changed);
	Base::MarkIfChanged(patch.timeStepSqu, params_.time.timeStepSqu, changed);

	if (!changed) { return; } // nothing to change

	// apply changes
	BeginWrite();
	if (patch.timeStep) { params_.time.timeStep = *patch.timeStep; }
	if (patch.timeStepInv) { params_.time.timeStepInv = *patch.timeStepInv; }
	if (patch.timeStepSqu) { params_.time.timeStepSqu = *patch.timeStepSqu; }
	EndWrite();
}
*/

void SimParamsMgr::Set_TimeStep(float newTimeStep)
{
	SPDLOG_INFO("Changed 'timeStep' from {} to {}",
		params_.timeStep, newTimeStep);

	BeginWrite();
	params_.timeStep = newTimeStep;
	params_.timeStepInv = (newTimeStep != 0.0f) ? 1.0f / newTimeStep : 0.0f;
	params_.timeStepSqu = newTimeStep * newTimeStep;
	EndWrite();
}


void SimParamsMgr::Set_ParticleSize(float newParticleSize)
{
	SPDLOG_INFO("Changed 'particleSize' from {} to {}",
		params_.particleSize, newParticleSize);

	BeginWrite();
	params_.particleSize = newParticleSize;
	EndWrite();
}


void SimParamsMgr::Set_MassFluid(float newMassFluid)
{
	SPDLOG_INFO("Changed 'massFluid' from {} to {}",
		params_.massFluid, newMassFluid);

	BeginWrite();
	params_.massFluid = newMassFluid;
	EndWrite();
}


void SimParamsMgr::Set_MassBound(float newMassBound)
{
	SPDLOG_INFO("Changed 'massBound' from {} to {}",
		params_.massBound, newMassBound);

	BeginWrite();
	params_.massBound = newMassBound;
	EndWrite();
}


void SimParamsMgr::Set_KernelRadius(float newKernelRadius)
{
	SPDLOG_INFO("Changed 'kernelRadius' from {} to {}",
		params_.kernelRadius, newKernelRadius);

	BeginWrite();
	params_.kernelRadius = newKernelRadius;
	params_.kernelRadiusSqu = newKernelRadius * newKernelRadius;
	EndWrite();
}


void SimParamsMgr::Set_RestDensity(float newRestDensity)
{
	SPDLOG_INFO("Changed 'restDensity' from {} to {}",
		params_.restDensity, newRestDensity);

	BeginWrite();
	params_.restDensity = newRestDensity;
	params_.restDensityInv = (newRestDensity != 0.0f) ? 1.0f / newRestDensity : 0.0f;
	params_.restDensitySqu = newRestDensity * newRestDensity;
	EndWrite();
}


void SimParamsMgr::Set_ViscosityFluid(float newViscosityFluid)
{
	SPDLOG_INFO("Changed 'viscosityFluid' from {} to {}",
		params_.viscosityFluid, newViscosityFluid);

	BeginWrite();
	params_.viscosityFluid = newViscosityFluid;
	EndWrite();
}


void SimParamsMgr::Set_ViscosityBound(float newViscosityBound)
{
	SPDLOG_INFO("Changed 'viscosityBound' from {} to {}",
		params_.viscosityBound, newViscosityBound);

	BeginWrite();
	params_.viscosityBound = newViscosityBound;
	EndWrite();
}


void SimParamsMgr::BeginWrite()
{
	version_.fetch_add(1);
}


void SimParamsMgr::EndWrite()
{
	version_.fetch_add(1);
}
