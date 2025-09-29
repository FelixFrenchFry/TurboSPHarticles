// parameters_sim.cpp

// external
#include <cmath>
#include <spdlog/spdlog.h>
#include <thread>

// internal
#include "Source/Base/helpers.h"
#include "parameters_sim.h"



using namespace Sim;

std::pair<SimParams, int64_t> SimParamsMgr::SnapshotParams() const
{
	while (true)
	{
		// check for ongoing write
		int64_t before = version_.load(std::memory_order_acquire);
		if (before % 2 != 0) { std::this_thread::yield(); continue; }

		// copy parameters
		SimParams res = params_;

		// make sure no write happened during copy
		int64_t after = version_.load(std::memory_order_acquire);
		if (before == after) { return { res, after }; }

		// else try again
		std::this_thread::yield();
	}
}

int64_t SimParamsMgr::StableVersion() const
{
	while (true)
	{
		// make sure no write is ongoing
		int64_t a = version_.load(std::memory_order_acquire);
		if (a % 2 == 0)
		{
			int64_t b = version_.load(std::memory_order_acquire);
			if (a == b) { return a; }
		}

		// else try again
		std::this_thread::yield();
	}
}

void SimParamsMgr::Set_SimRunning(bool simRunning)
{
	// detect if parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(simRunning, params_.simRunning, changed);

	if (!changed) { return; } // nothing to change

	SPDLOG_INFO("Changed 'simRunning' from {} to {}",
		params_.simRunning, simRunning);

	// apply changes
	BeginWrite();
	params_.simRunning = simRunning;
	EndWrite();
}

void SimParamsMgr::Set_SimRunNextStep(bool simRunNextStep)
{
	// detect if parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(simRunNextStep, params_.simRunNextStep, changed);

	if (!changed) { return; } // nothing to change

	if (params_.simRunning) { return; } // sim already fully running

	SPDLOG_INFO("Changed 'simRunNextStep' from {} to {}",
		params_.simRunNextStep, simRunNextStep);

	// apply changes
	BeginWrite();
	params_.simRunNextStep = simRunNextStep;
	EndWrite();
}

void SimParamsMgr::Set_NumParticles(uint32_t N, Base::ParticleType type)
{
	// detect if parameter would actually change
	bool changed = false;

	if (type == Base::ParticleType::Fluid)
	{
		Base::MarkIfChanged(N, params_.N_F, changed);
	}
	else if (type == Base::ParticleType::Boundary)
	{
		Base::MarkIfChanged(N, params_.N_B, changed);
	}
	else
	{
		SPDLOG_INFO("Change of 'N' not implemented for this type");
	}

	if (!changed) { return; } // nothing to change

	BeginWrite();

	if (type == Base::ParticleType::Fluid)
	{
		SPDLOG_INFO("Changed 'N_F' from {} to {} "
			"(and dependent params accordingly)",
			params_.N_F, N);

		// apply changes
		params_.N_F = N;
	}
	else if (type == Base::ParticleType::Boundary)
	{
		SPDLOG_INFO("Changed 'N_B' from {} to {} "
			"(and dependent params accordingly)",
			params_.N_B, N);

		// apply changes
		params_.N_B = N;
	}

	EndWrite();
}

void SimParamsMgr::Set_TimeStep(float timeStep)
{
	if (!(timeStep > 0.0f))
	{
		SPDLOG_ERROR("Param change cancelled, 'timeStep' must be > 0");
		return;
	}

	// detect if parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(timeStep, params_.timeStep, changed);

	if (!changed) { return; } // nothing to change

	SPDLOG_INFO("Changed 'timeStep' from {} to {}",
		params_.timeStep, timeStep);

	// apply changes
	BeginWrite();
	params_.timeStep = timeStep;
	EndWrite();
}

void SimParamsMgr::Set_ParticleSize(float particleSize, Base::ParticleType type)
{
	if (!(particleSize > 0.0f))
	{
		SPDLOG_ERROR("Param change cancelled, 'particleSize' must be > 0");
		return;
	}

	// detect if parameter would actually change
	bool changed = false;

	if (type == Base::ParticleType::Fluid)
	{
		Base::MarkIfChanged(particleSize, params_.particleSizeF, changed);
	}
	else if (type == Base::ParticleType::Boundary)
	{
		Base::MarkIfChanged(particleSize, params_.particleSizeB, changed);
	}
	else
	{
		SPDLOG_INFO("Change of 'particleSize' not implemented for this type");
	}

	if (!changed) { return; } // nothing to change

	BeginWrite();

	if (type == Base::ParticleType::Fluid)
	{
		SPDLOG_INFO("Changed 'particleSizeF' from {} to {} "
					"(and dependent params accordingly)",
			params_.particleSizeF, particleSize);

		// apply changes
		params_.particleSizeF = particleSize;
		params_.particleRadiusF = 0.5f * particleSize;
		params_.massF = params_.restDensityF * particleSize
											 * particleSize
											 * particleSize;
	}
	else if (type == Base::ParticleType::Boundary)
	{
		SPDLOG_INFO("Changed 'particleSizeB' from {} to {} "
					"(and dependent params accordingly)",
			params_.particleSizeB, particleSize);

		// apply changes
		params_.particleSizeB = particleSize;
		params_.particleRadiusB = 0.5f * particleSize;
		params_.massB = params_.restDensityB * particleSize
											 * particleSize
											 * particleSize;
	}

	EndWrite();
}

void SimParamsMgr::Set_RestDensity(float restDensity, Base::ParticleType type)
{
	if (!(restDensity > 0.0f))
	{
		SPDLOG_ERROR("Param change cancelled, 'restDensity' must be > 0");
		return;
	}

	// detect if parameter would actually change
	bool changed = false;

	if (type == Base::ParticleType::Fluid)
	{
		Base::MarkIfChanged(restDensity, params_.restDensityF, changed);
	}
	else if (type == Base::ParticleType::Boundary)
	{
		Base::MarkIfChanged(restDensity, params_.restDensityB, changed);
	}
	else
	{
		SPDLOG_INFO("Change of 'restDensity' not implemented for this type");
	}

	if (!changed) { return; } // nothing to change

	BeginWrite();

	if (type == Base::ParticleType::Fluid)
	{
		SPDLOG_INFO("Changed 'restDensityF' from {} to {} "
					"(and dependent params accordingly)",
			params_.restDensityF, restDensity);

		// apply changes
		params_.restDensityF = restDensity;
		params_.massF = restDensity * params_.particleSizeF
									* params_.particleSizeF
									* params_.particleSizeF;
	}
	else if (type == Base::ParticleType::Boundary)
	{
		SPDLOG_INFO("Changed 'restDensityB' from {} to {} "
					"(and dependent params accordingly)",
			params_.restDensityB, restDensity);

		// apply changes
		params_.restDensityB = restDensity;
		params_.massB = restDensity * params_.particleSizeB
									* params_.particleSizeB
									* params_.particleSizeB;
	}

	EndWrite();
}

void SimParamsMgr::Set_Mass(float mass, Base::ParticleType type)
{
	if (!(mass > 0.0f))
	{
		SPDLOG_ERROR("Param change cancelled, 'mass' must be > 0");
		return;
	}

	// detect if parameter would actually change
	bool changed = false;

	if (type == Base::ParticleType::Fluid)
	{
		Base::MarkIfChanged(mass, params_.massF, changed);
	}
	else if (type == Base::ParticleType::Boundary)
	{
		Base::MarkIfChanged(mass, params_.massB, changed);
	}
	else
	{
		SPDLOG_INFO("Change of 'mass' not implemented for this type");
	}

	if (!changed) { return; } // nothing to change

	BeginWrite();

	if (type == Base::ParticleType::Fluid)
	{
		SPDLOG_INFO("Changed 'massF' from {} to {} "
			"(and dependent params accordingly)",
			params_.massF, mass);

		// apply changes
		params_.massF = mass;
		// TODO: adjust rest density too?
	}
	else if (type == Base::ParticleType::Boundary)
	{
		SPDLOG_INFO("Changed 'massB' from {} to {} "
			"(and dependent params accordingly)",
			params_.massB, mass);

		// apply changes
		params_.massB = mass;
		// TODO: adjust rest density too?
	}

	EndWrite();
}

void SimParamsMgr::Set_TaitExponent(float taitExponent)
{
	// detect if parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(taitExponent, params_.taitExponent, changed);

	if (!changed) { return; } // nothing to change

	SPDLOG_INFO("Changed 'taitExponent' from {} to {}",
		params_.taitExponent, taitExponent);

	// apply changes
	BeginWrite();
	params_.taitExponent = taitExponent;
	EndWrite();
}

void SimParamsMgr::Set_Stiffness(float stiffness)
{
	// detect if parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(stiffness, params_.stiffness, changed);

	if (!changed) { return; } // nothing to change

	SPDLOG_INFO("Changed 'stiffness' from {} to {}",
		params_.stiffness, stiffness);

	// apply changes
	BeginWrite();
	params_.stiffness = stiffness;
	EndWrite();
}

void SimParamsMgr::Set_Viscosity(float viscosity, Base::ParticleType type)
{
	// detect if parameter would actually change
	bool changed = false;

	if (type == Base::ParticleType::Fluid)
	{
		Base::MarkIfChanged(viscosity, params_.viscosityF, changed);
	}
	else if (type == Base::ParticleType::Boundary)
	{
		Base::MarkIfChanged(viscosity, params_.viscosityB, changed);
	}
	else
	{
		SPDLOG_INFO("Change of 'viscosity' not implemented for this type");
	}

	if (!changed) { return; } // nothing to change

	BeginWrite();

	if (type == Base::ParticleType::Fluid)
	{
		SPDLOG_INFO("Changed 'viscosityF' from {} to {}",
			params_.viscosityF, viscosity);

		// apply changes
		params_.viscosityF = viscosity;
	}
	else if (type == Base::ParticleType::Boundary)
	{
		SPDLOG_INFO("Changed 'viscosityB' from {} to {}",
			params_.viscosityB, viscosity);

		// apply changes
		params_.viscosityB = viscosity;
	}

	EndWrite();
}

void SimParamsMgr::Set_Gravity(glm::vec3 gravity)
{
	// detect if parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(gravity, params_.gravity, changed);

	if (!changed) { return; } // nothing to change

	SPDLOG_INFO("Changed 'gravity' from ({}, {}, {}) to ({}, {}, {})",
		params_.gravity.x, params_.gravity.y, params_.gravity.z,
		gravity.x, gravity.y, gravity.z);

	// apply changes
	BeginWrite();
	params_.gravity = gravity;
	EndWrite();
}

void SimParamsMgr::Set_KernelSuppRatio(float kernelSuppRatio)
{
	if (!(kernelSuppRatio >= 1.0f))
	{
		SPDLOG_ERROR("Param change cancelled, 'kernelSuppRatio' must be >= 1");
		return;
	}

	// detect if parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(kernelSuppRatio, params_.kernelSuppRatio, changed);

	if (!changed) { return; } // nothing to change

	SPDLOG_INFO("Changed 'kernelSuppRatio' from {} to {}",
		params_.kernelSuppRatio, kernelSuppRatio);

	// apply changes
	BeginWrite();
	params_.kernelSuppRatio = kernelSuppRatio;
	params_.kernelRadius = kernelSuppRatio * params_.particleSizeF;
	EndWrite();
}

void SimParamsMgr::Set_KernelRadius(float kernelRadius)
{
	if (!(kernelRadius > 0.0f))
	{
		SPDLOG_ERROR("Param change cancelled, 'kernelRadius' must be > 0");
		return;
	}

	// detect if parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(kernelRadius, params_.kernelRadius, changed);

	if (!changed) { return; } // nothing to change

	SPDLOG_INFO("Changed 'kernelRadius' from {} to {}",
		params_.kernelRadius, kernelRadius);

	// apply changes
	BeginWrite();
	params_.kernelRadius = kernelRadius;
	params_.kernelSuppRatio = kernelRadius / params_.particleSizeF;
	EndWrite();
}

void SimParamsMgr::Set_BoundaryLayers(int boundaryLayers)
{
	if (!(boundaryLayers >= 0))
	{
		SPDLOG_ERROR("Param change cancelled, 'boundaryLayers' must be >= 0");
		return;
	}

	// detect if parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(boundaryLayers, params_.boundaryLayers, changed);

	if (!changed) { return; } // nothing to change

	SPDLOG_INFO("Changed 'boundaryLayers' from {} to {}",
		params_.boundaryLayers, boundaryLayers);

	// apply changes
	BeginWrite();
	params_.boundaryLayers = boundaryLayers;
	EndWrite();
}

void SimParamsMgr::PresetParams_A()
{
	// TODO: differentiate between fluid and boundary particles?

	float timeStep			= 0.0001f;
	float particleSize		= 0.01f;
	float restDensity		= 1'000.0f;
	float taitExponent		= 7.0f;
	float stiffness			= 57000.0f;
	float viscosity			= 0.000003f;
	float gravity			= -9.81f;
	float kernelSuppRatio	= 2.0f;

	Set_SimRunning(false);
	Set_SimRunNextStep(false);
	Set_TimeStep(timeStep);
	Set_ParticleSize(particleSize, Base::ParticleType::Fluid);
	Set_ParticleSize(particleSize, Base::ParticleType::Boundary);
	Set_RestDensity(restDensity, Base::ParticleType::Fluid);
	Set_RestDensity(restDensity, Base::ParticleType::Boundary);
	Set_TaitExponent(taitExponent);
	Set_Stiffness(stiffness);
	Set_Viscosity(viscosity, Base::ParticleType::Fluid);
	Set_Viscosity(viscosity, Base::ParticleType::Boundary);
	Set_Gravity(glm::vec3{ 0.0f, gravity, 0.0f });
	Set_KernelSuppRatio(kernelSuppRatio);
}

void SimParamsMgr::PresetParams_B()
{
	// TODO: differentiate between fluid and boundary particles?

	float timeStep = 0.001f;
	float particleSize = 0.1f;
	float restDensity = 1'000.0f;
	float taitExponent = 1700.0f;
	float stiffness = 57000.0f;
	float viscosity = 0.3f;
	float gravity = -9.81f;
	float kernelSuppRatio = 2.0f;

	Set_SimRunning(false);
	Set_SimRunNextStep(false);
	Set_TimeStep(timeStep);
	Set_ParticleSize(particleSize, Base::ParticleType::Fluid);
	Set_ParticleSize(particleSize, Base::ParticleType::Boundary);
	Set_RestDensity(restDensity, Base::ParticleType::Fluid);
	Set_RestDensity(restDensity, Base::ParticleType::Boundary);
	Set_TaitExponent(taitExponent);
	Set_Stiffness(stiffness);
	Set_Viscosity(viscosity, Base::ParticleType::Fluid);
	Set_Viscosity(viscosity, Base::ParticleType::Boundary);
	Set_Gravity(glm::vec3{ 0.0f, gravity, 0.0f });
	Set_KernelSuppRatio(kernelSuppRatio);
}

void SimParamsMgr::BeginWrite()
{
	version_.fetch_add(1, std::memory_order_acq_rel);
}

void SimParamsMgr::EndWrite()
{
	version_.fetch_add(1, std::memory_order_release);
}
