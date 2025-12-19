// parameters_sim.cpp

// external
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <thread>

// internal
#include "Source/Base/helpers.h"
#include "parameters_sim.h"



using namespace Sim;
using nlohmann::json;

std::pair<SimParams, uint64_t> SimParamsMgr::SnapshotParams() const
{
	while (true)
	{
		// check for ongoing write
		uint64_t before = version_.load(std::memory_order_acquire);
		if (before % 2 != 0) { std::this_thread::yield(); continue; }

		// copy parameters
		SimParams res = params_;

		// make sure no write happened during copy
		uint64_t after = version_.load(std::memory_order_acquire);
		if (before == after) { return { res, after }; }

		// else try again
		std::this_thread::yield();
	}
}

uint64_t SimParamsMgr::StableVersion() const
{
	while (true)
	{
		// make sure no write is ongoing
		uint64_t a = version_.load(std::memory_order_acquire);
		if (a % 2 == 0)
		{
			uint64_t b = version_.load(std::memory_order_acquire);
			if (a == b) { return a; }
		}

		// else try again
		std::this_thread::yield();
	}
}

bool SimParamsMgr::LoadSimParamsFromJsonFile(
	const std::string& fileName, bool verbose)
{
	namespace fs = std::filesystem;

	// use TSP_CONFIG_DIR from the build system as base path
	fs::path base(TSP_CONFIG_DIR);
	fs::path full = base / fileName;
	std::string path = full.string();

	std::ifstream in(path);

	if (!in)
	{
		SPDLOG_ERROR("Failed to open SimParams JSON file: {}",
			path.c_str());
		return false;
	}

	json jfile;
	try
	{
		in >> jfile;
	}
	catch (const std::exception& e)
	{
		SPDLOG_ERROR("Failed to parse JSON {}: {}",
			path.c_str(), e.what());
		return false;
	}

	// --- [ modes ] -----------------------------------------------------------

	// helpers for reading enums
	auto ParseBackendVersion = [](const json& v, Base::BackendVersion& out) -> bool
		{
			if (v.is_string())
			{
				const std::string s = v.get<std::string>();
				if (s == "CPU01") { out = Base::BackendVersion::CPU01; return true; }
				if (s == "CPU02") { out = Base::BackendVersion::CPU02; return true; }
				if (s == "GPU01") { out = Base::BackendVersion::GPU01; return true; }

				// TODO: add more

				return false;
			}
			else if (v.is_number_integer())
			{
				out = static_cast<Base::BackendVersion>(v.get<int>());

				return true;
			}

			return false;
		};

	auto ParseScenarioType = [](const json& v, Base::ScenarioType& out) -> bool
		{
			if (v.is_string())
			{
				const std::string s = v.get<std::string>();
				if (s == "BreakingDam") { out = Base::ScenarioType::BreakingDam; return true; }

				// TODO: add more

				return false;
			}
			else if (v.is_number_integer())
			{
				out = static_cast<Base::ScenarioType>(v.get<int>());

				return true;
			}

			return false;
		};

	// TODO: implement actual setters for backend version and scenario type?
	{
		BeginWrite();

		if (auto it = jfile.find("backendVersion"); it != jfile.end())
		{
			Base::BackendVersion bv;
			if (ParseBackendVersion(*it, bv))
			{
				params_.backendVersion = bv;
			}
			else
			{
				SPDLOG_WARN("Invalid 'backendVersion' in JSON");
			}
		}

		if (auto it = jfile.find("scenarioType"); it != jfile.end())
		{
			Base::ScenarioType st;
			if (ParseScenarioType(*it, st))
			{
				params_.scenarioType = st;
			}
			else
			{
				SPDLOG_WARN("Invalid 'scenarioType' in JSON");
			}
		}

		if (auto it = jfile.find("maxSteps"); it != jfile.end())
		{
			params_.maxSteps = it->get<int64_t>();
		}

		EndWrite();
	}

	// --- [ simulation state flags ] ------------------------------------------

	Set_SimRunning(false, verbose);
	Set_SimRunNextStep(false, verbose);

	// --- [ time step ] -------------------------------------------------------

	if (auto it = jfile.find("timeStep"); it != jfile.end())
	{
		Set_TimeStep(it->get<float>(), verbose);
	}

	// --- [ particle size ] ---------------------------------------------------

	if (auto it = jfile.find("particleSize"); it != jfile.end())
	{
		Set_ParticleSize(it->get<float>(),
			Base::ParticleType::Fluid, verbose);
		Set_ParticleSize(it->get<float>(),
			Base::ParticleType::Boundary, verbose);
	}

	// --- [ kernel support ratio ] --------------------------------------------

	if (auto it = jfile.find("kernelSuppRatio"); it != jfile.end())
	{
		Set_KernelSuppRatio(it->get<float>(), verbose);
	}

	// --- [ rest density ] ----------------------------------------------------

	if (auto it = jfile.find("restDensityF"); it != jfile.end())
	{
		Set_RestDensity(it->get<float>(),
			Base::ParticleType::Fluid, verbose);
	}

	if (auto it = jfile.find("restDensityB"); it != jfile.end())
	{
		Set_RestDensity(it->get<float>(),
			Base::ParticleType::Boundary, verbose);
	}

	// --- [ viscosity ] -------------------------------------------------------

	if (auto it = jfile.find("viscosityF"); it != jfile.end())
	{
		Set_Viscosity(it->get<float>(),
			Base::ParticleType::Fluid, verbose);
	}

	if (auto it = jfile.find("viscosityB"); it != jfile.end())
	{
		Set_Viscosity(it->get<float>(),
			Base::ParticleType::Boundary, verbose);
	}

	// --- [ pressure ] --------------------------------------------------------

	if (auto it = jfile.find("gammaPress"); it != jfile.end())
	{
		Set_GammaPress(it->get<float>(), verbose);
	}

	if (auto it = jfile.find("jacobiOmega"); it != jfile.end())
	{
		Set_JacobiOmega(it->get<float>(), verbose);
	}

	// TODO: implement setters for these
	{
		BeginWrite();

		if (auto it = jfile.find("densityDevTarget"); it != jfile.end())
		{
			params_.densityDevTarget = it->get<float>();
		}

		if (auto it = jfile.find("fixedIters"); it != jfile.end())
		{
			params_.fixedIters = it->get<uint32_t>();
		}

		if (auto it = jfile.find("minSolverIters"); it != jfile.end())
		{
			params_.minSolverIters = it->get<uint32_t>();
		}

		if (auto it = jfile.find("maxSolverIters"); it != jfile.end())
		{
			params_.maxSolverIters = it->get<uint32_t>();
		}

		if (auto it = jfile.find("solverTerminationCondition"); it != jfile.end())
		{
			using STC = Base::SolverTerminationCondition;

			const std::string s = it->get<std::string>();

			if (s == "FixedIters") { params_.solverTerminationCondition = STC::FixedIters; }
			else if (s == "RelDensityError_Raw") { params_.solverTerminationCondition = STC::RelDensityError_Raw; }
			else if (s == "RelDensityError_Pos") { params_.solverTerminationCondition = STC::RelDensityError_Pos; }
			else if (s == "RelDensityError_Abs") { params_.solverTerminationCondition = STC::RelDensityError_Abs; }
			else
			{
				SPDLOG_WARN("Invalid 'solverTerminationCondition' string in JSON: {}", s);
			}
		}

		EndWrite();
	}

	// --- [ gravity ] ---------------------------------------------------------

	if (auto it = jfile.find("gravity"); it != jfile.end() &&
		it->is_array() && it->size() == 3)
	{
		glm::vec3 g;
		g.x = (*it)[0].get<float>();
		g.y = (*it)[1].get<float>();
		g.z = (*it)[2].get<float>();
		Set_Gravity(g, verbose);
	}

	// --- [ jitter ] ----------------------------------------------------------

	// TODO: implement setters for jitterFrac/jitterSeed stuff?
	{
		BeginWrite();

		if (auto it = jfile.find("jitterFracF"); it != jfile.end())
		{
			params_.jitterFracF = it->get<float>();
		}

		if (auto it = jfile.find("jitterSeedF"); it != jfile.end())
		{
			params_.jitterSeedF = it->get<int64_t>();
		}

		if (auto it = jfile.find("jitterFracB"); it != jfile.end())
		{
			params_.jitterFracB = it->get<float>();
		}

		if (auto it = jfile.find("jitterSeedB"); it != jfile.end())
		{
			params_.jitterSeedB = it->get<int64_t>();
		}

		EndWrite();
	}

	// --- [ boundary layers ] -------------------------------------------------

	if (auto it = jfile.find("boundaryLayers"); it != jfile.end())
	{
		Set_BoundaryLayers(it->get<uint32_t>(), verbose);
	}

	// --- [ statistics, benchmarks, and exports ] -----------------------------

	// TODO: implement setters for maxStatRecs/maxBenchRecs?
	{
		BeginWrite();

		if (auto it = jfile.find("maxStatRecs"); it != jfile.end())
		{
			params_.maxStatRecs = it->get<uint32_t>();
		}

		if (auto it = jfile.find("maxBenchRecs"); it != jfile.end())
		{
			params_.maxBenchRecs = it->get<uint32_t>();
		}

		if (auto it = jfile.find("exportEnabled"); it != jfile.end())
		{
			params_.exportEnabled = it->get<bool>();
		}

		if (auto it = jfile.find("exportDir"); it != jfile.end())
		{
			params_.exportDir = it->get<std::string>();
		}

		if (auto it = jfile.find("exportInterval"); it != jfile.end())
		{
			params_.exportInterval = it->get<uint32_t>();
		}

		EndWrite();
	}

	if (!AssertParamDerivationCorrectness())
	{
		SPDLOG_ERROR("Parameter derivation is incorrect");
		return false;
	}

	return true;
}

void SimParamsMgr::Set_SimRunning(
	bool simRunning, bool verbose)
{
	// detect if parameter would actually change
	bool changed = false;
	Base::MarkIfChanged(simRunning, params_.simRunning, changed);
	if (!changed) { return; } // nothing to change

	if (verbose)
	{
		SPDLOG_INFO("Changed 'simRunning' from {} to {}",
			params_.simRunning, simRunning);
	}

	// apply changes
	BeginWrite();
	params_.simRunning = simRunning;
	EndWrite();
}

void SimParamsMgr::Set_SimRunNextStep(
	bool simRunNextStep, bool verbose)
{
	// detect if parameter would actually change
	bool changed = false;
	Base::MarkIfChanged(simRunNextStep, params_.simRunNextStep, changed);
	if (!changed) { return; } // nothing to change

	if (params_.simRunning) { return; } // sim already fully running

	if (verbose)
	{
		SPDLOG_INFO("Changed 'simRunNextStep' from {} to {}",
			params_.simRunNextStep, simRunNextStep);
	}

	// apply changes
	BeginWrite();
	params_.simRunNextStep = simRunNextStep;
	EndWrite();
}

void SimParamsMgr::Set_NumParticles(
	uint32_t N, Base::ParticleType type, bool verbose)
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
		SPDLOG_WARN("Change of 'N' not implemented for this type");
	}

	if (!changed) { return; } // nothing to change

	BeginWrite();

	if (type == Base::ParticleType::Fluid)
	{
		if (verbose)
		{
			SPDLOG_INFO("Changed 'N_F' from {} to {}",
				params_.N_F, N);
		}

		// apply changes
		params_.N_F = N;
	}
	else if (type == Base::ParticleType::Boundary)
	{
		if (verbose)
		{
			SPDLOG_INFO("Changed 'N_B' from {} to {}",
				params_.N_B, N);
		}

		// apply changes
		params_.N_B = N;
	}

	EndWrite();
}

void SimParamsMgr::Set_TimeStep(
	float timeStep, bool verbose)
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

	if (verbose)
	{
		SPDLOG_INFO("Changed 'timeStep' from {} to {}",
			params_.timeStep, timeStep);
	}

	// apply changes
	BeginWrite();
	params_.timeStep = timeStep;
	EndWrite();
}

void SimParamsMgr::Set_ParticleSize(
	float particleSize, Base::ParticleType type, bool verbose)
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
		SPDLOG_WARN("Change of 'particleSize' not implemented for this type");
	}

	if (!changed) { return; } // nothing to change

	BeginWrite();

	if (type == Base::ParticleType::Fluid)
	{
		if (verbose)
		{
			SPDLOG_INFO("Changed 'particleSizeF' from {} to {} "
						"(and dependent params accordingly)",
				params_.particleSizeF, particleSize);
		}

		// apply changes
		params_.particleSizeF = particleSize;
		params_.particleSizeInvF = 1.0f / particleSize;
		params_.particleRadiusF = 0.5f * particleSize;
		params_.massF = params_.restDensityF * particleSize
											 * particleSize
											 * particleSize;
	}
	else if (type == Base::ParticleType::Boundary)
	{
		if (verbose)
		{
			SPDLOG_INFO("Changed 'particleSizeB' from {} to {} "
						"(and dependent params accordingly)",
				params_.particleSizeB, particleSize);
		}

		// apply changes
		params_.particleSizeB = particleSize;
		params_.particleSizeInvB = 1.0f / particleSize;
		params_.particleRadiusB = 0.5f * particleSize;
		params_.massB = params_.restDensityB * particleSize
											 * particleSize
											 * particleSize;
	}

	EndWrite();
}

void SimParamsMgr::Set_KernelSuppRatio(
	float kernelSuppRatio, bool verbose)
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

	if (verbose)
	{
		SPDLOG_INFO("Changed 'kernelSuppRatio' from {} to {}",
			params_.kernelSuppRatio, kernelSuppRatio);
	}

	// apply changes
	BeginWrite();
	params_.kernelSuppRatio = kernelSuppRatio;
	params_.kernelRadius = kernelSuppRatio * params_.particleSizeF;
	params_.kernelRadiusSqu = params_.kernelRadius * params_.kernelRadius;
	EndWrite();
}

void SimParamsMgr::Set_KernelRadius(
	float kernelRadius, bool verbose)
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

	if (verbose)
	{
		SPDLOG_INFO("Changed 'kernelRadius' from {} to {}",
			params_.kernelRadius, kernelRadius);
	}

	// apply changes
	BeginWrite();
	params_.kernelRadius = kernelRadius;
	params_.kernelRadiusSqu = kernelRadius * kernelRadius;
	params_.kernelSuppRatio = kernelRadius / params_.particleSizeF;
	EndWrite();
}

void SimParamsMgr::Set_RestDensity(
	float restDensity, Base::ParticleType type, bool verbose)
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
		SPDLOG_WARN("Change of 'restDensity' not implemented for this type");
	}

	if (!changed) { return; } // nothing to change

	BeginWrite();

	if (type == Base::ParticleType::Fluid)
	{
		if (verbose)
		{
			SPDLOG_INFO("Changed 'restDensityF' from {} to {} "
						"(and dependent params accordingly)",
				params_.restDensityF, restDensity);
		}

		// apply changes
		params_.restDensityF = restDensity;
		params_.massF = restDensity * params_.particleSizeF
									* params_.particleSizeF
									* params_.particleSizeF;
	}
	else if (type == Base::ParticleType::Boundary)
	{
		if (verbose)
		{
			SPDLOG_INFO("Changed 'restDensityB' from {} to {} "
						"(and dependent params accordingly)",
				params_.restDensityB, restDensity);
		}

		// apply changes
		params_.restDensityB = restDensity;
		params_.massB = restDensity * params_.particleSizeB
									* params_.particleSizeB
									* params_.particleSizeB;
	}

	EndWrite();
}

void SimParamsMgr::Set_Mass(
	float mass, Base::ParticleType type, bool verbose)
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
		SPDLOG_WARN("Change of 'mass' not implemented for this type");
	}

	if (!changed) { return; } // nothing to change

	BeginWrite();

	if (type == Base::ParticleType::Fluid)
	{
		if (verbose)
		{
			SPDLOG_INFO("Changed 'massF' from {} to {} "
						"(and dependent params accordingly)",
				params_.massF, mass);
		}

		// apply changes
		params_.massF = mass;
		params_.restDensityF = mass / (params_.particleSizeF 
									* params_.particleSizeF
									* params_.particleSizeF);
	}
	else if (type == Base::ParticleType::Boundary)
	{
		if (verbose)
		{
			SPDLOG_INFO("Changed 'massB' from {} to {} "
						"(and dependent params accordingly)",
				params_.massB, mass);
		}

		// apply changes
		params_.massB = mass;
		params_.restDensityB = mass / (params_.particleSizeB
									* params_.particleSizeB
									* params_.particleSizeB);
	}

	EndWrite();
}

void SimParamsMgr::Set_Viscosity(
	float viscosity, Base::ParticleType type, bool verbose)
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
		SPDLOG_WARN("Change of 'viscosity' not implemented for this type");
	}

	if (!changed) { return; } // nothing to change

	BeginWrite();

	if (type == Base::ParticleType::Fluid)
	{
		if (verbose)
		{
			SPDLOG_INFO("Changed 'viscosityF' from {} to {}",
				params_.viscosityF, viscosity);
		}

		// apply changes
		params_.viscosityF = viscosity;
	}
	else if (type == Base::ParticleType::Boundary)
	{
		if (verbose)
		{
			SPDLOG_INFO("Changed 'viscosityB' from {} to {}",
				params_.viscosityB, viscosity);
		}

		// apply changes
		params_.viscosityB = viscosity;
	}

	EndWrite();
}

void SimParamsMgr::Set_GammaPress(
	float gammaPress, bool verbose)
{
	// detect if parameter would actually change
	bool changed = false;
	Base::MarkIfChanged(gammaPress, params_.gammaPress, changed);
	if (!changed) { return; } // nothing to change

	if (verbose)
	{
		SPDLOG_INFO("Changed 'gammaPress' from {} to {}",
			params_.gammaPress, gammaPress);
	}

	// apply changes
	BeginWrite();
	params_.gammaPress = gammaPress;
	EndWrite();
}

void SimParamsMgr::Set_JacobiOmega(
	float jacobiOmega, bool verbose)
{
	// detect if parameter would actually change
	bool changed = false;
	Base::MarkIfChanged(jacobiOmega, params_.jacobiOmega, changed);
	if (!changed) { return; } // nothing to change

	if (verbose)
	{
		SPDLOG_INFO("Changed 'jacobiOmega' from {} to {}",
			params_.jacobiOmega, jacobiOmega);
	}

	// apply changes
	BeginWrite();
	params_.jacobiOmega = jacobiOmega;
	EndWrite();
}

void SimParamsMgr::Set_DensityDevTarget(
	float densityDevTarget, bool verbose)
{
	// detect if parameter would actually change
	bool changed = false;
	Base::MarkIfChanged(densityDevTarget, params_.densityDevTarget, changed);
	if (!changed) { return; } // nothing to change

	if (verbose)
	{
		SPDLOG_INFO("Changed 'densityDevTarget' from {} to {}",
			params_.densityDevTarget, densityDevTarget);
	}

	// apply changes
	BeginWrite();
	params_.densityDevTarget = densityDevTarget;
	EndWrite();
}

void SimParamsMgr::Set_MinSolverIters(
	uint32_t minSolverIters, bool verbose)
{
	if (minSolverIters <= 0)
	{
		SPDLOG_ERROR("Param change cancelled, 'minSolverIters' must be > 0");
		return;
	}

	// detect if parameter would actually change
	bool changed = false;
	Base::MarkIfChanged(minSolverIters, params_.minSolverIters, changed);
	if (!changed) { return; } // nothing to change

	if (verbose)
	{
		SPDLOG_INFO("Changed 'minSolverIters' from {} to {}",
			params_.minSolverIters, minSolverIters);
	}

	// apply changes
	BeginWrite();
	params_.minSolverIters = minSolverIters;
	EndWrite();
}

void SimParamsMgr::Set_MaxSolverIters(
	uint32_t maxSolverIters, bool verbose)
{
	if (maxSolverIters <= 0)
	{
		SPDLOG_ERROR("Param change cancelled, 'maxSolverIters' must be > 0");
		return;
	}

	// detect if parameter would actually change
	bool changed = false;
	Base::MarkIfChanged(maxSolverIters, params_.maxSolverIters, changed);
	if (!changed) { return; } // nothing to change

	if (verbose)
	{
		SPDLOG_INFO("Changed 'maxSolverIters' from {} to {}",
			params_.maxSolverIters, maxSolverIters);
	}

	// apply changes
	BeginWrite();
	params_.maxSolverIters = maxSolverIters;
	EndWrite();
}

void SimParamsMgr::Set_Gravity(
	glm::vec3 gravity, bool verbose)
{
	// detect if parameter would actually change
	bool changed = false;
	Base::MarkIfChanged(gravity, params_.gravity, changed);
	if (!changed) { return; } // nothing to change

	if (verbose)
	{
		SPDLOG_INFO("Changed 'gravity' from ({}, {}, {}) to ({}, {}, {})",
			params_.gravity.x, params_.gravity.y, params_.gravity.z,
			gravity.x, gravity.y, gravity.z);
	}

	// apply changes
	BeginWrite();
	params_.gravity = gravity;
	EndWrite();
}

void SimParamsMgr::Set_BoundaryLayers(
	uint32_t boundaryLayers, bool verbose)
{
	// detect if parameter would actually change
	bool changed = false;
	Base::MarkIfChanged(boundaryLayers, params_.boundaryLayers, changed);
	if (!changed) { return; } // nothing to change

	if (verbose)
	{
		SPDLOG_INFO("Changed 'boundaryLayers' from {} to {}",
			params_.boundaryLayers, boundaryLayers);
	}

	// apply changes
	BeginWrite();
	params_.boundaryLayers = boundaryLayers;
	EndWrite();
}

bool SimParamsMgr::AssertParamDerivationCorrectness() const
{
	SimParams par = SnapshotParams().first;

	auto Check = [&](std::string_view msg, bool fail, auto&& pred) -> bool
	{
		const bool pass = static_cast<bool>(pred());

		if (!pass)
		{
			if (fail == true)
			{
				SPDLOG_ERROR("[SimParams] {}", msg);
			#ifndef NDEBUG
				assert(false && "SimParam invariant failed");
			#endif
			}
			else
			{
				SPDLOG_WARN("[SimParams] {}", msg);
			}
		}

		return pass;
	};

	// --- [ hard invariants ] -------------------------------------------------

	bool ok = true;

	// non-zero positive values
	ok &= Check("timeStep <= 0", true, [&]()
	{
		return par.timeStep > 0.0f;
	});
	ok &= Check("particleSizeF <= 0", true, [&]()
	{
		return par.particleSizeF > 0.0f;
	});
	ok &= Check("particleSizeB <= 0", true, [&]()
	{
		return par.particleSizeB > 0.0f;
	});
	ok &= Check("particleSizeInvF <= 0", true, [&]()
	{
		return par.particleSizeInvF > 0.0f;
	});
	ok &= Check("particleSizeInvB <= 0", true, [&]()
	{
		return par.particleSizeInvB > 0.0f;
	});
	ok &= Check("particleRadiusF <= 0", true, [&]()
	{
		return par.particleRadiusF > 0.0f;
	});
	ok &= Check("particleRadiusB <= 0", true, [&]()
	{
		return par.particleRadiusB > 0.0f;
	});
	ok &= Check("restDensityF <= 0", true, [&]()
	{
		return par.restDensityF > 0.0f;
	});
	ok &= Check("restDensityB <= 0", true, [&]()
	{
		return par.restDensityB > 0.0f;
	});
	ok &= Check("massF <= 0", true, [&]()
	{
		return par.massF > 0.0f;
	});
	ok &= Check("massB <= 0", true, [&]()
	{
		return par.massB > 0.0f;
	});
	ok &= Check("gammaPress < 0", true, [&]()
	{
		return par.gammaPress >= 0.0f;
	});
	ok &= Check("jacobiOmega < 0", true, [&]()
	{
		return par.jacobiOmega >= 0.0f;
	});
	ok &= Check("viscosityF < 0", true, [&]()
	{
		return par.viscosityF >= 0.0f;
	});
	ok &= Check("viscosityB < 0", true, [&]()
	{
		return par.viscosityB >= 0.0f;
	});

	// relationships
	ok &= Check("particleSizeInvF != 1 / particleSizeF", true, [&]()
	{
		float expect = 1.0f / par.particleSizeF;
		return Base::ApproxEqu(par.particleSizeInvF, expect);
	});
	ok &= Check("particleSizeInvB != 1 / particleSizeB", true, [&]()
	{
		float expect = 1.0f / par.particleSizeB;
		return Base::ApproxEqu(par.particleSizeInvB, expect);
	});
	ok &= Check("particleRadiusF != 0.5 * particleSizeF", true, [&]()
	{
		float expect = 0.5f * par.particleSizeF;
		return Base::ApproxEqu(par.particleRadiusF, expect);
	});
	ok &= Check("particleRadiusB != 0.5 * particleSizeB", true, [&]()
	{
		float expect = 0.5f * par.particleSizeB;
		return Base::ApproxEqu(par.particleRadiusB, expect);
	});
	ok &= Check("massF != restDensityF * particleSizeF^3", true, [&]()
	{
		float expect = par.restDensityF * par.particleSizeF
										* par.particleSizeF
										* par.particleSizeF;
		return Base::ApproxEqu(par.massF, expect);
	});
	ok &= Check("massB != restDensityB * particleSizeB^3", true, [&]()
	{
		float expect = par.restDensityB * par.particleSizeB
										* par.particleSizeB
										* par.particleSizeB;
		return Base::ApproxEqu(par.massB, expect);
	});
	ok &= Check("kernelRadius != kernelSuppRatio * particleSizeF", true, [&]()
	{
		float expect = par.kernelSuppRatio * par.particleSizeF;
		return Base::ApproxEqu(par.kernelRadius, expect);
	});
	ok &= Check("kernelSuppRatio < 1", true, [&]()
	{
		return par.kernelSuppRatio >= 1.0f;
	});

	return ok;
}

inline void SimParamsMgr::BeginWrite()
{
	version_.fetch_add(1, std::memory_order_acq_rel);
}

inline void SimParamsMgr::EndWrite()
{
	version_.fetch_add(1, std::memory_order_release);
}
