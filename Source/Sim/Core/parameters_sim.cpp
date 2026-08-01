// parameters_sim.cpp

// external
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <spdlog/spdlog.h>
#include <thread>

// internal
#include "Source/Base/helpers.h"
#include "parameters_sim.h"



constexpr float PI = std::numbers::pi_v<float>;
using namespace Sim;

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

bool SimParamsMgr::LoadSimParamsFromJson(
	const nlohmann::json& j, bool verbose)
{
	// --- [ modes ] ---------------------------------------------------------------------------------

	// helpers for reading enums
	auto ParseBackendVersion = [](const nlohmann::json& v, Base::BackendVersion& out) -> bool
		{
			if (v.is_string())
			{
				const std::string s = v.get<std::string>();
				if (s == "CPU01") { out = Base::BackendVersion::CPU01; return true; }
				if (s == "GPU01") { out = Base::BackendVersion::GPU01; return true; }
				return false;
			}
			else if (v.is_number_integer())
			{
				const int i = v.get<int>();
				switch (i)
				{
				case 0:
					out = Base::BackendVersion::CPU01;
					return true;
				case 1:
					out = Base::BackendVersion::GPU01;
					return true;
				default:
					return false;
				}
			}

			return false;
		};

	auto ParseScenarioType = [](const nlohmann::json& v, Base::ScenarioType& out) -> bool
		{
			if (v.is_string())
			{
				const std::string s = v.get<std::string>();
				if (s == "BreakingDam") { out = Base::ScenarioType::BreakingDam; return true; }
				if (s == "ConvergingDam") { out = Base::ScenarioType::ConvergingDam; return true; }
				if (s == "BigDroplet") { out = Base::ScenarioType::BigDroplet; return true; }
				if (s == "RigidCube") { out = Base::ScenarioType::RigidCube; return true; }
				if (s == "SampledMesh") { out = Base::ScenarioType::SampledMesh; return true; }
				return false;
			}
			else if (v.is_number_integer())
			{
				out = static_cast<Base::ScenarioType>(v.get<int>());
				return true;
			}

			return false;
		};

	auto ParseNeighborSearchMethod = [](const nlohmann::json& v, Base::NeighborSearchMethod& out) -> bool
		{
			if (v.is_string())
			{
				const std::string s = v.get<std::string>();
				if (s == "BruteForce")
					{ out = Base::NeighborSearchMethod::BruteForce; return true; }
				if (s == "CellLinkedList_Compact")
					{ out = Base::NeighborSearchMethod::CellLinkedList_Compact; return true; }
				return false;
			}
			else if (v.is_number_integer())
			{
				const int i = v.get<int>();
				switch (i)
				{
				case 0:
					out = Base::NeighborSearchMethod::BruteForce;
					return true;
				case 1:
					out = Base::NeighborSearchMethod::CellLinkedList_Compact;
					return true;
				default:
					return false;
				}
			}
			return false;
		};

	auto ParseNeighborListMethod = [](const nlohmann::json& v, Base::NeighborListMethod& out) -> bool
		{
			if (v.is_string())
			{
				const std::string s = v.get<std::string>();
				if (s == "ChunkedSELLX_Explicit")
					{ out = Base::NeighborListMethod::ChunkedSELLX_Explicit; return true; }
				if (s == "ChunkedSELLX_AdaptCompress")
					{ out = Base::NeighborListMethod::ChunkedSELLX_AdaptCompress; return true; }
				return false;
			}
			else if (v.is_number_integer())
			{
				const int i = v.get<int>();
				switch (i)
				{
				case 0:
					out = Base::NeighborListMethod::ChunkedSELLX_Explicit;
					return true;
				case 1:
					out = Base::NeighborListMethod::ChunkedSELLX_AdaptCompress;
					return true;
				default:
					return false;
				}
			}
			return false;
		};

	auto ParseDomainDecompMethod = [](const nlohmann::json& v, Base::DomainDecompMethod& out) -> bool
		{
			if (v.is_string())
			{
				const std::string s = v.get<std::string>();
				if (s == "AABB_Uniform")
					{ out = Base::DomainDecompMethod::AABB_Uniform; return true; }
				return false;
			}
			else if (v.is_number_integer())
			{
				const int i = v.get<int>();
				switch (i)
				{
				case 0:
					out = Base::DomainDecompMethod::AABB_Uniform;
					return true;
				default:
					return false;
				}
			}
			return false;
		};

	// TODO: implement actual setters for backend version and scenario type?
	{
		BeginWrite();

		if (auto it = j.find("backendVersion"); it != j.end())
		{
			Base::BackendVersion bv;
			if (ParseBackendVersion(*it, bv))
			{
				params_.backendVersion = bv;
			}
			else
			{
				SPDLOG_WARN("Invalid 'backendVersion' value in JSON.");
				EndWrite();
				return false;
			}
		}

		if (auto it = j.find("scenarioType"); it != j.end())
		{
			Base::ScenarioType st;
			if (ParseScenarioType(*it, st))
			{
				params_.scenarioType = st;
			}
			else
			{
				SPDLOG_WARN("Invalid 'scenarioType' value in JSON.");
			}
		}

		if (auto it = j.find("neighborSearchMethod"); it != j.end())
		{
			Base::NeighborSearchMethod nsm;
			if (ParseNeighborSearchMethod(*it, nsm))
			{
				params_.neighborSearchMethod = nsm;
			}
			else
			{
				SPDLOG_WARN("Invalid 'neighborSearchMethod' value in JSON.");
				EndWrite();
				return false;
			}
		}

		if (auto it = j.find("neighborListStorage"); it != j.end())
		{
			Base::NeighborListMethod nls;
			if (ParseNeighborListMethod(*it, nls))
			{
				params_.neighborListMethod = nls;
			}
			else
			{
				SPDLOG_WARN("Invalid 'neighborListStorage' value in JSON.");
				EndWrite();
				return false;
			}
		}

		if (auto it = j.find("domainDecompMethod"); it != j.end())
		{
			Base::DomainDecompMethod ddm;
			if (ParseDomainDecompMethod(*it, ddm))
			{
				params_.domainDecompMethod = ddm;
			}
			else
			{
				SPDLOG_WARN("Invalid 'domainDecompMethod' value in JSON.");
				EndWrite();
				return false;
			}
		}

		if (auto it = j.find("maxSteps"); it != j.end())
		{
			params_.maxSteps = it->get<int64_t>();
		}

		if (auto it = j.find("maxThreadsSim"); it != j.end())
		{
			if (!it->is_number_integer())
			{
				SPDLOG_WARN("Invalid 'maxThreadsSim' value in JSON (must be -1 or > 0).");
			}
			else
			{
				const int64_t maxThreadsSim = it->get<int64_t>();
				if (maxThreadsSim == -1 || maxThreadsSim > 0)
				{
					params_.maxThreadsSim = maxThreadsSim;
				}
				else if (maxThreadsSim == 0)
				{
					SPDLOG_WARN("Invalid 'maxThreadsSim' value in JSON (0 is not allowed).");
				}
				else
				{
					SPDLOG_WARN("Invalid 'maxThreadsSim' value in JSON (must be -1 or > 0).");
				}
			}
		}

		EndWrite();
	}

	// --- [ simulation state flags ] ----------------------------------------------------------------

	Set_SimRunning(false, verbose);
	Set_SimRunNextStep(false, verbose);

	// --- [ time step ] -----------------------------------------------------------------------------

	if (auto it = j.find("timeStep"); it != j.end())
	{
		Set_TimeStep(it->get<float>(), verbose);
	}

	// --- [ particle size ] -------------------------------------------------------------------------

	if (auto it = j.find("particleSize"); it != j.end())
	{
		Set_ParticleSize(it->get<float>(),
			Base::ParticleType::Fluid, verbose);
		Set_ParticleSize(it->get<float>(),
			Base::ParticleType::Boundary, verbose);
	}

	// --- [ kernel support ratio ] ------------------------------------------------------------------

	if (auto it = j.find("kernelSuppRatio"); it != j.end())
	{
		Set_KernelSuppRatio(it->get<float>(), verbose);
	}

	// --- [ neighbor count heuristics ] -------------------------------------------------------------

	BeginWrite();
	UpdateNeighborCountHeuristics_();
	EndWrite();

	// --- [ rest density ] --------------------------------------------------------------------------

	if (auto it = j.find("restDensity"); it != j.end())
	{
		Set_RestDensity(it->get<float>(), verbose);
	}

	// --- [ non-pressure accelerations ] ------------------------------------------------------------

	if (auto it = j.find("viscosityF"); it != j.end())
	{
		Set_Viscosity(it->get<float>(),
			Base::ParticleType::Fluid, verbose);
	}

	if (auto it = j.find("viscosityB"); it != j.end())
	{
		Set_Viscosity(it->get<float>(),
			Base::ParticleType::Boundary, verbose);
	}

	if (auto it = j.find("gravity"); it != j.end() &&
		it->is_array() && it->size() == 3)
	{
		glm::vec3 g;
		g.x = (*it)[0].get<float>();
		g.y = (*it)[1].get<float>();
		g.z = (*it)[2].get<float>();
		Set_Gravity(g, verbose);
	}

	// --- [ pressure ] ------------------------------------------------------------------------------

	if (auto it = j.find("pressGamma"); it != j.end())
	{
		Set_PressGamma(it->get<float>(), verbose);
	}

	if (auto it = j.find("jacobiOmega"); it != j.end())
	{
		Set_JacobiOmega(it->get<float>(), verbose);
	}

	// TODO: implement setters for these
	{
		BeginWrite();

		if (auto it = j.find("densityDevTarget"); it != j.end())
		{
			params_.densityDevTarget = it->get<float>();
		}

		if (auto it = j.find("fixedIters"); it != j.end())
		{
			params_.fixedIters = it->get<uint32_t>();
		}

		if (auto it = j.find("minSolverIters"); it != j.end())
		{
			params_.minSolverIters = it->get<uint32_t>();
		}

		if (auto it = j.find("maxSolverIters"); it != j.end())
		{
			params_.maxSolverIters = it->get<uint32_t>();
		}

		auto ParseSolverTerminationCond = [](const nlohmann::json& v, Base::SolverTerminationCond& out) -> bool
			{
				if (v.is_string())
				{
					const std::string s = v.get<std::string>();
					if (s == "FixedIters")
						{ out = Base::SolverTerminationCond::FixedIters; return true; }
					if (s == "RelDensityError_Raw")
						{ out = Base::SolverTerminationCond::RelDensityError_Raw; return true; }
					if (s == "RelDensityError_Pos")
						{ out = Base::SolverTerminationCond::RelDensityError_Pos; return true; }
					if (s == "RelDensityError_Abs")
						{ out = Base::SolverTerminationCond::RelDensityError_Abs; return true; }
					return false;
				}
				else if (v.is_number_integer())
				{
					out = static_cast<Base::SolverTerminationCond>(v.get<int>());
					return true;
				}
				return false;
			};

		if (auto it = j.find("solverTerminationCond"); it != j.end())
		{
			Base::SolverTerminationCond stc;
			if (ParseSolverTerminationCond(*it, stc))
			{
				params_.solverTerminationCond = stc;
			}
			else
			{
				SPDLOG_WARN("Invalid 'solverTerminationCond' value in JSON.");
			}
		}

		EndWrite();
	}

	// --- [ data structuring ] ----------------------------------------------------------------------

	if (auto it = j.find("gridCellSizeRatio"); it != j.end())
	{
		BeginWrite();

		const float gridCellSizeRatio = it->get<float>();
		if (gridCellSizeRatio > 0.0f)
		{
			params_.gridCellSizeRatio = gridCellSizeRatio;
		}
		else
		{
			SPDLOG_WARN("Invalid 'gridCellSizeRatio' value in JSON (must be > 0).");
		}

		EndWrite();
	}

	// --- [ jitter ] --------------------------------------------------------------------------------

	// TODO: implement setters for jitterFrac/jitterSeed stuff?
	{
		BeginWrite();

		if (auto it = j.find("jitterFracF"); it != j.end())
		{
			params_.jitterFracF = it->get<float>();
		}

		if (auto it = j.find("jitterSeedF"); it != j.end())
		{
			params_.jitterSeedF = it->get<int64_t>();
		}

		if (auto it = j.find("jitterFracB"); it != j.end())
		{
			params_.jitterFracB = it->get<float>();
		}

		if (auto it = j.find("jitterSeedB"); it != j.end())
		{
			params_.jitterSeedB = it->get<int64_t>();
		}

		EndWrite();
	}

	// --- [ boundary layers ] -----------------------------------------------------------------------

	if (auto it = j.find("boundaryLayers"); it != j.end())
	{
		Set_BoundaryLayers(it->get<uint32_t>(), verbose);
	}

	// --- [ surface sampling ] ----------------------------------------------------------------------

	{
		float minDistanceRatio = params_.surfaceSamplingMinDistanceRatio;
		float maxDistanceRatio = params_.surfaceSamplingMaxDistanceRatio;
		float coverageProbeSpacingRatio = params_.surfaceSamplingCoverageProbeSpacingRatio;
		uint32_t candidateMultiplier = params_.surfaceSamplingCandidateMultiplier;
		uint32_t maxTrialsPerCell = params_.surfaceSamplingMaxTrialsPerCell;
		uint64_t seed = params_.surfaceSamplingSeed;


		if (auto it = j.find("surfaceSamplingMinDistanceRatio"); it != j.end())
		{
			minDistanceRatio = it->get<float>();
		}
		if (auto it = j.find("surfaceSamplingMaxDistanceRatio"); it != j.end())
		{
			maxDistanceRatio = it->get<float>();
		}
		if (auto it = j.find("surfaceSamplingCoverageProbeSpacingRatio"); it != j.end())
		{
			coverageProbeSpacingRatio = it->get<float>();
		}
		if (auto it = j.find("surfaceSamplingCandidateMultiplier"); it != j.end())
		{
			candidateMultiplier = it->get<uint32_t>();
		}
		if (auto it = j.find("surfaceSamplingMaxTrialsPerCell"); it != j.end())
		{
			maxTrialsPerCell = it->get<uint32_t>();
		}
		if (auto it = j.find("surfaceSamplingSeed"); it != j.end())
		{
			seed = it->get<uint64_t>();
		}
		else if (j.find("jitterSeedB") != j.end())
		{
			// preserve the sampler seed used by existing parameter profiles
			seed = static_cast<uint64_t>(params_.jitterSeedB);
		}


		if (!SurfaceSamplingDistanceRatiosValid_(
			minDistanceRatio,
			maxDistanceRatio,
			coverageProbeSpacingRatio,
			params_.particleSizeF,
			params_.particleSizeB))
		{
			SPDLOG_WARN("Invalid surface sampling distance ratios in JSON.");
			return false;
		}
		if (candidateMultiplier == 0 || maxTrialsPerCell == 0)
		{
			SPDLOG_WARN("Surface sampling candidate multiplier and trials per cell must both be > 0.");
			return false;
		}

		BeginWrite();
		params_.surfaceSamplingMinDistanceRatio = minDistanceRatio;
		params_.surfaceSamplingMaxDistanceRatio = maxDistanceRatio;
		params_.surfaceSamplingCoverageProbeSpacingRatio = coverageProbeSpacingRatio;
		params_.surfaceSamplingCandidateMultiplier = candidateMultiplier;
		params_.surfaceSamplingMaxTrialsPerCell = maxTrialsPerCell;
		params_.surfaceSamplingSeed = seed;

		UpdateSurfaceSamplingDistances_();
		EndWrite();
	}

	// --- [ statistics, benchmarks, and exports ] ---------------------------------------------------

	// TODO: implement setters for maxStatRecs/maxBenchRecs?
	{
		BeginWrite();

		if (auto it = j.find("maxBenchRecs"); it != j.end())
		{
			params_.maxBenchRecs = it->get<uint32_t>();
		}

		if (auto it = j.find("maxStatRecs"); it != j.end())
		{
			params_.maxStatRecs = it->get<uint32_t>();
		}

		auto ParseTimeUnit = [](const nlohmann::json& v, Bench::TimeUnit& out) -> bool
			{
				if (v.is_string())
				{
					const std::string s = v.get<std::string>();
					if (s == "s") { out = Bench::TimeUnit::s; return true; }
					if (s == "ms") { out = Bench::TimeUnit::ms; return true; }
					if (s == "us") { out = Bench::TimeUnit::us; return true; }
					return false;
				}
				else if (v.is_number_integer())
				{
					out = static_cast<Bench::TimeUnit>(v.get<int>());
					return true;
				}
				return false;
			};

		if (auto it = j.find("timeUnit"); it != j.end())
		{
			Bench::TimeUnit tu;
			if (ParseTimeUnit(*it, tu))
			{
				params_.timeUnit = tu;
			}
			else
			{
				SPDLOG_WARN("Invalid 'timeUnit' value in JSON.");
			}
		}

		auto ParseByteUnit = [](const nlohmann::json& v, Mem::ByteUnit& out) -> bool
			{
				if (v.is_string())
				{
					const std::string s = v.get<std::string>();
					if (s == "GB") { out = Mem::ByteUnit::GB; return true; }
					if (s == "MB") { out = Mem::ByteUnit::MB; return true; }
					if (s == "KB") { out = Mem::ByteUnit::KB; return true; }
					if (s == "B") { out = Mem::ByteUnit::B; return true; }
					return false;
				}
				else if (v.is_number_integer())
				{
					out = static_cast<Mem::ByteUnit>(v.get<int>());
					return true;
				}
				return false;
			};

		if (auto it = j.find("byteUnit"); it != j.end())
		{
			Mem::ByteUnit bu;
			if (ParseByteUnit(*it, bu))
			{
				params_.byteUnit = bu;
			}
			else
			{
				SPDLOG_WARN("Invalid 'byteUnit' value in JSON.");
			}
		}

		if (auto it = j.find("statGroupsEnabled"); it != j.end() && it->is_object())
		{
			auto& sge = params_.statGroupsEnabled;

			if (auto v = it->find("particles"); v != it->end() && v->is_boolean())
			{
				sge[static_cast<uint8_t>(Stats::StatGroup::Particles)] = v->get<bool>();
			}

			if (auto v = it->find("density"); v != it->end() && v->is_boolean())
			{
				sge[static_cast<uint8_t>(Stats::StatGroup::Density)] = v->get<bool>();
			}

			if (auto v = it->find("velocity"); v != it->end() && v->is_boolean())
			{
				sge[static_cast<uint8_t>(Stats::StatGroup::Velocity)] = v->get<bool>();
			}
		}

		if (auto it = j.find("exportEnabled"); it != j.end())
		{
			params_.exportEnabled = it->get<bool>();
		}

		if (auto it = j.find("exportDir"); it != j.end())
		{
			params_.exportDir = it->get<std::string>();
		}

		if (auto it = j.find("exportInterval"); it != j.end())
		{
			params_.exportInterval = it->get<uint32_t>();
		}

		EndWrite();
	}

	// --- [ export surface classification ] ---------------------------------------------------------

	{
		float surfaceNbrCountRatio = params_.surfaceNbrCountRatio;

		if (auto it = j.find("surfaceNbrCountRatio"); it != j.end())
		{
			surfaceNbrCountRatio = it->get<float>();
		}

		if (!SurfaceNbrCountRatioValid_(surfaceNbrCountRatio))
		{
			SPDLOG_WARN("Invalid surface neighbor count ratio in JSON.");
			return false;
		}

		BeginWrite();
		params_.surfaceNbrCountRatio = surfaceNbrCountRatio;

		UpdateSurfaceNbrCountThreshold_();
		EndWrite();
	}

	if (!AssertParamDerivationCorrectness())
	{
		SPDLOG_ERROR("Derived simulation parameters are invalid.");
		return false;
	}

	return true;
}

bool SimParamsMgr::LoadSimParamsFromProfile(const std::string& profile, bool verbose)
{
	nlohmann::json j;
	if (!Base::LoadJsonObjectFromFile(profile + "/params.json", j))
	{
		return false;
	}

	return LoadSimParamsFromJson(j, verbose);
}

bool SimParamsMgr::LoadSimParamsFromCliOverrides(
	const std::vector<std::string>& overrides, bool verbose)
{
	if (overrides.empty()) { return true; }

	auto Trim = [](std::string s) -> std::string
		{
			const auto IsSpace = [](unsigned char c) -> bool
				{
					return std::isspace(c) != 0;
				};

			size_t left = 0;
			while (left < s.size() && IsSpace(static_cast<unsigned char>(s[left])))
			{
				left++;
			}

			size_t right = s.size();
			while (right > left && IsSpace(static_cast<unsigned char>(s[right - 1])))
			{
				right--;
			}

			return s.substr(left, right - left);
		};

	nlohmann::json j = nlohmann::json::object();

	for (const std::string& rawOverride : overrides)
	{
		const size_t iEq = rawOverride.find('=');
		if (iEq == std::string::npos)
		{
			SPDLOG_ERROR("Invalid '-sim/--sim' override '{}': Expected key=value.",
				rawOverride);
			return false;
		}

		const std::string path = Trim(rawOverride.substr(0, iEq));
		const std::string valueText = Trim(rawOverride.substr(iEq + 1));

		if (path.empty())
		{
			SPDLOG_ERROR("Invalid '-sim/--sim' override '{}': Missing key.", rawOverride);
			return false;
		}

		nlohmann::json value{};
		try
		{
			// supports numbers, booleans, arrays/objects and quoted strings
			value = nlohmann::json::parse(valueText);
		}
		catch (...)
		{
			// fallback for unquoted strings like CellLinkedList_Compact
			value = valueText;
		}

		nlohmann::json* cursor = &j;
		size_t begin = 0;

		while (true)
		{
			const size_t iDot = path.find('.', begin);
			const bool isLast = (iDot == std::string::npos);

			const size_t end = isLast ? path.size() : iDot;
			const std::string key = path.substr(begin, end - begin);
			if (key.empty())
			{
				SPDLOG_ERROR("Invalid '-sim/--sim' override '{}': Malformed key path.",
					rawOverride);
				return false;
			}

			if (isLast)
			{
				(*cursor)[key] = value;
				break;
			}

			nlohmann::json& next = (*cursor)[key];
			if (!next.is_object() && !next.is_null())
			{
				SPDLOG_ERROR(
					"Invalid '-sim/--sim' override '{}': Key path collides with a non-object.",
					rawOverride);
				return false;
			}
			if (next.is_null())
			{
				next = nlohmann::json::object();
			}

			cursor = &next;
			begin = iDot + 1;
		}
	}

	return LoadSimParamsFromJson(j, verbose);
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

void SimParamsMgr::Set_TimeStep(
	float timeStep, bool verbose)
{
	if (!(timeStep > 0.0f))
	{
		SPDLOG_ERROR("Parameter change cancelled: 'timeStep' must be > 0.");
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
		SPDLOG_ERROR("Parameter change cancelled: 'particleSize' must be > 0.");
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
		SPDLOG_WARN("Changing 'particleSize' is not implemented for this particle type.");
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
		params_.massF = params_.restDensity * particleSize
											* particleSize
											* particleSize;
		params_.kernelRadius = params_.kernelSuppRatio * params_.particleSizeF;
		params_.kernelRadiusSqu = params_.kernelRadius * params_.kernelRadius;
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
		params_.massB = params_.restDensity * particleSize
											* particleSize
											* particleSize;
		params_.kernelRadius = params_.kernelSuppRatio * params_.particleSizeF;
		params_.kernelRadiusSqu = params_.kernelRadius * params_.kernelRadius;
		UpdateSurfaceSamplingDistances_();
	}

	EndWrite();
}

void SimParamsMgr::Set_KernelSuppRatio(
	float kernelSuppRatio, bool verbose)
{
	if (!Base::ApproxEqu(kernelSuppRatio, CubicSplineKernelSuppRatio))
	{
		SPDLOG_ERROR("Parameter change cancelled: 'kernelSuppRatio' must be {} for hardcoded cubic-spline kernels.",
			CubicSplineKernelSuppRatio);
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
	UpdateNeighborCountHeuristics_();
	EndWrite();
}

void SimParamsMgr::Set_KernelRadius(
	float kernelRadius, bool verbose)
{
	if (!(kernelRadius > 0.0f))
	{
		SPDLOG_ERROR("Parameter change cancelled: 'kernelRadius' must be > 0.");
		return;
	}

	const float kernelSuppRatio = kernelRadius / params_.particleSizeF;
	if (!Base::ApproxEqu(kernelSuppRatio, CubicSplineKernelSuppRatio))
	{
		SPDLOG_ERROR("Parameter change cancelled: 'kernelRadius' must match hardcoded cubic-spline support radius {}h.",
			CubicSplineKernelSuppRatio);
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
	params_.kernelSuppRatio = kernelSuppRatio;
	UpdateNeighborCountHeuristics_();
	EndWrite();
}

void SimParamsMgr::Set_RestDensity(
	float restDensity, bool verbose)
{
	if (!(restDensity > 0.0f))
	{
		SPDLOG_ERROR("Parameter change cancelled: 'restDensity' must be > 0.");
		return;
	}

	// detect if parameter would actually change
	bool changed = false;
	Base::MarkIfChanged(restDensity, params_.restDensity, changed);
	if (!changed) { return; } // nothing to change

	if (verbose)
	{
		SPDLOG_INFO("Changed 'restDensity' from {} to {} "
					"(and dependent params accordingly)",
			params_.restDensity, restDensity);
	}

	BeginWrite();
	params_.restDensity = restDensity;
	params_.massF = restDensity * params_.particleSizeF
								* params_.particleSizeF
								* params_.particleSizeF;
	params_.massB = restDensity * params_.particleSizeB
								* params_.particleSizeB
								* params_.particleSizeB;
	EndWrite();
}

void SimParamsMgr::Set_Mass(
	float mass, Base::ParticleType type, bool verbose)
{
	if (!(mass > 0.0f))
	{
		SPDLOG_ERROR("Parameter change cancelled: 'mass' must be > 0.");
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
		SPDLOG_WARN("Changing 'mass' is not implemented for this particle type.");
	}

	if (!changed) { return; } // nothing to change

	const float particleSize = type == Base::ParticleType::Fluid
		? params_.particleSizeF
		: params_.particleSizeB;
	const float restDensity = mass / (particleSize * particleSize * particleSize);

	if (type == Base::ParticleType::Fluid)
	{
		if (verbose)
		{
			SPDLOG_INFO("Changed 'massF' from {} to {} "
						"(and dependent params accordingly)",
				params_.massF, mass);
		}
	}
	else if (type == Base::ParticleType::Boundary)
	{
		if (verbose)
		{
			SPDLOG_INFO("Changed 'massB' from {} to {} "
						"(and dependent params accordingly)",
				params_.massB, mass);
		}
	}

	BeginWrite();
	params_.restDensity = restDensity;
	params_.massF = restDensity * params_.particleSizeF
								* params_.particleSizeF
								* params_.particleSizeF;
	params_.massB = restDensity * params_.particleSizeB
								* params_.particleSizeB
								* params_.particleSizeB;
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
		SPDLOG_WARN("Changing 'viscosity' is not implemented for this particle type.");
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

void SimParamsMgr::Set_PressGamma(
	float pressGamma, bool verbose)
{
	// detect if parameter would actually change
	bool changed = false;
	Base::MarkIfChanged(pressGamma, params_.pressGamma, changed);
	if (!changed) { return; } // nothing to change

	if (verbose)
	{
		SPDLOG_INFO("Changed 'pressGamma' from {} to {}",
			params_.pressGamma, pressGamma);
	}

	// apply changes
	BeginWrite();
	params_.pressGamma = pressGamma;
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
		SPDLOG_ERROR("Parameter change cancelled: 'minSolverIters' must be > 0.");
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
		SPDLOG_ERROR("Parameter change cancelled: 'maxSolverIters' must be > 0.");
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

void SimParamsMgr::Set_SurfaceSamplingDistanceRatios(
	const float minDistanceRatio,
	const float maxDistanceRatio,
	const float coverageProbeSpacingRatio,
	const bool verbose)
{
	if (!SurfaceSamplingDistanceRatiosValid_(
		minDistanceRatio,
		maxDistanceRatio,
		coverageProbeSpacingRatio,
		params_.particleSizeF,
		params_.particleSizeB))
	{
		SPDLOG_ERROR(
			"Parameter change cancelled: Surface sampling ratios must be finite and > 0, "
			"with maxDistanceRatio >= minDistanceRatio + coverageProbeSpacingRatio "
			"and maxDistanceRatio * particleSizeB <= particleSizeF.");
		return;
	}

	bool changed = false;
	Base::MarkIfChanged(
		minDistanceRatio,
		params_.surfaceSamplingMinDistanceRatio,
		changed);
	Base::MarkIfChanged(
		maxDistanceRatio,
		params_.surfaceSamplingMaxDistanceRatio,
		changed);
	Base::MarkIfChanged(
		coverageProbeSpacingRatio,
		params_.surfaceSamplingCoverageProbeSpacingRatio,
		changed);
	if (!changed) { return; }

	if (verbose)
	{
		SPDLOG_INFO("Changed surface sampling distance ratios from [{}, {}, {}] to [{}, {}, {}].",
			params_.surfaceSamplingMinDistanceRatio,
			params_.surfaceSamplingMaxDistanceRatio,
			params_.surfaceSamplingCoverageProbeSpacingRatio,
			minDistanceRatio,
			maxDistanceRatio,
			coverageProbeSpacingRatio);
	}

	BeginWrite();
	params_.surfaceSamplingMinDistanceRatio = minDistanceRatio;
	params_.surfaceSamplingMaxDistanceRatio = maxDistanceRatio;
	params_.surfaceSamplingCoverageProbeSpacingRatio = coverageProbeSpacingRatio;
	UpdateSurfaceSamplingDistances_();
	EndWrite();
}

void SimParamsMgr::Set_SurfaceNbrCountRatio(
	const float surfaceNbrCountRatio, const bool verbose)
{
	if (!SurfaceNbrCountRatioValid_(surfaceNbrCountRatio))
	{
		SPDLOG_ERROR(
			"Parameter change cancelled: Surface neighbor count ratio must be finite, "
			"> 0 and <= {}.", surfaceNbrCountRatioMax_);
		return;
	}

	// detect if parameter would actually change
	bool changed = false;
	Base::MarkIfChanged(surfaceNbrCountRatio, params_.surfaceNbrCountRatio, changed);
	if (!changed) { return; } // nothing to change

	if (verbose)
	{
		SPDLOG_INFO("Changed 'surfaceNbrCountRatio' from {} to {}",
			params_.surfaceNbrCountRatio, surfaceNbrCountRatio);
	}

	// apply changes
	BeginWrite();
	params_.surfaceNbrCountRatio = surfaceNbrCountRatio;
	UpdateSurfaceNbrCountThreshold_();
	EndWrite();
}

void SimParamsMgr::UpdateNeighborCountHeuristics_()
{
	params_.expctdNbrCountSoftMax_FF = HeuristicExpectedNeighborCount_SoftMax(
		params_.kernelSuppRatio, Base::ParticleType::Fluid);
	params_.expctdNbrCountSoftMax_FB = HeuristicExpectedNeighborCount_SoftMax(
		params_.kernelSuppRatio, Base::ParticleType::Boundary);
	params_.expctdNbrCountHardMax_FF = HeuristicExpectedNeighborCount_HardMax(
		params_.kernelSuppRatio, Base::ParticleType::Fluid);
	params_.expctdNbrCountHardMax_FB = HeuristicExpectedNeighborCount_HardMax(
		params_.kernelSuppRatio, Base::ParticleType::Boundary);
	params_.expctdAvgNbrCountHardMax_FF = HeuristicExpectedAvgNeighborCount_HardMax(
		params_.kernelSuppRatio, Base::ParticleType::Fluid);
	params_.expctdAvgNbrCountHardMax_FB = HeuristicExpectedAvgNeighborCount_HardMax(
		params_.kernelSuppRatio, Base::ParticleType::Boundary);

	// the surface threshold is a fraction of the expected fluid neighbor count
	UpdateSurfaceNbrCountThreshold_();
}

void SimParamsMgr::UpdateSurfaceSamplingDistances_()
{
	params_.surfaceSamplingMinDistance =
		params_.surfaceSamplingMinDistanceRatio * params_.particleSizeB;
	params_.surfaceSamplingMaxDistance =
		params_.surfaceSamplingMaxDistanceRatio * params_.particleSizeB;
	params_.surfaceSamplingCoverageProbeSpacing =
		params_.surfaceSamplingCoverageProbeSpacingRatio * params_.particleSizeB;
}

void SimParamsMgr::UpdateSurfaceNbrCountThreshold_()
{
	params_.surfaceNbrCountThreshold = HeuristicSurfaceNbrCountThreshold(
		params_.surfaceNbrCountRatio, params_.expctdAvgNbrCountHardMax_FF);
}

bool SimParamsMgr::SurfaceSamplingDistanceRatiosValid_(
	const float minDistanceRatio,
	const float maxDistanceRatio,
	const float coverageProbeSpacingRatio,
	const float particleSizeF,
	const float particleSizeB)
{
	if (!std::isfinite(minDistanceRatio) || minDistanceRatio <= 0.0f ||
		!std::isfinite(maxDistanceRatio) || maxDistanceRatio <= 0.0f ||
		!std::isfinite(coverageProbeSpacingRatio) || coverageProbeSpacingRatio <= 0.0f)
	{
		return false;
	}

	const float ratioEpsilon = 32.0f * std::numeric_limits<float>::epsilon() *
		std::max(
			maxDistanceRatio,
			minDistanceRatio + coverageProbeSpacingRatio);
	if (maxDistanceRatio + ratioEpsilon <
		minDistanceRatio + coverageProbeSpacingRatio)
	{
		return false;
	}

	const float maxDistance = maxDistanceRatio * particleSizeB;
	const float distanceEpsilon = 32.0f * std::numeric_limits<float>::epsilon() *
		std::max(maxDistance, particleSizeF);
	return maxDistance <= particleSizeF + distanceEpsilon;
}

bool SimParamsMgr::SurfaceNbrCountRatioValid_(const float surfaceNbrCountRatio)
{
	if (!std::isfinite(surfaceNbrCountRatio) || surfaceNbrCountRatio <= 0.0f)
	{
		return false;
	}

	const float ratioEpsilon = 32.0f * std::numeric_limits<float>::epsilon() * surfaceNbrCountRatioMax_;
	return surfaceNbrCountRatio <= surfaceNbrCountRatioMax_ + ratioEpsilon;
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

	// --- [ hard invariants ] -----------------------------------------------------------------------

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
	ok &= Check("restDensity <= 0", true, [&]()
	{
		return par.restDensity > 0.0f;
	});
	ok &= Check("massF <= 0", true, [&]()
	{
		return par.massF > 0.0f;
	});
	ok &= Check("massB <= 0", true, [&]()
	{
		return par.massB > 0.0f;
	});
	ok &= Check("pressGamma < 0", true, [&]()
	{
		return par.pressGamma >= 0.0f;
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
	ok &= Check("surfaceSamplingCandidateMultiplier == 0", true, [&]()
	{
		return par.surfaceSamplingCandidateMultiplier > 0;
	});
	ok &= Check("surfaceSamplingMaxTrialsPerCell == 0", true, [&]()
	{
		return par.surfaceSamplingMaxTrialsPerCell > 0;
	});
	ok &= Check("surfaceNbrCountThreshold == 0", true, [&]()
	{
		return par.surfaceNbrCountThreshold > 0;
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
	ok &= Check("massF != restDensity * particleSizeF^3", true, [&]()
	{
		float expect = par.restDensity * par.particleSizeF
									* par.particleSizeF
									* par.particleSizeF;
		return Base::ApproxEqu(par.massF, expect);
	});
	ok &= Check("massB != restDensity * particleSizeB^3", true, [&]()
	{
		float expect = par.restDensity * par.particleSizeB
									* par.particleSizeB
									* par.particleSizeB;
		return Base::ApproxEqu(par.massB, expect);
	});
	ok &= Check("surface sampling distance ratios invalid", true, [&]()
	{
		return SurfaceSamplingDistanceRatiosValid_(
			par.surfaceSamplingMinDistanceRatio,
			par.surfaceSamplingMaxDistanceRatio,
			par.surfaceSamplingCoverageProbeSpacingRatio,
			par.particleSizeF,
			par.particleSizeB);
	});
	ok &= Check("surfaceSamplingMinDistance derivation mismatch", true, [&]()
	{
		const float expect = par.surfaceSamplingMinDistanceRatio * par.particleSizeB;
		return Base::ApproxEqu(par.surfaceSamplingMinDistance, expect);
	});
	ok &= Check("surfaceSamplingMaxDistance derivation mismatch", true, [&]()
	{
		const float expect = par.surfaceSamplingMaxDistanceRatio * par.particleSizeB;
		return Base::ApproxEqu(par.surfaceSamplingMaxDistance, expect);
	});
	ok &= Check("surfaceSamplingCoverageProbeSpacing derivation mismatch", true, [&]()
	{
		const float expect =
			par.surfaceSamplingCoverageProbeSpacingRatio * par.particleSizeB;
		return Base::ApproxEqu(par.surfaceSamplingCoverageProbeSpacing, expect);
	});
	ok &= Check("surfaceNbrCountRatio invalid", true, [&]()
	{
		return SurfaceNbrCountRatioValid_(par.surfaceNbrCountRatio);
	});
	ok &= Check("surfaceNbrCountThreshold derivation mismatch", true, [&]()
	{
		return par.surfaceNbrCountThreshold == HeuristicSurfaceNbrCountThreshold(
			par.surfaceNbrCountRatio, par.expctdAvgNbrCountHardMax_FF);
	});
	ok &= Check("kernelSuppRatio != cubic spline support ratio", true, [&]()
	{
		return Base::ApproxEqu(par.kernelSuppRatio, CubicSplineKernelSuppRatio);
	});
	ok &= Check("gridCellSizeRatio <= 0", true, [&]()
	{
		return par.gridCellSizeRatio > 0.0f;
	});
	ok &= Check("kernelRadius != kernelSuppRatio * particleSizeF", true, [&]()
	{
		float expect = par.kernelSuppRatio * par.particleSizeF;
		return Base::ApproxEqu(par.kernelRadius, expect);
	});
	ok &= Check("kernelRadiusSqu != kernelRadius^2", true, [&]()
	{
		float expect = par.kernelRadius * par.kernelRadius;
		return Base::ApproxEqu(par.kernelRadiusSqu, expect);
	});
	ok &= Check("expctdNbrCountSoftMax_FF heuristic mismatch", true, [&]()
	{
		return par.expctdNbrCountSoftMax_FF == HeuristicExpectedNeighborCount_SoftMax(
			par.kernelSuppRatio, Base::ParticleType::Fluid);
	});
	ok &= Check("expctdNbrCountSoftMax_FB heuristic mismatch", true, [&]()
	{
		return par.expctdNbrCountSoftMax_FB == HeuristicExpectedNeighborCount_SoftMax(
			par.kernelSuppRatio, Base::ParticleType::Boundary);
	});
	ok &= Check("expctdNbrCountHardMax_FF heuristic mismatch", true, [&]()
	{
		return par.expctdNbrCountHardMax_FF == HeuristicExpectedNeighborCount_HardMax(
			par.kernelSuppRatio, Base::ParticleType::Fluid);
	});
	ok &= Check("expctdNbrCountHardMax_FB heuristic mismatch", true, [&]()
	{
		return par.expctdNbrCountHardMax_FB == HeuristicExpectedNeighborCount_HardMax(
			par.kernelSuppRatio, Base::ParticleType::Boundary);
	});
	ok &= Check("expctdAvgNbrCountHardMax_FF heuristic mismatch", true, [&]()
	{
		return par.expctdAvgNbrCountHardMax_FF == HeuristicExpectedAvgNeighborCount_HardMax(
			par.kernelSuppRatio, Base::ParticleType::Fluid);
	});
	ok &= Check("expctdAvgNbrCountHardMax_FB heuristic mismatch", true, [&]()
	{
		return par.expctdAvgNbrCountHardMax_FB == HeuristicExpectedAvgNeighborCount_HardMax(
			par.kernelSuppRatio, Base::ParticleType::Boundary);
	});
	ok &= Check("expctdNbrCountSoftMax_FF > expctdNbrCountHardMax_FF", true, [&]()
	{
		return par.expctdNbrCountSoftMax_FF <= par.expctdNbrCountHardMax_FF;
	});
	ok &= Check("expctdNbrCountSoftMax_FB > expctdNbrCountHardMax_FB", true, [&]()
	{
		return par.expctdNbrCountSoftMax_FB <= par.expctdNbrCountHardMax_FB;
	});
	ok &= Check("expctdAvgNbrCountHardMax_FF > expctdNbrCountHardMax_FF", true, [&]()
	{
		return par.expctdAvgNbrCountHardMax_FF <= par.expctdNbrCountHardMax_FF;
	});
	ok &= Check("expctdAvgNbrCountHardMax_FB > expctdNbrCountHardMax_FB", true, [&]()
	{
		return par.expctdAvgNbrCountHardMax_FB <= par.expctdNbrCountHardMax_FB;
	});

	return ok;
}

uint32_t SimParamsMgr::HeuristicExpectedNeighborCount_SoftMax(
	float kernelSuppRatio, Base::ParticleType type) const
{
	float volume = (4.0f / 3.0f) * PI * std::pow(kernelSuppRatio, 3.0f);
	float occupancy = type == Base::ParticleType::Fluid ? 1.0f : 0.5f;
	float margin = 0.2f;

	float heuristic = std::ceil(volume * occupancy * (1.0f + margin));
	return static_cast<uint32_t>(heuristic);
}

uint32_t SimParamsMgr::HeuristicExpectedNeighborCount_HardMax(
	float kernelSuppRatio, Base::ParticleType type) const
{
	float volume = (4.0f / 3.0f) * PI * std::pow(kernelSuppRatio, 3.0f);
	float occupancy = type == Base::ParticleType::Fluid ? 1.0f : 0.75f;
	float margin = 0.8f;

	float heuristic = std::ceil(volume * occupancy * (1.0f + margin));
	return static_cast<uint32_t>(heuristic);
}

uint32_t SimParamsMgr::HeuristicExpectedAvgNeighborCount_HardMax(
	float kernelSuppRatio, Base::ParticleType type) const
{
	const SimParams refParams{};

	float volume = (4.0f / 3.0f) * PI * std::pow(kernelSuppRatio, 3.0f);
	float refVolume = (4.0f / 3.0f) * PI * std::pow(CubicSplineKernelSuppRatio, 3.0f);
	float occupancy = type == Base::ParticleType::Fluid
		? static_cast<float>(refParams.expctdAvgNbrCountHardMax_FF) / refVolume
		: static_cast<float>(refParams.expctdAvgNbrCountHardMax_FB) / refVolume;
	float margin = 0.0f;

	float heuristic = std::ceil(volume * occupancy * (1.0f + margin));
	return static_cast<uint32_t>(heuristic);
}

uint32_t SimParamsMgr::HeuristicSurfaceNbrCountThreshold(
	float surfaceNbrCountRatio, uint32_t expctdAvgNbrCountHardMax_FF) const
{
	// a fluid particle is near the surface once its kernel support is truncated far enough,
	// so the threshold tracks the expected fluid neighbor count of a full support
	float heuristic = std::round(
		surfaceNbrCountRatio * static_cast<float>(expctdAvgNbrCountHardMax_FF));

	return std::max(static_cast<uint32_t>(heuristic), 1u);
}

inline void SimParamsMgr::BeginWrite()
{
	version_.fetch_add(1, std::memory_order_acq_rel);
}

inline void SimParamsMgr::EndWrite()
{
	version_.fetch_add(1, std::memory_order_release);
}
