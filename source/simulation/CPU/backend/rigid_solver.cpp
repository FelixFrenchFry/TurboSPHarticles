// rigid_solver.cpp

// external
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// internal
#include "rigid_solver.h"
#include "cpu_backend.h"
#include "source/simulation/core/solver_constants.h"
#include "source/simulation/CPU/simd.h"



namespace
{
	template<class Function>
	void ForEachContactNeighbor(const bool boundary, const uint32_t i,
		const Sim::I_NeighborListSELL& neighborList,
		const CPU::RigidBoundaryContactNeighbors& contactNeighbors,
		const Function& function)
	{
		if (boundary)
		{
			const uint64_t begin = contactNeighbors.offsetsBR[i];
			const uint64_t end = contactNeighbors.offsetsBR[i + 1];
			for (uint64_t k = begin; k < end; k++) { function(contactNeighbors.indicesBR[k], false); }
			return;
		}
		const Sim::NeighborSpan nbrRR = neighborList.RR(i);
		for (uint32_t k = 0; k < nbrRR.N; k++) { function(nbrRR.data[k], false); }
		if (!contactNeighbors.offsetsRB.empty())
		{
			const uint64_t begin = contactNeighbors.offsetsRB[i];
			const uint64_t end = contactNeighbors.offsetsRB[i + 1];
			for (uint64_t k = begin; k < end; k++) { function(contactNeighbors.indicesRB[k], true); }
		}
	}

	float ContactKernel(const float dx, const float dy, const float dz,
		const float particleSizeInvF, const float kernelRadiusSqu, const float cubicSplineAlpha,
		float& xkg, float& ykg, float& zkg)
	{
		const float distSqu = std::fmaf(dx, dx, std::fmaf(dy, dy, dz * dz));
		if (distSqu >= kernelRadiusSqu) { return 0.0f; }
		const float q = std::sqrt(distSqu) * particleSizeInvF;
		const float t1 = std::max(1.0f - q, 0.0f);
		const float t2 = std::max(2.0f - q, 0.0f);
		CPU::KernelGradient_CubicSpline_Scalar(dx, dy, dz, particleSizeInvF,
			kernelRadiusSqu, cubicSplineAlpha, xkg, ykg, zkg);
		return cubicSplineAlpha * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);
	}

	struct BodyGradient
	{
		uint32_t bodyIdx;
		float x = 0.0f, y = 0.0f, z = 0.0f;
		float tx = 0.0f, ty = 0.0f, tz = 0.0f;
	};

	void AddBodyGradient(std::vector<BodyGradient>& gradients, const uint32_t bodyIdx,
		const float x, const float y, const float z, const float rx, const float ry, const float rz)
	{
		auto it = std::find_if(gradients.begin(), gradients.end(),
			[&](const BodyGradient& gradient) { return gradient.bodyIdx == bodyIdx; });
		if (it == gradients.end())
		{
			gradients.push_back({ bodyIdx });
			it = gradients.end() - 1;
		}
		it->x += x; it->y += y; it->z += z;
		it->tx += ry * z - rz * y;
		it->ty += rz * x - rx * z;
		it->tz += rx * y - ry * x;
	}

} // namespace anonymous

bool CPU::BuildRigidBoundaryNeighbors(
	const Sim::I_NeighborListSELL& neighborList,
	const uint32_t N_R,
	RigidBoundaryContactNeighbors& contactNeighbors)
{
	contactNeighbors.Clear();
	auto& localIdxsB = contactNeighbors.localIdxsB;
	auto& offsetsRB = contactNeighbors.offsetsRB;
	auto& indicesRB = contactNeighbors.indicesRB;
	auto& offsetsBR = contactNeighbors.offsetsBR;
	auto& indicesBR = contactNeighbors.indicesBR;

	// assign contact indices only to boundary particles referenced by rigid neighbors
	std::unordered_map<uint32_t, uint32_t> contactIdx_FromLocalIdxB;
	bool hasRigidContact = false;
	for (uint32_t i = 0; i < N_R; i++)
	{
		const Sim::NeighborSpan nbrRB = neighborList.RB(i);
		if (offsetsRB.empty())
		{
			if (nbrRB.N == 0)
			{
				hasRigidContact = hasRigidContact || neighborList.CountRR(i) > 0;
				continue;
			}
			offsetsRB.resize(static_cast<uint64_t>(N_R) + 1ull, 0);
			offsetsBR.push_back(0);
		}
		for (uint32_t k = 0; k < nbrRB.N; k++)
		{
			const uint32_t localIdxB = nbrRB.data[k];
			const auto [it, inserted] = contactIdx_FromLocalIdxB.try_emplace(
				localIdxB, static_cast<uint32_t>(localIdxsB.size()));
			if (inserted)
			{
				localIdxsB.push_back(localIdxB);
				offsetsBR.push_back(0);
			}
			const uint32_t contactIdxB = it->second;
			indicesRB.push_back(contactIdxB);
			offsetsBR[contactIdxB + 1]++;
		}
		offsetsRB[i + 1] = indicesRB.size();
	}
	if (offsetsRB.empty()) { return hasRigidContact; }

	// invert the compact rigid-to-boundary neighbor list
	const uint32_t N_contactB = static_cast<uint32_t>(localIdxsB.size());
	for (uint32_t i = 0; i < N_contactB; i++) { offsetsBR[i + 1] += offsetsBR[i]; }
	indicesBR.resize(offsetsBR.back());
	auto writeOffsetsBR = offsetsBR;
	for (uint32_t i = 0; i < N_R; i++)
	{
		for (uint64_t k = offsetsRB[i]; k < offsetsRB[i + 1]; k++)
		{
			indicesBR[writeOffsetsBR[indicesRB[k]]++] = i;
		}
	}
	return true;
}

void CPU::RigidContactGeometry(
	const float* RESTRICT xPosRelR,
	const float* RESTRICT yPosRelR,
	const float* RESTRICT zPosRelR,
	const uint32_t* RESTRICT bodyIdR,
	const float* RESTRICT a00RB,
	const float* RESTRICT a01RB,
	const float* RESTRICT a02RB,
	const float* RESTRICT a10RB,
	const float* RESTRICT a11RB,
	const float* RESTRICT a12RB,
	const float* RESTRICT a20RB,
	const float* RESTRICT a21RB,
	const float* RESTRICT a22RB,
	const float* RESTRICT iBodyInv00RB,
	const float* RESTRICT iBodyInv01RB,
	const float* RESTRICT iBodyInv02RB,
	const float* RESTRICT iBodyInv10RB,
	const float* RESTRICT iBodyInv11RB,
	const float* RESTRICT iBodyInv12RB,
	const float* RESTRICT iBodyInv20RB,
	const float* RESTRICT iBodyInv21RB,
	const float* RESTRICT iBodyInv22RB,
	const uint32_t N_R,
	const uint32_t N_RB,
	// destinations
	float* RESTRICT xPosRelWorldR,
	float* RESTRICT yPosRelWorldR,
	float* RESTRICT zPosRelWorldR,
	float* RESTRICT iWorldInv00RB,
	float* RESTRICT iWorldInv01RB,
	float* RESTRICT iWorldInv02RB,
	float* RESTRICT iWorldInv10RB,
	float* RESTRICT iWorldInv11RB,
	float* RESTRICT iWorldInv12RB,
	float* RESTRICT iWorldInv20RB,
	float* RESTRICT iWorldInv21RB,
	float* RESTRICT iWorldInv22RB)
{
	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N_RB),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const float t00 = a00RB[i] * iBodyInv00RB[i] + a01RB[i] * iBodyInv10RB[i] + a02RB[i] * iBodyInv20RB[i];
				const float t01 = a00RB[i] * iBodyInv01RB[i] + a01RB[i] * iBodyInv11RB[i] + a02RB[i] * iBodyInv21RB[i];
				const float t02 = a00RB[i] * iBodyInv02RB[i] + a01RB[i] * iBodyInv12RB[i] + a02RB[i] * iBodyInv22RB[i];
				const float t10 = a10RB[i] * iBodyInv00RB[i] + a11RB[i] * iBodyInv10RB[i] + a12RB[i] * iBodyInv20RB[i];
				const float t11 = a10RB[i] * iBodyInv01RB[i] + a11RB[i] * iBodyInv11RB[i] + a12RB[i] * iBodyInv21RB[i];
				const float t12 = a10RB[i] * iBodyInv02RB[i] + a11RB[i] * iBodyInv12RB[i] + a12RB[i] * iBodyInv22RB[i];
				const float t20 = a20RB[i] * iBodyInv00RB[i] + a21RB[i] * iBodyInv10RB[i] + a22RB[i] * iBodyInv20RB[i];
				const float t21 = a20RB[i] * iBodyInv01RB[i] + a21RB[i] * iBodyInv11RB[i] + a22RB[i] * iBodyInv21RB[i];
				const float t22 = a20RB[i] * iBodyInv02RB[i] + a21RB[i] * iBodyInv12RB[i] + a22RB[i] * iBodyInv22RB[i];
				iWorldInv00RB[i] = t00 * a00RB[i] + t01 * a01RB[i] + t02 * a02RB[i];
				iWorldInv01RB[i] = t00 * a10RB[i] + t01 * a11RB[i] + t02 * a12RB[i];
				iWorldInv02RB[i] = t00 * a20RB[i] + t01 * a21RB[i] + t02 * a22RB[i];
				iWorldInv10RB[i] = t10 * a00RB[i] + t11 * a01RB[i] + t12 * a02RB[i];
				iWorldInv11RB[i] = t10 * a10RB[i] + t11 * a11RB[i] + t12 * a12RB[i];
				iWorldInv12RB[i] = t10 * a20RB[i] + t11 * a21RB[i] + t12 * a22RB[i];
				iWorldInv20RB[i] = t20 * a00RB[i] + t21 * a01RB[i] + t22 * a02RB[i];
				iWorldInv21RB[i] = t20 * a10RB[i] + t21 * a11RB[i] + t22 * a12RB[i];
				iWorldInv22RB[i] = t20 * a20RB[i] + t21 * a21RB[i] + t22 * a22RB[i];
			}
		}, tbb::auto_partitioner{});
	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N_R),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const uint32_t bodyIdx = bodyIdR[i];
				const float lx = xPosRelR[i], ly = yPosRelR[i], lz = zPosRelR[i];
				xPosRelWorldR[i] = std::fmaf(a00RB[bodyIdx], lx, std::fmaf(a01RB[bodyIdx], ly, a02RB[bodyIdx] * lz));
				yPosRelWorldR[i] = std::fmaf(a10RB[bodyIdx], lx, std::fmaf(a11RB[bodyIdx], ly, a12RB[bodyIdx] * lz));
				zPosRelWorldR[i] = std::fmaf(a20RB[bodyIdx], lx, std::fmaf(a21RB[bodyIdx], ly, a22RB[bodyIdx] * lz));
			}
		}, tbb::auto_partitioner{});
}

void CPU::RigidContactDensityAndDiagonal(
	const float* RESTRICT xPosR,
	const float* RESTRICT yPosR,
	const float* RESTRICT zPosR,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT volR,
	const float* RESTRICT volB,
	const float* RESTRICT xPosRelWorldR,
	const float* RESTRICT yPosRelWorldR,
	const float* RESTRICT zPosRelWorldR,
	const uint32_t* RESTRICT bodyIdR,
	const float* RESTRICT massInvRB,
	const float* RESTRICT iWorldInv00RB,
	const float* RESTRICT iWorldInv01RB,
	const float* RESTRICT iWorldInv02RB,
	const float* RESTRICT iWorldInv10RB,
	const float* RESTRICT iWorldInv11RB,
	const float* RESTRICT iWorldInv12RB,
	const float* RESTRICT iWorldInv20RB,
	const float* RESTRICT iWorldInv21RB,
	const float* RESTRICT iWorldInv22RB,
	const Sim::I_NeighborListSELL& neighborList,
	const RigidBoundaryContactNeighbors& contactNeighbors,
	const float particleSizeInvF,
	const float kernelRadiusSqu,
	const float cubicSplineAlpha,
	const float timeStep,
	const uint32_t N_R,
	// destinations
	float* RESTRICT densityContactR,
	float* RESTRICT densityContactB,
	float* RESTRICT diagElemContactR,
	float* RESTRICT diagElemContactB,
	float* RESTRICT pressureContactR,
	float* RESTRICT pressureContactB,
	float* RESTRICT sourceTermContactR,
	float* RESTRICT sourceTermContactB,
	float* RESTRICT jacobiOmegaContactR,
	float* RESTRICT jacobiOmegaContactB)
{
	const auto compute = [&](const bool boundary, const uint32_t N)
	{
		const float* xPos = boundary ? xPosB : xPosR;
		const float* yPos = boundary ? yPosB : yPosR;
		const float* zPos = boundary ? zPosB : zPosR;
		const float* vol = boundary ? volB : volR;
		float* density = boundary ? densityContactB : densityContactR;
		float* diagonal = boundary ? diagElemContactB : diagElemContactR;
		float* pressure = boundary ? pressureContactB : pressureContactR;
		float* source = boundary ? sourceTermContactB : sourceTermContactR;
		float* omega = boundary ? jacobiOmegaContactB : jacobiOmegaContactR;
		tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				std::vector<BodyGradient> gradients;
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					gradients.clear();
					const uint32_t localIdx = boundary ? contactNeighbors.localIdxsB[i] : i;
					const float volRest = RIGID_CONTACT_VOLUME_FACTOR * vol[localIdx];
					float rho = RIGID_CONTACT_VOLUME_FACTOR;
					bool hasContact = false;
					ForEachContactNeighbor(boundary, i, neighborList, contactNeighbors,
						[&](const uint32_t j, const bool neighborBoundary)
						{
							const uint32_t nbrLocalIdx = neighborBoundary ? contactNeighbors.localIdxsB[j] : j;
							float xkg, ykg, zkg;
							const float weight = ContactKernel(
								xPos[localIdx] - (neighborBoundary ? xPosB[nbrLocalIdx] : xPosR[j]),
								yPos[localIdx] - (neighborBoundary ? yPosB[nbrLocalIdx] : yPosR[j]),
								zPos[localIdx] - (neighborBoundary ? zPosB[nbrLocalIdx] : zPosR[j]),
								particleSizeInvF, kernelRadiusSqu, cubicSplineAlpha, xkg, ykg, zkg);
							if (!(weight > 0.0f)) { return; }
							hasContact = true;
							const float neighborVolume = RIGID_CONTACT_VOLUME_FACTOR * (neighborBoundary ? volB[nbrLocalIdx] : volR[j]);
							rho += neighborVolume * weight;
							xkg *= neighborVolume; ykg *= neighborVolume; zkg *= neighborVolume;
							if (!boundary)
							{
								AddBodyGradient(gradients, bodyIdR[i], xkg, ykg, zkg,
									xPosRelWorldR[i], yPosRelWorldR[i], zPosRelWorldR[i]);
							}
							if (!neighborBoundary)
							{
								AddBodyGradient(gradients, bodyIdR[j], -xkg, -ykg, -zkg,
									xPosRelWorldR[j], yPosRelWorldR[j], zPosRelWorldR[j]);
							}
						});
					// combine each body's gradients before applying its inverse mass and inertia
					double diag = 0.0;
					for (const BodyGradient& g : gradients)
					{
						const uint32_t b = g.bodyIdx;
						const float wx = iWorldInv00RB[b] * g.tx + iWorldInv01RB[b] * g.ty + iWorldInv02RB[b] * g.tz;
						const float wy = iWorldInv10RB[b] * g.tx + iWorldInv11RB[b] * g.ty + iWorldInv12RB[b] * g.tz;
						const float wz = iWorldInv20RB[b] * g.tx + iWorldInv21RB[b] * g.ty + iWorldInv22RB[b] * g.tz;
						diag += massInvRB[b] * (g.x * g.x + g.y * g.y + g.z * g.z)
							+ g.tx * wx + g.ty * wy + g.tz * wz;
					}
					diagonal[i] = static_cast<float>(timeStep * static_cast<double>(timeStep) * volRest / (rho * rho) * diag);
					if (!std::isfinite(diagonal[i]) || diagonal[i] < 0.0f)
					{
						throw std::runtime_error("Invalid CPU rigid contact diagonal.");
					}
					density[i] = rho;
					pressure[i] = 0.0f;
					source[i] = rho - 1.0f;
					omega[i] = hasContact ? 0.5f : 0.0f;
				}
			}, tbb::auto_partitioner{});
	};
	compute(false, N_R);
	compute(true, static_cast<uint32_t>(contactNeighbors.localIdxsB.size()));
}

uint64_t CPU::RigidContactRelaxation(
	const uint32_t* RESTRICT particleOffsetsR_ByBody,
	const uint32_t* RESTRICT particleLocalIdxsR_ByBody,
	const uint32_t* RESTRICT bodyIdR,
	const float* RESTRICT massInvRB,
	const uint64_t* RESTRICT offsetsBR,
	const uint32_t* RESTRICT indicesBR,
	const float* RESTRICT diagElemContactB,
	const uint32_t N_RB,
	const uint32_t N_B,
	// destinations
	float* RESTRICT jacobiOmegaContactR,
	float* RESTRICT jacobiOmegaContactB)
{
	uint64_t N_contacts = 0;
	for (uint32_t bodyIdx = 0; bodyIdx < N_RB; bodyIdx++)
	{
		const uint32_t begin = particleOffsetsR_ByBody[bodyIdx];
		const uint32_t end = particleOffsetsR_ByBody[bodyIdx + 1];
		uint32_t count = 0;
		for (uint32_t k = begin; k < end; k++) { count += jacobiOmegaContactR[particleLocalIdxsR_ByBody[k]] > 0.0f; }
		if (count == 0) { continue; }
		for (uint32_t k = begin; k < end; k++) { jacobiOmegaContactR[particleLocalIdxsR_ByBody[k]] /= count; }
		N_contacts += count;
	}
	if (N_B == 0) { return N_contacts; }

	// count each boundary particle once per movable rigid body
	std::vector<uint32_t> N_contactsB_ByBody(N_RB, 0);
	std::vector<uint32_t> lastContactIdxB_ByBody(N_RB, N_B);
	for (uint32_t i = 0; i < N_B; i++)
	{
		if (jacobiOmegaContactB[i] <= 0.0f) { continue; }
		N_contacts++;
		if (diagElemContactB[i] <= 0.0f) { continue; }
		for (uint64_t k = offsetsBR[i]; k < offsetsBR[i + 1]; k++)
		{
			const uint32_t bodyIdx = bodyIdR[indicesBR[k]];
			if (massInvRB[bodyIdx] <= 0.0f || lastContactIdxB_ByBody[bodyIdx] == i) { continue; }
			lastContactIdxB_ByBody[bodyIdx] = i;
			N_contactsB_ByBody[bodyIdx]++;
		}
	}

	// use the largest contact count for boundary particles shared by multiple bodies
	for (uint32_t i = 0; i < N_B; i++)
	{
		if (jacobiOmegaContactB[i] <= 0.0f || diagElemContactB[i] <= 0.0f) { continue; }
		uint32_t N_maxContactsB = 0;
		for (uint64_t k = offsetsBR[i]; k < offsetsBR[i + 1]; k++)
		{
			const uint32_t bodyIdx = bodyIdR[indicesBR[k]];
			N_maxContactsB = std::max(N_maxContactsB, N_contactsB_ByBody[bodyIdx]);
		}
		if (N_maxContactsB > 0) { jacobiOmegaContactB[i] /= N_maxContactsB; }
	}
	return N_contacts;
}

void CPU::RigidContactPressureForces(
	const float* RESTRICT xPosR,
	const float* RESTRICT yPosR,
	const float* RESTRICT zPosR,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT volR,
	const float* RESTRICT volB,
	const float* RESTRICT densityContactR,
	const float* RESTRICT densityContactB,
	const float* RESTRICT pressureContactR,
	const float* RESTRICT pressureContactB,
	const Sim::I_NeighborListSELL& neighborList,
	const RigidBoundaryContactNeighbors& contactNeighbors,
	const float particleSizeInvF,
	const float kernelRadiusSqu,
	const float cubicSplineAlpha,
	const uint32_t N_R,
	// destinations
	float* RESTRICT xForceContactR,
	float* RESTRICT yForceContactR,
	float* RESTRICT zForceContactR)
{
	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N_R),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float fx = 0.0f, fy = 0.0f, fz = 0.0f;
				const float pi = pressureContactR[i] / (densityContactR[i] * densityContactR[i]);
				ForEachContactNeighbor(false, i, neighborList, contactNeighbors,
					[&](const uint32_t j, const bool boundary)
					{
						const uint32_t nbrLocalIdx = boundary ? contactNeighbors.localIdxsB[j] : j;
						float xkg, ykg, zkg;
						const float weight = ContactKernel(
							xPosR[i] - (boundary ? xPosB[nbrLocalIdx] : xPosR[j]),
							yPosR[i] - (boundary ? yPosB[nbrLocalIdx] : yPosR[j]),
							zPosR[i] - (boundary ? zPosB[nbrLocalIdx] : zPosR[j]),
							particleSizeInvF, kernelRadiusSqu, cubicSplineAlpha, xkg, ykg, zkg);
						if (!(weight > 0.0f)) { return; }
						const float rho = boundary ? densityContactB[j] : densityContactR[j];
						const float pj = (boundary ? pressureContactB[j] : pressureContactR[j]) / (rho * rho);
						const float forceCoeff = -RIGID_CONTACT_VOLUME_FACTOR * volR[i]
							* RIGID_CONTACT_VOLUME_FACTOR * (boundary ? volB[nbrLocalIdx] : volR[j]) * (pi + pj);
						fx += forceCoeff * xkg; fy += forceCoeff * ykg; fz += forceCoeff * zkg;
					});
				xForceContactR[i] = fx; yForceContactR[i] = fy; zForceContactR[i] = fz;
			}
		}, tbb::auto_partitioner{});
}

void CPU::RigidContactParticleVelocities(
	const float* RESTRICT xPosRelWorldR,
	const float* RESTRICT yPosRelWorldR,
	const float* RESTRICT zPosRelWorldR,
	const uint32_t* RESTRICT bodyIdR,
	const float* RESTRICT massInvRB,
	const float* RESTRICT iWorldInv00RB,
	const float* RESTRICT iWorldInv01RB,
	const float* RESTRICT iWorldInv02RB,
	const float* RESTRICT iWorldInv10RB,
	const float* RESTRICT iWorldInv11RB,
	const float* RESTRICT iWorldInv12RB,
	const float* RESTRICT iWorldInv20RB,
	const float* RESTRICT iWorldInv21RB,
	const float* RESTRICT iWorldInv22RB,
	const float* RESTRICT xForceContactRB,
	const float* RESTRICT yForceContactRB,
	const float* RESTRICT zForceContactRB,
	const float* RESTRICT xTorqueContactRB,
	const float* RESTRICT yTorqueContactRB,
	const float* RESTRICT zTorqueContactRB,
	const float timeStep,
	const uint32_t N_R,
	// destinations
	float* RESTRICT xVelContactR,
	float* RESTRICT yVelContactR,
	float* RESTRICT zVelContactR)
{
	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N_R),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const uint32_t b = bodyIdR[i];
				const float tx = xTorqueContactRB[b], ty = yTorqueContactRB[b], tz = zTorqueContactRB[b];
				const float wx = iWorldInv00RB[b] * tx + iWorldInv01RB[b] * ty + iWorldInv02RB[b] * tz;
				const float wy = iWorldInv10RB[b] * tx + iWorldInv11RB[b] * ty + iWorldInv12RB[b] * tz;
				const float wz = iWorldInv20RB[b] * tx + iWorldInv21RB[b] * ty + iWorldInv22RB[b] * tz;
				const float rx = xPosRelWorldR[i], ry = yPosRelWorldR[i], rz = zPosRelWorldR[i];
				xVelContactR[i] = timeStep * (massInvRB[b] * xForceContactRB[b] + wy * rz - wz * ry);
				yVelContactR[i] = timeStep * (massInvRB[b] * yForceContactRB[b] + wz * rx - wx * rz);
				zVelContactR[i] = timeStep * (massInvRB[b] * zForceContactRB[b] + wx * ry - wy * rx);
			}
		}, tbb::auto_partitioner{});
}

double CPU::RigidContactDensityError(
	const float* RESTRICT xPosR,
	const float* RESTRICT yPosR,
	const float* RESTRICT zPosR,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT volR,
	const float* RESTRICT volB,
	const float* RESTRICT xVelR,
	const float* RESTRICT yVelR,
	const float* RESTRICT zVelR,
	const float* RESTRICT sourceTermContactR,
	const float* RESTRICT sourceTermContactB,
	const float* RESTRICT pressureContactR,
	const float* RESTRICT pressureContactB,
	const Sim::I_NeighborListSELL& neighborList,
	const RigidBoundaryContactNeighbors& contactNeighbors,
	const float particleSizeInvF,
	const float kernelRadiusSqu,
	const float cubicSplineAlpha,
	const float timeStep,
	const uint32_t N_R,
	// destinations
	float* RESTRICT densityErrorContactR,
	float* RESTRICT densityErrorContactB)
{
	const auto compute = [&](const bool boundary, const uint32_t N) -> double
	{
		const float* xPos = boundary ? xPosB : xPosR;
		const float* yPos = boundary ? yPosB : yPosR;
		const float* zPos = boundary ? zPosB : zPosR;
		const float* source = boundary ? sourceTermContactB : sourceTermContactR;
		const float* pressure = boundary ? pressureContactB : pressureContactR;
		float* error = boundary ? densityErrorContactB : densityErrorContactR;
		return tbb::parallel_reduce(tbb::blocked_range<uint32_t>(0, N), 0.0,
			[&](const tbb::blocked_range<uint32_t>& r, double sum) -> double
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const uint32_t localIdx = boundary ? contactNeighbors.localIdxsB[i] : i;
					const float vx = boundary ? 0.0f : xVelR[i];
					const float vy = boundary ? 0.0f : yVelR[i];
					const float vz = boundary ? 0.0f : zVelR[i];
					float residual = source[i];
					ForEachContactNeighbor(boundary, i, neighborList, contactNeighbors,
						[&](const uint32_t j, const bool neighborBoundary)
						{
							const uint32_t nbrLocalIdx = neighborBoundary ? contactNeighbors.localIdxsB[j] : j;
							float xkg, ykg, zkg;
							const float weight = ContactKernel(
								xPos[localIdx] - (neighborBoundary ? xPosB[nbrLocalIdx] : xPosR[j]),
								yPos[localIdx] - (neighborBoundary ? yPosB[nbrLocalIdx] : yPosR[j]),
								zPos[localIdx] - (neighborBoundary ? zPosB[nbrLocalIdx] : zPosR[j]),
								particleSizeInvF, kernelRadiusSqu, cubicSplineAlpha, xkg, ykg, zkg);
							if (!(weight > 0.0f)) { return; }
							const float dvx = (neighborBoundary ? 0.0f : xVelR[j]) - vx;
							const float dvy = (neighborBoundary ? 0.0f : yVelR[j]) - vy;
							const float dvz = (neighborBoundary ? 0.0f : zVelR[j]) - vz;
							const float divergence = timeStep * (dvx * xkg + dvy * ykg + dvz * zkg);
							residual -= RIGID_CONTACT_VOLUME_FACTOR * (neighborBoundary ? volB[nbrLocalIdx] : volR[j]) * divergence;
						});
					if (!std::isfinite(residual)) { throw std::runtime_error("Non-finite CPU rigid contact density error."); }
					error[i] = residual;
					// also release excess pressure when a contact separates
					sum += pressure[i] > 0.0f ? std::abs(residual) : std::max(residual, 0.0f);
				}
				return sum;
			}, std::plus<double>{}, tbb::auto_partitioner{});
	};
	return compute(false, N_R) + compute(true, static_cast<uint32_t>(contactNeighbors.localIdxsB.size()));
}

void CPU::RigidContactPressureRelaxedJacobi(
	const float* RESTRICT densityErrorContact,
	const float* RESTRICT diagElemContact,
	const float* RESTRICT jacobiOmegaContact,
	const uint32_t N,
	// destinations
	float* RESTRICT pressureContact)
{
	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				if (!(diagElemContact[i] > 0.0f)) { continue; }
				const float pressure = pressureContact[i] + jacobiOmegaContact[i] * densityErrorContact[i] / diagElemContact[i];
				if (!std::isfinite(pressure)) { throw std::runtime_error("Non-finite CPU rigid contact pressure."); }
				pressureContact[i] = std::max(0.0f, pressure);
			}
		}, tbb::auto_partitioner{});
}

void Sim::SimBackend_CPU::SolveRigidContacts(const SimParams& params)
{
	if (!std::isfinite(params.rigidContactDensityDevTarget) ||
		params.rigidContactDensityDevTarget <= 0.0f ||
		params.rigidContactMaxSolverIters == 0)
	{
		throw std::runtime_error("Invalid CPU rigid contact solver tolerance or iteration limit.");
	}
	const bool previouslyConverged = rigidContactReport_.converged;
	rigidContactReport_ = {};
	const bool hasContact = CPU::BuildRigidBoundaryNeighbors(*neighborList_, N_R_, rigidBoundaryContactNeighbors_);
	const uint32_t N_contactB = static_cast<uint32_t>(rigidBoundaryContactNeighbors_.localIdxsB.size());
	densityContactB_.resize(N_contactB);
	diagElemContactB_.resize(N_contactB);
	pressureContactB_.resize(N_contactB);
	sourceTermContactB_.resize(N_contactB);
	densityErrorContactB_.resize(N_contactB);
	jacobiOmegaContactB_.resize(N_contactB);
	if (!hasContact) { return; }

	const SolverConstants constants = BuildSolverConstants(params.particleSizeF, params.restDensity);

	// resize contact work and prepare fixed geometry
	xPosRelWorldR_.resize(N_R_);
	yPosRelWorldR_.resize(N_R_);
	zPosRelWorldR_.resize(N_R_);
	xVelContactR_.resize(N_R_);
	yVelContactR_.resize(N_R_);
	zVelContactR_.resize(N_R_);
	xForceContactR_.resize(N_R_);
	yForceContactR_.resize(N_R_);
	zForceContactR_.resize(N_R_);
	densityContactR_.resize(N_R_);
	diagElemContactR_.resize(N_R_);
	pressureContactR_.resize(N_R_);
	sourceTermContactR_.resize(N_R_);
	densityErrorContactR_.resize(N_R_);
	jacobiOmegaContactR_.resize(N_R_);
	iWorldInv00RB_.resize(N_RB_);
	iWorldInv01RB_.resize(N_RB_);
	iWorldInv02RB_.resize(N_RB_);
	iWorldInv10RB_.resize(N_RB_);
	iWorldInv11RB_.resize(N_RB_);
	iWorldInv12RB_.resize(N_RB_);
	iWorldInv20RB_.resize(N_RB_);
	iWorldInv21RB_.resize(N_RB_);
	iWorldInv22RB_.resize(N_RB_);
	xForceContactRB_.resize(N_RB_);
	yForceContactRB_.resize(N_RB_);
	zForceContactRB_.resize(N_RB_);
	xTorqueContactRB_.resize(N_RB_);
	yTorqueContactRB_.resize(N_RB_);
	zTorqueContactRB_.resize(N_RB_);

	CPU::RigidContactGeometry(
		xPosRelR_.data(), yPosRelR_.data(), zPosRelR_.data(),
		bodyIdR_.data(),
		a00RB_.data(), a01RB_.data(), a02RB_.data(),
		a10RB_.data(), a11RB_.data(), a12RB_.data(),
		a20RB_.data(), a21RB_.data(), a22RB_.data(),
		iBodyInv00RB_.data(), iBodyInv01RB_.data(), iBodyInv02RB_.data(),
		iBodyInv10RB_.data(), iBodyInv11RB_.data(), iBodyInv12RB_.data(),
		iBodyInv20RB_.data(), iBodyInv21RB_.data(), iBodyInv22RB_.data(),
		N_R_,
		N_RB_,
		xPosRelWorldR_.data(), yPosRelWorldR_.data(), zPosRelWorldR_.data(),
		iWorldInv00RB_.data(), iWorldInv01RB_.data(), iWorldInv02RB_.data(),
		iWorldInv10RB_.data(), iWorldInv11RB_.data(), iWorldInv12RB_.data(),
		iWorldInv20RB_.data(), iWorldInv21RB_.data(), iWorldInv22RB_.data());
	CPU::RigidContactDensityAndDiagonal(
		xPosR_.data(), yPosR_.data(), zPosR_.data(),
		xPosB_.data(), yPosB_.data(), zPosB_.data(),
		volR_.data(), volB_.data(),
		xPosRelWorldR_.data(), yPosRelWorldR_.data(), zPosRelWorldR_.data(),
		bodyIdR_.data(),
		massInvRB_.data(),
		iWorldInv00RB_.data(), iWorldInv01RB_.data(), iWorldInv02RB_.data(),
		iWorldInv10RB_.data(), iWorldInv11RB_.data(), iWorldInv12RB_.data(),
		iWorldInv20RB_.data(), iWorldInv21RB_.data(), iWorldInv22RB_.data(),
		*neighborList_,
		rigidBoundaryContactNeighbors_,
		constants.sizeInvF, params.kernelRadiusSqu, constants.cubicSplineAlpha,
		params.timeStep,
		N_R_,
		densityContactR_.data(), densityContactB_.data(),
		diagElemContactR_.data(), diagElemContactB_.data(),
		pressureContactR_.data(), pressureContactB_.data(),
		sourceTermContactR_.data(), sourceTermContactB_.data(),
		jacobiOmegaContactR_.data(), jacobiOmegaContactB_.data());
	const uint64_t N_contacts = CPU::RigidContactRelaxation(
		particleOffsetsR_ByBody_.data(),
		particleLocalIdxsR_ByBody_.data(),
		bodyIdR_.data(), massInvRB_.data(),
		rigidBoundaryContactNeighbors_.offsetsBR.data(), rigidBoundaryContactNeighbors_.indicesBR.data(),
		diagElemContactB_.data(),
		N_RB_,
		N_contactB,
		jacobiOmegaContactR_.data(), jacobiOmegaContactB_.data());

	// source motion includes external and fluid forces, but no contact forces
	PredictRigidBodies(params.timeStep);
	PredictRigidParticleVelocities();
	(void)CPU::RigidContactDensityError(
		xPosR_.data(), yPosR_.data(), zPosR_.data(),
		xPosB_.data(), yPosB_.data(), zPosB_.data(),
		volR_.data(), volB_.data(),
		xVelPredR_.data(), yVelPredR_.data(), zVelPredR_.data(),
		sourceTermContactR_.data(), sourceTermContactB_.data(),
		pressureContactR_.data(), pressureContactB_.data(),
		*neighborList_,
		rigidBoundaryContactNeighbors_,
		constants.sizeInvF, params.kernelRadiusSqu, constants.cubicSplineAlpha,
		params.timeStep,
		N_R_,
		densityErrorContactR_.data(), densityErrorContactB_.data());

	sourceTermContactR_.swap(densityErrorContactR_);
	sourceTermContactB_.swap(densityErrorContactB_);

	for (;;)
	{
		CPU::RigidContactPressureForces(
			xPosR_.data(), yPosR_.data(), zPosR_.data(),
			xPosB_.data(), yPosB_.data(), zPosB_.data(),
			volR_.data(), volB_.data(),
			densityContactR_.data(), densityContactB_.data(),
			pressureContactR_.data(), pressureContactB_.data(),
			*neighborList_, rigidBoundaryContactNeighbors_,
			constants.sizeInvF, params.kernelRadiusSqu, constants.cubicSplineAlpha,
			N_R_,
			xForceContactR_.data(), yForceContactR_.data(), zForceContactR_.data());
		std::ranges::fill(xForceContactRB_, 0.0f);
		std::ranges::fill(yForceContactRB_, 0.0f);
		std::ranges::fill(zForceContactRB_, 0.0f);
		std::ranges::fill(xTorqueContactRB_, 0.0f);
		std::ranges::fill(yTorqueContactRB_, 0.0f);
		std::ranges::fill(zTorqueContactRB_, 0.0f);
		ReduceRigidParticleForcesToBody(RigidForceSource::Contact);
		CPU::RigidContactParticleVelocities(
			xPosRelWorldR_.data(), yPosRelWorldR_.data(), zPosRelWorldR_.data(),
			bodyIdR_.data(),
			massInvRB_.data(),
			iWorldInv00RB_.data(), iWorldInv01RB_.data(), iWorldInv02RB_.data(),
			iWorldInv10RB_.data(), iWorldInv11RB_.data(), iWorldInv12RB_.data(),
			iWorldInv20RB_.data(), iWorldInv21RB_.data(), iWorldInv22RB_.data(),
			xForceContactRB_.data(), yForceContactRB_.data(), zForceContactRB_.data(),
			xTorqueContactRB_.data(), yTorqueContactRB_.data(), zTorqueContactRB_.data(),
			params.timeStep,
			N_R_,
			xVelContactR_.data(), yVelContactR_.data(), zVelContactR_.data());
		const double errorSum = CPU::RigidContactDensityError(
			xPosR_.data(), yPosR_.data(), zPosR_.data(),
			xPosB_.data(), yPosB_.data(), zPosB_.data(),
			volR_.data(), volB_.data(),
			xVelContactR_.data(), yVelContactR_.data(), zVelContactR_.data(),
			sourceTermContactR_.data(), sourceTermContactB_.data(),
			pressureContactR_.data(), pressureContactB_.data(),
			*neighborList_,
			rigidBoundaryContactNeighbors_,
			constants.sizeInvF, params.kernelRadiusSqu, constants.cubicSplineAlpha,
			params.timeStep,
			N_R_,
			densityErrorContactR_.data(), densityErrorContactB_.data());

		rigidContactReport_.avgDensityError = N_contacts > 0 ? static_cast<float>(errorSum / N_contacts) : 0.0f;
		rigidContactReport_.converged = rigidContactReport_.avgDensityError <= params.rigidContactDensityDevTarget;
		if (rigidContactReport_.converged || rigidContactReport_.N_solverIters == params.rigidContactMaxSolverIters) { break; }
		CPU::RigidContactPressureRelaxedJacobi(
			densityErrorContactR_.data(),
			diagElemContactR_.data(),
			jacobiOmegaContactR_.data(),
			N_R_,
			pressureContactR_.data());
		CPU::RigidContactPressureRelaxedJacobi(
			densityErrorContactB_.data(),
			diagElemContactB_.data(),
			jacobiOmegaContactB_.data(),
			N_contactB,
			pressureContactB_.data());
		rigidContactReport_.N_solverIters++;
	}

	// add the final contact loads once
	for (uint32_t bodyIdx = 0; bodyIdx < N_RB_; bodyIdx++)
	{
		xForceRB_[bodyIdx] += xForceContactRB_[bodyIdx];
		yForceRB_[bodyIdx] += yForceContactRB_[bodyIdx];
		zForceRB_[bodyIdx] += zForceContactRB_[bodyIdx];
		xTorqueRB_[bodyIdx] += xTorqueContactRB_[bodyIdx];
		yTorqueRB_[bodyIdx] += yTorqueContactRB_[bodyIdx];
		zTorqueRB_[bodyIdx] += zTorqueContactRB_[bodyIdx];
	}
	if (!rigidContactReport_.converged && previouslyConverged)
	{
		SPDLOG_WARN("CPU rigid contact solver reached {} iterations with density error {} (target {}).",
			rigidContactReport_.N_solverIters, rigidContactReport_.avgDensityError, params.rigidContactDensityDevTarget);
	}
}
