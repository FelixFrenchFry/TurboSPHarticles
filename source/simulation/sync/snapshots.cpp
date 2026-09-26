// snapshots.cpp

// external
#include <stdexcept>
#include <utility>

// internal
#include "snapshots.h"



namespace Sync
{
	void CollectBoundaryMeshParticles(
		const BoundaryConstView& particlesB,
		const std::optional<uint64_t> meshIdBeginB,
		std::shared_ptr<const std::vector<glm::vec3>>& meshPosB,
		std::shared_ptr<const std::vector<float>>& meshVolB)
	{
		if (!meshIdBeginB)
		{
			meshPosB.reset();
			meshVolB.reset();
			return;
		}
		const uint32_t N_B = particlesB.N_;
		if (N_B > particlesB.id.size() ||
			N_B > particlesB.xPos.size() || N_B > particlesB.yPos.size() ||
			N_B > particlesB.zPos.size() || N_B > particlesB.volB.size())
		{
			throw std::runtime_error("Invalid boundary range during mesh snapshot packing.");
		}

		// boundary meshes occupy the final global boundary ID range
		auto posB = std::make_shared<std::vector<glm::vec3>>();
		auto volB = std::make_shared<std::vector<float>>();
		for (uint32_t i = 0; i < N_B; i++)
		{
			if (particlesB.id[i] == Sim::INVALID_PARTICLE_ID)
			{
				throw std::runtime_error("Invalid boundary particle during mesh snapshot packing.");
			}
			if (particlesB.id[i] < *meshIdBeginB) { continue; }
			posB->push_back({ particlesB.xPos[i], particlesB.yPos[i], particlesB.zPos[i] });
			volB->push_back(particlesB.volB[i]);
		}
		meshPosB = std::move(posB);
		meshVolB = std::move(volB);
	}

} // namespace Sync
