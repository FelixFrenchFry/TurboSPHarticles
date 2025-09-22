// backend_sim.cpp

// external
#include <stdexcept>

// internal
#include "backend_sim.h"
#include "Source/Sim/CPU/CPU_Backend_01/cpu_backend_01.h"



namespace Sim
{
	std::unique_ptr<ISimBackend> MakeSimBackend(Base::BackendVersion version, const SimParams& params)
	{
		switch (version)
		{
		case Base::BackendVersion::CPU_01:
			{
				return std::make_unique<SimBackendCPU_01>(params);
			}
		default:
			{
				throw std::runtime_error("Unsupported sim backend version request");
			}
		}
	}

} // namespace Sim
