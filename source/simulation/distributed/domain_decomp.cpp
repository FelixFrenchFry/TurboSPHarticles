// domain_decomp.cpp

// external
#include <memory>
#include <stdexcept>

// internal
#include "domain_decomp.h"
#include "layouts/decomp_aabb_columns.h"
#include "layouts/decomp_aabb_orb.h"
#include "layouts/decomp_aabb_slabs.h"



namespace Dist
{
	std::unique_ptr<IDomainDecomp> MakeDomainDecomp(const Base::DomainDecompMethod method)
	{
		switch (method)
		{
		case Base::DomainDecompMethod::AABB_Slabs:
		{
			return std::make_unique<Decomp_AABB_Slabs>();
		}
		case Base::DomainDecompMethod::AABB_Columns:
		{
			return std::make_unique<Decomp_AABB_Columns>();
		}
		case Base::DomainDecompMethod::AABB_Orb:
		{
			return std::make_unique<Decomp_AABB_Orb>();
		}
		}

		throw std::runtime_error("Unsupported domain decomp method requested.");
	}

} // namespace Dist
