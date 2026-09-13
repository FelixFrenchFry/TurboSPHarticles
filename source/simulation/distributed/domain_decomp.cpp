// domain_decomp.cpp

// external
#include <memory>
#include <stdexcept>

// internal
#include "domain_decomp.h"
#include "layouts/decomp_aabb_columns.h"
#include "layouts/decomp_aabb_slabs.h"



namespace Dist
{
	void IDomainDecomp::Restore(const DomainDecompInput&, std::span<const AABB>)
	{
		throw std::runtime_error("Selected domain decomp does not support checkpoint restart.");
	}

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
		}

		throw std::runtime_error("Unsupported domain decomp method requested.");
	}

} // namespace Dist
