// domain_decomp.cpp

// external
#include <memory>
#include <stdexcept>

// internal
#include "domain_decomp.h"
#include "Layouts/decomp_aabb_dynamic.h"
#include "Layouts/decomp_aabb_uniform.h"



namespace Dist
{
	std::unique_ptr<IDomainDecomp> MakeDomainDecomp(const Base::DomainDecompMethod method)
	{
		switch (method)
		{
		case Base::DomainDecompMethod::AABB_Uniform:
		{
			return std::make_unique<Decomp_AABB_Uniform>();
		}
		case Base::DomainDecompMethod::AABB_Dynamic:
		{
			return std::make_unique<Decomp_AABB_Dynamic>();
		}
		}

		throw std::runtime_error("Unsupported domain decomp method requested.");
	}

} // namespace Dist
