// common.cuh
//
// TODO

#pragma once

// external
#include <thrust/tuple.h>



namespace GPU01
{
	// host+device struct for AABB accumulation
	struct MinMax3HD
	{
		float xMin, yMin, zMin;
		float xMax, yMax, zMax;
	};

	// map (x, y, z) to MinMax3HD
	struct ToMinMax3
	{
		__host__ __device__
		MinMax3HD operator()(const thrust::tuple<float, float, float>& t) const
		{
			const float x = thrust::get<0>(t);
			const float y = thrust::get<1>(t);
			const float z = thrust::get<2>(t);

			return MinMax3HD{ x, y, z, x, y, z };
		}
	};

	// reduce two MinMax3HDs to their combined min/max
	struct ReduceMinMax3
	{
		__host__ __device__
		MinMax3HD operator()(const MinMax3HD& a, const MinMax3HD& b) const
		{
			MinMax3HD r;

			r.xMin = (a.xMin < b.xMin) ? a.xMin : b.xMin;
			r.yMin = (a.yMin < b.yMin) ? a.yMin : b.yMin;
			r.zMin = (a.zMin < b.zMin) ? a.zMin : b.zMin;
			r.xMax = (a.xMax > b.xMax) ? a.xMax : b.xMax;
			r.yMax = (a.yMax > b.yMax) ? a.yMax : b.yMax;
			r.zMax = (a.zMax > b.zMax) ? a.zMax : b.zMax;

			return r;
		}
	};

} // namespace GPU01
