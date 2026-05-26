// simd8.h

#pragma once

// external
#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <xsimd/xsimd.hpp>



namespace CPU
{
	constexpr uint32_t N_SIMD_WIDTH = 8u;
	constexpr float ZERO_EQU_LIM_F = 1e-7f;

	// 8-wide CPU SIMD types used by SELL8-style kernels
	using SimdArch = xsimd::avx2;
	using Float32x8 = xsimd::batch<float, SimdArch>;
	using Int32x8 = xsimd::batch<int32_t, SimdArch>;
	using UInt32x8 = xsimd::batch<uint32_t, SimdArch>;
	using Mask32x8 = xsimd::batch_bool<float, SimdArch>;
	using UIntMask32x8 = xsimd::batch_bool<uint32_t, SimdArch>;

	static_assert(Float32x8::size == N_SIMD_WIDTH);
	static_assert(Int32x8::size == N_SIMD_WIDTH);
	static_assert(UInt32x8::size == N_SIMD_WIDTH);

	// broadcast one scalar float to all 8 SIMD lanes
	inline Float32x8 SetFloat32x8(
		const float v) noexcept
	{
		return xsimd::broadcast<float, SimdArch>(v);
	}

	// create a float SIMD vector with all lanes set to zero
	inline Float32x8 ZeroFloat32x8() noexcept
	{
		return xsimd::broadcast<float, SimdArch>(0.0f);
	}

	// load 8 contiguous float values from unaligned memory
	inline Float32x8 LoadFloat32x8(
		const float* p) noexcept
	{
		return xsimd::load_unaligned<SimdArch>(p);
	}

	// load 8 contiguous uint32 values from unaligned memory
	inline UInt32x8 LoadUInt32x8(
		const uint32_t* p) noexcept
	{
		return xsimd::load_unaligned<SimdArch>(p);
	}

	// build 8 particle indices for one SIMD chunk, padding the last partial chunk safely
	inline UInt32x8 ParticleIdx32x8(
		const uint32_t begin,
		const uint32_t end) noexcept
	{
		alignas(32) uint32_t idx[N_SIMD_WIDTH]{};
		const uint32_t last = end - 1u;

		for (uint32_t lane = 0; lane < N_SIMD_WIDTH; lane++)
		{
			idx[lane] = std::min(begin + lane, last);
		}

		return LoadUInt32x8(idx);
	}

	// gather 8 scattered float values by particle index
	inline Float32x8 GatherFloat32x8(
		const float* p,
		const UInt32x8 idx) noexcept
	{
		return xsimd::batch<float, SimdArch>::gather(p, idx);
	}

	// fused multiply-add for hot inner-loop arithmetic
	inline Float32x8 FmaFloat32x8(
		const Float32x8 a,
		const Float32x8 b,
		const Float32x8 c) noexcept
	{
	#if defined(__FMA__) || defined(_MSC_VER)
		return Float32x8(_mm256_fmadd_ps(
			static_cast<__m256>(a),
			static_cast<__m256>(b),
			static_cast<__m256>(c)));
	#else
		return Float32x8(_mm256_add_ps(
			_mm256_mul_ps(static_cast<__m256>(a), static_cast<__m256>(b)),
			static_cast<__m256>(c)));
	#endif
	}

	// cast a uint32 lane mask to a float lane mask for float select operations
	inline Mask32x8 SelectMask32x8(
		const UIntMask32x8 mask) noexcept
	{
		return xsimd::batch_bool_cast<float>(mask);
	}

	// return the valid lanes for neighbor-column k in a SELL8 chunk
	inline Mask32x8 MaskKLessThanCounts32x8(
		const uint32_t k,
		const uint32_t* counts) noexcept
	{
		const __m256i countV = static_cast<__m256i>(LoadUInt32x8(counts));
		const __m256i kV = _mm256_set1_epi32(static_cast<int>(k));
		return Mask32x8(_mm256_castsi256_ps(_mm256_cmpgt_epi32(countV, kV)));
	}

	// select per lane between two float SIMD vectors
	inline Float32x8 SelectFloat32x8(
		const Mask32x8 mask,
		const Float32x8 a,
		const Float32x8 b) noexcept
	{
		return xsimd::select(mask, a, b);
	}

	// keep only lanes selected by a float SIMD mask
	inline Float32x8 AndMaskedFloat32x8(
		const Float32x8 v,
		const Mask32x8 mask) noexcept
	{
		return Float32x8(_mm256_and_ps(
			static_cast<__m256>(v),
			static_cast<__m256>(mask)));
	}

	// keep valid lanes and set masked-out lanes to zero
	inline Float32x8 ZeroMaskedFloat32x8(
		const Float32x8 v,
		const Mask32x8 mask) noexcept
	{
		return AndMaskedFloat32x8(v, mask);
	}

	// take the absolute value by clearing the sign bit
	inline Float32x8 AbsBitwiseFloat32x8(
		const Float32x8 v) noexcept
	{
		return Float32x8(_mm256_andnot_ps(
			_mm256_set1_ps(-0.0f),
			static_cast<__m256>(v)));
	}

	// take the absolute value of all 8 float lanes
	inline Float32x8 AbsFloat32x8(
		const Float32x8 v) noexcept
	{
		return AbsBitwiseFloat32x8(v);
	}

	// return the low 8 lane bits of a float SIMD mask
	inline uint32_t MaskBits32x8(
		const Mask32x8 mask) noexcept
	{
		return static_cast<uint32_t>(_mm256_movemask_ps(static_cast<__m256>(mask)));
	}

	// store a full SIMD chunk or only the valid lanes of the final partial chunk
	inline void StoreFloat32x8Partial(
		float* dst,
		const uint32_t begin,
		const uint32_t end,
		const Float32x8 v) noexcept
	{
		if (end - begin == N_SIMD_WIDTH)
		{
			v.store_unaligned(dst + begin);
			return;
		}

		alignas(32) float tmp[N_SIMD_WIDTH]{};
		v.store_aligned(tmp);
		for (uint32_t i = begin; i < end; i++)
		{
			dst[i] = tmp[i - begin];
		}
	}

	// compute cubic-spline kernel weights for 8 particle-pair offsets
	inline Float32x8 KernelWeight_CubicSpline_SIMD8(
		const Float32x8 dx,
		const Float32x8 dy,
		const Float32x8 dz,
		const float particleSizeInv,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha) noexcept
	{
		const Float32x8 zero = xsimd::broadcast<float, SimdArch>(0.0f);
		const Float32x8 one = xsimd::broadcast<float, SimdArch>(1.0f);
		const Float32x8 two = xsimd::broadcast<float, SimdArch>(2.0f);
		const Float32x8 four = xsimd::broadcast<float, SimdArch>(4.0f);
		const Float32x8 eps = xsimd::broadcast<float, SimdArch>(ZERO_EQU_LIM_F);
		const Float32x8 sizeInv = xsimd::broadcast<float, SimdArch>(particleSizeInv);
		const Float32x8 radiusSqu = xsimd::broadcast<float, SimdArch>(kernelRadiusSqu);
		const Float32x8 alpha = xsimd::broadcast<float, SimdArch>(cubicSplineAlpha);

		const Float32x8 distSqu = xsimd::fma(dx, dx, xsimd::fma(dy, dy, dz * dz));
		const Mask32x8 inside = distSqu < radiusSqu;
		const Mask32x8 zeroDist = distSqu < eps;
		const Float32x8 dist = xsimd::sqrt(xsimd::max(distSqu, eps));
		const Float32x8 distNorm = dist * sizeInv;

		const Float32x8 t1 = xsimd::max(one - distNorm, zero);
		const Float32x8 t2 = xsimd::max(two - distNorm, zero);
		const Float32x8 t1_2 = t1 * t1;
		const Float32x8 t2_2 = t2 * t2;
		const Float32x8 t1_3 = t1_2 * t1;
		const Float32x8 t2_3 = t2_2 * t2;
		const Float32x8 kw = alpha * (t2_3 - four * t1_3);
		const Float32x8 kwZeroDist = four * alpha;
		const Float32x8 kwFinal = SelectFloat32x8(zeroDist, kwZeroDist, kw);

		return ZeroMaskedFloat32x8(kwFinal, inside);
	}

	// compute cubic-spline kernel gradients for 8 particle-pair offsets
	inline void KernelGradient_CubicSpline_SIMD8(
		const Float32x8 dx,
		const Float32x8 dy,
		const Float32x8 dz,
		const float particleSizeInv,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		// destinations
		Float32x8& xkg,
		Float32x8& ykg,
		Float32x8& zkg) noexcept
	{
		const Float32x8 zero = xsimd::broadcast<float, SimdArch>(0.0f);
		const Float32x8 one = xsimd::broadcast<float, SimdArch>(1.0f);
		const Float32x8 two = xsimd::broadcast<float, SimdArch>(2.0f);
		const Float32x8 three = xsimd::broadcast<float, SimdArch>(3.0f);
		const Float32x8 twelve = xsimd::broadcast<float, SimdArch>(12.0f);
		const Float32x8 eps = xsimd::broadcast<float, SimdArch>(ZERO_EQU_LIM_F);
		const Float32x8 sizeInv = xsimd::broadcast<float, SimdArch>(particleSizeInv);
		const Float32x8 radiusSqu = xsimd::broadcast<float, SimdArch>(kernelRadiusSqu);
		const Float32x8 alpha = xsimd::broadcast<float, SimdArch>(cubicSplineAlpha);

		const Float32x8 distSqu = xsimd::fma(dx, dx, xsimd::fma(dy, dy, dz * dz));
		const Mask32x8 inside = distSqu < radiusSqu;
		const Mask32x8 nonZero = distSqu >= eps;
		const Mask32x8 valid = inside && nonZero;
		const Float32x8 dist = xsimd::sqrt(xsimd::max(distSqu, eps));
		const Float32x8 distNorm = dist * sizeInv;

		const Float32x8 t1 = xsimd::max(one - distNorm, zero);
		const Float32x8 t2 = xsimd::max(two - distNorm, zero);
		const Float32x8 t1_2 = t1 * t1;
		const Float32x8 t2_2 = t2 * t2;
		const Float32x8 commonFac = alpha
			* ((zero - three) * t2_2 + twelve * t1_2)
			* (sizeInv / dist);

		xkg = ZeroMaskedFloat32x8(commonFac * dx, valid);
		ykg = ZeroMaskedFloat32x8(commonFac * dy, valid);
		zkg = ZeroMaskedFloat32x8(commonFac * dz, valid);
	}

} // namespace CPU
