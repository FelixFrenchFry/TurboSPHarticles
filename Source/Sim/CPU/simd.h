// simd.h

#pragma once

// external
#include <algorithm>
#include <cstdint>
#include <xsimd/xsimd.hpp>

// internal
#if defined(TSP_CPU_SIMD_ARCH_AVX512)
#include "Source/Sim/CPU/simd_avx512.h"
#elif defined(TSP_CPU_SIMD_ARCH_AVX256) || !defined(TSP_CPU_SIMD_WIDTH)
#include "Source/Sim/CPU/simd_avx256.h"
#else
#error "Unsupported CPU SIMD architecture. Expected AVX256 or AVX512."
#endif



namespace CPU
{
	constexpr float ZERO_EQU_LIM_F = 1e-7f;

	static_assert(Float32xSIMD::size == N_SIMD_WIDTH);
	static_assert(Int32xSIMD::size == N_SIMD_WIDTH);
	static_assert(UInt32xSIMD::size == N_SIMD_WIDTH);

#if defined(TSP_CPU_SIMD_WIDTH)
	static_assert(TSP_CPU_SIMD_WIDTH == N_SIMD_WIDTH);
#endif

	// broadcast one scalar float to all SIMD lanes
	inline Float32xSIMD SetFloat32xSIMD(
		const float v) noexcept
	{
		return xsimd::broadcast<float, SimdArch>(v);
	}

	// broadcast one scalar uint32 to all SIMD lanes
	inline UInt32xSIMD SetUInt32xSIMD(
		const uint32_t v) noexcept
	{
		return xsimd::broadcast<uint32_t, SimdArch>(v);
	}

	// create a float SIMD vector with all lanes set to zero
	inline Float32xSIMD ZeroFloat32xSIMD() noexcept
	{
		return xsimd::broadcast<float, SimdArch>(0.0f);
	}

	// load contiguous float values from unaligned memory
	inline Float32xSIMD LoadFloat32xSIMD(
		const float* p) noexcept
	{
		return xsimd::load_unaligned<SimdArch>(p);
	}

	// load contiguous uint32 values from unaligned memory
	inline UInt32xSIMD LoadUInt32xSIMD(
		const uint32_t* p) noexcept
	{
		return xsimd::load_unaligned<SimdArch>(p);
	}

	// build particle indices for one SIMD chunk, padding the last partial chunk safely
	inline UInt32xSIMD ParticleIdx32xSIMD(
		const uint32_t begin,
		const uint32_t end) noexcept
	{
		alignas(N_SIMD_ALIGNMENT) uint32_t idx[N_SIMD_WIDTH]{};
		const uint32_t last = end - 1u;

		for (uint32_t lane = 0; lane < N_SIMD_WIDTH; lane++)
		{
			idx[lane] = std::min(begin + lane, last);
		}

		return LoadUInt32xSIMD(idx);
	}

	// gather scattered float values by particle index
	inline Float32xSIMD GatherFloat32xSIMD(
		const float* p,
		const UInt32xSIMD idx) noexcept
	{
		return Float32xSIMD::gather(p, idx);
	}

	// fused multiply-add for hot inner-loop arithmetic
	inline Float32xSIMD FmaFloat32xSIMD(
		const Float32xSIMD a,
		const Float32xSIMD b,
		const Float32xSIMD c) noexcept
	{
		return xsimd::fma(a, b, c);
	}

	// cast a uint32 lane mask to a float lane mask for float select operations
	inline Mask32xSIMD SelectMask32xSIMD(
		const UIntMask32xSIMD mask) noexcept
	{
		return xsimd::batch_bool_cast<float>(mask);
	}

	// return the valid lanes for neighbor-column k in a SELL chunk
	inline Mask32xSIMD MaskKLessThanCounts32xSIMD(
		const uint32_t k,
		const UInt32xSIMD countV) noexcept
	{
		return SelectMask32xSIMD(countV > SetUInt32xSIMD(k));
	}

	// return the valid lanes for neighbor-column k in a SELL chunk
	inline Mask32xSIMD MaskKLessThanCounts32xSIMD(
		const uint32_t k,
		const uint32_t* counts) noexcept
	{
		return MaskKLessThanCounts32xSIMD(k, LoadUInt32xSIMD(counts));
	}

	// return the lanes a partial chunk covers, so its padding lanes can be masked in registers
	inline Mask32xSIMD ValidLaneMask32xSIMD(
		const uint32_t begin,
		const uint32_t end) noexcept
	{
		alignas(N_SIMD_ALIGNMENT) uint32_t lanes[N_SIMD_WIDTH]{};

		for (uint32_t lane = 0; lane < N_SIMD_WIDTH; lane++)
		{
			lanes[lane] = lane;
		}

		return SelectMask32xSIMD(LoadUInt32xSIMD(lanes) < SetUInt32xSIMD(end - begin));
	}

	// select per lane between two float SIMD vectors
	inline Float32xSIMD SelectFloat32xSIMD(
		const Mask32xSIMD mask,
		const Float32xSIMD a,
		const Float32xSIMD b) noexcept
	{
		return xsimd::select(mask, a, b);
	}

	// keep only lanes selected by a float SIMD mask
	inline Float32xSIMD AndMaskedFloat32xSIMD(
		const Float32xSIMD v,
		const Mask32xSIMD mask) noexcept
	{
		return SelectFloat32xSIMD(mask, v, ZeroFloat32xSIMD());
	}

	// keep valid lanes and set masked-out lanes to zero
	inline Float32xSIMD ZeroMaskedFloat32xSIMD(
		const Float32xSIMD v,
		const Mask32xSIMD mask) noexcept
	{
		return AndMaskedFloat32xSIMD(v, mask);
	}

	// add one to every uint32 lane selected by a float lane mask, keeping counters in registers
	inline UInt32xSIMD AddMaskedOneUInt32xSIMD(
		const UInt32xSIMD v,
		const Mask32xSIMD mask) noexcept
	{
		const UIntMask32xSIMD maskU = xsimd::batch_bool_cast<uint32_t>(mask);

		return v + xsimd::select(maskU, SetUInt32xSIMD(1u), SetUInt32xSIMD(0u));
	}

	// take the absolute value by clearing the sign bit
	inline Float32xSIMD AbsBitwiseFloat32xSIMD(
		const Float32xSIMD v) noexcept
	{
		return xsimd::abs(v);
	}

	// take the absolute value of all float lanes
	inline Float32xSIMD AbsFloat32xSIMD(
		const Float32xSIMD v) noexcept
	{
		return AbsBitwiseFloat32xSIMD(v);
	}

	// return the low lane bits of a float SIMD mask
	inline uint32_t MaskBits32xSIMD(
		const Mask32xSIMD mask) noexcept
	{
		return static_cast<uint32_t>(mask.mask());
	}

	// store a full SIMD chunk or only the valid lanes of the final partial chunk
	inline void StoreFloat32xSIMDPartial(
		float* dst,
		const uint32_t begin,
		const uint32_t end,
		const Float32xSIMD v) noexcept
	{
		if (end - begin == N_SIMD_WIDTH)
		{
			v.store_unaligned(dst + begin);
			return;
		}

		alignas(N_SIMD_ALIGNMENT) float tmp[N_SIMD_WIDTH]{};
		v.store_aligned(tmp);
		for (uint32_t i = begin; i < end; i++)
		{
			dst[i] = tmp[i - begin];
		}
	}

	// compute cubic-spline kernel weights for SIMD particle-pair offsets
	inline Float32xSIMD KernelWeight_CubicSpline_SIMD(
		const Float32xSIMD dx,
		const Float32xSIMD dy,
		const Float32xSIMD dz,
		const float particleSizeInv,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha) noexcept
	{
		const Float32xSIMD zero = xsimd::broadcast<float, SimdArch>(0.0f);
		const Float32xSIMD one = xsimd::broadcast<float, SimdArch>(1.0f);
		const Float32xSIMD two = xsimd::broadcast<float, SimdArch>(2.0f);
		const Float32xSIMD four = xsimd::broadcast<float, SimdArch>(4.0f);
		const Float32xSIMD eps = xsimd::broadcast<float, SimdArch>(ZERO_EQU_LIM_F);
		const Float32xSIMD sizeInv = xsimd::broadcast<float, SimdArch>(particleSizeInv);
		const Float32xSIMD radiusSqu = xsimd::broadcast<float, SimdArch>(kernelRadiusSqu);
		const Float32xSIMD alpha = xsimd::broadcast<float, SimdArch>(cubicSplineAlpha);

		const Float32xSIMD distSqu = xsimd::fma(dx, dx, xsimd::fma(dy, dy, dz * dz));
		const Mask32xSIMD inside = distSqu < radiusSqu;
		const Mask32xSIMD zeroDist = distSqu < eps;
		const Float32xSIMD dist = xsimd::sqrt(xsimd::max(distSqu, eps));
		const Float32xSIMD distNorm = dist * sizeInv;

		const Float32xSIMD t1 = xsimd::max(one - distNorm, zero);
		const Float32xSIMD t2 = xsimd::max(two - distNorm, zero);
		const Float32xSIMD t1_2 = t1 * t1;
		const Float32xSIMD t2_2 = t2 * t2;
		const Float32xSIMD t1_3 = t1_2 * t1;
		const Float32xSIMD t2_3 = t2_2 * t2;
		const Float32xSIMD kw = alpha * (t2_3 - four * t1_3);
		const Float32xSIMD kwZeroDist = four * alpha;
		const Float32xSIMD kwFinal = SelectFloat32xSIMD(zeroDist, kwZeroDist, kw);

		return ZeroMaskedFloat32xSIMD(kwFinal, inside);
	}

	// compute cubic-spline kernel gradients for SIMD particle-pair offsets
	inline void KernelGradient_CubicSpline_SIMD(
		const Float32xSIMD dx,
		const Float32xSIMD dy,
		const Float32xSIMD dz,
		const float particleSizeInv,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		// destinations
		Float32xSIMD& xkg,
		Float32xSIMD& ykg,
		Float32xSIMD& zkg) noexcept
	{
		const Float32xSIMD zero = xsimd::broadcast<float, SimdArch>(0.0f);
		const Float32xSIMD one = xsimd::broadcast<float, SimdArch>(1.0f);
		const Float32xSIMD two = xsimd::broadcast<float, SimdArch>(2.0f);
		const Float32xSIMD three = xsimd::broadcast<float, SimdArch>(3.0f);
		const Float32xSIMD twelve = xsimd::broadcast<float, SimdArch>(12.0f);
		const Float32xSIMD eps = xsimd::broadcast<float, SimdArch>(ZERO_EQU_LIM_F);
		const Float32xSIMD sizeInv = xsimd::broadcast<float, SimdArch>(particleSizeInv);
		const Float32xSIMD radiusSqu = xsimd::broadcast<float, SimdArch>(kernelRadiusSqu);
		const Float32xSIMD alpha = xsimd::broadcast<float, SimdArch>(cubicSplineAlpha);

		const Float32xSIMD distSqu = xsimd::fma(dx, dx, xsimd::fma(dy, dy, dz * dz));
		const Mask32xSIMD inside = distSqu < radiusSqu;
		const Mask32xSIMD nonZero = distSqu >= eps;
		const Mask32xSIMD valid = inside && nonZero;
		const Float32xSIMD dist = xsimd::sqrt(xsimd::max(distSqu, eps));
		const Float32xSIMD distNorm = dist * sizeInv;

		const Float32xSIMD t1 = xsimd::max(one - distNorm, zero);
		const Float32xSIMD t2 = xsimd::max(two - distNorm, zero);
		const Float32xSIMD t1_2 = t1 * t1;
		const Float32xSIMD t2_2 = t2 * t2;
		const Float32xSIMD commonFac = alpha
			* ((zero - three) * t2_2 + twelve * t1_2)
			* (sizeInv / dist);

		xkg = ZeroMaskedFloat32xSIMD(commonFac * dx, valid);
		ykg = ZeroMaskedFloat32xSIMD(commonFac * dy, valid);
		zkg = ZeroMaskedFloat32xSIMD(commonFac * dz, valid);
	}

} // namespace CPU
