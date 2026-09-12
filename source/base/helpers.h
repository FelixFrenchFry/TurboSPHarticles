// helpers.h
#pragma once

// external
#include <cmath>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>



namespace Base
{
    // --- [ approximate equality helpers for detecting parameter changes ] -----------------------

    // float comparison
    inline bool ApproxEqu(const float a, const float b, const float eps = 1e-7f)
	{
        return std::fabs(a - b) <= eps;
    }

    // double comparison
    inline bool ApproxEqu(const double a, const double b, const double eps = 1e-14)
    {
        return std::fabs(a - b) <= eps;
    }

    // element-wise vec2 comparison
    inline bool ApproxEqu(const glm::vec2& a, const glm::vec2& b, const float eps = 1e-7f)
	{
        return ApproxEqu(a.x, b.x, eps) &&
               ApproxEqu(a.y, b.y, eps);
    }

	// element-wise vec3 comparison
    inline bool ApproxEqu(const glm::vec3& a, const glm::vec3& b, const float eps = 1e-7f)
	{
        return ApproxEqu(a.x, b.x, eps) &&
               ApproxEqu(a.y, b.y, eps) &&
               ApproxEqu(a.z, b.z, eps);
    }

    // element-wise vec4 comparison
    inline bool ApproxEqu(const glm::vec4& a, const glm::vec4& b, const float eps = 1e-7f)
	{
        return ApproxEqu(a.x, b.x, eps) &&
               ApproxEqu(a.y, b.y, eps) &&
               ApproxEqu(a.z, b.z, eps) &&
               ApproxEqu(a.w, b.w, eps);
    }

    // element-wise mat4 comparison
    inline bool ApproxEqu(const glm::mat4& a, const glm::mat4& b, const float eps = 1e-7f)
	{
        float maxDiff = 0.0f;

        for (int c = 0; c < 4; c++)
        {
            for (int r = 0; r < 4; r++)
            {
                maxDiff = fmaxf(maxDiff, std::fabs(a[c][r] - b[c][r]));
            }
        }

        return maxDiff <= eps;
    }

    // fallback for exact types (int, bool, enums, etc...)
    template <class T>
	inline bool ApproxEquDefault(const T& a, const T& b) { return a == b; }

    // helpers for std::optional fields
    template <class T, class Eq>
    inline void MarkIfChanged(const std::optional<T>& incoming, const T& current, bool& changed, Eq eq)
	{
        if (incoming && !eq(*incoming, current)) changed = true;
    }
    template <class T>
    inline void MarkIfChanged(const std::optional<T>& incoming, const T& current, bool& changed)
	{
        MarkIfChanged(incoming, current, changed, ApproxEquDefault<T>);
    }

	// overloads that select the right comparator automatically
    inline void MarkIfChanged(const std::optional<int32_t>& old, const int32_t& cur, bool& ch)
    {
        if (old && old != cur) { ch = true; }
    }
    inline void MarkIfChanged(const std::optional<int64_t>& old, const int64_t& cur, bool& ch)
    {
        if (old && old != cur) { ch = true; }
    }
    inline void MarkIfChanged(const std::optional<uint32_t>& old, const uint32_t& cur, bool& ch)
    {
        if (old && old != cur) { ch = true; }
    }
    inline void MarkIfChanged(const std::optional<uint64_t>& old, const uint64_t& cur, bool& ch)
    {
        if (old && old != cur) { ch = true; }
    }
    inline void MarkIfChanged(const std::optional<float>& old, const float& cur, bool& ch)
    {
	    if (old && !ApproxEqu(*old, cur)) { ch = true; }
    }
    inline void MarkIfChanged(const std::optional<glm::vec2>& old, const glm::vec2& cur, bool& ch)
    {
	    if (old && !ApproxEqu(*old, cur)) { ch = true; }
    }
    inline void MarkIfChanged(const std::optional<glm::vec3>& old, const glm::vec3& cur, bool& ch)
    {
	    if (old && !ApproxEqu(*old, cur)) { ch = true; }
    }
    inline void MarkIfChanged(const std::optional<glm::vec4>& old, const glm::vec4& cur, bool& ch)
    {
	    if (old && !ApproxEqu(*old, cur)) { ch = true; }
    }
    inline void MarkIfChanged(const std::optional<glm::mat4>& old, const glm::mat4& cur, bool& ch)
    {
	    if (old && !ApproxEqu(*old, cur)) { ch = true; }
    }

    // --- [ json ] -------------------------------------------------------------------------------

    // relative config paths under TSP_CONFIG_DIR
    std::string ParamsPathForProfile(const std::string& profile);
    std::string ScenariosPathForProfile(const std::string& profile);

    bool LoadJsonObjectFromFile(
        const std::string& fileName,
        nlohmann::json& out);

    // --- [ misc ] -------------------------------------------------------------------------------

    inline bool Float_AlmostEqual(const float a, const float b,
                                  const float rel_epsilon = 1e-7f,
                                  const float abs_epsilon = 1e-7f) noexcept
    {
        const float diff = std::fabs(a - b);
        if (diff <= abs_epsilon) { return true; }

        return diff <= rel_epsilon * fmaxf(std::fabs(a), std::fabs(b));
    }

    constexpr uint64_t ExtraPercent(const uint64_t n, const float pct) noexcept
	{
		if (pct <= 0.0f || ApproxEqu(pct, 0.0f)) { return 0u; }

        const float extra = static_cast<float>(n) * pct / 100.0f;
        if (!std::isfinite(extra) ||
            extra >= static_cast<float>(std::numeric_limits<uint64_t>::max()))
        {
            return std::numeric_limits<uint64_t>::max();
        }
        const uint64_t roundedDown = static_cast<uint64_t>(extra);

        return roundedDown + (static_cast<float>(roundedDown) < extra ? 1u : 0u);
    }

    // reserve scratch storage without having to preserve its contents if capacity grows
    template <class T>
    inline void ReserveScratchVector(
        std::vector<T>& values,
        const uint64_t N_elements,
        const float extraAllocCapacityPct = 0.0f)
    {
        if (values.capacity() < N_elements)
        {
            const uint64_t extra = ExtraPercent(N_elements, extraAllocCapacityPct);
            if (extra > std::numeric_limits<uint64_t>::max() - N_elements)
            {
                throw std::length_error("Scratch vector reserve size overflow.");
            }
            const uint64_t capacityTarget = N_elements + extra;
            if (capacityTarget > values.max_size())
            {
                throw std::length_error("Scratch vector reserve exceeds max_size().");
            }
            values.clear();
            values.reserve(capacityTarget);
        }
    }

    // resize scratch storage without having to preserve its contents if capacity grows
    template <class T>
    inline void ResizeScratchVector(
        std::vector<T>& values,
        const uint64_t N_elements,
        const float extraAllocCapacityPct = 0.0f)
    {
        ReserveScratchVector(values, N_elements, extraAllocCapacityPct);
        values.resize(N_elements);
    }

    // grow scratch storage without shrinking or having to preserve its contents if capacity grows
    template <class T>
    inline void ResizeGrowScratchVector(
        std::vector<T>& values,
        const uint64_t N_elements,
        const float extraAllocCapacityPct = 0.0f)
    {
        if (values.size() >= N_elements) { return; }

        ResizeScratchVector(values, N_elements, extraAllocCapacityPct);
    }

    // release all vector storage
    template <class T>
    static void ReleaseVectorMemory(std::vector<T>& v)
    {
        std::vector<T>().swap(v);
    }

} // namespace Base

namespace Bench
{
    enum class TimeUnit : uint8_t
    {
        s, ms, us
    };

    static inline double ConvertFromNs(TimeUnit unit, uint64_t ns)
    {
        switch (unit)
        {
        case TimeUnit::s: { return static_cast<double>(ns) / 1'000'000'000.0; }
        case TimeUnit::ms: { return static_cast<double>(ns) / 1'000'000.0; }
        case TimeUnit::us: { return static_cast<double>(ns) / 1'000.0; }
        }
        return static_cast<double>(ns);
    }

    static inline const char* UnitLabel(TimeUnit unit)
    {
        switch (unit)
        {
        case TimeUnit::s: { return "s"; }
        case TimeUnit::ms: { return "ms"; }
        case TimeUnit::us: { return "us"; }
        }
        return "";
    }

} // namespace Bench

namespace Mem
{
    enum class ByteUnit : uint8_t
    {
        GB, MB, KB, B
    };

    static inline double ConvertFromBytes(ByteUnit unit, uint64_t N_bytes)
    {
        switch (unit)
        {
        case ByteUnit::GB: { return static_cast<double>(N_bytes) / (1000.0 * 1000.0 * 1000.0); }
        case ByteUnit::MB: { return static_cast<double>(N_bytes) / (1000.0 * 1000.0); }
        case ByteUnit::KB: { return static_cast<double>(N_bytes) / 1000.0; }
        case ByteUnit::B: { return static_cast<double>(N_bytes); }
        }
        return static_cast<double>(N_bytes);
    }

    static inline const char* UnitLabel(ByteUnit u)
    {
        switch (u)
        {
        case ByteUnit::GB: { return "GB"; }
        case ByteUnit::MB: { return "MB"; }
        case ByteUnit::KB: { return "KB"; }
        case ByteUnit::B: { return "B"; }
        }
        return "";
    }
	
} // namespace Mem
