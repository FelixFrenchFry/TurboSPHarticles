// helpers.h
//
// TODO: comment

#pragma once

// external
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <optional>



namespace Base
{
	// ===== approximate equality helpers for detecting parameter changes =====

    // float comparison
    inline bool ApproxEqu(float a, float b, float eps = 1e-6f)
	{
        return std::fabs(a - b) <= eps;
    }

    // element-wise vec2 comparison
    inline bool ApproxEqu(const glm::vec2& a, const glm::vec2& b, float eps = 1e-6f)
	{
        return ApproxEqu(a.x, b.x, eps) &&
               ApproxEqu(a.y, b.y, eps);
    }

	// element-wise vec3 comparison
    inline bool ApproxEqu(const glm::vec3& a, const glm::vec3& b, float eps = 1e-6f)
	{
        return ApproxEqu(a.x, b.x, eps) &&
               ApproxEqu(a.y, b.y, eps) &&
               ApproxEqu(a.z, b.z, eps);
    }

    // element-wise vec4 comparison
    inline bool ApproxEqu(const glm::vec4& a, const glm::vec4& b, float eps = 1e-6f)
	{
        return ApproxEqu(a.x, b.x, eps) &&
               ApproxEqu(a.y, b.y, eps) &&
               ApproxEqu(a.z, b.z, eps) &&
               ApproxEqu(a.w, b.w, eps);
    }

    // element-wise mat4 comparison
    inline bool ApproxEqu(const glm::mat4& a, const glm::mat4& b, float eps = 1e-6f)
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
    inline void MarkIfChanged(const std::optional<float>& o, const float& cur, bool& ch)
    {
	    if (o && !ApproxEqu(*o, cur)) ch = true;
    }
    inline void MarkIfChanged(const std::optional<glm::vec2>& o, const glm::vec2& cur, bool& ch)
    {
	    if (o && !ApproxEqu(*o, cur)) ch = true;
    }
    inline void MarkIfChanged(const std::optional<glm::vec3>& o, const glm::vec3& cur, bool& ch)
    {
	    if (o && !ApproxEqu(*o, cur)) ch = true;
    }
    inline void MarkIfChanged(const std::optional<glm::vec4>& o, const glm::vec4& cur, bool& ch)
    {
	    if (o && !ApproxEqu(*o, cur)) ch = true;
    }
    inline void MarkIfChanged(const std::optional<glm::mat4>& o, const glm::mat4& cur, bool& ch)
    {
	    if (o && !ApproxEqu(*o, cur)) ch = true;
    }

    // ===== misc =====

    inline bool Float_AlmostEqual(float a, float b,
                                  float rel_epsilon = 1e-5f,
                                  float abs_epsilon = 1e-8f) noexcept
    {
        float diff = std::fabs(a - b);
        if (diff <= abs_epsilon) return true;
        return diff <= rel_epsilon * fmaxf(std::fabs(a), std::fabs(b));
    }

} // namespace Base
