// helpers.h
//
// TODO: comment

#pragma once

// external
#include <algorithm>
#include <cmath>
#include <fstream>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/spdlog.h>



namespace Base
{
    // --- [ approximate equality helpers for detecting parameter changes ] ----

    // float comparison
    inline bool ApproxEqu(const float a, const float b, const float eps = 1e-6f)
	{
        return std::fabs(a - b) <= eps;
    }

    // double comparison
    inline bool ApproxEqu(const double a, const double b, const double eps = 1e-6)
    {
        return std::fabs(a - b) <= eps;
    }

    // element-wise vec2 comparison
    inline bool ApproxEqu(const glm::vec2& a, const glm::vec2& b, const float eps = 1e-6f)
	{
        return ApproxEqu(a.x, b.x, eps) &&
               ApproxEqu(a.y, b.y, eps);
    }

	// element-wise vec3 comparison
    inline bool ApproxEqu(const glm::vec3& a, const glm::vec3& b, const float eps = 1e-6f)
	{
        return ApproxEqu(a.x, b.x, eps) &&
               ApproxEqu(a.y, b.y, eps) &&
               ApproxEqu(a.z, b.z, eps);
    }

    // element-wise vec4 comparison
    inline bool ApproxEqu(const glm::vec4& a, const glm::vec4& b, const float eps = 1e-6f)
	{
        return ApproxEqu(a.x, b.x, eps) &&
               ApproxEqu(a.y, b.y, eps) &&
               ApproxEqu(a.z, b.z, eps) &&
               ApproxEqu(a.w, b.w, eps);
    }

    // element-wise mat4 comparison
    inline bool ApproxEqu(const glm::mat4& a, const glm::mat4& b, const float eps = 1e-6f)
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

    // --- [ json ] ------------------------------------------------------------

    inline bool LoadJsonObjectFromFile(
        const std::string& fileName,
        nlohmann::json& out)
    {
        namespace fs = std::filesystem;

        // use TSP_CONFIG_DIR from the build system as base path
        fs::path base(TSP_CONFIG_DIR);
        fs::path full = base / fileName;
        std::string path = full.string();

        std::ifstream in(path);
        if (!in)
        {
            SPDLOG_ERROR("Failed to open JSON file: {}",
                path);
            return false;
        }

        try
        {
            in >> out;
        }
        catch (const std::exception& e)
        {
            SPDLOG_ERROR("Failed to parse JSON {}: {}",
                path, e.what());
            return false;
        }

        // return JSON object
        return true;
    }

    // --- [ misc ] ------------------------------------------------------------

    inline bool Float_AlmostEqual(const float a, const float b,
                                  const float rel_epsilon = 1e-6f,
                                  const float abs_epsilon = 1e-9f) noexcept
    {
        const float diff = std::fabs(a - b);
        if (diff <= abs_epsilon) { return true; }

        return diff <= rel_epsilon * fmaxf(std::fabs(a), std::fabs(b));
    }

} // namespace Base
