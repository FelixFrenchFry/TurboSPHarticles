// scenario_json.h
#pragma once

// external
#include <cmath>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>



namespace Scene
{
	inline const nlohmann::json& ReadScenarioField(
		const nlohmann::json& j, const char* field, const std::string& context)
	{
		if (!j.is_object())
		{
			throw std::runtime_error(context + " must be an object.");
		}
		const auto it = j.find(field);
		if (it == j.end())
		{
			throw std::runtime_error(context + " requires '" + field + "'.");
		}
		return *it;
	}

	inline const nlohmann::json& ReadScenarioObject(
		const nlohmann::json& j, const char* field, const std::string& context)
	{
		const auto& value = ReadScenarioField(j, field, context);
		if (!value.is_object())
		{
			throw std::runtime_error(context + "." + field + " must be an object.");
		}
		return value;
	}

	inline const nlohmann::json& ReadScenarioNonEmptyArray(
		const nlohmann::json& j, const char* field, const std::string& context)
	{
		const auto& value = ReadScenarioField(j, field, context);
		if (!value.is_array() || value.empty())
		{
			throw std::runtime_error(context + "." + field + " must be a non-empty array.");
		}
		return value;
	}

	inline void ValidateEmptyScenarioArray(
		const nlohmann::json& j, const char* field, const std::string& context)
	{
		const auto it = j.find(field);
		if (it != j.end() && (!it->is_array() || !it->empty()))
		{
			throw std::runtime_error(context + "." + field + " must be absent or an empty array.");
		}
	}

	inline float ParseScenarioFloat(const nlohmann::json& value, const std::string& field)
	{
		if (!value.is_number())
		{
			throw std::runtime_error(field + " must be a finite number.");
		}
		const double number = value.get<double>();
		if (!std::isfinite(number) || std::abs(number) > std::numeric_limits<float>::max())
		{
			throw std::runtime_error(field + " must be a finite number within float range.");
		}
		return static_cast<float>(number);
	}

	inline float ReadScenarioFloat(
		const nlohmann::json& j, const char* field, const std::string& context)
	{
		return ParseScenarioFloat(ReadScenarioField(j, field, context), context + "." + field);
	}

	template <glm::length_t N>
	inline glm::vec<N, float> ReadScenarioVector(
		const nlohmann::json& j, const char* field, const std::string& context)
	{
		const auto& value = ReadScenarioField(j, field, context);
		const std::string path = context + "." + field;
		if (!value.is_array() || value.size() != N)
		{
			throw std::runtime_error(path + " must contain " + std::to_string(N) + " numbers.");
		}
		glm::vec<N, float> result{};
		for (glm::length_t i = 0; i < N; i++)
		{
			result[i] = ParseScenarioFloat(value[i], path + "[" + std::to_string(i) + "]");
		}
		return result;
	}

	inline std::string ReadScenarioString(
		const nlohmann::json& j, const char* field, const std::string& context)
	{
		const auto& value = ReadScenarioField(j, field, context);
		if (!value.is_string() || value.get_ref<const std::string&>().empty())
		{
			throw std::runtime_error(context + "." + field + " must be a non-empty string.");
		}
		return value.get<std::string>();
	}

	inline void ValidateScenarioType(const nlohmann::json& j, const std::string& scenarioName)
	{
		if (ReadScenarioString(j, "type", scenarioName) != scenarioName)
		{
			throw std::runtime_error(scenarioName + ".type must be '" + scenarioName + "'.");
		}
	}

} // namespace Scene
