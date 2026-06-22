// app_runtime.h

#pragma once

// external
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

// internal
#include "Source/Sim/Core/parameters_sim.h"



namespace App
{
	struct MainOptions
	{
		std::string profile = "Default";
		std::vector<std::string> simOverrides{};
	};

	struct ParseResult
	{
		MainOptions options{};
		int exitCode = 0;
		bool shouldExit = false;
	};

	struct ThreadBudget
	{
		uint32_t N_totalThreads = 1;
		uint32_t N_simThreads = 1;
		uint32_t N_visThreads = 0;
	};

	void SetupLogging();
	void PrintPlatformBanner();

	ParseResult ParseMainOptions(
		int argc,
		char** argv,
		const char* appName,
		bool logErrors = true);

	bool LoadSimParams(
		Sim::SimParamsMgr& simParamsMgr,
		std::string& profile,
		const std::vector<std::string>& simOverrides,
		bool logStatus = true);

	ThreadBudget ResolveThreadBudget(
		int64_t maxThreadsSim,
		uint32_t N_visThreads);

	class AppStopWatcher
	{
	public: // ---------------------------------------------------------------------------------------

		AppStopWatcher() = default;
		~AppStopWatcher();

		bool Start(std::function<void()> stopCallback);
		void Stop();

	private: // --------------------------------------------------------------------------------------

		std::jthread watcherThread_{};
		bool handlerInstalled_ = false;
	};

} // namespace App
