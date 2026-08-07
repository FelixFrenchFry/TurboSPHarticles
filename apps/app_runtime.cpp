// app_runtime.cpp

// external
#include <algorithm>
#include <atomic>
#include <chrono>
#include <CLI/CLI.hpp>
#include <cctype>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include <thread>
#include <utility>
#if defined(_WIN32)
	#include <Windows.h>
#endif

// internal
#include "app_runtime.h"



namespace
{
	#if defined(_WIN32)
	std::atomic<bool> g_consoleStopRequested{ false };
	#else
	volatile std::sig_atomic_t g_consoleStopRequested = 0;
	#endif

	#if defined(_WIN32)
	BOOL WINAPI RequestConsoleStop(DWORD ctrlType)
	{
		switch (ctrlType)
		{
		case CTRL_C_EVENT:
		case CTRL_BREAK_EVENT:
		case CTRL_CLOSE_EVENT:
		case CTRL_LOGOFF_EVENT:
		case CTRL_SHUTDOWN_EVENT:
			g_consoleStopRequested.store(true, std::memory_order_release);
			return TRUE;
		default:
			return FALSE;
		}
	}
	#else
	void RequestConsoleStop(int)
	{
		g_consoleStopRequested = 1;
	}
	#endif

	void ResetConsoleStopRequest()
	{
		#if defined(_WIN32)
		g_consoleStopRequested.store(false, std::memory_order_release);
		#else
		g_consoleStopRequested = 0;
		#endif
	}

	bool ConsoleStopRequested()
	{
		#if defined(_WIN32)
		return g_consoleStopRequested.load(std::memory_order_acquire);
		#else
		return (g_consoleStopRequested != 0);
		#endif
	}

	bool InstallConsoleStopHandler()
	{
		#if defined(_WIN32)
		return SetConsoleCtrlHandler(RequestConsoleStop, TRUE) != 0;
		#else
		return std::signal(SIGINT, RequestConsoleStop) != SIG_ERR;
		#endif
	}

	void UninstallConsoleStopHandler()
	{
		#if defined(_WIN32)
		SetConsoleCtrlHandler(RequestConsoleStop, FALSE);
		#else
		std::signal(SIGINT, SIG_DFL);
		#endif
	}

	void NormalizeProfile(std::string& p)
	{
		if (p.empty()) { p = "default"; return; }

		// letters -> uppercase
		for (char& c : p)
		{
			c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
		}

		// the only profile directory that is not upper case
		if (p == "DEFAULT") { p = "default"; }
	}

	bool IsValidProfile(const std::string& p)
	{
		if (p == "default") { return true; }
		if (p.empty()) { return false; }

		// letters and digits only, which also keeps the name a single path segment
		return std::all_of(p.begin(), p.end(), [](const char c)
		{
			return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
		});
	}

} // namespace anonymous

void App::SetupLogging()
{
	// [year-month-day 12:34:56.789] [type] message
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
	spdlog::flush_on(spdlog::level::err);
}

void App::PrintPlatformBanner()
{
	#if defined(_WIN32)
	printf("(Running on Windows)\n");
	#elif defined(__linux__)
	printf("(Running on Linux)\n");
	#endif
}

App::ParseResult App::ParseMainOptions(
	int argc,
	char** argv,
	const char* appName,
	bool logErrors)
{
	CLI::App application{ appName };
	MainOptions options{};

	application.add_option("-p,--profile", options.profile,
			"Name of configuration profile /configs/<...> with 'A-Z' and '0-9' "
			"(e.g. 'A' or 'A2') or 'default'");
	application.add_option("--sim", options.simOverrides,
		"Master override simulation parameter via key=value "
		"(repeatable, CLI wins over config file)");

	// allow single-dash alias "-sim" by normalizing to "--sim" before CLI11 parsing
	std::vector<std::string> cliArgs{};
	cliArgs.reserve((argc > 0) ? static_cast<size_t>(argc) : 0);
	if (argc > 0)
	{
		cliArgs.push_back(argv[0]);
	}
	for (int i = 1; i < argc; i++)
	{
		std::string arg = argv[i];
		if (arg == "-sim")
		{
			arg = "--sim";
		}
		else if (arg.rfind("-sim=", 0) == 0)
		{
			arg.replace(0, 4, "--sim");
		}

		cliArgs.push_back(arg);
	}

	std::vector<char*> cliArgv{};
	cliArgv.reserve(cliArgs.size());
	for (auto& arg : cliArgs)
	{
		cliArgv.push_back(arg.data());
	}

	try
	{
		application.parse(static_cast<int>(cliArgv.size()), cliArgv.data());
	}
	catch (const CLI::ParseError& e)
	{
		return ParseResult
		{
			.options = std::move(options),
			.exitCode = logErrors ? application.exit(e) : e.get_exit_code(),
			.shouldExit = true
		};
	}

	NormalizeProfile(options.profile);
	if (!IsValidProfile(options.profile))
	{
		if (logErrors)
		{
			SPDLOG_ERROR("Invalid --profile arg '{}'. Use letters 'A-Z' and digits '0-9' "
				"or leave empty for 'default'.", options.profile);
		}

		return ParseResult
		{
			.options = std::move(options),
			.exitCode = EXIT_FAILURE,
			.shouldExit = true
		};
	}

	return ParseResult
	{
		.options = std::move(options),
		.exitCode = EXIT_SUCCESS,
		.shouldExit = false
	};
}

bool App::LoadSimParams(
	Sim::SimParamsMgr& simParamsMgr,
	std::string& profile,
	const std::vector<std::string>& simOverrides,
	const bool logStatus)
{
	if (!simParamsMgr.LoadSimParamsFromProfile(profile))
	{
		if (profile != "default")
		{
			if (logStatus)
			{
				SPDLOG_WARN("Failed to load '/{}/params.json'. Falling back to '/default/params.json'.", profile);
			}
			profile = "default";

			if (!simParamsMgr.LoadSimParamsFromProfile(profile))
			{
				SPDLOG_ERROR("Failed to load '/default/params.json' too.");
				return false;
			}
		}
		else
		{
			SPDLOG_ERROR("Failed to load '/default/params.json'.");
			return false;
		}
	}
	if (logStatus)
	{
		SPDLOG_INFO("Using config profile '{}'", profile);
	}

	if (!simOverrides.empty())
	{
		if (!simParamsMgr.LoadSimParamsFromCliOverrides(simOverrides, logStatus))
		{
			SPDLOG_ERROR("Failed to apply one or more '-sim/--sim' overrides.");
			return false;
		}

		if (logStatus)
		{
			SPDLOG_INFO("Applied {} simulation override(s) from CLI args",
				simOverrides.size());
		}
	}

	return true;
}

App::ThreadBudget App::ResolveThreadBudget(
	const int64_t maxThreadsSim,
	const uint32_t N_visThreads)
{
	const uint32_t N_totalThreads = std::max<uint32_t>(1, std::thread::hardware_concurrency());
	const uint32_t N_simThreadsDefault = N_totalThreads > N_visThreads
									   ? N_totalThreads - N_visThreads : 1;

	uint32_t N_simThreads = N_simThreadsDefault;
	if (maxThreadsSim == -1)
	{
		N_simThreads = N_simThreadsDefault;
	}
	else if (maxThreadsSim == 0)
	{
		SPDLOG_WARN("Invalid 'maxThreadsSim' value 0 (must be -1 or > 0). "
			"Falling back to default simulation thread budget {}.",
			N_simThreadsDefault);
	}
	else if (maxThreadsSim < -1)
	{
		SPDLOG_WARN("Invalid 'maxThreadsSim' value {} (must be -1 or > 0). "
			"Falling back to default simulation thread budget {}.",
			maxThreadsSim, N_simThreadsDefault);
	}
	else
	{
		const uint64_t requested = static_cast<uint64_t>(maxThreadsSim);
		if (requested <= static_cast<uint64_t>(N_simThreadsDefault))
		{
			N_simThreads = static_cast<uint32_t>(requested);
		}
	}

	return ThreadBudget
	{
		.N_totalThreads = N_totalThreads,
		.N_simThreads = N_simThreads,
		.N_visThreads = N_visThreads
	};
}

App::AppStopWatcher::~AppStopWatcher()
{
	Stop();
}

bool App::AppStopWatcher::Start(std::function<void()> stopCallback)
{
	Stop();
	if (!stopCallback) { return false; }

	ResetConsoleStopRequest();
	if (!InstallConsoleStopHandler())
	{
		SPDLOG_WARN("Failed to install Ctrl+C handler; final shutdown summary may be skipped.");
		return false;
	}

	handlerInstalled_ = true;
	watcherThread_ = std::jthread(
		[stopCallback = std::move(stopCallback)](const std::stop_token& stopToken)
		{
			while (!stopToken.stop_requested())
			{
				if (ConsoleStopRequested())
				{
					stopCallback();
				}

				std::this_thread::sleep_for(std::chrono::milliseconds(20));
			}
		});

	return true;
}

void App::AppStopWatcher::Stop()
{
	if (watcherThread_.joinable())
	{
		watcherThread_.request_stop();
		watcherThread_.join();
	}

	if (handlerInstalled_)
	{
		UninstallConsoleStopHandler();
		handlerInstalled_ = false;
	}
}
