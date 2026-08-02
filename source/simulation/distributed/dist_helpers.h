// dist_helpers.h

#pragma once

// external
#include <spdlog/spdlog.h>



// root-only spdlog helpers for global distributed status messages
#define SPDLOG_INFO_MPI(message, mpiContext, ...) \
	do { if ((mpiContext).IsRoot()) { SPDLOG_INFO(message __VA_OPT__(,) __VA_ARGS__); } } while (false)

#define SPDLOG_WARN_MPI(message, mpiContext, ...) \
	do { if ((mpiContext).IsRoot()) { SPDLOG_WARN(message __VA_OPT__(,) __VA_ARGS__); } } while (false)

#define SPDLOG_ERROR_MPI(message, mpiContext, ...) \
	do { if ((mpiContext).IsRoot()) { SPDLOG_ERROR(message __VA_OPT__(,) __VA_ARGS__); } } while (false)
