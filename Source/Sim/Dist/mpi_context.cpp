// mpi_context.cpp

// external
#include <cstdint>
#include <stdexcept>

// internal
#include "mpi_context.h"



using namespace Dist;

MpiContext::MpiContext(
	int& argc,
	char**& argv,
	const int requestedThreadSupport)
{
	int alreadyInitialized = 0;
	CheckMpiResult(MPI_Initialized(&alreadyInitialized), "MPI_Initialized");

	if (!alreadyInitialized)
	{
		// request thread mode once, before any rank metadata is queried
		CheckMpiResult(MPI_Init_thread(&argc, &argv, requestedThreadSupport, &threadSupport_),
			"MPI_Init_thread");
		ownsMpiLifetime_ = true;
	}
	else
	{
		CheckMpiResult(MPI_Query_thread(&threadSupport_), "MPI_Query_thread");
	}

	if (threadSupport_ < requestedThreadSupport)
	{
		throw std::runtime_error("MPI implementation does not provide the requested thread support.");
	}

	CheckMpiResult(MPI_Comm_rank(MPI_COMM_WORLD, &rank_), "MPI_Comm_rank");
	CheckMpiResult(MPI_Comm_size(MPI_COMM_WORLD, &size_), "MPI_Comm_size");
}

MpiContext::~MpiContext()
{
	// only finalize MPI if this context initialized it
	if (!ownsMpiLifetime_) { return; }

	int alreadyFinalized = 0;
	if (MPI_Finalized(&alreadyFinalized) != MPI_SUCCESS) { return; }
	if (!alreadyFinalized)
	{
		(void)MPI_Finalize();
	}
}

void MpiContext::Barrier() const
{
	CheckMpiResult(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
}

void MpiContext::BroadcastInt(int& value, const int root) const
{
	CheckMpiResult(MPI_Bcast(&value, 1, MPI_INT, root, MPI_COMM_WORLD), "MPI_Bcast(int)");
}

void MpiContext::BroadcastString(std::string& value, const int root) const
{
	// broadcast length first, so non-root ranks can resize their receive buffer
	int length = (rank_ == root) ? static_cast<int>(value.size()) : 0;

	BroadcastInt(length, root);
	if (length < 0)
	{
		throw std::runtime_error("Invalid negative MPI string broadcast length.");
	}

	if (rank_ != root)
	{
		value.resize(static_cast<uint64_t>(length));
	}

	if (length > 0)
	{
		CheckMpiResult(MPI_Bcast(value.data(), length, MPI_CHAR, root, MPI_COMM_WORLD),
			"MPI_Bcast(string)");
	}
}

bool MpiContext::AllTrue(const bool localValue) const
{
	const int local = localValue ? 1 : 0;
	int global = 0;
	CheckMpiResult(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
		"MPI_Allreduce(bool)");
	return global != 0;
}

const char* MpiContext::ThreadSupportName(const int threadSupport)
{
	switch (threadSupport)
	{
	case MPI_THREAD_SINGLE:
		return "MPI_THREAD_SINGLE";
	case MPI_THREAD_FUNNELED:
		return "MPI_THREAD_FUNNELED";
	case MPI_THREAD_SERIALIZED:
		return "MPI_THREAD_SERIALIZED";
	case MPI_THREAD_MULTIPLE:
		return "MPI_THREAD_MULTIPLE";
	default:
		return "MPI_THREAD_<unknown>";
	}
}

void MpiContext::CheckMpiResult(const int result, const char* opName)
{
	if (result == MPI_SUCCESS) { return; }

	throw std::runtime_error(std::string(opName) + " failed with MPI error code " +
		std::to_string(result));
}
