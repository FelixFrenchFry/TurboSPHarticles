// mpi_context.cpp

// external
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
	#include <process.h>
#else
	#include <unistd.h>
#endif

// internal
#include "mpi_context.h"



using namespace Dist;

#if MPI_VERSION < 3 || (MPI_VERSION == 3 && MPI_SUBVERSION < 1)
#error "Distributed snapshots require MPI 3.1 nonblocking collective file I/O."
#endif

namespace
{
	constexpr uint64_t MPI_WAIT_CLOCK_CHECK_INTERVAL = 64;

	int64_t CurrentProcessId()
	{
	#ifdef _WIN32
		return static_cast<int64_t>(_getpid());
	#else
		return static_cast<int64_t>(getpid());
	#endif
	}

	int CheckedMpiCount(const uint64_t N_values)
	{
		if (N_values > static_cast<uint64_t>(std::numeric_limits<int>::max()))
		{
			throw std::runtime_error("MPI chunk exceeds int count range.");
		}

		return static_cast<int>(N_values);
	}

	std::string MpiErrorMessage(
		const int result,
		const char* opName)
	{
		// MPI_Error_string is unavailable before initialization succeeds
		const char* operationName = opName && opName[0] != '\0'
			? opName
			: "MPI operation";
		return std::string(operationName) + " failed with MPI error code " +
			std::to_string(result) + ".";
	}

	[[noreturn]] void AbortMpiJob(
		const std::string_view reason,
		const int errorCode) noexcept
	{
		// avoid allocations in the abort path
		const int abortErrorCode = errorCode == MPI_SUCCESS ? MPI_ERR_OTHER : errorCode;
		if (reason.empty())
		{
			std::fputs(errorCode == MPI_SUCCESS ? "Application failure" : "MPI operation", stderr);
		}
		else { std::fwrite(reason.data(), 1, reason.size(), stderr); }
		if (errorCode != MPI_SUCCESS)
		{
			std::fprintf(stderr, " (MPI error code %d", errorCode);
			char errorString[MPI_MAX_ERROR_STRING]{};
			int errorStringLength = 0;
			if (MPI_Error_string(errorCode, errorString, &errorStringLength) == MPI_SUCCESS &&
				errorStringLength > 0)
			{
				std::fprintf(stderr, ": %.*s", errorStringLength, errorString);
			}
			std::fputs(")", stderr);
		}
		std::fputs(". Aborting MPI job.\n", stderr);
		std::fflush(stderr);

		(void)MPI_Abort(MPI_COMM_WORLD, abortErrorCode);
		std::terminate();
	}

	void CheckMpiResultImpl(
		const int result,
		const char* opName,
		const bool abortOnError = true)
	{
		if (result == MPI_SUCCESS) { return; }

		if (abortOnError)
		{
			AbortMpiJob(opName ? opName : "MPI operation", result);
		}
		throw std::runtime_error(MpiErrorMessage(result, opName));
	}

	template <typename T, typename MpiOp>
	void ForEachMpiVectorChunk(
		std::vector<T>& values,
		MpiOp&& mpiOp)
	{
		constexpr uint64_t chunkCapacity = MPI_CONTEXT_BYTE_CHUNK_SIZE / sizeof(T);

		uint64_t valueOffset = 0;
		while (valueOffset < values.size())
		{
			const uint64_t N_chunkValues = std::min(
				chunkCapacity,
				static_cast<uint64_t>(values.size()) - valueOffset);
			mpiOp(
				values.data() + valueOffset,
				CheckedMpiCount(N_chunkValues));
			valueOffset += N_chunkValues;
		}
	}

} // namespace anonymous

MpiContext::MpiContext(
	int& argc,
	char**& argv,
	const int requestedThreadSupport,
	const MpiWaitConfig waitConfig)
	: waitConfig_(waitConfig)
{
	if (waitConfig_.warningTimeout.count() <= 0 ||
		waitConfig_.abortTimeout < waitConfig_.warningTimeout)
	{
		throw std::runtime_error("Invalid MPI wait timeout configuration.");
	}

	int alreadyInitialized = 0;
	// MPI_Abort requires successful initialization
	CheckMpiResultImpl(
		MPI_Initialized(&alreadyInitialized),
		"MPI_Initialized",
		false);

	if (!alreadyInitialized)
	{
		// request thread mode once, before any rank metadata is queried
		CheckMpiResultImpl(
			MPI_Init_thread(&argc, &argv, requestedThreadSupport, &threadSupport_),
			"MPI_Init_thread",
			false);
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
	CheckMpiResult(MPI_Comm_size(MPI_COMM_WORLD, &N_ranks_), "MPI_Comm_size");
	CheckMpiResult(
		MPI_Comm_get_errhandler(MPI_COMM_WORLD, &previousErrorHandler_),
		"MPI_Comm_get_errhandler");
	CheckMpiResult(
		MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
		"MPI_Comm_set_errhandler");

	std::array<char, MPI_MAX_PROCESSOR_NAME> processorName{};
	int processorNameLength = 0;
	CheckMpiResult(
		MPI_Get_processor_name(processorName.data(), &processorNameLength),
		"MPI_Get_processor_name");
	if (processorNameLength > 0)
	{
		processorName_.assign(
			processorName.data(),
			static_cast<std::string::size_type>(processorNameLength));
	}
	processId_ = CurrentProcessId();
}

MpiContext::~MpiContext()
{
	int alreadyFinalized = 0;
	if (MPI_Finalized(&alreadyFinalized) != MPI_SUCCESS) { return; }
	if (alreadyFinalized) { return; }

	// restore the communicator before leaving or finalizing the MPI lifetime
	if (previousErrorHandler_ != MPI_ERRHANDLER_NULL)
	{
		(void)MPI_Comm_set_errhandler(MPI_COMM_WORLD, previousErrorHandler_);
		(void)MPI_Errhandler_free(&previousErrorHandler_);
	}

	// only finalize MPI if this context initialized it
	if (ownsMpiLifetime_)
	{
		(void)MPI_Finalize();
	}
}

[[noreturn]] void MpiContext::Abort(
	const std::string_view reason,
	const int errorCode) const noexcept
{
	std::fprintf(stderr, "[rank %d/%d host=%s pid=%lld] ",
		rank_, N_ranks_, processorName_.c_str(), static_cast<long long>(processId_));
	AbortMpiJob(reason, errorCode);
}

void MpiContext::Barrier() const
{
	MPI_Request request = MPI_REQUEST_NULL;
	CheckMpiResult(MPI_Ibarrier(MPI_COMM_WORLD, &request), "MPI_Ibarrier");
	Wait(&request, 1, "MPI barrier", nullptr);
}

void MpiContext::BroadcastInt(int& value, const int root) const
{
	MPI_Request request = MPI_REQUEST_NULL;
	CheckMpiResult(
		MPI_Ibcast(&value, 1, MPI_INT, root, MPI_COMM_WORLD, &request),
		"MPI_Ibcast(int)");
	Wait(&request, 1, "MPI int broadcast", nullptr);
}

void MpiContext::BroadcastUint64(uint64_t& value, const int root) const
{
	MPI_Request request = MPI_REQUEST_NULL;
	CheckMpiResult(
		MPI_Ibcast(&value, 1, MPI_UINT64_T, root, MPI_COMM_WORLD, &request),
		"MPI_Ibcast(uint64)");
	Wait(&request, 1, "MPI uint64 broadcast", nullptr);
}

void MpiContext::BroadcastFloats(
	std::vector<float>& values,
	const int root) const
{
	ForEachMpiVectorChunk(
		values,
		[this, root](float* chunkData, const int N_chunkValues)
		{
			MPI_Request request = MPI_REQUEST_NULL;
			CheckMpiResult(
				MPI_Ibcast(
					chunkData,
					N_chunkValues,
					MPI_FLOAT,
					root,
					MPI_COMM_WORLD,
					&request),
				"MPI_Ibcast(floats)");
			Wait(&request, 1, "MPI float broadcast", nullptr);
		});
}

void MpiContext::BroadcastString(std::string& value, const int root) const noexcept
{
	try
	{
		if (rank_ == root && value.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
		{
			Abort("MPI string broadcast length exceeds int range");
		}

		// broadcast length first, so non-root ranks can resize their receive buffer
		int N_chars = (rank_ == root) ? static_cast<int>(value.size()) : 0;
		BroadcastInt(N_chars, root);
		if (N_chars < 0) { Abort("Invalid negative MPI string broadcast length"); }

		if (rank_ != root)
		{
			value.resize(static_cast<uint64_t>(N_chars));
		}

		if (N_chars > 0)
		{
			MPI_Request request = MPI_REQUEST_NULL;
			CheckMpiResult(
				MPI_Ibcast(
					value.data(), N_chars, MPI_CHAR, root, MPI_COMM_WORLD, &request),
				"MPI_Ibcast(string)");
			Wait(&request, 1, "MPI string broadcast", nullptr);
		}
	}
	catch (...)
	{
		// a failed receive allocation must not leave peers waiting in the broadcast
		Abort("Failed to broadcast MPI string");
	}
}

bool MpiContext::AllTrue(
	const bool localValue,
	const char* context) const
{
	const int local = localValue ? 1 : 0;
	int global = 0;
	MPI_Request request = MPI_REQUEST_NULL;
	CheckMpiResult(
		MPI_Iallreduce(
			&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD, &request),
		"MPI_Iallreduce(bool)");
	Wait(&request, 1, context ? context : "MPI all-true reduction", nullptr);
	return global != 0;
}

uint64_t MpiContext::AllReduceSumUint64(const uint64_t value) const
{
	uint64_t result = 0;
	MPI_Request request = MPI_REQUEST_NULL;
	CheckMpiResult(
		MPI_Iallreduce(
			&value,
			&result,
			1,
			MPI_UINT64_T,
			MPI_SUM,
			MPI_COMM_WORLD,
			&request),
		"MPI_Iallreduce(uint64 sum)");
	Wait(&request, 1, "MPI uint64-sum reduction", nullptr);
	return result;
}

void MpiContext::AllReduceSumUint64s(
	const std::span<uint64_t> values) const
{
	MPI_Request request = MPI_REQUEST_NULL;
	CheckMpiResult(
		MPI_Iallreduce(
			MPI_IN_PLACE,
			values.data(),
			CheckedMpiCount(values.size()),
			MPI_UINT64_T,
			MPI_SUM,
			MPI_COMM_WORLD,
			&request),
		"MPI_Iallreduce(uint64 sum)");
	Wait(&request, 1, "MPI uint64-array sum reduction", nullptr);
}

void MpiContext::AllReduceSumUint64s(std::vector<uint64_t>& values) const
{
	ForEachMpiVectorChunk(
		values,
		[this](uint64_t* chunkData, const int N_chunkValues)
		{
			MPI_Request request = MPI_REQUEST_NULL;
			CheckMpiResult(
				MPI_Iallreduce(
					MPI_IN_PLACE,
					chunkData,
					N_chunkValues,
					MPI_UINT64_T,
					MPI_SUM,
					MPI_COMM_WORLD,
					&request),
				"MPI_Iallreduce(uint64s)");
			Wait(&request, 1, "MPI uint64-vector sum reduction", nullptr);
		});
}

void MpiContext::AllReduceSumDoubles(
	const std::span<double> values,
	const char* context) const
{
	MPI_Request request = MPI_REQUEST_NULL;
	CheckMpiResult(
		MPI_Iallreduce(
			MPI_IN_PLACE,
			values.data(),
			CheckedMpiCount(values.size()),
			MPI_DOUBLE,
			MPI_SUM,
			MPI_COMM_WORLD,
			&request),
		"MPI_Iallreduce(double sum)");
	Wait(&request, 1, context ? context : "MPI double-sum reduction", nullptr);
}

void MpiContext::AllReduceMaxUint32s(const std::span<uint32_t> values) const
{
	MPI_Request request = MPI_REQUEST_NULL;
	CheckMpiResult(
		MPI_Iallreduce(
			MPI_IN_PLACE,
			values.data(),
			CheckedMpiCount(values.size()),
			MPI_UINT32_T,
			MPI_MAX,
			MPI_COMM_WORLD,
			&request),
		"MPI_Iallreduce(uint32 max)");
	Wait(&request, 1, "MPI uint32-array max reduction", nullptr);
}

float MpiContext::AllReduceMinFloat(const float value) const
{
	float result = 0.0f;
	MPI_Request request = MPI_REQUEST_NULL;
	CheckMpiResult(
		MPI_Iallreduce(
			&value,
			&result,
			1,
			MPI_FLOAT,
			MPI_MIN,
			MPI_COMM_WORLD,
			&request),
		"MPI_Iallreduce(float min)");
	Wait(&request, 1, "MPI float-min reduction", nullptr);
	return result;
}

float MpiContext::AllReduceMaxFloat(const float value) const
{
	float result = 0.0f;
	MPI_Request request = MPI_REQUEST_NULL;
	CheckMpiResult(
		MPI_Iallreduce(
			&value,
			&result,
			1,
			MPI_FLOAT,
			MPI_MAX,
			MPI_COMM_WORLD,
			&request),
		"MPI_Iallreduce(float max)");
	Wait(&request, 1, "MPI float-max reduction", nullptr);
	return result;
}

void MpiContext::AllGatherDouble(
	const double localValue,
	std::vector<double>& values) const
{
	values.resize(static_cast<uint64_t>(N_ranks_));
	MPI_Request request = MPI_REQUEST_NULL;
	CheckMpiResult(
		MPI_Iallgather(
			&localValue,
			1,
			MPI_DOUBLE,
			values.data(),
			1,
			MPI_DOUBLE,
			MPI_COMM_WORLD,
			&request),
		"MPI_Iallgather(double)");
	Wait(&request, 1, "MPI double all-gather", nullptr);
}

void MpiContext::AllReduceSumUint32s(std::vector<uint32_t>& values) const
{
	ForEachMpiVectorChunk(
		values,
		[this](uint32_t* chunkData, const int N_chunkValues)
		{
			MPI_Request request = MPI_REQUEST_NULL;
			CheckMpiResult(
				MPI_Iallreduce(
					MPI_IN_PLACE,
					chunkData,
					N_chunkValues,
					MPI_UINT32_T,
					MPI_SUM,
					MPI_COMM_WORLD,
					&request),
				"MPI_Iallreduce(uint32s)");
			Wait(&request, 1, "MPI uint32-vector sum reduction", nullptr);
		});
}

void MpiContext::AllReduceSumFloats(std::vector<float>& values) const
{
	ForEachMpiVectorChunk(
		values,
		[this](float* chunkData, const int N_chunkValues)
		{
			MPI_Request request = MPI_REQUEST_NULL;
			CheckMpiResult(
				MPI_Iallreduce(
					MPI_IN_PLACE,
					chunkData,
					N_chunkValues,
					MPI_FLOAT,
					MPI_SUM,
					MPI_COMM_WORLD,
					&request),
				"MPI_Iallreduce(floats)");
			Wait(&request, 1, "MPI float-vector sum reduction", nullptr);
		});
}

void MpiContext::AllToAllInts(
	const std::vector<int>& sendValues,
	std::vector<int>& recvValues,
	const int N_valuesPerRank) const
{
	if (N_valuesPerRank <= 0)
	{
		throw std::runtime_error("MPI all-to-all values-per-rank must be positive.");
	}

	const uint64_t N_expectedValues = static_cast<uint64_t>(N_ranks_) * static_cast<uint64_t>(N_valuesPerRank);
	if (sendValues.size() != N_expectedValues)
	{
		throw std::runtime_error("MPI all-to-all int send buffer size does not match world size.");
	}

	recvValues.resize(N_expectedValues);
	MPI_Request request = MPI_REQUEST_NULL;
	CheckMpiResult(
		MPI_Ialltoall(
			sendValues.data(),
			N_valuesPerRank,
			MPI_INT,
			recvValues.data(),
			N_valuesPerRank,
			MPI_INT,
			MPI_COMM_WORLD,
			&request),
		"MPI_Ialltoall(int)");
	Wait(&request, 1, "MPI int all-to-all", nullptr);
}

void MpiContext::IRecvBytes(
	void* recvData,
	const int N_recvBytes,
	const int peerRank,
	const int tag,
	MPI_Request& request) const
{
	if (N_recvBytes < 0)
	{
		throw std::runtime_error("Invalid negative MPI byte count.");
	}
	if (N_recvBytes > 0 && recvData == nullptr)
	{
		throw std::runtime_error("Invalid null MPI receive buffer.");
	}

	CheckMpiResult(
		MPI_Irecv(
			recvData,
			N_recvBytes,
			MPI_BYTE,
			peerRank,
			tag,
			MPI_COMM_WORLD,
			&request),
		"MPI_Irecv(bytes)");
}

void MpiContext::ISendBytes(
	const void* sendData,
	const int N_sendBytes,
	const int peerRank,
	const int tag,
	MPI_Request& request) const
{
	if (N_sendBytes < 0)
	{
		throw std::runtime_error("Invalid negative MPI byte count.");
	}
	if (N_sendBytes > 0 && sendData == nullptr)
	{
		throw std::runtime_error("Invalid null MPI send buffer.");
	}

	CheckMpiResult(
		MPI_Isend(
			sendData,
			N_sendBytes,
			MPI_BYTE,
			peerRank,
			tag,
			MPI_COMM_WORLD,
			&request),
		"MPI_Isend(bytes)");
}

void MpiContext::IRecvBytesChunked(
	void* recvData,
	const uint64_t N_recvBytes,
	const int peerRank,
	const int tag,
	std::vector<MPI_Request>& requests,
	std::vector<MpiRequestInfo>* requestInfos) const
{
	if (N_recvBytes == 0) { return; }
	if (recvData == nullptr)
	{
		throw std::runtime_error("Invalid null MPI chunked receive buffer.");
	}

	uint8_t* recvBytes = static_cast<uint8_t*>(recvData);
	uint64_t byteOffset = 0;

	// using the same tag for each chunk,
	// MPI preserves FIFO-ordering for same peer/tag/communicator
	while (byteOffset < N_recvBytes)
	{
		const uint64_t N_chunkBytes = std::min(
			MPI_CONTEXT_BYTE_CHUNK_SIZE, N_recvBytes - byteOffset);

		requests.emplace_back(MPI_REQUEST_NULL);
		IRecvBytes(
			recvBytes + byteOffset,
			CheckedMpiCount(N_chunkBytes),
			peerRank,
			tag,
			requests.back());
		if (requestInfos)
		{
			requestInfos->push_back({
				.peerRank = peerRank,
				.tag = tag,
				.N_bytes = N_chunkBytes,
				.direction = MpiRequestDirection::Receive
			});
		}

		byteOffset += N_chunkBytes;
	}
}

void MpiContext::ISendBytesChunked(
	const void* sendData,
	const uint64_t N_sendBytes,
	const int peerRank,
	const int tag,
	std::vector<MPI_Request>& requests,
	std::vector<MpiRequestInfo>* requestInfos) const
{
	if (N_sendBytes == 0) { return; }
	if (sendData == nullptr)
	{
		throw std::runtime_error("Invalid null MPI chunked send buffer.");
	}

	const uint8_t* sendBytes = static_cast<const uint8_t*>(sendData);
	uint64_t byteOffset = 0;

	// using the same tag for each chunk,
	// MPI preserves FIFO-ordering for same peer/tag/communicator
	while (byteOffset < N_sendBytes)
	{
		const uint64_t N_chunkBytes = std::min(
			MPI_CONTEXT_BYTE_CHUNK_SIZE, N_sendBytes - byteOffset);

		requests.emplace_back(MPI_REQUEST_NULL);
		ISendBytes(
			sendBytes + byteOffset,
			CheckedMpiCount(N_chunkBytes),
			peerRank,
			tag,
			requests.back());
		if (requestInfos)
		{
			requestInfos->push_back({
				.peerRank = peerRank,
				.tag = tag,
				.N_bytes = N_chunkBytes,
				.direction = MpiRequestDirection::Send
			});
		}

		byteOffset += N_chunkBytes;
	}
}

void MpiContext::WaitAll(
	std::vector<MPI_Request>& requests,
	const char* context,
	const std::vector<MpiRequestInfo>* requestInfos) const noexcept
{
	if (requestInfos && requestInfos->size() != requests.size())
	{
		Abort("MPI request diagnostics do not match the request count");
	}
	Wait(
		requests.data(), requests.size(), context,
		requestInfos ? requestInfos->data() : nullptr);
}

MPI_File MpiContext::OpenFileForWrite(
	const std::string& path,
	const uint64_t N_bytes) const
{
	try
	{
		if (N_bytes > static_cast<uint64_t>(std::numeric_limits<MPI_Offset>::max()))
		{
			Abort("MPI file size exceeds MPI_Offset range");
		}

		MPI_File file = MPI_FILE_NULL;
		CheckMpiResult(
			MPI_File_open(
				MPI_COMM_WORLD,
				path.c_str(),
				MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL,
				MPI_INFO_NULL,
				&file),
			"MPI_File_open");
		CheckMpiResult(
			MPI_File_set_errhandler(file, MPI_ERRORS_RETURN),
			"MPI_File_set_errhandler");
		CheckMpiResult(
			MPI_File_set_view(file, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL),
			"MPI_File_set_view");

		// ranks write disjoint file ranges
		CheckMpiResult(
			MPI_File_set_atomicity(file, 0),
			"MPI_File_set_atomicity");

		return file;
	}
	catch (...)
	{
		Abort("Exception during collective MPI file open");
	}
}

void MpiContext::WriteFileBytesAtAll(
	MPI_File file,
	const uint64_t offset,
	const void* data,
	const int N_bytes,
	const char* context) const
{
	try
	{
		const auto maxOffset = static_cast<uint64_t>(std::numeric_limits<MPI_Offset>::max());
		if (N_bytes < 0 || offset > maxOffset || static_cast<uint64_t>(N_bytes) > maxOffset - offset)
		{
			Abort("MPI file write exceeds offset/count range");
		}

		MPI_Request request = MPI_REQUEST_NULL;
		CheckMpiResult(
			MPI_File_iwrite_at_all(
				file,
				static_cast<MPI_Offset>(offset),
				data,
				N_bytes,
				MPI_BYTE,
				&request),
			"MPI_File_iwrite_at_all");
		Wait(&request, 1, context, nullptr);

		int N_written = 0;
		CheckMpiResult(
			MPI_Get_count(&waitStatuses_[0], MPI_BYTE, &N_written),
			"MPI_Get_count(file write)");
		if (N_written != N_bytes) { Abort("Incomplete collective MPI file write"); }
	}
	catch (...)
	{
		Abort("Exception during collective MPI file write");
	}
}

void MpiContext::CloseFile(MPI_File& file) const
{
	try
	{
		CheckMpiResult(MPI_File_sync(file), "MPI_File_sync");
		CheckMpiResult(MPI_File_close(&file), "MPI_File_close");
	}
	catch (...)
	{
		Abort("Exception during collective MPI file close");
	}
}

void MpiContext::Wait(
	MPI_Request* requests,
	const uint64_t N_requests,
	const char* context,
	const MpiRequestInfo* requestInfos) const noexcept
{
	if (N_requests == 0) { return; }
	if (N_requests > static_cast<uint64_t>(std::numeric_limits<int>::max()))
	{
		Abort("MPI request count exceeds int range");
	}

	const char* opName = context && context[0] != '\0'
		? context
		: "MPI request completion";
	try
	{
		waitStatuses_.resize(N_requests);
		const auto waitStart = std::chrono::steady_clock::now();
		bool warningPrinted = false;
		uint64_t N_polls = 0;

		while (true)
		{
			int complete = 0;
			const int result = MPI_Testall(
				static_cast<int>(N_requests),
				requests,
				&complete,
				waitStatuses_.data());
			if (result != MPI_SUCCESS)
			{
				if (result == MPI_ERR_IN_STATUS)
				{
					for (uint64_t requestIdx = 0; requestIdx != N_requests; requestIdx++)
					{
						const int requestError = waitStatuses_[requestIdx].MPI_ERROR;
						if (requestError == MPI_SUCCESS || requestError == MPI_ERR_PENDING) { continue; }

						std::fprintf(
							stderr,
							"[rank %d/%d host=%s pid=%lld] MPI request %llu failed: %s.\n",
							rank_, N_ranks_, processorName_.c_str(),
							static_cast<long long>(processId_),
							static_cast<unsigned long long>(requestIdx), opName);
						Abort(opName, requestError);
					}
				}
				Abort(opName, result);
			}
			if (complete) { return; }
			N_polls += 1;
			if (N_polls % MPI_WAIT_CLOCK_CHECK_INTERVAL != 0) { continue; }

			const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::steady_clock::now() - waitStart);
			if (!warningPrinted && elapsed >= waitConfig_.warningTimeout)
			{
				warningPrinted = true;
				std::fprintf(
					stderr,
					"[rank %d/%d host=%s pid=%lld] MPI wait is still pending after %lld ms: %s.\n",
					rank_, N_ranks_, processorName_.c_str(),
					static_cast<long long>(processId_),
					static_cast<long long>(elapsed.count()), opName);

				for (uint64_t requestIdx = 0; requestIdx != N_requests; requestIdx++)
				{
					if (requests[requestIdx] == MPI_REQUEST_NULL) { continue; }
					int requestComplete = 0;
					MPI_Status status{};
					CheckMpiResult(
						MPI_Request_get_status(requests[requestIdx], &requestComplete, &status),
						"MPI_Request_get_status");
					if (requestComplete) { continue; }

					if (requestInfos)
					{
						const MpiRequestInfo& info = requestInfos[requestIdx];
						std::fprintf(
							stderr,
							"[rank %d/%d host=%s pid=%lld] Pending MPI request %llu: direction=%s peer=%d tag=%d bytes=%llu.\n",
							rank_, N_ranks_, processorName_.c_str(),
							static_cast<long long>(processId_),
							static_cast<unsigned long long>(requestIdx),
							info.direction == MpiRequestDirection::Receive ? "receive" : "send",
							info.peerRank, info.tag,
							static_cast<unsigned long long>(info.N_bytes));
					}
					else
					{
						std::fprintf(
							stderr,
							"[rank %d/%d host=%s pid=%lld] Pending MPI request %llu.\n",
							rank_, N_ranks_, processorName_.c_str(),
							static_cast<long long>(processId_),
							static_cast<unsigned long long>(requestIdx));
					}
				}
				std::fflush(stderr);
			}

			if (elapsed >= waitConfig_.abortTimeout)
			{
				std::fprintf(
					stderr,
					"[rank %d/%d host=%s pid=%lld] MPI wait timed out after %lld ms: %s.\n",
					rank_, N_ranks_, processorName_.c_str(),
					static_cast<long long>(processId_),
					static_cast<long long>(elapsed.count()), opName);
				Abort("MPI wait timed out");
			}
		}
	}
	catch (...)
	{
		// requests are already active, so failure must abort the job
		Abort("Failed while waiting for MPI requests");
	}
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
	CheckMpiResultImpl(result, opName);
}
