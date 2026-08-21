// mpi_context.cpp

// external
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>

// internal
#include "mpi_context.h"



using namespace Dist;

namespace
{
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
		const char* opName,
		const bool includeErrorString = true)
	{
		char errorString[MPI_MAX_ERROR_STRING]{};
		int errorStringLength = 0;
		const bool hasErrorString = includeErrorString && MPI_Error_string(
			result, errorString, &errorStringLength) == MPI_SUCCESS &&
			errorStringLength > 0;

		const char* operationName = opName && opName[0] != '\0'
			? opName
			: "MPI operation";
		std::string message = std::string(operationName) + " failed with MPI error code " +
			std::to_string(result);
		if (hasErrorString)
		{
			message += " (" + std::string(errorString, errorStringLength) + ")";
		}
		return message + ".";
	}

	[[noreturn]] void AbortMpiJob(
		const std::string& message,
		const int errorCode)
	{
		const int abortErrorCode = errorCode == MPI_SUCCESS ? MPI_ERR_OTHER : errorCode;
		std::fprintf(stderr, "%s Aborting MPI job.\n", message.c_str());
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

		const std::string message = MpiErrorMessage(result, opName, abortOnError);
		if (abortOnError)
		{
			AbortMpiJob(message, result);
		}
		throw std::runtime_error(message);
	}

	template <typename T, typename MpiOp>
	void ForEachMpiVectorChunk(
		std::vector<T>& values,
		const char* opName,
		MpiOp&& mpiOp)
	{
		constexpr uint64_t chunkCapacity = MPI_CONTEXT_BYTE_CHUNK_SIZE / sizeof(T);

		uint64_t valueOffset = 0;
		while (valueOffset < values.size())
		{
			const uint64_t N_chunkValues = std::min(
				chunkCapacity,
				static_cast<uint64_t>(values.size()) - valueOffset);
			CheckMpiResultImpl(
				mpiOp(
					values.data() + valueOffset,
					CheckedMpiCount(N_chunkValues)),
				opName);
			valueOffset += N_chunkValues;
		}
	}

} // namespace anonymous

MpiContext::MpiContext(
	int& argc,
	char**& argv,
	const int requestedThreadSupport)
{
	int alreadyInitialized = 0;
	// MPI_Abort is not valid until MPI initialization has succeeded.
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

[[noreturn]] void MpiContext::Abort(
	const char* context,
	const int errorCode) const
{
	const int abortErrorCode = errorCode == MPI_SUCCESS ? MPI_ERR_OTHER : errorCode;
	const std::string message = "[rank " + std::to_string(rank_) + "/" +
		std::to_string(N_ranks_) + "] " +
		MpiErrorMessage(abortErrorCode, context);
	AbortMpiJob(message, abortErrorCode);
}

void MpiContext::Barrier() const
{
	CheckMpiResult(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
}

void MpiContext::BroadcastInt(int& value, const int root) const
{
	CheckMpiResult(MPI_Bcast(&value, 1, MPI_INT, root, MPI_COMM_WORLD), "MPI_Bcast(int)");
}

void MpiContext::BroadcastUint64(uint64_t& value, const int root) const
{
	CheckMpiResult(MPI_Bcast(&value, 1, MPI_UINT64_T, root, MPI_COMM_WORLD), "MPI_Bcast(uint64)");
}

void MpiContext::BroadcastFloats(
	std::vector<float>& values,
	const int root) const
{
	ForEachMpiVectorChunk(
		values,
		"MPI_Bcast(floats)",
		[root](float* chunkData, const int N_chunkValues)
		{
			return MPI_Bcast(
				chunkData,
				N_chunkValues,
				MPI_FLOAT,
				root,
				MPI_COMM_WORLD);
		});
}

void MpiContext::BroadcastString(std::string& value, const int root) const
{
	// broadcast length first, so non-root ranks can resize their receive buffer
	int N_chars = (rank_ == root) ? static_cast<int>(value.size()) : 0;

	BroadcastInt(N_chars, root);
	if (N_chars < 0)
	{
		throw std::runtime_error("Invalid negative MPI string broadcast length.");
	}

	if (rank_ != root)
	{
		value.resize(static_cast<uint64_t>(N_chars));
	}

	if (N_chars > 0)
	{
		CheckMpiResult(MPI_Bcast(value.data(), N_chars, MPI_CHAR, root, MPI_COMM_WORLD),
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

uint64_t MpiContext::AllReduceSumUint64(const uint64_t value) const
{
	uint64_t result = 0;
	CheckMpiResult(
		MPI_Allreduce(
			&value,
			&result,
			1,
			MPI_UINT64_T,
			MPI_SUM,
			MPI_COMM_WORLD),
		"MPI_Allreduce(uint64 sum)");
	return result;
}

void MpiContext::AllReduceSumUint64s(
	std::array<uint64_t, 4>& values) const
{
	CheckMpiResult(
		MPI_Allreduce(
			MPI_IN_PLACE,
			values.data(),
			static_cast<int>(values.size()),
			MPI_UINT64_T,
			MPI_SUM,
			MPI_COMM_WORLD),
		"MPI_Allreduce(uint64 sum)");
}

void MpiContext::AllReduceSumDoubles(
	std::array<double, 3>& values) const
{
	CheckMpiResult(
		MPI_Allreduce(
			MPI_IN_PLACE,
			values.data(),
			static_cast<int>(values.size()),
			MPI_DOUBLE,
			MPI_SUM,
			MPI_COMM_WORLD),
		"MPI_Allreduce(double sum)");
}

uint32_t MpiContext::AllReduceMaxUint32(const uint32_t value) const
{
	uint32_t result = 0;
	CheckMpiResult(
		MPI_Allreduce(
			&value,
			&result,
			1,
			MPI_UINT32_T,
			MPI_MAX,
			MPI_COMM_WORLD),
		"MPI_Allreduce(uint32 max)");
	return result;
}

double MpiContext::AllReduceSumDouble(const double value) const
{
	double result = 0.0;
	CheckMpiResult(
		MPI_Allreduce(
			&value,
			&result,
			1,
			MPI_DOUBLE,
			MPI_SUM,
			MPI_COMM_WORLD),
		"MPI_Allreduce(double sum)");
	return result;
}

float MpiContext::AllReduceMinFloat(const float value) const
{
	float result = 0.0f;
	CheckMpiResult(
		MPI_Allreduce(
			&value,
			&result,
			1,
			MPI_FLOAT,
			MPI_MIN,
			MPI_COMM_WORLD),
		"MPI_Allreduce(float min)");
	return result;
}

float MpiContext::AllReduceMaxFloat(const float value) const
{
	float result = 0.0f;
	CheckMpiResult(
		MPI_Allreduce(
			&value,
			&result,
			1,
			MPI_FLOAT,
			MPI_MAX,
			MPI_COMM_WORLD),
		"MPI_Allreduce(float max)");
	return result;
}

void MpiContext::AllGatherDouble(
	const double localValue,
	std::vector<double>& values) const
{
	values.resize(static_cast<uint64_t>(N_ranks_));
	CheckMpiResult(
		MPI_Allgather(
			&localValue,
			1,
			MPI_DOUBLE,
			values.data(),
			1,
			MPI_DOUBLE,
			MPI_COMM_WORLD),
		"MPI_Allgather(double)");
}

void MpiContext::AllReduceSumUint32s(std::vector<uint32_t>& values) const
{
	ForEachMpiVectorChunk(
		values,
		"MPI_Allreduce(uint32s)",
		[](uint32_t* chunkData, const int N_chunkValues)
		{
			return MPI_Allreduce(
				MPI_IN_PLACE,
				chunkData,
				N_chunkValues,
				MPI_UINT32_T,
				MPI_SUM,
				MPI_COMM_WORLD);
		});
}

void MpiContext::AllReduceSumFloats(std::vector<float>& values) const
{
	ForEachMpiVectorChunk(
		values,
		"MPI_Allreduce(floats)",
		[](float* chunkData, const int N_chunkValues)
		{
			return MPI_Allreduce(
				MPI_IN_PLACE,
				chunkData,
				N_chunkValues,
				MPI_FLOAT,
				MPI_SUM,
				MPI_COMM_WORLD);
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
	CheckMpiResult(
		MPI_Alltoall(
			sendValues.data(),
			N_valuesPerRank,
			MPI_INT,
			recvValues.data(),
			N_valuesPerRank,
			MPI_INT,
			MPI_COMM_WORLD),
		"MPI_Alltoall(int)");
}

void MpiContext::IRecvInt(
	int& recvValue,
	const int peerRank,
	const int tag,
	MPI_Request& request) const
{
	CheckMpiResult(
		MPI_Irecv(
			&recvValue,
			1,
			MPI_INT,
			peerRank,
			tag,
			MPI_COMM_WORLD,
			&request),
		"MPI_Irecv(int)");
}

void MpiContext::ISendInt(
	const int& sendValue,
	const int peerRank,
	const int tag,
	MPI_Request& request) const
{
	CheckMpiResult(
		MPI_Isend(
			&sendValue,
			1,
			MPI_INT,
			peerRank,
			tag,
			MPI_COMM_WORLD,
			&request),
		"MPI_Isend(int)");
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
	std::vector<MPI_Request>& requests) const
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

		byteOffset += N_chunkBytes;
	}
}

void MpiContext::ISendBytesChunked(
	const void* sendData,
	const uint64_t N_sendBytes,
	const int peerRank,
	const int tag,
	std::vector<MPI_Request>& requests) const
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

		byteOffset += N_chunkBytes;
	}
}

void MpiContext::WaitAll(
	std::vector<MPI_Request>& requests,
	const char* context) const
{
	if (requests.empty()) { return; }
	if (requests.size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
	{
		throw std::runtime_error("MPI request count exceeds int range.");
	}

	const std::string opName = context && context[0] != '\0'
		? "MPI_Waitall (" + std::string(context) + ")"
		: "MPI_Waitall";
	waitStatuses_.resize(requests.size());
	const int result = MPI_Waitall(
		static_cast<int>(requests.size()),
		requests.data(),
		waitStatuses_.data());
	if (result == MPI_SUCCESS) { return; }

	if (result == MPI_ERR_IN_STATUS)
	{
		for (uint64_t requestIdx = 0; requestIdx != requests.size(); requestIdx++)
		{
			if (waitStatuses_[requestIdx].MPI_ERROR == MPI_SUCCESS) { continue; }

			const std::string requestOpName = opName + " request " +
				std::to_string(requestIdx);
			Abort(requestOpName.c_str(), waitStatuses_[requestIdx].MPI_ERROR);
		}
	}

	Abort(opName.c_str(), result);
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
