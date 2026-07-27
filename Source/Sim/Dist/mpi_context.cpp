// mpi_context.cpp

// external
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

// internal
#include "mpi_context.h"



using namespace Dist;

namespace
{
	constexpr uint64_t MPI_CONTEXT_BYTE_CHUNK_SIZE =
		256ull * 1024ull * 1024ull; // ~ 256 MiB

	int CheckedMpiCount(const uint64_t valueCount)
	{
		if (valueCount > static_cast<uint64_t>(std::numeric_limits<int>::max()))
		{
			throw std::runtime_error("MPI chunk exceeds int count range.");
		}

		return static_cast<int>(valueCount);
	}

	void CheckMpiResultImpl(const int result, const char* opName)
	{
		if (result == MPI_SUCCESS) { return; }

		throw std::runtime_error(std::string(opName) + " failed with MPI error code " +
			std::to_string(result) + ".");
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
			const uint64_t chunkSize = std::min(
				chunkCapacity,
				static_cast<uint64_t>(values.size()) - valueOffset);
			CheckMpiResultImpl(
				mpiOp(
					values.data() + valueOffset,
					CheckedMpiCount(chunkSize)),
				opName);
			valueOffset += chunkSize;
		}
	}
}

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
		[root](float* chunkData, const int chunkSize)
		{
			return MPI_Bcast(
				chunkData,
				chunkSize,
				MPI_FLOAT,
				root,
				MPI_COMM_WORLD);
		});
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

void MpiContext::AllReduceSumUint32s(std::vector<uint32_t>& values) const
{
	ForEachMpiVectorChunk(
		values,
		"MPI_Allreduce(uint32s)",
		[](uint32_t* chunkData, const int chunkSize)
		{
			return MPI_Allreduce(
				MPI_IN_PLACE,
				chunkData,
				chunkSize,
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
		[](float* chunkData, const int chunkSize)
		{
			return MPI_Allreduce(
				MPI_IN_PLACE,
				chunkData,
				chunkSize,
				MPI_FLOAT,
				MPI_SUM,
				MPI_COMM_WORLD);
		});
}

void MpiContext::AllReduceMaxUint8s(std::vector<uint8_t>& values) const
{
	ForEachMpiVectorChunk(
		values,
		"MPI_Allreduce(uint8 max)",
		[](uint8_t* chunkData, const int chunkSize)
		{
			return MPI_Allreduce(
				MPI_IN_PLACE,
				chunkData,
				chunkSize,
				MPI_UINT8_T,
				MPI_MAX,
				MPI_COMM_WORLD);
		});
}

void MpiContext::AllToAllInts(
	const std::vector<int>& sendValues,
	std::vector<int>& recvValues,
	const int valuesPerRank) const
{
	if (valuesPerRank <= 0)
	{
		throw std::runtime_error("MPI all-to-all values-per-rank must be positive.");
	}

	const uint64_t expectedValueCount = static_cast<uint64_t>(size_) * static_cast<uint64_t>(valuesPerRank);
	if (sendValues.size() != expectedValueCount)
	{
		throw std::runtime_error("MPI all-to-all int send buffer size does not match world size.");
	}

	recvValues.resize(expectedValueCount);
	CheckMpiResult(
		MPI_Alltoall(
			sendValues.data(),
			valuesPerRank,
			MPI_INT,
			recvValues.data(),
			valuesPerRank,
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
	const int recvByteCount,
	const int peerRank,
	const int tag,
	MPI_Request& request) const
{
	if (recvByteCount < 0)
	{
		throw std::runtime_error("Invalid negative MPI byte count.");
	}
	if (recvByteCount > 0 && recvData == nullptr)
	{
		throw std::runtime_error("Invalid null MPI receive buffer.");
	}

	CheckMpiResult(
		MPI_Irecv(
			recvData,
			recvByteCount,
			MPI_BYTE,
			peerRank,
			tag,
			MPI_COMM_WORLD,
			&request),
		"MPI_Irecv(bytes)");
}

void MpiContext::ISendBytes(
	const void* sendData,
	const int sendByteCount,
	const int peerRank,
	const int tag,
	MPI_Request& request) const
{
	if (sendByteCount < 0)
	{
		throw std::runtime_error("Invalid negative MPI byte count.");
	}
	if (sendByteCount > 0 && sendData == nullptr)
	{
		throw std::runtime_error("Invalid null MPI send buffer.");
	}

	CheckMpiResult(
		MPI_Isend(
			sendData,
			sendByteCount,
			MPI_BYTE,
			peerRank,
			tag,
			MPI_COMM_WORLD,
			&request),
		"MPI_Isend(bytes)");
}

void MpiContext::IRecvBytesChunked(
	void* recvData,
	const uint64_t recvByteCount,
	const int peerRank,
	const int tag,
	std::vector<MPI_Request>& requests) const
{
	if (recvByteCount == 0) { return; }
	if (recvData == nullptr)
	{
		throw std::runtime_error("Invalid null MPI chunked receive buffer.");
	}

	uint8_t* recvBytes = static_cast<uint8_t*>(recvData);
	uint64_t byteOffset = 0;

	// use the same tag for each chunk; MPI preserves ordering for same peer/tag/communicator
	while (byteOffset < recvByteCount)
	{
		const uint64_t chunkByteCount = std::min(
			MPI_CONTEXT_BYTE_CHUNK_SIZE, recvByteCount - byteOffset);

		requests.emplace_back(MPI_REQUEST_NULL);
		IRecvBytes(
			recvBytes + byteOffset,
			CheckedMpiCount(chunkByteCount),
			peerRank,
			tag,
			requests.back());

		byteOffset += chunkByteCount;
	}
}

void MpiContext::ISendBytesChunked(
	const void* sendData,
	const uint64_t sendByteCount,
	const int peerRank,
	const int tag,
	std::vector<MPI_Request>& requests) const
{
	if (sendByteCount == 0) { return; }
	if (sendData == nullptr)
	{
		throw std::runtime_error("Invalid null MPI chunked send buffer.");
	}

	const uint8_t* sendBytes = static_cast<const uint8_t*>(sendData);
	uint64_t byteOffset = 0;

	// use the same tag for each chunk; MPI preserves ordering for same peer/tag/communicator
	while (byteOffset < sendByteCount)
	{
		const uint64_t chunkByteCount = std::min(
			MPI_CONTEXT_BYTE_CHUNK_SIZE, sendByteCount - byteOffset);

		requests.emplace_back(MPI_REQUEST_NULL);
		ISendBytes(
			sendBytes + byteOffset,
			CheckedMpiCount(chunkByteCount),
			peerRank,
			tag,
			requests.back());

		byteOffset += chunkByteCount;
	}
}

void MpiContext::WaitAll(std::vector<MPI_Request>& requests) const
{
	if (requests.empty()) { return; }
	if (requests.size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
	{
		throw std::runtime_error("MPI request count exceeds int range.");
	}

	CheckMpiResult(
		MPI_Waitall(
			static_cast<int>(requests.size()),
			requests.data(),
			MPI_STATUSES_IGNORE),
		"MPI_Waitall");
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
