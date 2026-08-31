// mpi_context.h

#pragma once

// external
#include <array>
#include <chrono>
#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>



namespace Dist
{
	inline constexpr uint64_t MPI_CONTEXT_BYTE_CHUNK_SIZE =
		256ull * 1024ull * 1024ull; // ~ 256 MiB

	struct MpiWaitConfig
	{
		std::chrono::milliseconds warningTimeout = std::chrono::minutes(5);
		std::chrono::milliseconds abortTimeout = std::chrono::minutes(30);
	};

	enum class MpiRequestDirection : uint8_t
	{
		Receive,
		Send,
	};

	struct MpiRequestInfo
	{
		int peerRank = -1;
		int tag = -1;
		uint64_t N_bytes = 0;
		MpiRequestDirection direction = MpiRequestDirection::Receive;
	};

	class MpiContext final
	{
	public:

		// initializes MPI if needed, otherwise attaches to the existing MPI lifetime
		MpiContext(
			int& argc,
			char**& argv,
			int requestedThreadSupport = MPI_THREAD_FUNNELED,
			MpiWaitConfig waitConfig = {});
		~MpiContext();

		MpiContext(const MpiContext&) = delete;
		MpiContext& operator=(const MpiContext&) = delete;
		MpiContext(MpiContext&&) = delete;
		MpiContext& operator=(MpiContext&&) = delete;

		// metadata about MPI environment
		int Rank() const { return rank_; }
		int Size() const { return N_ranks_; }
		int RootRank() const { return rootRank_; }
		int ThreadSupport() const { return threadSupport_; }
		bool IsRoot() const { return rank_ == rootRank_; }

		// collectives for MPI communication
		void Barrier() const;
		void BroadcastInt(int& value, int root = 0) const;
		void BroadcastUint64(uint64_t& value, int root = 0) const;
		void BroadcastFloats(std::vector<float>& values, int root = 0) const;
		void BroadcastString(std::string& value, int root = 0) const;

		bool AllTrue(bool localValue, const char* context = nullptr) const;
		uint64_t AllReduceSumUint64(uint64_t value) const;
		void AllReduceSumUint64s(std::array<uint64_t, 4>& values) const;
		void AllReduceSumDoubles(
			std::array<double, 3>& values,
			const char* context = nullptr) const;
		uint32_t AllReduceMaxUint32(uint32_t value) const;
		double AllReduceSumDouble(double value) const;
		float AllReduceMinFloat(float value) const;
		float AllReduceMaxFloat(float value) const;
		void AllGatherDouble(double localValue, std::vector<double>& values) const;
		void AllReduceSumUint32s(std::vector<uint32_t>& values) const;
		void AllReduceSumFloats(std::vector<float>& values) const;
		void AllToAllInts(
			const std::vector<int>& sendValues,
			std::vector<int>& recvValues,
			int N_valuesPerRank = 1) const;

		// point-to-point helpers for neighbor communication
		void IRecvInt(int& recvValue, int peerRank, int tag, MPI_Request& request) const;
		void ISendInt(const int& sendValue, int peerRank, int tag, MPI_Request& request) const;
		void IRecvBytes(
			void* recvData,
			int N_recvBytes,
			int peerRank,
			int tag,
			MPI_Request& request) const;
		void ISendBytes(
			const void* sendData,
			int N_sendBytes,
			int peerRank,
			int tag,
			MPI_Request& request) const;
		void IRecvBytesChunked(
			void* recvData,
			uint64_t N_recvBytes,
			int peerRank,
			int tag,
			std::vector<MPI_Request>& requests,
			std::vector<MpiRequestInfo>* requestInfos = nullptr) const;
		void ISendBytesChunked(
			const void* sendData,
			uint64_t N_sendBytes,
			int peerRank,
			int tag,
			std::vector<MPI_Request>& requests,
			std::vector<MpiRequestInfo>* requestInfos = nullptr) const;
		void WaitAll(
			std::vector<MPI_Request>& requests,
			const char* context = nullptr,
			const std::vector<MpiRequestInfo>* requestInfos = nullptr) const;
		// aborts all ranks after an unrecoverable MPI operation error
		[[noreturn]] void Abort(
			const char* context = nullptr,
			int errorCode = MPI_ERR_OTHER) const;
		[[noreturn]] void AbortJob(
			const std::string& reason,
			int errorCode = MPI_ERR_OTHER) const;

		// helpers
		static const char* ThreadSupportName(int threadSupport);

	private:

		static void CheckMpiResult(int result, const char* opName);
		void Wait(
			MPI_Request* requests,
			uint64_t N_requests,
			const char* context,
			const MpiRequestInfo* requestInfos) const;

		int rank_ = 0;
		int N_ranks_ = 1;
		int rootRank_ = 0;
		int threadSupport_ = MPI_THREAD_SINGLE;
		std::string processorName_ = "unknown";
		int64_t processId_ = 0;
		MPI_Errhandler previousErrorHandler_ = MPI_ERRHANDLER_NULL;
		MpiWaitConfig waitConfig_{};
		// only the context that called MPI_Init_thread is allowed to call MPI_Finalize
		bool ownsMpiLifetime_ = false;
		// reused by timed waits, so request-status diagnostics do not allocate per exchange
		mutable std::vector<MPI_Status> waitStatuses_;
	};

} // namespace Dist
