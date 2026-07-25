// mpi_context.h

#pragma once

// external
#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>



namespace Dist
{
	class MpiContext final
	{
	public:

		// initializes MPI if needed, otherwise attaches to the existing MPI lifetime
		MpiContext(int& argc, char**& argv, int requestedThreadSupport = MPI_THREAD_FUNNELED);
		~MpiContext();

		MpiContext(const MpiContext&) = delete;
		MpiContext& operator=(const MpiContext&) = delete;
		MpiContext(MpiContext&&) = delete;
		MpiContext& operator=(MpiContext&&) = delete;

		// metadata about MPI environment
		int Rank() const { return rank_; }
		int Size() const { return size_; }
		int RootRank() const { return rootRank_; }
		int ThreadSupport() const { return threadSupport_; }
		bool IsRoot() const { return rank_ == rootRank_; }

		// collectives for MPI communication
		void Barrier() const;
		void BroadcastInt(int& value, int root = 0) const;
		void BroadcastUint64(uint64_t& value, int root = 0) const;
		void BroadcastFloats(std::vector<float>& values, int root = 0) const;
		void BroadcastString(std::string& value, int root = 0) const;

		bool AllTrue(bool localValue) const;
		uint64_t AllReduceSumUint64(uint64_t value) const;
		uint32_t AllReduceMaxUint32(uint32_t value) const;
		double AllReduceSumDouble(double value) const;
		float AllReduceMinFloat(float value) const;
		float AllReduceMaxFloat(float value) const;
		void AllReduceSumUint32s(std::vector<uint32_t>& values) const;
		void AllReduceSumFloats(std::vector<float>& values) const;
		void AllReduceMaxUint8s(std::vector<uint8_t>& values) const;
		void AllToAllInts(const std::vector<int>& sendValues, std::vector<int>& recvValues) const;

		// point-to-point helpers for neighbor communication
		void IRecvInt(int& recvValue, int peerRank, int tag, MPI_Request& request) const;
		void ISendInt(const int& sendValue, int peerRank, int tag, MPI_Request& request) const;
		void IRecvBytes(
			void* recvData,
			int recvByteCount,
			int peerRank,
			int tag,
			MPI_Request& request) const;
		void ISendBytes(
			const void* sendData,
			int sendByteCount,
			int peerRank,
			int tag,
			MPI_Request& request) const;
		void IRecvBytesChunked(
			void* recvData,
			uint64_t recvByteCount,
			int peerRank,
			int tag,
			std::vector<MPI_Request>& requests) const;
		void ISendBytesChunked(
			const void* sendData,
			uint64_t sendByteCount,
			int peerRank,
			int tag,
			std::vector<MPI_Request>& requests) const;
		void WaitAll(std::vector<MPI_Request>& requests) const;

		// helpers
		static const char* ThreadSupportName(int threadSupport);

	private:

		static void CheckMpiResult(int result, const char* opName);

		int rank_ = 0;
		int size_ = 1;
		int rootRank_ = 0;
		int threadSupport_ = MPI_THREAD_SINGLE;
		// only the context that called MPI_Init_thread is allowed to call MPI_Finalize
		bool ownsMpiLifetime_ = false;
	};

} // namespace Dist
