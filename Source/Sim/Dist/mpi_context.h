// mpi_context.h

#pragma once

// external
#include <mpi.h>
#include <string>



namespace Dist
{
	class MpiContext final
	{
	public: // ---------------------------------------------------------------------------------------

		MpiContext(int& argc, char**& argv, int requestedThreadSupport = MPI_THREAD_FUNNELED);
		~MpiContext();

		MpiContext(const MpiContext&) = delete;
		MpiContext& operator=(const MpiContext&) = delete;
		MpiContext(MpiContext&&) = delete;
		MpiContext& operator=(MpiContext&&) = delete;

		// metadata
		int Rank() const { return rank_; }
		int Size() const { return size_; }
		bool IsRoot() const { return rank_ == rootRank_; }
		int RootRank() const { return rootRank_; }
		int ThreadSupport() const { return threadSupport_; }

		// collectives
		void Barrier() const;
		void BroadcastInt(int& value, int root = 0) const;
		void BroadcastString(std::string& value, int root = 0) const;
		bool AllTrue(bool localValue) const;

		// helpers
		static const char* ThreadSupportName(int threadSupport);

	private: // --------------------------------------------------------------------------------------

		static void CheckMpiResult(int result, const char* opName);

		int rank_ = 0;
		int size_ = 1;
		int rootRank_ = 0;
		int threadSupport_ = MPI_THREAD_SINGLE;
		bool ownsMpiLifetime_ = false;
	};

} // namespace Dist
