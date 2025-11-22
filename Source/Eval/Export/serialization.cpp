// serialization.cpp

// external
#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <vector>

// internal
#include "serialization.h"



using namespace HighFive;

// ===== helpers =====

namespace {

	inline DataSetCreateProps MakeDsProps(int64_t N_elements,
										  int64_t N_elemBytes,
										  const Export::SaveOptions& opt)
	{
		DataSetCreateProps props;

		// no elements -> nothing to do
		if (N_elements == 0) return props;

		// contiguous layout -> nothing to do
		if (!opt.compress) { return props; }

		// chunked + filters
		const int64_t wantChunkElems = std::max<int64_t>(1, opt.targetChunkBytes / N_elemBytes);
		const int64_t chunkElems = std::max<int64_t>(1, std::min(N_elements, wantChunkElems));

		props.add(Chunking({ static_cast<hsize_t>(chunkElems) }));

		// byte shuffle
		if (opt.shuffleFilter)
		{
			props.add(Shuffle());
		}

		// compression
		props.add(Deflate(opt.compressionLevel));

		return props;
	}

	template <class T>
	inline void Write1D(Group& group, const char* name,
		const std::vector<T>& vec,
		const Export::SaveOptions& opt)
	{
		auto dsp = DataSpace::From(vec);
		auto dcp = MakeDsProps(vec.size(), sizeof(T), opt);
		auto ds = group.createDataSet<T>(name, dsp, dcp);

		ds.write(vec);
	}

	template <class T>
	inline void Read1D(const Group& group, const char* name, std::vector<T>& vec)
	{
		auto ds = group.getDataSet(name);

		ds.read(vec);
	}

	inline File OpenFileForWrite(const std::string& path)
	{
		FileCreateProps fcp;
		FileAccessProps fap;

		return File(path, File::Overwrite, fcp, fap);
	}

	inline File OpenFileForRead(const std::string& path)
	{
		return File(path, File::ReadOnly);
	}

	inline void AtomicReplace(const std::filesystem::path& tmp,
		const std::filesystem::path& dst)
	{
		std::error_code ec;

		// try to rename directly
		std::filesystem::rename(tmp, dst, ec);
		if (!ec) return;

		// remove existing then rename as fallback
		std::filesystem::remove(dst, ec); // ignore error if not exists
		std::filesystem::rename(tmp, dst); // let this throw if it still fails
	}

	void CheckFluidConsistent(const Sync::FluidParticles& F)
	{
		const auto N = static_cast<size_t>(F.N_);
		assert(F.id.size() == N);

		assert(F.xPos.size() == N && F.yPos.size() == N && F.zPos.size() == N);
		assert(F.xVel.size() == N && F.yVel.size() == N && F.zVel.size() == N);
		assert(F.xAcc.size() == N && F.yAcc.size() == N && F.zAcc.size() == N);

		assert(F.xAccPress.size() == N && F.yAccPress.size() == N && F.zAccPress.size() == N);
		assert(F.xAccVisc.size() == N && F.yAccVisc.size() == N && F.zAccVisc.size() == N);
		assert(F.xAccExt.size() == N && F.yAccExt.size() == N && F.zAccExt.size() == N);

		assert(F.density.size() == N);
		assert(F.pressure.size() == N);
	}

	void CheckBoundaryConsistent(const Sync::BoundaryParticles& B)
	{
		const auto N = static_cast<size_t>(B.N_);
		assert(B.id.size() == N);

		assert(B.xPos.size() == N && B.yPos.size() == N && B.zPos.size() == N);
		assert(B.xVel.size() == N && B.yVel.size() == N && B.zVel.size() == N);
	}

} // anonymous namespace

// ===== saving and loading =====

namespace Export {

	void SaveUniversalSimState(const std::string& path,
							   const Sync::UniversalSimState& state,
							   const SaveOptions& opt)
	{
		// file path handling
		const std::filesystem::path dstPath{ path };
		const std::filesystem::path tmpPath = opt.atomicWrite ?
			std::filesystem::path(dstPath.string() + opt.tempSuffix) : dstPath;

		// open file (optionally with alignment)
		File file = OpenFileForWrite(tmpPath.string());

		// metadata
		auto root = file.createGroup("/State");
		root.createAttribute("schemaVersion", state.schemaVersion);
		root.createAttribute("step", state.step);
		root.createAttribute("time", state.time);

		// sanity check before writing
		CheckFluidConsistent(state.fluidParticles);
		CheckBoundaryConsistent(state.boundaryParticles);

		{ // open scope for writer

			// fluid data
			{
				const auto& F = state.fluidParticles;
				auto gF = root.createGroup("fluidParticles");

				gF.createAttribute("N_particles", F.N_);

				// write SoA arrays
				Write1D(gF, "xPos", F.xPos, opt);
				Write1D(gF, "yPos", F.yPos, opt);
				Write1D(gF, "zPos", F.zPos, opt);

				Write1D(gF, "xVel", F.xVel, opt);
				Write1D(gF, "yVel", F.yVel, opt);
				Write1D(gF, "zVel", F.zVel, opt);

				Write1D(gF, "xVelPred", F.xVelPred, opt);
				Write1D(gF, "yVelPred", F.yVelPred, opt);
				Write1D(gF, "zVelPred", F.zVelPred, opt);

				Write1D(gF, "xAcc", F.xAcc, opt);
				Write1D(gF, "yAcc", F.yAcc, opt);
				Write1D(gF, "zAcc", F.zAcc, opt);

				Write1D(gF, "xAccPress", F.xAccPress, opt);
				Write1D(gF, "yAccPress", F.yAccPress, opt);
				Write1D(gF, "zAccPress", F.zAccPress, opt);

				Write1D(gF, "xAccVisc", F.xAccVisc, opt);
				Write1D(gF, "yAccVisc", F.yAccVisc, opt);
				Write1D(gF, "zAccVisc", F.zAccVisc, opt);

				Write1D(gF, "xAccExt", F.xAccExt, opt);
				Write1D(gF, "yAccExt", F.yAccExt, opt);
				Write1D(gF, "zAccExt", F.zAccExt, opt);

				Write1D(gF, "sourceTerm", F.sourceTerm, opt);
				Write1D(gF, "diagElem", F.diagElem, opt);
				Write1D(gF, "sysVecElem", F.sysVecElem, opt);
				Write1D(gF, "pressure", F.pressure, opt);
				Write1D(gF, "density", F.density, opt);

				Write1D(gF, "id", F.id, opt);
			}

			// boundary data
			{
				const auto& B = state.boundaryParticles;
				auto gB = root.createGroup("boundaryParticles");

				gB.createAttribute("N_particles", B.N_);

				// write SoA arrays
				Write1D(gB, "xPos", B.xPos, opt);
				Write1D(gB, "yPos", B.yPos, opt);
				Write1D(gB, "zPos", B.zPos, opt);

				Write1D(gB, "xVel", B.xVel, opt);
				Write1D(gB, "yVel", B.yVel, opt);
				Write1D(gB, "zVel", B.zVel, opt);

				Write1D(gB, "id", B.id, opt);
			}

		} // end scope for writer

		// close file by going out of scope
		if (opt.atomicWrite)
		{
			AtomicReplace(tmpPath, dstPath);
		}
	}

	void LoadUniversalSimState(const std::string& path,
							   Sync::UniversalSimState& state)
	{
		File file = OpenFileForRead(path);

		// metadata
		// TODO: branch on schemaVersion?
		auto root = file.getGroup("/State");
		root.getAttribute("schemaVersion").read(state.schemaVersion);
		root.getAttribute("step").read(state.step);
		root.getAttribute("time").read(state.time);

		// fluid data
		{
			auto gF = root.getGroup("fluidParticles");
			uint32_t NF = 0;
			gF.getAttribute("N_particles").read(NF);
			state.ResizeFluid(NF);

			auto& F = state.fluidParticles;

			// read SoA arrays
			Read1D(gF, "xPos", F.xPos);
			Read1D(gF, "yPos", F.yPos);
			Read1D(gF, "zPos", F.zPos);

			Read1D(gF, "xVel", F.xVel);
			Read1D(gF, "yVel", F.yVel);
			Read1D(gF, "zVel", F.zVel);

			Read1D(gF, "xVelPred", F.xVelPred);
			Read1D(gF, "yVelPred", F.yVelPred);
			Read1D(gF, "zVelPred", F.zVelPred);

			Read1D(gF, "xAcc", F.xAcc);
			Read1D(gF, "yAcc", F.yAcc);
			Read1D(gF, "zAcc", F.zAcc);

			Read1D(gF, "xAccPress", F.xAccPress);
			Read1D(gF, "yAccPress", F.yAccPress);
			Read1D(gF, "zAccPress", F.zAccPress);

			Read1D(gF, "xAccVisc", F.xAccVisc);
			Read1D(gF, "yAccVisc", F.yAccVisc);
			Read1D(gF, "zAccVisc", F.zAccVisc);

			Read1D(gF, "xAccExt", F.xAccExt);
			Read1D(gF, "yAccExt", F.yAccExt);
			Read1D(gF, "zAccExt", F.zAccExt);

			Read1D(gF, "sourceTerm", F.sourceTerm);
			Read1D(gF, "diagElem", F.diagElem);
			Read1D(gF, "sysVecElem", F.sysVecElem);
			Read1D(gF, "pressure", F.pressure);
			Read1D(gF, "density", F.density);

			Read1D(gF, "id", F.id);

			// sanity check
			F.N_ = NF;
		}

		// boundary data
		{
			auto gB = root.getGroup("boundaryParticles");
			uint32_t NB = 0;
			gB.getAttribute("N_particles").read(NB);
			state.ResizeBoundary(NB);

			auto& B = state.boundaryParticles;

			// read SoA arrays
			Read1D(gB, "xPos", B.xPos);
			Read1D(gB, "yPos", B.yPos);
			Read1D(gB, "zPos", B.zPos);

			Read1D(gB, "xVel", B.xVel);
			Read1D(gB, "yVel", B.yVel);
			Read1D(gB, "zVel", B.zVel);

			Read1D(gB, "id", B.id);

			// sanity check
			B.N_ = NB;
		}
	}

} // namespace Export
