// serialization.cpp

// external
#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <sstream>
#include <vector>

// internal
#include "serialization.h"



using namespace HighFive;

namespace Export
{
	inline DataSetCreateProps MakeDsProps(uint64_t N_elements,
										  uint64_t N_elemBytes,
										  const Export::SaveOptions& opt)
	{
		DataSetCreateProps props;

		// no elements -> nothing to do
		if (N_elements == 0) return props;

		// contiguous layout -> nothing to do
		if (!opt.compress) { return props; }

		// chunked + filters
		const uint64_t wantChunkElems = std::max<uint64_t>(1, opt.targetChunkBytes / N_elemBytes);
		const uint64_t chunkElems = std::max<uint64_t>(1, std::min(N_elements, wantChunkElems));

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

	inline std::string FormatStepPadded(uint32_t step, uint32_t width = 9)
	{
		std::ostringstream oss;
		oss << std::setfill('0') << std::setw(width) << step;

		return oss.str();
	}

	void CheckFluidConsistent(const Sync::FluidParticles& F)
	{
		const auto N = static_cast<size_t>(F.N_);

		assert(F.id.size() == N);

		assert(F.xPos.size() == N && F.yPos.size() == N && F.zPos.size() == N);

		assert(F.xVel.size() == N && F.yVel.size() == N && F.zVel.size() == N);
		assert(F.xVelPred.size() == N && F.yVelPred.size() == N && F.zVelPred.size() == N);

		assert(F.xAccPress.size() == N && F.yAccPress.size() == N && F.zAccPress.size() == N);
		assert(F.xAccNonPress.size() == N && F.yAccNonPress.size() == N && F.zAccNonPress.size() == N);

		assert(F.sourceTerm.size() == N);
		assert(F.diagElem.size() == N);

		assert(F.density.size() == N);
		assert(F.pressure.size() == N);
	}

	void CheckBoundaryConsistent(const Sync::BoundaryParticles& B)
	{
		const auto N = static_cast<size_t>(B.N_);

		assert(B.id.size() == N);

		assert(B.xPos.size() == N && B.yPos.size() == N && B.zPos.size() == N);
	}

	void SerializationMgr::SaveUniversalSimState(
		const Sync::UniversalSimState& state,
		const std::string& path) const
	{
		// file path handling
		const std::filesystem::path dstPath{ path };
		const std::filesystem::path tmpPath = saveOptions_.atomicWrite
			? std::filesystem::path(dstPath.string() + saveOptions_.tempSuffix) : dstPath;

		// open scope for writer
		{
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

			// fluid data
			{
				const auto& F = state.fluidParticles;
				auto gF = root.createGroup("fluidParticles");

				gF.createAttribute("N_particles", F.N_);

				// write SoA arrays
				Write1D(gF, "xPos", F.xPos, saveOptions_);
				Write1D(gF, "yPos", F.yPos, saveOptions_);
				Write1D(gF, "zPos", F.zPos, saveOptions_);

				Write1D(gF, "xVel", F.xVel, saveOptions_);
				Write1D(gF, "yVel", F.yVel, saveOptions_);
				Write1D(gF, "zVel", F.zVel, saveOptions_);

				Write1D(gF, "xVelPred", F.xVelPred, saveOptions_);
				Write1D(gF, "yVelPred", F.yVelPred, saveOptions_);
				Write1D(gF, "zVelPred", F.zVelPred, saveOptions_);

				Write1D(gF, "xAccPress", F.xAccPress, saveOptions_);
				Write1D(gF, "yAccPress", F.yAccPress, saveOptions_);
				Write1D(gF, "zAccPress", F.zAccPress, saveOptions_);

				Write1D(gF, "xAccNonPress", F.xAccNonPress, saveOptions_);
				Write1D(gF, "yAccNonPress", F.yAccNonPress, saveOptions_);
				Write1D(gF, "zAccNonPress", F.zAccNonPress, saveOptions_);

				Write1D(gF, "sourceTerm", F.sourceTerm, saveOptions_);
				Write1D(gF, "diagElem", F.diagElem, saveOptions_);
				Write1D(gF, "pressure", F.pressure, saveOptions_);
				Write1D(gF, "density", F.density, saveOptions_);

				Write1D(gF, "id", F.id, saveOptions_);
			}

			// boundary data
			{
				const auto& B = state.boundaryParticles;
				auto gB = root.createGroup("boundaryParticles");

				gB.createAttribute("N_particles", B.N_);

				// write SoA arrays
				Write1D(gB, "xPos", B.xPos, saveOptions_);
				Write1D(gB, "yPos", B.yPos, saveOptions_);
				Write1D(gB, "zPos", B.zPos, saveOptions_);

				Write1D(gB, "id", B.id, saveOptions_);
			}

		} // end scope for writer

		// close file by going out of scope
		if (saveOptions_.atomicWrite)
		{
			AtomicReplace(tmpPath, dstPath);
		}
	}

	void SerializationMgr::LoadUniversalSimState(
		Sync::UniversalSimState& state,
		const std::string& path) const
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

			Read1D(gF, "xAccPress", F.xAccPress);
			Read1D(gF, "yAccPress", F.yAccPress);
			Read1D(gF, "zAccPress", F.zAccPress);

			Read1D(gF, "xAccNonPress", F.xAccNonPress);
			Read1D(gF, "yAccNonPress", F.yAccNonPress);
			Read1D(gF, "zAccNonPress", F.zAccNonPress);

			Read1D(gF, "sourceTerm", F.sourceTerm);
			Read1D(gF, "diagElem", F.diagElem);
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

			Read1D(gB, "id", B.id);

			// sanity check
			B.N_ = NB;
		}
	}

	void SerializationMgr::SaveExportSnapshot(
		const Sync::ExportSnapshot& snap,
		const std::string& dir) const
	{
		const std::string baseName = "pos_vel_step_" + FormatStepPadded(snap.simStep) + ".h5";

		// ensure export directory exists
		std::error_code ec;
		std::filesystem::create_directories(saveOptions_.baseDir / dir, ec);
		if (ec)
		{
			SPDLOG_ERROR("Failed to create export directory '{}': {}",
				saveOptions_.baseDir.string(), ec.message());
			return;
		}

		// make sure data for every particle is present
		const uint32_t NF_size = snap.N_F;
		assert(snap.xPosF.size() == NF_size);
		assert(snap.yPosF.size() == NF_size);
		assert(snap.zPosF.size() == NF_size);
		assert(snap.velMagF.size() == NF_size);

		// file path handling
		const std::filesystem::path dstPath = saveOptions_.baseDir / dir /  baseName;
		const std::filesystem::path tmpPath = saveOptions_.atomicWrite
			? std::filesystem::path(dstPath.string() + saveOptions_.tempSuffix)
			: dstPath;

		// open scope for writer
		{
			// open file
			File file = OpenFileForWrite(tmpPath.string());

			// metadata
			auto root = file.createGroup("/State");
			root.createAttribute("step", snap.simStep);
			root.createAttribute("time", snap.simTime);

			{
				auto gF = root.createGroup("fluidParticles");

				gF.createAttribute("N_particles", NF_size);

				Write1D(gF, "xPosF", snap.xPosF, saveOptions_);
				Write1D(gF, "yPosF", snap.yPosF, saveOptions_);
				Write1D(gF, "zPosF", snap.zPosF, saveOptions_);
				Write1D(gF, "velMagF", snap.velMagF, saveOptions_);
			}
		} // end scope for writer

		if (saveOptions_.atomicWrite)
		{
			AtomicReplace(tmpPath, dstPath);
		}
	}

	void SerializationMgr::SaveExportSnapXdmfInfo(
		const Sync::ExportSnapshot& snap,
		const std::string& dir) const
	{
		const std::string baseName_h5 = "pos_vel_step_" + FormatStepPadded(snap.simStep) + ".h5";
		const std::string baseName_xdmf = "pos_vel_step_" + FormatStepPadded(snap.simStep) + ".xdmf";

		// ensure export directory exists
		std::error_code ec;
		std::filesystem::create_directories(saveOptions_.baseDir / dir, ec);
		if (ec)
		{
			SPDLOG_ERROR("Failed to create export directory '{}': {}",
				saveOptions_.baseDir.string(), ec.message());
			return;
		}

		const std::filesystem::path xdmfPath = saveOptions_.baseDir / dir / baseName_xdmf;

		// open XDMF file
		std::ofstream ofs(xdmfPath, std::ios::out | std::ios::trunc);
		if (!ofs)
		{
			SPDLOG_ERROR("Failed to open XDMF file for write: {}",
				xdmfPath.string());
			return;
		}

		// make sure data for every particle is present
		const uint32_t NF_size = snap.N_F;
		assert(snap.xPosF.size() == NF_size);
		assert(snap.yPosF.size() == NF_size);
		assert(snap.zPosF.size() == NF_size);
		assert(snap.velMagF.size() == NF_size);

		// XDMF header
		ofs << R"(<?xml version="1.0" ?>)" << "\n";
		ofs << R"(<Xdmf Version="3.0">)" << "\n";
		ofs << R"(  <Domain>)" << "\n";

		// one uniform grid per file
		ofs << "    <Grid Name=\"Step_" << snap.simStep
			<< "\" GridType=\"Uniform\">" << "\n";

		// time metadata
		ofs << "      <Time Value=\""
			<< std::setprecision(16) << snap.simTime
			<< "\" />\n";

		// topology of point cloud with NF vertices
		ofs << "      <Topology TopologyType=\"Polyvertex\" "
			<< "Dimensions=\"" << NF_size << "\"/>\n";

		// geometry of XYZ coordinates
		ofs << "      <Geometry GeometryType=\"XYZ\">\n";
		ofs << "        <DataItem ItemType=\"Function\" "
			<< "Function=\"JOIN($0, $1, $2)\" "
			<< "Dimensions=\"" << NF_size << " 3\" "
			<< "NumberType=\"Float\" Precision=\"4\">\n";

		// x pos
		ofs << "          <DataItem Dimensions=\"" << NF_size
			<< "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
		ofs << "            " << baseName_h5
			<< ":/State/fluidParticles/xPosF\n";
		ofs << "          </DataItem>\n";

		// y pos
		ofs << "          <DataItem Dimensions=\"" << NF_size
			<< "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
		ofs << "            " << baseName_h5
			<< ":/State/fluidParticles/yPosF\n";
		ofs << "          </DataItem>\n";

		// z pos
		ofs << "          <DataItem Dimensions=\"" << NF_size
			<< "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
		ofs << "            " << baseName_h5
			<< ":/State/fluidParticles/zPosF\n";
		ofs << "          </DataItem>\n";
		ofs << "        </DataItem>\n";
		ofs << "      </Geometry>\n";

		// velocity magnitude
		ofs << "      <Attribute Name=\"velMagF\" "
			<< "AttributeType=\"Scalar\" Center=\"Node\">\n";
		ofs << "        <DataItem Dimensions=\"" << NF_size
			<< "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
		ofs << "          " << baseName_h5
			<< ":/State/fluidParticles/velMagF\n";
		ofs << "        </DataItem>\n";
		ofs << "      </Attribute>\n";

		// close everything
		ofs << "    </Grid>\n";
		ofs << "  </Domain>\n";
		ofs << "</Xdmf>\n";
	}

} // namespace Export
