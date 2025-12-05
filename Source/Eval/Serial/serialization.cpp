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

	inline std::string FormatStepPadded(int64_t step, uint32_t width = 9)
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

	void SerializationMgr::SaveUniversalSimState(
		const std::string& path,
		const Sync::UniversalSimState& state)
	{
		// file path handling
		const std::filesystem::path dstPath{ path };
		const std::filesystem::path tmpPath = saveOptions_.atomicWrite ?
			std::filesystem::path(dstPath.string() + saveOptions_.tempSuffix) : dstPath;

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

				Write1D(gF, "xAcc", F.xAcc, saveOptions_);
				Write1D(gF, "yAcc", F.yAcc, saveOptions_);
				Write1D(gF, "zAcc", F.zAcc, saveOptions_);

				Write1D(gF, "xAccPress", F.xAccPress, saveOptions_);
				Write1D(gF, "yAccPress", F.yAccPress, saveOptions_);
				Write1D(gF, "zAccPress", F.zAccPress, saveOptions_);

				Write1D(gF, "xAccVisc", F.xAccVisc, saveOptions_);
				Write1D(gF, "yAccVisc", F.yAccVisc, saveOptions_);
				Write1D(gF, "zAccVisc", F.zAccVisc, saveOptions_);

				Write1D(gF, "xAccExt", F.xAccExt, saveOptions_);
				Write1D(gF, "yAccExt", F.yAccExt, saveOptions_);
				Write1D(gF, "zAccExt", F.zAccExt, saveOptions_);

				Write1D(gF, "sourceTerm", F.sourceTerm, saveOptions_);
				Write1D(gF, "diagElem", F.diagElem, saveOptions_);
				Write1D(gF, "sysVecElem", F.sysVecElem, saveOptions_);
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

				Write1D(gB, "xVel", B.xVel, saveOptions_);
				Write1D(gB, "yVel", B.yVel, saveOptions_);
				Write1D(gB, "zVel", B.zVel, saveOptions_);

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
		const std::string& path,
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

	void SerializationMgr::SaveExportSnapshot(const Sync::ExportSnapshot& snap, const std::string& dir)
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

		// file path handling
		const std::filesystem::path dstPath = saveOptions_.baseDir / dir /  baseName;
		const std::filesystem::path tmpPath = saveOptions_.atomicWrite
			? std::filesystem::path(dstPath.string() + saveOptions_.tempSuffix)
			: dstPath;

		// open scope for writer
		{
			// open file (optionally with alignment)
			File file = OpenFileForWrite(tmpPath.string());

			// metadata
			auto root = file.createGroup("/State");
			root.createAttribute("step", snap.simStep);
			root.createAttribute("time", snap.simTime);

			{
				auto gF = root.createGroup("fluidParticles");

				gF.createAttribute("N_particles", snap.N_F);

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

		//SPDLOG_INFO("Exported sim snapshot for step {} to {}", snap.simStep, dstPath.string());
	}

	void SerializationMgr::SaveExportSnapXdmfInfo(const Sync::ExportSnapshot& snap, const std::string& dir)
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
			SPDLOG_ERROR("Failed to open XDMF file for write: {}", xdmfPath.string());
			return;
		}

		const uint32_t NF = snap.N_F;

		// XDMF header
		ofs << R"(<?xml version="1.0" ?>)" << "\n";
		ofs << R"(<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>)" << "\n";
		ofs << R"(<Xdmf Version="2.0">)" << "\n";
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
			<< "Dimensions=\"" << NF << "\"/>\n";

		// geometry of XYZ coordinates
		ofs << "      <Geometry GeometryType=\"XYZ\">\n";
		ofs << "        <DataItem ItemType=\"Function\" "
			<< "Function=\"JOIN($0, $1, $2)\" "
			<< "Dimensions=\"" << NF << " 3\" "
			<< "NumberType=\"Float\" Precision=\"4\">\n";

		// x pos
		ofs << "          <DataItem Dimensions=\"" << NF
			<< "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
		ofs << "            " << baseName_h5
			<< ":/State/fluidParticles/xPosF\n";
		ofs << "          </DataItem>\n";

		// y pos
		ofs << "          <DataItem Dimensions=\"" << NF
			<< "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
		ofs << "            " << baseName_h5
			<< ":/State/fluidParticles/yPosF\n";
		ofs << "          </DataItem>\n";

		// z pos
		ofs << "          <DataItem Dimensions=\"" << NF
			<< "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
		ofs << "            " << baseName_h5
			<< ":/State/fluidParticles/zPosF\n";
		ofs << "          </DataItem>\n";
		ofs << "        </DataItem>\n";
		ofs << "      </Geometry>\n";

		// velocity magnitude
		ofs << "      <Attribute Name=\"velMagF\" "
			<< "AttributeType=\"Scalar\" Center=\"Node\">\n";
		ofs << "        <DataItem Dimensions=\"" << NF
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
