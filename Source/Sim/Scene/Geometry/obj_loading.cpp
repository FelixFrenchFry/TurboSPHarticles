// obj_loading.cpp

// external
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

// internal
#include "obj_loading.h"



namespace Scene
{
	ObjMeshLoader::ObjMeshLoader(std::filesystem::path modelDirectory)
		:
		modelDirectory_(std::move(modelDirectory))
	{}

	TriangleSurfaceMesh ObjMeshLoader::Load(const std::string& modelFile) const
	{
		// validate the model path before resolving it against the asset directory
		const std::filesystem::path relativePath(modelFile);
		if (modelFile.empty() || relativePath.is_absolute())
		{
			throw std::runtime_error("Rigid mesh model file must be a non-empty relative path.");
		}
		for (const auto& component : relativePath)
		{
			if (component == "..")
			{
				throw std::runtime_error("Rigid mesh model file must remain inside the model directory.");
			}
		}
		if (relativePath.extension() != ".obj")
		{
			throw std::runtime_error("Rigid mesh model file must use the .obj extension.");
		}

		// open the normalized model path inside the configured directory
		const std::filesystem::path path =
			(modelDirectory_ / relativePath.lexically_normal()).lexically_normal();
		std::ifstream ifs(path);
		if (!ifs)
		{
			throw std::runtime_error("Could not open rigid mesh model file '" + path.string() + "'.");
		}

		// parse vertex positions and polygon faces line by line
		TriangleSurfaceMesh mesh;
		std::string line;
		uint32_t lineNumber = 0;

		while (std::getline(ifs, line))
		{
			lineNumber++;
			std::istringstream lineStream(line);
			std::string keyword;
			lineStream >> keyword;
			if (keyword.empty() || keyword[0] == '#') { continue; }

			if (keyword == "v")
			{
				glm::vec3 vertex{ 0.0f };
				if (!(lineStream >> vertex.x >> vertex.y >> vertex.z) ||
					!std::isfinite(vertex.x) ||
					!std::isfinite(vertex.y) ||
					!std::isfinite(vertex.z))
				{
					throw std::runtime_error("Rigid mesh OBJ contains an invalid vertex at line " +
						std::to_string(lineNumber) + ".");
				}
				if (mesh.vertices.size() >= std::numeric_limits<uint32_t>::max())
				{
					throw std::runtime_error("Rigid mesh OBJ vertex count exceeds uint32 limits.");
				}
				mesh.vertices.push_back(vertex);
				continue;
			}

			if (keyword != "f") { continue; }

			// resolve absolute and relative OBJ vertex indices for this face
			std::vector<uint32_t> faceIndices;
			std::string token;
			while (lineStream >> token)
			{
				if (!token.empty() && token[0] == '#') { break; }

				const std::string indexToken = token.substr(0, token.find('/'));
				if (indexToken.empty())
				{
					throw std::runtime_error("Rigid mesh OBJ contains an invalid face at line " +
						std::to_string(lineNumber) + ".");
				}

				int64_t objIndex = 0;
				try
				{
					size_t parsedChars = 0;
					objIndex = std::stoll(indexToken, &parsedChars);
					if (parsedChars != indexToken.size()) { throw std::invalid_argument("trailing characters"); }
				}
				catch (const std::exception&)
				{
					throw std::runtime_error("Rigid mesh OBJ contains an invalid face index at line " +
						std::to_string(lineNumber) + ".");
				}
				if (objIndex == 0)
				{
					throw std::runtime_error("Rigid mesh OBJ contains a zero face index at line " +
						std::to_string(lineNumber) + ".");
				}

				const int64_t vertexCount = static_cast<int64_t>(mesh.vertices.size());
				const int64_t vertexIdx = objIndex > 0
					? objIndex - 1
					: vertexCount + objIndex;
				if (vertexIdx < 0 || vertexIdx >= vertexCount)
				{
					throw std::runtime_error("Rigid mesh OBJ face index is out of range at line " +
						std::to_string(lineNumber) + ".");
				}
				faceIndices.push_back(static_cast<uint32_t>(vertexIdx));
			}

			if (faceIndices.size() < 3)
			{
				throw std::runtime_error("Rigid mesh OBJ contains a face with fewer than three vertices at line " +
					std::to_string(lineNumber) + ".");
			}

			// triangulate polygonal faces as a fan rooted at the first vertex
			for (uint32_t i = 1; i + 1 < faceIndices.size(); i++)
			{
				if (mesh.triangles.size() >= std::numeric_limits<uint32_t>::max())
				{
					throw std::runtime_error("Rigid mesh OBJ triangle count exceeds uint32 limits.");
				}
				mesh.triangles.emplace_back(
					faceIndices[0],
					faceIndices[i],
					faceIndices[i + 1]);
			}
		}

		// require at least one complete triangle surface
		if (mesh.vertices.empty() || mesh.triangles.empty())
		{
			throw std::runtime_error("Rigid mesh OBJ does not contain a triangle surface.");
		}

		return mesh;
	}

} // namespace Scene
