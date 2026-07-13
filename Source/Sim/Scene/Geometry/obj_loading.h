// obj_loading.h

#pragma once

// external
#include <filesystem>
#include <string>

// internal
#include "triangle_surface.h"



namespace Scene
{
	// loads triangle positions and faces from an OBJ model asset
	class ObjMeshLoader
	{
	public:

		explicit ObjMeshLoader(std::filesystem::path modelDirectory);

		TriangleSurfaceMesh Load(const std::string& modelFile) const;

	private:

		std::filesystem::path modelDirectory_;
	};

} // namespace Scene
