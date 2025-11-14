// neighbor_search_kernels.cu

// external

// internal
#include "Source/Sim/GPU/gpu_utils.h"
#include "neighbor_search_kernels.cuh"
#include "neighbor_search.h"



namespace GPU01
{
	__constant__ float c_cellSize;
	__constant__ float c_cellSizeInv;
	__constant__ float c_xBoxOrig;
	__constant__ float c_yBoxOrig;
	__constant__ float c_zBoxOrig;
	__constant__ uint32_t c_N_DimX;
	__constant__ uint32_t c_N_DimY;
	__constant__ uint32_t c_N_DimZ;
	__constant__ uint32_t c_N_Cells;

	inline void Sim::NS_CellLinkedList_Basic_GPU01::ReloadConstants(
		float cellSize, float xBoxOrig, float yBoxOrig, float zBoxOrig,
		uint32_t N_DimX, uint32_t N_DimY, uint32_t N_DimZ, uint32_t N_Cells,
		cudaStream_t stream)
	{
		float cellSizeInv = 1.0f / cellSize;

		// store values in constant memory on the GPU
		GPU::Check(cudaMemcpyToSymbol(c_cellSize, &cellSize, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_cellSizeInv, &cellSizeInv, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_xBoxOrig, &xBoxOrig, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_yBoxOrig, &yBoxOrig, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_zBoxOrig, &zBoxOrig, sizeof(float)));
		GPU::Check(cudaMemcpyToSymbol(c_N_DimX, &N_DimX, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_N_DimY, &N_DimY, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_N_DimZ, &N_DimZ, sizeof(uint32_t)));
		GPU::Check(cudaMemcpyToSymbol(c_N_Cells, &N_Cells, sizeof(uint32_t)));

		GPU::StreamSync(stream);
	}

} // namespace GPU01
