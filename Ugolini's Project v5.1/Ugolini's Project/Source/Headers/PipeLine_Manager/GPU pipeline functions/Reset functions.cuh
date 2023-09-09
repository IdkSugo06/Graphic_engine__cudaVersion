#pragma once
#include "Pipeline functions general defines.cuh"


__global__ void ResetRGBData(uint8* p_rgbMap);
__global__ void ResetFragmentData(Fragment* p_fragmentMap);
__global__ void ResetTriangleData(GPU_Triangle* p_triangleArray, const uint32 triangleArrayLenght);


//-- -- -- -- -- -- -- -- -- -- -- -- CODE -- -- -- -- -- -- -- -- -- -- -- -- 

__global__ void ResetRGBData(uint8* p_rgbMap) {//not used anymore
	uint32 id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id * PIXEL_PER_KERNEL >= SCREEN_PIXELNUMBER) return;
	for (uint32 pixelId = id * PIXEL_PER_KERNEL; pixelId < (id + 1) * PIXEL_PER_KERNEL; pixelId++) {
		p_rgbMap[pixelId * BMP_IMAGE_BYTEPP] = 0;
		p_rgbMap[pixelId * BMP_IMAGE_BYTEPP + 1] = 0;
		p_rgbMap[pixelId * BMP_IMAGE_BYTEPP + 2] = 0;
	}

}
__global__ void ResetFragmentData(Fragment* p_fragmentMap) {
	uint32 id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id * PIXEL_PER_KERNEL >= SCREEN_PIXELNUMBER)
		return;
	for (uint32 pixelId = id * PIXEL_PER_KERNEL; pixelId < (id + 1) * PIXEL_PER_KERNEL; pixelId++) {
		p_fragmentMap[pixelId].m_valid = false;
		p_fragmentMap[pixelId].m_relToCam_point.z = INFINITE;
	}
}
__global__ void ResetTriangleData(GPU_Triangle* p_triangleArray, const uint32 triangleArrayLenght) {
	uint32 id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= triangleArrayLenght)
		return;
	CLEAR_BIT(p_triangleArray[id].flag, TF_BP__VALIDITY);
}