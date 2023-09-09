#pragma once
#include "Pipeline functions general defines.cuh"


__global__ void Rasterizzation__subdividedSquareRasterizzation(GPU_subdividedRasterizzationInfo* p_sdSqI, Fragment* p_fragMap, uint32 lenght);
__global__ void Rasterizzation__triangleDivision(GPU_Triangle* p_triangleBuffer, GPU_subdividedRasterizzationInfo* p_sdSqI, Fragment* p_fragment, uint32 lenght);
__device__ void Rasterizzation__Line(GPU_Triangle tri, Fragment* p_fragmentMap, uint16 minX, uint16 maxX, uint16 y, float w1, float w2, float w1xStep, float w2xStep);


//-- -- -- -- -- -- -- -- -- -- -- -- CODE -- -- -- -- -- -- -- -- -- -- -- -- 
__global__ void Rasterizzation__triangleDivision(GPU_Triangle* p_triangleBuffer, GPU_subdividedRasterizzationInfo* p_sdSqI, Fragment* p_fragment, uint32 lenght) {
	uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
	p_sdSqI[idx].isValid = false;
	if (idx >= lenght) return;
	p_sdSqI[idx].Create(&(p_triangleBuffer[idx]), p_fragment);
}

__global__ void Rasterizzation__subdividedSquareRasterizzation(GPU_subdividedRasterizzationInfo* p_sdSqI, Fragment* p_fragMap, uint32 lenght) {
	uint32 realIdx = (threadIdx.x + blockIdx.x * blockDim.x);
	if (realIdx >= lenght) return;
	uint32 idx = realIdx / GPU_RST_NUMOF_THREADS_PER_TRI;
	uint16 threadId = realIdx % GPU_RST_NUMOF_THREADS_PER_TRI;

	//if (realIdx > 1024) return;
	//Switch the "2d array" columns and rows //I couldnt write in in english :P
	//I triangoli clippati sono più rari dei non clippati, questo scambio di colonne e righe
	//previene l'altrimenti comune caso in cui un blocco di thread viene speso nel rasterizzare
	//una porzione di triangleBuffer (il quale, ricordo, archivia in sequenza i derivati del clipping del triangolo) 
	//di un triangolo non clippato, quindi 3/4 dei thread sarebbero fermi perchè il triangolo non era valido
	idx = (idx / MAX_THREAD_GPU) + ((idx % MAX_THREAD_GPU) * GPU_TRIANGLEBUFFER_LENGHT);

	GPU_subdividedRasterizzationInfo* squareInfo = &(p_sdSqI[idx]);
	if (!squareInfo->isValid) return;
	if (!(squareInfo->tri)) return;
	if (!CHECK_BIT(squareInfo->tri->flag, TF_BP__VALIDITY)) return;

	uint16 startingX, startingY, endingX, endingY; float w1, w2;
	{
		uint32 tempX = threadId % RASTERIZZATION__SUBDIVSQUARE_NUMOF_COLOUMN;
		uint32 tempY = threadId / RASTERIZZATION__SUBDIVSQUARE_NUMOF_ROW;

		startingX = squareInfo->squareLenght * tempX;
		startingY = squareInfo->squareHeight * tempY;
	}

	w1 = squareInfo->startingW1 + squareInfo->w1_xStep * startingX + squareInfo->w1_yStep * startingY;
	w2 = squareInfo->startingW2 + squareInfo->w2_xStep * startingX + squareInfo->w2_yStep * startingY;

	startingX += squareInfo->minX;
	startingY += squareInfo->minY;
	endingX = startingX + squareInfo->squareLenght + 1;
	endingY = startingY + squareInfo->squareHeight + 1;

	if (SHOW_SUBDIVIDED_SQUARED) { endingX -= 2; endingY -= 2; }
	GPU_Triangle _tri = *(squareInfo->tri);
	float w1ys = squareInfo->w1_yStep, w2ys = squareInfo->w2_yStep, w1xs = squareInfo->w1_xStep, w2xs = squareInfo->w2_xStep;
	for (uint16 y = startingY; y < endingY; y++) {
		w1 += w1ys;
		w2 += w2ys;
		Rasterizzation__Line(_tri, p_fragMap, startingX, endingX, y, w1, w2, w1xs, w2xs);
	}
}

__device__ void Rasterizzation__Line(GPU_Triangle tri, Fragment* p_fragmentMap, uint16 minX, uint16 maxX, uint16 y, float w1, float w2, float w1xStep, float w2xStep) {
	float w3 = 0;
	bool pixelFound = false; 
	for (uint16 x = minX; x < maxX; x++) {
		w1 += w1xStep;
		w2 += w2xStep;
		w3 = (1.0f - w1 - w2);
		if (w1 > 0 && w2 > 0 && w3 > 0) {
			Fragment tempFrag; Fragment* p_frag = &(p_fragmentMap[x + y * SCREEN_WIDTH]);
			pixelFound = true;

			float invZComponent = w1 * tri.m_relToCam_points_interpolationInfo[1].z + w2 * tri.m_relToCam_points_interpolationInfo[2].z + w3 * tri.m_relToCam_points_interpolationInfo[0].z;
			if ((!p_frag->m_valid) || (p_frag->m_relToCam_point.z < invZComponent)) {
				tempFrag.m_valid = true;

				tempFrag.m_relToCam_point.x = w1 * tri.m_relToCam_points_interpolationInfo[1].x + w2 * tri.m_relToCam_points_interpolationInfo[2].x + w3 * tri.m_relToCam_points_interpolationInfo[0].x;
				tempFrag.m_relToCam_point.y = w1 * tri.m_relToCam_points_interpolationInfo[1].y + w2 * tri.m_relToCam_points_interpolationInfo[2].y + w3 * tri.m_relToCam_points_interpolationInfo[0].y;
				tempFrag.m_relToCam_point.z = invZComponent;

				tempFrag.m_relToCam_normal = tri.m_relToCam_normal; tempFrag.m_relToCam_tangent = tri.m_relToCam_tangent; tempFrag.m_relToCam_bitangent = tri.m_relToCam_bitangent;

				tempFrag.m_screenSpace_point.x = x;
				tempFrag.m_screenSpace_point.y = y;

				tempFrag.m_uv_coordinates.x = w1 * tri.m_uv_coordinates[1].x + w2 * tri.m_uv_coordinates[2].x + w3 * tri.m_uv_coordinates[0].x;
				tempFrag.m_uv_coordinates.y = w1 * tri.m_uv_coordinates[1].y + w2 * tri.m_uv_coordinates[2].y + w3 * tri.m_uv_coordinates[0].y;

				tempFrag.m_materialId = tri.m_materialId;
				*p_frag = tempFrag;
			}
		}
		else if (pixelFound) {
			return;
		}
	}
}