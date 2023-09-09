#pragma once
#include "Pipeline functions general defines.cuh"


//It'll project and clip the triangles, and put the final triangles in the triangleBuffer, in order to be taken by the rasterizer
__device__ bool Projection(GPU_Triangle* tri, const GPU_camInfo* p_ci);
//It'll clip and put the clipped triangles into a buffer (max lenght = 8) (for every triangles in the buffer it has to set the validity bit)
__global__ void Clipping(const GPU_camInfo* p_ci, const GPU_Triangle* triangles, GPU_Triangle* triangleBuffer, Fragment* p_fragmentMap, const uint32 triangleArrayLenght, const uint32 i_clippingCicle);
__device__ void Clipping__clip(GPU_Triangle* triBuffer);
//Left side functions
__device__ void Clipping__leftSide(GPU_Triangle* triangleBuffer, uint16& trianglesNumber);
__device__ float Clipping__edgeIntersection__leftSide(Vector2& p1, Vector2& p2);
__device__ void Clipping__buildClippedTriangle__leftSide__1pointOutside(GPU_Triangle* triangleBuffer, uint8 pOId, uint8 pI1Id, uint8 pI2Id);
__device__ void Clipping__buildClippedTriangle__leftSide__2pointOutside(GPU_Triangle* triangleBuffer, uint8 pIId, uint8 pO1Id, uint8 pO2Id);
//Upper side functions
__device__ void Clipping__upperSide(GPU_Triangle* triangleBuffer, uint16& trianglesNumber);
__device__ void Clipping__upperSide__cicle(GPU_Triangle* triangleBuffer, uint8& trianglesFound, uint8& boundary, uint8& trianglesCreated, uint8& i);
__device__ float Clipping__edgeIntersection__upperSide(Vector2& p1, Vector2& p2);
__device__ void Clipping__buildClippedTriangle__upperSide__1pointOutside(GPU_Triangle* triangleBuffer, uint8 pOId, uint8 pI1Id, uint8 pI2Id, uint8 tb_1, uint8 tb_2);
__device__ void Clipping__buildClippedTriangle__upperSide__2pointOutside(GPU_Triangle* triangleBuffer, uint8 pIId, uint8 pO1Id, uint8 pO2Id, uint8 tb_1);
//Right side functions
__device__ void Clipping__rightSide(GPU_Triangle* triangleBuffer, uint16& trianglesNumber);
__device__ void Clipping__rightSide__cicle(GPU_Triangle* triangleBuffer, uint8& trianglesFound, uint8& boundary, uint8& trianglesCreated, uint8& i);
__device__ float Clipping__edgeIntersection__rightSide(Vector2& p1, Vector2& p2);
__device__ void Clipping__buildClippedTriangle__rightSide__1pointOutside(GPU_Triangle* triangleBuffer, uint8 pOId, uint8 pI1Id, uint8 pI2Id, uint8 tb_1, uint8 tb_2);
__device__ void Clipping__buildClippedTriangle__rightSide__2pointOutside(GPU_Triangle* triangleBuffer, uint8 pIId, uint8 pO1Id, uint8 pO2Id, uint8 tb_1);
//Down side functions
__device__ void Clipping__downSide(GPU_Triangle* triangleBuffer, uint16& trianglesNumber);
__device__ void Clipping__downSide__cicle(GPU_Triangle* triangleBuffer, uint8& trianglesFound, uint8& boundary, uint8& trianglesCreated, uint8& i);
__device__ float Clipping__edgeIntersection__downSide(Vector2& p1, Vector2& p2);
__device__ void Clipping__buildClippedTriangle__downSide__1pointOutside(GPU_Triangle* triangleBuffer, uint8 pOId, uint8 pI1Id, uint8 pI2Id, uint8 tb_1, uint8 tb_2);
__device__ void Clipping__buildClippedTriangle__downSide__2pointOutside(GPU_Triangle* triangleBuffer, uint8 pIId, uint8 pO1Id, uint8 pO2Id, uint8 tb_1);
//Other functions
__device__ void Clipping__attributesInterpolation__verticalAxisIntersection__onePointOut(GPU_Triangle* triangle0, GPU_Triangle* triangle1, uint8 pOId, uint8 pI1Id, uint8 pI2Id, float t1, float t2, uint16 boundary); 
__device__ void Clipping__attributesInterpolation__horizontalAxisIntersection__onePointOut(GPU_Triangle* triangle0, GPU_Triangle* triangle1, uint8 pOId, uint8 pI1Id, uint8 pI2Id, float t1, float t2, uint16 boundary); 
__device__ void Clipping__attributesInterpolation__horizontalAxisIntersection__onePointOutPt2(GPU_Triangle* triangle0, GPU_Triangle* triangle1, uint8 pOId, uint8 pI1Id, uint8 pI2Id, float t1, float t2, uint16 boundary);


//-- -- -- -- -- -- -- -- -- -- -- -- CODE -- -- -- -- -- -- -- -- -- -- -- -- 
#define CLIPPING__LEFT_BOUNDARY 0
#define CLIPPING__UPPER_BOUNDARY (SCREEN_HEIGHT)
#define CLIPPING__RIGHT_BOUNDARY (SCREEN_WIDTH)
#define CLIPPING__DOWN_BOUNDARY 0


//Main functions
__device__ bool Projection(GPU_Triangle* tri, const GPU_camInfo* p_ci) {
	//Projection
	float projection_coefficent = (tri->m_relToCam_points[0].z * p_ci->m_TanFov);
	if (abs(projection_coefficent) < 0.00001f) return false;
	tri->m_screenSpace_points[0].x = (((tri->m_relToCam_points[0].x / projection_coefficent) * 0.5f) + 0.5f) * p_ci->m_screen_width;
	tri->m_screenSpace_points[0].y = (((tri->m_relToCam_points[0].y * p_ci->m_aspect_ratio / projection_coefficent) * 0.5f) + 0.5f) * p_ci->m_screen_height;

	projection_coefficent = (tri->m_relToCam_points[1].z * p_ci->m_TanFov);
	if (abs(projection_coefficent) < 0.00001f) return false;
	tri->m_screenSpace_points[1].x = (((tri->m_relToCam_points[1].x / projection_coefficent) * 0.5f) + 0.5f) * p_ci->m_screen_width;
	tri->m_screenSpace_points[1].y = (((tri->m_relToCam_points[1].y * p_ci->m_aspect_ratio / projection_coefficent) * 0.5f) + 0.5f) * p_ci->m_screen_height;

	projection_coefficent = (tri->m_relToCam_points[2].z * p_ci->m_TanFov);
	if (abs(projection_coefficent) < 0.00001f) return false;
	tri->m_screenSpace_points[2].x = (((tri->m_relToCam_points[2].x / projection_coefficent) * 0.5f) + 0.5f) * p_ci->m_screen_width;
	tri->m_screenSpace_points[2].y = (((tri->m_relToCam_points[2].y * p_ci->m_aspect_ratio / projection_coefficent) * 0.5f) + 0.5f) * p_ci->m_screen_height;
	return true;
}
__global__ void Clipping(const GPU_camInfo* p_ci, const GPU_Triangle* triangles, GPU_Triangle* triangleBuffer, Fragment* p_fragmentMap, const uint32 triangleArrayLenght, const uint32 i_clippingCicle) {
	uint32 partialId = threadIdx.x + blockDim.x * blockIdx.x;
	uint32 id = partialId + i_clippingCicle * MAX_THREAD_GPU;
	if (id >= triangleArrayLenght) return;
	uint32 tbId = partialId * GPU_TRIANGLEBUFFER_LENGHT;
	if (!CHECK_BIT(triangles[id].flag, TF_BP__VALIDITY)) {
		for (uint8 i = 0; i < GPU_TRIANGLEBUFFER_LENGHT; i++) { CLEAR_BIT(triangleBuffer[tbId + i].flag, TF_BP__VALIDITY); } return;
	}
	triangleBuffer[tbId] = triangles[id]; //Remember to set the validity bit to true for each triangle in the buffer when clipping

	//Projection
	if (!Projection(&(triangleBuffer[tbId]), p_ci)) return;
	SET_BIT(triangleBuffer[tbId].flag, TF_BP__VALIDITY);
	for (uint8 i = 1; i < GPU_TRIANGLEBUFFER_LENGHT; i++) { CLEAR_BIT(triangleBuffer[tbId + i].flag, TF_BP__VALIDITY); }

	Clipping__clip(&(triangleBuffer[tbId]));
}
__device__ void Clipping__clip(GPU_Triangle* triBuffer) {
	uint16 trianglesNumber = 1;
	Clipping__leftSide(triBuffer, trianglesNumber);
	Clipping__upperSide(triBuffer, trianglesNumber);
	Clipping__rightSide(triBuffer, trianglesNumber);
	Clipping__downSide(triBuffer, trianglesNumber);
}


//Specific side clipping
__device__ void Clipping__leftSide(GPU_Triangle* triangleBuffer, uint16& trianglesNumber) { //trianglesNumber has to be 1
	//Check how many vertecies are outside
	uint8 numOf_pointsOutside = 0;
	bool pointsOutside[3] = { false, false, false };
	if (triangleBuffer->m_screenSpace_points[0].x < CLIPPING__LEFT_BOUNDARY) { numOf_pointsOutside++;	pointsOutside[0] = true; }
	if (triangleBuffer->m_screenSpace_points[1].x < CLIPPING__LEFT_BOUNDARY) { numOf_pointsOutside++;	pointsOutside[1] = true; }
	if (triangleBuffer->m_screenSpace_points[2].x < CLIPPING__LEFT_BOUNDARY) { numOf_pointsOutside++;	pointsOutside[2] = true; }

	if (numOf_pointsOutside == 0) return;
	else if (numOf_pointsOutside == 1) {
		//Search the id of the outside points
		uint8 pI1Id, pI2Id, pOId; //Tri1 = (pI1Id, T1, pI2Id), Tri2 = (pI2Id, T1, T2); T1 = p2 -> p0; T2 = p1 -> p0
		if (pointsOutside[0]) { pOId = 0; pI1Id = 2; pI2Id = 1; }
		else {
			if (pointsOutside[1]) { pOId = 1; pI1Id = 0; pI2Id = 2; }
			else { pOId = 2; pI1Id = 1; pI2Id = 0; }
		}

		Clipping__buildClippedTriangle__leftSide__1pointOutside(triangleBuffer, pOId, pI1Id, pI2Id);
		trianglesNumber = 2;
	}
	else if (numOf_pointsOutside == 2) {
		//Search the id of the outside points
		uint8 pO1Id, pO2Id, pIId;
		if (pointsOutside[0]) {
			if (pointsOutside[1]) { pO1Id = 0; pO2Id = 1; pIId = 2; }
			else { pO1Id = 2; pO2Id = 0; pIId = 1; }
		}
		else { pIId = 0; pO1Id = 1; pO2Id = 2; }

		Clipping__buildClippedTriangle__leftSide__2pointOutside(triangleBuffer, pIId, pO1Id, pO2Id);
		trianglesNumber = 1;
	}
	else if (numOf_pointsOutside == 3) {
		trianglesNumber = 0;
		CLEAR_BIT(triangleBuffer[0].flag, TF_BP__VALIDITY);
		return;
	}
}
__device__ void Clipping__upperSide(GPU_Triangle* triangleBuffer, uint16& trianglesNumber) {
	uint8 boundary = trianglesNumber, trianglesCreated = 0, trianglesFound = 0;

	for (uint8 i = 0; i < trianglesNumber; i++) {
		if (CHECK_BIT(triangleBuffer[i].flag, TF_BP__VALIDITY)) {
			Clipping__upperSide__cicle(triangleBuffer, trianglesFound, boundary, trianglesCreated, i);
		}
	}
	trianglesNumber = boundary + trianglesCreated;
}
__device__ void Clipping__rightSide(GPU_Triangle* triangleBuffer, uint16& trianglesNumber) {
	uint8 boundary = trianglesNumber, trianglesCreated = 0, trianglesFound = 0;

	//! The law that defines the new triangles is: {Tri1 = (pI1Id, T1, pI2Id), Tri2 = (pI2Id, T1, T2); T1 = p2->p0; T2 = p1->p0}
	for (uint8 i = 0; i < trianglesNumber; i++) {
		if (CHECK_BIT(triangleBuffer[i].flag, TF_BP__VALIDITY)) {
			Clipping__rightSide__cicle(triangleBuffer, trianglesFound, boundary, trianglesCreated, i);
		}
	}
	trianglesNumber = boundary + trianglesCreated;
}
__device__ void Clipping__downSide(GPU_Triangle* triangleBuffer, uint16& trianglesNumber) {
	uint8 boundary = trianglesNumber, trianglesCreated = 0, trianglesFound = 0;

	//printf("%d\n", trianglesNumber);
	//! The law that defines the new triangles is: {Tri1 = (pI1Id, T1, pI2Id), Tri2 = (pI2Id, T1, T2); T1 = p2->p0; T2 = p1->p0}
	for (uint8 i = 0; i < trianglesNumber; i++) {
		if (CHECK_BIT(triangleBuffer[i].flag, TF_BP__VALIDITY)) {
			Clipping__downSide__cicle(triangleBuffer, trianglesFound, boundary, trianglesCreated, i);
		}
	}
	trianglesNumber = boundary + trianglesCreated;
}



//This code is the core of the logical part of the operation (exept for the left side)
	// EXPLAINATION 
	//"trianglesNumber" times (in order not to change triangles that have been just put in): if the triangle is valid,
	//Check if it is on outside of the screen, and if it is, the algorithm will clip it.
	//The first triangle clipped will be put into the tb[trianglesFound], wheras the second will eventually be stored 
	//into the tb[trianglesNumber + triangleFound]
	//When a triangle is removed, all the triangles stored after the trianglesNumber boundary has to be dragged into the previous position
__device__ void Clipping__upperSide__cicle(GPU_Triangle* triangleBuffer, uint8& trianglesFound, uint8& boundary, uint8& trianglesCreated, uint8& i) {
	//Check how many vertecies are outside
	uint8 numOf_pointsOutside = 0;
	bool pointsOutside[3] = { false, false, false };
	if (triangleBuffer[i].m_screenSpace_points[0].y > CLIPPING__UPPER_BOUNDARY) { numOf_pointsOutside++;	pointsOutside[0] = true; }
	if (triangleBuffer[i].m_screenSpace_points[1].y > CLIPPING__UPPER_BOUNDARY) { numOf_pointsOutside++;	pointsOutside[1] = true; }
	if (triangleBuffer[i].m_screenSpace_points[2].y > CLIPPING__UPPER_BOUNDARY) { numOf_pointsOutside++;	pointsOutside[2] = true; }

	//Different cases 
	if (numOf_pointsOutside == 0) {
		trianglesFound++;
	}
	else if (numOf_pointsOutside == 1) {
		//Search the id of the outside points
		uint8 pI1Id, pI2Id, pOId;  if (pointsOutside[0]) { pOId = 0; pI1Id = 2; pI2Id = 1; } //! The law that defines the new triangles is: {Tri1 = (pI1Id, T1, pI2Id), Tri2 = (pI2Id, T1, T2); T1 = p2->p0; T2 = p1->p0}
		else { if (pointsOutside[1]) { pOId = 1; pI1Id = 0; pI2Id = 2; } else { pOId = 2; pI1Id = 1; pI2Id = 0; } }
		Clipping__buildClippedTriangle__upperSide__1pointOutside(triangleBuffer, pOId, pI1Id, pI2Id, trianglesFound, boundary + trianglesCreated);
		trianglesCreated++; trianglesFound++;
	}
	else if (numOf_pointsOutside == 2) {
		//Search the id of the outside points
		uint8 pO1Id, pO2Id, pIId; if (pointsOutside[0]) { if (pointsOutside[1]) { pO1Id = 0; pO2Id = 1; pIId = 2; } else { pO1Id = 2; pO2Id = 0; pIId = 1; } }
		else { pIId = 0; pO1Id = 1; pO2Id = 2; }

		Clipping__buildClippedTriangle__upperSide__2pointOutside(triangleBuffer, pIId, pO1Id, pO2Id, i);
		trianglesFound++;
	}
	else if (numOf_pointsOutside == 3) {
		for (uint8 j = trianglesFound; j < (trianglesCreated + boundary - 1); j++) {
			triangleBuffer[j] = triangleBuffer[j + 1];
		}
		CLEAR_BIT(triangleBuffer[(trianglesCreated + boundary - 1)].flag, TF_BP__VALIDITY);
		boundary--; i--;
	}
}
__device__ void Clipping__rightSide__cicle(GPU_Triangle* triangleBuffer, uint8& trianglesFound, uint8& boundary, uint8& trianglesCreated, uint8& i) {
	//Check how many vertecies are outside
	uint8 numOf_pointsOutside = 0;
	bool pointsOutside[3] = { false, false, false };
	if (triangleBuffer[i].m_screenSpace_points[0].x > CLIPPING__RIGHT_BOUNDARY) { numOf_pointsOutside++;	pointsOutside[0] = true; }
	if (triangleBuffer[i].m_screenSpace_points[1].x > CLIPPING__RIGHT_BOUNDARY) { numOf_pointsOutside++;	pointsOutside[1] = true; }
	if (triangleBuffer[i].m_screenSpace_points[2].x > CLIPPING__RIGHT_BOUNDARY) { numOf_pointsOutside++;	pointsOutside[2] = true; }

	//Different cases
	if (numOf_pointsOutside == 0) {
		trianglesFound++;
	}
	else if (numOf_pointsOutside == 1) {
		//Search the id of the outside points
		uint8 pI1Id, pI2Id, pOId;  if (pointsOutside[0]) { pOId = 0; pI1Id = 2; pI2Id = 1; }
		else { if (pointsOutside[1]) { pOId = 1; pI1Id = 0; pI2Id = 2; } else { pOId = 2; pI1Id = 1; pI2Id = 0; } }

		Clipping__buildClippedTriangle__rightSide__1pointOutside(triangleBuffer, pOId, pI1Id, pI2Id, trianglesFound, boundary + trianglesCreated); ///!CHANGED it was boundary+trianglesFound
		trianglesCreated++; trianglesFound++;
	}
	else if (numOf_pointsOutside == 2) {
		//Search the id of the outside points
		uint8 pO1Id, pO2Id, pIId; if (pointsOutside[0]) { if (pointsOutside[1]) { pO1Id = 0; pO2Id = 1; pIId = 2; } else { pO1Id = 2; pO2Id = 0; pIId = 1; } }
		else { pIId = 0; pO1Id = 1; pO2Id = 2; }

		Clipping__buildClippedTriangle__rightSide__2pointOutside(triangleBuffer, pIId, pO1Id, pO2Id, i);
		trianglesFound++;
	}
	else if (numOf_pointsOutside == 3) {
		for (uint8 j = trianglesFound; j < (trianglesCreated + boundary - 1); j++) {
			triangleBuffer[j] = triangleBuffer[j + 1];
		}
		CLEAR_BIT(triangleBuffer[(trianglesCreated + boundary - 1)].flag, TF_BP__VALIDITY);
		boundary--; i--;
	}
}
__device__ void Clipping__downSide__cicle(GPU_Triangle* triangleBuffer, uint8& trianglesFound, uint8& boundary, uint8& trianglesCreated, uint8& i) {
	//Check how many vertecies are outside
	uint8 numOf_pointsOutside = 0;
	bool pointsOutside[3] = { false, false, false };
	if (triangleBuffer[i].m_screenSpace_points[0].y < CLIPPING__DOWN_BOUNDARY) { numOf_pointsOutside++;	pointsOutside[0] = true; }
	if (triangleBuffer[i].m_screenSpace_points[1].y < CLIPPING__DOWN_BOUNDARY) { numOf_pointsOutside++;	pointsOutside[1] = true; }
	if (triangleBuffer[i].m_screenSpace_points[2].y < CLIPPING__DOWN_BOUNDARY) { numOf_pointsOutside++;	pointsOutside[2] = true; }

	//Different cases 
	if (numOf_pointsOutside == 0) {
		trianglesFound++;
	}
	else if (numOf_pointsOutside == 1) {
		//Search the id of the outside points
		uint8 pI1Id, pI2Id, pOId;  if (pointsOutside[0]) { pOId = 0; pI1Id = 2; pI2Id = 1; } //! The law that defines the new triangles is: {Tri1 = (pI1Id, T1, pI2Id), Tri2 = (pI2Id, T1, T2); T1 = p2->p0; T2 = p1->p0}
		else { if (pointsOutside[1]) { pOId = 1; pI1Id = 0; pI2Id = 2; } else { pOId = 2; pI1Id = 1; pI2Id = 0; } }
		Clipping__buildClippedTriangle__downSide__1pointOutside(triangleBuffer, pOId, pI1Id, pI2Id, trianglesFound, boundary + trianglesCreated);
		trianglesCreated++; trianglesFound++;
	}
	else if (numOf_pointsOutside == 2) {
		//Search the id of the outside points
		uint8 pO1Id, pO2Id, pIId; if (pointsOutside[0]) { if (pointsOutside[1]) { pO1Id = 0; pO2Id = 1; pIId = 2; } else { pO1Id = 2; pO2Id = 0; pIId = 1; } }
		else { pIId = 0; pO1Id = 1; pO2Id = 2; }
		Clipping__buildClippedTriangle__downSide__2pointOutside(triangleBuffer, pIId, pO1Id, pO2Id, i);
		trianglesFound++;
	}
	else if (numOf_pointsOutside == 3) {
		for (uint8 j = trianglesFound; j < (trianglesCreated + boundary - 1); j++) {
			triangleBuffer[j] = triangleBuffer[j + 1];
		}
		CLEAR_BIT(triangleBuffer[(trianglesCreated + boundary - 1)].flag, TF_BP__VALIDITY);
		boundary--; i--;
	}
}

//Edge intersection functions
__device__ float Clipping__edgeIntersection__leftSide(Vector2& p1, Vector2& p2) { //p3 = t * p2 + (1 - t) * p1
	return (CLIPPING__LEFT_BOUNDARY - p2.x) / (p1.x - p2.x); //Cannot be the same, one of them is inside and the other's outside the screen, no x / 0 error can occur
}
__device__ float Clipping__edgeIntersection__upperSide(Vector2& p1, Vector2& p2) { //p3 = t * p2 + (1 - t) * p1
	return (CLIPPING__UPPER_BOUNDARY - p2.y) / (p1.y - p2.y);
}
__device__ float Clipping__edgeIntersection__rightSide(Vector2& p1, Vector2& p2) { //p3 = t * p2 + (1 - t) * p1
	return (CLIPPING__RIGHT_BOUNDARY - p2.x) / (p1.x - p2.x);
}
__device__ float Clipping__edgeIntersection__downSide(Vector2& p1, Vector2& p2) { //p3 = t * p2 + (1 - t) * p1
	return (CLIPPING__DOWN_BOUNDARY - p2.y) / (p1.y - p2.y);
}



__device__ void Clipping__attributesInterpolation__verticalAxisIntersection__onePointOut(GPU_Triangle* triangle0, GPU_Triangle* triangle1, uint8 pOId, uint8 pI1Id, uint8 pI2Id, float t1, float t2, uint16 boundary) {
	float ti1 = (1.0f - t1);
	float ti2 = (1.0f - t2);
	{
		//Interpolate the m_relToCam_points_interpolationInfo
		//I have to interpolate the 1/z or orthogonal view attributes to have a precise interpolation
		Vector3 T1 = triangle0->m_relToCam_points_interpolationInfo[pI1Id] * t1 + triangle0->m_relToCam_points_interpolationInfo[pOId] * ti1;
		Vector3 T2 = triangle0->m_relToCam_points_interpolationInfo[pI2Id] * t2 + triangle0->m_relToCam_points_interpolationInfo[pOId] * ti2;

		triangle0->m_relToCam_points_interpolationInfo[0] = triangle1->m_relToCam_points_interpolationInfo[pI1Id];
		triangle0->m_relToCam_points_interpolationInfo[1] = T1;
		triangle0->m_relToCam_points_interpolationInfo[2] = triangle1->m_relToCam_points_interpolationInfo[pI2Id];

		triangle1->m_relToCam_points_interpolationInfo[0] = triangle1->m_relToCam_points_interpolationInfo[pI2Id];
		triangle1->m_relToCam_points_interpolationInfo[1] = T1;
		triangle1->m_relToCam_points_interpolationInfo[2] = T2;
	}
	{
		//Interpolate the spaceScreen_vertecies
		Vector2 T1; T1.x = boundary; T1.y = triangle0->m_screenSpace_points[pI1Id].y * t1 + triangle0->m_screenSpace_points[pOId].y * ti1;
		Vector2 T2; T2.x = boundary; T2.y = triangle0->m_screenSpace_points[pI2Id].y * t2 + triangle0->m_screenSpace_points[pOId].y * ti2;

		triangle0->m_screenSpace_points[0] = triangle1->m_screenSpace_points[pI1Id];
		triangle0->m_screenSpace_points[1] = T1;
		triangle0->m_screenSpace_points[2] = triangle1->m_screenSpace_points[pI2Id];

		triangle1->m_screenSpace_points[0] = triangle1->m_screenSpace_points[pI2Id];
		triangle1->m_screenSpace_points[1] = T1;
		triangle1->m_screenSpace_points[2] = T2;

		//Interpolate the uv_coordinates
		T1 = triangle0->m_uv_coordinates[pI1Id] * t1 + triangle0->m_uv_coordinates[pOId] * ti1;
		T2 = triangle0->m_uv_coordinates[pI2Id] * t2 + triangle0->m_uv_coordinates[pOId] * ti2;

		triangle0->m_uv_coordinates[0] = triangle1->m_uv_coordinates[pI1Id];
		triangle0->m_uv_coordinates[1] = T1;
		triangle0->m_uv_coordinates[2] = triangle1->m_uv_coordinates[pI2Id];

		triangle1->m_uv_coordinates[0] = triangle1->m_uv_coordinates[pI2Id];
		triangle1->m_uv_coordinates[1] = T1;
		triangle1->m_uv_coordinates[2] = T2;
	}
}
__device__ void Clipping__attributesInterpolation__horizontalAxisIntersection__onePointOut(GPU_Triangle* triangle0, GPU_Triangle* triangle1, uint8 pOId, uint8 pI1Id, uint8 pI2Id, float t1, float t2, uint16 boundary) {
	float ti1 = (1.0f - t1);
	float ti2 = (1.0f - t2);
	{
		//Interpolate the m_relToCam_points_interpolationInfo
		//I have to interpolate the 1/z or orthogonal view attributes to have a precise interpolation
		Vector3 T1 = triangle0->m_relToCam_points_interpolationInfo[pI1Id] * t1 + triangle0->m_relToCam_points_interpolationInfo[pOId] * ti1;
		Vector3 T2 = triangle0->m_relToCam_points_interpolationInfo[pI2Id] * t2 + triangle0->m_relToCam_points_interpolationInfo[pOId] * ti2;
	
		triangle0->m_relToCam_points_interpolationInfo[0] = triangle1->m_relToCam_points_interpolationInfo[pI1Id];
		triangle0->m_relToCam_points_interpolationInfo[1] = T1;
		triangle0->m_relToCam_points_interpolationInfo[2] = triangle1->m_relToCam_points_interpolationInfo[pI2Id];
	
		triangle1->m_relToCam_points_interpolationInfo[0] = triangle1->m_relToCam_points_interpolationInfo[pI2Id];
		triangle1->m_relToCam_points_interpolationInfo[1] = T1;
		triangle1->m_relToCam_points_interpolationInfo[2] = T2;
	}
	Clipping__attributesInterpolation__horizontalAxisIntersection__onePointOutPt2(triangle0, triangle1, pOId, pI1Id, pI2Id, t1, t2, boundary);
}
__device__ void Clipping__attributesInterpolation__horizontalAxisIntersection__onePointOutPt2(GPU_Triangle* triangle0, GPU_Triangle* triangle1, uint8 pOId, uint8 pI1Id, uint8 pI2Id, float t1, float t2, uint16 boundary) {
	float ti1 = (1.0f - t1);
	float ti2 = (1.0f - t2);
	{
		//Interpolate the spaceScreen_vertecies
		Vector2 T1; T1.y = boundary; T1.x = triangle0->m_screenSpace_points[pI1Id].x * t1 + triangle0->m_screenSpace_points[pOId].x * ti1;
		Vector2 T2; T2.y = boundary; T2.x = triangle0->m_screenSpace_points[pI2Id].x * t2 + triangle0->m_screenSpace_points[pOId].x * ti2;
		
		triangle0->m_screenSpace_points[0] = triangle1->m_screenSpace_points[pI1Id];
		triangle0->m_screenSpace_points[1] = T1;
		triangle0->m_screenSpace_points[2] = triangle1->m_screenSpace_points[pI2Id];
		
		triangle1->m_screenSpace_points[0] = triangle1->m_screenSpace_points[pI2Id];
		triangle1->m_screenSpace_points[1] = T1;
		triangle1->m_screenSpace_points[2] = T2;
	
		//Interpolate the uv_coordinates
		T1 = triangle0->m_uv_coordinates[pI1Id] * t1 + triangle0->m_uv_coordinates[pOId] * ti1;
		T2 = triangle0->m_uv_coordinates[pI2Id] * t2 + triangle0->m_uv_coordinates[pOId] * ti2;
		
		triangle0->m_uv_coordinates[0] = triangle1->m_uv_coordinates[pI1Id];
		triangle0->m_uv_coordinates[1] = T1;
		triangle0->m_uv_coordinates[2] = triangle1->m_uv_coordinates[pI2Id];
		
		triangle1->m_uv_coordinates[0] = triangle1->m_uv_coordinates[pI2Id];
		triangle1->m_uv_coordinates[1] = T1;
		triangle1->m_uv_coordinates[2] = T2;
	}
}

//Build triangles functions
//Left
__device__ void Clipping__buildClippedTriangle__leftSide__1pointOutside(GPU_Triangle* triangleBuffer, uint8 pOId, uint8 pI1Id, uint8 pI2Id) {
	//Compute the interpolation value
	float t1 = Clipping__edgeIntersection__leftSide(triangleBuffer->m_screenSpace_points[pI1Id], triangleBuffer->m_screenSpace_points[pOId]);
	float t2 = Clipping__edgeIntersection__leftSide(triangleBuffer->m_screenSpace_points[pI2Id], triangleBuffer->m_screenSpace_points[pOId]);
	triangleBuffer[1] = triangleBuffer[0]; //Copy the triangle before (validity flag already set)
	Clipping__attributesInterpolation__verticalAxisIntersection__onePointOut(&(triangleBuffer[0]), &(triangleBuffer[1]), pOId, pI1Id, pI2Id, t1, t2, CLIPPING__LEFT_BOUNDARY);
}
__device__ void Clipping__buildClippedTriangle__leftSide__2pointOutside(GPU_Triangle* triangleBuffer, uint8 pIId, uint8 pO1Id, uint8 pO2Id) {
	//Compute the interpolation value
	float t1 = Clipping__edgeIntersection__leftSide(triangleBuffer->m_screenSpace_points[pIId], triangleBuffer->m_screenSpace_points[pO1Id]);
	float t2 = Clipping__edgeIntersection__leftSide(triangleBuffer->m_screenSpace_points[pIId], triangleBuffer->m_screenSpace_points[pO2Id]);
	float ti1 = (1.0f - t1);
	float ti2 = (1.0f - t2);

	{
		//Interpolate the m_relToCam_points_interpolationInfo
		Vector3 T1 = triangleBuffer[0].m_relToCam_points_interpolationInfo[pIId] * t1 + triangleBuffer[0].m_relToCam_points_interpolationInfo[pO1Id] * ti1;
		Vector3 T2 = triangleBuffer[0].m_relToCam_points_interpolationInfo[pIId] * t2 + triangleBuffer[0].m_relToCam_points_interpolationInfo[pO2Id] * ti2;

		triangleBuffer[0].m_relToCam_points_interpolationInfo[0] = triangleBuffer[0].m_relToCam_points_interpolationInfo[pIId];
		triangleBuffer[0].m_relToCam_points_interpolationInfo[1] = T1;
		triangleBuffer[0].m_relToCam_points_interpolationInfo[2] = T2;
	}
	{
		//Interpolate the spaceScreen_vertecies
		Vector2 T1; T1.x = CLIPPING__LEFT_BOUNDARY; T1.y = triangleBuffer[0].m_screenSpace_points[pIId].y * t1 + triangleBuffer[0].m_screenSpace_points[pO1Id].y * ti1;
		Vector2 T2; T2.x = CLIPPING__LEFT_BOUNDARY; T2.y = triangleBuffer[0].m_screenSpace_points[pIId].y * t2 + triangleBuffer[0].m_screenSpace_points[pO2Id].y * ti2;

		triangleBuffer[0].m_screenSpace_points[0] = triangleBuffer[0].m_screenSpace_points[pIId];
		triangleBuffer[0].m_screenSpace_points[1] = T1;
		triangleBuffer[0].m_screenSpace_points[2] = T2;

		//Interpolate the uv_coordinates
		T1 = triangleBuffer[0].m_uv_coordinates[pIId] * t1 + triangleBuffer[0].m_uv_coordinates[pO1Id] * ti1;
		T2 = triangleBuffer[0].m_uv_coordinates[pIId] * t2 + triangleBuffer[0].m_uv_coordinates[pO2Id] * ti2;

		triangleBuffer[0].m_uv_coordinates[0] = triangleBuffer[0].m_uv_coordinates[pIId];
		triangleBuffer[0].m_uv_coordinates[1] = T1;
		triangleBuffer[0].m_uv_coordinates[2] = T2;
	}
}

//Upper
__device__ void Clipping__buildClippedTriangle__upperSide__1pointOutside(GPU_Triangle* triangleBuffer, uint8 pOId, uint8 pI1Id, uint8 pI2Id, uint8 tb_1, uint8 tb_2) {
	//Compute the interpolation value
	float t1 = Clipping__edgeIntersection__upperSide(triangleBuffer[tb_1].m_screenSpace_points[pI1Id], triangleBuffer[tb_1].m_screenSpace_points[pOId]);
	float t2 = Clipping__edgeIntersection__upperSide(triangleBuffer[tb_1].m_screenSpace_points[pI2Id], triangleBuffer[tb_1].m_screenSpace_points[pOId]);
	triangleBuffer[tb_2] = triangleBuffer[tb_1]; //Copy the triangle before (validity flag already set)
	Clipping__attributesInterpolation__horizontalAxisIntersection__onePointOut(&(triangleBuffer[tb_1]), &(triangleBuffer[tb_2]), pOId, pI1Id, pI2Id, t1, t2, CLIPPING__UPPER_BOUNDARY);
}
__device__ void Clipping__buildClippedTriangle__upperSide__2pointOutside(GPU_Triangle* triangleBuffer, uint8 pIId, uint8 pO1Id, uint8 pO2Id, uint8 tb_1) {
	//Compute the interpolation value
	float t1 = Clipping__edgeIntersection__upperSide(triangleBuffer[tb_1].m_screenSpace_points[pIId], triangleBuffer[tb_1].m_screenSpace_points[pO1Id]);
	float t2 = Clipping__edgeIntersection__upperSide(triangleBuffer[tb_1].m_screenSpace_points[pIId], triangleBuffer[tb_1].m_screenSpace_points[pO2Id]);
	float ti1 = (1.0f - t1);
	float ti2 = (1.0f - t2);

	{
		//Interpolate the m_relToCam_points_interpolationInfo
		Vector3 T1 = triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pIId] * t1 + triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pO1Id] * ti1;
		Vector3 T2 = triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pIId] * t2 + triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pO2Id] * ti2;

		triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[0] = triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pIId];
		triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[1] = T1;
		triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[2] = T2;
	}
	{
		//Interpolate the spaceScreen_vertecies
		Vector2 T1; T1.y = CLIPPING__UPPER_BOUNDARY; T1.x = triangleBuffer[tb_1].m_screenSpace_points[pIId].x * t1 + triangleBuffer[tb_1].m_screenSpace_points[pO1Id].x * ti1;
		Vector2 T2; T2.y = CLIPPING__UPPER_BOUNDARY; T2.x = triangleBuffer[tb_1].m_screenSpace_points[pIId].x * t2 + triangleBuffer[tb_1].m_screenSpace_points[pO2Id].x * ti2;

		triangleBuffer[tb_1].m_screenSpace_points[0] = triangleBuffer[tb_1].m_screenSpace_points[pIId];
		triangleBuffer[tb_1].m_screenSpace_points[1] = T1;
		triangleBuffer[tb_1].m_screenSpace_points[2] = T2;

		//Interpolate the uv_coordinates
		T1 = triangleBuffer[tb_1].m_uv_coordinates[pIId] * t1 + triangleBuffer[tb_1].m_uv_coordinates[pO1Id] * ti1;
		T2 = triangleBuffer[tb_1].m_uv_coordinates[pIId] * t2 + triangleBuffer[tb_1].m_uv_coordinates[pO2Id] * ti2;

		triangleBuffer[tb_1].m_uv_coordinates[0] = triangleBuffer[tb_1].m_uv_coordinates[pIId];
		triangleBuffer[tb_1].m_uv_coordinates[1] = T1;
		triangleBuffer[tb_1].m_uv_coordinates[2] = T2;
	}
}

//Right
__device__ void Clipping__buildClippedTriangle__rightSide__1pointOutside(GPU_Triangle* triangleBuffer, uint8 pOId, uint8 pI1Id, uint8 pI2Id, uint8 tb_1, uint8 tb_2) {
	//Compute the interpolation value
	float t1 = Clipping__edgeIntersection__rightSide(triangleBuffer[tb_1].m_screenSpace_points[pI1Id], triangleBuffer[tb_1].m_screenSpace_points[pOId]);
	float t2 = Clipping__edgeIntersection__rightSide(triangleBuffer[tb_1].m_screenSpace_points[pI2Id], triangleBuffer[tb_1].m_screenSpace_points[pOId]);
	triangleBuffer[tb_2] = triangleBuffer[tb_1]; //Copy the triangle before (validity flag already set)
	Clipping__attributesInterpolation__verticalAxisIntersection__onePointOut(&(triangleBuffer[tb_1]), &(triangleBuffer[tb_2]), pOId, pI1Id, pI2Id, t1, t2, CLIPPING__RIGHT_BOUNDARY);
}
__device__ void Clipping__buildClippedTriangle__rightSide__2pointOutside(GPU_Triangle* triangleBuffer, uint8 pIId, uint8 pO1Id, uint8 pO2Id, uint8 tb_1) {
	//Compute the interpolation value
	float t1 = Clipping__edgeIntersection__rightSide(triangleBuffer[tb_1].m_screenSpace_points[pIId], triangleBuffer[tb_1].m_screenSpace_points[pO1Id]);
	float t2 = Clipping__edgeIntersection__rightSide(triangleBuffer[tb_1].m_screenSpace_points[pIId], triangleBuffer[tb_1].m_screenSpace_points[pO2Id]);
	float ti1 = (1.0f - t1);
	float ti2 = (1.0f - t2);

	{
		//Interpolate the m_relToCam_points_interpolationInfo
		Vector3 T1 = triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pIId] * t1 + triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pO1Id] * ti1;
		Vector3 T2 = triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pIId] * t2 + triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pO2Id] * ti2;

		triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[0] = triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pIId];
		triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[1] = T1;
		triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[2] = T2;
	}
	{
		//Interpolate the spaceScreen_vertecies
		Vector2 T1; T1.x = CLIPPING__RIGHT_BOUNDARY; T1.y = triangleBuffer[tb_1].m_screenSpace_points[pIId].y * t1 + triangleBuffer[tb_1].m_screenSpace_points[pO1Id].y * ti1;
		Vector2 T2; T2.x = CLIPPING__RIGHT_BOUNDARY; T2.y = triangleBuffer[tb_1].m_screenSpace_points[pIId].y * t2 + triangleBuffer[tb_1].m_screenSpace_points[pO2Id].y * ti2;

		triangleBuffer[tb_1].m_screenSpace_points[0] = triangleBuffer[tb_1].m_screenSpace_points[pIId];
		triangleBuffer[tb_1].m_screenSpace_points[1] = T1;
		triangleBuffer[tb_1].m_screenSpace_points[2] = T2;

		//Interpolate the uv_coordinates
		T1 = triangleBuffer[tb_1].m_uv_coordinates[pIId] * t1 + triangleBuffer[tb_1].m_uv_coordinates[pO1Id] * ti1;
		T2 = triangleBuffer[tb_1].m_uv_coordinates[pIId] * t2 + triangleBuffer[tb_1].m_uv_coordinates[pO2Id] * ti2;

		triangleBuffer[tb_1].m_uv_coordinates[0] = triangleBuffer[tb_1].m_uv_coordinates[pIId];
		triangleBuffer[tb_1].m_uv_coordinates[1] = T1;
		triangleBuffer[tb_1].m_uv_coordinates[2] = T2;
	}
}
//Down
__device__ void Clipping__buildClippedTriangle__downSide__1pointOutside(GPU_Triangle* triangleBuffer, uint8 pOId, uint8 pI1Id, uint8 pI2Id, uint8 tb_1, uint8 tb_2) {
	//Compute the interpolation value
	float t1 = Clipping__edgeIntersection__downSide(triangleBuffer[tb_1].m_screenSpace_points[pI1Id], triangleBuffer[tb_1].m_screenSpace_points[pOId]);
	float t2 = Clipping__edgeIntersection__downSide(triangleBuffer[tb_1].m_screenSpace_points[pI2Id], triangleBuffer[tb_1].m_screenSpace_points[pOId]);
	triangleBuffer[tb_2] = triangleBuffer[tb_1]; //Copy the triangle before (validity flag already set)
	Clipping__attributesInterpolation__horizontalAxisIntersection__onePointOut(&(triangleBuffer[tb_1]), &(triangleBuffer[tb_2]), pOId, pI1Id, pI2Id, t1, t2, CLIPPING__DOWN_BOUNDARY);
}
__device__ void Clipping__buildClippedTriangle__downSide__2pointOutside(GPU_Triangle* triangleBuffer, uint8 pIId, uint8 pO1Id, uint8 pO2Id, uint8 tb_1) {
	//Compute the interpolation value
	float t1 = Clipping__edgeIntersection__downSide(triangleBuffer[tb_1].m_screenSpace_points[pIId], triangleBuffer[tb_1].m_screenSpace_points[pO1Id]);
	float t2 = Clipping__edgeIntersection__downSide(triangleBuffer[tb_1].m_screenSpace_points[pIId], triangleBuffer[tb_1].m_screenSpace_points[pO2Id]);
	float ti1 = (1.0f - t1);
	float ti2 = (1.0f - t2);

	{
		//Interpolate the m_relToCam_points_interpolationInfo
		Vector3 T1 = triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pIId] * t1 + triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pO1Id] * ti1;
		Vector3 T2 = triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pIId] * t2 + triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pO2Id] * ti2;

		triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[0] = triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[pIId];
		triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[1] = T1;
		triangleBuffer[tb_1].m_relToCam_points_interpolationInfo[2] = T2;
	}
	{
		//Interpolate the spaceScreen_vertecies
		Vector2 T1; T1.y = CLIPPING__DOWN_BOUNDARY; T1.x = triangleBuffer[tb_1].m_screenSpace_points[pIId].x * t1 + triangleBuffer[tb_1].m_screenSpace_points[pO1Id].x * ti1;
		Vector2 T2; T2.y = CLIPPING__DOWN_BOUNDARY; T2.x = triangleBuffer[tb_1].m_screenSpace_points[pIId].x * t2 + triangleBuffer[tb_1].m_screenSpace_points[pO2Id].x * ti2;

		triangleBuffer[tb_1].m_screenSpace_points[0] = triangleBuffer[tb_1].m_screenSpace_points[pIId];
		triangleBuffer[tb_1].m_screenSpace_points[1] = T1;
		triangleBuffer[tb_1].m_screenSpace_points[2] = T2;

		//Interpolate the uv_coordinates
		T1 = triangleBuffer[tb_1].m_uv_coordinates[pIId] * t1 + triangleBuffer[tb_1].m_uv_coordinates[pO1Id] * ti1;
		T2 = triangleBuffer[tb_1].m_uv_coordinates[pIId] * t2 + triangleBuffer[tb_1].m_uv_coordinates[pO2Id] * ti2;

		triangleBuffer[tb_1].m_uv_coordinates[0] = triangleBuffer[tb_1].m_uv_coordinates[pIId];
		triangleBuffer[tb_1].m_uv_coordinates[1] = T1;
		triangleBuffer[tb_1].m_uv_coordinates[2] = T2;
	}
}









































/*

__device__ void Clipping__buildClippedTriangle__leftSide__1pointOutside_nuovamodifica(GPU_Triangle* triangleBuffer, uint8 pOId, uint8 pI1Id, uint8 pI2Id) {
	//Compute the interpolation value
	float t1 = Clipping__edgeIntersection__leftSide(triangleBuffer->m_screenSpace_points[pI1Id], triangleBuffer->m_screenSpace_points[pOId]);
	float t2 = Clipping__edgeIntersection__leftSide(triangleBuffer->m_screenSpace_points[pI2Id], triangleBuffer->m_screenSpace_points[pOId]);
	float ti1 = (1.0f - t1);
	float ti2 = (1.0f - t2);
	triangleBuffer[1] = triangleBuffer[0]; //Copy the triangle before (validity flag already set)

	bool triangle1In, triangle2In;
	SET_BIT(triangleBuffer[1].flag, TF_BP__VALIDITY);
	{
		//Interpolate the spaceScreen_vertecies
		Vector2 T1; T1.x = CLIPPING__LEFT_BOUNDARY; T1.y = triangleBuffer[0].m_screenSpace_points[pI1Id].y * t1 + triangleBuffer[0].m_screenSpace_points[pOId].y * ti1;
		Vector2 T2; T2.x = CLIPPING__LEFT_BOUNDARY; T2.y = triangleBuffer[0].m_screenSpace_points[pI2Id].y * t2 + triangleBuffer[0].m_screenSpace_points[pOId].y * ti2;

		triangle1In = triangleBuffer[1].m_screenSpace_points[pI1Id] < CLIPPING__UPPER_BOUNDARY && T1.y < CLIPPING__UPPER_BOUNDARY&& triangleBuffer[1].m_screenSpace_points[pI2Id].y < CLIPPING__UPPER_BOUNDARY;
		triangle2In = T1.y < CLIPPING__UPPER_BOUNDARY&& T2.y < CLIPPING__UPPER_BOUNDARY&& triangleBuffer[1].m_screenSpace_points[pI2Id].y < CLIPPING__UPPER_BOUNDARY;
		if (triangle1In) {
			triangleBuffer[0].m_screenSpace_points[0] = triangleBuffer[1].m_screenSpace_points[pI1Id];
			triangleBuffer[0].m_screenSpace_points[1] = T1;
			triangleBuffer[0].m_screenSpace_points[2] = triangleBuffer[1].m_screenSpace_points[pI2Id];

			if (triangle2In) {
				triangleBuffer[1].m_screenSpace_points[0] = triangleBuffer[1].m_screenSpace_points[pI2Id];
				triangleBuffer[1].m_screenSpace_points[1] = T1;
				triangleBuffer[1].m_screenSpace_points[2] = T2;
			}
			else {
				CLEAR_BIT(triangleBuffer[1].flag, TF_BP__VALIDITY);
			}
		}
		else if (triangle2In) {
			triangleBuffer[0].m_screenSpace_points[0] = triangleBuffer[1].m_screenSpace_points[pI2Id];
			triangleBuffer[0].m_screenSpace_points[1] = T1;
			triangleBuffer[0].m_screenSpace_points[2] = T2;
			SET_BIT(triangleBuffer[1].flag, TF_BP__VALIDITY);
		}

		//Interpolate the uv_coordinates
		T1 = triangleBuffer[0].m_uv_coordinates[pI1Id] * t1 + triangleBuffer[0].m_uv_coordinates[pOId] * ti1;
		T2 = triangleBuffer[0].m_uv_coordinates[pI2Id] * t2 + triangleBuffer[0].m_uv_coordinates[pOId] * ti2;

		if (triangle1In) {
			triangleBuffer[0].m_uv_coordinates[0] = triangleBuffer[1].m_uv_coordinates[pI1Id];
			triangleBuffer[0].m_uv_coordinates[1] = T1;
			triangleBuffer[0].m_uv_coordinates[2] = triangleBuffer[1].m_uv_coordinates[pI2Id];

			if (triangle2In) {
				triangleBuffer[1].m_uv_coordinates[0] = triangleBuffer[1].m_uv_coordinates[pI2Id];
				triangleBuffer[1].m_uv_coordinates[1] = T1;
				triangleBuffer[1].m_uv_coordinates[2] = T2;
			}
		}
		else if (triangle2In) {
			triangleBuffer[0].m_uv_coordinates[0] = triangleBuffer[1].m_uv_coordinates[pI2Id];
			triangleBuffer[0].m_uv_coordinates[1] = T1;
			triangleBuffer[0].m_uv_coordinates[2] = T2;
		}
	}
	{
		//Interpolate the m_relToCam_points_interpolationInfo
		//I have to interpolate the 1/z or orthogonal view attributes to have a precise interpolation
		Vector3 T1 = triangleBuffer[0].m_relToCam_points_interpolationInfo[pI1Id] * t1 + triangleBuffer[0].m_relToCam_points_interpolationInfo[pOId] * ti1;
		Vector3 T2 = triangleBuffer[0].m_relToCam_points_interpolationInfo[pI2Id] * t2 + triangleBuffer[0].m_relToCam_points_interpolationInfo[pOId] * ti2;

		if (triangle1In) {
			triangleBuffer[0].m_relToCam_points_interpolationInfo[0] = triangleBuffer[1].m_relToCam_points_interpolationInfo[pI1Id];
			triangleBuffer[0].m_relToCam_points_interpolationInfo[1] = T1;
			triangleBuffer[0].m_relToCam_points_interpolationInfo[2] = triangleBuffer[1].m_relToCam_points_interpolationInfo[pI2Id];

			if (triangle2In) {
				triangleBuffer[1].m_relToCam_points_interpolationInfo[0] = triangleBuffer[1].m_relToCam_points_interpolationInfo[pI2Id];
				triangleBuffer[1].m_relToCam_points_interpolationInfo[1] = T1;
				triangleBuffer[1].m_relToCam_points_interpolationInfo[2] = T2;
			}
		}
		else if (triangle2In) {
			triangleBuffer[0].m_relToCam_points_interpolationInfo[0] = triangleBuffer[1].m_relToCam_points_interpolationInfo[pI2Id];
			triangleBuffer[0].m_relToCam_points_interpolationInfo[1] = T1;
			triangleBuffer[0].m_relToCam_points_interpolationInfo[2] = T2;
		}
	}
}*/