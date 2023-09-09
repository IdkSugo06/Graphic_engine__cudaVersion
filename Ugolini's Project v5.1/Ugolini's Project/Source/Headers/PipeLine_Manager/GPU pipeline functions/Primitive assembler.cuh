#pragma once
#include "Pipeline functions general defines.cuh"


//For every triangleId, it computes the attributes needed and set its validity bit
__device__ bool PrimitiveAssembler__subComputetions(const GPU_camInfo* p_ci, const TriangleId* tId, const Vector3* asVertices, const Vector3* rtcVertices, GPU_Triangle* tri, uint32 id) {

	//compute relToCam normal, tangent and bitangent 
	float tangentX = rtcVertices[tId[id].m_vertexIds[2]].x - rtcVertices[tId[id].m_vertexIds[0]].x; //Used as temp variables
	float tangentY = rtcVertices[tId[id].m_vertexIds[2]].y - rtcVertices[tId[id].m_vertexIds[0]].y;
	float tangentZ = rtcVertices[tId[id].m_vertexIds[2]].z - rtcVertices[tId[id].m_vertexIds[0]].z;
	float mod = invSqrt(tangentX * tangentX + tangentY * tangentY + tangentZ * tangentZ); if (mod < 0.0001f) return false;
	tangentX *= mod; tangentY *= mod; tangentZ *= mod;

	float bitangentX = rtcVertices[tId[id].m_vertexIds[1]].x - rtcVertices[tId[id].m_vertexIds[0]].x; //That is a temp bitangent (it is a side of the triangle)
	float bitangentY = rtcVertices[tId[id].m_vertexIds[1]].y - rtcVertices[tId[id].m_vertexIds[0]].y; //That is a temp bitangent
	float bitangentZ = rtcVertices[tId[id].m_vertexIds[1]].z - rtcVertices[tId[id].m_vertexIds[0]].z; //That is a temp bitangent

	float normalX = tangentY * bitangentZ - tangentZ * bitangentY;
	float normalY = tangentZ * bitangentX - tangentX * bitangentZ;
	float normalZ = tangentX * bitangentY - tangentY * bitangentX;
	mod = invSqrt(normalX * normalX + normalY * normalY + normalZ * normalZ); if (mod < 0.0001f) return false;
	normalX *= mod; normalY *= mod; normalZ *= mod;

	bitangentX = normalY * tangentZ - normalZ * tangentY;
	bitangentY = normalZ * tangentX - normalX * tangentZ;
	bitangentZ = normalX * tangentY - normalY * tangentX;
	

	//Assign normal, tangent and bitangent attributes
	tri->m_relToCam_tangent.x = tangentX;
	tri->m_relToCam_tangent.y = tangentY;
	tri->m_relToCam_tangent.z = tangentZ;

	tri->m_relToCam_normal.x = normalX;
	tri->m_relToCam_normal.y = normalY;
	tri->m_relToCam_normal.z = normalZ;

	if (Vector3::DotProduct(tri->m_relToCam_points[0], tri->m_relToCam_normal) < 0) {
		CLEAR_BIT(tri->flag, TF_BP__VALIDITY); return false;
	}
	
	tri->m_relToCam_bitangent.x = bitangentX;
	tri->m_relToCam_bitangent.y = bitangentY;
	tri->m_relToCam_bitangent.z = bitangentZ;

	return true;
}
__global__ void PrimitiveAssembler(const GPU_camInfo* p_ci, const TriangleId* tId, GPU_Triangle* triangles, const Vector3* asVertices, Vector3* rtcVertices, const Vector3* uvCoordinates, const uint32 triangleArrayLenght, Fragment* p_fragmentMap) {
	uint32 id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= triangleArrayLenght)
		return;

	if (rtcVertices[tId[id].m_vertexIds[0]].z < 0 || rtcVertices[tId[id].m_vertexIds[1]].z < 0 || rtcVertices[tId[id].m_vertexIds[2]].z < 0) return;
	GPU_Triangle* tri = &(triangles[id]);
	tri->m_materialId = tId[id].m_materialId;
	
	//ASSIGN ABS SPACE VERTEX
	tri->m_absSpace_points[0] = asVertices[tId[id].m_vertexIds[0]];
	tri->m_absSpace_points[1] = asVertices[tId[id].m_vertexIds[1]];
	tri->m_absSpace_points[2] = asVertices[tId[id].m_vertexIds[2]];

	//ASSIGN ABS SPACE VERTEX
	tri->m_relToCam_points[0] = rtcVertices[tId[id].m_vertexIds[0]];
	tri->m_relToCam_points[1] = rtcVertices[tId[id].m_vertexIds[1]];
	tri->m_relToCam_points[2] = rtcVertices[tId[id].m_vertexIds[2]];

	//SPLIT THE COMPUTATION IN ORDER TO LIGHTHEN THE FUNCTION
	if (!PrimitiveAssembler__subComputetions(p_ci, tId, asVertices, rtcVertices, tri, id)) return;
	
	//ASSIGN RELTOCAM VERTEX
	if (tri->m_relToCam_points[0].z == 0) tri->m_relToCam_points[0].z += 0.00001f;
	if (tri->m_relToCam_points[1].z == 0) tri->m_relToCam_points[1].z += 0.00001f;
	if (tri->m_relToCam_points[2].z == 0) tri->m_relToCam_points[2].z += 0.00001f;
	float invZ0 = 1.0f / (tri->m_relToCam_points[0].z);
	float invZ1 = 1.0f / (tri->m_relToCam_points[1].z);
	float invZ2 = 1.0f / (tri->m_relToCam_points[2].z);
	
	tri->m_relToCam_points_interpolationInfo[0].z = invZ0;
	tri->m_relToCam_points_interpolationInfo[1].z = invZ1;
	tri->m_relToCam_points_interpolationInfo[2].z = invZ2;
	
	tri->m_relToCam_points_interpolationInfo[0].x = rtcVertices[tId[id].m_vertexIds[0]].x * invZ0;
	tri->m_relToCam_points_interpolationInfo[1].x = rtcVertices[tId[id].m_vertexIds[1]].x * invZ1;
	tri->m_relToCam_points_interpolationInfo[2].x = rtcVertices[tId[id].m_vertexIds[2]].x * invZ2;
	
	tri->m_relToCam_points_interpolationInfo[0].y = rtcVertices[tId[id].m_vertexIds[0]].y * invZ0;
	tri->m_relToCam_points_interpolationInfo[1].y = rtcVertices[tId[id].m_vertexIds[1]].y * invZ1;
	tri->m_relToCam_points_interpolationInfo[2].y = rtcVertices[tId[id].m_vertexIds[2]].y * invZ2;
	
	//ASSIGN UV COORDINATES	
	tri->m_uv_coordinates[0].x = uvCoordinates[tId[id].m_uvCoordinatesIds[0]].x * invZ0;
	tri->m_uv_coordinates[1].x = uvCoordinates[tId[id].m_uvCoordinatesIds[1]].x * invZ1;
	tri->m_uv_coordinates[2].x = uvCoordinates[tId[id].m_uvCoordinatesIds[2]].x * invZ2;
	
	tri->m_uv_coordinates[0].y = uvCoordinates[tId[id].m_uvCoordinatesIds[0]].y * invZ0;
	tri->m_uv_coordinates[1].y = uvCoordinates[tId[id].m_uvCoordinatesIds[1]].y * invZ1;
	tri->m_uv_coordinates[2].y = uvCoordinates[tId[id].m_uvCoordinatesIds[2]].y * invZ2;	
	
	SET_BIT(tri->flag, TF_BP__VALIDITY);
}