#pragma once
#include "Pipeline functions general defines.cuh"


__global__ void VertexShader(const GPU_camInfo* p_camInfo, const Vector3* p_absSpace_points, Vector3* p_relToCam_points, const uint32 vertexArrayLenght);
__global__ void LightRotation(const GPU_camInfo* p_camInfo, const Light* p_lights, const uint32 lightsNumber);
__global__ void FragmentShader(Fragment* p_fragmentMap, RgbVector* p_rgbHDRMap, const Material* p_materials, Light* p_lights, uint16 numOf_lights, uint64 lenght);
__device__ RgbVector ComputeColor(Fragment* frag, const Material* material, Light* p_lights, uint16 numOf_lights);


//-- -- -- -- -- -- -- -- -- -- -- -- CODE -- -- -- -- -- -- -- -- -- -- -- -- 

__global__ void VertexShader(const GPU_camInfo* p_camInfo, const Vector3* p_absSpace_points, Vector3* p_relToCam_points, const uint32 vertexArrayLenght) {
	uint32 id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= vertexArrayLenght)
		return;
	//Quaternion based transformation: absSpace position to relToCam space
	Vector3 rtcPos;
	rtcPos.x = p_absSpace_points[id].x - p_camInfo->m_position.x;
	rtcPos.y = p_absSpace_points[id].y - p_camInfo->m_position.y;
	rtcPos.z = p_absSpace_points[id].z - p_camInfo->m_position.z;

	float q1w = p_camInfo->m_orientation.w +
		p_camInfo->m_orientation.x * rtcPos.x +
		p_camInfo->m_orientation.y * rtcPos.y +
		p_camInfo->m_orientation.z * rtcPos.z;

	float q1x = p_camInfo->m_orientation.w * rtcPos.x -
		p_camInfo->m_orientation.x -
		p_camInfo->m_orientation.y * rtcPos.z +
		p_camInfo->m_orientation.z * rtcPos.y;

	float q1y = p_camInfo->m_orientation.w * rtcPos.y -
		p_camInfo->m_orientation.y -
		p_camInfo->m_orientation.z * rtcPos.x +
		p_camInfo->m_orientation.x * rtcPos.z;

	float q1z = p_camInfo->m_orientation.w * rtcPos.z -
		p_camInfo->m_orientation.z +
		p_camInfo->m_orientation.y * rtcPos.x -
		p_camInfo->m_orientation.x * rtcPos.y;

	p_relToCam_points[id].x = q1w * p_camInfo->m_orientation.x + q1x * p_camInfo->m_orientation.w + q1y * p_camInfo->m_orientation.z - q1z * p_camInfo->m_orientation.y;
	p_relToCam_points[id].y = q1w * p_camInfo->m_orientation.y + q1y * p_camInfo->m_orientation.w + q1z * p_camInfo->m_orientation.x - q1x * p_camInfo->m_orientation.z;
	p_relToCam_points[id].z = q1w * p_camInfo->m_orientation.z + q1z * p_camInfo->m_orientation.w - q1y * p_camInfo->m_orientation.x + q1x * p_camInfo->m_orientation.y;
	return;
}
__global__ void LightRotation(const GPU_camInfo* p_camInfo, Light* p_lights, const uint32 lightsNumber) {
	uint32 id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= lightsNumber) return;

	Light* light = &(p_lights[id]);
	float q1w = 0, q1x = 0, q1y = 0, q1z = 0;
	if (light->m_lightType == LIGHT_TYPE_SPOT_LIGHT || light->m_lightType == LIGHT_TYPE_POINT_LIGHT) {
		//Quaternion based transformation: light position absSpace to relToCam space
		Vector3 rtcPos;
		rtcPos.x = light->m_absSpace_position.x - p_camInfo->m_position.x;
		rtcPos.y = light->m_absSpace_position.y - p_camInfo->m_position.y;
		rtcPos.z = light->m_absSpace_position.z - p_camInfo->m_position.z;

		q1w = p_camInfo->m_orientation.w +
			p_camInfo->m_orientation.x * rtcPos.x +
			p_camInfo->m_orientation.y * rtcPos.y +
			p_camInfo->m_orientation.z * rtcPos.z;

		q1x = p_camInfo->m_orientation.w * rtcPos.x -
			p_camInfo->m_orientation.x -
			p_camInfo->m_orientation.y * rtcPos.z +
			p_camInfo->m_orientation.z * rtcPos.y;

		q1y = p_camInfo->m_orientation.w * rtcPos.y -
			p_camInfo->m_orientation.y -
			p_camInfo->m_orientation.z * rtcPos.x +
			p_camInfo->m_orientation.x * rtcPos.z;

		q1z = p_camInfo->m_orientation.w * rtcPos.z -
			p_camInfo->m_orientation.z +
			p_camInfo->m_orientation.y * rtcPos.x -
			p_camInfo->m_orientation.x * rtcPos.y;

		light->m_relToCam_position.x = q1w * p_camInfo->m_orientation.x + q1x * p_camInfo->m_orientation.w + q1y * p_camInfo->m_orientation.z - q1z * p_camInfo->m_orientation.y;
		light->m_relToCam_position.y = q1w * p_camInfo->m_orientation.y + q1y * p_camInfo->m_orientation.w + q1z * p_camInfo->m_orientation.x - q1x * p_camInfo->m_orientation.z;
		light->m_relToCam_position.z = q1w * p_camInfo->m_orientation.z + q1z * p_camInfo->m_orientation.w - q1y * p_camInfo->m_orientation.x + q1x * p_camInfo->m_orientation.y;
	}
	if (light->m_lightType == LIGHT_TYPE_DIRECTIONAL_LIGHT || light->m_lightType == LIGHT_TYPE_SPOT_LIGHT) {
		//Quaternion based transformation: light direction rotation
		Vector3 lightDir = light->m_direction;
		q1w = p_camInfo->m_orientation.w +
			p_camInfo->m_orientation.x * lightDir.x +
			p_camInfo->m_orientation.y * lightDir.y +
			p_camInfo->m_orientation.z * lightDir.z;

		q1x = p_camInfo->m_orientation.w * lightDir.x -
			p_camInfo->m_orientation.x -
			p_camInfo->m_orientation.y * lightDir.z +
			p_camInfo->m_orientation.z * lightDir.y;

		q1y = p_camInfo->m_orientation.w * lightDir.y -
			p_camInfo->m_orientation.y -
			p_camInfo->m_orientation.z * lightDir.x +
			p_camInfo->m_orientation.x * lightDir.z;

		q1z = p_camInfo->m_orientation.w * lightDir.z -
			p_camInfo->m_orientation.z +
			p_camInfo->m_orientation.y * lightDir.x -
			p_camInfo->m_orientation.x * lightDir.y;

		light->m_rotatedDirection.x = q1w * p_camInfo->m_orientation.x + q1x * p_camInfo->m_orientation.w + q1y * p_camInfo->m_orientation.z - q1z * p_camInfo->m_orientation.y;
		light->m_rotatedDirection.y = q1w * p_camInfo->m_orientation.y + q1y * p_camInfo->m_orientation.w + q1z * p_camInfo->m_orientation.x - q1x * p_camInfo->m_orientation.z;
		light->m_rotatedDirection.z = q1w * p_camInfo->m_orientation.z + q1z * p_camInfo->m_orientation.w - q1y * p_camInfo->m_orientation.x + q1x * p_camInfo->m_orientation.y;
	}
	return;
}
__global__ void FragmentShader(Fragment* p_fragmentMap, RgbVector* p_rgbHDRMap, const Material* p_materials, Light* p_lights, uint16 numOf_lights, uint64 lenght) {
	uint32 threadId = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadId >= lenght) return;

	for (uint32 pixelId = threadId * PIXEL_PER_KERNEL; pixelId < (threadId + 1) * PIXEL_PER_KERNEL; pixelId++) {
		if (p_fragmentMap[pixelId].m_valid) {
			Fragment* frag = &(p_fragmentMap[pixelId]);

			float z = 1.0f / frag->m_relToCam_point.z;
			frag->m_relToCam_point.x *= z; frag->m_relToCam_point.y *= z; frag->m_relToCam_point.z = z;
			frag->m_uv_coordinates.x *= z; frag->m_uv_coordinates.y *= z;

			RgbVector finalColor = ComputeColor(frag, &(p_materials[frag->m_materialId]), p_lights, numOf_lights);
			p_rgbHDRMap[pixelId] = finalColor;
		}
	}
}
__device__ RgbVector ComputeColor(Fragment* frag, const Material* material, Light* p_lights, uint16 numOf_lights) {
	const RgbVectorMap* txtMap = &(material->m_textureMap); const NormalMap* normalMap = &(material->m_normalMap);
	RgbVector baseColor;

	if (normalMap->m_isValid) {
		if (normalMap->p_memory != nullptr) {
			uint32 normalId = (uint32)(frag->m_uv_coordinates.x * normalMap->m_width) + (uint32)(frag->m_uv_coordinates.y * normalMap->m_height) * normalMap->m_width; //From (0->1) to pixels (0->width)
			if (normalId < normalMap->m_mapLenght) {
				//Matrix transformation
				Matrix_3x3 m(frag->m_relToCam_tangent, frag->m_relToCam_bitangent, frag->m_relToCam_normal);
				frag->m_relToCam_normal = m * material->m_normalMap.p_memory[normalId]; frag->m_relToCam_normal *= -1; //frag->m_relToCam_normal.z *= -1;
			}
		}
	}
	if (txtMap->m_isValid) {
		if (txtMap->p_memory != nullptr) {
			RgbVector* txtMem = txtMap->p_memory;
			uint32 txtId = (uint32)(frag->m_uv_coordinates.x * txtMap->m_width) + (uint32)(frag->m_uv_coordinates.y * txtMap->m_height) * (uint32)txtMap->m_width;
			if (txtId < txtMap->m_mapLenght) { baseColor.red = txtMem[txtId].red; baseColor.green = txtMem[txtId].green; baseColor.blue = txtMem[txtId].blue; }
		}
	}
	else { 
		baseColor = material->m_diffuseCoeff; 
	}

	float specExp = material->m_specularExponent; Vector3 normal = frag->m_relToCam_normal; Vector3 relToCam_point = frag->m_relToCam_point;
	RgbVector finalColor = baseColor * p_lights[DEFAULT_LIGHT_ID].ComputeColor__directionalLight(relToCam_point, normal, specExp);
	for (uint16 i = 0; i < numOf_lights; i++) {
		if (i != DEFAULT_LIGHT_ID) {
			finalColor += baseColor * p_lights[i].ComputeColor(relToCam_point, normal, specExp);
		}
	}
	finalColor += material->m_emissiveCoeff;
	finalColor += MATERIAL_DEF_AMBIENT_COEFF;
	return finalColor;
}