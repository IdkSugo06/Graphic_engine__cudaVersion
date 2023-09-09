#pragma once
#include "../Math/MyMath.hpp"

struct Fragment {
	bool m_valid = false;

	void* p_temp;
	Vector3 m_absSpace_point, m_relToCam_point;
	Vector3 m_relToCam_normal, m_relToCam_tangent, m_relToCam_bitangent;

	Vector3 m_uv_coordinates; //Uv coordinates will be between 0 and 1 (they'll be converted in screen space during the fragment shader) (careful bt prospective correction (primitive assembler -> fragment shader))
	Vector2 m_screenSpace_point;

	uint16 m_materialId;

	//float m_zComponent = 0; Contained in the relToCam_point
};