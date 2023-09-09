#pragma once

#include "Camera.hpp"
#include "ImagesStructs\RGB_structs.hpp"

#define DEFAULT_LIGHT_ID 0
#define DEFAULT_LIGHT_RGBVALUES 0.2,0.2,0.2
#define DEFAULT_LIGHT_DIRECTION -1.0f / 3, -1.0f / 3, 2.645f / 3

#define DEFAULT_POINTLIGHT_CONSTATT 0.05
#define DEFAULT_POINTLIGHT_LINATT 0.5
#define DEFAULT_POINTLIGHT_QUADATT 0.25

#define DEFAULT_SPOTLIGHT_CUTOFF_ANGLE (PI/12) //30°
#define DEFAULT_SPOTLIGHT_CUTOFF_VALUE cos(DEFAULT_SPOTLIGHT_CUTOFF_ANGLE/2)
#define DEFAULT_SPOTLIGHT_DEGRADATION_ANGLE (PI/18) //10°
#define DEFAULT_SPOTLIGHT_DEGRADATION_VALUE cos(DEFAULT_SPOTLIGHT_CUTOFF_ANGLE + (DEFAULT_SPOTLIGHT_DEGRADATION_ANGLE/2))

#define LIGHT_TYPE_NOTVALID 0
#define LIGHT_TYPE_DIRECTIONAL_LIGHT 1
#define LIGHT_TYPE_POINT_LIGHT 2
#define LIGHT_TYPE_SPOT_LIGHT 3

#define LIGHT_SHADERMAP_WIDTH 1280
#define LIGHT_SHADERMAP_HEIGHT 720

#define LIGHT_MAXDISTANCE_LIGHT 5000
struct Light {

	//--------------------------------------------------------------- Members
	//Space info (directional light wont use them)
	Vector3 m_absSpace_position = Vector3(0, 0, 1);

	//Temp variables: those variables will be used to store variables 
	//during the space trasformation process
	Vector3 m_rotatedDirection, m_relToCam_position;

	//Light specifics
	uint8 m_lightType = LIGHT_TYPE_NOTVALID; bool m_usingShaders = false;
	Vector3 m_direction = Vector3(0, 0, -1); RgbVector m_color = RgbVector::White;
	float m_maxDistanceSquared = 0, * p_shaderMap{ nullptr };

	//Attenuation variables
	float m_const_attenuation = 0, m_lin_attenuation = 0, m_quad_attenuation = 0;
	//Spot light variables
	//Spot light requires two angles, angle1: the area in this angle will not be attenuated; angle2: the area outside this angle will be completly dark, the area in between the two angles will be attenuated
	float m_cutOff = DEFAULT_SPOTLIGHT_CUTOFF_VALUE, m_cutOff_degradation = DEFAULT_SPOTLIGHT_DEGRADATION_VALUE; //m_cutOff_degradation will be the difference between the cos(angle1) and the cos(angle2) (cos(angle2) = cos(angle1) + degradation)
	float m_cutOff_invDegradation = 0; //This variable will store the inverse of the difference (m_cutOff - m_cutOff_degradation) in order not to compute it for every fragment


	//--------------------------------------------------------------- "Constructors"
	Light() = default;
	void operator= (Light const& l) {
		m_lightType = l.m_lightType; m_maxDistanceSquared = l.m_maxDistanceSquared;
		m_absSpace_position = l.m_absSpace_position;  m_relToCam_position = l.m_relToCam_position;
		m_direction = l.m_direction;  m_color = l.m_color;
		m_const_attenuation = l.m_const_attenuation;  m_lin_attenuation = l.m_lin_attenuation;  m_quad_attenuation = l.m_quad_attenuation;
		m_cutOff = l.m_cutOff;  m_cutOff_degradation = l.m_cutOff_degradation; m_cutOff_invDegradation = l.m_cutOff_invDegradation;
	}
	void DirectionalLight(Vector3 direction = Vector3(0, 0, 0), RgbVector color = RgbVector::White) {
		m_lightType = LIGHT_TYPE_DIRECTIONAL_LIGHT;
		m_direction = Vector3::Normalized(direction); m_color = color;
	}
	void PointLight(Vector3 position = Vector3(0, 0, 0), RgbVector color = RgbVector::White, float kconst = DEFAULT_POINTLIGHT_CONSTATT, float klin = DEFAULT_POINTLIGHT_LINATT, float kquad = DEFAULT_POINTLIGHT_QUADATT) {
		m_lightType = LIGHT_TYPE_POINT_LIGHT; m_maxDistanceSquared = LIGHT_MAXDISTANCE_LIGHT;
		m_absSpace_position = position; m_color = color;
		m_const_attenuation = kconst; m_lin_attenuation = klin; m_quad_attenuation = kquad;
	}
	void SpotLight(Vector3 position = Vector3(0, 0, 0), Vector3 direction = Vector3(0, 0, 0), RgbVector color = RgbVector::White, float cutOff = DEFAULT_SPOTLIGHT_CUTOFF_VALUE, float cutOff_degradation = DEFAULT_SPOTLIGHT_DEGRADATION_VALUE, float kconst = DEFAULT_POINTLIGHT_CONSTATT, float klin = DEFAULT_POINTLIGHT_LINATT, float kquad = DEFAULT_POINTLIGHT_QUADATT) {
		m_lightType = LIGHT_TYPE_SPOT_LIGHT; m_color = color; m_maxDistanceSquared = LIGHT_MAXDISTANCE_LIGHT;
		m_absSpace_position = position; m_direction = Vector3::Normalized(direction);
		m_const_attenuation = kconst; m_lin_attenuation = klin; m_quad_attenuation = kquad;
		m_cutOff = cutOff; m_cutOff_degradation = cutOff_degradation; m_cutOff_invDegradation = 1.0f / (m_cutOff - cutOff_degradation);
	}

	//--------------------------------------------------------------- Functions
	__device__ RgbVector ComputeColor(Vector3 relToCam_point, Vector3 normal, float specularExponent = 6) {
		if (m_lightType == LIGHT_TYPE_DIRECTIONAL_LIGHT) return ComputeColor__directionalLight(relToCam_point, normal, specularExponent);
		if (m_lightType == LIGHT_TYPE_POINT_LIGHT) return ComputeColor__pointLight(relToCam_point, normal, specularExponent);
		if (m_lightType == LIGHT_TYPE_SPOT_LIGHT) return ComputeColor__spotLight(relToCam_point, normal, specularExponent);
		return RgbVector(0,0,0);
	}
	__device__ RgbVector ComputeColor__directionalLight(Vector3 relToCam_point, Vector3 normal, float specularExponent) {
		return  m_color * myMax(0, (Vector3::DotProduct(m_rotatedDirection, normal)));
	}
	__device__ RgbVector ComputeColor__pointLight(Vector3 relToCam_point, Vector3 normal, float specularExponent) {
		Vector3 relToLight_point = relToCam_point - m_relToCam_position;
		float distance = relToLight_point.x * relToLight_point.x + relToLight_point.y * relToLight_point.y + relToLight_point.z * relToLight_point.z;
		if (distance > m_maxDistanceSquared) return RgbVector(0, 0, 0);

		//Compute attenuation
		float invAttenuationCoeff = m_const_attenuation; //Ill invert it later on
		invAttenuationCoeff += distance * m_quad_attenuation; //Distance still squared
		float invMod = invSqrt(distance); distance = 1.0f / invMod; relToLight_point = relToLight_point * invMod; //Square the lenght
		invAttenuationCoeff += distance * m_lin_attenuation;
		invAttenuationCoeff = 1.0f / myMax(1, invAttenuationCoeff);

		//Compute diffuse lighting
		float dp_dirNorm = Vector3::DotProduct(relToLight_point, normal); //dotProduct between direction and normal
		float diffuseCoeff = myMax(0, dp_dirNorm) * invAttenuationCoeff;

		//Compute specular lighting (Blinn - phong)
		//I have neither the usual fromPointToCam vector nor the fromPointToLight but the *-1 version of those: ill add, normalize them anyway and ill consider the -dotProd
		Vector3 half = (Vector3::Normalized(relToCam_point) + relToLight_point); half.Normalize();
		float dp_halfVNorm = Vector3::DotProduct(half, normal); dp_halfVNorm = myMax(0, dp_halfVNorm); //dotProduct between half vector and normal
		float specularCoeff = pow(dp_halfVNorm, specularExponent) * invAttenuationCoeff * (dp_dirNorm > 0);

		//return RgbVector(specularCoeff, specularCoeff, specularCoeff);
		return (m_color * diffuseCoeff + RgbVector(specularCoeff));
	}
	__device__ RgbVector ComputeColor__spotLight(Vector3 relToCam_point, Vector3 normal, float specularExponent) {
		Vector3 relToLight_point = relToCam_point - m_relToCam_position;
		
		//Compute the cutOff attenuation
		float dp_rtlPointDir = Vector3::DotProduct(Vector3::Normalized(relToLight_point), m_rotatedDirection);//This will store the dot product between the relToLight_point and the direction vector  
		float cutOff_attenuation = (dp_rtlPointDir - m_cutOff_degradation) * m_cutOff_invDegradation; cutOff_attenuation = myMin(cutOff_attenuation, 1);
		if (cutOff_attenuation < 0) return RgbVector(0);

		float distance = relToLight_point.x * relToLight_point.x + relToLight_point.y * relToLight_point.y + relToLight_point.z * relToLight_point.z;
		if (distance > m_maxDistanceSquared) return RgbVector(0, 0, 0);

		//Compute attenuation
		float invAttenuationCoeff = m_const_attenuation; //Ill invert it later on
		invAttenuationCoeff += distance * m_quad_attenuation; //Distance still squared
		float invMod = invSqrt(distance); distance = 1.0f / invMod; relToLight_point = relToLight_point * invMod; //Square the lenght
		invAttenuationCoeff += distance * m_lin_attenuation;
		invAttenuationCoeff = 1.0f / myMax(1, invAttenuationCoeff);

		//Compute diffuse lighting
		float dp_dirNorm = Vector3::DotProduct(relToLight_point, normal); //dotProduct between direction and normal
		float diffuseCoeff = myMax(0, dp_dirNorm) * invAttenuationCoeff * cutOff_attenuation;

		//Compute specular lighting (Blinn - phong)
		//I have neither the usual fromPointToCam vector nor the fromPointToLight but the *-1 version of those: ill add, normalize them anyway and ill consider the -dotProd
		Vector3 half = (Vector3::Normalized(relToCam_point) + relToLight_point); half.Normalize();
		float dp_halfVNorm = Vector3::DotProduct(half, normal); dp_halfVNorm = myMax(0, dp_halfVNorm); //dotProduct between half vector and normal
		float specularCoeff = pow(dp_halfVNorm, specularExponent) * invAttenuationCoeff * cutOff_attenuation * (dp_dirNorm > 0);

		//return RgbVector(specularCoeff, specularCoeff, specularCoeff);
		return (m_color * diffuseCoeff + RgbVector(specularCoeff));
	}
};