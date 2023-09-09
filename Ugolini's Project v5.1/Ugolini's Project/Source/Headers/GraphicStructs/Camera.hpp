#pragma once
#include <Windows.h>
#include "../Object/Object.hpp"

#define SCREEN_WIDTH 2500
#define SCREEN_HEIGHT (int)(SCREEN_WIDTH*9/16)
#define SCREEN_PIXELNUMBER SCREEN_WIDTH * SCREEN_HEIGHT

class Camera : public Object {

	//--------------------------------------------------------------- Members
public:
	uint16 m_screen_width, m_screen_height;
	float m_fov, m_aspect_ratio, m_inv_aspect_ratio, m_TanFov, m_invTanFov;
	Vector2 m_screenPortion[2];

	float m_friction = 0.5;


	//--------------------------------------------------------------- Constructors
public:
	Camera(Vector3 position, Quaternion rotation, uint16 width, uint16 height, float fov = PI * 1 / 3)
		: Object(position, Vector3(0, 0, 0), rotation) {
		m_screen_width = width; m_screen_height = height;
		m_fov = fov; m_TanFov = tan(fov);
		m_aspect_ratio = (float)width / height;
		m_inv_aspect_ratio = 1.0f / m_aspect_ratio;
		m_invTanFov = 1.0f / m_TanFov;
	}
	Camera(Vector3 position, uint16 width = SCREEN_WIDTH, uint16 height = SCREEN_HEIGHT, float fov = PI * 1 / 3)
		: Camera(position, Quaternion(), width, height, fov) {
	}
	Camera(uint16 width = SCREEN_WIDTH, uint16 height = SCREEN_HEIGHT, float fov = PI * 1 / 3)
		: Camera(Vector3(.2, 1.1, -5), Quaternion(Vector3(0,1,0), PI), width, height, fov) {
	}
	~Camera() = default;


	//--------------------------------------------------------------- Movement methods
	void GoTo(Vector3 vect) {
		m_position = vect; 
	}
	void Rotate(Quaternion quat) {
		m_orientation.Rotate(quat);
	}
	void AssRotation(Quaternion quat) {
		m_orientation = quat;
	}
};
Camera* p_camera = new Camera();


struct GPU_camInfo {
	uint16 m_screen_width, m_screen_height;
	float m_fov; //Fov (radiants)
	float m_aspect_ratio; //Width / height
	float m_inv_aspect_ratio; //height / width
	float m_TanFov, m_invTanFov; //tan(fov)

	Vector3 m_position;
	Quaternion m_orientation;

	GPU_camInfo(Camera* p_cam) {
		m_screen_width = p_cam->m_screen_width; m_screen_height = p_cam->m_screen_height;
		m_fov = p_cam->m_fov; m_TanFov = p_cam->m_TanFov; m_invTanFov = p_cam->m_invTanFov;
		m_aspect_ratio = (float)m_screen_width / m_screen_height;
		m_inv_aspect_ratio = 1 / m_aspect_ratio;
		m_position = p_cam->m_position; m_orientation = p_cam->m_orientation;
	}
	~GPU_camInfo() {

	}
	
	void operator= (GPU_camInfo const& ci) {
		m_screen_width = ci.m_screen_width; m_screen_height = ci.m_screen_height;
		m_fov = ci.m_fov; m_TanFov = ci.m_TanFov; m_invTanFov = ci.m_invTanFov;
		m_aspect_ratio = ci.m_aspect_ratio; 
		m_inv_aspect_ratio = ci.m_inv_aspect_ratio;
		m_position = ci.m_position; m_orientation = ci.m_orientation;
	}
	void Update(Camera* p_cam) {
		m_position = p_cam->m_position; m_orientation = p_cam->m_orientation;
	}
};