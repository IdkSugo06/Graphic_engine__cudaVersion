#pragma once
#include "..\Math\MyMath.hpp"


struct Object {

    //--------------------------------------------------------------- Members
public:
    Vector3 m_position = Vector3(0,0,0), m_velocity_ms = Vector3(0, 0, 0);

    //m_orientation is the rotation to apply at an object in absolute space in order to have the same orientation
    Quaternion m_orientation, m_rotation_ms;


    //--------------------------------------------------------------- Constructors
    Object(Vector3 position, Vector3 velocity, Quaternion rotation) {
        m_position = position; m_velocity_ms = velocity; m_orientation = rotation;
    }
    Object(Vector3 position, Vector3 velocity)
        : Object(position, velocity, Quaternion()) {
        m_position = position; m_velocity_ms = velocity;
    }
    Object(Vector3 position)
        : Object(position, Vector3::VectAss(0, 0, 0), Quaternion()) {
    }
    Object()
        : Object(Vector3::VectAss(0, 0, 0), Vector3::VectAss(0, 0, 0), Quaternion()) {
    }

    ~Object() = default;


    //--------------------------------------------------------------- Physic methods
public:
    //virtual void UpdateObj(uint16 time_ms) {
    //    MoveObj(m_velocity_ms * time_ms);
    //    RotateObj(m_rotation_ms * time_ms);
    //}
    //virtual void MoveObj(Vector3 dist) {
    //    m_position += dist;
    //}
    //virtual void RotateObj(Quaternion rotation, Vector3 origin = Vector3(0, 0, 0)) {
    //    m_orientation.Rotate(rotation);
    //}
};