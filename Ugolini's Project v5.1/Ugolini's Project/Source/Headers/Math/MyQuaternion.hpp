#pragma once
#include "MyVector3.hpp"


struct Quaternion {

    //--------------------------------------------------------------- Members
public:
    float x, y, z, w = 1;


    //--------------------------------------------------------------- Constructors
public:
    Quaternion() {
        x = 0; y = 0; z = 0; w = 1;
    }
    Quaternion(float argX, float argY, float argZ, float argW) {
        x = argX; y = argY; z = argZ; w = argW;
    }
    Quaternion(float realPart, Vector3 immPart) {
        x = immPart.x; y = immPart.y; z = immPart.z; w = realPart;
    }
    Quaternion(Vector3 axis, double angle) {
        axis.Normalize();
        axis *= sin(angle / 2);
        *this = axis;                //Assegno la parte immaginaria
        w = cos(angle / 2);         //Assegno la parte reale
        Normalize();
    }


    //--------------------------------------------------------------- Operator overload
        //Quaternions
    Quaternion operator+ (Quaternion const& q2) {
        return Quaternion(x + q2.x, y + q2.y, z + q2.z, w + q2.w);
    }
    Quaternion operator- (Quaternion const& q2) {
        return Quaternion(x - q2.x, y - q2.y, z - q2.z, w - q2.w);
    }
    Quaternion operator* (Quaternion const& q2) {
        Quaternion result;
        result.w = w * q2.w - x * q2.x - y * q2.y - z * q2.z;
        result.x = w * q2.x + x * q2.w + y * q2.z - z * q2.y;
        result.y = w * q2.y + y * q2.w + z * q2.x - x * q2.z;
        result.z = w * q2.z + z * q2.w + x * q2.y - y * q2.x;
        return result;
    }

    Quaternion operator+ (float const& f) {
        return Quaternion(x + f, y + f, z + f, w + f);
    }
    Quaternion operator- (float const& f) {
        return Quaternion(x - f, y - f, z - f, w - f);
    }
    Quaternion operator* (float const& f) {
        return Quaternion(x * f, y * f, z * f, w * f);
    }

    //Bool 
    bool operator== (Quaternion const& q2) {
        return (x == q2.x && y == q2.y && z == q2.z && w == q2.w);
    }
    bool operator> (Quaternion const& q2) {
        return (x > q2.x && y > q2.y && z > q2.z && w > q2.w);
    }
    bool operator< (Quaternion const& q2) {
        return (x < q2.x&& y < q2.y&& z < q2.z&& w < q2.w);
    }
    bool operator>= (Quaternion const& q2) {
        return (x >= q2.x && y >= q2.y && z >= q2.z && w >= q2.w);
    }
    bool operator<= (Quaternion const& q2) {
        return (x <= q2.x && y <= q2.y && z <= q2.z && w <= q2.w);
    }

    bool operator== (float const& f) {
        return (x == f && y == f && z == f);
    }
    bool operator> (float const& f) {
        return (x > f && y > f && z > f);
    }
    bool operator< (float const& f) {
        return (x < f&& y < f&& z < f);
    }
    bool operator>= (float const& f) {
        return (x >= f && y >= f && z >= f);
    }
    bool operator<= (float const& f) {
        return (x <= f && y <= f && z <= f);
    }

    //Void
    void operator+= (Quaternion const& q2) {
        x += q2.x; y += q2.y; z += q2.z; w += q2.w;
    }
    void operator-= (Quaternion const& q2) {
        x -= q2.x; y -= q2.y; z -= q2.z; w -= q2.w;
    }
    void operator*= (Quaternion const& v2) {
        (*this) = (*this) * v2;
    }
    void operator= (Quaternion const& q2) {
        x = q2.x; y = q2.y; z = q2.z; w = q2.w;
    }
    void operator= (Vector3 const& v2) {
        x = v2.x; y = v2.y; z = v2.z;
    }

    void operator+= (float const& f) {
        x += f; y += f; z += f; w += f;
    }
    void operator-= (float const& f) {
        x -= f; y -= f; z -= f; w -= f;
    }
    void operator*= (float const& f) {
        x *= f; y *= f; z *= f; w *= f;
    }
    void operator= (float const& f) {
        x = f; y = f; z = f; w = f;
    }


    //--------------------------------------------------------------- Methods
    void Ass(float _x, float _y, float _z, float _w) {
        x = _x; y = _y; z = _z; w = _w;
    }
    void Ass(Vector3 vect, float _w) {
        x = vect.x; y = vect.y; z = vect.z; w = _w;
    }

    float Length() {
        return sqrt(x * x + y * y + z * z + w * w);
    }
    void Coniugate() {
        x = -x; y = -y; z = -z;
    }
    void Normalize() {
        (*this) = (*this) * (1 / (*this).Length());
    }
    Quaternion Coniugated() {
        Quaternion quat(-x, -y, -z, w);
        return quat;
    }
    Quaternion Normalized() {
        Quaternion quat = *this;
        return quat * (1 / quat.Length());
    }


    //--------------------------------------------------------------- Static methods
    static Quaternion Normalized(Quaternion quat) {
        quat.Normalize();
        return quat;
    }
    static Quaternion Coniugated(Quaternion quat) {
        quat.Coniugate();
        return quat;
    }
    static float LengthOf(Quaternion quat) {
        return quat.Length();
    }
    static Quaternion QuatAss(float argX, float argY, float argZ, float argW) {
        return Quaternion(argX, argY, argZ, argZ);
    }
    static Vector3 QuatToVect(Quaternion quat) {
        return Vector3(quat.x, quat.y, quat.z);
    }
    static void PrintQ(Quaternion q, bool nl = true) {
        std::cout << "(" << q.x << "," << q.y << "," << q.z << "," << q.w << ")";
        if (nl) { std::cout << std::endl; }
    }


    //--------------------------------------------------------------- Rotation management
    void Rotate(Quaternion rotation) {
        *this *= rotation;
    }
    Vector3 RotatePoint(Vector3 point, Vector3 origin = Vector3(0, 0, 0)) {
        point -= origin;
        Quaternion translated_point = ((*this) * Quaternion(1, point)) * Coniugated(*this);
        return QuatToVect(translated_point) + origin;
    }
    static Vector3 RotatePoint(Quaternion rotationQuat, Vector3 point) {
        Quaternion translated_point = (rotationQuat * Quaternion(1, point)) * Coniugated(rotationQuat);
        return QuatToVect(translated_point);
    }
    static Vector3 RotatePoint(Quaternion rotationQuat, Vector3 point, Vector3 origin) {
        point = point - origin;
        Quaternion translated_point = (rotationQuat * Quaternion(1, point)) * Coniugated(rotationQuat);
        return QuatToVect(translated_point) + origin;
    }
    static void RotatePoints(Quaternion rotationQuat, Vector3* in_points, Vector3* out_points, uint32_t points_number, Vector3 origin = Vector3(0, 0, 0)) {
        //Doesnt affect *in_points
        for (uint32_t i = 0; i < points_number; i++) {
            out_points[i] = rotationQuat.RotatePoint(in_points[i], origin);
        }
    }
    static void RotatePoints(Quaternion rotationQuat, Vector3* points, uint32_t points_number, Vector3 origin = Vector3(0, 0, 0)) {
        //Affects *points
        for (uint32_t i = 0; i < points_number; i++) {
            points[i] = rotationQuat.RotatePoint(points[i], origin);
        }
    }
};