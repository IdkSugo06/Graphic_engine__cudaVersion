#pragma once
#include <iostream>
#include <cmath>
#include "MyVector2.hpp"


struct Vector3 {

    //--------------------------------------------------------------- Members
public:
    float x = 0, y = 0, z = 0;
    static const Vector3 m_nullVector;


    //--------------------------------------------------------------- Constructors
    __device__ __host__ Vector3() {
        x = 0, y = 0, z = 0;
    }
    __device__ __host__ Vector3(float num) {
        x = num, y = num, z = num;
    }
    __device__ __host__ Vector3(float argX, float argY, float argZ) {
        x = argX, y = argY, z = argZ;
    }
    __device__ __host__ Vector3(Vector2 vect, float argZ) {
        x = vect.x; y = vect.y; z = argZ;
    }
    ~Vector3() = default;


    //--------------------------------------------------------------- Operator overload
        //Vector3
    __device__ __host__ Vector3 operator+ (Vector3 const& v2) const {
        return Vector3(x + v2.x, y + v2.y, z + v2.z);
    }
    __device__ __host__ Vector3 operator- (Vector3 const& v2) const {
        return Vector3(x - v2.x, y - v2.y, z - v2.z);
    }
    __device__ __host__ Vector3 operator* (Vector3 const& v2) const {
        //Return the cross product
        Vector3 result;
        result.x = y * v2.z - z * v2.y;
        result.y = z * v2.x - x * v2.z;
        result.z = x * v2.y - y * v2.x;
        return result;
    }

    __device__ __host__ Vector3 operator+ (float const& f) const {
        return Vector3(x + f, y + f, z + f);
    }
    __device__ __host__ Vector3 operator- (float const& f) const {
        return Vector3(x - f, y - f, z - f);
    }
    __device__ __host__ Vector3 operator* (float const& f) const {
        return Vector3(x * f, y * f, z * f);
    }

    //Bool 
    __device__ __host__ bool operator== (Vector3 const& v2) const {
        return (x == v2.x && y == v2.y && z == v2.z);
    }
    __device__ __host__ bool operator> (Vector3 const& v2) const {
        return (x > v2.x && y > v2.y && z > v2.z);
    }
    __device__ __host__ bool operator< (Vector3 const& v2) const {
        return (x < v2.x&& y < v2.y&& z < v2.z);
    }
    __device__ __host__ bool operator>= (Vector3 const& v2) const {
        return (x >= v2.x && y >= v2.y && z >= v2.z);
    }
    __device__ __host__ bool operator<= (Vector3 const& v2) const {
        return (x <= v2.x && y <= v2.y && z <= v2.z);
    }

    __device__ __host__ bool operator== (float const& f) const {
        return (x == f && y == f && z == f);
    }
    __device__ __host__ bool operator> (float const& f) const {
        return (x > f && y > f && z > f);
    }
    __device__ __host__ bool operator< (float const& f) const {
        return (x < f&& y < f&& z < f);
    }
    __device__ __host__ bool operator>= (float const& f) const {
        return (x >= f && y >= f && z >= f);
    }
    __device__ __host__ bool operator<= (float const& f) const {
        return (x <= f && y <= f && z <= f);
    }

    //Void
    __device__ __host__ void operator+= (Vector3 const& v2) {
        x += v2.x; y += v2.y; z += v2.z;
    }
    __device__ __host__ void operator-= (Vector3 const& v2) {
        x -= v2.x; y -= v2.y; z -= v2.z;
    }
    __device__ __host__ void operator*= (Vector3 const& v2) {
        (*this) = (*this) * v2;
    }
    __device__ __host__ void operator= (Vector3 const& v2) {
        x = v2.x; y = v2.y; z = v2.z;
    }
    __device__ __host__ void operator= (Vector2 const& v2) {
        x = v2.x; y = v2.y; z = 0;
    }

    __device__ __host__ void operator+= (float const& f) {
        x += f; y += f; z += f;
    }
    __device__ __host__ void operator-= (float const& f) {
        x -= f; y -= f; z -= f;
    }
    __device__ __host__ void operator*= (float const& f) {
        x *= f; y *= f; z *= f;
    }
    __device__ __host__ void operator= (float const& f) {
        x = f; y = f; z = f;
    }


    //--------------------------------------------------------------- Methods
    __device__ __host__ float Length() const{
        return sqrtf(x * x + y * y + z * z);
    }
    __device__ __host__ void Normalize() {
        (*this) = (*this) * (1.0f / (*this).Length());
    }
    __device__ __host__ void vectPow(float f) {
        x = pow(x, f); y = pow(y, f); z = pow(z, f);
    }
    __device__ __host__ void ClipMaxValue(float f) {
        if (x > f) { x = f; }
        if (y > f) { y = f; }
        if (z > f) { z = f; }
    }
    __device__ __host__ void ClipMinValue(float f) {
        if (x < f) { x = f; }
        if (y < f) { y = f; }
        if (z < f) { z = f; }
    }


    //--------------------------------------------------------------- Slope coefficents methods
    __device__ __host__ static float XYSlopeCoefficent(Vector3 const& v1, Vector3 const& v2) { //Calculate the angular coefficent between 2 points (x = m * y)
        float distY = (v2.y - v1.y);
        if (distY == 0) distY = 0.0001f;
        return (v2.x - v1.x) / distY;
    }
    __device__ __host__ static float XZSlopeCoefficent(Vector3 const& v1, Vector3 const& v2) { //Calculate the angular coefficent between 2 points (y = m * x)
        float distZ = (v2.z - v1.z);
        if (distZ == 0) distZ = 0.0001f;
        return (v2.x - v1.x) / distZ;
    }
    __device__ __host__ static float YXSlopeCoefficent(Vector3 const& v1, Vector3 const& v2) { //Calculate the angular coefficent between 2 points (y = m * x)
        float distX = (v2.x - v1.x);
        if (distX == 0) distX = 0.0001f;
        return (v2.y - v1.y) / distX;
    }
    __device__ __host__ static float YZSlopeCoefficent(Vector3 const& v1, Vector3 const& v2) { //Calculate the angular coefficent between 2 points (y = m * x)
        float distZ = (v2.z - v1.z);
        if (distZ == 0) distZ = 0.0001f;
        return (v2.y - v1.y) / distZ;
    }
    __device__ __host__ static float ZXSlopeCoefficent(Vector3 const& v1, Vector3 const& v2) { //Calculate the angular coefficent between 2 points (y = m * x)
        float distX = (v2.x - v1.x);
        if (distX == 0) distX = 0.0001f;
        return (v2.z - v1.z) / distX;
    }
    __device__ __host__ static float ZYSlopeCoefficent(Vector3 const& v1, Vector3 const& v2) { //Calculate the angular coefficent between 2 points (y = m * x)
        float distY = (v2.y - v1.y);
        if (distY == 0) distY = 0.0001f;
        return (v2.z - v1.z) / distY;
    }


    //--------------------------------------------------------------- Static methods
    __device__ __host__ static Vector3 CrossProduct(Vector3 v1, Vector3 v2) {
        return v1 * v2;
    }
    __device__ __host__ static Vector3 Normalized(Vector3 const& vect) {
        Vector3 result = vect;
        result.Normalize();
        return result;
    }
    static float LengthOf(Vector3 const& vect) {
        return vect.Length();
    }
    static Vector3 VectAss(float x, float y, float z) {
        return Vector3(x, y, z);
    }
    __device__ __host__ static Vector2 ToVect2(Vector3 const& vect) {
        return Vector2(vect.x, vect.y);
    }

    __device__ __host__ static float DotProduct(Vector3 const& v1, Vector3 const& v2) {
        return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
    }

    static void PrintV(Vector3 const& vect, bool nl = true) {
        std::cout << "(" << vect.x << "," << vect.y << "," << vect.z << ")";
        if (nl) { std::cout << std::endl; }
    }
};

const Vector3 Vector3::m_nullVector = Vector3();

struct Matrix_3x3 {
    Vector3 fc, sc, tc;

    Matrix_3x3() = default;
    ~Matrix_3x3() = default;
    __device__ __host__ Matrix_3x3(Vector3 _fc, Vector3 _sc, Vector3 _tc) {
        fc = _fc; sc = _sc; tc = _tc;
    }

    __device__ __host__ Vector3 operator* (Vector3& v) {
        float x = fc.x * v.x + sc.x * v.y + tc.x * v.z;
        float y = fc.y * v.x + sc.y * v.y + tc.y * v.z;
        float z = fc.z * v.x + sc.z * v.y + tc.z * v.z;
        return Vector3(x, y, z);
    }
    __device__ __host__ static Vector3 Product(Matrix_3x3& m, Vector3& v) {
        float x = m.fc.x * v.x + m.sc.x * v.y + m.tc.x * v.z;
        float y = m.fc.y * v.x + m.sc.y * v.y + m.tc.y * v.z;
        float z = m.fc.z * v.x + m.sc.z * v.y + m.tc.z * v.z;
        return Vector3(x,y,z);
    }
};