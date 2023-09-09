#pragma once
#include <iostream>
#include <cmath>



struct Vector2 {

    //--------------------------------------------------------------- Members
public:
    float x = 0, y = 0;


    //--------------------------------------------------------------- Constructors
    __device__ __host__ Vector2() {
        x = 0, y = 0;
    }
    __device__ __host__ Vector2(float num) {
        x = num, y = num;
    }
    __device__ __host__ Vector2(float argX, float argY) {
        x = argX, y = argY;
    }
    ~Vector2() = default;


    //--------------------------------------------------------------- Operator overload
        //Vector2
    __device__ __host__ Vector2 operator+ (Vector2 const& v2) const {
        return Vector2(x + v2.x, y + v2.y);
    }
    __device__ __host__ Vector2 operator- (Vector2 const& v2) const {
        return Vector2(x - v2.x, y - v2.y);
    }

    __device__ __host__ Vector2 operator+ (float const& f) const {
        return Vector2(x + f, y + f);
    }
    __device__ __host__ Vector2 operator- (float const& f) const {
        return Vector2(x - f, y - f);
    }
    __device__ __host__ Vector2 operator* (float const& f) const {
        return Vector2(x * f, y * f);
    }

    //Bool 
    __device__ __host__ bool operator== (Vector2 const& v2) const {
        return (x == v2.x && y == v2.y);
    }
    __device__ __host__ bool operator> (Vector2 const& v2) const {
        return (x > v2.x && y > v2.y);
    }
    __device__ __host__ bool operator< (Vector2 const& v2) const {
        return (x < v2.x&& y < v2.y);
    }
    __device__ __host__ bool operator>= (Vector2 const& v2) const {
        return (x >= v2.x && y >= v2.y);
    }
    __device__ __host__ bool operator<= (Vector2 const& v2) const {
        return (x <= v2.x && y <= v2.y);
    }
    __device__ __host__ bool operator== (float const& f) const {
        return (x == f && y == f);
    }
    __device__ __host__ bool operator> (float const& f) const {
        return (x > f && y > f);
    }
    __device__ __host__ bool operator< (float const& f) const {
        return (x < f&& y < f);
    }
    __device__ __host__ bool operator>= (float const& f) const {
        return (x >= f && y >= f);
    }
    __device__ __host__ bool operator<= (float const& f) const {
        return (x <= f && y <= f);
    }

    //Void
    __device__ __host__ void operator+= (Vector2 const& v2) {
        x += v2.x; y += v2.y;
    }
    __device__ __host__ void operator-= (Vector2 const& v2) {
        x -= v2.x; y -= v2.y;
    }
    __device__ __host__ void operator= (Vector2 const& v2) {
        x = v2.x; y = v2.y;
    }

    __device__ __host__ void operator+= (float const& f) {
        x += f; y += f;
    }
    __device__ __host__ void operator-= (float const& f) {
        x -= f; y -= f;
    }
    __device__ __host__ void operator*= (float const& f) {
        x *= f; y *= f;
    }
    __device__ __host__ void operator= (float const& f) {
        x = f; y = f;
    }


    //--------------------------------------------------------------- Methods
    float Length() const {
        return sqrt(x * x + y * y);
    }
    void Normalize() {
        (*this) = (*this) * (1 / (*this).Length());
    }
    void ClipMaxValue(float f) {
        if (x > f) { x = f; }
        if (y > f) { y = f; }
    }
    void ClipMinValue(float f) {
        if (x < f) { x = f; }
        if (y < f) { y = f; }
    }


    //--------------------------------------------------------------- Slope coefficents methods
    __device__ __host__ static float XYSlopeCoefficent(Vector2 const& v1, Vector2 const& v2) { //Calculate the angular coefficent between 2 points (x = m * y)
        float distY = (v2.y - v1.y);
        if (distY == 0) distY = 0.000001f;
        return (v2.x - v1.x) / distY;
    }
    __device__ __host__ static float YXSlopeCoefficent(Vector2 const& v1, Vector2 const& v2) { //Calculate the angular coefficent between 2 points (y = m * x)
        float distX = (v2.x - v1.x);
        if (distX == 0) distX = 0.000001f;
        return (v2.y - v1.y) / distX;
    }


    //--------------------------------------------------------------- Static methods
    static Vector2 Normalized(Vector2 const& vect) {
        Vector2 result = vect;
        result.Normalize();
        return result;
    }
    static float LengthOf(Vector2 const& vect) {
        return vect.Length();
    }
    static Vector2 VectAss(float x, float y) {
        return Vector2(x, y);
    }

    static float DotProduct(Vector2 const& v1, Vector2 const& v2) {
        return (v1.x * v2.x + v1.y * v2.y);
    }

    static void PrintV(Vector2 const& vect, bool nl = true) {
        std::cout << "(" << vect.x << "," << vect.y << ")";
        if (nl) { std::cout << std::endl; }
    }
};