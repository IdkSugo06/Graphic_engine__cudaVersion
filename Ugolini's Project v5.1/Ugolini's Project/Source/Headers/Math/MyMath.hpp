#pragma once
#include "MyQuaternion.hpp"
#include "MyVector3.hpp"
#include "MyVector2.hpp"

#ifndef PI
#define PI (float)3.14159265357989
#endif

#ifndef MY_TYPEDEF
#define MY_TYPEDEF
typedef unsigned char uint8; //255
typedef unsigned short uint16; //65k
typedef unsigned int uint32; //4 bilion
typedef unsigned long long uint64; //1.8 * 10^19
#endif

float MaxFloat(float f1, float f2) { return (f1 > f2 ? f1 : f2); }
float MinFloat(float f1, float f2) { return (f1 < f2 ? f1 : f2); }

//(Copied, https://en.wikipedia.org/wiki/Fast_inverse_square_root)
__device__ __host__ float invSqrt(float number) {
    union {
        float f;
        uint32_t i;
    } conv;

    float x2;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    conv.f = number;
    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f = conv.f * (threehalfs - (x2 * conv.f * conv.f));
    return conv.f;
}

#define myMin(a, b) (a < b ? a : b)
#define myMax(a, b) (a > b ? a : b)
#define myBetween(a,b,c) myMin(myMax(a,b),c)			//A between b (lower) and c (higher)

#define and &&
#define or ||