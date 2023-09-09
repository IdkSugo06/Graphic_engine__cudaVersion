#pragma once
#include "../GlobalStructsHandler.cuh"

#define PIXEL_PER_KERNEL 1
#define NUMOF_THREAD_FOR_SCREEN_COMPUTATION (SCREEN_PIXELNUMBER / PIXEL_PER_KERNEL)
#define launch_bounds(x,y)  __launch_bounds__(x,y)

#define SHOW_SUBDIVIDED_SQUARED false