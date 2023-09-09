#pragma once

//Crash handler
#define CH_NO_CRASH_OCCURRED 0
#define CH_CPU_ALLOCATION_ERROR 1
#define CH_GPU_ALLOCATION_ERROR 2
#define CH_CUDA_STREAM_CREATION_ERROR 3
#define CH_FILE_NAME_NOT_FOUND 4
#define CH_FPS_THRESHOLD_REACHED 5

struct CrashHandler {
	unsigned short crashCode = CH_NO_CRASH_OCCURRED; // 16 bits
};
CrashHandler crashHandler;