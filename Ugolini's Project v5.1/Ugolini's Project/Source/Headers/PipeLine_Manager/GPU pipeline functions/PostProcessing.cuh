#pragma once
#include "Pipeline functions general defines.cuh"

#define BLUR_WEIGHTGRID_DIM 5
//Blur weight grid defining
#define BWG_00 (1.0f	/ 63)
#define BWG_01 (2.0f	/ 63)
#define BWG_02 (2.5f	/ 63)
#define BWG_03 (2.0f	/ 63)
#define BWG_04 (1.0f	/ 63)
					 
#define BWG_10 (2.0f    / 63)
#define BWG_11 (3.0f    / 63)
#define BWG_12 (4.0f    / 63)
#define BWG_13 (3.0f    / 63)
#define BWG_14 (2.0f    / 63)
					 
#define BWG_20 (2.5f	/ 63)
#define BWG_21 (4.0f	/ 63)
#define BWG_22 (5.0f	/ 63)
#define BWG_23 (4.0f	/ 63)
#define BWG_24 (2.5f	/ 63)

#define BWG_30 (2.0f	/ 63)
#define BWG_31 (3.0f	/ 63)
#define BWG_32 (4.0f    / 63)
#define BWG_33 (3.0f	/ 63)
#define BWG_34 (2.0f	/ 63)

#define BWG_40 (1.0f	/ 63)
#define BWG_41 (2.0f	/ 63)
#define BWG_42 (2.5f	/ 63)
#define BWG_43 (2.0f	/ 63)
#define BWG_44 (1.0f	/ 63)
//End blur weight grid defining

#define BLOOM_THRESHOLD 1.3f
#define BLOOM_COOL_BLOOM true
#define BLOOM_EXTENDED_BLOOM true
#define BLOOM_SUPER_EXTENDED_BLOOM true
#define TONE_MAPPING true
#define GAMMA 1.5f

//Blur
__device__ RgbVector blurPixel(RgbVector* firstImagePixel, uint32 firstImageWidth);
__device__ RgbVector blurPixel__oneSide__blurLeft(RgbVector* imagePixel, uint32 imageWidth);
__device__ RgbVector blurPixel__oneSide__blurRight(RgbVector* imagePixel, uint32 imageWidth);
//Bloom functions
__global__ void Bloom__thresholdPass(RgbVector* p_fullRes, RgbVector* p_rgbHDRMap, uint32 lenght);
__global__ void Bloom__generalDWS(RgbVector* p_smallerDWS, RgbVector* p_biggerDWS, uint32 bigDWSred);
__global__ void Bloom__generalBlur(RgbVector* p_blurredImg, RgbVector* p_DWSImg, uint32 DWSred);
__global__ void Bloom__addUp(RgbVector* p_fstB, RgbVector* p_sndB, RgbVector* p_trdB, RgbVector* p_frthB, RgbVector* p_fifthB, RgbVector* p_sixthB, RgbVector* p_svnthB, RgbVector* fullRes, RgbVector* p_rgbHDR, uint32 lenght);

struct BloomHandler {
	RgbVector* p_fullRes, *p_firstDWS, * p_secondDWS, * p_thirdDWS, * p_fourthDWS, * p_fifthDWS, * p_sixthDWS, * p_seventhDWS; //dws downscale
	RgbVector* p_firstBLUR, * p_secondBLUR, * p_thirdBLUR, * p_fourthBLUR, * p_fifthBLUR, * p_sixthBLUR, * p_seventhBLUR; //blur

	//----------------------------------------------------------------------------------------------------------- Constructors
#define BLOOM__UPDOWN_IMGBUFFER_LINE 3
	BloomHandler() { AlloatecBloomBuffers(); }
	~BloomHandler() { DeallocateBloomBuffers(); }
	void ResetBloom() {
		DeallocateBloomBuffers();
		AlloatecBloomBuffers();
	}
	void AlloatecBloomBuffers() {

		//Create a temp buffer to store the 0s values and copy them into the GPU in order to initialize the buffers
		void* p_tempAlloc{ nullptr };
		cudaError_t tempAlloc = cudaMallocHost((void**)&p_tempAlloc, (SCREEN_WIDTH * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER)) * sizeof(RgbVector));
		if ((tempAlloc != cudaSuccess)) { crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return; }
		for (uint32 i = 0; i < (SCREEN_WIDTH * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER)); i++) {
			((RgbVector*)p_tempAlloc)[i] = RgbVector(0);
		}

		//Allocate and initialize the downscale buffers
		{
			//Allocation
			cudaError_t fullRes = cudaMalloc((void**)&p_fullRes, (SCREEN_WIDTH * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER)) * sizeof(RgbVector));
			cudaError_t firstDWS = cudaMalloc((void**)&p_firstDWS, ((SCREEN_WIDTH / 2) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 4)) * sizeof(RgbVector));
			cudaError_t secondDWS = cudaMalloc((void**)&p_secondDWS, ((SCREEN_WIDTH / 4) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 16)) * sizeof(RgbVector));
			cudaError_t thirdDWS = cudaMalloc((void**)&p_thirdDWS, ((SCREEN_WIDTH / 8) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 64)) * sizeof(RgbVector));
			//Check for errors
			if ((fullRes != cudaSuccess) || (firstDWS != cudaSuccess) || (secondDWS != cudaSuccess) || (thirdDWS != cudaSuccess)) {
				crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
			//Initialization
			cudaMemcpy(p_fullRes, p_tempAlloc, (SCREEN_WIDTH * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER)) * sizeof(RgbVector), cudaMemcpyHostToDevice);  cudaDeviceSynchronize();
			cudaMemcpy(p_firstDWS, p_tempAlloc, ((SCREEN_WIDTH / 2) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 4)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			cudaMemcpy(p_secondDWS, p_tempAlloc, ((SCREEN_WIDTH / 4) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 16)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			cudaMemcpy(p_thirdDWS, p_tempAlloc, ((SCREEN_WIDTH / 8) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 64)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			p_fullRes = &(p_fullRes[SCREEN_WIDTH * BLOOM__UPDOWN_IMGBUFFER_LINE]); p_firstDWS = &(p_firstDWS[(SCREEN_WIDTH / 2) * BLOOM__UPDOWN_IMGBUFFER_LINE]);
			p_secondDWS = &(p_secondDWS[(SCREEN_WIDTH / 4) * BLOOM__UPDOWN_IMGBUFFER_LINE]); p_thirdDWS = &(p_thirdDWS[(SCREEN_WIDTH / 8) * BLOOM__UPDOWN_IMGBUFFER_LINE]);

			//Allocation
			cudaError_t fourthDWS = cudaMalloc((void**)&p_fourthDWS, ((SCREEN_WIDTH / 16) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 256)) * sizeof(RgbVector));
			cudaError_t fifthDWS = cudaMalloc((void**)&p_fifthDWS, ((SCREEN_WIDTH / 32) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 1024)) * sizeof(RgbVector));
			cudaError_t sixthDWS = cudaMalloc((void**)&p_sixthDWS, ((SCREEN_WIDTH / 64) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 4096)) * sizeof(RgbVector));
			cudaError_t seventhDWS = cudaMalloc((void**)&p_seventhDWS, ((SCREEN_WIDTH / 128) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 16384)) * sizeof(RgbVector));
			//Check for errors
			if ((fourthDWS != cudaSuccess) || (fifthDWS != cudaSuccess) || (sixthDWS != cudaSuccess) || (seventhDWS != cudaSuccess)) {
				crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
			//Initialization
			cudaMemcpy(p_fourthDWS, p_tempAlloc, ((SCREEN_WIDTH / 16) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 256)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			cudaMemcpy(p_fifthDWS, p_tempAlloc, ((SCREEN_WIDTH / 32) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 1024)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			cudaMemcpy(p_sixthDWS, p_tempAlloc, ((SCREEN_WIDTH / 64) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 4096)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			cudaMemcpy(p_seventhDWS, p_tempAlloc, ((SCREEN_WIDTH / 128) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 16384)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			p_fourthDWS = &(p_fourthDWS[(SCREEN_WIDTH / 16) * BLOOM__UPDOWN_IMGBUFFER_LINE]); p_fifthDWS = &(p_fifthDWS[(SCREEN_WIDTH / 32) * BLOOM__UPDOWN_IMGBUFFER_LINE]);
			p_sixthDWS = &(p_sixthDWS[(SCREEN_WIDTH / 64) * BLOOM__UPDOWN_IMGBUFFER_LINE]); p_seventhDWS = &(p_seventhDWS[(SCREEN_WIDTH / 128) * BLOOM__UPDOWN_IMGBUFFER_LINE]);
		}
		//Allocate and initialize the blur buffers
		{
			//Allocation
			cudaError_t firstBLUR = cudaMalloc((void**)&p_firstBLUR, ((SCREEN_WIDTH / 2) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 4)) * sizeof(RgbVector));
			cudaError_t secondBLUR = cudaMalloc((void**)&p_secondBLUR, ((SCREEN_WIDTH / 4) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 16)) * sizeof(RgbVector));
			cudaError_t thirdBLUR = cudaMalloc((void**)&p_thirdBLUR, ((SCREEN_WIDTH / 8) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 64)) * sizeof(RgbVector));
			//Check for errors
			if ((firstBLUR != cudaSuccess) || (secondBLUR != cudaSuccess) || (thirdBLUR != cudaSuccess)) {
				crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
			//Initialization
			cudaMemcpy(p_firstBLUR, p_tempAlloc, ((SCREEN_WIDTH / 2) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 4)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			cudaMemcpy(p_secondBLUR, p_tempAlloc, ((SCREEN_WIDTH / 4) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 16)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			cudaMemcpy(p_thirdBLUR, p_tempAlloc, ((SCREEN_WIDTH / 8) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 64)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			p_firstBLUR = &(p_firstBLUR[(SCREEN_WIDTH / 2) * BLOOM__UPDOWN_IMGBUFFER_LINE]); p_secondBLUR = &(p_secondBLUR[(SCREEN_WIDTH / 4) * BLOOM__UPDOWN_IMGBUFFER_LINE]);
			p_thirdBLUR = &(p_thirdBLUR[(SCREEN_WIDTH / 8) * BLOOM__UPDOWN_IMGBUFFER_LINE]);

			//Allocation
			cudaError_t fourthBLUR = cudaMalloc((void**)&p_fourthBLUR, ((SCREEN_WIDTH / 16) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 256)) * sizeof(RgbVector));
			cudaError_t fifthBLUR = cudaMalloc((void**)&p_fifthBLUR, ((SCREEN_WIDTH / 32) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 1024)) * sizeof(RgbVector));
			cudaError_t sixthBLUR = cudaMalloc((void**)&p_sixthBLUR, ((SCREEN_WIDTH / 64) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 4096)) * sizeof(RgbVector));
			cudaError_t seventhBLUR = cudaMalloc((void**)&p_seventhBLUR, ((SCREEN_WIDTH / 128) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 16384)) * sizeof(RgbVector));
			//Check for errors
			if ((fourthBLUR != cudaSuccess) || (fifthBLUR != cudaSuccess) || (sixthBLUR != cudaSuccess) || (seventhBLUR != cudaSuccess)) {
				crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
			cudaMemcpy(p_fourthBLUR, p_tempAlloc, ((SCREEN_WIDTH / 16) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 256)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			cudaMemcpy(p_fifthBLUR, p_tempAlloc, ((SCREEN_WIDTH / 32) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 1024)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			cudaMemcpy(p_sixthBLUR, p_tempAlloc, ((SCREEN_WIDTH / 64) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 4096)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			cudaMemcpy(p_seventhBLUR, p_tempAlloc, ((SCREEN_WIDTH / 128) * BLOOM__UPDOWN_IMGBUFFER_LINE * 2 + (SCREEN_PIXELNUMBER / 16384)) * sizeof(RgbVector), cudaMemcpyHostToDevice); cudaDeviceSynchronize();
			p_fourthBLUR = &(p_fourthBLUR[(SCREEN_WIDTH / 16) * BLOOM__UPDOWN_IMGBUFFER_LINE]); p_fifthBLUR = &(p_fifthBLUR[(SCREEN_WIDTH / 32) * BLOOM__UPDOWN_IMGBUFFER_LINE]);
			p_sixthBLUR = &(p_sixthBLUR[(SCREEN_WIDTH / 64) * BLOOM__UPDOWN_IMGBUFFER_LINE]); p_seventhBLUR = &(p_seventhBLUR[(SCREEN_WIDTH / 128) * BLOOM__UPDOWN_IMGBUFFER_LINE]);
		}
		cudaFree(p_tempAlloc);
	}
	void DeallocateBloomBuffers() {
		cudaFree(p_fullRes);			cudaFree(p_firstDWS);
		cudaFree(p_secondDWS);			cudaFree(p_thirdDWS);
		cudaFree(p_fourthDWS);			cudaFree(p_fifthDWS);
		cudaFree(p_sixthDWS);			cudaFree(p_seventhDWS);

		cudaFree(p_firstBLUR);
		cudaFree(p_secondBLUR);			cudaFree(p_thirdBLUR);
		cudaFree(p_fourthBLUR);			cudaFree(p_fifthBLUR);
		cudaFree(p_sixthBLUR);			cudaFree(p_seventhBLUR);
	}

	void Bloom(RgbVector* p_rgbHDR, cudaStream_t stream) {
		Bloom__thresholdPass<<<((SCREEN_PIXELNUMBER / 1) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_fullRes, p_rgbHDR, SCREEN_PIXELNUMBER);

		Bloom__generalDWS<<<((SCREEN_PIXELNUMBER  / 4    ) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_firstDWS,		p_fullRes,		1);
		Bloom__generalBlur<<<((SCREEN_PIXELNUMBER / 4    ) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_firstBLUR,		p_firstDWS,		2);
																			
		Bloom__generalDWS<<<((SCREEN_PIXELNUMBER  / 16   ) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_secondDWS,		p_firstBLUR,	2);
		Bloom__generalBlur<<<((SCREEN_PIXELNUMBER / 16   ) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_secondBLUR,    p_secondDWS,    4);
																			
		if(BLOOM_COOL_BLOOM || true) Bloom__generalDWS<<<((SCREEN_PIXELNUMBER  / 64   ) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_thirdDWS,      p_secondBLUR,	4);
		if(BLOOM_COOL_BLOOM || true) Bloom__generalBlur<<<((SCREEN_PIXELNUMBER / 64   ) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_thirdBLUR,	 p_thirdDWS,	8);
																		
		if((BLOOM_EXTENDED_BLOOM && BLOOM_COOL_BLOOM) || true) Bloom__generalDWS<<<((SCREEN_PIXELNUMBER  / 256  ) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_fourthDWS,      p_thirdBLUR,	8);
		if((BLOOM_EXTENDED_BLOOM && BLOOM_COOL_BLOOM) || true) Bloom__generalBlur<<<((SCREEN_PIXELNUMBER / 256  ) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_fourthBLUR,	    p_fourthDWS,	16);
																
		if((BLOOM_EXTENDED_BLOOM && BLOOM_COOL_BLOOM) || true) Bloom__generalDWS<<<((SCREEN_PIXELNUMBER  / 1024 ) / MAX_THREAD_GPU) + 2, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_fifthDWS,		p_fourthBLUR,	16);
		if((BLOOM_EXTENDED_BLOOM && BLOOM_COOL_BLOOM) || true) Bloom__generalBlur<<<((SCREEN_PIXELNUMBER / 1024 ) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_fifthBLUR,		p_fifthDWS,		16);
																			
		if((BLOOM_SUPER_EXTENDED_BLOOM && BLOOM_COOL_BLOOM) || true) Bloom__generalDWS<<<((SCREEN_PIXELNUMBER / 4096) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_sixthDWS, p_fifthBLUR, 32);
		if((BLOOM_SUPER_EXTENDED_BLOOM && BLOOM_COOL_BLOOM) || true) Bloom__generalBlur<<<((SCREEN_PIXELNUMBER / 4096) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_sixthBLUR, p_sixthDWS, 32);

		if((BLOOM_SUPER_EXTENDED_BLOOM && BLOOM_COOL_BLOOM) || true) Bloom__generalDWS<<<((SCREEN_PIXELNUMBER / 16384) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_seventhDWS, p_sixthBLUR, 64);
		if((BLOOM_SUPER_EXTENDED_BLOOM && BLOOM_COOL_BLOOM) || true) Bloom__generalBlur<<<((SCREEN_PIXELNUMBER / 16384) / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_seventhBLUR,	p_seventhDWS,	64);

		Bloom__addUp<<<(SCREEN_PIXELNUMBER / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, stream>>>(p_firstBLUR, p_secondBLUR, p_thirdBLUR, p_fourthBLUR, p_fifthBLUR, p_sixthBLUR, p_seventhBLUR, p_fullRes, p_rgbHDR, SCREEN_PIXELNUMBER);
	}
};
BloomHandler bloomHandler;

//Bloom
__global__ void Bloom__thresholdPass(RgbVector* p_fullRes, RgbVector* p_rgbHDRMap, uint32 lenght) { //SCREEN_PIXELNUMBER times
	uint32 threadId = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadId >= lenght) return;

	for (uint32 pixelId = threadId * PIXEL_PER_KERNEL; pixelId < (threadId + 1) * PIXEL_PER_KERNEL; pixelId++) {
		p_fullRes[pixelId] = p_rgbHDRMap[pixelId];
		p_fullRes[pixelId].Threshold(BLOOM_THRESHOLD);
	}
}
__global__ void Bloom__generalDWS(RgbVector* p_smallerDWS, RgbVector* p_biggerDWS, uint32 bigDWSred) {
	uint32 dwspid = threadIdx.x + blockDim.x * blockIdx.x; //downscale pixel id

	//Ress reduction litte/big image
	uint32 smallDWSred = bigDWSred * 2;
	uint32 widthBigDWS = SCREEN_WIDTH / bigDWSred, widthSmallDWS = SCREEN_WIDTH / smallDWSred;

	uint32 maxPixelNum = SCREEN_PIXELNUMBER / (bigDWSred * bigDWSred);
	if (dwspid >= maxPixelNum) return;
	uint32 column = (dwspid % widthSmallDWS) * 2;
	int row = (dwspid / widthSmallDWS) * 2;
	uint32 pixelId = row * widthBigDWS + column;

	if ((row < 2) || (row > (SCREEN_HEIGHT / bigDWSred))) return;
	p_smallerDWS[dwspid] = p_biggerDWS[pixelId];
	p_smallerDWS[dwspid] += p_biggerDWS[pixelId + widthBigDWS];
	if (p_smallerDWS[dwspid].red + p_smallerDWS[dwspid].blue + p_smallerDWS[dwspid].green > 17){ 
		p_smallerDWS[dwspid] = RgbVector(0, 0, 0); return;
	}

	if ((column < widthBigDWS - 2) && (column > 2)){
		p_smallerDWS[dwspid] += p_biggerDWS[pixelId + 1];
		p_smallerDWS[dwspid] += p_biggerDWS[pixelId + widthBigDWS + 1];
		p_smallerDWS[dwspid] *= 0.25;
	}else {
		p_smallerDWS[dwspid] *= 0.5;
	}
}
__global__ void Bloom__generalBlur(RgbVector* p_blurredImg, RgbVector* p_DWSImg, uint32 DWSred) {
	uint32 dwspid = threadIdx.x + blockDim.x * blockIdx.x; //downscale pixel id
	uint32 DWSImgWidth = SCREEN_WIDTH / DWSred;
	uint32 column = ((dwspid) % DWSImgWidth);
	if (dwspid >= SCREEN_PIXELNUMBER / (DWSred * DWSred)) return;

	if (column <= 2) {
		int upperLeftPixel = (dwspid - DWSImgWidth * 2);
		p_blurredImg[dwspid] = blurPixel__oneSide__blurRight(&(p_DWSImg[upperLeftPixel]), DWSImgWidth);
		return;
	}
	if (column - 2 >= DWSImgWidth) {
		int upperLeftPixel = (dwspid - 2 - DWSImgWidth * 2);
		p_blurredImg[dwspid] = blurPixel__oneSide__blurLeft(&(p_DWSImg[upperLeftPixel]), DWSImgWidth);
		return;
	}
	int upperLeftPixel = (dwspid - 2 - DWSImgWidth * 2);
	p_blurredImg[dwspid] = blurPixel(&(p_DWSImg[upperLeftPixel]), DWSImgWidth);
}
__device__ RgbVector Bloom_upScaleDWSBlurredPixel(RgbVector* p_DWSBlurredImg, float column, float row, uint32 DWSred) {
	uint32 tempScreenWidth = (SCREEN_WIDTH / DWSred);
	
	column = (column / DWSred); row = (row / DWSred);
	uint32 intColumn = (uint32)(myMin(column, tempScreenWidth - 1)), intRow = (uint32)(row);
	float decPartColumn = column - (float)intColumn, decPartRow = row - (float)intRow;
	
	int centralPixelId = intColumn + intRow * tempScreenWidth;
	int upPixelId = centralPixelId + tempScreenWidth;
	int downPixelId = centralPixelId - tempScreenWidth;
	
	//Cross interpolation
	//010
	//111
	//010 
	RgbVector finalColor = p_DWSBlurredImg[centralPixelId] * 0.2f;
	float denominator = 0.2f;
	if (intColumn < 1) {
		finalColor += p_DWSBlurredImg[centralPixelId - 1] * (1.0f - decPartColumn); denominator += (1.0f - decPartColumn);
	}else if (intColumn >= tempScreenWidth - 1) {
		finalColor += p_DWSBlurredImg[centralPixelId + 1] * decPartColumn; denominator += decPartColumn;
	}else {
		finalColor += p_DWSBlurredImg[centralPixelId - 1] * (1.0f - decPartColumn); 
		finalColor += p_DWSBlurredImg[centralPixelId + 1] * decPartColumn;
		denominator += 1;
	}

	if (downPixelId < 1) {
		finalColor += p_DWSBlurredImg[downPixelId] * (1.0f - decPartRow); denominator += (1.0f - decPartRow);
	}else if (upPixelId > SCREEN_PIXELNUMBER - SCREEN_WIDTH) {
		finalColor += p_DWSBlurredImg[upPixelId] * decPartRow; denominator += decPartRow;
	}
	else {
		finalColor += p_DWSBlurredImg[downPixelId] * (1.0f - decPartRow);
		finalColor += p_DWSBlurredImg[upPixelId] * decPartRow;
		denominator += 1;
	}
	finalColor *= (1.0f / denominator);
	return finalColor;
}
__global__ void Bloom__addUp(RgbVector* p_fstB, RgbVector* p_sndB, RgbVector* p_trdB, RgbVector* p_frthB, RgbVector* p_fifthB, RgbVector* p_sixthB, RgbVector* p_svnthB, RgbVector* fullRes, RgbVector* p_rgbHDR, uint32 lenght) { //(SCREEN_PIXELNUMBER) times
	uint32 pixelId = threadIdx.x + blockDim.x * blockIdx.x; 
	if (pixelId >= lenght) return;
	uint32 column = (pixelId % (SCREEN_WIDTH));
	uint32 row = ((pixelId / (SCREEN_WIDTH)));
	
	if (BLOOM_COOL_BLOOM) {
		p_rgbHDR[pixelId] += fullRes[pixelId];
		p_rgbHDR[pixelId] += Bloom_upScaleDWSBlurredPixel(p_fstB, column, row, 2) * 0.5;
		p_rgbHDR[pixelId] += Bloom_upScaleDWSBlurredPixel(p_sndB, column, row, 4) * 0.4;
		p_rgbHDR[pixelId] += Bloom_upScaleDWSBlurredPixel(p_trdB, column, row, 8) * 0.2;
		if (BLOOM_EXTENDED_BLOOM) p_rgbHDR[pixelId] += Bloom_upScaleDWSBlurredPixel(p_frthB, column, row, 16) * 0.04;
		if (BLOOM_EXTENDED_BLOOM) p_rgbHDR[pixelId] += Bloom_upScaleDWSBlurredPixel(p_fifthB, column, row, 32) * 0.04;
		if (BLOOM_SUPER_EXTENDED_BLOOM) p_rgbHDR[pixelId] += Bloom_upScaleDWSBlurredPixel(p_sixthB, column, row, 64) * 0.02;
		if (BLOOM_SUPER_EXTENDED_BLOOM) p_rgbHDR[pixelId] += Bloom_upScaleDWSBlurredPixel(p_svnthB, column, row, 128) * 0.02;
	}
	else {
		p_rgbHDR[pixelId] += fullRes[pixelId];
		p_rgbHDR[pixelId] += p_fstB[myMin((column / 2), (SCREEN_WIDTH / 2) - 1) + (row / 2) * (SCREEN_WIDTH / 2)] * 0.5;
		p_rgbHDR[pixelId] += p_sndB[myMin((column / 4), (SCREEN_WIDTH / 4) - 1) + (row / 4) * (SCREEN_WIDTH / 4)] * 0.4;
		//if (BLOOM_EXTENDED_BLOOM) p_rgbHDR[pixelId] = p_trdB[myMin((column / 8), (SCREEN_WIDTH / 8) - 1) + (row / 8) * (SCREEN_WIDTH / 8)] * 0.2;
		//if (BLOOM_EXTENDED_BLOOM) p_rgbHDR[pixelId] = p_frthB[	myMin((column / 16 ), (SCREEN_WIDTH / 16 ) - 1) + (row / 16 ) * (SCREEN_WIDTH / 16 )] * 0.04;
		//if (BLOOM_SUPER_EXTENDED_BLOOM && BLOOM_EXTENDED_BLOOM) p_rgbHDR[pixelId] += p_fifthB[	myMin((column / 32 ), (SCREEN_WIDTH / 32 ) - 1) + (row / 32 ) * (SCREEN_WIDTH / 32 )] * 0.04;
		//if (BLOOM_SUPER_EXTENDED_BLOOM && BLOOM_EXTENDED_BLOOM) p_rgbHDR[pixelId] += p_sixthB[	myMin((column / 64 ), (SCREEN_WIDTH / 64 ) - 1) + (row / 64 ) * (SCREEN_WIDTH / 64 )] * 0.02;
		//if (BLOOM_SUPER_EXTENDED_BLOOM && BLOOM_EXTENDED_BLOOM) p_rgbHDR[pixelId] += p_svnthB[	myMin((column / 128), (SCREEN_WIDTH / 128) - 1) + (row / 128) * (SCREEN_WIDTH / 128)] * 0.02;
	}
}

//Blur
__device__ RgbVector blurPixel(RgbVector* imagePixel, uint32 imageWidth) { //The pixel of the image should be the upper left one
	RgbVector tempColor(0);
	tempColor = imagePixel[0] * BWG_00;
	tempColor += imagePixel[1] * BWG_01;
	tempColor += imagePixel[2] * BWG_02;
	tempColor += imagePixel[3] * BWG_03;
	tempColor += imagePixel[4] * BWG_04;

	uint32 temp = imageWidth;
	tempColor += imagePixel[0 + temp] * BWG_10;
	tempColor += imagePixel[1 + temp] * BWG_11;
	tempColor += imagePixel[2 + temp] * BWG_12;
	tempColor += imagePixel[3 + temp] * BWG_13;
	tempColor += imagePixel[4 + temp] * BWG_14;

	temp += imageWidth;
	tempColor += imagePixel[0 + temp] * BWG_20;
	tempColor += imagePixel[1 + temp] * BWG_21;
	tempColor += imagePixel[2 + temp] * BWG_22;
	tempColor += imagePixel[3 + temp] * BWG_23;
	tempColor += imagePixel[4 + temp] * BWG_24;

	temp += imageWidth;
	tempColor += imagePixel[0 + temp] * BWG_30;
	tempColor += imagePixel[1 + temp] * BWG_31;
	tempColor += imagePixel[2 + temp] * BWG_32;
	tempColor += imagePixel[3 + temp] * BWG_33;
	tempColor += imagePixel[4 + temp] * BWG_34;

	temp += imageWidth;
	tempColor += imagePixel[0 + temp] * BWG_40;
	tempColor += imagePixel[1 + temp] * BWG_41;
	tempColor += imagePixel[2 + temp] * BWG_42;
	tempColor += imagePixel[3 + temp] * BWG_43;
	tempColor += imagePixel[4 + temp] * BWG_44;

	return tempColor;
}
__device__ RgbVector blurPixel__oneSide__blurLeft(RgbVector* imagePixel, uint32 imageWidth) { //The pixel of the image should be the upper left one
	RgbVector tempColor(0);
	tempColor = imagePixel[0] * BWG_00;
	tempColor += imagePixel[1] * BWG_01;
	tempColor += imagePixel[2] * BWG_02;

	uint32 temp = imageWidth;
	tempColor += imagePixel[0 + temp] * BWG_10;
	tempColor += imagePixel[1 + temp] * BWG_11;
	tempColor += imagePixel[2 + temp] * BWG_12;

	temp += imageWidth;
	tempColor += imagePixel[0 + temp] * BWG_20;
	tempColor += imagePixel[1 + temp] * BWG_21;
	tempColor += imagePixel[2 + temp] * BWG_22;
	
	temp += imageWidth;
	tempColor += imagePixel[0 + temp] * BWG_30;
	tempColor += imagePixel[1 + temp] * BWG_31;
	tempColor += imagePixel[2 + temp] * BWG_32;

	temp += imageWidth;
	tempColor += imagePixel[0 + temp] * BWG_40;
	tempColor += imagePixel[1 + temp] * BWG_41;
	tempColor += imagePixel[2 + temp] * BWG_42;

	return tempColor;
}
__device__ RgbVector blurPixel__oneSide__blurRight(RgbVector* imagePixel, uint32 imageWidth) { //The pixel of the image should be the upper left one
	RgbVector tempColor(0);
	tempColor = imagePixel[0] * BWG_02;
	tempColor += imagePixel[1] * BWG_03;
	tempColor += imagePixel[2] * BWG_04;

	uint32 temp = imageWidth;
	tempColor += imagePixel[0 + temp] * BWG_12;
	tempColor += imagePixel[1 + temp] * BWG_13;
	tempColor += imagePixel[2 + temp] * BWG_14;

	temp += imageWidth;
	tempColor += imagePixel[0 + temp] * BWG_22;
	tempColor += imagePixel[1 + temp] * BWG_23;
	tempColor += imagePixel[2 + temp] * BWG_24;

	temp += imageWidth;
	tempColor += imagePixel[0 + temp] * BWG_32;
	tempColor += imagePixel[1 + temp] * BWG_33;
	tempColor += imagePixel[2 + temp] * BWG_34;

	temp += imageWidth;
	tempColor += imagePixel[0 + temp] * BWG_42;
	tempColor += imagePixel[1 + temp] * BWG_43;
	tempColor += imagePixel[2 + temp] * BWG_44;

	return tempColor;
}


//Tone mapping
float gamma = GAMMA;
__global__ void ToneMapping(RgbVector* p_rgbHDRMap, byte* p_rgbMap, float gamma, uint32 lenght) { //Reinhard tonemapping
	uint32 threadId = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadId >= lenght) return;

	for (uint32 pixelId = threadId * PIXEL_PER_KERNEL; pixelId < (threadId + 1) * PIXEL_PER_KERNEL; pixelId++) {
		RgbVector finalColor; 
		if (TONE_MAPPING) {
			RgbVector temp = p_rgbHDRMap[pixelId];
			finalColor.red = ((temp.red * (temp.red * 2.51f + 0.03f)) / (temp.red * (temp.red * 2.43f + 0.59f) + 0.14f));
			finalColor.green = ((temp.green * (temp.green * 2.51f + 0.03f)) / (temp.green * (temp.green * 2.43f + 0.59f) + 0.14f));
			finalColor.blue = ((temp.blue * (temp.blue * 2.51f + 0.03f)) / (temp.blue * (temp.blue * 2.43f + 0.59f) + 0.14f));

			finalColor.powRgbVector(1.0f / gamma);
			finalColor.Cap();
		}
		else {
			finalColor = RgbVector(p_rgbHDRMap[pixelId].red, p_rgbHDRMap[pixelId].green, p_rgbHDRMap[pixelId].blue); 
			finalColor.Cap();
		}
		BMP_IMAGE_SETCOLOR(p_rgbMap, pixelId, (uint8)(finalColor.red * 255), (uint8)(finalColor.green * 255), (uint8)(finalColor.blue * 255));
	}
}