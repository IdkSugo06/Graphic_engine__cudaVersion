#pragma once
#include "GPU pipeline functions\Pipeline functions.cuh"
#include "chrono"

struct GPU_pipelineManager {
	void InitialDataTransfer(cudaStream_t stream = cudaStreamHandler.mainStream) {
		//Maps
		cudaMemcpyAsync(gpuSH.p_fragmentMap, cpuSH.p_nullFragmentMap, SCREEN_PIXELNUMBER * sizeof(Fragment), cudaMemcpyHostToDevice, stream);	//FragmentMap reset
		cudaMemcpyAsync(gpuSH.p_rgbHDRMap, cpuSH.p_nullRgbHDRMap, SCREEN_PIXELNUMBER * sizeof(RgbVector), cudaMemcpyHostToDevice, stream);		//RgbHDRMap reset

		//Vertecies and triangles
		cudaMemcpyAsync(gpuSH.p_absSpace_vertices, cpuSH.p_vertexArray, cpuSH.m_vectors_number * sizeof(Vector3), cudaMemcpyHostToDevice, stream); //VertexTransfer_CPU_to_GPU
		cudaMemcpyAsync(gpuSH.p_uv_coordinates, cpuSH.p_uvCoordinatesArray, cpuSH.m_uv_coordinates_number * sizeof(Vector3), cudaMemcpyHostToDevice, stream); //UvVertexTransfer_CPU_to_GPU
		cudaMemcpyAsync(gpuSH.p_trianglesId, cpuSH.p_triangleIdArray, cpuSH.m_triangles_number * sizeof(TriangleId), cudaMemcpyHostToDevice, stream); //TriangleIdTransfer_CPU_to_GPU

		//General info
		cudaMemcpyAsync(gpuSH.p_camInfo, cpuSH.p_camInfo, sizeof(GPU_camInfo), cudaMemcpyHostToDevice, stream); //CamInfoTransfer_CPU_to_GPU
		cudaStreamSynchronize(stream);
	}
	void FinalDataTransfer(cudaStream_t stream = cudaStreamHandler.copyStream) {
		cudaMemcpyAsync(cpuSH.p_rgbMap, gpuSH.p_rgbMap, SCREEN_PIXELNUMBER * BMP_IMAGE_BYTEPP, cudaMemcpyDeviceToHost, stream); //RgbMap
	}

	void Execute() {
		cpuSH.p_camInfo->Update(p_camera);
		
		if (!cudaStreamHandler.executionGraphCreated) {
			cudaStreamBeginCapture(cudaStreamHandler.executionStream, cudaStreamCaptureModeGlobal);
			ResetTriangleData<<<(cpuSH.m_triangles_number / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, cudaStreamHandler.executionStream>>>(gpuSH.p_triangles, cpuSH.m_triangles_number);
			VertexShader<<<(cpuSH.m_vectors_number / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, cudaStreamHandler.executionStream>>>(gpuSH.p_camInfo, gpuSH.p_absSpace_vertices, gpuSH.p_relToCam_vertices, cpuSH.m_vectors_number);
			LightRotation<<<(gpuSH.m_lights_number / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, cudaStreamHandler.executionStream>>>(gpuSH.p_camInfo, gpuSH.p_lights, gpuSH.m_lights_number);
			PrimitiveAssembler<<<(cpuSH.m_triangles_number / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, cudaStreamHandler.executionStream>>>(gpuSH.p_camInfo, gpuSH.p_trianglesId, gpuSH.p_triangles, gpuSH.p_absSpace_vertices, gpuSH.p_relToCam_vertices, gpuSH.p_uv_coordinates, cpuSH.m_triangles_number, gpuSH.p_fragmentMap);
			
			for (uint32 i_clcicle = 0; i_clcicle < ((cpuSH.m_triangles_number / MAX_THREAD_GPU) + 1); i_clcicle++) {
				Clipping<<<4, MAX_THREAD_GPU/4, SHARED_MEMORY, cudaStreamHandler.executionStream>>>(gpuSH.p_camInfo, gpuSH.p_triangles, gpuSH.p_clippedTriangleBuffer, gpuSH.p_fragmentMap, cpuSH.m_triangles_number, i_clcicle);
				Rasterizzation__triangleDivision<<<GPU_TRIANGLEBUFFER_LENGHT, MAX_THREAD_GPU, SHARED_MEMORY, cudaStreamHandler.executionStream>>>(gpuSH.p_clippedTriangleBuffer, gpuSH.p_subdivRstInfo, gpuSH.p_fragmentMap, cpuSH.m_triangles_number * GPU_TRIANGLEBUFFER_LENGHT);
				Rasterizzation__subdividedSquareRasterizzation<<<GPU_TRIANGLEBUFFER_LENGHT * GPU_RST_NUMOF_THREADS_PER_TRI, MAX_THREAD_GPU, SHARED_MEMORY, cudaStreamHandler.executionStream>>>(gpuSH.p_subdivRstInfo, gpuSH.p_fragmentMap, MAX_THREAD_GPU * GPU_TRIANGLEBUFFER_LENGHT * GPU_RST_NUMOF_THREADS_PER_TRI);
			}
		
			FragmentShader<<<(NUMOF_THREAD_FOR_SCREEN_COMPUTATION / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, cudaStreamHandler.executionStream>>>(gpuSH.p_fragmentMap, gpuSH.p_rgbHDRMap, gpuSH.p_materials, gpuSH.p_lights, gpuSH.m_lights_number, SCREEN_PIXELNUMBER);
			if(gpuSH.bloomEnabled) bloomHandler.Bloom(gpuSH.p_rgbHDRMap, cudaStreamHandler.executionStream);

			ToneMapping<<<(NUMOF_THREAD_FOR_SCREEN_COMPUTATION / MAX_THREAD_GPU) + 1, MAX_THREAD_GPU, SHARED_MEMORY, cudaStreamHandler.executionStream>>>(gpuSH.p_rgbHDRMap, gpuSH.p_rgbMap, gamma, SCREEN_PIXELNUMBER);
			
			cudaStreamEndCapture(cudaStreamHandler.executionStream, &cudaStreamHandler.executionGraph);
			cudaGraphInstantiate(&cudaStreamHandler.executionGraphInstance, cudaStreamHandler.executionGraph, 0, 0, 0);
			cudaStreamHandler.executionGraphCreated = true;
		}
		else {
			cudaGraphLaunch(cudaStreamHandler.executionGraphInstance, cudaStreamHandler.executionStream);
		}
	}
};
GPU_pipelineManager pipelineManager;