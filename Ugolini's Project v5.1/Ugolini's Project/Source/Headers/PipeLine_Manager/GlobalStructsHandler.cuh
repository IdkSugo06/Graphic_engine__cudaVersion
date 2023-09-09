#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Singletons/CrashHandler.hpp"
#include "../GraphicStructs/Fragment.hpp"
#include "../GraphicStructs/Triangle.hpp"
#include "../GraphicStructs/Lights.hpp"

//Kernel options
#define MAX_THREAD_GPU 1024
#define SHARED_MEMORY 0

//triangle BL options
#define GPU_TRIANGLEBUFFER_LENGHT 13
#define GPU_RST_NUMOF_THREADS_PER_TRI (MAX_THREAD_GPU / GPU_RST_NUMOF_TRI_PER_MAXTHREADS)

//Upload file constants
#define FILE_CHAR_BUFFER_LENGHT 100

//Cuda streams
struct CudaStreamHandler {
	cudaStream_t mainStream = 0;
	cudaStream_t copyStream, executionStream;

	bool executionGraphCreated = false;
	cudaGraph_t executionGraph{0}; cudaGraphExec_t executionGraphInstance{0};

	CudaStreamHandler() {
		if (cudaStreamCreate(&copyStream) != cudaSuccess) {
			crashHandler.crashCode = CH_CUDA_STREAM_CREATION_ERROR;
		}
		if (cudaStreamCreate(&executionStream) != cudaSuccess) {
			crashHandler.crashCode = CH_CUDA_STREAM_CREATION_ERROR;
		}
	}
	~CudaStreamHandler() {
		cudaStreamDestroy(mainStream);
		cudaStreamDestroy(copyStream);
		cudaStreamDestroy(executionStream);
	}
};
CudaStreamHandler cudaStreamHandler;


//Malloc options: starting buffer lenghts
#define MYMAOPT_STARTING_VERTEX_BL 50000
#define MYMAOPT_STARTING_TRIANGLE_BL 50000
#define MYMAOPT_STARTING_MATERIAL_BL 20
#define MYMAOPT_STARTING_LIGHT_BL 20

//Resizing functions
void ResizeVertexArray(uint32 lenght);
void ResizeUVCoordinatesArray(uint32 lenght);
void ResizeTriangleArray(uint32 lenght);
void ResizeMaterialArray(uint32 lenght);
void ResizeLightArray(uint32 lenght);

//CPU struct handler
struct CPU_SH {

#pragma region CPU MEMBERS
	//General info
	GPU_camInfo* p_camInfo{ nullptr };

	//Material & light
	Material* p_tempMaterial{nullptr};		//Used as temp variable to transfer data on gpu, !!has to be allocated with cudaMallocHost
	Light* p_tempLight{ nullptr };

	//Vertices 
	Vector3* p_vertexArray{nullptr};
	Vector3* p_uvCoordinatesArray{nullptr};

	//Triangles 
	TriangleId* p_triangleIdArray{nullptr};

	//Screen maps (reset purposes)
	RgbVector* p_nullRgbHDRMap{ nullptr };	//Used to reset the GPU p_rgbHDRMap
	Fragment* p_nullFragmentMap{nullptr};	//Used to reset the GPU fragmentMap

	//Screen maps
	uint8* p_rgbMap{nullptr};

	//Buffers
	uint32 m_vectorsBL = MYMAOPT_STARTING_VERTEX_BL, m_uv_coordinatesBL = MYMAOPT_STARTING_VERTEX_BL, m_triangleIdsBL = MYMAOPT_STARTING_TRIANGLE_BL;	//Buffer Lenght
	uint32 m_vectors_number = 0, m_uv_coordinates_number = 0, m_triangles_number = 0;
#pragma endregion CPU MEMBERS

	//Constructor
	CPU_SH() {
		{
			cudaError_t vertexAllocation = cudaMallocHost((void**)&p_vertexArray, m_vectorsBL * sizeof(Vector3));
			cudaError_t uvCoordinatesAllocation = cudaMallocHost((void**)&p_uvCoordinatesArray, m_uv_coordinatesBL * sizeof(Vector3));
			cudaError_t triangleAllocation = cudaMallocHost((void**)&p_triangleIdArray, m_triangleIdsBL * sizeof(TriangleId));
			cudaError_t camInfoAllocation = cudaMallocHost((void**)&p_camInfo, sizeof(GPU_camInfo));
			*p_camInfo = GPU_camInfo(p_camera);
			if ((vertexAllocation != cudaSuccess) || (uvCoordinatesAllocation != cudaSuccess) || (triangleAllocation != cudaSuccess) || (camInfoAllocation != cudaSuccess)) {
				crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
		}{
			cudaError_t rgbMapAllocation = cudaMallocHost((void**)&p_rgbMap, SCREEN_PIXELNUMBER * BMP_IMAGE_BYTEPP);
			cudaError_t nullRgbHDRMapAllocation = cudaMallocHost((void**)&p_nullRgbHDRMap, SCREEN_PIXELNUMBER * sizeof(RgbVector));
			cudaError_t nullFragmentMapAllocation = cudaMallocHost((void**)&p_nullFragmentMap, SCREEN_PIXELNUMBER * sizeof(Fragment));
			if ((nullFragmentMapAllocation != cudaSuccess) || (rgbMapAllocation != cudaSuccess) || (nullRgbHDRMapAllocation != cudaSuccess)) {
				crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
		} {
			cudaError_t tempMaterialAllocation = cudaMallocHost((void**)&p_tempMaterial, sizeof(Material));
			cudaError_t tempMaterialTextureAllocation = cudaMallocHost((void**)&(p_tempMaterial->m_textureMap.p_memory), TEXTURE_MAX_HEIGHT * TEXTURE_MAX_WIDTH * sizeof(RgbVector));
			cudaError_t tempMaterialNormalMapAllocation = cudaMallocHost((void**)&(p_tempMaterial->m_normalMap.p_memory), TEXTURE_MAX_HEIGHT * TEXTURE_MAX_WIDTH * sizeof(Vector3));
			if ((tempMaterialAllocation != cudaSuccess) || (tempMaterialTextureAllocation != cudaSuccess) || (tempMaterialNormalMapAllocation != cudaSuccess)) {
				crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
		} {
			cudaError_t tempLightAllocation = cudaMallocHost((void**)&p_tempLight, sizeof(Light));
			cudaError_t tempShaderMapAllocation = cudaMallocHost((void**)&(p_tempLight->p_shaderMap), LIGHT_SHADERMAP_WIDTH * LIGHT_SHADERMAP_HEIGHT * sizeof(float));
			if ((tempLightAllocation != cudaSuccess) || (tempShaderMapAllocation != cudaSuccess)) {
				crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
		}
	}
	~CPU_SH() {
		cudaFree(p_tempMaterial);
		cudaFree(p_tempMaterial->m_textureMap.p_memory);
		cudaFree(p_tempMaterial->m_normalMap.p_memory);

		cudaFree(p_tempLight);
		cudaFree(p_tempLight->p_shaderMap);

		cudaFree(p_vertexArray);
		cudaFree(p_uvCoordinatesArray);
		cudaFree(p_triangleIdArray);

		cudaFree(p_rgbMap);
		cudaFree(p_nullRgbHDRMap);
		cudaFree(p_nullFragmentMap);

		cudaFree(p_camInfo);
	}

	//ADD & REMOVE POINTS
	uint32 AddPoint(Vector3 vect) {
		if (m_vectors_number >= m_vectorsBL) {
			ResizeVertexArray(m_vectorsBL + MYMAOPT_STARTING_VERTEX_BL);
		}
		p_vertexArray[m_vectors_number] = vect;
		m_vectors_number++;
		return (m_vectors_number - 1);
	}
	void RemovePoint(uint32 pos) {
		m_vectors_number--; //I do sub 1 before the next operation cause i need the last element position, which is the lenght - 1
		p_vertexArray[pos] = p_vertexArray[m_vectors_number];
	}

	void AddUVcoordinate(Vector2 uv) { //Uv coordinates has to be between 0 and 1
		if (m_uv_coordinates_number >= m_uv_coordinatesBL) {
			ResizeUVCoordinatesArray(m_uv_coordinatesBL + MYMAOPT_STARTING_VERTEX_BL);
		}
		uv.x = myBetween(uv.x, 0, 1);
		uv.y = myBetween(uv.y, 0, 1);
		p_uvCoordinatesArray[m_uv_coordinates_number] = Vector3(uv.x, uv.y, 0);
		m_uv_coordinates_number++;
	}
	void RemoveUVcoordinate(uint32 pos) {
		m_uv_coordinates_number--;
		p_uvCoordinatesArray[pos] = p_uvCoordinatesArray[m_uv_coordinates_number];
	}

	void AddPoint(Vector3 vect, Vector2 uv) {
		AddPoint(vect); AddUVcoordinate(uv);
	}
	void RemovePoint(uint32 vect, uint32 uv) {
		RemovePoint(vect); RemoveUVcoordinate(uv);
	}

	void AddTriangle(TriangleId tId) {
		if (m_triangles_number >= m_triangleIdsBL) {
			ResizeTriangleArray(m_triangleIdsBL + MYMAOPT_STARTING_TRIANGLE_BL);
		}
		p_triangleIdArray[m_triangles_number] = tId;
		m_triangles_number++;
	}
	void RemoveTriangle(uint32 pos) {
		m_triangles_number--;
		p_triangleIdArray[pos] = p_triangleIdArray[m_triangles_number];
	}

	void LoadInTempMaterial(Material& material) {
		*p_tempMaterial = material;
		if (!p_tempMaterial->m_textureMap.m_isValid) { 
			p_tempMaterial->m_textureMap.p_memory = nullptr; 
		}	
		if (!p_tempMaterial->m_normalMap.m_isValid) {
			p_tempMaterial->m_normalMap.p_memory = nullptr;
		}
	}
	void LoadInTempLight(Light& light) {
		*p_tempLight = light;
		if (!p_tempLight->m_usingShaders) {
			p_tempLight->p_shaderMap = nullptr;
		}
	}
};
CPU_SH cpuSH;

//GPU struct handler
struct GPU_SH {


	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ Members
#pragma region GPU MEMBERS
	bool bloomEnabled = false;

	//General info
	GPU_camInfo* p_camInfo{ nullptr };

	//Materials
	Material* p_materials{ nullptr };	
	void** pp_txtMemories{ nullptr };	//Store the p_ (to VRAM mem) in order to free the txts (CPU ALLOCATED)
	void** pp_normalMapsMemories{ nullptr };

	//Lights
	Light* p_lights{ nullptr }; //yet to be implemented
	void** pp_shaderMapMemories{ nullptr };	//Store the p_ (to VRAM mem) in order to free the txts (CPU ALLOCATED)

	//Vertices
	Vector3* p_absSpace_vertices{ nullptr };
	Vector3* p_relToCam_vertices{ nullptr };
	Vector3* p_uv_coordinates{ nullptr };

	//Triangles
	TriangleId* p_trianglesId{ nullptr };
	GPU_Triangle* p_triangles{ nullptr };
	GPU_Triangle* p_clippedTriangleBuffer{ nullptr };	//This variable will store a max of MAX_THREAD_GPU * GPU_TRIANGLEBUFFER_LENGHT triangles, in order to temporarly store the clipped triangles
	GPU_subdividedRasterizzationInfo* p_subdivRstInfo{nullptr};

	//Screen maps
	Fragment* p_fragmentMap{ nullptr };				//zBuffer contained in the fragmentMap (m_relToCam_point.z c:)
	RgbVector* p_rgbHDRMap{ nullptr };
	uint8* p_rgbMap{ nullptr };				//No padding, neither alfa


	//Buffers
	uint16 m_materialsBL = MYMAOPT_STARTING_MATERIAL_BL, m_lightsBL = MYMAOPT_STARTING_LIGHT_BL;
	uint16 m_materials_number = 0, m_lights_number = 0;
#pragma endregion GPU MEMBERS



	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ Constructors
	GPU_SH() {
		{
			cudaError_t materialsAllocation = cudaMalloc((void**)&p_materials, m_materialsBL * sizeof(Material));
			cudaError_t lightsAllocation = cudaMalloc((void**)&p_lights, m_lightsBL * sizeof(Light));
			if ((materialsAllocation != cudaSuccess) || (lightsAllocation != cudaSuccess)) {
				crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
			pp_shaderMapMemories = (void**)malloc(m_lightsBL * sizeof(void*));
			pp_txtMemories = (void**)malloc(m_materialsBL * sizeof(void*));
			pp_normalMapsMemories = (void**)malloc(m_materialsBL * sizeof(void*));
			if ((!pp_txtMemories) || (!pp_normalMapsMemories) || (!pp_shaderMapMemories)) {
				crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
		}{
			cudaError_t AbsSpaceVerticesAllocation = cudaMalloc((void**)&p_absSpace_vertices, cpuSH.m_vectorsBL * sizeof(Vector3));
			cudaError_t RelToCamVerticesAllocation = cudaMalloc((void**)&p_relToCam_vertices, cpuSH.m_vectorsBL * sizeof(Vector3));
			cudaError_t UvCoordinatesAllocation = cudaMalloc((void**)&p_uv_coordinates, cpuSH.m_uv_coordinatesBL * sizeof(Vector3));
			if ((AbsSpaceVerticesAllocation != cudaSuccess) || (RelToCamVerticesAllocation != cudaSuccess) || (UvCoordinatesAllocation != cudaSuccess)) {
				crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
		}{
			cudaError_t triangleIDAllocation = cudaMalloc((void**)&p_trianglesId, cpuSH.m_triangleIdsBL * sizeof(TriangleId));
			cudaError_t triangleAllocation = cudaMalloc((void**)&p_triangles, 200000 * sizeof(GPU_Triangle));
			cudaError_t triangleBufferAllocation = cudaMalloc((void**)&p_clippedTriangleBuffer, MAX_THREAD_GPU * GPU_TRIANGLEBUFFER_LENGHT * sizeof(GPU_Triangle));
			cudaError_t subDivRstInfoAllocation = cudaMalloc((void**)&p_subdivRstInfo, MAX_THREAD_GPU * GPU_TRIANGLEBUFFER_LENGHT * sizeof(GPU_subdividedRasterizzationInfo));
			if ((triangleAllocation != cudaSuccess) || (triangleBufferAllocation != cudaSuccess) || (triangleIDAllocation != cudaSuccess) || (subDivRstInfoAllocation != cudaSuccess)) {
				crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
		}{
			cudaError_t fragmentMapAllocation = cudaMalloc((void**)&p_fragmentMap, SCREEN_PIXELNUMBER * sizeof(Fragment));
			cudaError_t rgbHDRMapAllocation = cudaMalloc((void**)&p_rgbHDRMap, SCREEN_PIXELNUMBER * sizeof(RgbVector));
			cudaError_t rgbMapAllocation = cudaMalloc((void**)&p_rgbMap, SCREEN_PIXELNUMBER * BMP_IMAGE_BYTEPP);
			if ((fragmentMapAllocation != cudaSuccess) || (rgbMapAllocation != cudaSuccess) || (rgbHDRMapAllocation != cudaSuccess)) {
				crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
		} {
			cudaError_t camInfoAllocation = cudaMalloc((void**)&p_camInfo, sizeof(GPU_camInfo));
			if ((camInfoAllocation != cudaSuccess)) {
				crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR; std::cout << "[ALLOCATION ERROR]: \n"; return;
			}
		}

		DefaultMaterialSetUp();
		DefaultLightSetUp();
	}
	~GPU_SH() {
		for (uint16 i = 0; i < m_materials_number; i++) {
			if (pp_txtMemories[i] != nullptr) cudaFree(pp_txtMemories[i]);
		}
		for (uint16 i = 0; i < m_materials_number; i++) {
			if (pp_normalMapsMemories[i] != nullptr) cudaFree(pp_normalMapsMemories[i]);
		}
		cudaFree(p_materials);

		for (uint16 i = 0; i < m_lights_number; i++) {
			if (pp_shaderMapMemories[i] != nullptr) cudaFree(pp_shaderMapMemories[i]);
		}
		cudaFree(p_lights);

		cudaFree(p_absSpace_vertices);
		cudaFree(p_relToCam_vertices);

		cudaFree(p_trianglesId);
		cudaFree(p_triangles);
		cudaFree(p_clippedTriangleBuffer);

		cudaFree(p_fragmentMap);
		cudaFree(p_rgbHDRMap);
		cudaFree(p_rgbMap);

		cudaFree(p_camInfo);
	}


	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ Temp material functions
	void DefaultMaterialSetUp() {
		RgbVectorMap* txt = &(cpuSH.p_tempMaterial->m_textureMap);
		txt->m_width = TEXTURE_MAX_WIDTH; txt->m_height = TEXTURE_MAX_HEIGHT;
		txt->m_sizeof_line = TEXTURE_MAX_WIDTH * txt->m_sizeof_element; txt->m_sizeof_column = TEXTURE_MAX_HEIGHT * txt->m_sizeof_element;
		txt->m_sizeof_memory = (uint32)txt->m_sizeof_element * TEXTURE_MAX_WIDTH * TEXTURE_MAX_HEIGHT;
		txt->m_isValid = true;

		uint16 step = 100;
		for (uint16 y = 0; y < txt->m_height; y++) {
			for (uint16 x = 0; x < txt->m_width; x++) {
				bool b_x = (x % step) < step * 2;
				bool b_y = (y % step) < step * 2;
				if (b_x ^ b_y) {
					txt->p_memory[x + y * txt->m_width].red = 255;
					txt->p_memory[x + y * txt->m_width].green = 0;
					txt->p_memory[x + y * txt->m_width].blue = 255;
					txt->p_memory[x + y * txt->m_width].alfa = 0;
				}
			}
		}
		CopyTempMaterial();
	}
	void CopyTempMaterial(cudaStream_t stream = cudaStreamHandler.mainStream, bool wait = false) { //This function will copy whatever there is on the CPU tempMaterial and will put in the gpu
		if (m_materials_number >= m_materialsBL) {
			ResizeMaterialArray(m_materialsBL + MYMAOPT_STARTING_MATERIAL_BL);
		}
		RgbVectorMap* txtMap = &(cpuSH.p_tempMaterial->m_textureMap);
		NormalMap* normalMap = &(cpuSH.p_tempMaterial->m_normalMap);

		//TEXTURE MAP ALLOCATION
		void* p_txtCPU_memory = nullptr, *p_txtGPU_memory = nullptr;
		if (txtMap->m_isValid) {
			//Save a copy of the CPU memory pointer
			p_txtCPU_memory = txtMap->p_memory;

			//The space for the new material has already been allocated, we only have to allocate the memory of the texture and copy whats inside
			cudaError_t tempTextureMemoryAllocation = cudaMalloc((void**)&(p_txtGPU_memory), txtMap->m_sizeof_memory);
			if ((tempTextureMemoryAllocation != cudaSuccess)) {
				crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR; return;
			}

			//We can only copy exactly whats on CPU memory on GPU memory, so it'll overwrite the tempMaterial RAM pointer to the VRAM pointer
			txtMap->p_memory = (RgbVector*)p_txtGPU_memory;
			txtMap->m_isInCudaMem = true;
		}
		else {
			//Ill copy in the cpu all the default options, then cpy them in the gpu (isValid in gpu will remain false)
			txtMap->m_width = TEXTURE_MAX_WIDTH; txtMap->m_height = TEXTURE_MAX_HEIGHT;
			txtMap->m_sizeof_line = TEXTURE_MAX_WIDTH * txtMap->m_sizeof_element; txtMap->m_sizeof_column = TEXTURE_MAX_HEIGHT * txtMap->m_sizeof_element;
			txtMap->m_sizeof_memory = (uint32)txtMap->m_sizeof_element * TEXTURE_MAX_WIDTH * TEXTURE_MAX_HEIGHT;
			txtMap->m_mapLenght = TEXTURE_MAX_WIDTH * TEXTURE_MAX_HEIGHT;

			//txtMap->m_isValid = true;
			txtMap->p_memory = (RgbVector*)(pp_txtMemories[DEFAULT_MATERIAL_ID]);		//The first will be the default pointer
			txtMap->m_isInCudaMem = true;
		}

		//NORMAL MAP ALLOCATION
		void* p_normalMapCPU_memory = nullptr, *p_normalMapGPU_memory = nullptr;
		if (normalMap->m_isValid) {
			//Save a copy of the CPU memory pointer
			p_normalMapCPU_memory = normalMap->p_memory;

			//The space for the new material has already been allocated, we only have to allocate the memory of the texture and copy whats inside
			cudaError_t tempNormalMapMemoryAllocation = cudaMalloc((void**)&(p_normalMapGPU_memory), normalMap->m_sizeof_memory);
			if ((tempNormalMapMemoryAllocation != cudaSuccess)) {
				crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR; return;
			}

			//We can only copy exactly whats on CPU memory on GPU memory, so it'll overwrite the tempMaterial RAM pointer to the VRAM pointer
			normalMap->p_memory = (Vector3*)p_normalMapGPU_memory;
			normalMap->m_isInCudaMem = true;
		}

		//MATERIAL COPYING
		cudaMemcpyAsync(&(p_materials[m_materials_number]), cpuSH.p_tempMaterial, sizeof(Material), cudaMemcpyHostToDevice, stream);		
		cudaDeviceSynchronize();

		//RESET TEMP MATERIAL
		if (txtMap->m_isValid) {
			//Copy the texture map
			cudaMemcpyAsync(p_txtGPU_memory, p_txtCPU_memory, txtMap->m_sizeof_memory, cudaMemcpyHostToDevice, stream);

			//Load in the tempMaterial the pointer that there were before
			txtMap->p_memory = (RgbVector*)p_txtCPU_memory;
			txtMap->m_isInCudaMem = false;

			//It'll save the pointer in the VRAM so that i can free it when the program finishes
			pp_txtMemories[m_materials_number] = p_txtGPU_memory;
		}
		else {
			cudaMemcpyAsync(&(p_materials[m_materials_number]), cpuSH.p_tempMaterial, sizeof(Material), cudaMemcpyHostToDevice, stream);
			txtMap->p_memory = nullptr;
			txtMap->m_isInCudaMem = false;
			txtMap->m_isValid = false;

			pp_txtMemories[m_materials_number] = nullptr; //exception will be dealt in the gpu deconstructor
		}
		if (normalMap->m_isValid) {
			//Copy the texture map
			cudaMemcpyAsync(p_normalMapGPU_memory, p_normalMapCPU_memory, normalMap->m_sizeof_memory, cudaMemcpyHostToDevice, stream);

			//Load in the tempMaterial the pointer that there were before
			normalMap->p_memory = (Vector3*)p_normalMapCPU_memory;
			normalMap->m_isInCudaMem = false;

			//It'll save the pointer in the VRAM so that i can free it when the program finishes
			pp_normalMapsMemories[m_materials_number] = p_normalMapGPU_memory;
		}
		else {
			pp_normalMapsMemories[m_materials_number] = nullptr; //exception will be dealt in the gpu deconstructor
		}

		m_materials_number++;
		if (wait) cudaDeviceSynchronize();
	}

	void CreateMaterial(Material& mat) { //Upload one material on the gpu
		cpuSH.LoadInTempMaterial(mat);
		CopyTempMaterial();
	}
	void CreateMaterial(RgbVector kd, RgbVector ks, float specularExponent, RgbVector ke, const char* txtMap_filePath = nullptr, const char* normalMap_filePath = nullptr) { //Create and upload one material on the gpu
		cpuSH.LoadInTempMaterial(Material(kd, ks, specularExponent, ke, txtMap_filePath, normalMap_filePath));
		CopyTempMaterial();
	}
	void CreateMaterial(const char* txtMap_filePath = nullptr, const char* normalMap_filePath = nullptr) { //Create and upload one material on the gpu
		CreateMaterial(RgbVector::Red, RgbVector::Red, 1, RgbVector::Black, txtMap_filePath, normalMap_filePath);
	}



	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ Temp light functions
	void DefaultLightSetUp() {
		CreateLight__directionalLight(Vector3(DEFAULT_LIGHT_DIRECTION), RgbVector(DEFAULT_LIGHT_RGBVALUES));
	}
	void CopyTempLight(cudaStream_t stream = cudaStreamHandler.mainStream, bool wait = false) { //This function will copy whatever there is on the CPU tempMaterial and will put in the gpu
		if (m_lights_number >= m_lightsBL) {
			ResizeLightArray(m_lightsBL + MYMAOPT_STARTING_LIGHT_BL);
		}
		Light* light = cpuSH.p_tempLight;

		//ShaderMap allocation
		//void* p_shaderMapCPU_memory = nullptr, * p_shaderMapGPU_memory = nullptr;
		if (light->m_usingShaders) {
			//yet to be implemented
		}
	
		//Light copying
		cudaMemcpyAsync(&(p_lights[m_lights_number]), light, sizeof(Light), cudaMemcpyHostToDevice, stream);
		cudaDeviceSynchronize();

		//Reset light light
		if (light->m_usingShaders) {
			//yet to be implemented
		}

		m_lights_number++;
		if (wait) cudaDeviceSynchronize();
	}
	void CreateLight__directionalLight(Vector3 direction, RgbVector color) { //Create and upload one material on the gpu
		cpuSH.p_tempLight->DirectionalLight(direction, color);
		CopyTempLight();
	}
	void CreateLight__pointLight(Vector3 position, RgbVector color, float kconst = DEFAULT_POINTLIGHT_CONSTATT, float klin = DEFAULT_POINTLIGHT_LINATT, float kquad = DEFAULT_POINTLIGHT_QUADATT) { //Create and upload one material on the gpu
		cpuSH.p_tempLight->PointLight(position, color, kconst, klin, kquad);
		CopyTempLight();
	}
	void CreateLight__spotLight(Vector3 position, Vector3 direction, RgbVector color, float cutOff = DEFAULT_SPOTLIGHT_CUTOFF_VALUE, float cutOff_degradation = DEFAULT_SPOTLIGHT_DEGRADATION_VALUE, float kconst = DEFAULT_POINTLIGHT_CONSTATT, float klin = DEFAULT_POINTLIGHT_LINATT, float kquad = DEFAULT_POINTLIGHT_QUADATT) { //Create and upload one material on the gpu
		cpuSH.p_tempLight->SpotLight(position, direction, color, cutOff, cutOff_degradation, kconst, klin, kquad);
		CopyTempLight();
	}
};
GPU_SH gpuSH;


//Resizing functions
void ResizeVertexArray(uint32 lenght) {
	//Function called
	if (cpuSH.m_vectorsBL >= lenght) return; //Buffer already long enough

	// --- --- --- CPU array resizing --- --- ---
	//Useful variables
	uint32 previous_lenght = cpuSH.m_vectorsBL + 1;
	void* previousCPU_array = cpuSH.p_vertexArray;

	//Resizing
	cpuSH.m_vectorsBL = lenght;
	cudaError_t vertexAllocation = cudaMallocHost((void**)&cpuSH.p_vertexArray, lenght * sizeof(Vector3));
	if ((vertexAllocation != cudaSuccess)) {
		crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR;
		std::cout << "[ERRORE]: Ridimensionamento vertex buffer (CPU) fallito, attenderà 5 sec\n";
		Sleep(5000);
		return;
	}
	//Copying
	memcpy(cpuSH.p_vertexArray, previousCPU_array, previous_lenght * sizeof(Vector3));
	//Freeing
	cudaFree(previousCPU_array);

	// --- --- --- GPU array resizing --- --- ---
	//Useful variables
	void* previousGPU_array = gpuSH.p_absSpace_vertices;
	//Resizing
	vertexAllocation = cudaMalloc((void**)&gpuSH.p_absSpace_vertices, lenght * sizeof(Vector3));
	if ((vertexAllocation != cudaSuccess)) {
		crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR;
		std::cout << "[ERRORE]: Ridimensionamento vertex buffer (GPU) fallito, attenderà 5 sec\n";
		Sleep(5000);
		return;
	}
	//Copying
	cudaMemcpy(gpuSH.p_absSpace_vertices, previousGPU_array, previous_lenght * sizeof(Vector3), cudaMemcpyDeviceToDevice);
	//Freeing
	cudaFree(previousGPU_array);

	// --- --- --- GPU array resizing --- --- ---
	//Useful variables
	previousGPU_array = gpuSH.p_relToCam_vertices;
	//Resizing
	vertexAllocation = cudaMalloc((void**)&gpuSH.p_relToCam_vertices, lenght * sizeof(Vector3));
	if ((vertexAllocation != cudaSuccess)) {
		crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR;
		std::cout << "[ERRORE]: Ridimensionamento vertex buffer (GPU) fallito, attenderà 5 sec\n";
		Sleep(5000);
		return;
	}
	//Copying
	cudaMemcpy(gpuSH.p_relToCam_vertices, previousGPU_array, previous_lenght * sizeof(Vector3), cudaMemcpyDeviceToDevice);
	//Freeing
	cudaFree(previousGPU_array);
	cudaDeviceSynchronize();
}
void ResizeUVCoordinatesArray(uint32 lenght) {
	if (cpuSH.m_uv_coordinatesBL >= lenght) return; //Buffer already long enough

	// --- --- --- CPU array resizing --- --- ---
	//Useful variables
	uint32 previous_lenght = cpuSH.m_uv_coordinatesBL +1;
	void* previousCPU_array = cpuSH.p_uvCoordinatesArray;
	//Resizing
	cpuSH.m_uv_coordinatesBL = lenght;
	cudaError_t vertexAllocation = cudaMallocHost((void**)&cpuSH.p_uvCoordinatesArray, lenght * sizeof(Vector3));
	if ((vertexAllocation != cudaSuccess)) {
		crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR;
		std::cout << "[ERRORE]: Ridimensionamento vertex buffer (CPU) fallito, attenderà 5 sec\n";
		Sleep(5000);
		return;
	}
	//Copying
	memcpy(cpuSH.p_uvCoordinatesArray, previousCPU_array, previous_lenght * sizeof(Vector3));
	//Freeing
	cudaFree(previousCPU_array);

	// --- --- --- GPU array resizing --- --- ---
	//Useful variables
	void* previousGPU_array = gpuSH.p_uv_coordinates;
	//Resizing
	vertexAllocation = cudaMalloc((void**)&gpuSH.p_uv_coordinates, lenght * sizeof(Vector3));
	if ((vertexAllocation != cudaSuccess)) {
		crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR;
		std::cout << "[ERRORE]: Ridimensionamento vertex buffer (GPU) fallito, attenderà 5 sec\n";
		Sleep(5000);
		return;
	}
	//Copying
	cudaMemcpyAsync(gpuSH.p_uv_coordinates, previousGPU_array, previous_lenght * sizeof(Vector3), cudaMemcpyDeviceToDevice, cudaStreamHandler.copyStream);
	//Freeing
	cudaFree(previousGPU_array);
	cudaDeviceSynchronize();
}
void ResizeTriangleArray(uint32 lenght) {
	if (cpuSH.m_triangleIdsBL >= lenght) return; //Buffer already long enough

	// --- --- --- CPU array resizing --- --- ---
	//Useful variables
	uint32 previous_lenght = cpuSH.m_triangleIdsBL + 1;
	void* previousCPU_array = cpuSH.p_triangleIdArray;
	//Resizing
	cpuSH.m_triangleIdsBL = lenght;
	void* newPtr = nullptr;
	cudaError_t triangleAllocation = cudaMallocHost((void**)&newPtr, lenght * sizeof(TriangleId));
	if ((triangleAllocation != cudaSuccess)) {
		crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR;
		std::cout << "[ERRORE]: Ridimensionamento triangleId buffer (CPU) fallito, attenderà 5 sec\n";
		Sleep(5000);
		return;
	}
	cpuSH.p_triangleIdArray = (TriangleId*)newPtr;
	//Copying
	memcpy(cpuSH.p_triangleIdArray, previousCPU_array, previous_lenght * sizeof(TriangleId));
	//Freeing
	cudaFree(previousCPU_array);

	// --- --- --- GPU array resizing --- --- ---
	//Useful variables
	void* previousGPU_array = gpuSH.p_triangles;
	//Resizing
	newPtr = nullptr;
	triangleAllocation = cudaMalloc((void**)&newPtr, lenght * sizeof(GPU_Triangle));
	if ((triangleAllocation != cudaSuccess)) {
		crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR;
		std::cout << "[ERRORE]: Ridimensionamento triangle buffer (GPU) fallito, attenderà 5 sec\n";
		Sleep(5000);
		return;
	}
	gpuSH.p_triangles = (GPU_Triangle*)newPtr;
	//Copying
	cudaMemcpy(gpuSH.p_triangles, previousGPU_array, previous_lenght * sizeof(GPU_Triangle), cudaMemcpyDeviceToDevice);
	//Freeing
	cudaFree(previousGPU_array);

	// --- --- --- GPU array resizing --- --- ---
	//Useful variables
	previousGPU_array = gpuSH.p_trianglesId;
	//Resizing
	newPtr = nullptr;
	triangleAllocation = cudaMalloc((void**)&newPtr, lenght * sizeof(TriangleId));
	if ((triangleAllocation != cudaSuccess)) {
		crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR;
		std::cout << "[ERRORE]: Ridimensionamento triangle buffer (GPU) fallito, attenderà 5 sec\n";
		Sleep(5000);
		return;
	}
	gpuSH.p_trianglesId = (TriangleId*)newPtr;
	//Copying
	cudaMemcpy(gpuSH.p_trianglesId, previousGPU_array, previous_lenght * sizeof(TriangleId), cudaMemcpyDeviceToDevice);
	//Freeing
	cudaFree(previousGPU_array);

	cudaDeviceSynchronize();
}
void ResizeMaterialArray(uint32 lenght) {
	if (gpuSH.m_materialsBL >= lenght) return; //Buffer already long enough
	
	// --- --- --- CPU pp_txtMap array resizing --- --- ---
	//Useful variables
	void*  previousGPU_array = gpuSH.pp_txtMemories;
	uint16 previous_lenght = gpuSH.m_materialsBL;
	//Resizing
	cudaError_t materialAllocation = cudaMallocHost((void**)&gpuSH.pp_txtMemories, lenght * sizeof(void*));
	if ((materialAllocation != cudaSuccess)) {
		crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR;
		std::cout << "[ERRORE]: Ridimensionamento textureMapMemories buffer (GPU) fallito, attenderà 5 sec\n";
		Sleep(5000);
		return;
	}
	//Copying
	memcpy(gpuSH.pp_txtMemories, previousGPU_array, previous_lenght * sizeof(void*));
	//Freeing
	cudaFree(previousGPU_array);

	// --- --- --- CPU pp_normalMap array resizing --- --- ---
	//Useful variables
	previousGPU_array = gpuSH.pp_normalMapsMemories;
	//Resizing
	materialAllocation = cudaMallocHost((void**)&gpuSH.pp_normalMapsMemories, lenght * sizeof(void*));
	if ((materialAllocation != cudaSuccess)) {
		crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR;
		std::cout << "[ERRORE]: Ridimensionamento normalMapMemories buffer (GPU) fallito, attenderà 5 sec\n";
		Sleep(5000);
		return;
	}
	//Copying
	memcpy(gpuSH.pp_normalMapsMemories, previousGPU_array, previous_lenght * sizeof(void*));
	//Freeing
	cudaFree(previousGPU_array);

	// --- --- --- GPU material array resizing --- --- ---
	//Useful variables
	previousGPU_array = gpuSH.p_materials;
	//Resizing
	materialAllocation = cudaMalloc((void**)&gpuSH.p_materials, lenght * sizeof(Material));
	if ((materialAllocation != cudaSuccess)) {
		crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR;
		std::cout << "[ERRORE]: Ridimensionamento material buffer (GPU) fallito, attenderà 5 sec\n";
		Sleep(5000);
		return;
	}
	//Copying
	cudaMemcpyAsync(gpuSH.p_materials, previousGPU_array, previous_lenght * sizeof(Material), cudaMemcpyDeviceToDevice, cudaStreamHandler.copyStream);
	//Freeing
	cudaFree(previousGPU_array);
	cudaDeviceSynchronize();
}
void ResizeLightArray(uint32 lenght) {
	if (gpuSH.m_lightsBL >= lenght) return; //Buffer already long enough

	// --- --- --- CPU pp_shaderMap array resizing --- --- ---
	//Useful variables
	void* previousGPU_array = gpuSH.pp_shaderMapMemories;
	uint16 previous_lenght = gpuSH.m_lightsBL;
	//Resizing
	cudaError_t lightsAllocation = cudaMallocHost((void**)&gpuSH.pp_shaderMapMemories, lenght * sizeof(void*));
	if ((lightsAllocation != cudaSuccess)) {
		crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR;
		std::cout << "[ERRORE]: Ridimensionamento shaderMapMemories buffer (GPU) fallito, attenderà 5 sec\n";
		Sleep(5000);
		return;
	}
	//Copying
	memcpy(gpuSH.pp_shaderMapMemories, previousGPU_array, previous_lenght * sizeof(void*));
	//Freeing
	cudaFree(previousGPU_array);

	// --- --- --- GPU material array resizing --- --- ---
	//Useful variables
	previousGPU_array = gpuSH.p_lights;
	//Resizing
	lightsAllocation = cudaMalloc((void**)&gpuSH.p_lights, lenght * sizeof(Light));
	if ((lightsAllocation != cudaSuccess)) {
		crashHandler.crashCode = CH_GPU_ALLOCATION_ERROR;
		std::cout << "[ERRORE]: Ridimensionamento lightsBuffer (GPU) fallito, attenderà 5 sec\n";
		Sleep(5000);
		return;
	}
	//Copying
	cudaMemcpyAsync(gpuSH.p_lights, previousGPU_array, previous_lenght * sizeof(Light), cudaMemcpyDeviceToDevice, cudaStreamHandler.copyStream);
	//Freeing
	cudaFree(previousGPU_array);
	cudaDeviceSynchronize();
}