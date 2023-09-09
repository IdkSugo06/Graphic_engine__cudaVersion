#pragma once
#include "ImagesStructs\NormalMap.hpp"
#include "..\Singletons\CrashHandler.hpp"

//Materials options
#define DEFAULT_MATERIAL_ID 0
#define MATERIAL_TAG_BUFFER 50 //has to be under 255 to work, (uint8 used)
#define TEXTURE_MAX_WIDTH 1280
#define TEXTURE_MAX_HEIGHT 720

#define MATERIAL_SPECULAREXPONENT_COEFFREDUCTION 10

#define MATERIAL_DEF_AMBIENT_COEFF RgbVector::GetGrayScale(0.01);
struct Material {

	static const uint16 m_defaultMaterialId;
	static RgbVector m_ambientCoeff;

	//--------------------------------------------------------------- Members
public:
	RgbVector m_diffuseCoeff, m_specularCoeff, m_emissiveCoeff; //(0->1, 0->1, 0->1)
	float m_specularExponent; //(1->1000)
	RgbVectorMap m_textureMap; NormalMap m_normalMap;

	//--------------------------------------------------------------- Constructors
public:
	Material(const char* file_path) {
		//m_diffuseCoeff = kd;  m_specularCoeff = ks; m_emissiveCoeff = ke; m_specularExponent = specularExponent;
		m_textureMap = RgbVectorMap(file_path);
		m_normalMap = NormalMap(file_path);
	}
	Material(RgbVector kd, RgbVector ks, float specularExponent, RgbVector ke, const char* txtMap_filePath = nullptr, const char* normalMap_filePath = nullptr, float normalMapScale = 1) {
		m_diffuseCoeff = kd;  m_specularCoeff = ks; m_emissiveCoeff = ke; m_specularExponent = specularExponent;
		if (txtMap_filePath != nullptr) m_textureMap = RgbVectorMap(txtMap_filePath);
		if (normalMap_filePath != nullptr) m_normalMap = NormalMap(normalMap_filePath, normalMapScale);
	}
	Material(const char* txtMap_filePath = nullptr, const char* normalMap_filePath = nullptr) : Material(RgbVector::Red, RgbVector::Red, 1, RgbVector::Black, txtMap_filePath, normalMap_filePath) {}

	void operator= (Material& material) {
		m_diffuseCoeff = material.m_diffuseCoeff; m_specularCoeff = material.m_specularCoeff; m_emissiveCoeff = material.m_emissiveCoeff; m_specularExponent = material.m_specularExponent;
		m_textureMap = material.m_textureMap; material.m_textureMap.m_isValid = false;
		m_normalMap = material.m_normalMap; material.m_normalMap.m_isValid = false;
	}

	static void UploadMtllib(const char* mtl_filePath); //Defined in the AdvancedFileFunction.hpp file

	__device__ static RgbVector gpu_GetAmbientCoefficent() {
		return MATERIAL_DEF_AMBIENT_COEFF;
	}
	static void PrintMTL(Material const& mtl) {
		std::cout << "Diffuse:\n\t";
		RgbVector::PrintRGB(mtl.m_diffuseCoeff);
		std::cout << "Specular:\n\t";
		RgbVector::PrintRGB(mtl.m_specularCoeff);
		std::cout << "Emissive:\n\t";
		RgbVector::PrintRGB(mtl.m_emissiveCoeff);
		std::cout << "Specular exponent:\n\t" << mtl.m_specularExponent << "\n";
		if (mtl.m_textureMap.m_isValid) std::cout << "Texture map available" << "\n";
		else std::cout << "Texture map unavailable" << "\n";
	}
};
const uint16 Material::m_defaultMaterialId = DEFAULT_MATERIAL_ID;
RgbVector Material::m_ambientCoeff = MATERIAL_DEF_AMBIENT_COEFF;


//MTL FILE 
struct Mtltag { //Remeber the names of the mtl during the read of the obj file
	char m_tag[MATERIAL_TAG_BUFFER];

	Mtltag(const char* _tag, uint8 tagLenght) {
		uint8 i = 0;
		while ((i < tagLenght) && (i < MATERIAL_TAG_BUFFER - 1)){
			m_tag[i] = _tag[i];
			i++;
		}
		m_tag[i] = '\0';
	}
	void operator= (Mtltag& _mtltag) {
		for (uint8 i = 0; i < MATERIAL_TAG_BUFFER; i++) {
			m_tag[i] = _mtltag.m_tag[i];
		}
	}
	bool operator== (Mtltag& _mtltag) {
		bool answer = true;
		for (uint8 i = 0; i < MATERIAL_TAG_BUFFER; i++) {
			if (_mtltag.m_tag[i] == '\0') return answer;
			answer &= (m_tag[i] == _mtltag.m_tag[i]);
		}
		return answer;
	}
	bool operator== (const char* _tag) {
		for (uint8 i = 0; (i < MATERIAL_TAG_BUFFER); i++) {
			if (m_tag[i] != _tag[i]) return false;
			if (_tag[i] == '\0') return true;
		}
	}
};
struct Mtllib { //It will store all the tag found during the reading of the matllib file
	//This mtllib will store an array of mtltags, to get the index who points at the actual stored material, 
	//we will have to add the index of the tag and the material previously added
	//The index can be used to get the material from the "gpuSH.m_materials" array (see "GlobalStructHandler.cuh")

	//This tags wont be freed unless a material is removed (yet to be implemented)
	//This tags will be stored into the mtllibStorage
	Mtltag* p_tags{nullptr};
	uint16* p_tagsId{nullptr};
	uint16 m_tagNum = 0, m_tagNumBL = 50;

	//------------------------------------------------------------------------ Constructors 
	Mtllib() = default;
	~Mtllib() {}
	void operator= (Mtllib& mtllib) { //It copies the mtltags and ids
		p_tags = mtllib.p_tags; p_tagsId = mtllib.p_tagsId;
		m_tagNum = mtllib.m_tagNum; m_tagNumBL = mtllib.m_tagNumBL;
	}

	//------------------------------------------------------------------------ Functions 
	//void Copy(Mtllib& mtllib) {
	//	for (uint16 i = 0; i < mtllib.m_tagNum; i++) {
	//		AddTag(mtllib.p_tags[i], mtllib.p_tagsId[i]);
	//	}
	//}
	void Deallocate() {
		if (p_tags != nullptr) {
			free(p_tags);
		}
		if (p_tagsId != nullptr) {
			free(p_tagsId);
		}
	}

	//------------------------------------------------------------------------ Functions 
	void Resize(uint16 lenght) {
		if (lenght >= m_tagNumBL) {
			m_tagNumBL = lenght;
			p_tags = (Mtltag*)realloc(p_tags, m_tagNumBL * sizeof(Mtltag));
			p_tagsId = (uint16*)realloc(p_tagsId, m_tagNumBL * sizeof(uint16));
		}
	}
	void AddTag(Mtltag& newTag, uint16 id) {//Add a tag
		//Increase the tag number, then check if the mtltag array is null, in this case we allocate it
		//If not, we reallocate it and check 4 errors, then we copy whats inside (operator=)
		if (!p_tags) {
			p_tags = (Mtltag*)malloc(m_tagNumBL * sizeof(Mtltag));
			if (!p_tags) {
				std::cout << "[ERRORE]: allocazione memoria fallita (CPU)\n";
				crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR;
				return;
			}
		}
		if (!p_tagsId) {
			p_tagsId = (uint16*)malloc(m_tagNumBL * sizeof(uint16));
			if (!p_tagsId) {
				std::cout << "[ERRORE]: allocazione memoria fallita (CPU)\n";
				crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR;
				return;
			}
		}

		if (m_tagNum >= m_tagNumBL) {
			Resize(m_tagNumBL + 50);
		}

		p_tags[m_tagNum] = newTag;
		p_tagsId[m_tagNum] = id;
		m_tagNum++;
	}
	uint16 SearchTag(const char* _tagToSearch) {//It will return the (relToMtllibIndex + 1) (1 -> tagNum), 0 if not found
		for (uint16 i = 0; i < m_tagNum; i++) {
			if (p_tags[i] == _tagToSearch) return i + 1;
		}
		return 0; //return false;
	}
	uint16 GetId(uint16 relToMtllibIndex) {
		return p_tagsId[relToMtllibIndex];
	}
	uint16 GetId(const char* _tagToSearch) { //It will return the id relative to the gpuSH.p_materials
		return GetId(SearchTag(_tagToSearch) - 1);
	}
};