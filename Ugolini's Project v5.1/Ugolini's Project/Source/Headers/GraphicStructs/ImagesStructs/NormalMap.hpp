#pragma once

#include "RGB_structs.hpp"

struct NormalMap {
	static const uint16 m_sizeof_element = sizeof(Vector3);

	//--------------------------------------------------------------- Members
	bool m_isValid = false, m_isInCudaMem = false;
	uint16 m_width = 0, m_height = 0;					//Number of elements
	uint32 m_mapLenght = 0, m_sizeof_memory = 0;		//"sizeof" In bytes
	uint32 m_sizeof_line = 0, m_sizeof_column = 0;
	Vector3* p_memory{ nullptr };

	//--------------------------------------------------------------- Constructors
	NormalMap() = default;

	NormalMap(const char* file_path, float scale = 10) {
		BitmapImage bmpImage(file_path);
		if (!bmpImage.m_isValid) {
			m_isValid = false; return;
		}

		//Members
		m_width = bmpImage.m_width; m_height = bmpImage.m_height;
		m_sizeof_line = m_width * m_sizeof_element; m_sizeof_column = m_height * m_sizeof_element;
		m_mapLenght = m_width * m_height;
		m_sizeof_memory = m_mapLenght * m_sizeof_element;

		//Allocation
		p_memory = (Vector3*)malloc(m_sizeof_memory);
		if (!p_memory) {
			m_isValid = false; return;
		}
		m_isValid = true; scale = 1;

		//From rgb to float
		for (uint16 y = 0; y < m_height; y++) {
			for (uint16 x = 0; x < m_width; x++) {
				uint32 id = (x + y * m_width);
				uint32 bmpImage_id = (x + y * m_width) * BMP_IMAGE_BYTEPP;	//In bytes
				uint8* cell = &((byte*)bmpImage.p_memory)[bmpImage_id];

				p_memory[id].z = ((float)(cell[0] - 128)) * (-2.0f / 256);
				p_memory[id].y = (((float)cell[1]) * (2.0f / 256)) - 1;
				p_memory[id].x = (((float)cell[2]) * (2.0f / 256)) - 1;

				if (scale > 1) { p_memory[id].vectPow(scale); }
				p_memory[id].Normalize();
			}
		}
	}

	~NormalMap() {
		if (!m_isValid) return;
		if (m_isInCudaMem) { cudaFree(p_memory); return; }
		free(p_memory);
	}

	void operator= (NormalMap& nmap) {
		m_isValid = nmap.m_isValid; nmap.m_isValid = false;
		m_isInCudaMem = nmap.m_isInCudaMem;
		m_width = nmap.m_width; m_height = nmap.m_height;					
		m_mapLenght = nmap.m_mapLenght; m_sizeof_memory = nmap.m_sizeof_memory;	
		m_sizeof_line = nmap.m_sizeof_line; m_sizeof_column = nmap.m_sizeof_column;
		p_memory = nmap.p_memory;
	}
};