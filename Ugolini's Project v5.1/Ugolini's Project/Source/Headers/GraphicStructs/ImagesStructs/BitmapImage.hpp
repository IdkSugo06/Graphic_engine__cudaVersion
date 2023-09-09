#pragma once

#include <Windows.h>
#include "..\..\Singletons\BasicFileFunctions.hpp"

#ifndef MY_TYPEDEF
#define MY_TYPEDEF
typedef unsigned char uint8;		//255
typedef unsigned short uint16;		//65536
typedef unsigned int uint32;		//4 000 000 000
typedef unsigned long long uint64;	//simply too much
#endif

#define BMP_IMAGE_BYTEPP 4
#define BMP_IMAGE_BITPP (BMP_IMAGE_BYTEPP * 8)
#define BMP_IMAGE_SETCOLOR(p_mem,i_pos,r,g,b) p_mem[i_pos * BMP_IMAGE_BYTEPP + 2] = r; p_mem[i_pos * BMP_IMAGE_BYTEPP + 1] = g; p_mem[i_pos * BMP_IMAGE_BYTEPP] = b; 

void createBMI(BITMAPINFO& m_bmi, uint16 _w, uint16 _h) {
	m_bmi.bmiHeader.biSize = sizeof(m_bmi.bmiHeader);
	m_bmi.bmiHeader.biWidth = _w;
	m_bmi.bmiHeader.biHeight = _h;
	m_bmi.bmiHeader.biBitCount = BMP_IMAGE_BITPP;
	m_bmi.bmiHeader.biCompression = BI_RGB;
	m_bmi.bmiHeader.biPlanes = 1;
}
struct BitmapImage{ //No padding, protocol [b1,g1,r1,b2,g2,r2,...,bn,gn,rn]
	bool m_isValid = false; bool m_isInCudaMem = false;
	BITMAPINFO m_bmi;
	uint32 m_width = 0, m_height = 0, m_padding = 0;
	uint32 m_bytesPerLine = 0, m_sizeofImage = 0;
	uint32 m_memoryLenght;
	void* p_memory{ nullptr };

	BitmapImage(const char* file_path) {
		m_isValid = false;

		//file opening
		std::fstream fin;
		fin.open(file_path, std::ios::in | std::ios::binary);
		uint8 charBuffer[FILE_CHAR_BUFFER_LENGHT];

		if (!fin) {//Se il file non si è aperto
			std::cout << "Errore nel caricamento dell'immagine bitmap (\"" << file_path << "\"), attenderà 5 sec" << std::endl;
			Sleep(5000);
			return;
		}

		//HEADERS
		myFgets(fin, charBuffer, 2);
		if (charBuffer[0] != 66 || charBuffer[1] != 77) return;

		//uint32 fileSize;
		myFgets(fin, charBuffer, 4);
		//fileSize = (charBuffer[0] | (charBuffer[1] << 8) | (charBuffer[2] << 16) | (charBuffer[3] << 24));

		myFgets(fin, charBuffer, 4); // reserved bytes (4)
		myFgets(fin, charBuffer, 4); // pixelArray offset (4)

		uint16 headerDim;
		myFgets(fin, charBuffer, 4);
		headerDim = (charBuffer[0] | (charBuffer[1] << 8) | (charBuffer[2] << 16) | (charBuffer[3] << 24));

		myFgets(fin, charBuffer, 4);
		m_width = (charBuffer[0] | (charBuffer[1] << 8) | (charBuffer[2] << 16) | (charBuffer[3] << 24));
		myFgets(fin, charBuffer, 4);
		m_height = (charBuffer[0] | (charBuffer[1] << 8) | (charBuffer[2] << 16) | (charBuffer[3] << 24));

		m_padding = (4 - ((m_width * 3) % 4)) % 4;
		m_bytesPerLine = m_width * 3;
		m_sizeofImage = (uint32)m_bytesPerLine * m_height * sizeof(uint8);
		m_memoryLenght = (uint32)m_width * m_height;
		p_memory = (uint8*)malloc((uint32)m_width * m_height * BMP_IMAGE_BYTEPP);
		 
		if (!p_memory) return;
		m_isValid = true;
		myFgets(fin, charBuffer, headerDim - 12); //skip the header (-12 cause of the chars just read)

		//PIXEL ARRAY
		for (uint32 y = 0; y < m_height; y++) {	//In bytes
			for (uint32 x = 0; x < m_width * BMP_IMAGE_BYTEPP; x += BMP_IMAGE_BYTEPP) { //In bytes
				myFgets(fin, charBuffer, 3);
				((byte*)p_memory)[y * m_width * BMP_IMAGE_BYTEPP + x + 0] = (byte)(charBuffer[0]);
				((byte*)p_memory)[y * m_width * BMP_IMAGE_BYTEPP + x + 1] = (byte)(charBuffer[1]);
				((byte*)p_memory)[y * m_width * BMP_IMAGE_BYTEPP + x + 2] = (byte)(charBuffer[2]);
				for (uint8 i = 0; i < BMP_IMAGE_BYTEPP - 3; i++) {
					((byte*)p_memory)[y * m_width * BMP_IMAGE_BYTEPP + x + 3 + i] = 0;
				}
			}
			myFgets(fin, charBuffer, m_padding);
		}
		fin.close();
		return;
	}
	BitmapImage(uint16 _w, uint16 _h, void* mem = nullptr) {
		m_width = _w;
		m_height = _h;
		m_padding = (_w * 3) % 4;
		m_bytesPerLine = _w * 3;
		m_sizeofImage = m_bytesPerLine * _h * sizeof(uint8);
		m_memoryLenght = _w * _h;
		createBMI(m_bmi, _w, _h);
		if (mem) p_memory = mem; m_isValid = true; return;
		p_memory = (uint8*)malloc(m_sizeofImage);
		if (p_memory) m_isValid = true;
		MyFillMemory(255, 255, 255);
	}
	BitmapImage() : BitmapImage(1, 1) {};
	~BitmapImage() {
		if (m_isValid) {
			if (m_isInCudaMem) cudaFree(p_memory); return;
			free(p_memory); return;
		}
	}

	void MyFillMemory(uint8 red, uint8 green, uint8 blue) {
		for (uint16 y = 0; y < m_height; y++) {
			uint32 offset = y * m_bytesPerLine; //Padding included here
			for (uint16 x = 0; x < m_width; x += 3) {
				((uint8*)p_memory)[x + offset + 0] = blue;
				((uint8*)p_memory)[x + offset + 1] = green;
				((uint8*)p_memory)[x + offset + 2] = red;
			}
		}
	}
};