#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../Math/MyMath.hpp"
#include "BitmapImage.hpp"


struct RgbVector { //That vector will store rgba values (0->1)
	//--------------------------------------------------------------- Members
public:
	float red = 0, green = 0, blue = 0, alfa = 0;


	//--------------------------------------------------------------- Constructor
	__device__ __host__ RgbVector() {
		red = 0; green = 0; blue = 0; alfa = 0;
	}
	__device__ __host__ RgbVector(float f, float _alfa = 0) {
		red = f; green = f; blue = f; alfa = _alfa;
	}
	__device__ __host__ RgbVector(float _red, float _green, float _blue, float _alfa = 0) {
		red = _red; green = _green; blue = _blue; alfa = _alfa;
	}
	~RgbVector() = default;

#pragma region operators overload
	//--------------------------------------------------------------- Operand overloads
	__device__ __host__ RgbVector operator+ (RgbVector const& vect) {
		return RgbVector(red + vect.red, green + vect.green, blue + vect.blue, alfa + vect.alfa);
	}
	__device__ __host__ RgbVector operator- (RgbVector const& vect) {
		return RgbVector(red - vect.red, green - vect.green, blue - vect.blue, alfa - vect.alfa);
	}
	__device__ __host__ RgbVector operator* (RgbVector const& vect) {
		return RgbVector(red * vect.red, green * vect.green, blue * vect.blue, alfa * vect.alfa);
	}
	__device__ __host__ RgbVector operator/ (RgbVector const& vect) {	//Not capped
		return RgbVector(red / vect.red, green / vect.green, blue / vect.blue, alfa / vect.alfa);
	}
	__device__ __host__ RgbVector operator* (float num) {
		return RgbVector(red * num, green * num, blue * num, alfa * num);
	}

	__device__ __host__ void operator+= (RgbVector const& vect) {	//Not capped
		red += vect.red; green += vect.green; blue += vect.blue; alfa += vect.alfa;
	}
	__device__ __host__ void operator-= (RgbVector const& vect) {	//Not capped
		red -= vect.red; green -= vect.green; blue -= vect.blue; alfa -= vect.alfa;
	}
	__device__ __host__ void operator*= (RgbVector const& vect) { //Not capped
		red *= vect.red, green *= vect.green, blue *= vect.blue, alfa *= vect.alfa;
	}
	__device__ __host__ void operator/= (RgbVector const& vect) {	//Not capped
		red = red / vect.red, green = green / vect.green, blue = blue / vect.blue, alfa = alfa / vect.alfa;
	}
	__device__ __host__ void operator*= (float num) {	//Not capped
		red = red * num, green = green * num, blue = blue * num, alfa = alfa * num;
	}


	__device__ __host__ void operator= (RgbVector const& vect) {	//Not capped
		red = vect.red; green = vect.green; blue = vect.blue; alfa = vect.alfa;
	}
	__device__ __host__ bool operator== (RgbVector const& vect) {
		return (red == vect.red && green == vect.green && blue == vect.blue && alfa == vect.alfa);
	}

	__device__ __host__ void powRgbVector(float num) {
		red = pow(red, num); green = pow(green, num); blue = pow(blue, num); alfa = pow(alfa, num);
	}
	__device__ __host__ void Cap() {
		red = myBetween(red, 0, 1); green = myBetween(green, 0, 1); blue = myBetween(blue, 0, 1); alfa = myBetween(alfa, 0, 1);
	}
	__device__ __host__ void Threshold(float f) {
		if (red < f) {red = 0;}
		if (green < f) { green = 0; }
		if (blue < f) { blue = 0; }
		if (alfa < f) { alfa = 0; }
	}
	__device__ __host__ static RgbVector Capped(RgbVector vect) {
		 vect.Cap(); return vect;
	}

#pragma endregion operators overload

	static void PrintRGB(RgbVector const& vect) {
		std::cout << "(red: " << vect.red << ", green: " << vect.green << ", blue: " << vect.blue << ")\n";
	}

	//--------------------------------------------------------------- Static const Variables
	static const RgbVector Red;
	static const RgbVector Green;
	static const RgbVector Blue;
	static const RgbVector White;
	static const RgbVector LGrey;
	static const RgbVector Grey;
	static const RgbVector DGrey;
	static const RgbVector Black;

	__device__ __host__ static RgbVector GetGrayScale(float darkness = 0.3) {
		return RgbVector(darkness, darkness, darkness);
	}
};

struct RgbVectorMap { //Map of 4 floats per element (r,g,b,a) (0->1)
	static const uint16 m_sizeof_element = sizeof(RgbVector);						//In bytes

	//--------------------------------------------------------------- Members
	bool m_isValid = false, m_isInCudaMem = false;
	uint16 m_width = 0, m_height = 0;					//Number of elements
	uint32 m_mapLenght = 0, m_sizeof_memory = 0;		//"sizeof" In bytes
	uint32 m_sizeof_line = 0, m_sizeof_column = 0;		
	RgbVector* p_memory{nullptr};

	//--------------------------------------------------------------- Constructors
	RgbVectorMap() = default;
	RgbVectorMap(const char* file_path) {
		BitmapImage bmpImage(file_path);
		if (!bmpImage.m_isValid) { 
			m_isValid = false; return;
		}

		//Members
		m_width = bmpImage.m_width; m_height = bmpImage.m_height;
		m_sizeof_line = m_width * m_sizeof_element; m_sizeof_column = m_height * m_sizeof_element;
		m_mapLenght = (uint32)m_width * m_height;
		m_sizeof_memory = m_mapLenght * m_sizeof_element;

		//for (uint32 y = 0; y < m_height; y++) { //Print every pixel of the bitmap image
		//	for (uint32 x = 0; x < m_width; x++) {
		//		std::cout << (int)((byte*)bmpImage.p_memory)[(x + y * bmpImage.m_width) * BMP_IMAGE_BYTEPP + 0] << "\n";
		//		std::cout << (int)((byte*)bmpImage.p_memory)[(x + y * bmpImage.m_width) * BMP_IMAGE_BYTEPP + 1] << "\n";
		//		std::cout << (int)((byte*)bmpImage.p_memory)[(x + y * bmpImage.m_width) * BMP_IMAGE_BYTEPP + 2] << "\n";
		//		std::cout << (int)((byte*)bmpImage.p_memory)[(x + y * bmpImage.m_width) * BMP_IMAGE_BYTEPP + 3] << "\n\n";
		//	}
		//}Sleep(20000);

		//Allocation
		p_memory = (RgbVector*)malloc(m_sizeof_memory);
		if (!p_memory) { 
			m_isValid = false; return; }
		m_isValid = true;

		//From (0->255) to (0->1)
		for (uint16 y = 0; y < m_height; y++) {
			for (uint16 x = 0; x < m_width; x++) {
				uint32 id = (x + y * m_width);
				uint32 bmpImage_id = (x + y * m_width) * BMP_IMAGE_BYTEPP;	//In bytes
				uint8* cell = &((byte*)bmpImage.p_memory)[bmpImage_id];
				p_memory[id].blue		= ((float)cell[0]) / 255;
				p_memory[id].green		= ((float)cell[1]) / 255;
				p_memory[id].red		= ((float)cell[2]) / 255;
				if (BMP_IMAGE_BYTEPP > 3) {
					p_memory[id].alfa	= ((float)cell[3]) / 255;
				}
			}
		}

		//for (uint16 y = 0; y < m_height; y++) {
		//	for (uint16 x = 0; x < m_width; x++) {
		//		std::cout << (float)(p_memory[y * m_width + x].red) << "\n";
		//		std::cout << (float)(p_memory[y * m_width + x].green) << "\n";
		//		std::cout << (float)(p_memory[y * m_width + x].blue) << "\n";
		//		std::cout << (float)(p_memory[y * m_width + x].alfa) << "\n";
		//	}
		//}
	}
	RgbVectorMap(uint16 _w, uint16 _h, RgbVector* mem = nullptr) {
		m_width = _w; m_height = _h;
		m_sizeof_line = m_width * m_sizeof_element; m_sizeof_column = m_height * m_sizeof_element;
		m_mapLenght = (uint32)m_width * m_height;
		m_sizeof_memory = m_mapLenght * m_sizeof_element;
		if (mem) { p_memory = mem; }
		else { p_memory = (RgbVector*)malloc(m_sizeof_memory); }
		if (!p_memory) m_isValid = false; return;
	}
	~RgbVectorMap() {
		if (m_isValid) {
			if (m_isInCudaMem) cudaFree(p_memory); return;
			free(p_memory); return;
		}
	}

	void operator= (RgbVectorMap& map) {
		m_isValid = map.m_isValid;
		if (!m_isValid) return;
		map.m_isValid = false; m_isInCudaMem = map.m_isInCudaMem;
		m_width = map.m_width; m_height = map.m_height; m_mapLenght = map.m_mapLenght;
		m_sizeof_line = map.m_sizeof_line; m_sizeof_column = map.m_sizeof_column; m_sizeof_memory = map.m_sizeof_memory;
		p_memory = map.p_memory;
	}
};

RgbVector const RgbVector::Red(1, 0, 0);
RgbVector const RgbVector::Green(0, 1, 0);
RgbVector const RgbVector::Blue(0, 0, 1);
RgbVector const RgbVector::White(1, 1, 1);
RgbVector const RgbVector::LGrey(0.7, 0.7, 0.7);
RgbVector const RgbVector::Grey(0.5, 0.5, 0.5);
RgbVector const RgbVector::DGrey(0.2, 0.2, 0.2);
RgbVector const RgbVector::Black(0, 0, 0);