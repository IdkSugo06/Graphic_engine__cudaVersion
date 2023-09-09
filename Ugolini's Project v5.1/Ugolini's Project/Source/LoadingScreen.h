#pragma once
#include "Headers/Singletons/GameHandler.hpp"
#include <thread>
using namespace std::chrono_literals;

#define LOADING_LOGO_CENTER_X (SCREEN_WIDTH - 80)
#define LOADING_LOGO_CENTER_Y 50

#define LOADING_LOGO_RINGOUT 20
#define LOADING_LOGO_RINGIN 10

#define LOADING_LOGO_STARTING_X (LOADING_LOGO_CENTER_X - LOADING_LOGO_RINGOUT)
#define LOADING_LOGO_STARTING_Y (LOADING_LOGO_CENTER_Y - LOADING_LOGO_RINGOUT)

#define LOADING_LOGO_QUADLENGHT_X (2 * LOADING_LOGO_RINGOUT)
#define LOADING_LOGO_QUADLENGHT_Y (2 * LOADING_LOGO_RINGOUT)

#define LOADING_LOGO_RED 1
#define LOADING_LOGO_GREEN 0
#define LOADING_LOGO_BLUE 1
#define LOADING_LOGO_COLOR LOADING_LOGO_RED,LOADING_LOGO_GREEN,LOADING_LOGO_BLUE

#define LOADING_LOGO_SPEED 4
#define LOADING_LOGO_SPEED_RED (LOADING_LOGO_SPEED * LOADING_LOGO_RED)
#define LOADING_LOGO_SPEED_GREEN (LOADING_LOGO_SPEED * LOADING_LOGO_GREEN)
#define LOADING_LOGO_SPEED_BLUE (LOADING_LOGO_SPEED * LOADING_LOGO_BLUE)


bool isLoading = true;
void InitLoadingScreen(byte* p_memory) {
	uint32 pixelId = LOADING_LOGO_STARTING_Y * SCREEN_WIDTH + LOADING_LOGO_STARTING_X;
	for (uint16 y = LOADING_LOGO_STARTING_Y; y < (LOADING_LOGO_STARTING_Y + LOADING_LOGO_QUADLENGHT_Y); y++) {
		for (uint16 x = LOADING_LOGO_STARTING_X; x < (LOADING_LOGO_STARTING_X + LOADING_LOGO_QUADLENGHT_X); x++) {
			Vector2 vect(x - LOADING_LOGO_CENTER_X, y - LOADING_LOGO_CENTER_Y);
			float coeff = (vect.x * vect.x + vect.y * vect.y);
			bool _b1 = coeff < LOADING_LOGO_RINGOUT * LOADING_LOGO_RINGOUT;
			bool _b2 = coeff > LOADING_LOGO_RINGIN * LOADING_LOGO_RINGIN;
			if (_b1 && _b2) {
				vect.Normalize();
				vect.x = (2 * (vect.y > 0) - 1) * (0.5f * vect.x + 0.5f) * 0.5f;
				float coeff = vect.x + 0.5f; RgbVector color(LOADING_LOGO_COLOR); color *= coeff; 
				BMP_IMAGE_SETCOLOR(p_memory, pixelId, (uint8)(255 * color.red), (uint8)(255 * color.green), (uint8)(255 * color.blue));
			}pixelId++;
		} pixelId += (SCREEN_WIDTH - LOADING_LOGO_QUADLENGHT_X);
	}
}
void UpdateLoadingLogo(byte* p_memory) {
	uint32 pixelId = LOADING_LOGO_STARTING_Y * SCREEN_WIDTH + LOADING_LOGO_STARTING_X;
	for (uint16 y = LOADING_LOGO_STARTING_Y; y < (LOADING_LOGO_STARTING_Y + LOADING_LOGO_QUADLENGHT_Y); y++) {
		for (uint16 x = LOADING_LOGO_STARTING_X; x < (LOADING_LOGO_STARTING_X + LOADING_LOGO_QUADLENGHT_X); x++) {
			float coeff = ((x - LOADING_LOGO_CENTER_X) * (x - LOADING_LOGO_CENTER_X) + (y - LOADING_LOGO_CENTER_Y) * (y - LOADING_LOGO_CENTER_Y));
			bool _b1 = coeff < LOADING_LOGO_RINGOUT * LOADING_LOGO_RINGOUT;
			bool _b2 = coeff > LOADING_LOGO_RINGIN * LOADING_LOGO_RINGIN;
			if (_b1 && _b2) {
				uint8 r = p_memory[pixelId * BMP_IMAGE_BYTEPP + 2] - LOADING_LOGO_SPEED_RED;
				uint8 g = p_memory[pixelId * BMP_IMAGE_BYTEPP + 1] - LOADING_LOGO_SPEED_GREEN;
				uint8 b = p_memory[pixelId * BMP_IMAGE_BYTEPP + 0] - LOADING_LOGO_SPEED_BLUE;
				BMP_IMAGE_SETCOLOR(p_memory, pixelId, r, g, b);
			}pixelId++;
		} pixelId += (SCREEN_WIDTH - LOADING_LOGO_QUADLENGHT_X);
	}
}
void LoadingScreen(MyWindow* window, BitmapImage* bmpImage) {
	while (isLoading) {
		UpdateLoadingLogo(cpuSH.p_rgbMap);
		window->DrawWindowFrame(&((*window).m_window_handler), bmpImage);
		std::this_thread::sleep_for(1ms);
	}
}