#pragma once 
#include "AdvancedFileFunction.hpp"
#include "WindowClass.hpp"
#include "chrono"
typedef std::chrono::steady_clock::time_point chronoTimePoint;


#define FPS_SAMPLE_NUMBER 5
struct GameHandler {
	float deltaTime = 0;
	chronoTimePoint time = std::chrono::steady_clock::now();
	chronoTimePoint lastTime = std::chrono::steady_clock::now();
	uint16 fps[FPS_SAMPLE_NUMBER];
	uint32 frameSinceStart;


	void Update() {
		lastTime = time;
		time = std::chrono::steady_clock::now();
		deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(time - lastTime).count();
		if (deltaTime != 0) {
			uint16 newFpsSample = (uint16)(1000.0f / deltaTime);
			UpdateFps(newFpsSample);
		}
		frameSinceStart++;
	}
	void UpdateFps(uint16 newFpsSample) {
		for (uint8 i = 0; i < FPS_SAMPLE_NUMBER - 1; i++)
			fps[i + 1] = fps[i];
		fps[0] = newFpsSample;
	}
	float Compute_fps() {
		uint32 total = 0;
		for (uint8 i = 0; i < FPS_SAMPLE_NUMBER; i++)
			total += fps[i];
		return (((float)total) / FPS_SAMPLE_NUMBER);
	}
};
GameHandler game;