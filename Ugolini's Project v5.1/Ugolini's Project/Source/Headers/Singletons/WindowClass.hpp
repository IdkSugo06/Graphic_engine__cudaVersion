#pragma once

#include "Windows.h"
#include "Input.hpp"
#include "../GraphicStructs\Camera.hpp"
#include "..\PipeLine_Manager\GPU_PipelineManager.cuh"


#define WINDOW_STARTING_POINT_X 	0
#define WINDOW_STARTING_POINT_Y 	0

#define WINDOW_CLASS_NAME			"Uoglini's window"
#define WINDOW_CLASS_NAME_WCHAR		L"Ugolini's window"
#define WINDOW_NAME "Ugolini's Project"


//------------------------------------------------------------------------------------ Messages handler
LRESULT CALLBACK WindowMessageProcedure(HWND window_handler, UINT msg, WPARAM wParam, LPARAM lParam) {

	switch (msg) {
	case WM_CLOSE:
		DestroyWindow(window_handler);
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	case WM_KEYDOWN:
		InputHandler.keyChange(wParam, true);
		break;
	case WM_KEYUP:
		InputHandler.keyChange(wParam, false);
		break;
	}
	return DefWindowProc(window_handler, msg, wParam, lParam);
}


class MyWindow {
	//---------------------------------------------------- Members
public:
	HINSTANCE m_handler_instance;
	HWND m_window_handler;


	//---------------------------------------------------- Constructors
	MyWindow() : m_handler_instance(GetModuleHandle(nullptr)) {
		const wchar_t* CLASS_NAME = WINDOW_CLASS_NAME_WCHAR;

		WNDCLASS window_class = {};
		window_class.lpszClassName = (LPCSTR)CLASS_NAME;  	//Class name
		window_class.hInstance = m_handler_instance;
		window_class.hIcon = LoadIcon(NULL, IDI_WINLOGO);
		window_class.hCursor = LoadCursor(NULL, IDC_ARROW);
		window_class.lpfnWndProc = WindowMessageProcedure;	//Associo un puntatore a funzione
		RegisterClass(&window_class);

		RECT rect;
		rect.left = WINDOW_STARTING_POINT_X;
		rect.top = WINDOW_STARTING_POINT_Y;
		rect.right = rect.left + SCREEN_WIDTH;
		rect.bottom = rect.top + SCREEN_HEIGHT;

		DWORD style = WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU;
		AdjustWindowRect(&rect, style, false);

		m_window_handler = CreateWindowEx(0, (LPCSTR)CLASS_NAME, (LPCSTR)WINDOW_NAME, style,
			rect.left,
			rect.top,
			rect.right - rect.left,
			rect.bottom - rect.top,
			NULL, NULL, m_handler_instance, NULL
		);

		ShowWindow(m_window_handler, SW_SHOW);
	}
	MyWindow(MyWindow const&) = delete;
	~MyWindow() {
		const wchar_t* CLASS_NAME = WINDOW_CLASS_NAME_WCHAR;
		UnregisterClassW(CLASS_NAME, m_handler_instance);
	}


	//---------------------------------------------------- Operator overload
	void operator= (MyWindow const&) = delete;

	void DrawWindowFrame(HWND* window_handler, BitmapImage& bmpImage) {
		HDC DeviceContext = GetDC(*window_handler);
		StretchDIBits(DeviceContext, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, 0, bmpImage.m_width, bmpImage.m_height, bmpImage.p_memory, &(bmpImage.m_bmi), DIB_RGB_COLORS, SRCCOPY);
		ReleaseDC(*window_handler, DeviceContext);
	}
	void DrawWindowFrame(HWND* window_handler, BitmapImage* bmpImage) {
		HDC DeviceContext = GetDC(*window_handler);
		StretchDIBits(DeviceContext, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, 0, bmpImage->m_width, bmpImage->m_height, bmpImage->p_memory, &(bmpImage->m_bmi), DIB_RGB_COLORS, SRCCOPY);
		ReleaseDC(*window_handler, DeviceContext);
	}

	//---------------------------------------------------- Messages processor
	bool ProcessMessages() {
		MSG msg = {};
		while (PeekMessage(&msg, nullptr, 0u, 0u, PM_REMOVE)) {
			if (msg.message == WM_QUIT) {
				return false;
			}

			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		return true;
	}
};