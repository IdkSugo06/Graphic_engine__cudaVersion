// credits for input handler: Daniel Blagy https://youtu.be/BogrUrFuQ6A 

#ifndef MY_TYPEDEF
#define MY_TYPEDEF
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;
#endif

#define DC_MAX_KEYS 52

#define DC_A			0
#define DC_B			1
#define DC_C			2
#define DC_D			3
#define DC_E			4
#define DC_F			5
#define DC_G			6
#define DC_H			7
#define DC_I			8
#define DC_J			9
#define DC_K			10
#define DC_L			11
#define DC_M			12
#define DC_N			13
#define DC_O			14
#define DC_P			15
#define DC_Q			16
#define DC_R			17
#define DC_S			18
#define DC_T			19
#define DC_U			20
#define DC_V			21
#define DC_W			22
#define DC_X			23
#define DC_Y			24
#define DC_Z			25

#define DC_UP			26
#define DC_DOWN			27
#define DC_LEFT			28
#define DC_RIGHT		29

#define DC_0			30
#define DC_1			31
#define DC_2			32
#define DC_3			33
#define DC_4			34
#define DC_5			35
#define DC_6			36
#define DC_7			37
#define DC_8			38
#define DC_9			39
#define DC_MINUS		40
#define DC_PLUS			41

#define DC_SHIFT		42
#define DC_CONTROL		43
#define DC_ALT			44
#define DC_SPACE		45
#define DC_ESCAPE		46
#define DC_CAPSLOCK		47
#define DC_TAB			48
#define DC_ENTER		49
#define DC_BACKSPACE	50
#define DC_TILDE		51


struct keyState {
	bool isDown;
};
struct keyMapInfo {
	keyState map[DC_MAX_KEYS];
};


struct Input {
	keyMapInfo keyMapInfo;
	void keyChange(uint32 VK_code, uint8 status = 255) {
		uint32 code = 0;

		//Useful
		if (VK_code >= 'A' && VK_code <= 'Z')
			code = VK_code - 'A';
		else if (VK_code == VK_SPACE)
			code = DC_SPACE;
		else if (VK_code == VK_SHIFT)
			code = DC_SHIFT;
		else if (VK_code == VK_CONTROL)
			code = DC_CONTROL;
		else if (VK_code == VK_CAPITAL)
			code = DC_CAPSLOCK;
		else if (VK_code == VK_ESCAPE)
			code = DC_ESCAPE;

		//A little less
		else if (VK_code == VK_UP)
			code = DC_UP;
		else if (VK_code == VK_DOWN)
			code = DC_DOWN;
		else if (VK_code == VK_LEFT)
			code = DC_LEFT;
		else if (VK_code == VK_RIGHT)
			code = DC_RIGHT;
		else if (VK_code >= '0' && VK_code <= '9')
			code = VK_code - '0' + DC_0;

		//I dont think someone has ever pressed them
		else if (VK_code == VK_OEM_MINUS)
			code = DC_MINUS;
		else if (VK_code == VK_OEM_PLUS)
			code = DC_PLUS;
		else if (VK_code == VK_MENU)
			code = DC_ALT;
		
		if (status == 255) //If the status of the button is not specified, invert it
			keyMapInfo.map[code].isDown = !keyMapInfo.map[code].isDown;
		else
			keyMapInfo.map[code].isDown = status;
	}
	bool isPressed(uint32 keyCode) {
		return keyMapInfo.map[keyCode].isDown;
	}
};
Input InputHandler;
