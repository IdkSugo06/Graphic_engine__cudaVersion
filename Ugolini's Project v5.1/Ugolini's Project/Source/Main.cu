#include "LoadingScreen.h"

#define CoffeTableObjRelPath "Files/Objects/Coffe table object/Coffee_Table_01_BLEND.obj"
#define TableLampObjRelPath "Files/Objects/Table lamp object/Pendulum.obj"
#define BlueLampObjRelPath "Files/Objects/Lamp object/Luce blu.obj"
#define MagentaLampObjRelPath "Files/Objects/Lamp object/Luce magenta.obj"
#define BricksObjRelPath "Files/Objects/Briks object/Tiles.obj"
#define BulletObjRelPath "Files/Objects/Bullet object/Bullet.obj"
#define BObjRelPath "Files/Objects/B object/B.obj"
#define GunObjRelPath "Files/Objects/Gun object/g18.obj"

#define GrassBlockObjRelPath "Files/Objects/Minecraft block object/Grass_Block.obj"
#define GrassBlockPosition 0, -1.5, -10.5


#define CAMERA_STEP 0.2f / 1000
#define CAMERA_ANGLE (1.0f/35 * PI) / 1000 


void ProcessInput();
static bool b_BobjSpinning = false;
static bool b_GrassBlockSpinning = true;
static int cicleTo_optionChange = 0;
void loop() {
	ProcessInput();
	if (b_BobjSpinning) { 
		objectCollections.MyGetObject(-1)->RotateObject(Quaternion(Vector3(0, 1, 0), (-PI / 1500) * game.deltaTime), Vector3(0, 1, 5)); 
	}
	if (b_GrassBlockSpinning) { 
		objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(0, 1, 0), (-PI / 2500) * game.deltaTime), Vector3(GrassBlockPosition)); 
	}
}

void LoadObjects();
int main() {
	//Open the window and create the rgb memory
	MyWindow* p_Window = new MyWindow();
	BitmapImage bitmapImage(SCREEN_WIDTH, SCREEN_HEIGHT, cpuSH.p_rgbMap);
	InitLoadingScreen(cpuSH.p_rgbMap);
	p_Window->DrawWindowFrame(&((*p_Window).m_window_handler), bitmapImage);
	std::thread loadingThread(LoadingScreen, p_Window, &bitmapImage);

	//Load the scene
	LoadObjects();
	isLoading = false;

	loadingThread.join();

	//Wait for the operation to becompleted
	cudaStreamSynchronize(cudaStreamHandler.copyStream);
	Sleep(400);

	//Check for errors
	if (crashHandler.crashCode != CH_NO_CRASH_OCCURRED) {
		std::cout << "Interruzione imposta dallo crash handler, codice (" << crashHandler.crashCode << ")\n";
		Sleep(5000);
		return;
	}

	//Set up the loop
	bool running = true;
	
	while (running) {
		if (crashHandler.crashCode != CH_NO_CRASH_OCCURRED) {
			std::cout << "Interruzione imposta dallo crash handler, codice (" << crashHandler.crashCode << ")\n";
			Sleep(5000);
			running = false; break;
		}
		if (!p_Window->ProcessMessages()) {
			std::cout << "Closing window\n";
			running = false; break;
		}

		//START CODE HERE
		game.Update();
		loop();

		//END CODE HERE
		pipelineManager.InitialDataTransfer();
		cudaStreamSynchronize(cudaStreamHandler.copyStream);
		pipelineManager.Execute();

		//Fps cout
		std::cout << "Fps: " << game.Compute_fps() << "\n";
		//std::cout << "\tPunti totali: " << cpuSH.m_vectors_number << "\n";
		//std::cout << "\tTriangoli totali: " << cpuSH.m_triangles_number << "\n";
		if (game.Compute_fps() > 300) { crashHandler.crashCode = CH_FPS_THRESHOLD_REACHED; }
		cudaStreamSynchronize(cudaStreamHandler.executionStream);
		pipelineManager.FinalDataTransfer();
		cudaStreamSynchronize(cudaStreamHandler.copyStream);
		p_Window->DrawWindowFrame(&((*p_Window).m_window_handler), bitmapImage);
	}
	delete p_Window;
	return 0;
}

void LoadObjects() {
#define ERRORE_GENERALE_LETTURA_OBJ std::cout << "[ERRORE]: errore generico rilevato durante la letttura di un fle .obj :C\n"; Sleep(5000);
	if (!ReadOBJfile(CoffeTableObjRelPath)) { ERRORE_GENERALE_LETTURA_OBJ return; }
	objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(0, 1, 0), PI));
	objectCollections.CurrentObject()->MoveObject(Vector3(0, 0, 0));
	if (!ReadOBJfile(TableLampObjRelPath)) { ERRORE_GENERALE_LETTURA_OBJ return; }
	objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(0, 1, 0), 2 * PI / 8));
	objectCollections.CurrentObject()->MoveObject(Vector3(0.6f, 0.3f, .1f));
	if (!ReadOBJfile(BlueLampObjRelPath)) { ERRORE_GENERALE_LETTURA_OBJ return; }
	objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(0, 1, 0), -65 * PI / 96));
	objectCollections.CurrentObject()->MoveObject(Vector3(2, 1.5f, 5));
	if (!ReadOBJfile(MagentaLampObjRelPath)) { ERRORE_GENERALE_LETTURA_OBJ return; }
	objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(0, 1, 0), 65 * PI / 96));
	objectCollections.CurrentObject()->MoveObject(Vector3(-2, 1.5f, 5));
	if (!ReadOBJfile(BlueLampObjRelPath)) { ERRORE_GENERALE_LETTURA_OBJ return; }
	objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(0, 1, 0), 25 * PI / 96));
	objectCollections.CurrentObject()->MoveObject(Vector3(-2.5f, 2.5f, 9));
	if (!ReadOBJfile(MagentaLampObjRelPath)) { ERRORE_GENERALE_LETTURA_OBJ return; }
	objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(0, 1, 0), -25 * PI / 96));
	objectCollections.CurrentObject()->MoveObject(Vector3(2.5f, 1, 9));
	if (!ReadOBJfile(BricksObjRelPath)) { ERRORE_GENERALE_LETTURA_OBJ return; }
	objectCollections.CurrentObject()->MoveObject(Vector3(-1, 1, 15));
	if (!ReadOBJfile(BulletObjRelPath)) { ERRORE_GENERALE_LETTURA_OBJ return; }
	objectCollections.CurrentObject()->MoveObject(Vector3(.1f, .55f, .25f));
	if (!ReadOBJfile(BulletObjRelPath)) { ERRORE_GENERALE_LETTURA_OBJ return; }
	objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(1, 0, 0), -PI / 2));
	objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(0, 1, 0), 7 * PI / 8));
	objectCollections.CurrentObject()->MoveObject(Vector3(.25f, .71f, -.1f));
	if (!ReadOBJfile(GunObjRelPath)) { ERRORE_GENERALE_LETTURA_OBJ return; }
	objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(0, 0, 1), -PI / 2));
	objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(1, 0, 0), -PI / 16));
	objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(0, 1, 0), 59 * PI / 96));
	objectCollections.CurrentObject()->MoveObject(Vector3(0, 0.42f, -.27f));
	if (!ReadOBJfile(BObjRelPath)) { ERRORE_GENERALE_LETTURA_OBJ return; }
	objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(1, 0, 0), -PI / 2));
	objectCollections.CurrentObject()->MoveObject(Vector3(0, 1, 5));
	objectCollections.CurrentObject()->RotateObject(Quaternion(Vector3(0, 1, 0), PI + (PI * 11 / 60)), Vector3(0, 1, 5));
	if (!ReadOBJfile(GrassBlockObjRelPath)) { ERRORE_GENERALE_LETTURA_OBJ return; }
	objectCollections.CurrentObject()->MoveObject(Vector3(GrassBlockPosition));

	gpuSH.CreateLight__spotLight(Vector3(0.6f, .5f, .1f), Vector3(-.7f, -.2f, -.4f), RgbVector(4), DEFAULT_SPOTLIGHT_CUTOFF_VALUE, DEFAULT_SPOTLIGHT_DEGRADATION_VALUE, 0.05, 1, 1.5);
	gpuSH.CreateLight__spotLight(Vector3(-2, 1, 5), Vector3(.9f, 0, -.1f), RgbVector(1.5f, 0, 1.5f), cos(PI / 6), cos(PI / 6 + PI / 36), 0.05, 0.2, 0.2);
	gpuSH.CreateLight__spotLight(Vector3(2, 1, 5), Vector3(-.9f, 0, -.1f), RgbVector(0, 0.5f, 2.5f), cos(PI / 6), cos(PI / 6 + PI / 36), 0.05, 0.2, 0.2);
	gpuSH.CreateLight__spotLight(Vector3(2.5f, 0.2f, 9), Vector3(-.6f, .3f, .3f), RgbVector(1.2f, 0, 1.2f), cos(PI / 4), cos(PI / 5 + PI / 9), 0.05, 0.2, 0.2);
	gpuSH.CreateLight__spotLight(Vector3(-2.5f, 1.4f, 9), Vector3(.6f, -.3f, .3f), RgbVector(0, .7f, 3.5f), cos(PI / 4), cos(PI / 5 + PI / 9), 0.05, 0.2, 0.2);

	gpuSH.CreateLight__pointLight(Vector3(2, .7, -8), RgbVector(1, 1, 1));
	gpuSH.CreateLight__pointLight(Vector3(-1.5f, -1, -9.5f), RgbVector(0.5f, 0, 0.5f));
	gpuSH.CreateLight__pointLight(Vector3(.3f, 1.2f, -11), RgbVector(.7f, .7f, 2.2f));
}

void ProcessInput() {
	if (cicleTo_optionChange > 0) {
		cicleTo_optionChange--;
	}

	{ //CAMERA INPUT 
		float camStep = CAMERA_STEP;
		float camAngleRotation = CAMERA_ANGLE;
		if (InputHandler.keyMapInfo.map[DC_SHIFT].isDown) {
			camStep *= 5; camAngleRotation *= 10;
		}
		if (InputHandler.keyMapInfo.map[DC_B].isDown) {
			camStep /= 5; camAngleRotation /= 10;
		}

		//Movement
		if (InputHandler.keyMapInfo.map[DC_D].isDown) {
			p_camera->m_position += Quaternion::RotatePoint(p_camera->m_orientation, Vector3(camStep * game.deltaTime, 0, 0));
		}
		if (InputHandler.keyMapInfo.map[DC_A].isDown) {
			p_camera->m_position += Quaternion::RotatePoint(p_camera->m_orientation, Vector3(-camStep * game.deltaTime, 0, 0));
		}
		if (InputHandler.keyMapInfo.map[DC_SPACE].isDown) {
			p_camera->m_position += Quaternion::RotatePoint(p_camera->m_orientation, Vector3(0, camStep * game.deltaTime, 0));
		}
		if (InputHandler.keyMapInfo.map[DC_CONTROL].isDown) {
			p_camera->m_position += Quaternion::RotatePoint(p_camera->m_orientation, Vector3(0, -camStep * game.deltaTime, 0));
		}
		if (InputHandler.keyMapInfo.map[DC_W].isDown) {
			p_camera->m_position += Quaternion::RotatePoint(p_camera->m_orientation, Vector3(0, 0, camStep * game.deltaTime));
		}
		if (InputHandler.keyMapInfo.map[DC_S].isDown) {
			p_camera->m_position += Quaternion::RotatePoint(p_camera->m_orientation, Vector3(0, 0, -camStep * game.deltaTime));
		}

		//Rotations
		if (InputHandler.keyMapInfo.map[DC_R].isDown) {
			p_camera->m_orientation.Rotate(Quaternion(Vector3(1, 0, 0), camAngleRotation * game.deltaTime));
		}
		if (InputHandler.keyMapInfo.map[DC_F].isDown) {
			p_camera->m_orientation.Rotate(Quaternion(Vector3(1, 0, 0), -camAngleRotation * game.deltaTime));
		}
		if (InputHandler.keyMapInfo.map[DC_E].isDown) {
			Vector3 dir = Quaternion::RotatePoint(p_camera->m_orientation.Coniugated(), Vector3(0, 1, 0));
			p_camera->m_orientation.Rotate(Quaternion(dir, camAngleRotation * game.deltaTime));
		}
		if (InputHandler.keyMapInfo.map[DC_Q].isDown) {
			Vector3 dir = Quaternion::RotatePoint(p_camera->m_orientation.Coniugated(), Vector3(0, 1, 0));
			p_camera->m_orientation.Rotate(Quaternion(dir, -camAngleRotation * game.deltaTime));
		}
		if (InputHandler.keyMapInfo.map[DC_T].isDown) {
			Vector3 dir = Quaternion::RotatePoint(p_camera->m_orientation.Coniugated(), Vector3(0, 0, 1));
			p_camera->m_orientation.Rotate(Quaternion(dir, camAngleRotation * game.deltaTime));
		}
		if (InputHandler.keyMapInfo.map[DC_G].isDown) {
			Vector3 dir = Quaternion::RotatePoint(p_camera->m_orientation.Coniugated(), Vector3(0, 0, 1));
			p_camera->m_orientation.Rotate(Quaternion(dir, -camAngleRotation * game.deltaTime));
		}

		if (InputHandler.keyMapInfo.map[DC_P].isDown) {
			if (cicleTo_optionChange <= 0) {
				cicleTo_optionChange = 15;
				gpuSH.bloomEnabled = !gpuSH.bloomEnabled;
				p_camera->AssRotation(Quaternion(Vector3(1, 0, 0), 14 * PI / 144));
				p_camera->GoTo(Vector3(0, .57, -.44));
				p_camera->m_position += Quaternion::RotatePoint(p_camera->m_orientation, Vector3(0, 0, -camStep * 400));
				cudaStreamHandler.executionGraphCreated = false;
			}
		}

		if (InputHandler.keyMapInfo.map[DC_O].isDown) {
			if (cicleTo_optionChange <= 0) {
				cicleTo_optionChange = 10;
				b_GrassBlockSpinning = !b_GrassBlockSpinning;
			}
		}
		if (InputHandler.keyMapInfo.map[DC_I].isDown) {
			if (cicleTo_optionChange <= 0) {
				cicleTo_optionChange = 10;
				b_BobjSpinning = !b_BobjSpinning;
			}
		}
		if (InputHandler.keyMapInfo.map[DC_U].isDown) {
			if (cicleTo_optionChange <= 0) {
				cicleTo_optionChange = 10;
				if (gamma == 1.5f) { gamma = 2.2f; }
				else if (gamma == 2.2f) { gamma = 0.5f; }
				else { gamma = 1.5f; }
				cudaStreamHandler.executionGraphCreated = false;
			}
		}
		if (InputHandler.keyMapInfo.map[DC_L].isDown) {
			if (cicleTo_optionChange <= 0) {
				cicleTo_optionChange = 20;
				bloomHandler.ResetBloom();
				std::cout << "Reset bloom buffers avvenuto\n";
				cudaStreamHandler.executionGraphCreated = false;
			}
		}
	}
}