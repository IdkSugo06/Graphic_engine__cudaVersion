#pragma once
#include "..\PipeLine_Manager\GPU_PipelineManager.cuh"
#include "..\Object\Graphic\GraphicObject.hpp"

#define DEBUG_SHOW_TXT_NAME_WHEN_LOADED false

struct Path {
    char m_path[FILE_FILEDIRECTORY_BUFFER_LENGHT];

    Path(const char* path) {
        uint32 i = 0;
        while (path[i] != '\0') {
            m_path[i] = path[i];
            i++;
        } m_path[i] = '\0';
    }
    void operator= (Path path) {
        uint32 i = 0;
        while (path.m_path[i] != '\0') {
            m_path[i] = path.m_path[i];
            i++;
        } m_path[i] = '\0';
    }

    bool CheckPath(const char* path) {
        uint32 i = 0;
        while (m_path[i] != '\0') {
            if (m_path[i] != path[i]) return false;
            i++;
        }
        if (path[i] == '\0') return true; //path to check could be longer (m_path = walk; path = walking)
        return false;
    }
};

struct MtllibStorage {
    Mtllib* p_mtllibs{ nullptr }; Path* p_paths;
    uint16 m_mtllibs_number = 0, m_mtllibsBL = 10;

    MtllibStorage() = default;
    ~MtllibStorage() {
        if (p_paths != nullptr) { free(p_paths); }
        if (p_mtllibs != nullptr) {
            for (uint16 i = 0; i < m_mtllibs_number; i++) {
                p_mtllibs[i].Deallocate();
            }
        }
    }

    Mtllib* AddMtllib(const char* path, Mtllib mtllib = Mtllib()) { //HAS TO BE EXECUTED AFTER THE gpuSH.LoadMaterial()
        if (!p_paths) {
            p_paths = (Path*)malloc(m_mtllibsBL * sizeof(Path));
            if (!p_paths) {
                std::cout << "[ERRORE]: allocazione memoria fallita (CPU)\n";
                crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR;
                return nullptr;
            }
        }
        if (!p_mtllibs) {
            p_mtllibs = (Mtllib*)malloc(m_mtllibsBL * sizeof(Mtllib));
            if (!p_mtllibs) {
                std::cout << "[ERRORE]: allocazione memoria fallita (CPU)\n";
                crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR;
                return nullptr;
            }
        }

        if (m_mtllibs_number >= m_mtllibsBL) {
            p_paths = (Path*)realloc(p_paths, m_mtllibsBL * sizeof(Path));
            p_mtllibs = (Mtllib*)realloc(p_mtllibs, m_mtllibsBL * sizeof(Mtllib));
            m_mtllibs_number += 10;
        }

        p_paths[m_mtllibs_number] = Path(path);
        p_mtllibs[m_mtllibs_number] = mtllib;
        m_mtllibs_number++;
        return &(p_mtllibs[m_mtllibs_number - 1]);
    }
    Mtllib* SearchMtllib(const char* path) { //Return the pointer to the mtllib, nullptr if not found
        for (uint16 i = 0; i < m_mtllibs_number; i++) {
            if (p_paths[i].CheckPath(path)) return &(p_mtllibs[i]);
        }
        return nullptr;
    }
};
MtllibStorage mtllibStorage;

//.MTL FILES
Material ReadMtl(std::fstream& fin, char* file_directory, char* charBuffer) { //It will read until the blank line
	bool txtMap = false;
    bool normalMap = false;
	char txtMap_filePath[FILE_FILEDIRECTORY_BUFFER_LENGHT]; //Will store the absolute path (directory + name)
    char normalMap_filePath[FILE_FILEDIRECTORY_BUFFER_LENGHT];

    RgbVector kd, ks, ke; float specularExponent = 1; float normalMapScale = 1;
	bool running = true; 

	myFgets(fin, charBuffer, 1);
	while (running) {

		if (charBuffer[0] == '\n') {
			running = false;
            if (txtMap && !normalMap) return Material(kd, ks, specularExponent, ke, txtMap_filePath);
            if (!txtMap && normalMap) return Material(kd, ks, specularExponent, ke, nullptr, normalMap_filePath);
            if (txtMap && normalMap) return Material(kd, ks, specularExponent, ke, txtMap_filePath, normalMap_filePath);
            return Material(kd, ks, specularExponent, ke);
		}
		else {
			if (charBuffer[0] == 'N') {
				myFgets(fin, charBuffer, 2);
				if (charBuffer[0] == 's' && (charBuffer[1] == ' ' || charBuffer[1] == '\t')) { //Ns (specular exponent)
					specularExponent = ReadFloat(fin, charBuffer) / MATERIAL_SPECULAREXPONENT_COEFFREDUCTION;
					myFgets(fin, charBuffer, 1);
				}
				else if (charBuffer[0] == 'i' && (charBuffer[1] == ' ' || charBuffer[1] == '\t')) { //Ni, optical density
					myFendline(fin);
					myFgets(fin, charBuffer, 1);
				}
				else {
					myFendline(fin);
					myFgets(fin, charBuffer, 1);
				}
			}
			else if (charBuffer[0] == 'K') {
				myFgets(fin, charBuffer, 2);

				if (charBuffer[0] == 'e' && (charBuffer[1] == ' ' || charBuffer[1] == '\t')) { //ke, emissive
					ke.red = ReadFloat(fin, charBuffer);
					ke.green = ReadFloat(fin, charBuffer);
					ke.blue = ReadFloat(fin, charBuffer);
					myFgets(fin, charBuffer, 1);
				}
				else if (charBuffer[0] == 's' && (charBuffer[1] == ' ' || charBuffer[1] == '\t')) { //ke, specular
					ks.red = ReadFloat(fin, charBuffer);
					ks.green = ReadFloat(fin, charBuffer);
					ks.blue = ReadFloat(fin, charBuffer);
					myFgets(fin, charBuffer, 1);
				}
                else if (charBuffer[0] == 'd' && (charBuffer[1] == ' ' || charBuffer[1] == '\t')) { //ke, specular
                    kd.red = ReadFloat(fin, charBuffer);
                    kd.green = ReadFloat(fin, charBuffer);
                    kd.blue = ReadFloat(fin, charBuffer);
                    myFgets(fin, charBuffer, 1);
                }
				else if (charBuffer[0] == 'a' && (charBuffer[1] == ' ' || charBuffer[1] == '\t')) {//ka, ambient color
					myFendline(fin);
					myFgets(fin, charBuffer, 1);
				}
				else {
					myFendline(fin);
					myFgets(fin, charBuffer, 1);
				}
			}
			else if (charBuffer[0] == 'm') {
				myFgets(fin, charBuffer, 3);
				if (charBuffer[0] == 'a' && charBuffer[1] == 'p' && charBuffer[2] == '_') { //Map
                    myFgets(fin, charBuffer, 3);
                    if (charBuffer[0] == 'K' && charBuffer[1] == 'd' && (charBuffer[2] == ' ' || charBuffer[2] == '\t')) { //Texture

                        //Copy the directory into the txtMap_filePath
                        uint16 i = 0; //will represent the lenght
                        while (file_directory[i] != '\0' && (i < FILE_FILEDIRECTORY_BUFFER_LENGHT - 1)) {
                            txtMap_filePath[i] = file_directory[i];
                            i++;
                        }

                        //Check if the buffer has been exceeded
                        if (file_directory[i] != '\0') {
                            std::cout << "[AVVERTENZA]: Un percorso file ha superato il buffer prestabilito (di lunghezza " << FILE_FILEDIRECTORY_BUFFER_LENGHT << "), attenderà 2.5 secondi\n";
                            Sleep(2500);
                        }

                        //No error found
                        else {
                            txtMap = true;

                            //Read the texture map name
                            myFgets(fin, charBuffer, 1);
                            while (charBuffer[0] != '\n' && (i < FILE_FILENAME_BUFFER_LENGHT - 1)) {
                                txtMap_filePath[i] = charBuffer[0];
                                myFgets(fin, charBuffer, 1);
                                i++;
                            }
                            txtMap_filePath[i] = '\0';
                            if (DEBUG_SHOW_TXT_NAME_WHEN_LOADED) std::cout << "Texture map loaded: " << txtMap_filePath << "\n";

                            //Check if the buffer has been exceeded
                            if (charBuffer[0] != '\n') {
                                std::cout << "[AVVERTENZA]: Un nome file ha superato il buffer prestabilito (di lunghezza " << FILE_FILENAME_BUFFER_LENGHT << "), attenderà 2.5 secondi\n";
                                Sleep(2500);

                                myFendline(fin);
                                myFgets(fin, charBuffer, 1);
                            }
                            myFgets(fin, charBuffer, 1);
                        }
                    }else if (charBuffer[0] == 'B' && charBuffer[1] == 'u' && charBuffer[2] == 'm') { //Normal map
                        myFgets(fin, charBuffer, 2);
                        if (charBuffer[0] == 'p' && (charBuffer[1] == ' ' || charBuffer[1] == '\t')) {
                            myFgets(fin, charBuffer, 1); 
                            if (charBuffer[0] == '-') {
                                myFgets(fin, charBuffer, 3); //read 'bm '                             
                                normalMapScale = ReadFloat(fin, charBuffer); //read the float + ' '
                                myFgets(fin, charBuffer, 1); //read the (' ' or '\t') and load the first letter of the map in the charBuffer
                            }
                            else { normalMapScale = 1; }

                            //Copy the directory into the txtMap_filePath
                            uint16 i = 0; //will represent the lenght
                            while (file_directory[i] != '\0' && (i < FILE_FILEDIRECTORY_BUFFER_LENGHT - 1)) {
                                normalMap_filePath[i] = file_directory[i];
                                i++;
                            }

                            //Check if the buffer has been exceeded
                            if (file_directory[i] != '\0') {
                                std::cout << "[AVVERTENZA]: Un percorso file ha superato il buffer prestabilito (di lunghezza " << FILE_FILEDIRECTORY_BUFFER_LENGHT << "), attenderà 2.5 secondi\n";
                                Sleep(2500);
                            }

                            //No error found
                            else {
                                normalMap = true;

                                //Read the texture map name
                                //myFgets(fin, charBuffer, 1); //first char already red (in order to see if the -bm num is specified)
                                while (charBuffer[0] != '\n' && (i < FILE_FILENAME_BUFFER_LENGHT - 1)) {
                                    normalMap_filePath[i] = charBuffer[0];
                                    myFgets(fin, charBuffer, 1);
                                    i++;
                                }
                                normalMap_filePath[i] = '\0';
                                if(DEBUG_SHOW_TXT_NAME_WHEN_LOADED) std::cout << "Normal map loaded: " << normalMap_filePath << "\n";

                                //Check if the buffer has been exceeded
                                if (charBuffer[0] != '\n') {
                                    std::cout << "[AVVERTENZA]: Un nome file ha superato il buffer prestabilito (di lunghezza " << FILE_FILENAME_BUFFER_LENGHT << "), attenderà 2.5 secondi\n";
                                    Sleep(2500);

                                    myFendline(fin);
                                    myFgets(fin, charBuffer, 1);
                                }
                                myFgets(fin, charBuffer, 1);
                            }
                        }
                    }
				}
			}
			else {
                myFendline(fin); charBuffer[0] = '\n';
				myFgets(fin, charBuffer, 1);
			}
		}
	}
}
Mtllib* ReadMtllib(const char* file_path, uint16 numOf_materialAlreadyLoaded = gpuSH.m_materials_number, char* file_directory = nullptr) { //! CAREFUL, the file has to end with a '\n'
	//file_directory is not necessary, although it will save some computations
    Mtllib* p_mtllib = mtllibStorage.SearchMtllib(file_path);
    if (p_mtllib != nullptr) return p_mtllib;
    p_mtllib = mtllibStorage.AddMtllib(file_path);
    if (p_mtllib == nullptr) return nullptr;

	if (!file_directory) {
		file_directory = (char*)malloc(sizeof(char) * FILE_FILEDIRECTORY_BUFFER_LENGHT);
		if (!file_directory) {
			std::cout << "[ERRORE]: Durante l'allocazione di memoria (CPU), attenderà 5 sec" << std::endl;
			crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR;
			Sleep(5000);
			return p_mtllib;
		}
		uint16 i = 0;
		while (file_path[i] != '\0' && i < FILE_FILEDIRECTORY_BUFFER_LENGHT) {
			file_directory[i] = file_path[i];
			i++;
		}
		while (file_directory[i] != '\\') {
			file_directory[i] = '\0';
			i--;
		}
	}

	std::fstream fin;
	fin.open(file_path, std::ios::in | std::ios::binary);
	char charBuffer[FILE_CHAR_BUFFER_LENGHT];

	if (!fin) {//Se il file non si è aperto
		std::cout << "Errore nella lettura del file mtl specificato (\"" << file_path << "\"), attenderà 5 sec" << std::endl;
		crashHandler.crashCode = CH_FILE_NAME_NOT_FOUND;
		Sleep(5000);
		return p_mtllib;
	}

	bool running = true;
	char tempTag[MATERIAL_TAG_BUFFER];

	myFgets(fin, charBuffer, 1);
	while (running) {
        //std::cout << charBuffer[0] << "\n";
		//EOF check, when a file has come to the end, the fstream.read(charBuffer) function will not modify the charBuffer, that means that the char will remain the last read
		//So knowing that the mtl file has to finish with a '\n', we can check id it reamains the same for n times
		//If that function return three or more '\n', we know that the file's finished
		if (charBuffer[0] == '\n') {
			myFgets(fin, charBuffer, 1);
			if (charBuffer[0] == '\n') {
				myFgets(fin, charBuffer, 1);
				if (charBuffer[0] == '\n') {
					running = false; break;
				}
			}
		}
		else if (charBuffer[0] == 'n') {
			myFgets(fin, charBuffer, 6);
			if (charBuffer[0] == 'e' && charBuffer[1] == 'w' && charBuffer[2] == 'm' && charBuffer[3] == 't' && charBuffer[4] == 'l' && (charBuffer[5] == ' ' || charBuffer[5] == '\t')) {
				//Read the tag
				uint16 i = 0; //will represent the lenght
				myFgets(fin, charBuffer, 1);
				while (charBuffer[0] != '\n' && (i < MATERIAL_TAG_BUFFER - 1)) {
					tempTag[i] = charBuffer[0];
					myFgets(fin, charBuffer, 1);
					i++;
				}
				tempTag[i] = '\0';

				//Check for errors
				if (charBuffer[0] != '\n') {
					std::cout << "[AVVERTENZA]: Un tag di un mtl ha superato il buffer prestabilito (di lunghezza " << MATERIAL_TAG_BUFFER << "), attenderà 2.5 secondi\n";
					Sleep(2500);
					myFendline(fin); charBuffer[0] = '\n';
				}

                Material m = ReadMtl(fin, file_directory, charBuffer);
                cpuSH.LoadInTempMaterial(m);
                gpuSH.CopyTempMaterial(cudaStreamHandler.mainStream, true);
                p_mtllib->AddTag(Mtltag(tempTag, i), gpuSH.m_materials_number - 1);
			}
			else { 
                myFendline(fin); charBuffer[0] = '\n'; myFgets(fin, charBuffer, 1);
            }
		}
		else {
			myFendline(fin); charBuffer[0] = '\n';
			myFgets(fin, charBuffer, 1);
		}
	}
	return p_mtllib;
}
void Material::UploadMtllib(const char* mtl_filePath) {
    ReadMtllib(mtl_filePath);
}

//.OBJ FILES
void ReadFaceVertex(std::fstream& fin, char* charBuffer, char* str_number, uint8 verticesNum, uint32* p_vertices, uint32* p_txtVertices = nullptr) { //The next char read has to be the first digit, the next char to read is the one after the ' ' or '\t' or '\n' (charBuffer[0] = '\n')
    uint32 v, vt, vn;

    uint8 i = 0;
    myFgets(fin, (char*)charBuffer, 1);
    while (charBuffer[0] != '\\' && charBuffer[0] != '/' && charBuffer[0] != ' ' && charBuffer[0] != '\t' && charBuffer[0] != '\0' && charBuffer[0] != '\n') {
        str_number[i] = charBuffer[0];
        myFgets(fin, charBuffer, 1);
        i++;
    }str_number[i] = '\0';
    v = atof(str_number);

    i = 0;
    myFgets(fin, (char*)charBuffer, 1);
    while (charBuffer[0] != '\\' && charBuffer[0] != '/' && charBuffer[0] != ' ' && charBuffer[0] != '\t' && charBuffer[0] != '\0' && charBuffer[0] != '\n') {
        str_number[i] = charBuffer[0];
        myFgets(fin, charBuffer, 1);
        i++;
    }str_number[i] = '\0';
    vt = atof(str_number);

    i = 0;
    myFgets(fin, (char*)charBuffer, 1);
    while (charBuffer[0] != '\\' && charBuffer[0] != '/' && charBuffer[0] != ' ' && charBuffer[0] != '\t' && charBuffer[0] != '\0' && charBuffer[0] != '\n') {
        str_number[i] = charBuffer[0];
        myFgets(fin, charBuffer, 1);
        i++;
    }str_number[i] = '\0';
    vn = atof(str_number);

    p_vertices[verticesNum] = v;
    if (p_txtVertices) {
        p_txtVertices[verticesNum] = vt;
    }
}
void ReadFace(std::fstream& fin, char* charBuffer, uint32* p_vertices, uint32* p_txtVertices = nullptr) {//The next char read has to be the first digit, the next char to read is the one after the ' ' or '\t' or '\n' (charBuffer[0] = '\n')
    char str_number[FILE_STR_TO_FLOAT_BUFFER_LENGHT];

    ReadFaceVertex(fin, charBuffer, str_number, 0, p_vertices, p_txtVertices);
    ReadFaceVertex(fin, charBuffer, str_number, 1, p_vertices, p_txtVertices);
    ReadFaceVertex(fin, charBuffer, str_number, 2, p_vertices, p_txtVertices);
}

bool ReadOBJfile(const char* file_path, bool debug = false) {

    //Variables for 2nd phase
    uint16 currentMtlId = 0; //It'll store the id of the gpuSH.p_materials array of the last material read, (it'll be used in order to initialize a triangle)
    uint16 currentObjId = GraphicObject::m_objectNumber - 1; //It'll store the id of the GraphicObject::p_gObjects array of the last object read, (it'll be used in order to initialize the right attributes)
    //Those variables will be used to compute the absolute id (in order to access at the arrays in the cpuSH struct)
    uint32 numOfVerticesPreviouslyPresent = cpuSH.m_vectors_number;
    uint32 numOfTrianglesPreviouslyPresent = cpuSH.m_triangles_number;
    uint32 numOfUVcoordinatesPreviouslyPresent = cpuSH.m_uv_coordinates_number;

    // TRANSLATE DIRECTORY PHASE ------------------------------------------------------------------------------------------------------------- TRANSLATE DIRECTORY PHASE
    //Compute the file directory
    char file_directory[FILE_FILEDIRECTORY_BUFFER_LENGHT]; //File obj directory
    uint16 fileDir_lenght = 0;
    while (file_path[fileDir_lenght] != '\0' && fileDir_lenght < FILE_FILEDIRECTORY_BUFFER_LENGHT) {
        file_directory[fileDir_lenght] = file_path[fileDir_lenght];
        fileDir_lenght++;
    }
    while (file_directory[fileDir_lenght] != '\\' && file_directory[fileDir_lenght] != '/') {
        file_directory[fileDir_lenght] = '\0';
        fileDir_lenght--;
    }

    //Copy the file directory into the mtllib path (useful to read any .mtl file)
    char mtllib_path[FILE_FILEDIRECTORY_BUFFER_LENGHT]; //File obj directory + file mtl name
    fileDir_lenght = 0;
    while (file_directory[fileDir_lenght] != '\0' && fileDir_lenght < FILE_FILEDIRECTORY_BUFFER_LENGHT) {
        mtllib_path[fileDir_lenght] = file_directory[fileDir_lenght];
        fileDir_lenght++;
    }
    mtllib_path[fileDir_lenght] = '\0';

#pragma region 1st phase
    // READING FILE | FIRST PHASE -------------------------------------------------------------------------------------------------------------  READING FILE | FIRST PHASE
    //This phase will count the vertecies, faces and txtVertecies
    //It will also create the gObjects and allocate all the attributes found
    //The attributes allocated will not be initialized

    //Opne the file stream
    std::fstream fin;
    fin.open(file_path, std::ios::in | std::ios::binary);
    char charBuffer[FILE_CHAR_BUFFER_LENGHT];

    if (!fin) {//Se il file non si è aperto
        std::cout << "Errore nel caricamento del file OBJ (\"" << file_path << "\"), attenderà 5 sec" << std::endl;
        crashHandler.crashCode = CH_FILE_NAME_NOT_FOUND;
        Sleep(5000);
        return false;
    }

    //Variables
    uint16 objectCount = 0, mtlCount = 0;
    uint32 vertexCount = 0, txtVertexCount = 0, facesCount = 0;

    //Stores the "perObject" attributes (they'll be reset every new object created)
    uint32 objVertexCount = 0, objTxtVertexCount = 0, objFacesCount = 0;
    Mtllib* p_mtllib = nullptr;
    objectCollections.AddObjCollection();
    ObjectCollection* currentObjCollection = objectCollections.CurrentObject();

    //Code
    bool running = true;
    myFgets(fin, charBuffer, 1);
    while (running) {//Object counting

        //If it reads 3 '\n' in a row it stops (it reads the last char of the file multiple times when the file finishes), if not it just repeat the cicle while having the char read in the charBuffer[0]
        if (charBuffer[0] == '\n') {
            myFgets(fin, charBuffer, 1);
            if (charBuffer[0] == '\n') {
                myFgets(fin, charBuffer, 1);
                if (charBuffer[0] == '\n') {
                    running = false; break;
                }
            }
        }
        else if (charBuffer[0] == 'o') { //      "o" found
            if (objectCount != 0) { //Create the object
                GraphicObject::AddObject(objVertexCount, objFacesCount, objTxtVertexCount);
                currentObjCollection->AddObj(GraphicObject::m_objectNumber - 1);
                objVertexCount = 0; objTxtVertexCount = 0; objFacesCount = 0;
            }
            objectCount++;
            myFendline(fin); charBuffer[0] = '\n';
            myFgets(fin, charBuffer, 1);
        }
        else if (charBuffer[0] == 'v') { //      "v" found
            myFgets(fin, charBuffer, 1);
            if (charBuffer[0] == ' ' || charBuffer[0] == '\t') {
                vertexCount++; objVertexCount++;
            }
            if (charBuffer[0] == 't') {
                txtVertexCount++; objTxtVertexCount++;
            }
            if (charBuffer[0] == 'n') {
                //normals not implemented in my solution
            }
            myFendline(fin); charBuffer[0] = '\n';
            myFgets(fin, charBuffer, 1);
        }
        else if (charBuffer[0] == 'f') { //      "f" found 
            myFgets(fin, charBuffer, 1);
            facesCount++; objFacesCount++;
            //for (uint8 i = 0; i < 3; i++) //To check whether or not they are tri or quad faces (yet to be implemented)
            myFendline(fin); charBuffer[0] = '\n';
            myFgets(fin, charBuffer, 1);
        }
        else if (charBuffer[0] == 'u') { //      "usemtl"
            myFgets(fin, charBuffer, 5);
            if (charBuffer[0] == 's' && charBuffer[1] == 'e' && charBuffer[2] == 'm' && charBuffer[3] == 't' && charBuffer[4] == 'l') {
                mtlCount++;
            }
            myFendline(fin); charBuffer[0] = '\n';
            myFgets(fin, charBuffer, 1);
        }
        else if (charBuffer[0] == 'm') { //     "mtllib" 
            myFgets(fin, charBuffer, 6);
            if (charBuffer[0] == 't' && charBuffer[1] == 'l' && charBuffer[2] == 'l' && charBuffer[3] == 'i' && charBuffer[4] == 'b' && (charBuffer[5] == ' ' || charBuffer[5] == '\t')) {
                myFgets(fin, charBuffer, 1);
                uint16 i = fileDir_lenght;
                while (charBuffer[0] != '\n') {
                    mtllib_path[i] = charBuffer[0];
                    myFgets(fin, charBuffer, 1);
                    i++;
                }
                mtllib_path[i] = '\0';
            }
            p_mtllib = ReadMtllib(mtllib_path, gpuSH.m_materials_number, file_directory);
            if (p_mtllib == nullptr) return false;
            myFgets(fin, charBuffer, 1);
        }
        else {
            myFendline(fin); charBuffer[0] = '\n';
            myFgets(fin, charBuffer, 1);
        }
    }
    GraphicObject::AddObject(objVertexCount, objFacesCount, objTxtVertexCount);
    currentObjCollection->AddObj(GraphicObject::m_objectNumber - 1);
    
    fin.close();
    if (debug) {
        std::cout << "Counting finished\n";
        std::cout << "\tObject found: " << (int)objectCount << "\n";
        std::cout << "\tMtl found: " << (int)mtlCount << "\n";
        std::cout << "\tVertex (absSpace) found: " << (int)vertexCount << "\n";
        std::cout << "\tVertex (txtSpace) found: " << (int)txtVertexCount << "\n";
        std::cout << "\tTriangles found: " << (int)facesCount << "\n";
    }
#pragma endregion 1st phase

#pragma region 2nd phase
    // READING FILE | SECOND PHASE -------------------------------------------------------------------------------------------------------------  READING FILE | SECOND PHASE
    //This phase will instead initialize all the already allocated attributes

    //Opne the file stream
    fin.open(file_path, std::ios::in | std::ios::binary);
    if (!fin) {//Se il file non si è aperto
        std::cout << "Errore nel caricamento del file OBJ (\"" << file_path << "\"), attenderà 5 sec" << std::endl;
        crashHandler.crashCode = CH_FILE_NAME_NOT_FOUND;
        Sleep(5000);
        return false;
    }

    //"perObj" attributes
    objVertexCount = 0; objTxtVertexCount = 0; objFacesCount = 0;
    vertexCount = 0, txtVertexCount = 0, facesCount = 0;

    //Code
    running = true;
    myFgets(fin, charBuffer, 1);
    while (running) {
        //If it reads 3 '\n' in a row it stops (when the file finishes, the read function does not change the charBuffer, so it should remain the last char, which is supposed to be '\n')
        if (charBuffer[0] == '\n') {
            myFgets(fin, charBuffer, 1);
            if (charBuffer[0] == '\n') {
                myFgets(fin, charBuffer, 1);
                if (charBuffer[0] == '\n') {
                    running = false; break;
                }
            }
        }
        else if (charBuffer[0] == 'o') { //      "o" found
            objVertexCount = 0; objTxtVertexCount = 0; objFacesCount = 0;
            currentObjId++;
            myFendline(fin); charBuffer[0] = '\n';
            myFgets(fin, charBuffer, 1);
        }
        else if (charBuffer[0] == 'v') { //      "v" found
            myFgets(fin, charBuffer, 1);
            if (charBuffer[0] == ' ' || charBuffer[0] == '\t') { //Vertex
                uint32 vertexId = vertexCount + numOfVerticesPreviouslyPresent;
                GraphicObject* obj = &(GraphicObject::p_gObjects[currentObjId]);
                obj->p_verteciesId[objVertexCount] = vertexId;
                cpuSH.p_vertexArray[vertexId] = ReadVector3(fin, charBuffer);
                if (charBuffer[0] != '\n') {//obj vertices could have a fourth component, which will be ignored
                    myFendline(fin); charBuffer[0] = '\n';
                }
                objVertexCount++; vertexCount++;
            }
            else if (charBuffer[0] == 't') { //Txt vertex
                myFgets(fin, charBuffer, 1);
                uint32 txtVertexId = txtVertexCount + numOfUVcoordinatesPreviouslyPresent;
                GraphicObject::p_gObjects[currentObjId].p_uvCoordinatesId[objTxtVertexCount] = txtVertexId;
                cpuSH.p_uvCoordinatesArray[txtVertexId] = Vector3(ReadVector2(fin, charBuffer), 0);
                if (charBuffer[0] != '\n') {//obj txtVertices could have a third component, which will be ignored
                    myFendline(fin); charBuffer[0] = '\n';
                }
                objTxtVertexCount++; txtVertexCount++;
            }
            else {
                myFendline(fin); charBuffer[0] = '\n';
                myFgets(fin, charBuffer, 1);
            }
        }
        else if (charBuffer[0] == 'f') { //      "f" found 
            myFgets(fin, charBuffer, 1);
            uint32 v[3], vt[3];
            ReadFace(fin, charBuffer, v, vt);

            v[0] += numOfVerticesPreviouslyPresent - 1; v[1] += numOfVerticesPreviouslyPresent - 1; v[2] += numOfVerticesPreviouslyPresent - 1;
            vt[0] += numOfUVcoordinatesPreviouslyPresent - 1; vt[1] += numOfUVcoordinatesPreviouslyPresent - 1; vt[2] += numOfUVcoordinatesPreviouslyPresent - 1;
            uint32 triangleId = facesCount + numOfTrianglesPreviouslyPresent;
            GraphicObject::p_gObjects[currentObjId].p_trianglesId[objFacesCount] = triangleId;
            cpuSH.p_triangleIdArray[triangleId] = TriangleId(v[0], v[1], v[2], vt[0], vt[1], vt[2], currentMtlId);

            objFacesCount++; facesCount++;
            if (charBuffer[0] != '\n') {//just to be sure
                myFendline(fin); charBuffer[0] = '\n';
            }
            myFgets(fin, charBuffer, 1);
        }
        else if (charBuffer[0] == 'u') { //      "usemtl"
            myFgets(fin, charBuffer, 6);
            if (charBuffer[0] == 's' && charBuffer[1] == 'e' && charBuffer[2] == 'm' && charBuffer[3] == 't' && charBuffer[4] == 'l' && (charBuffer[5] == ' ' || charBuffer[5] == '\t')) {
                char mtltag[MATERIAL_TAG_BUFFER];
                uint8 i = 0;
                myFgets(fin, charBuffer, 1);
                while (charBuffer[0] != '\n' && i < MATERIAL_TAG_BUFFER) {
                    mtltag[i] = charBuffer[0];
                    myFgets(fin, charBuffer, 1);
                    i++;
                } mtltag[i] = '\0';
                currentMtlId = p_mtllib->GetId(mtltag);
                myFgets(fin, charBuffer, 1);
            }
            else {
                myFendline(fin); charBuffer[0] = '\n';
                myFgets(fin, charBuffer, 1);
            }
        }
        else {
            myFendline(fin); charBuffer[0] = '\n';
            myFgets(fin, charBuffer, 1);
        }
    }
#pragma endregion 2nd phase
    return true;
}