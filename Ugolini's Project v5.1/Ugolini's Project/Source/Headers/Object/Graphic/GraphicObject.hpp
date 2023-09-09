#pragma once
#include "..\Object.hpp"
#include "..\..\PipeLine_Manager\GPU_PipelineManager.cuh"


struct GraphicObject : Object {
    
    static GraphicObject* p_gObjects;
    static uint32 m_objectNumber;
    
    //--------------------------------------------------------------- Members
public:
    bool isValid = true;
    uint32 *p_verteciesId{ nullptr }, m_verteciesNumber = 0; //Array of index (used for the cpuSH.p_vertexArray)
    uint32 *p_trianglesId{ nullptr }, m_trianglesNumber = 0; //Array of index (used for the cpuSH.p_triangleIdArray) 
    uint32 *p_uvCoordinatesId{ nullptr }, m_uvCoordinatesNumber = 0; //Array of index (used for the cpuSH.p_uvCoordinatesArray)
    
    //--------------------------------------------------------------- Constructors
    GraphicObject(uint32 verteciesNum, uint32 trianglesNum, uint32 uvCoordinatesNum = 0) { //It allocate the right amount of mem, initialize everything to (0,0,0)
        m_verteciesNumber = verteciesNum; m_trianglesNumber = trianglesNum; m_uvCoordinatesNumber = uvCoordinatesNum;
        p_verteciesId = (uint32*)malloc(m_verteciesNumber * sizeof(uint32));
        p_trianglesId = (uint32*)malloc(m_trianglesNumber * sizeof(uint32));
        if(m_uvCoordinatesNumber != 0) p_uvCoordinatesId = (uint32*)malloc(m_uvCoordinatesNumber * sizeof(uint32));
        bool uvC_alloc = (!p_uvCoordinatesId) ^ (m_uvCoordinatesNumber == 0); //True if all good
        if ((!p_verteciesId) || (!p_trianglesId) || uvC_alloc) {
            std::cout << "Errore durante l'allocazione di un oggetto, attenderà 5 sec" << std::endl;
            crashHandler.crashCode = CH_CPU_ALLOCATION_ERROR;
            Sleep(5000);
            return;
        }
        ResizeVertexArray(cpuSH.m_vectors_number + verteciesNum);
        for (uint32 i = 0; i < m_verteciesNumber; i++) {
            p_verteciesId[i] = cpuSH.m_vectors_number;
            cpuSH.AddPoint(Vector3(0, 0, 0)); 
        }
        ResizeTriangleArray(cpuSH.m_triangles_number + trianglesNum);
        for (uint32 i = 0; i < m_trianglesNumber; i++) {
            p_trianglesId[i] = cpuSH.m_triangles_number;
            cpuSH.AddTriangle(TriangleId(0, 0, 0, 0, 0, 0));
        }
        ResizeTriangleArray(cpuSH.m_uv_coordinates_number + uvCoordinatesNum);
        for (uint32 i = 0; i < m_uvCoordinatesNumber; i++) {
            p_uvCoordinatesId[i] = cpuSH.m_uv_coordinates_number;
            cpuSH.AddUVcoordinate(Vector2(0, 0));
        }
        isValid = true;
    }
    ~GraphicObject() {
        if (!isValid) return;
        for (uint32 i = 0; i < m_verteciesNumber; i++) {
            cpuSH.RemovePoint(p_verteciesId[i]);
        }   free(p_verteciesId);
        for (uint32 i = 0; i < m_trianglesNumber; i++) {
            cpuSH.RemoveTriangle(p_trianglesId[i]);
        }   free(p_trianglesId);
        for (uint32 i = 0; i < m_uvCoordinatesNumber; i++) {
            cpuSH.RemoveUVcoordinate(p_uvCoordinatesId[i]);
        }   if(p_uvCoordinatesId) free(p_uvCoordinatesId);
    }

    void operator= (GraphicObject& gObj) {
        gObj.isValid = false;
        p_verteciesId = gObj.p_verteciesId;         m_verteciesNumber = gObj.m_verteciesNumber;
        p_trianglesId = gObj.p_trianglesId;         m_trianglesNumber = gObj.m_trianglesNumber;
        p_uvCoordinatesId = gObj.p_uvCoordinatesId; m_uvCoordinatesNumber = gObj.m_uvCoordinatesNumber;
    }

    //--------------------------------------------------------------- Static functions
#define MYMAOPT_GRAPHICOBJECTS_BUFFERLENGHT 50
    static void AddObject(uint32 verteciesNum, uint32 trianglesNum, uint32 uvCoordinatesNum = 0) {
        static uint32 m_objectBL = MYMAOPT_GRAPHICOBJECTS_BUFFERLENGHT;
        if (!p_gObjects) {
            p_gObjects = (GraphicObject*)malloc(m_objectBL * sizeof(GraphicObject));
        } else if (m_objectNumber >= m_objectBL){
            m_objectBL += MYMAOPT_GRAPHICOBJECTS_BUFFERLENGHT;
            p_gObjects = (GraphicObject*)realloc(p_gObjects, m_objectBL * sizeof(GraphicObject));
        }
        p_gObjects[m_objectNumber] = GraphicObject(verteciesNum, trianglesNum, uvCoordinatesNum);
        m_objectNumber++;
    }

    //--------------------------------------------------------------- Functions
    void MoveObject(Vector3 vect) {
        m_position += vect;
        for (uint32 i = 0; i < m_verteciesNumber; i++) {
            cpuSH.p_vertexArray[p_verteciesId[i]] += vect;
        }
    }
    void RotateObject(Quaternion rotation, Vector3 origin = Vector3(0,0,0)) {
        m_position = Quaternion::RotatePoint(rotation, m_position, origin);
        for (uint32 i = 0; i < m_verteciesNumber; i++) {
            cpuSH.p_vertexArray[p_verteciesId[i]] = Quaternion::RotatePoint(rotation, cpuSH.p_vertexArray[p_verteciesId[i]], origin);
        }
    }
};
GraphicObject* GraphicObject::p_gObjects{nullptr};
uint32 GraphicObject::m_objectNumber = 0;


struct ObjectCollection {
    uint32* m_objectIndexes; //This array will contain id that can be used to access an object in the GraphicObject::p_gObjects array
    uint32 m_objIdsBL = 10, m_objIds_number = 0;

    ObjectCollection() {
        m_objectIndexes = (uint32*)malloc(m_objIdsBL * sizeof(uint32));
    }
    ~ObjectCollection() {}

    void Resize_objIds(uint32 lenght) {
        m_objIdsBL = lenght;
        if (!m_objectIndexes) {
            m_objectIndexes = (uint32*)malloc(m_objIdsBL * sizeof(uint32)); return;
        }
        m_objectIndexes = (uint32*)realloc(m_objectIndexes, m_objIdsBL * sizeof(uint32));
    }
    void AddObj(uint32 id) {
        if (m_objIds_number >= m_objIdsBL) Resize_objIds(m_objIdsBL + 10);
        m_objectIndexes[m_objIds_number] = id;
        m_objIds_number++;
    }
   
    void MoveObject(Vector3 vect) {
        for (uint32 i = 0; i < m_objIds_number; i++) {
            GraphicObject::p_gObjects[m_objectIndexes[i]].MoveObject(vect);
        }
    }
    void RotateObject(Quaternion rotation, Vector3 origin = Vector3(0, 0, 0)) {
        for (uint32 i = 0; i < m_objIds_number; i++) {
            GraphicObject::p_gObjects[m_objectIndexes[i]].RotateObject(rotation, origin);
        }
    }
    void RotateObject(Vector3 axis, double angle , Vector3 origin = Vector3(0, 0, 0)) {
        RotateObject(Quaternion(axis, angle), origin);
    }
};
struct ObjectCollections {
    ObjectCollection* p_objCollections;
    uint32 m_objCollectionsBL = 10, m_objCollection_number = 0;
    int m_currentCollectionId = -1; //This will store the id (relative to the .m_objCollections) of the last collection

    ObjectCollections() {
        p_objCollections = (ObjectCollection*)malloc(m_objCollectionsBL * sizeof(ObjectCollection));
        for (uint32 i = 0; i < m_objCollectionsBL; i++) {
            p_objCollections[i] = ObjectCollection();
        }
    }
    ~ObjectCollections() {
        if (m_objCollection_number) free(p_objCollections);
    }

    ObjectCollection* CurrentObject() {
        return &(p_objCollections[m_currentCollectionId]);
    }
    ObjectCollection* MyGetObject(int id) { //if the id is less than 1, itll take start from the top of the array and proceed backwards
        if(id < 0) return &(p_objCollections[m_objCollection_number + id - 1]);
        return &(p_objCollections[id]);
    }
    void AddObjCollection() {
        if (m_objCollection_number >= m_objCollectionsBL) Resize_objCollections(m_objCollectionsBL + 10);
        m_objCollection_number++; m_currentCollectionId++;
    }
    void Resize_objCollections(uint32 lenght) {
        uint32 previousLenght = m_objCollectionsBL;
        m_objCollectionsBL = lenght;
        p_objCollections = (ObjectCollection*)realloc(p_objCollections, m_objCollectionsBL * sizeof(ObjectCollection));

        for (uint32 i = previousLenght; i < lenght; i++) {
            p_objCollections[i] = ObjectCollection();
        }
    }
};
ObjectCollections objectCollections;