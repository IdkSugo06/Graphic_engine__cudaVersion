#pragma once
#include "Material.hpp"

#define SET_BIT(flag,bitPos) (flag |= (1<<bitPos))
#define CLEAR_BIT(flag,bitPos) (flag &= ~(1<<bitPos))
#define CHECK_BIT(flag,bitPos) ((flag & (1<<bitPos)) == (1<<bitPos)) //if( CHECK_BIT(flag, int(0->(sizeof(flag) * 8))) )

#define TF_BP__VALIDITY 0 //stores the position of the bit of the flag of my gpu triangle struct 


struct GPU_Triangle {
    uint8 flag;

    Vector3 m_absSpace_points[3];
    Vector3 m_relToCam_points[3];
    Vector2 m_screenSpace_points[3]; //(0,1)

    Vector3 m_relToCam_points_interpolationInfo[3]; //(x/z,y/z,1/z)
    Vector2 m_uv_coordinates[3]; //(x/z,y/z)

    Vector3 m_relToCam_normal, m_relToCam_tangent, m_relToCam_bitangent;

    uint16 m_materialId;
    GPU_Triangle() {
        flag = 0;
        m_materialId = Material::m_defaultMaterialId;
    }
    ~GPU_Triangle() = default;

    __device__ void operator=(GPU_Triangle const* t) {
        m_absSpace_points[0] = t->m_absSpace_points[0];
        m_absSpace_points[1] = t->m_absSpace_points[1];
        m_absSpace_points[2] = t->m_absSpace_points[2];

        m_relToCam_points[0] = t->m_relToCam_points[0];
        m_relToCam_points[1] = t->m_relToCam_points[1];
        m_relToCam_points[2] = t->m_relToCam_points[2];

        m_relToCam_points_interpolationInfo[0] = t->m_relToCam_points_interpolationInfo[0];
        m_relToCam_points_interpolationInfo[1] = t->m_relToCam_points_interpolationInfo[1];
        m_relToCam_points_interpolationInfo[2] = t->m_relToCam_points_interpolationInfo[2];

        m_screenSpace_points[0] = t->m_screenSpace_points[0];
        m_screenSpace_points[1] = t->m_screenSpace_points[1];
        m_screenSpace_points[2] = t->m_screenSpace_points[2];

        m_uv_coordinates[0] = t->m_uv_coordinates[0];
        m_uv_coordinates[1] = t->m_uv_coordinates[1];
        m_uv_coordinates[2] = t->m_uv_coordinates[2];

        m_relToCam_normal = t->m_relToCam_normal;
        m_relToCam_tangent = t->m_relToCam_tangent;
        m_relToCam_bitangent = t->m_relToCam_bitangent;

        m_materialId = t->m_materialId;
    }
};
struct TriangleId {
    uint32 m_vertexIds[3]{ 0 };  //This id is used to point at the vertex in "vertexArray[id]"
    uint32 m_uvCoordinatesIds[3]{0};  //This id is used to point at the Vector2 in "uvCoordinatesArray[id]"
    uint16 m_materialId;

    //Constructors
    TriangleId(uint32 p1, uint32 p2, uint32 p3, uint32 uv1, uint32 uv2, uint32 uv3, uint16 m = Material::m_defaultMaterialId) {
        m_vertexIds[0] = p1; m_vertexIds[1] = p2; m_vertexIds[2] = p3;
        m_uvCoordinatesIds[0] = uv1; m_uvCoordinatesIds[1] = uv2; m_uvCoordinatesIds[2] = uv3;
        m_materialId = m;
    }
    TriangleId(uint32 p1, uint32 p2, uint32 p3, uint16 m = Material::m_defaultMaterialId) {
        m_vertexIds[0] = p1; m_vertexIds[1] = p2; m_vertexIds[2] = p3;
        m_materialId = m;
    }
    ~TriangleId() = default;

    //Functions
    static void PrintTid(TriangleId& tid) {
        std::cout << "v:  (" << tid.m_vertexIds[0] << "," << tid.m_vertexIds[1] << "," << tid.m_vertexIds[2] << ")\n";
        std::cout << "vt: (" << tid.m_uvCoordinatesIds[0] << "," << tid.m_uvCoordinatesIds[1] << "," << tid.m_uvCoordinatesIds[2] << ")\n";
        std::cout << "mtl: " << tid.m_materialId << "\n";
    }
};

#define GPU_RST_NUMOF_TRI_PER_MAXTHREADS 4 //has to be a uint8 (255 max)
#define RASTERIZZATION__SUBDIVSQUARE_NUMOF_ROW 16
#define RASTERIZZATION__SUBDIVSQUARE_NUMOF_COLOUMN 16
//RASTERIZZATION__SUBDIVSQUARE_NUMOF_ROW * RASTERIZZATION__SUBDIVSQUARE_NUMOF_COLOUMN has to be equal to MAX_THREAD_GPU / GPU_RST_NUMOF_THREADS_PER_TRI
 
struct GPU_subdividedRasterizzationInfo { //This struct will contain useful info for the new rasterizzation method (subdivide the triangle in multiple squares)
    bool isValid = false;
    uint16 minX, maxX, minY, maxY;
    float squareLenght, squareHeight;

    GPU_Triangle* tri;
    float Bx, By, Cx, Cy;
    float w1_xStep, w1_yStep, w2_xStep, w2_yStep;
    float startingW1, startingW2;

    GPU_subdividedRasterizzationInfo() = default;
    __device__ GPU_subdividedRasterizzationInfo(GPU_Triangle* _tri) {
        if (!_tri) return;
        tri = _tri;
        if (!CHECK_BIT(tri->flag, TF_BP__VALIDITY)) return;
        isValid = true;

        minX = myMin(myMin(tri->m_screenSpace_points[0].x, tri->m_screenSpace_points[1].x), tri->m_screenSpace_points[2].x);
        minY = myMin(myMin(tri->m_screenSpace_points[0].y, tri->m_screenSpace_points[1].y), tri->m_screenSpace_points[2].y);
        maxX = myMax(myMax(tri->m_screenSpace_points[0].x, tri->m_screenSpace_points[1].x), tri->m_screenSpace_points[2].x);
        maxY = myMax(myMax(tri->m_screenSpace_points[0].y, tri->m_screenSpace_points[1].y), tri->m_screenSpace_points[2].y);

        Bx = tri->m_screenSpace_points[1].x - tri->m_screenSpace_points[0].x;
        By = tri->m_screenSpace_points[1].y - tri->m_screenSpace_points[0].y;
        Cx = tri->m_screenSpace_points[2].x - tri->m_screenSpace_points[0].x;
        Cy = tri->m_screenSpace_points[2].y - tri->m_screenSpace_points[0].y;
        float invDenominator = 1.0f / (Bx * Cy - By * Cx);

        w1_xStep = Cy * invDenominator;
        w1_yStep = -(Cx * invDenominator);
        w2_xStep = -(By * invDenominator);
        w2_yStep = Bx * invDenominator;

        startingW1 = (((minX * Cy - minY * Cx) - (tri->m_screenSpace_points[0].x * Cy - tri->m_screenSpace_points[0].y * Cx))) * invDenominator;
        startingW2 = (-((minX * By - minY * Bx) - (tri->m_screenSpace_points[0].x * By - tri->m_screenSpace_points[0].y * Bx))) * invDenominator;
    }
    
    ~GPU_subdividedRasterizzationInfo() = default;

    __device__ void operator= (GPU_subdividedRasterizzationInfo const& subSquareInfo) {
        if (!subSquareInfo.tri) return;
        isValid = subSquareInfo.isValid;
        tri = subSquareInfo.tri;
        minX = subSquareInfo.minX; maxX = subSquareInfo.maxX; minY = subSquareInfo.minY; maxY = subSquareInfo.maxY;
        squareLenght = subSquareInfo.squareLenght; squareHeight = subSquareInfo.squareHeight;
        w1_xStep = subSquareInfo.w1_xStep; w1_yStep = subSquareInfo.w1_yStep; w2_xStep = subSquareInfo.w2_xStep; w2_yStep = subSquareInfo.w2_yStep;
        startingW1 = subSquareInfo.startingW1; startingW2 = subSquareInfo.startingW2;
    }

    __device__ void Create(GPU_Triangle* _tri, Fragment* p_fragment) {
        if (!_tri) return;
        tri = _tri;
        if (!CHECK_BIT(tri->flag, TF_BP__VALIDITY)) return;
        isValid = true;

        minX = myMin(myMin(tri->m_screenSpace_points[0].x, tri->m_screenSpace_points[1].x), tri->m_screenSpace_points[2].x);
        minY = myMin(myMin(tri->m_screenSpace_points[0].y, tri->m_screenSpace_points[1].y), tri->m_screenSpace_points[2].y);
        maxX = myMax(myMax(tri->m_screenSpace_points[0].x, tri->m_screenSpace_points[1].x), tri->m_screenSpace_points[2].x);
        maxY = myMax(myMax(tri->m_screenSpace_points[0].y, tri->m_screenSpace_points[1].y), tri->m_screenSpace_points[2].y);

        squareLenght = (maxX - minX) * (1.0f / RASTERIZZATION__SUBDIVSQUARE_NUMOF_COLOUMN);
        squareHeight = (maxY - minY) * (1.0f / RASTERIZZATION__SUBDIVSQUARE_NUMOF_ROW);

        Bx = tri->m_screenSpace_points[1].x - tri->m_screenSpace_points[0].x;
        By = tri->m_screenSpace_points[1].y - tri->m_screenSpace_points[0].y;
        Cx = tri->m_screenSpace_points[2].x - tri->m_screenSpace_points[0].x;
        Cy = tri->m_screenSpace_points[2].y - tri->m_screenSpace_points[0].y;
        float invDenominator = 1.0f / (Bx * Cy - By * Cx);

        w1_xStep = Cy * invDenominator;
        w1_yStep = -(Cx * invDenominator);
        w2_xStep = -(By * invDenominator);
        w2_yStep = Bx * invDenominator;

        startingW1 = (((minX * Cy - minY * Cx) - (tri->m_screenSpace_points[0].x * Cy - tri->m_screenSpace_points[0].y * Cx))) * invDenominator;
        startingW2 = (-((minX * By - minY * Bx) - (tri->m_screenSpace_points[0].x * By - tri->m_screenSpace_points[0].y * Bx))) * invDenominator;
    }
};