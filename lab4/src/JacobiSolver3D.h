#pragma once

#include <mpi.h>
#include <vector>

typedef std::vector<float> fvector;

class JacobiSolver3D {
 public:
    JacobiSolver3D(int rank, int size);
    float solve();

 private:
    void distributeLayers();
    void initBuffers();

    float calcCore();
    float calcBorder();

    void isendrecvBorders();
    void waitBorders();

    float phi(int i, int j, int k) const;
    float rho(int i, int j, int k) const;

    int getBorderIdx(int j, int k) const;
    float& curr(int i, int j, int k);
    float& prev(int i, int j, int k);

    const int procRank, commSize;
    const float x_0, y_0, z_0;
    const float D_x, D_y, D_z;
    const float parA, eps;
    const int N_x, N_y, N_z;
    const float h_x, h_y, h_z;
    const float h_x2, h_y2, h_z2;
    const float coef;

    int locN_x, locOffset;
    fvector prevBuffer, currBuffer;
    fvector topBorder, bottomBorder;

    MPI_Request reqSendTop, reqSendBottom;
    MPI_Request reqRecvTop, reqRecvBottom;
};
