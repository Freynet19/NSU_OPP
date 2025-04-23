#include "JacobiSolver3D.h"

#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>

JacobiSolver3D::JacobiSolver3D(int rank, int size)
: procRank(rank), commSize(size),
x_0(-1), y_0(-1), z_0(-1),
D_x(2), D_y(2), D_z(2),
parA(1e5), eps(1e-6),
N_x(400), N_y(400), N_z(400),
h_x(D_x / (N_x - 1)), h_y(D_y / (N_y - 1)), h_z(D_z / (N_z - 1)),
h_x2(h_x * h_x), h_y2(h_y * h_y), h_z2(h_z * h_z),
coef(1 / (2 / h_x2 + 2 / h_y2 + 2 / h_z2 + parA)) {
    distributeLayers();
    initBuffers();
}

float JacobiSolver3D::solve() {
    float globDiff = eps;
    while (globDiff >= eps) {
        std::swap(prevBuffer, currBuffer);

        isendrecvBorders();
        float locCoreDiff = calcCore();
        waitBorders();
        float locBorderDiff = calcBorders();
        float locDiff = std::max(locCoreDiff, locBorderDiff);

        MPI_Allreduce(&locDiff, &globDiff, 1, MPI_FLOAT,
            MPI_MAX, MPI_COMM_WORLD);
    }
    return globDiff;
}

void JacobiSolver3D::distributeLayers() {
    std::vector<int> allLocN_x(commSize), allOffsets(commSize);
    int shift = 0;
    for (int i = 0; i < commSize; i++) {
        allLocN_x[i] = N_x / commSize;
        allLocN_x[i] += i < N_x % commSize;
        allOffsets[i] = shift;
        shift += allLocN_x[i];
    }
    locN_x = allLocN_x[procRank];
    locOffset = allOffsets[procRank];
}

void JacobiSolver3D::initBuffers() {
    prevBuffer = fvector(locN_x * N_y * N_z);
    currBuffer = fvector(locN_x * N_y * N_z);
    for (int i = 0; i < locN_x; i++) {
        for (int j = 0; j < N_y; j++) {
            for (int k = 0; k < N_z; k++) {
                int iOff = i + locOffset;
                bool isBorder = iOff == 0 || iOff == N_x - 1 ||
                    j == 0 || j == N_y - 1 ||
                    k == 0 || k == N_z - 1;
                if (isBorder) {
                    curr(i, j, k) = phi(iOff, j, k);
                    prev(i, j, k) = phi(iOff, j, k);
                }
            }
        }
    }
    topBorder = fvector(N_y * N_z);
    bottomBorder = fvector(N_y * N_z);
}

float JacobiSolver3D::calcCore() {
    float maxDiff = 0;
    for (int i = 1; i < locN_x - 1; i++) {
        for (int j = 1; j < N_y - 1; j++) {
            for (int k = 1; k < N_z - 1; k++) {
                float f_i = (prev(i + 1, j, k) + prev(i - 1, j, k)) / h_x2;
                float f_j = (prev(i, j + 1, k) + prev(i, j - 1, k)) / h_y2;
                float f_k = (prev(i, j, k + 1) + prev(i, j, k - 1)) / h_z2;

                curr(i, j, k) = coef *
                    (f_i + f_j + f_k - rho(i + locOffset, j, k));
                maxDiff = std::max(maxDiff,
                    std::abs(curr(i, j, k) - prev(i, j, k)));
            }
        }
    }
    return maxDiff;
}

float JacobiSolver3D::calcBorders() {
    float maxDiff = 0;
    for (int j = 1; j < N_y - 1; j++) {
        for (int k = 1; k < N_z - 1; k++) {
            if (procRank != 0) {
                int i = 0;
                float f_i = (prev(i + 1, j, k) +
                    topBorder[getBorderIdx(j, k)]) / h_x2;
                float f_j = (prev(i, j + 1, k) + prev(i, j - 1, k)) / h_y2;
                float f_k = (prev(i, j, k + 1) + prev(i, j, k - 1)) / h_z2;

                curr(i, j, k) = coef *
                    (f_i + f_j + f_k - rho(i + locOffset, j, k));
                maxDiff = std::max(maxDiff,
                    std::abs(curr(i, j, k) - prev(i, j, k)));
            }

            if (procRank != commSize - 1) {
                int i = locN_x - 1;
                float f_i = (prev(i - 1, j, k) +
                    bottomBorder[getBorderIdx(j, k)]) / h_x2;
                float f_j = (prev(i, j + 1, k) + prev(i, j - 1, k)) / h_y2;
                float f_k = (prev(i, j, k + 1) + prev(i, j, k - 1)) / h_z2;

                curr(i, j, k) = coef *
                    (f_i + f_j + f_k - rho(i + locOffset, j, k));
                maxDiff = std::max(maxDiff,
                    std::abs(curr(i, j, k) - prev(i, j, k)));
            }
        }
    }
    return maxDiff;
}

void JacobiSolver3D::isendrecvBorders() {
    if (procRank != 0) {
        float* prevTopBorderPtr = prevBuffer.data();
        MPI_Isend(prevTopBorderPtr, N_y * N_z, MPI_FLOAT,
            procRank - 1, procRank, MPI_COMM_WORLD, &reqSendTop);
        MPI_Irecv(topBorder.data(), N_y * N_z, MPI_FLOAT,
            procRank - 1, procRank - 1, MPI_COMM_WORLD, &reqRecvTop);
    }
    if (procRank != commSize - 1) {
        float* prevBottomBorderPtr =
            prevBuffer.data() + (locN_x - 1) * N_y * N_z;
        MPI_Isend(prevBottomBorderPtr, N_y * N_z, MPI_FLOAT,
            procRank + 1, procRank, MPI_COMM_WORLD, &reqSendBottom);
        MPI_Irecv(bottomBorder.data(), N_y * N_z, MPI_FLOAT,
            procRank + 1, procRank + 1, MPI_COMM_WORLD, &reqRecvBottom);
    }
}

void JacobiSolver3D::waitBorders() {
    if (procRank != 0) {
        MPI_Wait(&reqSendTop, MPI_STATUS_IGNORE);
        MPI_Wait(&reqRecvTop, MPI_STATUS_IGNORE);
    }
    if (procRank != commSize - 1) {
        MPI_Wait(&reqSendBottom, MPI_STATUS_IGNORE);
        MPI_Wait(&reqRecvBottom, MPI_STATUS_IGNORE);
    }
}

float JacobiSolver3D::phi(int i, int j, int k) const {
    float x = x_0 + i * h_x;
    float y = y_0 + j * h_y;
    float z = z_0 + k * h_z;
    return x * x + y * y + z * z;
}

float JacobiSolver3D::rho(int i, int j, int k) const {
    return 6 - parA * phi(i, j, k);
}

int JacobiSolver3D::getBorderIdx(int j, int k) const { return j * N_z + k; }

float& JacobiSolver3D::curr(int i, int j, int k) {
    return currBuffer[i * N_y * N_z + j * N_z + k];
}

float& JacobiSolver3D::prev(int i, int j, int k) {
    return prevBuffer[i * N_y * N_z + j * N_z + k];
}
