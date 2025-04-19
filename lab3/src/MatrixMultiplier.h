#pragma once

#include <mpi.h>
#include <vector>

typedef std::vector<float> fvector;

class MatrixMultiplier {
 public:
    explicit MatrixMultiplier(int size, int rank, int N1, int N2, int N3);
    void locMultiply();
    void gatherMatC();
    void printC() const;

 private:
    void createGrid();
    void distributeMatA();
    void distributeMatB();
    static void fillMatrix(fvector& mat, int rows, int columns);

    const int gridSize, procRank;
    const int N1, N2, N3;
    int procCoords[2];

    fvector globA, globB, globC;
    fvector locA, locB, locC;
    int locN1, locN3;
    std::vector<int> allLocN1, allLocN3;

    MPI_Comm COMM_GRID;
    MPI_Comm COMM_ROW;
    MPI_Comm COMM_COLUMN;
};
