#include "MatrixMultiplier.h"
#include <iostream>
#include <vector>

MatrixMultiplier::MatrixMultiplier(int size, int rank, int N1, int N2, int N3)
: gridSize(size), procRank(rank), N1(N1), N2(N2), N3(N3) {
    createGrid();
    if (procRank == 0) {
        fillMatrix(globA, N1, N2);
        fillMatrix(globB, N2, N3);
        globC = fvector(N1 * N3);
    }
    distributeMatA();
    distributeMatB();
}

void MatrixMultiplier::createGrid() {
    int dims[2] = {0, 0};
    MPI_Dims_create(gridSize, 2, dims);

    int periods[2] = {0, 0};
    int reorder = 0;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &COMM_GRID);
    MPI_Cart_coords(COMM_GRID, procRank, 2, procCoords);

    int remainDims[2] = {0, 1};
    MPI_Cart_sub(COMM_GRID, remainDims, &COMM_ROW);

    remainDims[0] = 1;
    remainDims[1] = 0;
    MPI_Cart_sub(COMM_GRID, remainDims, &COMM_COLUMN);
}

void MatrixMultiplier::distributeMatA() {
    int colCommSize;
    MPI_Comm_size(COMM_COLUMN, &colCommSize);
    int rowCommSize = gridSize / colCommSize;

    allLocN1 = std::vector<int>(gridSize);

    std::vector<int> sendcounts(colCommSize);
    std::vector<int> displs(colCommSize);
    const int baseSize = N1 / colCommSize;
    const int remainder = N1 % colCommSize;
    int shift = 0;
    for (int i = 0; i < colCommSize; ++i) {
        int locRowsCount = baseSize + (i < remainder);
        for (int j = 0; j < rowCommSize; ++j) {
            allLocN1[i * rowCommSize + j] = locRowsCount;
        }
        sendcounts[i] = locRowsCount * N2;
        displs[i] = shift;
        shift += sendcounts[i];
    }

    locN1 = allLocN1[procRank];
    const int locSize = locN1 * N2;
    locA = fvector(locSize);

    if (procCoords[1] == 0) {
        MPI_Scatterv(globA.data(), sendcounts.data(), displs.data(), MPI_FLOAT,
            locA.data(), locSize, MPI_FLOAT, 0, COMM_COLUMN);
    }

    MPI_Bcast(locA.data(), locSize, MPI_FLOAT, 0, COMM_ROW);
}

void MatrixMultiplier::distributeMatB() {
    int rowCommSize;
    MPI_Comm_size(COMM_ROW, &rowCommSize);
    int colCommSize = gridSize / rowCommSize;

    allLocN3 = std::vector<int>(gridSize);

    std::vector<int> sendcounts(rowCommSize);
    std::vector<int> displs(rowCommSize);
    const int baseSize = N3 / rowCommSize;
    const int remainder = N3 % rowCommSize;
    int shift = 0;
    for (int i = 0; i < rowCommSize; ++i) {
        sendcounts[i] = baseSize + (i < remainder);
        for (int j = 0; j < colCommSize; ++j) {
            allLocN3[j * rowCommSize + i] = sendcounts[i];
        }
        displs[i] = shift;
        shift += sendcounts[i];
    }

    locN3 = allLocN3[procRank];
    const int locSize = locN3 * N2;
    locB = fvector(locSize);

    if (procCoords[0] == 0) {
        MPI_Datatype col;
        MPI_Type_vector(N2, 1, N3, MPI_FLOAT, &col);
        MPI_Type_create_resized(col, 0, sizeof(float), &col);
        MPI_Type_commit(&col);

        MPI_Scatterv(globB.data(), sendcounts.data(), displs.data(), col,
            locB.data(), locSize, MPI_FLOAT, 0, COMM_ROW);

        MPI_Type_free(&col);
    }

    MPI_Bcast(locB.data(), locSize, MPI_FLOAT, 0, COMM_COLUMN);
}

void MatrixMultiplier::locMultiply() {
    locC = fvector(locN1 * locN3);
    for (int i = 0; i < locN1; ++i) {
        for (int j = 0; j < locN3; ++j) {
            for (int k = 0; k < N2; ++k) {
                locC[i * locN3 + j] += locA[i * N2 + k] * locB[j * N2 + k];
            }
        }
    }
}

void MatrixMultiplier::gatherMatC() {
    int rowCommSize;
    MPI_Comm_size(COMM_ROW, &rowCommSize);

    std::vector<int> bigRowSendcounts(gridSize), smallRowSendcounts(gridSize);
    std::vector<int> bigRowRecvcounts(gridSize), smallRowRecvcounts(gridSize);
    std::vector<int> displs(gridSize);

    int baseDispls = 0;
    for (int i = 0; i < gridSize; ++i) {
        const int recvcount = allLocN3[i];
        if (recvcount == allLocN3[0]) {
            bigRowSendcounts[i] = allLocN1[i] * allLocN3[i];
            bigRowRecvcounts[i] = recvcount;
            smallRowSendcounts[i] = 0;
            smallRowRecvcounts[i] = 0;
        } else {
            bigRowSendcounts[i] = 0;
            bigRowRecvcounts[i] = 0;
            smallRowSendcounts[i] = allLocN1[i] * allLocN3[i];
            smallRowRecvcounts[i] = recvcount;
        }
        if (i == 0) continue;
        if (i % rowCommSize == 0) {
            baseDispls += N3 * allLocN1[i - 1];
            displs[i] = baseDispls;
        } else {
            displs[i] = displs[i - 1] +
                bigRowRecvcounts[i - 1] + smallRowRecvcounts[i - 1];
        }
    }

    MPI_Datatype bigRowBlock, smallRowBlock;
    MPI_Type_vector(allLocN1[0], allLocN3[0],
        N3, MPI_FLOAT, &bigRowBlock);
    MPI_Type_vector(allLocN1[0], allLocN3[0] - 1,
        N3, MPI_FLOAT, &smallRowBlock);
    MPI_Type_create_resized(bigRowBlock, 0, sizeof(float), &bigRowBlock);
    MPI_Type_create_resized(smallRowBlock, 0, sizeof(float), &smallRowBlock);
    MPI_Type_commit(&bigRowBlock);
    MPI_Type_commit(&smallRowBlock);

    MPI_Gatherv(locC.data(), bigRowSendcounts[procRank], MPI_FLOAT,
        globC.data(), bigRowRecvcounts.data(), displs.data(),
        bigRowBlock, 0, MPI_COMM_WORLD);
    MPI_Gatherv(locC.data(), smallRowSendcounts[procRank], MPI_FLOAT,
        globC.data(), smallRowRecvcounts.data(), displs.data(),
        smallRowBlock, 0, MPI_COMM_WORLD);

    MPI_Type_free(&bigRowBlock);
    MPI_Type_free(&smallRowBlock);
}

void MatrixMultiplier::printC() const {
    if (procRank == 0) {
        for (int i = 0; i < N1; ++i) {
            for (int j = 0; j < N3; ++j) {
                std::cout << globC[i * N3 + j] << " ";
            }
            std::cout << std::endl;
        }
    }
}

void MatrixMultiplier::fillMatrix(fvector& mat, int rows, int columns) {
    size_t size = rows * columns;
    mat.resize(size);
    for (size_t i = 0; i < size; i++) {
        mat[i] = static_cast<float>(i % columns);
    }
}
