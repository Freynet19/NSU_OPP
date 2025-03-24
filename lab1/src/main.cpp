#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cfloat>

#include "sleSolver.h"
#include "BinIO.h"

#define LOOPS 8

void solveSLE(const std::vector<float>& localMatA,
const std::vector<float>& vecB, int rank, int commSize) {
    if (rank == 0) {
        std::cout << "Running with " <<
            commSize << " processes..." << std::endl;
    }

    std::vector<int> vecSendcounts = std::vector<int>(commSize);
    std::vector<int> vecDispls = std::vector<int>(commSize);

    const int baseSize = static_cast<int>(vecB.size()) / commSize;
    const int remainder = static_cast<int>(vecB.size()) % commSize;
    int shift = 0;
    for (int i = 0; i < commSize; ++i) {
        vecSendcounts[i] = baseSize + (i < remainder);
        vecDispls[i] = shift;
        shift += vecSendcounts[i];
    }

    double duration = DBL_MAX;
    for (int i = 0; i < LOOPS; i++) {
        sleSolver solver(localMatA, vecB, rank, vecSendcounts, vecDispls);
        const double start = MPI_Wtime();
        solver.solve();
        const double end = MPI_Wtime();
        duration = std::min(duration, end - start);
    }

    if (rank == 0) {
        std::cout << "Done! Time taken: " << duration << " sec" << std::endl;

        // BinIO::writeVecToBin(solver.getActualX(), "actualX.bin");

        // const float diff = solver.getDiff("vecX.bin");
        // const float avg = diff / static_cast<float>(vecB.size());
        // std::cout << "Total diff: " << diff <<
        //     " (avg " << avg << ")" << std::endl;
        std::cout << "================================" <<
                     "================================" << std::endl;
    }
}

void prepareScatterv(const int vecSize, const int commSize,
std::vector<int>& matSendcounts, std::vector<int>& matDispls) {
    const int baseSize = vecSize / commSize;
    const int remainder = vecSize % commSize;
    int shift = 0;
    for (int i = 0; i < commSize; ++i) {
        matSendcounts[i] = (baseSize + (i < remainder)) * vecSize;
        matDispls[i] = shift;
        shift += matSendcounts[i];
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        std::vector<float> matA, vecB;
        if (rank == 0) {
            matA = BinIO::readVecFromBin("matA.bin");
            vecB = BinIO::readVecFromBin("vecB.bin");
            if (matA.size() != vecB.size() * vecB.size()) {
                throw std::runtime_error(
                    "Invalid matrix size: matA must be of size N x N.");
            }
        }

        int N = static_cast<int>(vecB.size());
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) vecB.resize(N);
        MPI_Bcast(vecB.data(), N, MPI_FLOAT, 0, MPI_COMM_WORLD);

        std::vector<int> matSendcounts = std::vector<int>(size);
        std::vector<int> matDispls = std::vector<int>(size);
        prepareScatterv(N, size, matSendcounts, matDispls);

        std::vector<float> localMatA(matSendcounts[rank]);
        MPI_Scatterv(matA.data(), matSendcounts.data(),
            matDispls.data(), MPI_FLOAT, localMatA.data(),
            matSendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

        solveSLE(localMatA, vecB, rank, size);
    } catch (const std::runtime_error& e) {
        std::cerr << "Process " << rank << ": " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
