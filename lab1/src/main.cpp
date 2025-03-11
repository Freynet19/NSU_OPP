#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>

#include "sleSolver.h"
#include "BinIO.h"

void solveSLE(const std::vector<float>& localMatA,
const std::vector<float>& vecB, int rank, int commSize) {
    sleSolver solver(localMatA, vecB, rank, commSize);

    const double start = MPI_Wtime();
    solver.solve();
    const double end = MPI_Wtime();

    if (rank == 0) {
        const double duration = end - start;
        std::cout << "Done! Time taken: " << duration << " sec" << std::endl;

        // BinIO::writeVecToBin(solver.getActualX(), "actualX.bin");

        const float diff = solver.getDiff("vecX.bin");
        const float avg = diff / static_cast<float>(vecB.size());
        std::cout << "Total diff: " << diff <<
            " (avg " << avg << ")" << std::endl;
        std::cout << "--------------------------------" <<
                     "--------------------------------" << std::endl;
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

        auto matSendcounts = std::vector<int>(size);
        auto matDispls = std::vector<int>(size);
        prepareScatterv(N, size, matSendcounts, matDispls);

        std::vector<float> localMatA(matSendcounts[rank]);
        MPI_Scatterv(matA.data(), matSendcounts.data(),
            matDispls.data(), MPI_FLOAT, localMatA.data(),
            matSendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
        /*std::cout << "Process " << rank
          << " received " << localMatA.size()
          << " elements (" << matSendcounts[rank]/N << " rows)"
          << std::endl;*/

        solveSLE(localMatA, vecB, rank, size);
    } catch (const std::runtime_error& e) {
        std::cerr << "Process " << rank << ": " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
