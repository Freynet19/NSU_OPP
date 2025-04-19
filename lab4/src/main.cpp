#include <mpi.h>
#include <cstdlib>
#include <cfloat>
#include <iostream>
#include <algorithm>
#include "JacobiSolver3D.h"

#define LOOPS 1

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double duration = DBL_MAX;
    float diff = -1;
    for (int i = 0; i < LOOPS; i++) {
        JacobiSolver3D solver(rank, size);
        const double start = MPI_Wtime();
        diff = solver.solve();
        const double end = MPI_Wtime();
        duration = std::min(duration, end - start);
    }

    if (rank == 0) {
        std::cout << "Time taken: " << duration << " sec" << std::endl;
        std::cout << "Max diff: " << diff << std::endl;
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
