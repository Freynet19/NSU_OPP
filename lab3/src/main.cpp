#include <mpi.h>
#include <cfloat>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include "MatrixMultiplier.h"

#define LOOPS 8

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) std::cerr <<
            "Usage: " << argv[0] << " N1 N2 N3" << std::endl;
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int N1 = std::atoi(argv[1]);
    int N2 = std::atoi(argv[2]);
    int N3 = std::atoi(argv[3]);
    if (N1 <= 0 || N2 <= 0 || N3 <= 0) throw std::invalid_argument(
        "All N must be positive");

    double duration = DBL_MAX;
    for (int i = 0; i < LOOPS; i++) {
        const double start = MPI_Wtime();
        MatrixMultiplier mm(size, rank, N1, N2, N3);
        mm.locMultiply();
        mm.gatherMatC();
        mm.printC();
        const double end = MPI_Wtime();
        duration = std::min(duration, end - start);
    }

    if (rank == 0) std::cout <<
        "Time taken: " << duration << " sec" << std::endl;

    MPI_Finalize();
    return EXIT_SUCCESS;
}
