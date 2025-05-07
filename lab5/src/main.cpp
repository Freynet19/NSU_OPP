#include <mpi.h>
#include "TaskManager.h"

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    TaskManager manager(MPI_COMM_WORLD);
    manager.run();

    MPI_Finalize();
    return 0;
}
