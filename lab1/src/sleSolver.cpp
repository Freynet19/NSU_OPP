#include "sleSolver.h"

#include <mpi.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include "BinIO.h"

sleSolver::sleSolver(const fvector& locMatA, const fvector& globVecB,
int rank, int commSize) : locMatA(locMatA), globVecB(globVecB),
globN(static_cast<int>(globVecB.size())),
locN(static_cast<int>(locMatA.size()) / globN), rank(rank) {
    globX = fvector(globN);
    globY = fvector(globN);
    locY = fvector(locN);
    globTau = globNormY = 0;
    vecSendcounts = std::vector<int>(commSize);
    vecDispls = std::vector<int>(commSize);

    const int baseSize = globN / commSize;
    const int remainder = globN % commSize;
    int shift = 0;
    for (int i = 0; i < commSize; ++i) {
        vecSendcounts[i] = baseSize + (i < remainder);
        vecDispls[i] = shift;
        shift += vecSendcounts[i];
    }
}

float sleSolver::getDiff(const std::string& vecXbin) const {
    fvector expectedX;
    try {
        expectedX = BinIO::readVecFromBin(vecXbin);
        if (expectedX.size() != globN) return -1;
    } catch (const std::runtime_error&) {
        return -1;
    }

    float diff = 0;
    for (int i = 0; i < globN; i++) {
        diff += std::abs(expectedX[i] - globX[i]);
    }
    return diff;
}

sleSolver::fvector sleSolver::getActualX() const {
    return globX;
}

void sleSolver::computeYAndNorm() {
    float locNormY = 0;
    for (int i = 0; i < locN; i++) {
        float Ax_i = 0;
        for (int j = 0; j < globN; j++) {
            Ax_i += locMatA[i * globN + j] * globX[j];
        }
        locY[i] = Ax_i - globVecB[vecDispls[rank] + i];
        locNormY += locY[i] * locY[i];
    }

    MPI_Allgatherv(locY.data(), vecSendcounts[rank], MPI_FLOAT,
        globY.data(), vecSendcounts.data(),
        vecDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);

    MPI_Reduce(&locNormY, &globNormY, 1, MPI_FLOAT,
        MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        globNormY = std::sqrt(globNormY);
    }
}

void sleSolver::computeTau() {
    float locNumerator = 0, locDenominator = 0;
    for (int i = 0; i < locN; i++) {
        float Ay_i = 0;
        for (int j = 0; j < globN; j++) {
            Ay_i += locMatA[i * globN + j] * globY[j];
        }
        locNumerator += locY[i] * Ay_i;
        locDenominator += Ay_i * Ay_i;
    }

    float globNumerator, globDenominator;
    MPI_Allreduce(&locNumerator, &globNumerator,
        1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&locDenominator, &globDenominator,
        1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    globTau = globNumerator / globDenominator;
}

void sleSolver::computeX() {
    auto locX = fvector(locN);
    for (int i = 0; i < locN; i++) {
        locX[i] -= globTau * locY[i];
    }
    MPI_Allgatherv(locX.data(), vecSendcounts[rank], MPI_FLOAT,
        globX.data(), vecSendcounts.data(),
        vecDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
}

void sleSolver::solve() {
    float normB = 1;
    if (rank == 0) {
        normB = 0;
        for (int i = 0; i < globN; ++i) {
            normB += globVecB[i] * globVecB[i];
        }
        normB = std::sqrt(normB);
    }

    bool done = false;
    while (true) {
        computeYAndNorm();

        if (rank == 0) {
            if (globNormY / normB < EPSILON) done = true;
            // else std::cerr << globNormY / normB << std::endl;
        }
        MPI_Bcast(&done, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        if (done) break;

        computeTau();
        computeX();
    }
}
