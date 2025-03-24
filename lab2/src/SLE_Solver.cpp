#include "SLE_Solver.h"

#include <filesystem>
#include <iostream>
#include <string>
#include "BinIO.h"

SLE_Solver::SLE_Solver(const fvector& matA, const fvector& vecB):
matA(matA), vecB(vecB), N(vecB.size()) {
    vecX = fvector(N);
    vecY = fvector(N);
    tau = 0;
}

float SLE_Solver::getDiff(const std::string& vecXbin) const {
    fvector expectedX;
    try {
        expectedX = BinIO::readVecFromBin(vecXbin);
        if (expectedX.size() != N) return -1;
    } catch (const std::runtime_error&) {
        return -1;
    }

    float diff = 0;
    for (size_t i = 0; i < N; i++) {
        diff += std::abs(expectedX[i] - vecX[i]);
    }
    return diff;
}

SLE_Solver::fvector SLE_Solver::getActualX() const {
    return vecX;
}
