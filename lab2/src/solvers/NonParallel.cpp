#include "NonParallel.h"

#include <cmath>

NonParallel::NonParallel(
    const fvector& matA, const fvector& vecB) : SLE_Solver(matA, vecB) {}

float NonParallel::norm(const fvector& x) {
    float result = 0;
    for (size_t i = 0; i < N; i++) {
        result += x[i] * x[i];
    }
    return std::sqrt(result);
}

void NonParallel::computeY() {
    for (size_t i = 0; i < N; i++) {
        float Ax_i = 0;
        for (size_t j = 0; j < N; j++) {
            Ax_i += matA[i * N + j] * vecX[j];
        }
        vecY[i] = Ax_i - vecB[i];
    }
}

void NonParallel::computeTau() {
    float numerator = 0, denominator = 0;
    for (size_t i = 0; i < N; i++) {
        float Ay_i = 0;
        for (size_t j = 0; j < N; j++) {
            Ay_i += matA[i * N + j] * vecY[j];
        }
        numerator += vecY[i] * Ay_i;
        denominator += Ay_i * Ay_i;
    }
    tau = numerator / denominator;
}

void NonParallel::computeX() {
    for (size_t i = 0; i < N; i++) {
        vecX[i] -= tau * vecY[i];
    }
}

void NonParallel::solve() {
    const float normB = norm(vecB);
    while (true) {
        computeY();
        if (norm(vecY) / normB < EPSILON) break;
        computeTau();
        computeX();
    }
}
