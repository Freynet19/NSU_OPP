#include "MultipleSectionsOMP.h"

#include <cmath>

MultipleSectionsOMP::MultipleSectionsOMP(
    const fvector& matA, const fvector& vecB) : SLE_Solver(matA, vecB) {}


float MultipleSectionsOMP::norm(const fvector& x) {
    float result = 0;
    #pragma omp parallel for default(none) shared(N, x) reduction(+:result)
    for (size_t i = 0; i < N; i++) {
        result += x[i] * x[i];
    }
    return std::sqrt(result);
}

void MultipleSectionsOMP::computeY() {
    #pragma omp parallel for default(none) shared(N, matA, vecB, vecX)
    for (size_t i = 0; i < N; i++) {
        float Ax_i = 0;
        for (size_t j = 0; j < N; j++) {
            Ax_i += matA[i * N + j] * vecX[j];
        }
        vecY[i] = Ax_i - vecB[i];
    }
}

void MultipleSectionsOMP::computeTau() {
    float numerator = 0, denominator = 0;
    #pragma omp parallel for default(none) shared(N, matA, vecY) \
    reduction(+:numerator, denominator)
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

void MultipleSectionsOMP::computeX() {
    #pragma omp parallel for default(none) shared(tau, vecX, vecY)
    for (size_t i = 0; i < N; i++) {
        vecX[i] -= tau * vecY[i];
    }
}

void MultipleSectionsOMP::solve() {
    const float normB = norm(vecB);
    while (true) {
        computeY();
        if (norm(vecY) / normB < EPSILON) break;
        computeTau();
        computeX();
    }
}
