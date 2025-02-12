#include "OneSectionOMP.h"

#include <cmath>
#include <iostream>
#include <ostream>

OneSectionOMP::OneSectionOMP(
    const fvector& matA, const fvector& vecB) : SLE_Solver(matA, vecB) {
    numerator = denominator = 0;
}

float OneSectionOMP::norm(const fvector& x) {
    float result = 0;
    #pragma omp parallel for default(none) shared(N, x) reduction(+:result)
    for (size_t i = 0; i < N; i++) {
        result += x[i] * x[i];
    }
    return std::sqrt(result);
}

void OneSectionOMP::computeY() {
    #pragma omp for
    for (size_t i = 0; i < N; i++) {
        float Ax_i = 0;
        for (size_t j = 0; j < N; j++) {
            Ax_i += matA[i * N + j] * vecX[j];
        }
        vecY[i] = Ax_i - vecB[i];
    }
}

void OneSectionOMP::computeTau() {
    #pragma omp for reduction(+:numerator, denominator)
    for (size_t i = 0; i < N; i++) {
        float Ay_i = 0;
        for (size_t j = 0; j < N; j++) {
            Ay_i += matA[i * N + j] * vecY[j];
        }
        numerator += vecY[i] * Ay_i;
        denominator += Ay_i * Ay_i;
    }
    #pragma omp single
    {
        tau = numerator / denominator;
        numerator = denominator = 0;
    }
}

void OneSectionOMP::computeX() {
    #pragma omp for
    for (size_t i = 0; i < N; i++) {
        vecX[i] -= tau * vecY[i];
    }
}

void OneSectionOMP::solve() {
    const float normB = norm(vecB);
    bool exit_flag = false;
    #pragma omp parallel default(none) \
    shared(matA, vecB, vecX, vecY, N, tau, normB, exit_flag)
    {
        while (!exit_flag) {
            computeY();

            float normY;
            #pragma omp single
            {
                normY = norm(vecY);
                if (normY / normB < EPSILON) exit_flag = true;
            }

            if (exit_flag) break;

            computeTau();
            computeX();
        }
    }
}
