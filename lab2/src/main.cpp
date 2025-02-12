#include <omp.h>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>

#include "NonParallel.h"
#include "MultipleSectionsOMP.h"
#include "OneSectionOMP.h"
#include "BinIO.h"

template<typename SolverType>
void solveSLE(const std::vector<float>& matA, const std::vector<float>& vecB) {
    SolverType solver(matA, vecB);
    const std::string type = typeid(SolverType).name();

    std::cout << "using " << type << " solver..." << std::endl;

    const auto start = std::chrono::system_clock::now();
    solver.solve();
    const auto end = std::chrono::system_clock::now();

    const double duration = std::chrono::duration_cast<
        std::chrono::duration<double>>(end - start).count();
    std::cout << "Done! Time taken: " << duration << " sec" << std::endl;

    BinIO::writeVecToBin(solver.getActualX(), "actualX.bin");

    const float diff = solver.getDiff("vecX.bin");
    const float avg = diff / static_cast<float>(vecB.size());
    std::cout << "Total diff: " << diff
              << " (avg " << avg << ")" << std::endl;
    std::cout << "--------------------------------" <<
                 "--------------------------------" << std::endl;
}

int main() {
    try {
        const std::vector<float> matA = BinIO::readVecFromBin("matA.bin");
        const std::vector<float> vecB = BinIO::readVecFromBin("vecB.bin");

        solveSLE<NonParallel>(matA, vecB);

        const int maxThreads = omp_get_max_threads();

        for (int t = 2; t <= maxThreads; ++t) {
            omp_set_num_threads(t);
            std::cout << "Set " << t << " threads" << std::endl;
            solveSLE<MultipleSectionsOMP>(matA, vecB);
        }

        for (int t = 2; t <= maxThreads; ++t) {
            omp_set_num_threads(t);
            std::cout << "Set " << t << " threads, ";
            solveSLE<OneSectionOMP>(matA, vecB);
        }
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
