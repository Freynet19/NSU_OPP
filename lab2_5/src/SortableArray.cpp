#include "SortableArray.h"

#include <omp.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <vector>
#include <utility>

SortableArray::SortableArray()
: unsorted(fillRandom()), parallelThreshold(0),
TEST_PT_NUM_THREADS(omp_get_max_threads() / 2),
TEST_NT_MAX_THREADS(omp_get_max_threads()) {
    std::cout << "Created array of length " << ARR_SIZE << std::endl;
}

void SortableArray::testParallelThreshold() {
    std::cout << "Testing parallelThreshold..." << std::endl;

    std::ofstream file(TEST_PT_FILE_NAME);
    if (!file.is_open()) throw std::runtime_error(
        "Could not open file " + TEST_PT_FILE_NAME);

    omp_set_num_threads(TEST_PT_NUM_THREADS);
    file << "parallelThreshold,Time (sec)" << std::endl;
    for (parallelThreshold = TEST_PT_INIT_PARALLEL_THRESHOLD;
         parallelThreshold > 0; parallelThreshold /= TEST_PT_DENOMINATOR) {
        const double duration = qSortMeasureTime(TEST_PT_NUM_LOOPS);
        validate();
        file << parallelThreshold << ',' << duration << std::endl;
    }

    file.close();
    std::cout << "Data written in " << TEST_PT_FILE_NAME << std::endl;
}

void SortableArray::testNumThreads() {
    std::cout << "Testing numThreads..." << std::endl;

    std::ofstream file(TEST_NT_FILE_NAME);
    if (!file.is_open()) throw std::runtime_error(
        "Could not open file " + TEST_NT_FILE_NAME);

    parallelThreshold = TEST_NT_PARALLEL_THRESHOLD;
    file << "numThreads,Time (sec)" << std::endl;
    for (int numThreads = 1; numThreads <= TEST_NT_MAX_THREADS; ++numThreads) {
        omp_set_num_threads(numThreads);
        const double duration = qSortMeasureTime(TEST_NT_NUM_LOOPS);
        validate();
        file << numThreads << ',' << duration << std::endl;
    }

    file.close();
    std::cout << "Data written in " << TEST_NT_FILE_NAME << std::endl;
}

void SortableArray::quickSortOMP(size_t beginIdx, size_t sortSize) {
    if (sortSize <= 1) return;

    const size_t pivot = partition(beginIdx, sortSize) + 1;
    const size_t sizeL = pivot - beginIdx;
    const size_t sizeR = sortSize - sizeL;

    #pragma omp task default(none) shared(sorted, beginIdx, sizeL) \
        if (sizeL > parallelThreshold)
    quickSortOMP(beginIdx, sizeL);
    #pragma omp task default(none) shared(sorted, pivot, sizeR) \
        if (sizeR > parallelThreshold)
    quickSortOMP(pivot, sizeR);
}

size_t SortableArray::partition(size_t beginIdx, size_t sortSize) {
    const float pivot = sorted[beginIdx];
    size_t l = beginIdx;
    size_t r = beginIdx + sortSize - 1;

    while (true) {
        while (l < beginIdx + sortSize && sorted[l] < pivot) ++l;
        while (r > beginIdx && sorted[r] > pivot) --r;

        if (l >= r) return r;
        std::swap(sorted[l], sorted[r]);
        ++l;
        --r;
    }
}

double SortableArray::qSortMeasureTime(int loops) {
    auto minDuration = std::chrono::high_resolution_clock::duration::max();
    for (int i = 0; i < loops; ++i) {
        sorted = unsorted;
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel default(shared)
        {
            #pragma omp single
            {
                #pragma omp task default(none) shared(sorted)
                quickSortOMP(0, ARR_SIZE);
                #pragma omp taskwait
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        minDuration = std::min(minDuration, end - start);
    }
    return std::chrono::duration_cast<
        std::chrono::duration<double>>(minDuration).count();
}

void SortableArray::validate() const {
    for (size_t i = 1; i < ARR_SIZE; ++i) {
        if (sorted[i - 1] > sorted[i]) {
            throw std::runtime_error("Array sorted incorrectly!");
        }
    }
}

std::vector<float> SortableArray::fillRandom() {
    auto arr = std::vector<float>(ARR_SIZE);
    std::mt19937 gen(23201);
    std::uniform_real_distribution dist(0.0f, 1000.0f);
    std::generate(arr.begin(), arr.end(), [&] { return dist(gen); });
    return arr;
}
