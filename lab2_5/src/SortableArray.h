#pragma once

#include <vector>
#include <string>

class SortableArray {
 public:
    SortableArray();
    void testParallelThreshold();
    void testNumThreads();

 private:
    static std::vector<float> fillRandom();
    void quickSortOMP(size_t beginIdx, size_t sortSize);
    size_t partition(size_t beginIdx, size_t sortSize);
    double qSortMeasureTime(int loops);
    void validate() const;

    const std::vector<float> unsorted;
    std::vector<float> sorted;
    size_t parallelThreshold;

    static constexpr size_t ARR_SIZE = 128 * 1024 * 1024;

    static inline const std::string TEST_PT_FILE_NAME = "testPT.csv";
    static constexpr int TEST_PT_NUM_LOOPS = 4;
    static constexpr size_t TEST_PT_INIT_PARALLEL_THRESHOLD = ARR_SIZE / 128;
    static constexpr float TEST_PT_DENOMINATOR = 1.25;
    const int TEST_PT_NUM_THREADS;

    static inline const std::string TEST_NT_FILE_NAME = "testNT.csv";
    static constexpr int TEST_NT_NUM_LOOPS = 8;
    static constexpr size_t TEST_NT_PARALLEL_THRESHOLD = 1024;
    const int TEST_NT_MAX_THREADS;
};
