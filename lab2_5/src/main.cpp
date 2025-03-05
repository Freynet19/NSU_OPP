#include <iostream>
#include "SortableArray.h"

int main() {
    try {
        SortableArray arr;
        arr.testParallelThreshold();
        arr.testNumThreads();
        std::cout << "Done! Exiting..." << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
