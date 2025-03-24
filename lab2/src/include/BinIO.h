#pragma once

#include <string>
#include <vector>

class BinIO {
 public:
    using fvector = std::vector<float>;

    static fvector readVecFromBin(const std::string& filename);
    static void writeVecToBin(const fvector& vec,
                              const std::string& filename);
};
