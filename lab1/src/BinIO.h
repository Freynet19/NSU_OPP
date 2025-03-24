#pragma once

#include <string>
#include <vector>

class BinIO {
 public:
    typedef std::vector<float> fvector;

    static fvector readVecFromBin(const std::string& filename);
    static void writeVecToBin(const fvector& vec,
                              const std::string& filename);
};
