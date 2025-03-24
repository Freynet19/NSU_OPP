#include "BinIO.h"

#include <fstream>
#include <string>

BinIO::fvector BinIO::readVecFromBin(const std::string& filename) {
    std::ifstream file(filename.c_str(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    size_t fileSize = file.tellg();
    if (fileSize % sizeof(float) != 0) {
        throw std::runtime_error(
            "File size is not a multiple of sizeof(float): " + filename);
    }

    fvector floats(fileSize / sizeof(float));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(floats.data()),
        static_cast<std::streamsize>(fileSize));

    file.close();
    return floats;
}

void BinIO::writeVecToBin(const fvector& vec, const std::string& filename) {
    std::ofstream file(filename.c_str(), std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    file.write(reinterpret_cast<const char*>(vec.data()),
        static_cast<std::streamsize>(vec.size() * sizeof(float)));
    file.close();
}
