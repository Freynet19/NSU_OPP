#include "BinIO.h"

#include <fstream>
#include <filesystem>
#include <string>

BinIO::fvector BinIO::readVecFromBin(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::uintmax_t fileSize = std::filesystem::file_size(filename);
    if (fileSize % sizeof(float) != 0) {
        throw std::runtime_error(
            "File size is not a multiple of sizeof(float): " + filename);
    }

    fvector floats(fileSize / sizeof(float));
    file.read(reinterpret_cast<char*>(floats.data()),
        static_cast<std::streamsize>(fileSize));

    file.close();
    return floats;
}

void BinIO::writeVecToBin(const fvector& vec, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    file.write(reinterpret_cast<const char*>(vec.data()),
        static_cast<std::streamsize>(vec.size() * sizeof(float)));
    file.close();
}
