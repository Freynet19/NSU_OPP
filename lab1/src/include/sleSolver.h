#pragma once

#include <string>
#include <vector>

class sleSolver {
 public:
    using fvector = std::vector<float>;

    explicit sleSolver(const fvector& locMatA, const fvector& globVecB,
        int rank, int size);

    void solve();
    [[nodiscard]] float getDiff(const std::string& vecXbin) const;
    [[nodiscard]] fvector getActualX() const;

 private:
    void computeYAndNorm();
    void computeTau();
    void computeX();

    const fvector& locMatA, globVecB;
    const int globN, locN;
    const int rank;
    std::vector<int> vecSendcounts, vecDispls;

    fvector globX, globY, locY;
    float globTau, globNormY;

    static constexpr float EPSILON = 1e-3;
};
