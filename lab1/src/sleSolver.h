#pragma once

#include <string>
#include <vector>

class sleSolver {
 public:
    typedef std::vector<float> fvector;

    explicit sleSolver(const fvector& locMatA, const fvector& globVecB,
        int rank, int size);

    void solve();
    float getDiff(const std::string& vecXbin) const;
    fvector getActualX() const;

 private:
    void computeYAndNorm();
    void computeTau();
    void computeX();

    const fvector& locMatA, globVecB;
    const int globN, locN;
    const int rank;
    std::vector<int> vecSendcounts, vecDispls;

    fvector globX, globY, locX, locY;
    float globTau, globNormY;

    const float EPSILON;
};
