#pragma once

#include "SLE_Solver.h"

class NonParallel : public SLE_Solver {
 public:
    explicit NonParallel(const fvector& matA, const fvector& vecB);
    void solve() override;

 private:
    float norm(const fvector& x) override;
    void computeY() override;
    void computeTau() override;
    void computeX() override;
};
