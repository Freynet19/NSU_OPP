#pragma once

#include "SLE_Solver.h"

class OneSectionOMP : public SLE_Solver {
 public:
    explicit OneSectionOMP(const fvector& matA, const fvector& vecB);
    void solve() override;

 private:
    float norm(const fvector& x) override;
    void computeY() override;
    void computeTau() override;
    void computeX() override;

    float numerator, denominator;
};
