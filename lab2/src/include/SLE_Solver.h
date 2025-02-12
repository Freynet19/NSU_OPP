#pragma once

#include <string>
#include <vector>

class SLE_Solver {
 public:
    using fvector = std::vector<float>;

    explicit SLE_Solver(const fvector& matA, const fvector& vecB);
    virtual ~SLE_Solver() = default;

    virtual void solve() = 0;
    [[nodiscard]] float getDiff(const std::string& vecXbin) const;
    [[nodiscard]] fvector getActualX() const;

 protected:
    const fvector& matA, vecB;
    const size_t N;

    fvector vecX, vecY;
    float tau;

    static constexpr float EPSILON = 1e-5;

 private:
    virtual float norm(const fvector& x) = 0;
    virtual void computeY() = 0;
    virtual void computeTau() = 0;
    virtual void computeX() = 0;
};
