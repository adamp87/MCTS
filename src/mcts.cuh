#ifndef MCTS_CUH
#define MCTS_CUH

#include <vector>

#include "defs.hpp"
#include "hearts.hpp"

class RolloutCUDA {
    struct impl;
    impl* pimpl;

public:
    RolloutCUDA(unsigned int* iterations, unsigned int seed);
    ~RolloutCUDA();

    bool hasGPU() const {
        return pimpl != 0;
    }

    bool cuRollout(const Hearts& state,
                   const Hearts::Player& ai,
                   unsigned int iteration,
                   std::vector<uint8>& points) const;
};

#endif // MCTS_CUH
