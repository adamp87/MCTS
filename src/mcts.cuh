#ifndef MCTS_CUH
#define MCTS_CUH

#include <vector>

#include "defs.hpp"
#include "hearts.hpp"

//! Execute multiple rollout on CUDA
/*!
 * \details TODO
 * \author adamp87
*/
class RolloutCUDA {
    struct impl;
    impl* pimpl;

public:
    RolloutCUDA(unsigned int* iterations, unsigned int seed);
    ~RolloutCUDA();

    bool hasGPU() const {
        return pimpl != 0;
    }

    //! TODO
    bool cuRollout(const uint8 idxAi,
                   const Hearts& state,
                   unsigned int iteration,
                   double& winSum) const;
};

#endif // MCTS_CUH
