#include "mcts.cuh"

// dummy functions for CPU compilation
RolloutCUDA::RolloutCUDA(unsigned int*, unsigned int) {
    pimpl = 0;
}

RolloutCUDA::~RolloutCUDA() {
    pimpl = 0;
}

bool RolloutCUDA::cuRollout(const uint8,
                            const Hearts&,
                            unsigned int,
                            double&) const {
    return false; // should not be called
}
