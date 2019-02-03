#include "mcts.cuh"

// dummy functions for CPU compilation
RolloutCUDA::RolloutCUDA(unsigned int*, unsigned int) {
    pimpl = 0;
}

RolloutCUDA::~RolloutCUDA() {
    pimpl = 0;
}

bool RolloutCUDA::cuRollout(const Hearts&,
                            const Hearts::Player&,
                            unsigned int,
                            double&) const {
    return false; // should not be called
}
