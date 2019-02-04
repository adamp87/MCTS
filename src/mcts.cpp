#include "mcts.cuh"

class Hearts;
template class RolloutCUDA<Hearts>;

// dummy functions for CPU compilation
template <typename TProblem>
RolloutCUDA<TProblem>::RolloutCUDA(unsigned int*, unsigned int) {
    pimpl = 0;
}

template <typename TProblem>
RolloutCUDA<TProblem>::~RolloutCUDA() {
    pimpl = 0;
}

template <typename TProblem>
bool RolloutCUDA<TProblem>::cuRollout(const uint8,
                                      const TProblem&,
                                      unsigned int,
                                      double&) const {
    return false; // should not be called
}
