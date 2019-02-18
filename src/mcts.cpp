#include "mcts.cuh"

class Chess;
class Hearts;
template class RolloutCUDA<Chess>;
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
bool RolloutCUDA<TProblem>::cuRollout(int,
                                      int,
                                      const TProblem&,
                                      unsigned int,
                                      double&) const {
    return false; // should not be called
}
