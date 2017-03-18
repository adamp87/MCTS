#include "mcts.cuh"

// dummy functions for CPU compilation
RolloutContainerCPP::RolloutContainerCPP(uint32) {
    data = 0;
}

RolloutContainerCPP::~RolloutContainerCPP() {
    data = 0;
}

unsigned int* RolloutContainerCPP::_cuRollout(const Hearts::State&,
                                              const Hearts::Player&,
                                              const uint8*,
                                              uint8) {
    return 0; // should not be called
}
