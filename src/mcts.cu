#include <cuda.h>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>

#include "defs.hpp"
#include "mcts.cuh"
#include "hearts.hpp"

__host__ unsigned int* cuRollout(const Hearts::State& state,
                                 const Hearts::Player& ai,
                                 const uint8* cards,
                                 uint8 nCards,
                                 RolloutContainer* data) {
    return 0; // implement
}

__host__ RolloutContainer* init(uint32 rollout) {
    return 0; // implement
}

__host__ void free(RolloutContainer* data) {
    // implement
}

RolloutContainerCPP::RolloutContainerCPP(uint32 iterations) {
    data = init(iterations);
}

RolloutContainerCPP::~RolloutContainerCPP() {
    free(data);
    data = 0;
}

unsigned int* RolloutContainerCPP::_cuRollout(const Hearts::State& state,
                                              const Hearts::Player& ai,
                                              const uint8* cards,
                                              uint8 nCards) {
    return cuRollout(state, ai, cards, nCards, data);
}
