#include <cuda.h>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>

#include <iostream>

#include "defs.hpp"
#include "mcts.cuh"
#include "hearts.hpp"

struct RolloutContainer {
    struct Input {
        Hearts::State state;
        Hearts::Player ai;
        uint8 cards[52];
        uint8 nCards;
    };

    Input* u_input;
    curandState* d_rnd;
    unsigned int* u_result; //52*28- cards*wincount, atomic

    unsigned int threadCount;
    unsigned int blockCountPerCard;
};

__global__ void rollout(const RolloutContainer::Input* src,
                        curandState* rnd,
                        unsigned int* dst) {
    uint cardIdx = blockIdx.x / (gridDim.x / src->nCards);
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint8 cards[52];
    Hearts::Player ai(src->ai);
    Hearts::State state(src->state);
    Hearts::update(state, src->cards[cardIdx]);

    // Copy state to local memory for efficiency
    curandState localRnd = rnd[idx];

    while (!state.isTerminal()) {
        uint8 count = Hearts::getPossibleCards(state, ai, cards);
        uint8 pick = curand(&localRnd) % count;
        Hearts::update(state, cards[pick]);
    }

    // Copy state back to global memory
    rnd[idx] = localRnd;

    uint8 points[4];
    Hearts::computePoints(state, points);
    uint* address = dst + cardIdx * 28 + Hearts::mapPoints2Wins(ai, points);
    atomicAdd(address, 1);
}

__global__ void setup_random(curandState* state, unsigned int seed) {
    uint id = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(seed, id, 0, &state[id]);
}

__host__ unsigned int* cuRollout(const Hearts::State& state,
                                 const Hearts::Player& ai,
                                 const uint8* cards,
                                 uint8 nCards,
                                 RolloutContainer* data) {
    data->u_input->state = state;
    data->u_input->ai = ai;
    std::copy(cards, cards + nCards, data->u_input->cards);
    data->u_input->nCards = nCards;
    std::fill(data->u_result, data->u_result + 52 * 28, 0);

    dim3 blocks(data->blockCountPerCard * nCards);
    dim3 threads(data->threadCount);
    rollout<<<blocks, threads>>>(data->u_input, data->d_rnd, data->u_result);
    cudaDeviceSynchronize();

    return data->u_result;
}

__host__ RolloutContainer* initData(uint32 rollout, unsigned int seed) {
    int deviceCount = 0;
    if(cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
        std::cout << "Failed to get device count" << std::endl;
        return 0;
    }
    if (deviceCount == 0) {
        std::cout << "No CUDA device found" << std::endl;
        return 0;
    }

    if ((rollout & (rollout - 1)) != 0) {
        std::cout << "Not power of two" << std::endl;
        return 0;
    }

    std::unique_ptr<RolloutContainer> data(new RolloutContainer());
    data->threadCount = 32;
    data->blockCountPerCard = rollout / data->threadCount;
    uint maxThreads = data->blockCountPerCard * data->threadCount * 52;
    if(cudaMallocManaged(&data->u_input, sizeof(RolloutContainer::Input)) != cudaSuccess) {
        std::cout << "Failed to allocate input" << std::endl;
        return 0;
    }
    if(cudaMalloc(&data->d_rnd, sizeof(curandState) * maxThreads) != cudaSuccess) {
        std::cout << "Failed to allocate random" << std::endl;
        return 0;
    }
    if(cudaMallocManaged(&data->u_result, sizeof(unsigned int) * 52 * 28) != cudaSuccess) {
        std::cout << "Failed to allocate results" << std::endl;
        return 0;
    }

    setup_random<<<data->blockCountPerCard * 52, data->threadCount>>>(data->d_rnd, seed);
    cudaDeviceSynchronize();

    return data.release();
}

__host__ void freeData(RolloutContainer* data) {
    if (data == 0)
        return;
    cudaFree(data->d_rnd);
    cudaFree(data->u_input);
    cudaFree(data->u_result);
}

RolloutContainerCPP::RolloutContainerCPP(uint32 iterations, unsigned int seed) {
    data = initData(iterations, seed);
}

RolloutContainerCPP::~RolloutContainerCPP() {
    freeData(data);
    data = 0;
}

unsigned int* RolloutContainerCPP::_cuRollout(const Hearts::State& state,
                                              const Hearts::Player& ai,
                                              const uint8* cards,
                                              uint8 nCards) {
    return cuRollout(state, ai, cards, nCards, data);
}
