#include <cuda.h>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>

#include <mutex>
#include <iostream>

#include "defs.hpp"
#include "mcts.cuh"
#include "hearts.hpp"

// NOTE: this is a dirty solution, compile flow.cpp for cuda here
#include "flownetwork.cpp"

struct RolloutCUDA::impl {
    Hearts* u_state;
    Hearts::Player* u_player;
    uint8* u_result; //players(4)*maxrollout
    curandState* d_rnd;

    std::mutex lock;
    constexpr static int nThread = 32;
};

__global__ void rollout(const Hearts* u_state,
                        const Hearts::Player* u_player,
                        curandState* d_rnd,
                        uint8* u_result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint8 cards[52];
    Hearts state(*u_state);
    Hearts::Player player(*u_player);

    // Copy state to local memory for efficiency
    curandState localRnd = d_rnd[idx];

    while (!state.isGameOver()) {
        uint8 count = state.getPossibleCards(player, cards);
        uint8 pick = curand(&localRnd) % count;
        state.update(cards[pick]);
    }

    // Copy state back to global memory
    d_rnd[idx] = localRnd;

    state.computePoints(u_result + 4 * idx);
}

__global__ void setup_random(curandState* state, unsigned int seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(seed, id, 0, &state[id]);
}

RolloutCUDA::RolloutCUDA(unsigned int* iterations, unsigned int seed) {
    pimpl = 0;
    int deviceCount = 0;
    if(cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
        std::cout << "Failed to get device count" << std::endl;
        return;
    }
    if (deviceCount == 0) {
        std::cout << "No CUDA device found" << std::endl;
        return;
    }

    unsigned int maxIterations = iterations[0];
    for (int i = 0; i < 4; ++i) {
        if (iterations[i] == 1)
            continue; // cuda rollout disabled for player
        if (iterations[i] % RolloutCUDA::impl::nThread != 0) {
            std::cout << "Rollout " << i << " is not dividable with " << RolloutCUDA::impl::nThread << std::endl;
            return;
        }
        if (maxIterations < iterations[i]) {
            maxIterations = iterations[i];
        }
    }

    std::unique_ptr<RolloutCUDA::impl> ptr(new RolloutCUDA::impl());
    dim3 threads(RolloutCUDA::impl::nThread);
    dim3 blocks(maxIterations / threads.x);
    if(cudaMallocManaged(&ptr->u_state, sizeof(Hearts)) != cudaSuccess) {
        std::cout << "Failed to allocate state" << std::endl;
        return;
    }
    if(cudaMallocManaged(&ptr->u_player, sizeof(Hearts::Player)) != cudaSuccess) {
        std::cout << "Failed to allocate player" << std::endl;
        return;
    }
    if(cudaMalloc(&ptr->d_rnd, sizeof(curandState) * maxIterations) != cudaSuccess) {
        std::cout << "Failed to allocate random" << std::endl;
        return;
    }
    if(cudaMallocManaged(&ptr->u_result, sizeof(uint8) * 4 * maxIterations) != cudaSuccess) {
        std::cout << "Failed to allocate results" << std::endl;
        return;
    }

    setup_random<<<blocks, threads>>>(ptr->d_rnd, seed);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cout << "Failed to sync CUDA call" << std::endl;
        return;
    }

    pimpl = ptr.release();
}

RolloutCUDA::~RolloutCUDA() {
    if (pimpl == 0)
        return;
    cudaFree(pimpl->d_rnd);
    cudaFree(pimpl->u_state);
    cudaFree(pimpl->u_player);
    cudaFree(pimpl->u_result);
    delete pimpl;
    pimpl = 0;
}

__host__ bool RolloutCUDA::cuRollout(const Hearts& state,
                                     const Hearts::Player& player,
                                     unsigned int iterations,
                                     std::vector<uint8>& points) const {
    std::unique_lock<std::mutex> lock(pimpl->lock, std::defer_lock);
    if (lock.try_lock() == false)
        return false;
    dim3 threads(RolloutCUDA::impl::nThread);
    dim3 blocks(iterations / threads.x);
    *pimpl->u_state = state;
    *pimpl->u_player = player;
    rollout<<<blocks, threads>>>(pimpl->u_state,
                                 pimpl->u_player,
                                 pimpl->d_rnd,
                                 pimpl->u_result);
    if(cudaDeviceSynchronize() != cudaSuccess) {
        std::cout << "Failed to sync CUDA call" << std::endl;
        return false;
    }

    points.clear();
    points.insert(points.end(), pimpl->u_result, pimpl->u_result + 4 * iterations);
    return true;
}
