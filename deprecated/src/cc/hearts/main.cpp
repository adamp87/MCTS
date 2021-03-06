#include <array>
#include <random>

#include <sstream>
#include <fstream>
#include <iostream>

#include "defs.hpp"
#include "mcts.hpp"
#include "mcts.cuh"
#include "mctree.hpp"
#include "hearts.hpp"
#include "mcts_debug.hpp"

#ifdef __linux__
#include <ctime>
#include <unistd.h>
int getSeed() {
    return ((time(NULL) & 0xFFFF) | (getpid() << 16));
}
#else
#include <ctime>
#include <windows.h>
int getSeed() {
    return ((time(NULL) & 0xFFFF) | (GetCurrentProcessId() << 16));
}
#endif

#ifdef _OPENMP
typedef MCTreeDynamic<MCTSNodeBaseMT<uint8> > TreeContainer;

// keep static_assert, might be usefull later
static_assert(std::is_same<TreeContainer, MCTreeDynamic<MCTSNodeBaseMT<uint8> > >::value,
              "Only MCTreeDynamic<MCTSNodeBaseMT<> > can be compiled with OpenMP");
#else
typedef MCTreeStaticArray<uint8, Hearts::MaxChildPerNode> TreeContainer;
#endif

//typedef MCTS_Policy_Debug PolicyDebug;
typedef MCTS_Policy_Debug_Dummy PolicyDebug;

int main(int argc, char** argv) {
    int cheat = 0;
    int writeTree = 0;
    std::string workDir = "";
    auto timestamp = std::time(0);
    unsigned int seed = getSeed();
    unsigned int policyIter[4] = {100, 1000, 10000, 100000};
    unsigned int rolloutIter[4] = {1, 1, 1, 1};

    if (argc == 2 && (argv[1] == std::string("-h") || argv[1] == std::string("--help"))) {
        std::cout << "Paramaters:" << std::endl;
        std::cout << "cheat 0" << std::endl;
        std::cout << "writeTree 0" << std::endl;
        std::cout << "workDir path/" << std::endl;
        std::cout << "seed 123" << std::endl;
        std::cout << "p0 100 (policy iteration for player0)" << std::endl;
        std::cout << "p[1,2,3] 100" << std::endl;
        std::cout << "r0 100 (rollout iteration for player0)" << std::endl;
        std::cout << "r[1,2,3] 100" << std::endl;
        return 0;
    }

    if (argc % 2 == 0) {
        std::cout << "Invalid input, exe key1 value1 key2 value2" << std::endl;
        return -1;
    }

    for (int i = 1; i < argc; i+=2) {
        std::string key(argv[i+0]);
        std::string val(argv[i+1]);
        if (key == "cheat") {
            cheat = (val != "0");
        } else if (key == "writeTree") {
            writeTree = (val != "0");
        } else if (key == "seed") {
            seed = std::stoi(val);
        } else if (key == "p0") {
            policyIter[0] = std::stoi(val);
        } else if (key == "p1") {
            policyIter[1] = std::stoi(val);
        } else if (key == "p2") {
            policyIter[2] = std::stoi(val);
        } else if (key == "p3") {
            policyIter[3] = std::stoi(val);
        } else if (key == "r0") {
            rolloutIter[0] = std::stoi(val);
        } else if (key == "r1") {
            rolloutIter[1] = std::stoi(val);
        } else if (key == "r2") {
            rolloutIter[2] = std::stoi(val);
        } else if (key == "r3") {
            rolloutIter[3] = std::stoi(val);
        } else if (key == "workDir") {
            workDir = val;
        } else {
            std::cout << "Unknown Key: " << key << std::endl;
            return -1;
        }
    }

    std::cout << "Seed " << seed << std::endl;
    std::cout << "Cheat " << cheat << std::endl;
    std::cout << "Results at: " << (writeTree == 0 ? "Disabled" : workDir) << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "P" << i << " PIter: " << policyIter[i] << " Riter: " << rolloutIter[i] << std::endl;
    }

    // init program
    std::srand(seed);
    Hearts state(cheat != 0);
    std::vector<Hearts::ActType> history;
    PolicyDebug policyDebug(writeTree, workDir, "hearts", timestamp);
    std::array<MCTS<TreeContainer, Hearts, PolicyDebug>, 4> ai;

    // int cuda rollout
    std::unique_ptr<RolloutCUDA<Hearts> > rolloutCuda(new RolloutCUDA<Hearts>(rolloutIter, seed));
    if (rolloutCuda->hasGPU()) {
        std::cout << "GPU Mode" << std::endl;
    } else {
        std::cout << "CPU Mode" << std::endl;
    }

    // print player cards
    for (int p = 0; p < 4; ++p) {
        std::cout << "P" << int(p) << " ";
        for (uint8 color = 0; color < 4; ++color) {
            for (uint8 value = 0; value < 13; ++value) {
                uint8 card = 13 * color + value;
                if (state.isCardAtPlayer(p, card)) {
                    std::cout << Hearts::act2str(card) << " ";
                }
            }
        }
        std::cout << std::endl;
    }

    // execute game
    for (uint8 round = 0; round < 13; ++round) {
        std::cout << "R" << round + 1 << " ";
        for (uint8 turn = 0; turn < 4; ++turn) {
            int player = state.getPlayer(round * 4 + turn);
            uint8 card = ai[player].execute(player,
                                            0,
                                            true,
                                            state,
                                            policyIter[player],
                                            rolloutIter[player],
                                            history,
                                            rolloutCuda.get(),
                                            policyDebug);
            state.update(card);
            history.push_back(card);

            std::cout << "P" << int(player) << " ";
            std::cout << Hearts::act2str(card) << " ";
        }
        std::cout << std::endl;
    }

    // print points
    std::array<uint8, 4> points;
    state.computePoints(points);
    for (int p = 0; p < 4; ++p) {
        std::cout << "P" << p << " " << int(points[p]) << std::endl;
    }

    // save tree
    for (int p = 0; p < 4; ++p) {
        if (writeTree == 0)
            continue;

        std::ofstream file;
        std::stringstream filename;
        filename << workDir << "hearts_" << timestamp << "_player_" << p << ".csv";

        file.open(filename.str());
        float maxIter = float(policyIter[p] * rolloutIter[p]);
        ai[p].writeResults(state, p, maxIter, history, file);
        file.close();

        { // filter results, write only first level branch nodes
            std::ifstream src;
            std::ofstream dst;
            std::string line;

            src.open(filename.str());
            filename << "_filtered.csv";
            dst.open(filename.str());
            std::getline(src, line); // get header
            dst << line << std::endl;
            while (std::getline(src, line)) {
                if (line[0] != '0')
                    continue; // skip lines that not first level branch
                dst << line << std::endl;
            }
            dst.close();
            src.close();
        }
    }

    return 0;
}
