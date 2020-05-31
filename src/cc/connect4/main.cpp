#include <array>
#include <chrono>

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include "mcts.hpp"
#include "connect4.hpp"

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
typedef MCTreeDynamic<MCTSNodeBaseMT<Connect4::ActType> > TreeContainer;

// keep static_assert, might be usefull later
static_assert(std::is_same<TreeContainer, MCTreeDynamic<MCTSNodeBaseMT<Connect4::ActType> > >::value,
              "Only MCTreeDynamic<MCTSNodeBaseMT<> > can be compiled with OpenMP");
#else
typedef MCTreeDynamic<MCTSNodeBase<Connect4::ActType> > TreeContainer;
#endif

Connect4::ActType getCmdInput(const Connect4& state, int player) {
    bool valid = false;
    Connect4::ActType act;
    Connect4::ActType actions[Connect4::MaxActions];
    Connect4::ActCounterType nActions = state.getPossibleActions(player, player, actions);
    while (!valid) {
        std::string actStr;
        std::cout << "Player" << player << ": ";
        std::cin >> actStr;
        std::transform(actStr.begin(), actStr.end(), actStr.begin(), [](unsigned char c){ return ::toupper(c); });
        if (actStr.size() < 4)
            continue;
        act = Connect4::ActType(actStr[1]-'1', actStr[3]-'1');
        for (Connect4::ActCounterType i = 0; i < nActions; ++i) {
            if (act == actions[i]) {
                valid = true;
            }
        }
    }
    return act;
}

int main(int argc, char** argv) {
    int writeTree = 0;
    std::string workDir = "";
    bool isDeterministic = true;
    std::string portWhite = "tcp://localhost:5555";
    std::string portBlack = "tcp://localhost:5555";
    auto timestamp = std::time(0);
    unsigned int seed = getSeed();
    unsigned int policyIter[2] = {1600, 1600};

    if (argc == 2 && (argv[1] == std::string("-h") || argv[1] == std::string("--help"))) {
        std::cout << "Paramaters:" << std::endl;
        std::cout << "deterministic 1 (deterministic, or 0 for stochastic)" << std::endl;
        std::cout << "portW tcp://localhost:5555 (port for DNN decisions)" << std::endl;
        std::cout << "portB tcp://localhost:5555 (port for DNN decisions)" << std::endl;
        std::cout << "writeTree 0" << std::endl;
        std::cout << "workDir path/" << std::endl;
        std::cout << "seed 123" << std::endl;
        std::cout << "p0 100 (policy iteration for player0, zero for human player)" << std::endl;
        std::cout << "p[1] 100" << std::endl;
        return 0;
    }

    if (argc % 2 == 0) {
        std::cout << "Invalid input, exe key1 value1 key2 value2" << std::endl;
        return -1;
    }

    for (int i = 1; i < argc; i+=2) {
        std::string key(argv[i+0]);
        std::string val(argv[i+1]);
        if (key == "writeTree") {
            writeTree = (val != "0");
        } else if (key == "deterministic") {
            isDeterministic = (val != "0");
        } else if (key == "seed") {
            seed = std::stoi(val);
        } else if (key == "workDir") {
            workDir = val;
        } else if (key == "portW") {
            portWhite = val;
        } else if (key == "portB") {
            portBlack = val;
        } else if (key == "p0") {
            policyIter[0] = std::stoi(val);
        } else if (key == "p1") {
            policyIter[1] = std::stoi(val);
        } else {
            std::cout << "Unknown Key: " << key << std::endl;
            return -1;
        }
    }

    std::cout << "Seed " << seed << std::endl;
    std::cout << "Port White: " << portWhite << std::endl;
    std::cout << "Port Black: " << portBlack << std::endl;
    std::cout << "Deterministic: " << isDeterministic << std::endl;
    std::cout << "Results at: " << (writeTree == 0 ? "Disabled" : workDir) << std::endl;
    for (int i = 0; i < 2; ++i) {
        std::cout << "P" << i << " PIter: " << policyIter[i] << std::endl;
    }

    // init program
    zmq::context_t zmq_context(16);
    std::vector<Connect4::ActType> history;
    Connect4 state(zmq_context, portWhite, portBlack);
    std::array<MCTS<TreeContainer, Connect4>, 2> ai = {seed, seed};

    // execute game
    for (int time = 0; !state.isFinished(); ++time) {
        int player = state.getPlayer(time);
        auto t0 = std::chrono::high_resolution_clock::now();
        Connect4::ActType act = (policyIter[player] == 0)  ?
                                getCmdInput(state, player) :
                                ai[player].execute(player,
                                                   isDeterministic,
                                                   state,
                                                   policyIter[player],
                                                   history);
        auto t1 = std::chrono::high_resolution_clock::now();
        state.update(act);
        history.push_back(act);

        std::cout << "T" << time << " ";
        std::cout << "P" << int(player) << " ";
        std::cout << Connect4::act2str(act) << " ";
        std::cout << std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count() << " sec";
        std::cout << std::endl;
        std::cout << state.getBoardDescription();
        std::cout << std::endl;
    }
    std::cout << state.getEndOfGameString() << std::endl;

    return 0;
}
