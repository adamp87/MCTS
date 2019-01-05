#include <array>
#include <random>

#include <sstream>
#include <fstream>
#include <iostream>

#include "defs.hpp"
#include "mcts.hpp"
#include "mctree.hpp"
#include "hearts.hpp"

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

std::string formatCard(uint8 card) {
    const char colors[] = {'C', 'D', 'S', 'H'};
    const char values[] = {'2', '3', '4', '5', '6', '7',
                           '8', '9', '0', 'J', 'Q', 'K', 'A'};

    std::string str("XX");
    str[0] = colors[card / 13];
    str[1] = values[card % 13];
    return str;
}

int main(int argc, char** argv) {
    int cheat = 0;
    int writeTree = 0;
    std::string workDir = "";
    unsigned int seed = getSeed();
    unsigned int policyIter[4] = {100, 1000, 10000, 100000};
    unsigned int rolloutIter[4] = {1, 1, 1, 1};

    if (argc == 2 && (argv[1] == "-h" || argv[1] == "--help")) {
        std::cout << "Please look in main.cpp for parameters" << std::endl;
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
    std::array<Hearts::Player, 4> players;
    std::array<MCTS<MCTreeStaticArray>, 4> ai;
    Hearts state(players);

    // cheat, play with open cards
    for (uint8 p = 0; p < 4; ++p) {
        for (uint8 color = 0; color < 4; ++color) {
            for (uint8 value = 0; value < 13; ++value) {
                uint8 card = 13 * color + value;
                for (uint8 p2 = 0; p2 < 4; ++p2) {
                    if (cheat != 0 && players[p2].hand[card] == p2) {
                        players[p].hand[card] = p2; // player see cards of other players
                    }
                }
            }
        }
    }

    // print player cards
    for (uint8 p = 0; p < 4; ++p) {
        std::cout << "P" << int(p) << " ";
        for (uint8 color = 0; color < 4; ++color) {
            for (uint8 value = 0; value < 13; ++value) {
                uint8 card = 13 * color + value;
                if (players[p].hand[card] == p) {
                    std::cout << formatCard(card) << " ";
                }
            }
        }
        std::cout << std::endl;
    }

    // execute game
    for (uint8 round = 0; round < 13; ++round) {
        std::cout << "R" << round + 1 << " ";
        for (uint8 turn = 0; turn < 4; ++turn) {
            uint8 player = state.getPlayer(round * 4 + turn);
            uint8 card = ai[player].execute(state, players[player], policyIter[player], rolloutIter[player]);
            state.update(card);

            std::cout << "P" << int(player) << " ";
            std::cout << formatCard(card) << " ";
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
        filename << workDir << "seed_" << seed << "_player_" << p << ".csv";

        file.open(filename.str());
        float maxIter = float(policyIter[p] * rolloutIter[p]);
        ai[p].writeResults(state, p, maxIter, file);

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
        }
    }

    return 0;
}
