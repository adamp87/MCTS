#ifndef HEARTS_HPP
#define HEARTS_HPP

#include <array>
#include <vector>

#include <random>
#include <numeric>
#include <algorithm>

#include "defs.hpp"
#include "flownetwork.hpp"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

//! Stores the state of the game that is known by all players
/*!
 * \details This class implements the function getPossibleCards for applying AI.
 *          This function returns those cards, which can be played in the next turn without breaking the rules.
 *          It can find cards for both opponent and for itself.
 *          Other functionalities are helping to simulate a game and to get the number of points at the end of the game.
 *          The rules of the game are described in Wikipedia under "Hearts (Card Game)".
 * \author adamp87
*/
class Hearts {
public:
    //! Stores the cards that only one player knows
    struct Player {
        uint8 player; //!< fixed player id
        uint8 hand[52]; //!< known cards: index card-value playerid; can store info after swap/open cards
    };

private:
    uint8 turn; //!< current turn, one round has 4 turns
    uint8 round; //!< current round, one game has 13 rounds
    uint8 orderInTime[52]; //!< game state: index time-value card
    uint8 orderAtCard[52]; //!< game state: index card-value time
    uint8 orderPlayer[52]; //!< game state: index time-value player

    friend class FlowNetwork;

    static constexpr uint8 order_unset = 255; //!< static variable for unset orders and cards

    //! Set the players order for the next round
    CUDA_CALLABLE_MEMBER void setPlayerOrder(uint8 player) {
        const uint8 cplayer_map[4][4] = { { 0, 1, 2, 3 },
                                          { 1, 2, 3, 0 },
                                          { 2, 3, 0, 1 },
                                          { 3, 0, 1, 2 } };
        for (uint8 i = 0; i < 4; ++i) {
            orderPlayer[round * 4 + i] = cplayer_map[player][i];
        }
    }

    //! Return player who has to take cards at end of turn
    CUDA_CALLABLE_MEMBER uint8 getPlayerToTakeCards(uint8 round) const {
        uint8 player = orderPlayer[round * 4];
        uint8 color_first = orderInTime[round * 4] / 13;
        uint8 highest_value = orderInTime[round * 4] % 13;
        for (uint8 i = 1; i < 4; ++i) {
            uint8 color_next = orderInTime[round * 4 + i] / 13;
            uint8 value_next = orderInTime[round * 4 + i] % 13;
            if (color_first == color_next && highest_value < value_next) {
                highest_value = value_next;
                player = orderPlayer[round * 4 + i];
            }
        }
        return player;
    }

public:
    //! Shuffle deck and set initial state
    Hearts(std::array<Player, 4>& players) {
        turn = 0;
        round = 0;
        uint8 unset = Hearts::order_unset;
        std::fill(orderPlayer, orderPlayer + 52, unset);
        std::fill(orderInTime, orderInTime + 52, unset);
        std::fill(orderAtCard, orderAtCard + 52, unset);

        // shuffle deck
        std::vector<uint8> cards(52);
        std::iota(cards.begin(), cards.end(), 0);
        std::random_shuffle(cards.begin(), cards.end());

        // set cards
        uint8 startPlayer = 0;
        for (uint8 i = 0; i < 4; ++i) {
            players[i].player = i;
            std::fill(players[i].hand, players[i].hand + 52, unset);//set to unknown
            for (uint8 j = 0; j < 13; ++j) {
                uint8 card = cards[i * 13 + j];
                players[i].hand[card] = i; //set card to own id
            }
            if (players[i].hand[0] == i) { // set first player
                startPlayer = i;
            }
        }
        setPlayerOrder(startPlayer);
    }

    uint8 getPlayer(uint8 time = 255) const {
        if (time == 255)
            time = round * 4 + turn;
        return orderPlayer[time];
    }

    uint8 getCardAtTime(uint8 time) const {
        return orderInTime[time];
    }

    uint8 isCardAtTimeValid(uint8 time) const {
        return orderInTime[time] != order_unset;
    }

    CUDA_CALLABLE_MEMBER bool isGameOver() const {
        return round == 13;
    }

    //! Implements game logic, return the possible cards that next player can play
    CUDA_CALLABLE_MEMBER uint8 getPossibleCards(const Player& ai, uint8* cards) const {
        uint8 count = 0;
        uint8 player = orderPlayer[round * 4 + turn]; // possible cards for this player
        uint8 colorFirst = orderInTime[round * 4] / 13; //note, must check if not first run

        if (round == 0 && turn == 0) { // starting card
            cards[0] = 0;
            return 1;
        }

        if (player == ai.player) { // own
            // check if has color
            bool aiHasNoColor[4];
            for (uint8 color = 0; color < 4; ++color) {
                aiHasNoColor[color] = true;
                for (uint8 value = 0; value < 13; ++value) {
                    uint8 card = color * 13 + value;
                    if (ai.hand[card] == ai.player && orderAtCard[card] == Hearts::order_unset) {
                        aiHasNoColor[color] = false;
                        break;
                    }
                }
            }

            // check if hearts broken
            bool heartsBroken = false;
            for(uint8 time = 0; time < round * 4 + turn; ++time){
                uint8 color = orderInTime[time] / 13;
                if (color == 3) {
                    heartsBroken = true;
                    break;
                }
            }

            // select cards
            for (uint8 color = 0; color < 4; ++color) {
                if (aiHasNoColor[color] == true)
                    continue; // ai has no card of this color
                if (turn != 0 && color != colorFirst && aiHasNoColor[colorFirst] == false)
                    continue; // must play same color
                if (round == 0 && color == 3 && !(aiHasNoColor[0] && aiHasNoColor[1] && aiHasNoColor[2]))
                    continue; // in first round no hearts (only if he has only hearts, quite impossible:))
                if (turn == 0 && color == 3 && !(heartsBroken || (aiHasNoColor[0] && aiHasNoColor[1] && aiHasNoColor[2])))
                    continue; // if hearts not broken or has other color, no hearts as first card
                for (uint8 value = 0; value < 13; ++value) {
                    if (round == 0 && color == 2 && value == 10)
                        continue; // in first round no queen

                    uint8 card = color * 13 + value;
                    if (orderAtCard[card] != order_unset)
                        continue; //card has been played

                    if (ai.hand[card] == ai.player) {
                        cards[count++] = card; // select card
                    }
                }
            }
        }
        else { // opponent

            // check if it is know if opponent has color (swap, open cards)
            bool knownToHaveColor[4] = { false };
            for (uint8 color = 0; color < 4; ++color) {
                for (uint8 value = 0; value < 13; ++value) {
                    uint8 card = color * 13 + value;
                    if (orderAtCard[card] != order_unset)
                        continue; //card has been played
                    if (ai.hand[card] == player) {
                        knownToHaveColor[color] = true;
                        break;
                    }
                }
            }

            // check if hearts are broken
            // check players, if they played other color than the one on the first turn
            bool heartsBroken = false;
            bool hasNoColor[4 * 4] = { false };
            for (uint8 time = 0; time < round * 4 + turn; ++time) {
                uint8 _round = time / 4;
                uint8 _turn = time % 4;
                uint8 player = orderPlayer[time];
                uint8 firstColor = orderInTime[_round * 4] / 13;
                uint8 nextColor = orderInTime[_round * 4 + _turn] / 13;
                if (_turn == 0 && firstColor == 3 && heartsBroken == false) {
                    // player starts with hearts and breaks hearts, he has only hearts
                    for (uint8 i = 0; i < 3; ++i) {
                        hasNoColor[player * 4 + i] = true;
                    }
                }
                if (firstColor != nextColor) {
                    // player placed different color
                    hasNoColor[player * 4 + firstColor] = true;
                }
                // must be checked last
                heartsBroken |= (nextColor == 3);
            }

            FlowNetwork flow(*this, ai.player, ai.hand, hasNoColor);

            for (uint8 color = 0; color < 4; ++color) {
                if (hasNoColor[player * 4 + color] == true)
                    continue; // ai has no card of this color
                if (turn != 0 && color != colorFirst && knownToHaveColor[colorFirst] == true)
                    continue; // it is known that ai has card of the first color
                if (turn == 0 && color == 3 && !heartsBroken && (knownToHaveColor[0] || knownToHaveColor[1] || knownToHaveColor[2]))
                    continue; // player has other color therefore cant start with hearth
                // theoretically opponent can play any color (he could play hearts even in first round(if he has no other color))

                // verify if game becomes invalid by playing a card from the given color
                if (knownToHaveColor[color] == false) {
                    if (flow.verifyOneCard(player, color) == false)
                        continue;
                }

                // verify if game becomes invalid by playing hearts as first card
                if (turn == 0 && color == 3 && !heartsBroken && knownToHaveColor[3] == false) {
                    if (flow.verifyHeart(player) == false)
                        continue;
                }

                // verify if game becomes invalid if not the first color is played
                if (turn != 0 && color != colorFirst && knownToHaveColor[color] == false) {
                    if (flow.verifyOneColor(player, colorFirst) == false)
                        continue;
                }

                for (uint8 value = 0; value < 13; ++value) {
                    if (round == 0 && color == 2 && value == 10)
                        continue; // in first round no queen

                    uint8 card = color * 13 + value;
                    if (orderAtCard[card] != order_unset)
                        continue; //card has been played

                    if (ai.hand[card] == player || ai.hand[card] == Hearts::order_unset) {
                        cards[count++] = card; // select from known or unknown opponent cards
                    }
                }
            }
        }
        return count;

        //TODO: filter cards (values next to each other), dont forget result is the same for all, dont discard here?
    }

    //! Update the game state according to card
    CUDA_CALLABLE_MEMBER void update(uint8 card) {
        //set card order
        uint8 time = round * 4 + turn;
        orderInTime[time] = card;
        orderAtCard[card] = time;

        //next player
        turn += 1;
        if (turn == 4) { //end of round
            uint8 startPlayer = getPlayerToTakeCards(round);
            round += 1;
            turn = 0;
            if (round != 13)
                setPlayerOrder(startPlayer);
        }
    }

    //! Compute points for each player
    void computePoints(std::array<uint8, 4>& points) const {
        computePoints(points.data());
    }

    //! Compute points for each player
    CUDA_CALLABLE_MEMBER void computePoints(uint8* points) const {
        const uint8 value_map[52] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

        std::fill(points, points + 4, 0);
        for (uint8 round = 0; round < 13; ++round) {
            uint8 point = 0;
            uint8 playerToTake = getPlayerToTakeCards(round);
            for (uint8 turn = 0; turn < 4; ++turn) {
                uint8 card = orderInTime[round * 4 + turn];
                point += value_map[card]; // add points
            }
            points[playerToTake] += point;
        }
    }
};

#endif //HEARTS_HPP
