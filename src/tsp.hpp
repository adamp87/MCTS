#ifndef TSP_HPP
#define TSP_HPP

#include <string>
#include <vector>
#include <istream>

#include <atomic>
#include <numeric>
#include <algorithm>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class TSP_Base {
protected:
    static std::vector<double> edgeWeights; // NOTE: only one object can be created

public:
    //! Interface
    CUDA_CALLABLE_MEMBER int getPlayer() const {
        return 0;
    }
    CUDA_CALLABLE_MEMBER int getPlayer(int) const {
        return 0;
    }
};

//! TODO
/*!
 * \details This class implements the function getPossibleMoves for applying AI.
 *          This function returns those moves, which can be played in the next turn without breaking the rules.
 *          Other functionalities are helping to simulate a game and to get the an evaluation of the current state.
 * \author adamp87
*/
class TSP_Vertex : public TSP_Base {
public:
    typedef std::uint_fast16_t MoveType; //!< interface
    typedef std::uint_fast16_t MoveCounterType; //!< interface
    constexpr static double UCT_C = 1.4; //!< interface, constant for exploration in uct formula
    constexpr static unsigned int MaxMoves = 127; //!< interface
    constexpr static unsigned int MaxChildPerNode = 127; //!< interface, TODO

private:
    double lb;
    double ub;
    double* weights;
    int nodeCount;

    int visitedCount;
    int tour[MaxMoves]; // TODO: dynamic
    bool visited[MaxMoves];

public:
    //! Set initial state
    TSP_Vertex(const std::string& filename)
        : visited{false} {
        visitedCount = 1;
        tour[0] = 0; // artificial root

        // open the binary file created by python script
        std::ifstream file(filename, std::ios::binary);

        // read the data
        std::vector<char> dataCh((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
        double* data = (double*)dataCh.data();

        // set up values
        nodeCount = static_cast<int>(data[0]);
        lb = 0;//data[1];
        ub = data[1]*2;//data[2];
        edgeWeights.insert(edgeWeights.end(), data+3, data+3+nodeCount*nodeCount);
        weights = &edgeWeights[0];

        if (MaxMoves < nodeCount) {
            // TODO
            throw -1;
        }
    }

    //! Interface
    CUDA_CALLABLE_MEMBER bool isFinished() const {
        return visitedCount == nodeCount;
    }

    //! Interface, Implements game logic, return the possible moves that idxAi can play
    /*!
    * \param idxMe ID of player who executes function
    * \param idxAi ID of player to get possible moves for
    * \param moves Allocated array to store possible moves
    * \return Number of possible moves
    * \note Using hands[idxAi] is hard-coded cheating
    */
    CUDA_CALLABLE_MEMBER MoveCounterType getPossibleMoves(int, int, MoveType* moves) const {
        MoveCounterType nMoves = 0;
        for (int i = 1; i < nodeCount; ++i) {
            if (!visited[i])
                moves[nMoves++] = i;
        }
        return nMoves;
    }

    //! Interface, Update the game state according to move
    CUDA_CALLABLE_MEMBER void update(MoveType& move) {
        visited[move] = true;
        tour[visitedCount++] = move;
    }

    //! Interface, Compute win value for MCTreeSearch, between 0-1
    CUDA_CALLABLE_MEMBER double computeMCTSWin(int) const {
        double sum = getTourLength();
        //double win = 1.0-exp((sum-lb)/(ub-lb)); //exponential
        double win = (ub-sum)/(ub-lb); // linear
        return win;
    }

    ///! Interface, convert move to string
    static std::string move2str(MoveType& move) {
        return std::to_string(move);
    }

    double getTourLength() const {
        double sum = 0.0;
        for (int i = 1; i < visitedCount; ++i) {
            int v1 = tour[i-1];
            int v2 = tour[i];
            sum += weights[v1*nodeCount+v2];
        }
        if (nodeCount == visitedCount) {
            int v1 = tour[1];
            int v2 = tour[nodeCount-1];
            sum += weights[v1*nodeCount+v2];
        }
        return sum;
    }
};

//! TODO
/*!
 * \details This class implements the function getPossibleMoves for applying AI.
 *          This function returns those moves, which can be played in the next turn without breaking the rules.
 *          Other functionalities are helping to simulate a game and to get the an evaluation of the current state.
 * \author adamp87
*/
class TSP_Edge : public TSP_Base {
public:
    //!< interface
    struct MoveType {
        int v1;
        int v2;

        CUDA_CALLABLE_MEMBER MoveType():v1(0),v2(0) {}
        CUDA_CALLABLE_MEMBER MoveType(int v1,int v2):v1(v1),v2(v2) {}

        bool operator==(const MoveType& other) const {
            return v1 == other.v1 &&
                   v2 == other.v2;
        }
        operator std::string() const {
            return std::to_string(v1) + " " + std::to_string(v2);
        }
    };

    typedef std::uint_fast16_t MoveCounterType; //!< interface
    constexpr static double UCT_C = 1.4; //!< interface, constant for exploration in uct formula
    constexpr static unsigned int MaxMoves = 1028; //!< interface
    constexpr static unsigned int MaxChildPerNode = 1028; //!< interface, TODO

private:
    double lb;
    double ub;
    double* weights;
    int nodeCount;
    int vIn[MaxMoves];
    int vOut[MaxMoves]; // TODO: dynamic

public:
    //! Set initial state
    TSP_Edge(const std::string& filename)
        : vIn{0}, vOut{0} {
        // open the binary file created by python script
        std::ifstream file(filename, std::ios::binary);

        // read the data
        std::vector<char> dataCh((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
        double* data = (double*)dataCh.data();

        // set up values
        nodeCount = static_cast<int>(data[0]);
        lb = 0;//data[1];
        ub = data[1]*2;//data[2];
        edgeWeights.insert(edgeWeights.end(), data+3, data+3+nodeCount*nodeCount);
        weights = &edgeWeights[0];

        if (MaxMoves < nodeCount) {
            // TODO
            throw -1;
        }
    }

    //! Interface
    CUDA_CALLABLE_MEMBER bool isFinished() const {
        for (int i = 1; i < nodeCount; ++i) {
            if (vIn[i] == 0)
                return false;
        }
        return true;
    }

    //! Interface, Implements game logic, return the possible moves that idxAi can play
    /*!
    * \param idxMe ID of player who executes function
    * \param idxAi ID of player to get possible moves for
    * \param moves Allocated array to store possible moves
    * \return Number of possible moves
    * \note Using hands[idxAi] is hard-coded cheating
    */
    CUDA_CALLABLE_MEMBER MoveCounterType getPossibleMoves(int, int, MoveType* moves) const {
        MoveCounterType nMoves = 0;
        for (int i = 1; i < nodeCount; ++i) {
            if (vIn[i] != 0)
                continue;
            for (int j = 1; j < nodeCount; ++j) {
                if (i == j)
                    continue;
                if (vOut[j] != 0)
                    continue;

                // check loop
                int idx = i;
                int loopSize = 0;
                bool hasLoop = false;
                while (vOut[idx] != 0) {
                    ++loopSize;
                    idx = vOut[idx];
                    if (idx == j) {
                        hasLoop = true;
                        break;
                    }
                }
                if (!hasLoop || loopSize == nodeCount-2)
                    moves[nMoves++] = MoveType(i, j);
            }
        }

        return nMoves;
    }

    //! Interface, Update the game state according to move
    CUDA_CALLABLE_MEMBER void update(MoveType& move) {
        vIn[move.v1] = move.v2;
        vOut[move.v2] = move.v1;
    }

    //! Interface, Compute win value for MCTreeSearch, between 0-1
    CUDA_CALLABLE_MEMBER double computeMCTSWin(int) const {
        double sum = 0.0;
        for (int i = 1; i < nodeCount; ++i) {
            if (vIn[i] == 0)
                continue;
            int v1 = i;
            int v2 = vIn[i];
            sum += weights[v1*nodeCount+v2];
        }
        //double win = 1.0-exp((sum-lb)/(ub-lb)); //exponential
        double win = (ub-sum)/(ub-lb); // linear

        return win;
    }

    ///! Interface, convert move to string
    static std::string move2str(MoveType& move) {
        return static_cast<std::string>(move);
    }

    double getTourLength() const {
        double sum = 0.0;
        for (int i = 1; i < nodeCount; ++i) {
            if (vIn[i] == 0)
                continue;
            int v1 = i;
            int v2 = vIn[i];
            sum += weights[v1*nodeCount+v2];
        }
        return sum;
    }
};


#endif //TSP_HPP
