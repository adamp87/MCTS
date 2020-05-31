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
 * \details This class implements the function getPossibleActions for applying AI.
 *          This function returns those actions, which can be played in the next turn without breaking the rules.
 *          Other functionalities are helping to simulate a game and to get the an evaluation of the current state.
 * \author adamp87
*/
class TSP_Vertex : public TSP_Base {
public:
    typedef std::uint_fast16_t ActType; //!< interface
    typedef std::uint_fast16_t ActCounterType; //!< interface
    constexpr static double UCT_C = 1.4; //!< interface, constant for exploration in uct formula
    constexpr static unsigned int MaxActions = 127; //!< interface
    constexpr static unsigned int MaxChildPerNode = 127; //!< interface, TODO
    constexpr static double DirichletAlpha = 0.3; //!< interface

private:
    double lb;
    double ub;
    double* weights;
    int nodeCount;

    int visitedCount;
    int tour[MaxActions]; // TODO: dynamic
    bool visited[MaxActions];

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

        if (MaxActions < nodeCount) {
            // TODO
            throw -1;
        }
    }

    //! Interface
    CUDA_CALLABLE_MEMBER bool isFinished() const {
        return visitedCount == nodeCount;
    }

    //! Interface, Implements game logic, return the possible actions that idxAi can play
    /*!
    * \param idxMe ID of player who executes function
    * \param idxAi ID of player to get possible action for
    * \param actions Allocated array to store possible actions
    * \return Number of possible actions
    */
    CUDA_CALLABLE_MEMBER ActCounterType getPossibleActions(int, int, ActType* actions) const {
        ActCounterType nActions = 0;
        for (int i = 1; i < nodeCount; ++i) {
            if (!visited[i])
                actions[nActions++] = i;
        }
        return nActions;
    }

    //! Interface, Update the game state according to action
    CUDA_CALLABLE_MEMBER void update(ActType& action) {
        visited[action] = true;
        tour[visitedCount++] = action;
    }

    //! Interface, Compute W and P values for MCTS
    //! * \param idxAi ID of player who executes function
    CUDA_CALLABLE_MEMBER void computeMCTS_WP(int idxAi, ActType*, ActCounterType nActions, double* P, double& W) const {
        for (ActCounterType i = 0; i < nActions; ++i)
            P[i] = 1;
        W = computeMCTS_W(idxAi);
    }

    //! Interface, Compute win value for MCTreeSearch, between 0-1
    CUDA_CALLABLE_MEMBER double computeMCTS_W(int) const {
        double sum = getTourLength();
        //double win = 1.0-exp((sum-lb)/(ub-lb)); //exponential
        double win = (ub-sum)/(ub-lb); // linear
        return win;
    }

    ///! Interface, convert action to string
    static std::string act2str(ActType& action) {
        return std::to_string(action);
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

    void getGameStateDNN(std::vector<float>&, int) const {

    }

    void getPolicyTrainDNN(std::vector<float>&, int, std::vector<std::pair<ActType, double> >&) const {

    }

    void storeGamePolicyDNN(std::vector<float>&, std::vector<float>&) const {

    }
};

//! TODO
/*!
 * \details This class implements the function getPossibleAxtions for applying AI.
 *          This function returns those actions, which can be played in the next turn without breaking the rules.
 *          Other functionalities are helping to simulate a game and to get the an evaluation of the current state.
 * \author adamp87
*/
class TSP_Edge : public TSP_Base {
public:
    //!< interface
    struct ActType {
        int v1;
        int v2;

        CUDA_CALLABLE_MEMBER ActType():v1(0),v2(0) {}
        CUDA_CALLABLE_MEMBER ActType(int v1,int v2):v1(v1),v2(v2) {}

        bool operator==(const ActType& other) const {
            return v1 == other.v1 &&
                   v2 == other.v2;
        }
        operator std::string() const {
            return std::to_string(v1) + " " + std::to_string(v2);
        }
    };

    typedef std::uint_fast16_t ActCounterType; //!< interface
    constexpr static double UCT_C = 1.4; //!< interface, constant for exploration in uct formula
    constexpr static unsigned int MaxActions = 1028; //!< interface
    constexpr static unsigned int MaxChildPerNode = 1028; //!< interface, TODO

private:
    double lb;
    double ub;
    double* weights;
    int nodeCount;
    int vIn[MaxActions];
    int vOut[MaxActions]; // TODO: dynamic

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

        if (MaxActions < nodeCount) {
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

    //! Interface, Implements game logic, return the possible actions that idxAi can play
    /*!
    * \param idxMe ID of player who executes function
    * \param idxAi ID of player to get possible actions for
    * \param actions Allocated array to store possible actions
    * \return Number of possible actions
    */
    CUDA_CALLABLE_MEMBER ActCounterType getPossibleActions(int, int, ActType* actions) const {
        ActCounterType nActions = 0;
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
                    actions[nActions++] = ActType(i, j);
            }
        }

        return nActions;
    }

    //! Interface, Update the game state according to action
    CUDA_CALLABLE_MEMBER void update(ActType& act) {
        vIn[act.v1] = act.v2;
        vOut[act.v2] = act.v1;
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

    ///! Interface, convert act to string
    static std::string act2str(ActType& act) {
        return static_cast<std::string>(act);
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
