#ifndef FLOWNETWORK_HPP
#define FLOWNETWORK_HPP

#include "defs.hpp"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class Hearts;

class FlowNetwork {
    static constexpr uint8 node_s = 0;
    static constexpr uint8 node_t = 9;
    static constexpr uint8 node_cA[4] = { 1, 2, 3, 4 };
    static constexpr uint8 node_pA[4] = { 5, 6, 7, 8 };
    static constexpr uint8 nodeCount = 10;

    constexpr static CUDA_CALLABLE_MEMBER uint8 get_node_c(uint8 idx) { return node_cA[idx]; }
    constexpr static CUDA_CALLABLE_MEMBER uint8 get_node_p(uint8 idx) { return node_pA[idx]; }

    static CUDA_CALLABLE_MEMBER void get_node(uint8* c, uint8* p) {
        c[0] = get_node_c(0); c[1] = get_node_c(1); c[2] = get_node_c(2); c[3] = get_node_c(3);
        p[0] = get_node_p(0); p[1] = get_node_p(1); p[2] = get_node_p(2); p[3] = get_node_p(3);
    }

    CUDA_CALLABLE_MEMBER static uint8 getEdge(uint8 i, uint8 j) {
        return j*nodeCount + i; //NOTE: findPath() speed is critical: use j*n+i
    }

    CUDA_CALLABLE_MEMBER static uint8 findPath(const uint8* graph, uint8* path) {
        uint8 v = 0;
        uint8 nStack = 1;
        uint8 stack[nodeCount];
        uint8 parent[nodeCount] = { 0 };
        bool discovered[nodeCount] = { false };

        stack[0] = node_s;
        while (nStack != 0 && v != node_t) {
            v = stack[--nStack]; // pop
            if (discovered[v] == true)
                continue; // already discovered
            discovered[v] = true; // marked as discovered
            for (uint8 i = 1; i < nodeCount; ++i) { //start from 1, as source is always discovered
                if (discovered[i] == true)
                    continue; // already discovered
                if (graph[getEdge(v, i)] == 0)
                    continue; // no edge
                parent[i] = v; // set predecessor
                stack[nStack++] = i; // push
            }
        }

        if (parent[node_t] == 0) {
            return 0; // no path found
        }
        //use stack to get reverse path
        nStack = 0;
        stack[0] = parent[node_t];
        while (stack[nStack] != 0)
        { // get reverse path
            stack[nStack + 1] = parent[stack[nStack]];
            ++nStack;
        }
        //return path in normal order
        for (uint8 i = 0; i <= nStack; ++i) {
            path[i] = stack[nStack - i];
        }
        path[++nStack] = node_t; // add sink to path
        return nStack;
    }

    CUDA_CALLABLE_MEMBER static bool _verify(uint8* graph, uint8 player) {
        uint8 path[nodeCount];

        while (true) {
            uint8 pathLength = findPath(graph, path);
            if (pathLength == 0)
                break;
            // get capacity of path
            uint8 cap = 52; //set to max
            uint8 v1 = node_s;
            uint8 v2 = node_s;
            for (uint8 i = 1; i <= pathLength; ++i) {
                v1 = v2;
                v2 = path[i];
                uint8 edgeCap = graph[getEdge(v1, v2)];
                if (edgeCap < cap)
                    cap = edgeCap;
            }

            //send flow
            v1 = node_s;
            v2 = node_s;
            for (uint8 i = 1; i <= pathLength; ++i) {
                v1 = v2;
                v2 = path[i];
                uint8 e1 = getEdge(v1, v2);
                uint8 e2 = getEdge(v2, v1);
                graph[e1] -= cap;
                graph[e2] += cap;
            }
        }

        // if not all source edges are saturaded, game would be invalid
        return graph[getEdge(node_s, get_node_c(0))] == 0
            && graph[getEdge(node_s, get_node_c(1))] == 0
            && graph[getEdge(node_s, get_node_c(2))] == 0
            && graph[getEdge(node_s, get_node_c(3))] == 0;
    }

public:
    typedef uint8 Graph[nodeCount * nodeCount];

    CUDA_CALLABLE_MEMBER static void init(const Hearts& state, uint8 ai, const uint8* ai_hand, const bool* hasNoColor, Graph& graph);

    CUDA_CALLABLE_MEMBER static bool verifyOneColor(const uint8* _graph, uint8 player, uint8 color) {
        Graph graph;
        uint8 node_c[4];
        uint8 node_p[4];
        get_node(node_c, node_p);
        for (uint8 i = 0; i < nodeCount * nodeCount; ++i) {
            graph[i] = _graph[i];
        }
        graph[getEdge(node_c[color], node_p[player])] = 0; // deactivate edge

        return _verify(graph, player);
    }

    CUDA_CALLABLE_MEMBER static bool verifyOneCard(const uint8* _graph, uint8 player, uint8 color) {
        Graph graph;
        uint8 node_c[4];
        uint8 node_p[4];
        get_node(node_c, node_p);
        for (uint8 i = 0; i < nodeCount * nodeCount; ++i) {
            graph[i] = _graph[i];
        }
        if (graph[getEdge(node_p[player], node_t)] == 0) //TODO:should not happen (keep this?)
            return false;
        if (graph[getEdge(node_s, node_c[color])] == 0)
            return false; // all unknown cards of this color have been played
        graph[getEdge(node_s, node_c[color])] -= 1; // decrement unknow cards for color
        graph[getEdge(node_p[player], node_t)] -= 1; // decrement unknow cards for player

        return _verify(graph, player);
    }

    CUDA_CALLABLE_MEMBER static bool verifyHeart(const uint8* _graph, uint8 player) {
        Graph graph;
        uint8 node_c[4];
        uint8 node_p[4];
        get_node(node_c, node_p);
        for (uint8 i = 0; i < nodeCount * nodeCount; ++i) {
            graph[i] = _graph[i];
        }
        for (uint8 i = 0; i < 3; ++i)
            graph[getEdge(node_c[i], node_p[player])] = 0; // deactivate all edges, except heart

        return _verify(graph, player);
    }

    static void printGraph(const uint8* graph) {
        for (uint8 i = 0; i < nodeCount; ++i) {
            for (uint8 j = 0; j < nodeCount; ++j) {
                if (graph[getEdge(i, j)] == 0)
                    continue;
                //std::cout << "(" << i << "," << j << "):" << graph[getEdge(i, j)] << std::endl;
            }
        }
    }
};

#endif //FLOWNETWORK_HPP
