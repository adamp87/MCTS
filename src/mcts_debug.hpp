#ifndef MCTS_DEBUG_HPP
#define MCTS_DEBUG_HPP

#include <mutex>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class MCTS_Policy_Debug_Dummy {
public:
    MCTS_Policy_Debug_Dummy(bool writeTree, const std::string&, int) {
        if (writeTree)
            std::cout << "Warning: PolicyDebug was not compiled" << std::endl;
    }

    template <class TTree, class TProblem, class NodePtr>
    void push(const TTree&,
              const TProblem&,
              const std::vector<NodePtr>&,
              const NodePtr,
              int,
              int,
              int)
    { }
};

class MCTS_Policy_Debug {

private:
    bool writeTree;
    std::mutex mutex;
    std::ofstream policyTrace;

public:
    MCTS_Policy_Debug(bool writeTree, const std::string& workDir, int seed) {
        this->writeTree = writeTree;
        std::stringstream filename;
        filename << workDir << "seed_" << seed << "_policy.csv";
        policyTrace.open(filename.str());
        policyTrace << "NodeID;ParentID;Depth;Time;Move;Iter;Win;Visit;SubRootVisit;Value;" << std::endl;
    }

    ~MCTS_Policy_Debug() {
        policyTrace.close();
    }

    template <class TTree, class TProblem, class NodePtr>
    void push(const TTree& tree,
              const TProblem& cstate,
              const std::vector<NodePtr>& policyNodes,
              const NodePtr subRoot,
              int idxAi,
              int iteration_i,
              int time)
    {
        if (!writeTree)
            return;
        std::lock_guard<std::mutex> lock(mutex); (void)lock;
        TProblem state(cstate);
        NodePtr parent = policyNodes[0];
        for (unsigned int depth = 0; depth < policyNodes.size(); ++depth) {
            NodePtr node = policyNodes[depth];

            double value = tree.value(node, log(static_cast<double>(subRoot->visits)), state.getPlayer() != idxAi, TProblem::UCT_C);
            state.update(node->move);
            policyTrace << tree.getNodeId(node) << ";";
            policyTrace << tree.getNodeId(parent) << ";";
            policyTrace << depth << ";";
            policyTrace << time << ";";
            policyTrace << TProblem::move2str(node->move) << ";";
            policyTrace << iteration_i << ";";
            policyTrace << node->wins << ";";
            policyTrace << node->visits << ";";
            policyTrace << subRoot->visits << ";";
            policyTrace << value << ";";
            policyTrace << std::endl;
            parent = node;
        }
    }
};

#endif //MCTS_DEBUG_HPP
