#ifndef MCTS_DEBUG_HPP
#define MCTS_DEBUG_HPP

#include <mutex>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class MCTS_Policy_Debug_Dummy {
public:
    MCTS_Policy_Debug_Dummy(bool writeTree, const std::string&, const std::string&, int) {
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
    MCTS_Policy_Debug(bool writeTree, const std::string& workDir, const std::string& prog, int timestamp) {
        this->writeTree = writeTree;
        if (!writeTree)
            return;
        std::stringstream filename;
        filename << workDir << prog << "_" << timestamp << "_policy.csv";
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
        std::stringstream txt;
        TProblem state(cstate);
        NodePtr parent = policyNodes[0];
        for (unsigned int depth = 1; depth < policyNodes.size(); ++depth) {
            //NOTE: must start from 1, 0 is the subroot and it should not be used to update
            NodePtr node = policyNodes[depth];

            double value = tree.value(node, log(static_cast<double>(subRoot->visits)), state.getPlayer() != idxAi, TProblem::UCT_C);
            state.update(node->move);
            txt << tree.getNodeId(node) << ";";
            txt << tree.getNodeId(parent) << ";";
            txt << depth << ";";
            txt << time << ";";
            txt << TProblem::move2str(node->move) << ";";
            txt << iteration_i << ";";
            txt << node->wins << ";";
            txt << node->visits << ";";
            txt << subRoot->visits << ";";
            txt << value << ";";
            txt << std::endl;
            parent = node;
        }

        { // thread-safe scope
            std::lock_guard<std::mutex> lock(mutex); (void)lock;
            policyTrace << txt.str();
        }
    }
};

#endif //MCTS_DEBUG_HPP
