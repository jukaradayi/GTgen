#include <stdexcept>
#include<iostream>

#include <unordered_set>
#include <utility>
#include <vector>
#include "edge_swap.hpp"

namespace swapper {


    EdgeSwap::EdgeSwap () {}

    EdgeSwap::EdgeSwap (std::vector<swapper::Edge> edge_vect, std::unordered_set<swapper::Edge> edge_set) {
    //EdgeSwap::EdgeSwap (std::unordered_set<swapper::Edge> edge_set) {

        this->edge_vect = edge_vect;
        this->edge_set = edge_set;

    }

    EdgeSwap::~EdgeSwap () {}
    
     
    // check if two edges share a node
    bool EdgeSwap::shareNode (swapper::Edge edge1, swapper::Edge edge2) {
        if (edge1.u == edge2.u || edge1.u == edge2.v) {
            return true;
        } else if (edge1.u == edge2.u || edge1.v == edge2.v) {
            return true;
        } else {
            return false;
        }
    }
    
    // replace edges
    void EdgeSwap::replaceEdge_Array(int edge1_idx, int edge2_idx, swapper::Edge new_edge1, swapper::Edge new_edge2){
        edge_vect[edge1_idx] = new_edge1;
        edge_vect[edge2_idx] = new_edge2;   
    }
    
    void EdgeSwap::replaceEdge_Set(swapper::Edge edge1, swapper::Edge edge2, swapper::Edge new_edge1, swapper::Edge new_edge2){

        edge_set.erase(edge1);
        edge_set.erase(edge2);

        edge_set.insert(new_edge1);
        edge_set.insert(new_edge2);

    }
    
    std::unordered_set<Edge> EdgeSwap::getEdges () {
        return edge_set;
    };

    // perform all edges swaps
    void EdgeSwap::edge_swaps (int N_swaps){
        int n_swap = 0;

        srand (time(NULL));
       
        //std::cout << N_swaps << endl; 
        int vect_size = edge_vect.size();
        while (n_swap < N_swaps) {
            //if (n_swap % 10000 == 0) {
            //    std::cout << n_swap << endl;
            //}
            //int edge1_idx = rand() % edges.size();
            int edge1_idx = rand() % vect_size;
            int edge2_idx = rand() % vect_size;
            //int edge2_idx = rand() % edges.size();

            float r = ((double) rand() / (RAND_MAX)) + 1;

            if (edge1_idx == edge2_idx) {
                //std::cout << "same edge" << endl;
                continue;
            }

            swapper::Edge edge1 = edge_vect[edge1_idx];
            swapper::Edge edge2 = edge_vect[edge2_idx];
            //auto it = std::begin(edge_set);
            //std::advance(it,edge1_idx);

            //auto edge1 =  *it;
            //it = std::begin(edge_set);
            //std::advance(it,edge2_idx);

            //auto edge2 = *it;

            //auto edge1 = *select_random(edge_set, edge1_idx);
            //auto edge2 = *select_random(edge_set, edge2_idx);

            swapper::Edge new_edge1(edge1.u, edge2.v);
            swapper::Edge new_edge2(edge1.v, edge2.u);
            //Edge new_edge1(Edge[edge1_idx].u, Edge[edge2_idx].v);
            //Edge new_edge2(Edge[edge1_idx].v, Edge[edge2_idx].u);


            if (edge_set.find(new_edge1) != edge_set.end()) continue;
            if (edge_set.find(new_edge2) != edge_set.end()) continue;
            //std::cout << "good edge" << endl;
            replaceEdge_Set(edge1, edge2, new_edge1, new_edge2);
            replaceEdge_Array(edge1_idx, edge2_idx, new_edge1, new_edge2);
            ++n_swap;
        }

    } 
}

