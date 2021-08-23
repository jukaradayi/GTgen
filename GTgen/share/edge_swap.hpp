#ifndef EDGE_SWAPPER_HPP_
#define EDGE_SWAPPER_HPP_


#include <cstdint>
#include <array>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;


//using namespace boost;
//
//namespace boost {
//    size_t hash_value(std::tuple<int, int> const & t) {
//        return get<0>(t) * 10 + get<1>(t) ;
//    }
//}
//struct key_hash : public std::unary_function<std::std::tuple<int,int> , std::size_t>
//{
//     std::size_t operator()(const std::std::tuple<int,int>& k) const
//          {
//                 return std::get<0>(k) ^ std::get<1>(k) ^ std::get<2>(k);
//                  }
//};
//namespace std {
//    size_t hash_value(std::std::tuple<int, int> const& t) {
//        return get<0>(t) * 10 + get<1>(t) ;
//        }
//}
namespace swapper {
struct Edge {
    uint64_t u, v;
    Edge() : u(std::numeric_limits<uint64_t>::max()), v(std::numeric_limits<uint64_t>::max()) {}

    Edge(int _u, int _v) {
        u = std::min(_u, _v);
        v = std::max(_u, _v);
    }

    //size_t hash_value(std::std::tuple<int, int> const& t) {
    //return get<0>(t) * 10 + get<1>(t) ;
    //}

};
}

namespace std {
    template<> struct hash<swapper::Edge> {
        inline size_t operator()(const swapper::Edge& e) const {
            return e.u * 10 + e.v ;
        }
    };

    template <> struct equal_to<swapper::Edge> {
        inline bool operator()(const swapper::Edge& a, const swapper::Edge& b) const {
            bool comparison;
            if (a.u == b.u && a.v == b.v) {
                comparison = true;
            } else {
                comparison = false;
            }
            return comparison;
    };
    };
}


namespace swapper {


    class EdgeSwap {
        public: 
            std::vector<Edge> edge_vect;
            std::unordered_set<Edge> edge_set;
            EdgeSwap();
            EdgeSwap(std::vector<Edge> edge_vect, std::unordered_set<Edge> edge_set);
            //EdgeSwap( std::unordered_set<Edge> edge_set);

            ~EdgeSwap();

             
            // check if two edges share a node
            bool shareNode(Edge edge1, Edge edge2);
            
            // replace edges
            void replaceEdge_Array(int edge1_idx, int edge2_idx, Edge new_edge1, Edge new_edge2);
            void replaceEdge_Set(swapper::Edge edge1, swapper::Edge edge2, Edge new_edge1, Edge new_edge2);
            
            
            // perform all edges swaps
            void edge_swaps (int N_swaps);

            // get edges
            std::unordered_set<Edge> getEdges ();
    };
}


#endif
