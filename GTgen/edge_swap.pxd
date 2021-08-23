from cython.operator import dereference

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set

cdef extern from "share/edge_swap.cpp":
    pass

cdef extern from "share/edge_swap.hpp" namespace "swapper":
    cdef cppclass Edge "swapper::Edge":
        int u
        int v
        Edge() except +
        Edge(int u, int v) except +

    cdef cppclass EdgeSwap "swapper::EdgeSwap":
        EdgeSwap() except +
        EdgeSwap(vector[Edge], unordered_set[Edge]) except +
        #EdgeSwap( unordered_set[Edge]) except +
        #vector[Edge] edges
        unordered_set[Edge] edge_set
        bool shareNode(Edge, Edge) except +
        #void replaceEdge_Array(int, int, Edge, Edge) except +
        void replaceEdge_Set(Edge, Edge, Edge, Edge) except +
        void edge_swaps(int) except +
        unordered_set[Edge] getEdges() except +


