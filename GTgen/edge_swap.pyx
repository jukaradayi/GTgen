# distutils: language = c++
from cython.operator import dereference

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set

from GTgen.edge_swap cimport Edge, EdgeSwap


def main():
    # This is only to make cython generate initialization code
    pass
main()

cdef class pyEdge():
    cdef Edge* _this
    def __cinit__(self, int u, int v):
        self._this = new Edge(u, v)

        #self.u = u
        #self.v = v
        #return init_edge()

    def __dealloc__(self):
        del self._this
    #def init_edge():
    #    return _Edge(self.u, self.v)

    def get_edge(self):
        return (self._this.u, self._this.v)

cdef class pyEdgeSwap():
    cdef EdgeSwap* _this
    def __cinit__(self, edges, edge_set):

        cdef vector[Edge] _edge_vect
        #cdef int i
        cdef unordered_set[Edge] _edge_set
        for e in edges:
            _edge = pyEdge(e[0], e[1])
            _edge_vect.push_back(dereference(_edge._this))
            _edge_set.insert(dereference(_edge._this))
        #cdef vector[Edge]
        #self.edges = edges
        #self.edge_set = edge_set
        #print("juste avant instance")
        self._this = new EdgeSwap(_edge_vect, _edge_set)
        #self._this = new EdgeSwapper(_edge_set)


    def pyEdge_swaps(self, N_swaps):
        new_edges = self._this.edge_swaps(N_swaps)
        cdef unordered_set[Edge] _edge_set
        _edge_set = self._this.getEdges()
        edge_set = set()
        edge_vect = []
        cdef Edge it = dereference(pyEdge(0, 0)._this)
        for it in _edge_set:
            edge_set.add((it.u, it.v))
            edge_vect.append((it.u, it.v))
        return edge_vect, edge_set
               


#class Main():
#    def __init__():
#        a=None
#        pass
#
#a=None
