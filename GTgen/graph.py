""" Simple data structure for Graph.
    A graph object stores:
        - its nodes as a list of int (the node names are 
          restricted to integers)
        - its edges as: 
            - a list of (u,v) where u<v (to iterate, when order is
              required)
            - a set of (u,v) where u<v (for quick search, for example
              for edge-swap process)
        - its degree sequence as a dict {node:degree}
"""

import ipdb
import time
import random
import numpy as np

from collections import Counter

class Graph():
    """ Graph Class

    Attributes:
    -----------
    edges: list of tuple of int
        all the edges (u,v) in the graph, with u<v, stored as a list
    edge_set: set of tuple of int
        all the edges (u,v) in the graph, with u<v, stored as a set
    nodes: list of int,
        the node names
    degree: list of int
        the degree sequence of the graph
    logger: logger,
        a logger
    """
    def __init__(self, edges=[], nodes=set(),
            edge_set = None,
            degrees=None,
            weight=np.empty((0,)),
            logger=None,
            merge_multiedges=False):
        self.edges = edges
        if edge_set is None:
            self.edge_set = set(edges)
        else:
            self.edge_set = edge_set
        self.nodes = nodes
        self.weight = weight
        self._degrees = degrees
        self.logger = logger

        # When adding graphs, if merge_multiedges set to True, 
        # increment edge weight when both graph contain the same edge
        self.merge_multiedges = merge_multiedges

    def __repr__(self):
        repr_string = f'nodes:\n {self.nodes}\n'
        repr_string += f'edges:\n {self.edges}\n'
        if len(self.weight) > 0:
            repr_string += f'weights:\n {self.weight}\n'
        return repr_string

    def __add__(self, other):
        """ Merge two graphs.
            When node names are shared, assume same nodes
            Raise Attribute Error when edges are present in both.
        """
        # raise error if multiple edge detected
        #if (len(self.edge_set.intersection(other.edge_set)) > 0 
        #    and not self.merge_multiedges):
        #    raise AttributeError
        #else:
        total_nodes = self.nodes.union(other.nodes)
        total_edge_set = self.edge_set.union(other.edge_set)
        #total_degrees = Counter(elem for elem in list(sum(toedges, ())))
        
        # get list of multi edges
        multiedges = list(self.edge_set.intersection(other.edge_set))

        if self.merge_multiedges and len(multiedges) > 0:
            # when merging multi edges, remove them from other edge list
            # and increment weight
            multiedges = list(self.edge_set.intersection(other.edge_set))
            #_other_edges = other.edges.copy()
            other_edges = []
            other_mask = []

            for edge in multiedges:
                edge_idx = self.edges.index(edge)
                self.weight[edge_idx] += 1

                # remove from other graph edges list
                other_idx = other.edges.index(edge)
                #del(other_edges[other_idx])
                other_mask.append(other_idx)
            else:
                other_edges = [edge for idx, edge in enumerate(other.edges) if idx not in other_mask]

                # merge updated list and weights
                total_edges = self.edges + other_edges
                other_weight = other.weight[[i for i in range(len(other.weight)) if i not in other_mask]]
                total_weight = np.concatenate([self.weight, other_weight])
        elif len(multiedges) > 0 and not self.merge_multiedges:
            # raise error if can't merge multiedges
            raise AttributeError
        else:
            # no multi edges detected
            total_edges = self.edges + other.edges
            total_weight = np.concatenate([self.weight, other.weight])

        return Graph(edges = total_edges, nodes=total_nodes,
                edge_set=total_edge_set, weight=total_weight,
                logger = self.logger,
                merge_multiedges=self.merge_multiedges)

    def init_weights(self):
        """ Initiate weights as array of 0s.
            Needs to be done when graph is computed
        """
        self.weight = np.ones((len(self.edges)))

    def shuffle_weights(self):
        assert self.weight is not None, ("Attempting to shuffle weights,"
                                         "but weights not defined.")
        np.random.shuffle(self.weight)


    def _write_graphOnly(self, output):
        with open(output, 'w') as fout:
            for edge in self.edges:
                fout.write('{}-{}'.format(edge[0], edge[1]))

    def _write_weightedGraph(self, output):
        assert len(self.edges) == len(self.weight), ("Graph has {} weights"
              "and {} edges, should have the same number".format(
                  len(self.weight), len(self.edges)))
        with open(output, 'w') as fout:
            for edge, weight in sorted(zip(self.edges, self.weight)):
                fout.write('{}-{} {}\n'.format(edge[0], edge[1], weight))

    def write_graph(self, output):
        if self.weight is None:
            self._write_graphOnly(output)
        else:
            self._write_weightedGraph(output)

    @property
    def degrees(self):
        """ list of node degrees, unordered"""
        if self._degrees is None:
            # list(sum(edges, ())) flattens [(n1, n2), (n1, n3)] in [n1,n2,n1,n3]
            # so just count number of occurence of each node in flattened list
            self._degrees = Counter(elem for elem in list(sum(self.edges, ())))
        return self._degrees

    def hasEdge(self, _edge): ## TODO should check order of u,v
        return _edge in self.edge_set

    # TODO
    #def renameNodes(self, mapping):
    #    """ rename all nodes"""

    #def hasEdge(self, _edge):
    #    """ Check if the graph has requested edge (u,v). 
    #        Reorder edge if v>u.
    #    """
    #    (u, v) = _edge
    #    edge = (u, v) if u < v else (v, u)
    #    if edge in self.edge_set:
    #        return True
    #    else:
    #        return False
    #    #if ((v>u and (v,u) in self.edge_set)
    #    #  or (u<v and (u,v) in self.edge_set)):
    #    #    return True
    #    #else:
    #    #    return False

    def _replaceEdges(self, edge1_idx, edge2_idx, new_edge1, new_edge2):
        self.edge_set.remove(self.edges[edge1_idx])
        self.edge_set.remove(self.edges[edge2_idx])

        self.edge_set.add(new_edge1)
        self.edge_set.add(new_edge2)

        self.edges[edge1_idx] = new_edge1
        self.edges[edge2_idx] = new_edge2


    def swapEdge(self, edge1_idx, edge2_idx, reverse=False):
        """ Given two edges e1=(e1_n1, e1_n2) and e2=(e2_n1, e2_n2), 
           swap the edges to get e1'=(e1_n1, e2_n2), e2'=(e2_n1, e1_n2):

           >>>  e1_n1 --- e1_n2       e1_n1 \\ / e1_n2
           >>>                   =>          x
           >>>  e2_n1 --- e2_n2       e2_n1 / \\ e2_n2

             Parameters:
             ----------
             edge1_idx, edge2_idx: ints
                index of edges to be swapped
        """
        #assert type(edge1_idx) == int, 'expecting integer for swapEdgeIdx'
        #assert type(edge2_idx) == int, 'expecting integer for swapEdgeIdx'
        #assert edge1_idx < len(self.edges), 
        #    "edge index for swap is out of bound"
        #assert edge2_idx < len(self.edges), 
        #    "edge index for swap is out of bound"
        #assert edge1_idx != edge2_idx, "swapping edge with itself"

        # get edges
        (e1_n1, e1_n2)  = self.edges[edge1_idx]

        # 1/2 chance of reversing _edge2 : equivalent to picking both
        # directions at random
        edge2 = self.edges[edge2_idx]

        if e1_n1 in edge2 or e1_n2 in edge2:
            return False

        # reversing edge 2 or not
        if reverse:
            (e2_n2, e2_n1) = edge2 #self.edges[edge2_idx]
        else:
            (e2_n1, e2_n2) = edge2 #self.edges[edge2_idx]

        # get new edges
        new_edge1 = (e1_n1, e2_n2) if e1_n1 < e2_n2 else (e2_n2, e1_n1)
        new_edge2 = (e2_n1, e1_n2) if e2_n1 < e1_n2 else (e1_n2, e2_n1)
        
        #assert self.hasEdge((e1_n1, e1_n2)), 
        #   "attempting to swap an edge that is not in graph"
        #assert self.hasEdge((e2_n1, e2_n2)),
        #   "attempting to swap an edge that is not in graph"
        #assert not self.hasEdge(new_edge1), 
        #    "swapped edge already exist in graph"
        #assert not self.hasEdge(new_edge2),
        #   "swapped edge already exist in graph"

        # check if new edges already exist
        if new_edge1 in self.edge_set or new_edge2 in self.edge_set:
            return False

        # replace edges
        self._replaceEdges(edge1_idx, edge2_idx, new_edge1, new_edge2)

        # if swap was successful return true
        return True

    @property
    def numberOfNodes(self):
        if self.nodes is None:
            self.nodes = set(sum(edges, ()))
        return len(self.nodes)

    @property
    def numberOfEdges(self):
        return len(self.edges)

    def addEdge(self, edge):
        """ add an edge in the graph"""
        (n1, n2) = edge
        #assert n1 != n2, "can't add self loop"
        #assert not self.hasEdge(edge), "can't add multi edge"

        # keep nodes sorted
        if n1 < n2:
            self.edges.append((n1, n2))
            self.edge_set.add((n1, n2))
        else:
            self.edges.append((n2, n1))
            self.edge_set.add((n2, n1))

        # append nodes
        self.nodes.add(n1)
        self.nodes.add(n2)

    def addEdgesFrom(self, edge_list):
        """ add a list of edges"""
        for edge in edge_list:
            self.addEdge(edge)

    def shuffle_edges(self):
        """ shuffle all edges"""
        np.random.shuffle(self.edges)
