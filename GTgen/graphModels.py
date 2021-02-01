"""
    Implementation of Havel Hakimi and GNM graph models.
    To implement other graph Model, simply inherit the class 
    AbstractGraphGenerator, and implement the __init__ and run methods

    Usage example:


    >>> from GTgen.graphModels import HavelHakimi

    >>> # list of (nodes, degree)
    >>> sequence = [(0,4), (1,2), (2,2), (3,4), (5,2), (6,2)]
    >>> N_swap = 0
    >>> logger = lo
    >>> logging.basicConfig(
    >>>         level=logging.INFO,
    >>>         format='%(asctime)s %(levelname)-8s %(message)s',
    >>>         datefmt='%m-%d %H:%M'
    >>>         )

    >>> # instantiate logger
    >>> logger = logging.getLogger()

    >>> model = HavelHakimi(sequence  , N_swap, logger, seed=None)

    >>> # run Havel Hakimi
    >>> model.run()

    >>> print(model.graph)
"""


import ipdb
import time
import random
import logging
import argparse
import numpy as np

from GTgen.graph import *
from collections import defaultdict

class AbstractGraphGenerator():
    """ Abstract Class for graph Generators
        Generate a random graph, picked uniformely, given a number of nodes,
        a number of edges, and a degree sequence.

        Attributes:
        -----------
        n: int
            the number of nodes 
        m: int
            the number of edges
        seq: list of int
            the degree sequence

    """
    def __init__(self, N_swap, graph):
        self.graph = graph
        self.N_swap = N_swap

    #def is_realisable():
    #    raise NotImplementedError

    def _edge_swap(self): 
        """ Do N_swap edge swap to get more random graph"""

        assert self.N_swap is not None, ("attempting to swap edges but no "
                "swap number defined")

        # first shuffle edges, than swap two by two
        n_swap = 0
        self.logger.info('{} swaps needed'.format(self.N_swap))

        # keep track of time
        t0 = time.time()
        #reverse_array = np.random.uniform(0,1,self.N_swap*10)

        # pick random edges
        #rdm_index1 = np.random.choice(len(self.graph.edges), size=10*self.N_swap)
        #rdm_index2 = np.random.choice(len(self.graph.edges), size=10*self.N_swap)

        # Keep track of number of attemps before a successful swap
        n_attempts = []
        _n_attempts = 0
        #N_fail = 0
        while n_swap < self.N_swap:
            #for edge1_idx, edge2_idx, reverse in zip(rdm_index1, rdm_index2, reverse_array):
            edge1_idx = np.random.choice(len(self.graph.edges))
            edge2_idx = np.random.choice(len(self.graph.edges))
            reverse = np.random.uniform(0,1)

            if edge1_idx == edge2_idx:
                continue

            (e1_n1, e1_n2)  = self.graph.edges[edge1_idx]
            edge2 = self.graph.edges[edge2_idx]

            if (e1_n1 in edge2 or e1_n2 in edge2):
                continue

            # 1/2 chance of reversing _edge2 : equivalent to picking both
            # directions at random
            #if reverse_array[n_swap] >= 0.5:
            if reverse >= 0.5:
                (e2_n2, e2_n1) = edge2 #self.edges[edge2_idx]
            else:
                (e2_n1, e2_n2) = edge2 #self.edges[edge2_idx]

            # get new edges
            new_edge1 = (e1_n1, e2_n2) if e1_n1 < e2_n2 else (e2_n2, e1_n1)
            new_edge2 = (e2_n1, e1_n2) if e2_n1 < e1_n2 else (e1_n2, e2_n1)

            # skip when edge exist 
            if (new_edge1 in self.graph.edge_set 
             or new_edge2 in self.graph.edge_set):
                _n_attempts += 1
                continue
            else:
                n_attempts.append(_n_attempts)
                _n_attempts = 0

                # replace previous edges in set
                self.graph.edge_set.remove(self.graph.edges[edge1_idx])
                self.graph.edge_set.remove(self.graph.edges[edge2_idx])
                self.graph.edge_set.add(new_edge1)
                self.graph.edge_set.add(new_edge2)
            
                self.graph.edges[edge1_idx] = new_edge1
                self.graph.edges[edge2_idx] = new_edge2
                n_swap += 1
                if n_swap % 10**7 == 0:
                    self.logger.debug('{} for {} swaps, mean n_attempts {}'.format(
                        time.time() - t0, n_swap, np.mean(n_attempts)))

            if n_swap >= self.N_swap :
                break
            #else:
            #    #when run out of indexes pick some again
            #    reverse_array = np.random.uniform(0,1,self.N_swap*10)
            #    #self.graph.shuffle_edges()
            #    rdm_index1 = np.random.choice(len(self.graph.edges), size=10*self.N_swap)
            #    rdm_index2 = np.random.choice(len(self.graph.edges), size=10*self.N_swap)

                #self.graph.shuffle_edges()

        self.logger.debug('{} for {} swap, mean n_attempts {},'
            'min n_attempts {}, max n_attempts {}'.format(
                time.time()-t0, self.N_swap, np.mean(n_attempts),
                min(n_attempts), max(n_attempts)))
        #self.logger.debug('FAIL {}'.format(N_fail))

    #def write_graph(self, weights = None):
    #    if weights is not None:
    #        #iterator = zip(self.graph.iterEdges(), weights)
    #        iterator = zip(self.graph.edges(), weights)
    #    else:
    #        #iterator = self.graph.iterEdges() # TODO add zip (ones) pour ajouter des poids fictifs ? 
    #        iterator = self.graph.edges() # TODO add zip (ones) pour ajouter des poids fictifs ? 
    #    for ((u, v), weight) in iterator:
    #        if u<v :
    #            fout.write(f'{u},{v} {weight}\n')
    #        else:
    #            fout.write(f'{v},{u} {weight}\n')

    def run(self):
        raise NotImplementedError

class HavelHakimi(AbstractGraphGenerator):
    """
        Python implementation of Networkit Havel Hakimi with in-house 
        graph data structure.
        
        Attributes:
        -----------
        sequence : np.array
            Degree sequence and node names to realize. Must be non-increasing.
        N_swap: int
            After generation, N_swap * N_edges edge swap will be performed.
        seed: int
            The random seed to be used in numpy
    """

    def __init__(self, sequence , N_swap, logger, seed=None):
        self.graph = Graph(edges=[],
                #nodes=set(range(len(self.sequence))),
                nodes=set(sequence[:,0]),
                #is_sorted=False,
                degrees=sequence, logger=logger)

        super().__init__(N_swap=N_swap, graph=self.graph)
        self.logger = logger
        if seed is not None:
            np.random.seed(seed)
        if len(sequence.shape) > 1 and sequence.shape[1] > 1:
            self.sequence = sequence
        else:
            # when nodes not provided, simply name them in order
            self.sequence = np.empty((sequence.shape[0], 2))
            self.sequence[:,1] = sequence
            self.sequence[:,0] = np.arange(sequence.shape[0])

        self.sequence = sequence
        self.N_swap = N_swap

        self.graph = Graph(edges=[],
                #nodes=set(range(len(self.sequence))),
                nodes=set(self.sequence[:,0]),
                degrees=self.sequence, logger=self.logger)

    def run(self):
        # Using Networkit implementation
        numDegVals = np.max(self.sequence[:,1]) + 1
        nodesByDeficit = defaultdict(list)

        #for node, degree in enumerate(self.sequence):
        for node, degree in self.sequence:
            nodesByDeficit[degree].insert(0, (degree, node))

        maxDeficit = numDegVals - 1
        while maxDeficit:
            # process node in largest bucket
            while len(nodesByDeficit[maxDeficit]) > 0:
                # get element
                deficit, currentVertex = nodesByDeficit[maxDeficit].pop(0)

                # connect vertex to following ones
                currentNeighborList = maxDeficit
                numToMove = []


                while(deficit > 0):
                    numDeleteFromCurrentList = 0
                    for _, nextNeighbor in nodesByDeficit[currentNeighborList]:
                        self.graph.addEdge((currentVertex, nextNeighbor))

                        deficit -= 1
                        numDeleteFromCurrentList += 1

                        if deficit == 0:
                            # due to -= 1  a few lines below
                            currentNeighborList += 1 
                            break
                    numToMove.append(numDeleteFromCurrentList)

                    if (currentNeighborList == 1):
                        raise RuntimeError('Havel Hakimi: degree sequence is not realisable')
                    currentNeighborList -= 1

                while len(numToMove) > 0:
                    num = numToMove.pop()

                    # move this many items from current list to next one
                    for i in range(num):
                        dan = nodesByDeficit[currentNeighborList][0]#.pop(0)
                        nodesByDeficit[currentNeighborList - 1].insert(0, (dan[0] -1, dan[1]))
                        nodesByDeficit[currentNeighborList].pop(0)
                    currentNeighborList += 1
            maxDeficit -= 1 

        counter_edge = Counter(self.graph.edges)
        multiple = [edge for edge in self.graph.edges if counter_edge[edge] >1]
        # initialize weights
        self.graph.init_weights()


class GNM(AbstractGraphGenerator):
    """ Reimplementation of NetworkX GNM model.
    
    The complete process is: 

    .. code-block:: python

        - if n == 0:
            - return empty graph
        - if m == n*(n-1)/2:
            # return clique as combinations of all nodes as edges
            - edges = itertools.combinations(nodes, 2)
        - while edge_count <m:
            - pick two nodes (u,v) such that u != v
            - if (u,v) not in graph:
                - edge_count += 1
                - add (u,v) to graph
        - return graph
    Attributes:
    -----------
        n: int
            number of nodes
        m: int
            number of edges
        seed: int
            random seed
    """
    def __init__(self, n, m, seed=None, nodes=None, logger=None):
        self.n = n
        self.m = m 

        if seed is not None:
            np.random.seed(seed)

        if nodes is None:
            self.nodes = set(range(n))
        else:
            self.nodes = nodes

        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s %(levelname)-8s %(message)s',
                datefmt='%m-%d %H:%M'
                )

            self.logger = logging.getLogger()
        else:
            self.logger = logger

    @staticmethod ## 
    def _clique(G):
        """ From Networkx.generators.classic.complete_graph
        """
        import itertools

        # generate clique by getting all possible combinations of 2 nodes
        nodes = G.nodes
        if len(nodes) > 1:
            edges = itertools.combinations(nodes, 2)
            G.addEdgesFrom(edges)

        # initialize weights
        G.init_weights()

        return G

    def run(self):
        """ can add nodes to existing GNM, and create GNM of these nodes """ 
        self.graph = Graph(edges=[], nodes=self.nodes,
                 degrees=None, logger=self.logger)

        # if n = 1, return 1 node graph,
        # if m = n(n-1)/2, return clique
        if self.n < 1:
            raise RuntimeError('trying to generate empty graph...')
        elif self.n == 1:
            return
        max_edges = self.n * (self.n - 1) / 2.0

        #if self.m <= 0:
        #    return
        if self.m >= max_edges:
            GNM._clique(self.graph)
            return

        # pick two nodes at random, create edge if it doesn't already exist
        #nlist = range(self.n)
        edge_count = 0
        while edge_count < self.m:
            u = np.random.choice(list(self.nodes))
            v = np.random.choice(list(self.nodes))
            edge = (u, v) if u<v else (v,u)
            if u == v or self.graph.hasEdge(edge):
                continue
            else:
                self.graph.addEdge(edge)
                edge_count = edge_count + 1

        # initialize weights
        self.graph.init_weights()
        return

