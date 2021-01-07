import ipdb
import time
import random
import argparse
import numpy as np

from GTgen.graph import *
#import networkit as nk
#import networkit.graphtools as gt

from collections import defaultdict
#from networkx.generators.random_graphs import gnm_random_graph
#from networkit.generators import EdgeSwitchingMarkovChainGenerator, HavelHakimiGenerator, ErdosRenyiGenerator

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
    def __init__(self, **kwargs):
        self.n = None
        self.m = None
        self.seq = None
        self.N_swap = None

    def is_realisable():
        raise NotImplementedError

    def _edge_swap(self): 
        """ Do N_swap edge swap to get more random graph"""
        import cProfile

        pr = cProfile.Profile()
        pr.enable()

        assert self.N_swap is not None, "attempting to swap edges but no swap number defined"
        # first shuffle edges, than swap two by two
        n_swap = 0
        self.logger.info('{} swaps needed'.format(self.N_swap))
        t0 = time.time()
        reverse_array = np.random.uniform(0,1,self.N_swap*10)
        #self.graph.shuffle_edges()
        rdm_index1 = np.random.choice(len(self.graph.edges), size=10*self.N_swap)
        rdm_index2 = np.random.choice(len(self.graph.edges), size=10*self.N_swap)

        while n_swap < self.N_swap:
            #for edge1_idx, edge1 in enumerate(self.graph.edges[::2]):
            for edge1_idx, edge2_idx, reverse in zip(rdm_index1, rdm_index2, reverse_array):
                #edge2_idx = edge1_idx + 1
                if edge1_idx == edge2_idx:
                    continue

                if n_swap % 10**4 == 0:

                    self.logger.debug('{} for {} swaps'.format(time.time() - t0, n_swap))

                (e1_n1, e1_n2)  = self.graph.edges[edge1_idx]

                # 1/2 chance of reversing _edge2 : equivalent to picking both
                # directions at random
                edge2 = self.graph.edges[edge2_idx]

                #if e1_n1 in edge2 or e1_n2 in edge2:
                #    accepted = False
                #else:
                #    accepted = True

                #if accepted:
                if reverse_array[n_swap] >= 0.5:
                    (e2_n2, e2_n1) = edge2 #self.edges[edge2_idx]
                else:
                    (e2_n1, e2_n2) = edge2 #self.edges[edge2_idx]

                # get new edges
                new_edge1 = (e1_n1, e2_n2) if e1_n1 < e2_n2 else (e2_n2, e1_n1)
                new_edge2 = (e2_n1, e1_n2) if e2_n1 < e1_n2 else (e1_n2, e2_n1)
                
                if new_edge1 in self.graph.edge_set or new_edge2 in self.graph.edge_set:
                    continue
                else:
                    self.graph.edge_set.remove(self.graph.edges[edge1_idx])
                    self.graph.edge_set.remove(self.graph.edges[edge2_idx])
            
                    self.graph.edge_set.add(new_edge1)
                    self.graph.edge_set.add(new_edge2)
            
                    self.graph.edges[edge1_idx] = new_edge1
                    self.graph.edges[edge2_idx] = new_edge2

                    n_swap += 1
                if n_swap >= self.N_swap :
                    break
            else:
                #when run out of indexes pick some again
                reverse_array = np.random.uniform(0,1,self.N_swap*10)
                #self.graph.shuffle_edges()
                rdm_index1 = np.random.choice(len(self.graph.edges), size=10*self.N_swap)
                rdm_index2 = np.random.choice(len(self.graph.edges), size=10*self.N_swap)

                #self.graph.shuffle_edges()

        pr.disable()
        pr.print_stats()

        pr.dump_stats('inside_complete.profile')

        self.logger.debug('{} for {} swap'.format(time.time()-t0, self.N_swap))


    def _edge_swap_shuffle(self): 
        """ Do N_swap edge swap to get more random graph"""
        import cProfile

        pr = cProfile.Profile()
        pr.enable()

        assert self.N_swap is not None, "attempting to swap edges but no swap number defined"
        # first shuffle edges, than swap two by two
        n_swap = 0
        self.logger.info('{} swaps needed'.format(self.N_swap))
        t0 = time.time()
        reverse_array = np.random.uniform(0,1,self.N_swap*100)
        self.graph.shuffle_edges()

        while n_swap < self.N_swap:
            #for edge1_idx, edge1 in enumerate(self.graph.edges[::2]):
            for edge1_idx, edge1 in enumerate(self.graph.edges[:]):
                edge2_idx = edge1_idx + 1

                if n_swap % 10**4 == 0:

                    self.logger.debug('{} for {} swaps'.format(time.time() - t0, n_swap))

                (e1_n1, e1_n2)  = self.graph.edges[edge1_idx]

                # 1/2 chance of reversing _edge2 : equivalent to picking both
                # directions at random
                edge2 = self.graph.edges[edge2_idx]

                #if e1_n1 in edge2 or e1_n2 in edge2:
                #    accepted = False
                #else:
                #    accepted = True

                #if accepted:
                if reverse_array[n_swap] >= 0.5:
                    (e2_n2, e2_n1) = edge2 #self.edges[edge2_idx]
                else:
                    (e2_n1, e2_n2) = edge2 #self.edges[edge2_idx]

                # get new edges
                new_edge1 = (e1_n1, e2_n2) if e1_n1 < e2_n2 else (e2_n2, e1_n1)
                new_edge2 = (e2_n1, e1_n2) if e2_n1 < e1_n2 else (e1_n2, e2_n1)
                
                if new_edge1 in self.graph.edge_set or new_edge2 in self.graph.edge_set:
                    continue
                else:
                    self.graph.edge_set.remove(self.graph.edges[edge1_idx])
                    self.graph.edge_set.remove(self.graph.edges[edge2_idx])
            
                    self.graph.edge_set.add(new_edge1)
                    self.graph.edge_set.add(new_edge2)
            
                    self.graph.edges[edge1_idx] = new_edge1
                    self.graph.edges[edge2_idx] = new_edge2

                    n_swap += 1
                if n_swap >= self.N_swap :
                    break
            else:
                self.graph.shuffle_edges()

        pr.disable()
        pr.print_stats()

        pr.dump_stats('inside_complete.profile')

        self.logger.debug('{} for {} swap'.format(time.time()-t0, self.N_swap))

    def write_graph(self, weights = None):
        if weights is not None:
            #iterator = zip(self.graph.iterEdges(), weights)
            iterator = zip(self.graph.edges(), weights)
        else:
            #iterator = self.graph.iterEdges() # TODO add zip (ones) pour ajouter des poids fictifs ? 
            iterator = self.graph.edges() # TODO add zip (ones) pour ajouter des poids fictifs ? 
        for ((u, v), weight) in iterator:
            if u<v :
                fout.write(f'{u},{v} {weight}\n')
            else:
                fout.write(f'{v},{u} {weight}\n')

    def run(self):
        raise NotImplementedError

class HavelHakimi(AbstractGraphGenerator):
    """
        Parameters:
        -----------
        sequence : vector
            Degree sequence to realize. Must be non-increasing.
        ignoreIfRealizable : bool, optional
            If true, generate the graph even if the degree sequence is not realizable. Some nodes may get lower degrees than requested in the sequence.
    """

    def __init__(self, sequence , N_swap, logger):
        self.logger = logger
        self.sequence = sequence
        #self.generator = HavelHakimiGenerator(self.sequence)
        self.N_swap = N_swap
        self.graph = Graph(edges=[],
                nodes=set(range(len(self.sequence))),
                is_sorted=False,
                degrees=self.sequence, logger=self.logger)

    def run(self):
        # Using Networkit implementation
        numDegVals = max(self.sequence) + 1 ## quid du +1 .. ? 
        nodesByDeficit = defaultdict(list)
        for node, degree in enumerate(self.sequence):
            nodesByDeficit[degree].insert(0, (degree, node))

        maxDeficit = numDegVals - 1
        while maxDeficit:
            # process node in largest bucket
            while len(nodesByDeficit[maxDeficit]) > 0:
                # get element
                #print(nodesByDeficit)
                #print(nodesByDeficit[maxDeficit])
                deficit, currentVertex = nodesByDeficit[maxDeficit].pop(0)

                # connect vertex to following ones
                currentNeighborList = maxDeficit
                numToMove = []


                while(deficit > 0):
                    numDeleteFromCurrentList = 0
                    for _, nextNeighbor in nodesByDeficit[currentNeighborList]:
                        #edge =  (currentVertex, nextNeighbor) if currentVertex < nextNeighbor else (nextNeighbor, currentVertex)
                        #print('current vertex {} nextNeighbor {} , deficit {} max deficit {}'.format(currentVertex, nextNeighbor, deficit, maxDeficit))
                        #print('before add')
                        #print(self.graph.edges)
                        self.graph.addEdge((currentVertex, nextNeighbor))
                        #print('after add')
                        #print(self.graph.edges)

                        deficit -= 1
                        numDeleteFromCurrentList += 1

                        if deficit == 0:
                            # dur to -= 1  a few lines below
                            currentNeighborList += 1 
                            break
                    numToMove.append(numDeleteFromCurrentList)

                    if (currentNeighborList == 1):
                        #print(self.graph.edges)
                        raise RuntimeError('Havel Hakimi: degree sequence is not realisable')
                    currentNeighborList -= 1

                while len(numToMove) > 0:
                    num = numToMove.pop()

                    # move this many items from current list to next one
                    for i in range(num):
                        dan = nodesByDeficit[currentNeighborList][0]#.pop(0)
                        #dan[0] -= 1
                        nodesByDeficit[currentNeighborList - 1].insert(0, (dan[0] -1, dan[1]))
                        nodesByDeficit[currentNeighborList].pop(0)
                    currentNeighborList += 1
            maxDeficit -= 1 

        t00 = time.time()

        #self.graph.edge_set = set(self.graph.edges)
        counter_edge = Counter(self.graph.edges)
        multiple = [edge for edge in self.graph.edges if counter_edge[edge] >1]
        #ipdb.set_trace()
        #assert sorted(list(self.graph.edge_set)) == sorted(self.graph_edges)
        #print('{} to convert to set'.format(time.time() - t00))
        #ipdb.set_trace()
        #self.graph = self.generator.generate()

class ModifiedHavelHakimi(AbstractGraphGenerator):
    """
        Parameters:
        -----------
        sequence : vector
            Degree sequence to realize. Must be non-increasing.
        ignoreIfRealizable : bool, optional
            If true, generate the graph even if the degree sequence is not realizable. Some nodes may get lower degrees than requested in the sequence.
    """

    def __init__(self, sequence , N_swap, logger):
        self.logger = logger
        self.sequence = sequence
        #self.generator = HavelHakimiGenerator(self.sequence)
        self.N_swap = N_swap
        self.logger.info('before instanciantiating graph')
        self.logger.info('{} nodes'.format(len(self.sequence)))
        t0 = time.time()
        self.graph = Graph(nodes=set(range(len(self.sequence))),
                           degrees=self.sequence)
        print('{} to instanciate graph'.format(time.time() - t0) )
    def run(self):
        # Using Networkit implementation
        numDegVals = max(self.sequence) + 1 ## quid du +1 .. ? 
        nodesByDeficit = defaultdict(list)
        for node, degree in enumerate(self.sequence):
            nodesByDeficit[degree].insert(0, (degree, node))

        maxDeficit = numDegVals - 1
        while maxDeficit:
            while len(nodesByDeficit[maxDeficit]) > 0:
                # get element
                deficit, currentVertex = nodesByDeficit[maxDeficit].pop(0)

                # connect vertex to following ones
                currentNeighborList = maxDeficit
                numToMove = []
                while(deficit):
                    numDeleteFromCurrentList = 0
                    for _, nextNeighbor in nodesByDeficit[currentNeighborList]:
                        self.graph.addEdge((currentVertex, nextNeighbor))
                        deficit -= 1
                        numDeleteFromCurrentList += 1

                        if deficit == 0:
                            # dur to -= 1  a few lines below
                            currentNeighborList += 1 
                            break
                    numToMove.append(numDeleteFromCurrentList)

                    if (currentNeighborList == 1):
                        #print(self.graph.edges)
                        raise RuntimeError('Havel Hakimi: degree sequence is not realisable')
                    currentNeighborList -= 1

                while len(numToMove) > 0:
                    num = numToMove.pop()

                    # move this many items from current list to next one
                    for i in range(num):
                        dan = nodesByDeficit[currentNeighborList].pop(0)
                        #dan[0] -= 1
                        nodesByDeficit[currentNeighborList - 1].insert(0, (dan[0] -1, dan[1]))
                    currentNeighborList += 1
            maxDeficit -= 1 

        #self.graph = self.generator.generate()

class GNM(AbstractGraphGenerator):
    """ Reimplementation of NetworkX GNM model Using Networkit API
        (useful to get randomEdge method, amongst other things)
        
    Parameters:
    -----------
        n: int
            number of nodes
        m: int
            number of edges
        seed: int
            random seed
    """
    def __init__(self, n, m, logger,seed=None):
        self.n = n
        self.m = m 
        self.logger = logger
        #self.seed = seed if not None else None ## TODO

    @staticmethod ## 
    def _clique(G):
        """ From Networkx.generators.classic.complete_graph
        """
        #n_name, nodes = n
        #G = empty_graph(n_name, create_using)
        import itertools
        nodes = range(G.numberOfNodes)
        if len(nodes) > 1:
            edges = itertools.combinations(nodes, 2)
            G.addEdgesFrom(edges)
        return G

    def run(self):
        """ can add nodes to existing GNM, and create GNM of these nodes """ 
        #self.graph = nk.graph.Graph()
        self.graph = Graph(edges=[], nodes=set(range(self.n)),
                is_sorted=False, degrees=None, logger=self.logger)
        
        #self.graph.addNodes(self.n)

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
        nlist = range(self.n)
        edge_count = 0
        while edge_count < self.m:
            u = np.random.choice(nlist)
            v = np.random.choice(nlist)
            edge = (u, v) if u<v else (v,u)
            if u == v or self.graph.hasEdge(edge):
                continue
            else:
                self.graph.addEdge(edge)
                edge_count = edge_count + 1
        return
#class compareNetworkit(AbstractGraphGenerator):
#    ## TODO ADD AS UNIT TEST <3 
#    def __init__(self, degree_list,
#            logger):
#
#        self.logger = logger
#        self.degree_list = degree_list
#
#    def run(self):
#        #ipdb.set_trace()
#        self.generator = HavelHakimiGenerator(self.degree_list, ignoreIfRealizable=False)
#        self.graph = self.generator.generate()
#        print(sorted(self.graph.edges()))
#        model = HavelHakimi(self.degree_list, 0, self.logger)
#        model.run()
#        print(sorted(model.graph.edges) )
