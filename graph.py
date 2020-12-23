import ipdb
import time
import random
import argparse
import numpy as np
import networkit as nk
#import networkit.graphtools as gt

#from networkx.generators.random_graphs import gnm_random_graph
from networkit.generators import EdgeSwitchingMarkovChainGenerator, HavelHakimiGenerator, ErdosRenyiGenerator

class AbstractGraphGenerator():
    """ Abstract Class for graph Generators
        Generate a random graph, picked uniformely, given a number of nodes,
        a number of edges, and a degree sequence.
        Attributes:
        -----------
        n : int
            the number of nodes 
        m : int
            the number of edges
        seq : list of int
            the degree sequence

    """
    def __init__(self, **kwargs):
        self.n = None
        self.m = None
        self.seq = None
        self.N_swap = None

    def is_realisable():
        raise NotImplementedError

    @property
    def degree_list(self):
        """ Return list of node degrees, unordered"""
        degree_list = []
        for u in self.graph.iterNodes():
            degree_list.append((u,self.graph.degree(u)))
        return degree_list

    # trying to optimize
    ## IDEE DE ML: graph NK (probablement - a verif) stocké comme 
    #      
    #         n1 -> (ni, ... nj)
    #         n2 -> (ni',... nj') (voisins)
    #           .
    #           .
    #           .
    #
    # pas super optimisé pour faire les swaps, donc changer pour avoir plutôt :
    #   [[n1, ni] ... [n1, nj], [n2, ni'] ... [n2, nj'] ...]
    # 
    # puis shuffle, faire des swap 2 à 2 avec voisin puis reshuffle au besoin et refaire swap, 
    # puis repasser a première visu

    def _edge_swap(self):
        """ Swap N_swap edges , picked uniformly over the set all all edges"""
        ## TODO could also defin default number ... ? 
        assert self.N_swap is not None, "attempting to swap edges but no swap number defined"
        self.logger.info('swapping {} edges in graph'.format(self.N_swap))

        #  track number of swaps done for debug print
        n = 0
        t1 = time.time()
        # trying to beat DEBUG:root:took 159.76628422737122s to do 10000 swaps
        edge_pool = nk.graphtools.randomEdges(self.graph, self.N_swap)
        t6 = time.time() 
        self.logger.info('took {} to pick {} random edges'.format(t6 -t1, self.N_swap))
        pool_index = 0
        while (n < self.N_swap):
            
            # print every 10^7 swaps
            if n%10000000 == 0 and n>0:
                self.logger.debug('{} switches dones'.format(n))
            (e1_n1, e1_n2), (e2_n1, e2_n2) = edge_pool[pool_index], edge_pool[pool_index + 1]
            pool_index += 2

            # get new pool if first one is finished
            if pool_index >= len(edge_pool):
                edge_pool = nk.graphtools.randomEdges(self.graph, self.N_swap)
                pool_index = 0

            #((e1_n1, e1_n2), (e2_n1, e2_n2)) = gt.randomEdges(self.graph, 2)


            # check if didn't pick two times the same edge 
            # or if switched edge already exists
            if ((e1_n1 == e2_n1) or (e1_n1 == e2_n2) 
                or (e1_n2 == e2_n1) or (e1_n2 == e2_n2)
                or self.graph.hasEdge(e1_n1, e2_n2) 
                or self.graph.hasEdge(e2_n1, e1_n2)
                or not self.graph.hasEdge(e1_n1, e1_n2)
                or not self.graph.hasEdge(e2_n1, e2_n2)):
                continue
            else:
                self.graph.swapEdge(e1_n1, e1_n2, e2_n1, e2_n2)
                n += 1
        t2 = time.time()
        self.logger.debug(
                'took {}s to do {} switches'.format(t2-t1, self.N_swap))

    #def _edge_swap(self):
    #    """ Swap N_swap edges , picked uniformly over the set all all edges"""
    #    ## TODO could also defin default number ... ? 
    #    assert self.N_swap is not None, "attempting to swap edges but no swap number defined"
    #    self.logger.info('swapping {} edges in graph'.format(self.N_swap))

    #    #  track number of swaps done for debug print
    #    n = 0
    #    t1 = time.time()
    #    while (n < self.N_swap):
    #        
    #        # print every 10^7 swaps
    #        #if n%10000000 == 0 and n>0:
    #        if n%1000 == 0:
    #            self.logger.debug('{} switches dones'.format(n))
    #        ((e1_n1, e1_n2), (e2_n1, e2_n2)) = nk.graphtools.randomEdges(self.graph, 2)


    #        # check if didn't pick two times the same edge 
    #        # or if switched edge already exists
    #        if ((e1_n1 == e2_n1) or (e1_n1 == e2_n2) 
    #            or (e1_n2 == e2_n1) or (e1_n2 == e2_n2)
    #            or self.graph.hasEdge(e1_n1, e2_n2) 
    #            or self.graph.hasEdge(e2_n1, e1_n2)):
    #            continue
    #        else:
    #            self.graph.swapEdge(e1_n1, e1_n2, e2_n1, e2_n2)
    #            n += 1
    #    t2 = time.time()
    #    self.logger.debug(
    #            'took {}s to do {} swaps'.format(t2-t1, self.N_swap))

    def write_graph(self, weights = None):
        if weights is not None:
            iterator = zip(self.graph.iterEdges(), weights)
        else:
            iterator = self.graph.iterEdges() # TODO add zip (ones) pour ajouter des poids fictifs ? 
        for ((u, v), weight) in iterator:
            if u<v :
                fout.write(f'{u},{v} {weight}\n')
            else:
                fout.write(f'{v},{u} {weight}\n')

    def run(self):
        raise NotImplementedError

class EdgeSwitchingMarkovChain(AbstractGraphGenerator):
    """ Wrapper of Networkit EdgeSwitchingMarkovChainGenerator 
        Also called 'configuration model' in networkit.
        Parameters:
        -----------
        degreeSequence : vector[count]
            The degree sequence that shall be generated
        ignoreIfRealizable : bool, optional
            If true, generate the graph even if the degree sequence is not realizable. Some nodes may get lower degrees than requested in the sequence.
    """

    def __init__(self, **kwargs):
        self.sequence = kwargs['seq']
        self.generator = EdgeSwitchingMarkovChainGenerator(self.sequence)
        self.generator.isRealizable()
        self.graph = None

    def run(self):
        self.graph = self.generator.generate()

class HavelHakimi(AbstractGraphGenerator):
    """ Wrapper of Networkit HavelHakimi
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
        self.generator = HavelHakimiGenerator(self.sequence)
        self.N_swap = N_swap
        self.graph = None

    def run(self):
        self.graph = self.generator.generate()

class ErdosRenyi(AbstractGraphGenerator):
    """ Wrapper of Networkit ErdosRenyiGenerator
        Parameters:
           -----------
           nNodes : count
               Number of nodes n in the graph.
           prob : double
               Probability of existence for each edge p.
           directed : bool
               Generates a directed
           selfLoops : bool
               Allows self-loops to be generated (only for directed graphs)
    """   
    def __init__(self, n, p, N_swap):
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.generator = ErdosRenyiGenerator(self.n, self.p,
                                directed = False, selfLoops=False)
        self.N_swap = N_swap
        self.graph = None

    def run(self):
        self.graph = self.generator.generate()

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
    def __init__(self, n, m, seed=None):
        self.n = n
        self.m = m 
        #self.seed = seed if not None else None ## TODO

    @staticmethod ## 
    def _clique(G):
        """ From Networkx.generators.classic.complete_graph
        """
        #n_name, nodes = n
        #G = empty_graph(n_name, create_using)
        import itertools
        nodes = range(G.numberOfNodes())
        if len(nodes) > 1:
            edges = itertools.combinations(nodes, 2)
            G.add_edges_from(edges)
        return G

    def run(self):
        """ can add nodes to existing GNM, and create GNM of these nodes """ 
        self.graph = nk.graph.Graph()
        self.graph.addNodes(self.n)

        # if n = 1, return 1 node graph,
        # if m = n(n-1)/2, return clique
        if self.n == 1:
            return self.graph
        max_edges = self.n * (self.n - 1) / 2.0
        if self.m >= max_edges:
            return GNM._clique(self.graph)

        # pick two nodes at random, create edge if it doesn't already exist
        nlist = range(self.n)
        edge_count = 0
        while edge_count <= self.m:
            u = np.random.choice(nlist)
            v = np.random.choice(nlist)
            if u == v or self.graph.hasEdge(u, v):
                continue
            else:
                self.graph.addEdge(u, v)
                edge_count = edge_count + 1

class FromDataWithAnomaly(AbstractGraphGenerator):
    ## TODO Move to other file
    """
        
        I tirer noir données réelles 
        
        II former rouge avec modèle anomalie (erdos renyii où on donne  le nombre de noeud/liens ?)
        
        
        III1 Tirer bleu (trouver endroit où rouge va cf iii) aléatoirement (havel hakimi + mélanges) 
        
        III2 si III1 marche pas, mélanger bleu ou rouge (avec proba de choix bleu ou rouge)
        
        III3 si III2 marche pas, switch spécifiquement 

        Implémentation:
        3 graphes : anomalie(s), normal & global (global = normal + anomalie)
        Dans l'implémentation, on part de la séquence global, on construit l'anomalie puis on construit le global

        Stocker graphes anomalies et graphe normal séparément, jamais construire graphe global, seulement garder en mémoire où "brancher" les noeds de l'anomalie sur le normal

        an_* = anomaly
        norm_* = normal graph
    """
    def __init__(self, degree_list,
            numberOfAnomaly,
            n_anomaly,
            m_anomaly, 
            N_swap,
            logger):

        self.logger = logger

        # global = normal graph + anomaly graph
        #self.dataset = kwargs['dataset']
        self.global_degree_list = np.array(degree_list)

        # anomaly parameters to generate Erdos Renyi
        self.numberOfAnomaly = numberOfAnomaly # number of anomalies
        self.n_anomaly = n_anomaly
        self.m_anomaly = m_anomaly # pick erdos renyi for anomaly
        self.anomaly_seq = dict()
        self.G_anomalies = [] # graph

        # normal graph
        self.G_normal = None
        self.N_swap = N_swap #18215698 #* 10 # N_swap

        # can break when 1 multiedge detected can start generation from scratch
        self.break_when_multiple = False 

        # mapping from current global graph to networkx (anomaly) indices
        self.norm2an = dict()
        self.an2norm = dict()

    def _generate_anomaly(self):
        """ Generate numberOfAnomaly anomalies
            Each anomaly is generated independently as disjoint graphs,
            with the same parameters TODO : allow different parameters for each anomaly
            stored as a list of graphs.
        """
        # instantiate and generate all anomalies
        for an_i in range(self.numberOfAnomaly):
            self.G_anomalies.append(GNM(self.n_anomaly, self.m_anomaly))
            self.G_anomalies[an_i].run()

    def _get_normality_degree_seq(self):
        """ Get degree sequence for 'normal' graph, by pluging anomaly node in 
            global graph degree sequence, and substracting the anomaly 
            degrees to get normal degree.
            global degree sequence = (gd1, gd2, ... gdn)
            normal degree sequence = (nd1, nd2, ... ndn)
            anomalies degree sequences = (ad1, ad2, ... adn)
            where
            ```
                gdi = ndi + adi
            ```
            In practice : # TODO pseudo code with nice math display 
            -For i in (1, n), get list of indices (k1, ... kj) such that
            ```
                for m in (1...j)  gd_km >= ad_km & gd_km not already selected
            ```
            -Pick n in (1...j) uniformly at random, attribute node _km of anomaly to
             node _km of global graph, and mark node _km of global graph as already selected.
            -When picking a node that has already been selected, start from scratch.
        """
        has_duplicate_node = True
        node_selection = []

        # get anomalies degree list
        an_degree_list = []
        an_degree_pos = [] # index of first node of each graph in degree list
        for an_i in range(self.numberOfAnomaly):
            an_degree_pos.append(len(an_degree_list))
            an_degree_list += self.G_anomalies[an_i].degree_list
        else:
            # convert to np.array to get nice numpy index seach
            an_degree_list = np.array(an_degree_list) 

        # plug all anomalies nodes to normal graph degree series
        # start from scratch if two anomaly nodes end up on the same normal
        # graph node
        while (has_duplicate_node):
            for (index, (an_node, degree)) in enumerate(an_degree_list):

                # get nodes with degree high enough to place anomaly node
                an = np.where(
                        np.array(an_degree_pos) <= index)[0][-1]

                # pick one at random
                candidate_indices = np.where(self.global_degree_list[:,1] > degree)
                candidate_node = np.random.choice(candidate_indices[0])

                # if node has already been chosen, pick all nodes again
                if (candidate_node, self.global_degree_list[candidate_node, 0]) in node_selection:
                    node_selection = []
                    has_duplicate_node = True
                    break
                node_selection.append((candidate_node, self.global_degree_list[candidate_node,0]))
            else:

                # when nodes are picked, substract anomaly degrees 
                has_duplicate_node = False
                for (index, 
                  ((an_node, an_degree), 
                  (node_idx, norm_node))) in enumerate(zip(an_degree_list, node_selection)):

                    # get index of graph in list of anomaly graphs
                    an_index = np.where(np.array(an_degree_pos) <= index)[0][-1] 

                    # set mapping from global to networkit node index
                    self.norm2an[norm_node] = (an_index, an_node)
                    self.an2norm[(an_index, an_node)] = norm_node

                    # substrace degrees
                    self.global_degree_list[node_idx,1] -= an_degree # substract degree from anomaly

    def _generate_normality(self):
        """ Generate 'normal' graph using Havel-Hakimi algorithm + edge switching"""
        self.G_normal = HavelHakimi(self.global_degree_list[:,1], self.N_swap, self.logger)
        self.G_normal.run()
        self.G_normal._edge_swap() ## TODO migrate edge switch in "run" when edge switch flag

    def _check_multiple_edges(self):
        """ check if Global graph has multiple edges """

        # keep track of multiple edges in case we can to switch them later
        multiple_edges = []
        for (norm_n1, norm_n2) in self.G_normal.graph.iterEdges():

            # if any of those nodes are not in G_anomaly,
            # current edge is not a multiple edge.
            if (norm_n1 in self.norm2an and norm_n2 in self.norm2an): 

                # get anomaly nodes names and check if they are in same graph
                an_index1, an_n1 = self.norm2an[norm_n1]
                an_index2, an_n2 = self.norm2an[norm_n2]

                # check if current edge already exists in anomaly
                if ((an_index1 == an_index2) and self.G_anomalies[an_index1].graph.hasEdge(an_n1, an_n2)): 
                    multiple_edges.append(((norm_n1, norm_n2), (an_index1, (an_n1, an_n2))))
                    if self.break_when_multiple:
                        self.logger.warning('Multiedge in normal+anomaly graph,'
                          'and break_when_multiple set to true, stopping generation.')
                        break

        if len(multiple_edges) > 0:
            self.logger.info(
              '{} Multiedge in normal+anomaly graph detected.'.format(len(multiple_edges)))
        return multiple_edges
    
    @staticmethod
    def _swap_edge(edge1, G):
        ''' Given graph and 1 edge pick another edge uniformly at random and
            swap edges
        '''
        (e1_n1, e1_n2) = edge1

        # check if didn't pick two times the same edge 
        # or if swaped edges already exist
        acceptable_swap = False
        while acceptable_swap == False:
            (e2_n1, e2_n2) = nk.graphtools.randomEdge(G, uniformDistribution=True)
            if ((e1_n1 == e2_n1) or (e1_n1 == e2_n2) or (e1_n2 == e2_n1) or (e1_n2 == e2_n2)
             or G.hasEdge(e1_n1, e2_n2) or G.hasEdge(e2_n1, e1_n2)):
                acceptable_swap = False
            else:
                acceptable_swap = True
                return (e2_n1, e2_n2)
    
    def swap_multiedges(self, multiple_edges):
        """ If multiple edges are detected between normal graph and anomaly, 
            target them specifically when switching
        """
        self.logger.info('Swapping edges, targetting multiedges specifically')
        self.logger.debug('Swapping {} edges'.format(len(multiple_edges)))
        while (len(multiple_edges) > 0):

            # pick an edge at random ## TODO not necessary ? 
            edge_index = np.random.randint(low=0, high=len(multiple_edges))
            ((norm_n1, norm_n2), (an_index, (an_n1, an_n2))) = multiple_edges[edge_index]
            (norm_edge1, (an_index, an_edge1)) = multiple_edges[edge_index]

            # choose at random which of the normal graph or anomaly to update
            p = random.uniform(0, 1)
            if p>=0.5: # TODO remonter proba comme paramètre 
                norm_edge2 = self._swap_edge(norm_edge1, self.G_normal.graph)

                # check if swap is not multiple link with anomaly
                # TODO should remove check for norm_n1, assum exist since multiple edge :)
                if ((norm_edge1[0] in self.norm2an and norm_edge2[1] in self.norm2an and norm_edge1[1] in self.norm2an and norm_edge2[0] in self.norm2an) 
                    and (self.G_anomalies[an_index].graph.hasEdge(self.norm2an[norm_edge1[0]][1], self.norm2an[norm_edge2[1]][1])
                        or self.G_anomalies[an_index].graph.hasEdge(self.norm2an[norm_edge2[0]][1], self.norm2an[norm_edge1[1]][1]))):
                        # don't do swap, pick new edge again ...
                        continue 
                else:
                    self.G_normal.graph.swapEdge(norm_edge1[0], norm_edge1[1], norm_edge2[0], norm_edge2[1])
                    multiple_edges.pop(edge_index)

            else:
                an_edge2 = self._swap_edge(an_edge1, self.G_anomalies[an_index].graph) ## TODO clean coherence arguments

                # check if swap is not multiple link with normal graph 
                if (self.G_normal.graph.hasEdge(self.an2norm[(an_index, an_edge1[0])], self.an2norm[(an_index, an_edge2[1])]) 
                       or self.G_normal.graph.hasEdge(self.an2norm[(an_index, an_edge2[0])], self.an2norm[(an_index, an_edge1[1])])):
                    # don't do swap, pick new edge again ...
                    continue
                else:
                    #try:
                    self.G_anomalies[an_index].graph.swapEdge(an_edge1[0], an_edge1[1], an_edge2[0], an_edge2[1])
                    #except:
                    #    ipdb.set_trace()
                    #    self.G_anomalies[an_index].graph.swapEdge(an_n1, an_n2, e2_n1, e2_n2)
                multiple_edges.pop(edge_index)
            self.logger.info('{} edges left'.format(len(multiple_edges)))

    def run(self):
        self.logger.info('generating anomaly')
        self._generate_anomaly()
        self.logger.info('getting normal graph degree sequence')
        self._get_normality_degree_seq()
        self.logger.info('generating normal graph')
        self._generate_normality()
        self.logger.info('checking for multiple links')
        multiple_edges = self._check_multiple_edges()
        #for ((norm1, norm2), (anind, (an1, an2))) in multiple_edges:
        #    print(self.G_anomalies[anind].graph.hasEdge(an1, an2))
        if len(multiple_edges) > 0:
            self.swap_multiedges(multiple_edges)


