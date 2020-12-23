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

    def is_realisable():
        raise NotImplementedError

    #@parameter
    def degree_list(self):
        degree_list = []
        for u in self.graph.iterNodes():
            degree_list.append((u,self.graph.degree(u)))
        return degree_list

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

    def generate_weight(self):
        raise NotImplementedError

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

    def __init__(self, sequence , N_switch, logger):
        self.logger = logger
        self.sequence = sequence
        self.generator = HavelHakimiGenerator(self.sequence)
        self.N_switch = N_switch
        self.graph = None
        #self.anomaly_sequence = kwargs['anomaly']['sequence'] # should be the same length as self.sequence, with 0s

    def _edge_swap(self):
        """ shuffle to make 'random' graph'"""
        self.logger.info('switching {} edges in Havel Hakimi graph'.format(self.N_switch))
        # regarder genbip swaps pour configuration model
        #raise NotImplementedError
        n = 0
        t1 = time.time()
        n_attempt = 0
        while (n < self.N_switch):
            
            if n%10000000 == 0 and n>0:
                self.logger.debug('{} switches dones'.format(n))
            #((e1_n1, e1_n2), (e2_n1, e2_n2)) = gt.randomEdges(self.graph, 2)
            (e1_n1, e1_n2) = nk.graphtools.randomEdge(self.graph, uniformDistribution=True)
            (e2_n1, e2_n2) = nk.graphtools.randomEdge(self.graph, uniformDistribution=True)


            # check if didn't pick two times the same edge 
            # or if switched edge already exists
            if ((e1_n1 == e2_n1)              or (e1_n1 == e2_n2)               or (e1_n2 == e2_n1)               or (e1_n2 == e2_n2)                or self.graph.hasEdge(e1_n1, e2_n2)               or self.graph.hasEdge(e2_n1, e1_n2)):
                continue
            else:
                self.graph.swapEdge(e1_n1, e1_n2, e2_n1, e2_n2)
                n += 1
        t2 = time.time()
        self.logger.info('took {}s to do {} switches'.format(t2-t1, self.N_switch))
        #print('edge pick took {}, swap took {}, total took {}, n is {}'.format(t2-t1, t3 -t2, t3-t1, n))

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
    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.generator = ErdosRenyiGenerator(self.n, self.p, directed = False, selfLoops=False)
        self.graph = None

    def run(self):
        #assert
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
        self.m = m #n * (n-1) / 2
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
        self.graph = nk.graph.Graph() # .Graph()?   
        self.graph.addNodes(self.n) 

        if self.n == 1:
            return self.graph
        max_edges = self.n * (self.n - 1) / 2.0
        if self.m >= max_edges:
            return GNM._clique(self.graph)

        nlist = range(self.n)
        edge_count = 0
        while edge_count <= self.m:
            # generate random edge,u,v
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
            N_switch,
            logger):

        self.logger = logger

        # global = normal graph + anomaly graph
        #self.dataset = kwargs['dataset']
        self.degree_list = np.array(degree_list)

        # anomaly parameters to generate Erdos Renyi
        self.numberOfAnomaly = numberOfAnomaly # number of anomalies
        self.n_anomaly = n_anomaly
        self.m_anomaly = m_anomaly # pick erdos renyi for anomaly
        self.anomaly_seq = dict()
        self.G_anomalies = [] # graph

        # normal graph
        self.G_normal = None
        self.N_switch = N_switch #18215698 #* 10 # N_switch
        self.break_when_multiple = False # can break when 1 multiple edge detected => can start generation from scratch

        # mapping from current global graph to networkx (anomaly) indices
        self.node2nx = dict()
        self.nx2node = dict()

    def _generate_anomaly(self):
        """ Generate anomly as Erdos Renyi, using Networkx GNM model """
        ## TODO allow other models .. ?
        # draft : G_anomalies is a list of GNMs
        # to get normality degree, iterate over list than degree list
        # nk2node et node2nk : keep track of index in list + index of node
        # 
        # #TODO for i in range(self.numberOfAnomaly):
        # instantiate and generate all anomalies
        for an_i in range(self.numberOfAnomaly):
            self.G_anomalies.append(GNM(self.n_anomaly, self.m_anomaly))
            self.G_anomalies[an_i].run()

    def _get_normality_degree_seq(self):
        """ Get degree sequence for 'normal' graph, by placing anomaly in 
            global graph degree sequence, and substracting the anomaly 
            degrees
        """
        #self.anomaly_seq = self.G.seq # todo later after selecting 
        #ipdb.set_trace()
        has_duplicate_node = True
        node_selection = []

        # place anomaly in global graph by choosing randomly
        # the nodes that have a degree high enough.
        # pick again if two anomaly nodes get mapped to the same node in 
        # global graph.
        an_degree_list = [] ## TODO Confusion degree_list et self.degree_list ... 
        an_degree_pos = [] # index of first node of each graph 
        for an_i in range(self.numberOfAnomaly): ## TODO do sequentially or "all at once" : if one anomaly placed, don't try again from scratch ? 
            an_degree_pos.append(len(an_degree_list))
            an_degree_list += self.G_anomalies[an_i].degree_list() # TODO concat degree lists
        # TODO for n_anomaly

        # plug all anomalies nodes to normal graph degree series
        # start from scratch if two anomaly nodes end up on the same normal graph node
        while (has_duplicate_node):
            for (index, (node, degree)) in enumerate(an_degree_list): # TODO remonter degree a attribut de GNM
                # get nodes with degree high enough to place current anomaly
                # node 
                anomaly_index = np.where(np.array(an_degree_pos) <= index)[0][-1] # TODO encore un peu cradot

                # get all available nodes with high enough degree in normal graph
                # pick one at random
                candidate_indices = np.where(self.degree_list[:,1] > degree)

                node_candidate = np.random.choice(candidate_indices[0])

                # if node has already been chosen, pick all nodes again
                if (node_candidate, self.degree_list[node_candidate, 0]) in node_selection:
                    node_selection = []
                    has_duplicate_node = True
                    break

                node_selection.append((node_candidate, self.degree_list[node_candidate,0]))

            else:
                
                # when nodes are picked, substract anomaly degrees 
                has_duplicate_node = False
                #for ((nx_node, an_degree), (nG_idx, node)) in zip(self.G_anomaly.graph.degree, node_selection):
                for ((nx_node, an_degree), (node_idx, node)) in zip(an_degree_list, node_selection):
                    # set mapping from global to networkx node index
                    self.node2nx[node] = (anomaly_index, nx_node) # TODO replace nx_node by nk_node
                    self.nx2node[(anomaly_index, nx_node)] = node
                    # substrace degrees
                    self.degree_list[node_idx,1] -= an_degree # substract degree from anomaly

    def _generate_normality(self):
        """ Generate 'normal' graph using Havel-Hakimi algorithm + edge switching"""
        self.G_normal = HavelHakimi(self.degree_list[:,1], self.N_switch, self.logger)
        self.G_normal.run()
        self.G_normal._edge_swap() ## TODO migrate edge switch in "run" when edge switch flag

    def _check_multiple_edges(self):
        """ check if Global graph has multiple edges """

        # keep track of multiple edges in case we can to switch them later
        has_multiple_edge = False
        multiple_edge_list = []
        for (node1, node2) in self.G_normal.graph.iterEdges():

            # if any of those nodes are not in G_anomaly,
            # current edge is not a multiple edge.
            if (node1 in self.node2nx and node2 in self.node2nx): 
                # TODO here get (graph_index, nx_node) 
                # TODO check that both nodes are in same graph
                an_index1, nx_node1 = self.node2nx[node1]
                an_index2, nx_node2 = self.node2nx[node2]

                # check if current edge already exists in anomaly
                # TODO get graph then check condition on graph
                if ((an_index1 == an_index2) and self.G_anomalies[an_index1].graph.hasEdge(nx_node1, nx_node2)): 
                    has_multiple_edge = True
                    multiple_edge_list.append(((node1, node2), (an_index1, (nx_node1, nx_node2)))) # TODO cradot
                    if self.break_when_multiple:
                        self.logger.info('detected multiple edge')
                        break
        if len(multiple_edge_list) > 0:
            self.logger.info('detected multiple edge')
        return multiple_edge_list
    
    @staticmethod
    def _swap_edge(node1, G):
        (e1_n1, e1_n2) = node1
        #(e1_n1, e1_n2) = gt.randomEdge(self.graph)

        # check if didn't pick two times the same edge 
        # or if switched edge already exists
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
        self.logger.info('swapping multiple edges')
        self.logger.info('{} edges left'.format(len(multiple_edges)))
        while (len(multiple_edges) > 0):
            #((node1, node2, an_node1, an_node2)) = np.random.choice(multiple_edge_list)
            # pick an edge at random
            edge_index = np.random.randint(low=0, high=len(multiple_edges))
            ((norm_n1, norm_n2), (an_index, (an_n1, an_n2))) = multiple_edges[edge_index] 
            p = random.uniform(0, 1)

            # choose at random which of the normal graph or anomaly to update
            if p>=0.5: 
                #(e1_n1, e1_n2) = (norm_n1, norm_n2)
                (e2_n1, e2_n2) = self._swap_edge((norm_n1, norm_n2), self.G_normal.graph)

                # check if swap is not multiple link with anomaly
                # TODO should remove check for norm_n1, assum exist since multiple edge :)
                if ((norm_n1 in self.node2nx and e2_n2 in self.node2nx and norm_n2 in self.node2nx and e2_n1 in self.node2nx) 
                    and (self.G_anomalies[an_index].graph.hasEdge(self.node2nx[norm_n1][1], self.node2nx[e2_n2][1])
                        or self.G_anomalies[an_index].graph.hasEdge(self.node2nx[e2_n1][1], self.node2nx[norm_n2][1]))):
                        # don't do swap, pick new edge again ...
                        continue 
                else:
                    self.G_normal.graph.swapEdge(norm_n1, norm_n2, e2_n1, e2_n2)
                    multiple_edges.pop(edge_index)

            else:
                #(e1_n1, e1_n2) = (an_n1, an_n2) # un peu cradot
                (e2_n1, e2_n2) = self._swap_edge((an_n1, an_n2), self.G_anomalies[an_index].graph) ## TODO clean coherence arguments

                # check if swap is not multiple link with normal graph 
                if (self.G_normal.graph.hasEdge(self.nx2node[(an_index, an_n1)], self.nx2node[(an_index, e2_n2)]) 
                       or self.G_normal.graph.hasEdge(self.nx2node[(an_index, e2_n1)], self.nx2node[(an_index, an_n2)])):
                    # don't do swap, pick new edge again ...
                    continue
                else:
                    self.G_anomalies[an_index].graph.swapEdge(an_n1, an_n2, e2_n1, e2_n2)
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
        if len(multiple_edges) > 0:
            self.swap_multiedges(multiple_edges)


