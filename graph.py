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

    def write_graph(self, weights = None):
        ## TODO random weight
        # sort nodes
        #sum_weight = 0
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

class ConfigurationModel(AbstractGraphGenerator):
    """ Implementation of configuration model
    """
    def __init__(self, **kwargs):
        self.seq = kwargs['seq']
        self.anomaly_seq = kwargs['anomaly_seq']
        self.anomaly_type = kwargs['anomaly_type']
        raise NotImplementedError

    def _clique(self):
        ## TODO: Idée à discuter avec Matthieu: on part de 2 graphes, un avec aucun lien G0, un avec tous GN, 
        # à chaque fois qu'on ajoute un lien dans G0, on enlève ce lien dans GN => on a toutes les cliques possibles,
        # reste à trouver celle de la taille qu'on veut (probably long)
        # possiblement très overkill
        raise NotImplementedError

    def _configuration_run(self):
        raise NotImplementedError

    def run(self):
        # generate anomaly first then feed into model
        anomaly = getattr(self, kwargs['anomaly_type'])
        self.anomaly(self.graph)
        
        self._configuration_run()
        raise NotImplementedError

class EdgeSwitchingMarkovChain(AbstractGraphGenerator):
    """ Wrapper of Networkit EdgeSwitchingMarkovChainGenerator 
        Also called 'configuration model' in networkit.
        TODO
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
        TODO
        Parameters:
        -----------
        sequence : vector
            Degree sequence to realize. Must be non-increasing.
        ignoreIfRealizable : bool, optional
            If true, generate the graph even if the degree sequence is not realizable. Some nodes may get lower degrees than requested in the sequence.
    """

    #def __init__(self, **kwargs):
    def __init__(self, sequence , N_switch, logger):
        self.logger = logger
        self.sequence = sequence
        self.generator = HavelHakimiGenerator(self.sequence)
        self.N_switch = N_switch
        self.graph = None
        #self.anomaly_sequence = kwargs['anomaly']['sequence'] # should be the same length as self.sequence, with 0s

    #def _havelhakimi(self):
    #    """ python reimplementation of networkit'havel-hakimi algo -- overkill .. ? """
    #    N = len(self.sequence)
    #    numDegVals = max(sequence)
    #    from collections import  defaultdict
    #    nodesByDeficit = defaultdict(list)
    #    for v, seq_v in enumerate(self.sequence):
    #        nodeByDeficit[seq_v].append((seq_v, v))

    #    maxDeficit = numDegVals - 1


    #    while maxDeficit:
    #        while len(nodesByDeficit[maxDeficit]) > 0:
    #            (deficit, currentVertex) = nodesByDeficit[maxDeficit]
    #            nodesByDeficit.pop(0)

    #            currentNeighborList = maxDeficit

    def _edge_switch(self):
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
            #(e1_n1, e1_n2) = self.graph.randomEdge()
            #(e2_n1, e2_n2) = self.graph.randomEdge()
            (e1_n1, e1_n2) = nk.graphtools.randomEdge(self.graph, uniformDistribution=True)
            (e2_n1, e2_n2) = nk.graphtools.randomEdge(self.graph, uniformDistribution=True)


            # check if didn't pick two times the same edge 
            # or if switched edge already exists
            if ((e1_n1 == e2_n1)              or (e1_n1 == e2_n2)               or (e1_n2 == e2_n1)               or (e1_n2 == e2_n2)                or self.graph.hasEdge(e1_n1, e2_n2)               or self.graph.hasEdge(e2_n1, e1_n2)):
                #print(n)
                #ipdb.set_trace()
                #n_attempt+=1
                continue
            else:
                #print('took {}s, {} attempt for successfule loop'.format(time.time() -t1, n_attempt))
                #n_attempt = 0
                #t1 = time.time()
                #self.logger.debug('took {} to switch'.format(n_attempt))
                #n_attempt = 0
                self.graph.swapEdge(e1_n1, e1_n2, e2_n1, e2_n2)
                n += 1
        t2 = time.time()
        self.logger.info('took {}s to do {} switches'.format(t2-t1, self.N_switch))
        #print('edge pick took {}, swap took {}, total took {}, n is {}'.format(t2-t1, t3 -t2, t3-t1, n))

    def run(self):
        self.graph = self.generator.generate()

class ErdosRenyi(AbstractGraphGenerator):
    """ Wrapper of Networkit ErdosRenyiGenerator
        TODO
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
    #def __init__(self, n, p):
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
    def __init__(self, n, m):
        #self.n = kwargs['n']
        #self.m = kwargs['m'] +1
        self.n = n
        #self.m = m + 1
        # TODO remove just for anomaly testing
        self.m = m #n * (n-1) / 2
        #self.seed = seed if not None else None

    def write_graph(self, out_path, weights):
        ## TODO random weight
        # sort nodes
        #sum_weight = 0
        if weights is not None:
            print(len(self.graph.edges()))
            print(len(weights))
            iterator = zip(self.graph.edges(), weights)
        else:
            iterator = self.graph.iterEdges()
        with open(out_path, 'w') as fout:


            for ((u, v), weight) in iterator:
                # don't write if weight is 0
                if weight <= 0:
                    continue

                if u<v :
                    fout.write(f'{u},{v} {weight}\n')
                else:
                    fout.write(f'{v},{u} {weight}\n')

    #@parameter
    def degree_list(self):
        degree_list = []
        for u in self.graph.iterNodes():
            degree_list.append((u,self.graph.degree(u)))
        return degree_list

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
        #if directed:
        #    G = nx.DiGraph()
        #else:
        #    G = nx.Graph()
        self.graph = nk.graph.Graph() # .Graph()?   
        #G.add_nodes_from(range(n))
        self.graph.addNodes(self.n) 

        if self.n == 1:
            return self.graph
        max_edges = self.n * (self.n - 1) / 2.0
        #if not directed:
        #    max_edges /= 2.0
        if self.m >= max_edges:
            return GNM._clique(self.graph) # TODO reimplem nk .. ? 

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
        #return G


    #def degree_seq(self):
    #    G.degree # in networkX .. networkit ?
    #    raise NotImplementedError

    #def run(self):
    #    self.graph = self.generate()#gnm_random_graph(self.n, self.m, seed=None, directed=False)



#class GNM_old(AbstractGraphGenerator):
#    """ Wrapper of Networkx gnm_random_graph
#    Parameters:
#    -----------
#        n: int
#            number of nodes
#        m: int
#            number of edges
#        seed: int
#            random seed
#    """
#    def __init__(self, n, m):
#        #self.n = kwargs['n']
#        #self.m = kwargs['m'] +1
#        self.n = n
#        #self.m = m + 1
#        # TODO remove just for anomaly testing
#        self.m = m #n * (n-1) / 2
#        #self.seed = seed if not None else None
#
#    def write_graph(self, out_path, weights):
#        ## TODO random weight
#        # sort nodes
#        #sum_weight = 0
#        if weights is not None:
#            print(len(self.graph.edges()))
#            print(len(weights))
#            iterator = zip(self.graph.edges(), weights)
#        else:
#            iterator = self.graph.iterEdges()
#        with open(out_path, 'w') as fout:
#
#
#            for ((u, v), weight) in iterator:
#                # don't write if weight is 0
#                if weight <= 0:
#                    continue
#
#                if u<v :
#                    fout.write(f'{u},{v} {weight}\n')
#                else:
#                    fout.write(f'{v},{u} {weight}\n')
#
#    def degree_seq(self):
#        G.degree # in networkX .. networkit ?
#        raise NotImplementedError
#
#    def run(self):
#        self.graph = gnm_random_graph(self.n, self.m, seed=None, directed=False)

class FromDataWithAnomaly(AbstractGraphGenerator):
    """
        
        I tirer noir données réelles 
        
        II former rouge avec modèle anomalie (erdos renyii où on donne  le nombre de noeud/liens ?)
        
        
        III1 Tirer bleu (trouver endroit où rouge va cf iii) aléatoirement (havel hakimi + mélanges) 
        
        III2 si III1 marche pas, mélanger bleu ou rouge (avec proba de choix bleu ou rouge)
        
        III3 si III2 marche pas, switch spécifiquement 
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
        self.G_anomaly = None # graph

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
        for i in range(self.numberOfAnomaly):
            self.G_anomaly = GNM(self.n_anomaly, self.m_anomaly)
            self.G_anomaly.run()

    def _get_normality_degree_seq(self):
        """ Get degree sequence for 'normal' graph, by placing anomaly in 
            global graph degree sequence, and substracting the anomaly 
            degrees
        """
        ### TODO get taxonomy normal/anomaly
        #self.anomaly_seq = self.G.seq # todo later after selecting 
        #ipdb.set_trace()
        has_duplicate_node = True
        node_selection = []

        # place anomaly in global graph by choosing randomly
        # the nodes that have a degree high enough.
        # pick again if two anomaly nodes get mapped to the same node in 
        # global graph.
        degree_list = self.G_anomaly.degree_list()
        while (has_duplicate_node):
            #for node, degree in self.G_anomaly.graph.degree: # TODO remonter degree a attribut de GNM
            for node, degree in degree_list: # TODO remonter degree a attribut de GNM
                # get nodes with degree high enough to place current anomaly
                # node 
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
                for ((nx_node, an_degree), (nG_idx, node)) in zip(degree_list, node_selection):
                    # set mapping from global to networkx node index
                    self.node2nx[node] = nx_node
                    self.nx2node[nx_node] = node
                    # substrace degrees
                    self.degree_list[nG_idx,1] -= an_degree # substract degree from anomaly

    def _generate_normality(self):
        """ Generate 'normal' graph using Havel-Hakimi algorithm + edge switching"""
        self.G_normal = HavelHakimi(self.degree_list[:,1], self.N_switch, self.logger)
        self.G_normal.run()
        self.G_normal._edge_switch() ## TODO migrate edge switch in "run" when edge switch flag

    def _check_multiple_edges(self):
        """ check if Global graph has multiple edges """

        # keep track of multiple edges in case we can to switch them later
        has_multiple_edge = False
        multiple_edge_list = []
        for (node1, node2) in self.G_normal.graph.iterEdges():

            # if any of those nodes are not in G_anomaly,
            # current edge is not a multiple edge.
            if (node1 in self.node2nx and node2 in self.node2nx): 
                nx_node1 = self.node2nx[node1]
                nx_node2 = self.node2nx[node2]

                # check if current edge already exists in anomaly
                if ((nx_node1, nx_node2) in self.G_anomaly.graph.edges() 
                    or (nx_node2, nx_node1) in self.G_anomaly.graph.edges()):
                    has_multiple_edge = True
                    multiple_edge_list.append((node1, node2, nx_node1, nx_node2))
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
            if ((e1_n1 == e2_n1)
             or (e1_n1 == e2_n2)
             or (e1_n2 == e2_n1)
             or (e1_n2 == e2_n2)
             or G.hasEdge(e1_n1, e2_n2)
             or G.hasEdge(e2_n1, e1_n2)):
                acceptable_swap = False
                #ipdb.set_trace()
                #n_attempt+=1
                #continue
            else:
                acceptable_swap = True
            #print('took {}s, {} attempt for successfule loop'.format(time.time() -t1, n_attempt))
            #n_attempt = 0
            #t1 = time.time()
            #self.logger.debug('took {} to switch'.format(n_attempt))
            #n_attempt = 0

        #raise NotImplementedError
        return (e2_n1, e2_n2) 
    
    #def _swap_anomaly(e1_n1, e1_n2):
    #    """ switch an edge in the anomaly graph"""
    #    # boolean to be set to true when an edge swap has been accepted
    #    acceptable_swap = False
    #    while (not acceptable_swap):
    #        # pick two edges at random
    #        #(e1_n1, e1_n2) = np.random.choice(self.G_anomaly.edges())
    #        (e1_n1, e2_n2) = an_node1
    #        (e2_n1, e2_n2) = np.random.choice(self.G_anomaly.edges())

    #        # swap them if the swapped edges don't already exist
    #        if (((e1_n1, e2_n2) not in self.G_anomaly.edges())
    #         and ((e2_n1, e1_n2) not in self.G_anomaly.edges())):
    #            self.G_anomaly.add_edge((e2_n1, e1_n2))
    #            self.G_anomaly.add_edge((e1_n1, e2_n2))
    #            self.G_anomaly.remove_edge((e1_n1, e1_n2))
    #            self.G_anomaly.remove_edge((e2_n1, e2_n2))

    #            # accept swap
    #            acceptable_swap = True

        return (e2_n1, e2_n2)

    def swap_multiedges(self, multiple_edges):
        """ If multiple edges are detected between normal graph and anomaly, 
            target them specifically when switching
        """
        self.logger.info('swapping multiple edges')
        self.logger.info('{} edges left'.format(len(multiple_edges)))
        while (len(multiple_edges) > 0):
            #((node1, node2, an_node1, an_node2)) = np.random.choice(multiple_edge_list)
            edge_index = np.random.randint(low=0, high=len(multiple_edges))
            ((norm_n1, norm_n2, an_n1, an_n2)) = multiple_edges[edge_index] 
            p = random.uniform(0, 1)

            # choose at random which of the normal graph or anomaly to update
            if p>=0.5: 
                (e1_n1, e1_n2) = (norm_n1, norm_n2)
                (e2_n1, e2_n2) = self._swap_edge((norm_n1, norm_n2), self.G_normal.graph)

                # check if swap is not multiple link with anomaly    
                if e1_n1 in self.node2nx and e2_n2 in self.node2nx:
                    if (self.G_anomaly.graph.hasEdge(self.node2nx[e1_n1], self.node2nx[e2_n2]) 
                             or self.G_anomaly.graph.hasEdge(self.node2nx[e2_n1], self.node2nx[e1_n2])):
                        # don't do swap, pick new edge again ...
                        continue 
                        #    multiple_edge_list.append((self.nx2node[(e1_n1, e2_n2)], (e1_n1, e2_n2)))
                        #if self.G_anomaly.graph.hasEdge(self.nx2node[(e2_n1, e1_n2)]):
                        #    multiple_edge_list.append((self.nx2node[(e1_n1, e2_n2)], (e1_n1, e2_n2)))
                else:
                    self.G_normal.graph.swapEdge(e1_n1, e1_n2, e2_n1, e2_n2)
                    multiple_edges.pop(edge_index)

            else:
                (e1_n1, e1_n2) = (an_n1, an_n2)
                (e2_n1, e2_n2) = self._swap_edge((an_n1, an_n2), self.G_anomaly.graph) ## TODO clean coherence arguments

                # check if swap is not multiple link with normal graph 
                if (self.G_normal.graph.hasEdge(self.nx2node[e1_n1], self.nx2node[e2_n2]) 
                       or self.G_normal.graph.hasEdge(self.nx2node[e2_n1], self.nx2node[e1_n2])):
                    # don't do swap, pick new edge again ...
                    continue
                    #    multiple_edge_list.append((self.nx2node[(e1_n1, e2_n2)], (e1_n1, e2_n2)))
                    #if self.G_normal.graph.hasEdge(self.nx2node[(e2_n1, e1_n2)]):
                    #    multiple_edge_list.append((self.nx2node[(e1_n1, e2_n2)], (e1_n1, e2_n2)))
                else:
                    self.G_anomaly.graph.swapEdge(e1_n1, e1_n2, e2_n1, e2_n2)
                multiple_edges.pop(edge_index)
            self.logger.info('{} edges left'.format(len(multiple_edges)))



            # check new edge doesn't already exist
            # maybe we messed up hard and created 2 multiple links from 1 ... ## TODO in that case reject change ??  

            #_check_new_edge()

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


