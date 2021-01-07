import time
import random
import numpy as np

from collections import defaultdict
from GTgen.graphModels import *

class GraphWithAnomaly():
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
            self.G_anomalies.append(GNM(self.n_anomaly, self.m_anomaly, self.logger))
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
            an_degree_list += self.G_anomalies[an_i].graph.degrees.items()
        else:
            # convert to np.array to get nice numpy index seach
            an_degree_list = np.array(an_degree_list) 

        # plug all anomalies nodes to normal graph degree series
        # start from scratch if two anomaly nodes end up on the same normal
        # graph node
        while (has_duplicate_node):
            #for (index, (an_node, degree)) in enumerate(an_degree_list):
            for index, (an_node, degree) in enumerate(an_degree_list):

                # get nodes with degree high enough to place anomaly node
                an = np.where(
                        np.array(an_degree_pos) <= index)[0][-1]

                # pick one at random
                candidate_indices = np.where(self.global_degree_list[:,1] >=  degree)
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
                    #ipdb.set_trace()
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
        #_generator = HavelHakimiGenerator(self.global_degree_list[:, 1], ignoreIfRealizable=False)

        #t0 = time.time()
        #_graph = _generator.generate()

        t0 = time.time() 
        #self.logger.info('{}s for nk hakimi'.format(t1 - t0))
        self.G_normal.run()
        t1 = time.time()
        self.logger.info('{}s for python hakimi'.format(t1 - t0))
        self.G_normal._edge_swap() ## TODO migrate edge switch in "run" when edge switch flag

    def _check_multiple_edges(self):
        """ check if Global graph has multiple edges """

        # keep track of multiple edges in case we can to switch them later
        multiple_edges = []
        #for (norm_n1, norm_n2) in self.G_normal.graph.iterEdges():
        #ipdb.set_trace()
        for  norm_edge in self.G_normal.graph.edges:
            (norm_n1, norm_n2) = norm_edge

            # if any of those nodes are not in G_anomaly,
            # current edge is not a multiple edge.
            if (norm_n1 in self.norm2an and norm_n2 in self.norm2an): 

                # get anomaly nodes names and check if they are in same graph
                an_index1, an_n1 = self.norm2an[norm_n1]
                an_index2, an_n2 = self.norm2an[norm_n2]

                an_edge = (an_n1, an_n2) if an_n1 < an_n2 else (an_n2, an_n1)

                # check if current edge already exists in anomaly
                if ((an_index1 == an_index2) and self.G_anomalies[an_index1].graph.hasEdge(an_edge)): 
                    multiple_edges.append((norm_edge, (an_index1, an_edge)))
                    if self.break_when_multiple:
                        self.logger.warning('Multiedge in normal+anomaly graph,'
                          'and break_when_multiple set to true, stopping generation.')
                        break

        if len(multiple_edges) > 0:
            self.logger.info(
              '{} Multiedge in normal+anomaly graph detected.'.format(len(multiple_edges)))
        return multiple_edges
    
    #@staticmethod
    #def _acceptable_edge(edge1, G):
    #    ''' Given graph and 1 edge pick another edge uniformly at random and
    #        swap edges
    #    '''
    #    print('in acceptable edge')
    #    (e1_n1, e1_n2) = edge1
    #    acceptable_swap = False
    #    while acceptable_swap == False:
    #        # pick random edge u,v , direction matters for swap
    #        #available_edges = G.edges + [(n2, n1) for (n1, n2) in G.edges]
    #        (e2_n1, e2_n2) = G.edges[np.random.choice(len(G.edges))]

    #        # check if swap is acceptable
    #        #if ((e1_n1 == e2_n1) or (e1_n1 == e2_n2) or (e1_n2 == e2_n1) or (e1_n2 == e2_n2)
    #        if (e1_n1 in (e2_n1, e2_n2) or e1_n2 in (e2_n1, e2_n2)):
    #            #or G.hasEdge((e1_n1, e2_n2)) or G.hasEdge((e2_n1, e1_n2))):
    #            acceptable_swap = False
    #        else:
    #            acceptable_swap = True
    #            print('exiting acceptable edge')
    #            return (e2_n1, e2_n2)
    
    def swap_multiedges(self, multiple_edges):
        """ If multiple edges are detected between normal graph and anomaly, 
            target them specifically when switching
        """
        self.logger.info('Swapping edges, targetting multiedges specifically')
        self.logger.debug('Swapping {} edges'.format(len(multiple_edges)))
        while (len(multiple_edges) > 0):
            # pick an edge at random ## TODO not necessary ? 
            edge_index = np.random.randint(low=0, high=len(multiple_edges))
            #((norm_n1, norm_n2), (an_index, (an_n1, an_n2))) = multiple_edges[edge_index]
            (norm_edge1, (an_index, an_edge1)) = multiple_edges[edge_index]

            # choose at random which of the normal graph or anomaly to update
            p = random.uniform(0, 1)
            if p>=0.5: # TODO remonter proba comme paramètre 
                #norm_edge2 = self._acceptable_edge(norm_edge1,
                #                                   self.G_normal.graph)
                norm_edge2 = self.G_normal.graph.edges[
                    np.random.choice(len(self.G_normal.graph.edges))]
                # 1/2 chance of reversing _edge2 : equivalent to picking both
                # directions at random
                reverse = random.uniform(0,1) >= 0.5
                norm_edge1_idx = self.G_normal.graph.edges.index(norm_edge1)
                norm_edge2_idx = self.G_normal.graph.edges.index(norm_edge2)

                # check if swap is not multiple link with anomaly
                # TODO TODO CLEAN THIS UP IT'S UNREADABLE 
                if (( norm_edge2[1] in self.norm2an and norm_edge2[0] in self.norm2an) 
                 and (self.G_anomalies[an_index].graph.hasEdge((self.norm2an[norm_edge1[0]][1], self.norm2an[norm_edge2[1]][1]))
                   or self.G_anomalies[an_index].graph.hasEdge((self.norm2an[norm_edge2[0]][1], self.norm2an[norm_edge1[1]][1])))):
                    # don't do swap, pick new edge again ...
                    continue 
                else:
                    try:
                        self.G_normal.graph.swapEdge(norm_edge1_idx, norm_edge2_idx)
                        multiple_edges.pop(edge_index)
                    except AssertionError as err:
                        self.logger.debug('Swap was rejected: {}'.format(err))
                        continue
                    multiple_edges = self._check_multiple_edges()
            else:
                #an_edge2 = self._acceptable_edge(an_edge1, self.G_anomalies[an_index].graph) ## TODO clean coherence arguments
                an_edge2 = self.G_anomalies[an_index].graph.edges[
                             np.random.choice(
                               len(self.G_anomalies[an_index].graph.edges))]
                # 1/2 chance of reversing _edge2 : equivalent to picking both
                # directions at random
                reverse = random.uniform(0,1) >= 0.5
                an_edge1_idx = self.G_anomalies[an_index].graph.edges.index(an_edge1)
                an_edge2_idx = self.G_anomalies[an_index].graph.edges.index(an_edge2)

                #if random.uniform(0,1) >= 0.5:
                #    an_edge2 = (_an_edge2[1], _an_edge2[0])
                #else:
                #    an_edge2 = _an_edge2

                # check if swap is not multiple link with normal graph 
                if (self.G_normal.graph.hasEdge((self.an2norm[(an_index, an_edge1[0])], self.an2norm[(an_index, an_edge2[1])])) 
                 or self.G_normal.graph.hasEdge((self.an2norm[(an_index, an_edge2[0])], self.an2norm[(an_index, an_edge1[1])]))):
                    # don't do swap, pick new edge again ...
                    continue
                else:
                    # try swap unless it's rejected
                    try:
                        self.G_anomalies[an_index].graph.swapEdge(an_edge1_idx, an_edge2_idx, reverse)
                        multiple_edges.pop(edge_index)
                    except AssertionError as err:
                        self.logger.debug('Swap was rejected: {}'.format(err))
                        # if swap is not accepted
                        continue

                    multiple_edges = self._check_multiple_edges()
                #if len(multiple_edges) % 10 == 0:
                self.logger.debug('{} multiple edges left'.format(len(multiple_edges)))
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
        self.logger.info('{} multiple edges'.format(len(multiple_edges)))
        #if len(multiple_edges) > 0:
        #    self.swap_multiedges(multiple_edges)


