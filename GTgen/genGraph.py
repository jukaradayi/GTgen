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
            weight,
            logger):

        self.logger = logger

        # global = normal graph + anomaly graph
        #self.dataset = kwargs['dataset']
        #self.global_degree_list = np.array(degree_list)
        self.global_degree_list = np.array(degree_list)
        self.normal_degree_list = np.empty((self.global_degree_list.shape[0], 2))

        # anomaly parameters to generate Erdos Renyi
        self.numberOfAnomaly = numberOfAnomaly # number of anomalies
        self.n_anomaly = n_anomaly
        self.m_anomaly = m_anomaly # pick erdos renyi for anomaly
        self.anomaly_seq = dict()
        self.G_anomalies = [] # graph

        # normal graph
        self.G_normal = None
        #self.N_swap = N_swap #18215698 #* 10 # N_swap
        N_edges = int(sum([deg for _,deg in degree_list])/2)
        #self.N_swap = int(N_swap * (degree_list))
        self.N_swap = int(N_swap * N_edges)

        # can break when 1 multiedge detected can start generation from scratch
        self.break_when_multiple = False 

        # mapping from current global graph to networkx (anomaly) indices
        self.norm2an = dict()
        self.an2norm = dict()

        # weight vector
        #self.weight = np.empty((int(sum([deg for _,deg in degree_list])/2),))
        self.weight = np.empty((N_edges,))
        prev_idx = 0
        for val, num in weight:
            self.weight[prev_idx:prev_idx+num] =  val
            prev_idx += num
    def _generate_anomaly(self):
        """ Generate numberOfAnomaly anomalies
            Each anomaly is generated independently as disjoint graphs,
            with the same parameters TODO : allow different parameters for each anomaly
            stored as a list of graphs.
        """
        # instantiate and generate all anomalies
        #for an_i in range(self.numberOfAnomaly):
        #    self.G_anomalies.append(GNM(self.n_anomaly, self.m_anomaly, set(range(self.n_anomaly * an_i, self.n_anomaly * (an_i+1))), self.logger))
        #    self.G_anomalies[an_i].run()
        #self.G_anomalie = GNM(self.n_anomaly, self.m_anomaly, set(range(
        self.G_anomaly = Graph(edges=[],
                #nodes=set(self.sequence[:,0]),
                nodes=set(),
                degrees=None, logger=self.logger)


        for an_i in range(self.numberOfAnomaly):
            anomalyModel = GNM(self.n_anomaly, self.m_anomaly, set(range(self.n_anomaly * an_i, self.n_anomaly * (an_i+1))), self.logger)
            anomalyModel.run()
            self.G_anomaly += anomalyModel.graph


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

        # check realisation is at least possible:
        anomaly_degrees = sorted([deg for _, deg in self.G_anomaly.degrees.items()])
        anomaly_degrees.reverse()
        mask = []
        antimask = []
        for deg in anomaly_degrees:
            mask = [idx for idx in range(len(self.global_degree_list[:,1])) if idx not in antimask]
            max_idx = np.argmax(self.global_degree_list[mask,1])
            if self.global_degree_list[mask[max_idx],1] < deg:
                raise ValueError('cannot fit anomaly in global graph')
            else:
                antimask.append(mask[max_idx])

        #global_degrees = sorted(self.global_degree_list
        #self.normal_degree_list[an_node, 0]
        #ipdb.set_trace()

        has_duplicate_node = True
        node_selection = []

        # get anomalies degree list
        #an_degree_list = []
        #an_degree_pos = [] # index of first node of each graph in degree list
        #for an_i in range(self.numberOfAnomaly):
        #    an_degree_pos.append(len(an_degree_list))
        #    an_degree_list += self.G_anomalies[an_i].graph.degrees.items()
        #else:
        #    # convert to np.array to get nice numpy index seach
        #    an_degree_list = np.array(an_degree_list) 

        # plug all anomalies nodes to normal graph degree series
        # start from scratch if two anomaly nodes end up on the same normal
        # graph node
        while (has_duplicate_node):
            #for (index, (an_node, degree)) in enumerate(an_degree_list):
            #for index, (an_node, degree) in enumerate(an_degree_list):
            for an_node, an_degree in self.G_anomaly.degrees.items():

                ## get nodes with degree high enough to place anomaly node
                #an = np.where(
                #        np.array(an_degree_pos) <= index)[0][-1]

                # pick one at random
                candidate_indices = np.where(self.global_degree_list[:,1] >=  an_degree)
                candidate_node = np.random.choice(candidate_indices[0])

                # if node has already been chosen, pick all nodes again
                #if (candidate_node, self.global_degree_list[candidate_node, 0]) in node_selection:
                #    node_selection = []
                #    has_duplicate_node = True
                #    break
                if candidate_node in node_selection:
                    node_selection = []
                    has_duplicate_node = True
                    break
                #node_selection.append((candidate_node, self.degree_list[candidate_node,0]))
                node_selection.append(candidate_node)
                #node_selection.append((candidate_node, self.degree_list[candidate_node,1]))
            else:
                # when nodes are picked, substract anomaly degrees 
                has_duplicate_node = False
                # an_node are numbered from 0 to numberOfAnomaly * m_anomaly 
                for (an_node, an_degree), node_idx in zip(self.G_anomaly.degrees.items(), node_selection):
                    self.normal_degree_list[an_node, 0] = an_node
                    self.normal_degree_list[an_node, 1] = self.global_degree_list[node_idx, 1] - an_degree
                last_index = len(self.G_anomaly.degrees.items())
                mask = [index for index in range(self.global_degree_list.shape[0]) if index not in node_selection]
                # assign rest of degrees
                for _norm_node, degree in enumerate(self.global_degree_list[mask,1]):
                    norm_node = last_index + _norm_node
                    self.normal_degree_list[norm_node, 0] = norm_node
                    self.normal_degree_list[norm_node, 1] = degree


                #for (index, 
                #  ((an_node, an_degree), 
                #  (node_idx, norm_node))) in enumerate(zip(an_degree_list, node_selection)):
                #    # set normal nodes with the same name as anomalies
                #    self.global_degree_list[node_idx,0] = an_node
                #    self.global_degree_list[node_idx,1] = self.degree_list[node_idx, 1] - an_degree
                #    #
                #    #ipdb.set_trace()
                #    # get index of graph in list of anomaly graphs

                #    #an_index = np.where(np.array(an_degree_pos) <= index)[0][-1] 

                #    # set mapping from global to networkit node index
                #    #self.norm2an[norm_node] = (an_index, an_node)
                #    #self.an2norm[(an_index, an_node)] = norm_node

                #    # substrace degrees
                #    #self.global_degree_list[node_idx,1] -= an_degree # substract degree from anomaly
                #for (index

    def _generate_normality(self):
        """ Generate 'normal' graph using Havel-Hakimi algorithm + edge switching"""
        #self.G_normal = HavelHakimi(self.global_degree_list[:,1], self.N_swap, self.logger)
        self.logger.info('initiate generator') 
        normalModel = HavelHakimi(self.normal_degree_list[self.normal_degree_list[:,1]>0,:], self.N_swap, self.logger)
        #_generator = HavelHakimiGenerator(self.global_degree_list[:, 1], ignoreIfRealizable=False)

        #t0 = time.time()
        #_graph = _generator.generate()

        t0 = time.time() 
        #self.logger.info('{}s for nk hakimi'.format(t1 - t0))
        self.logger.info('run generator')
        normalModel.run()
        #self.G_normal._edge_swap() ## TODO migrate edge switch in "run" when edge switch flag
        self.logger.info('make swaps')
        normalModel._edge_swap()
        #self.G_normal.run()
        self.G_normal = normalModel.graph
        t1 = time.time()
        self.logger.info('{}s for python hakimi'.format(t1 - t0))

    def _check_multiple_edges(self):
        """ check if Global graph has multiple edges """
        # to detect multiple edges simply compare sets
        multiple_edges = list(self.G_normal.edge_set.intersection(self.G_anomaly.edge_set))

        ## keep track of multiple edges in case we can to switch them later
        #multiple_edges = []
        ##for (norm_n1, norm_n2) in self.G_normal.graph.iterEdges():
        ##ipdb.set_trace()
        #for  norm_edge in self.G_normal.graph.edges:
        #    (norm_n1, norm_n2) = norm_edge

        #    # if any of those nodes are not in G_anomaly,
        #    # current edge is not a multiple edge.
        #    if (norm_n1 in self.norm2an and norm_n2 in self.norm2an): 

        #        # get anomaly nodes names and check if they are in same graph
        #        an_index1, an_n1 = self.norm2an[norm_n1]
        #        an_index2, an_n2 = self.norm2an[norm_n2]

        #        an_edge = (an_n1, an_n2) if an_n1 < an_n2 else (an_n2, an_n1)

        #        # check if current edge already exists in anomaly
        #        if ((an_index1 == an_index2) and self.G_anomalies[an_index1].graph.hasEdge(an_edge)): 
        #            multiple_edges.append((norm_edge, (an_index1, an_edge)))
        #            if self.break_when_multiple:
        #                self.logger.warning('Multiedge in normal+anomaly graph,'
        #                  'and break_when_multiple set to true, stopping generation.')
        #                break

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
            #(norm_edge1, (an_index, an_edge1)) = multiple_edges[edge_index]
            multiple_edge = multiple_edges[edge_index]

            # choose at random which of the normal graph or anomaly to update
            p = random.uniform(0, 1)
            if p>=0.5: # TODO remonter proba comme paramètre 
                norm_edge2_idx = np.random.choice(len(self.G_normal.edges))
                norm_edge2 = self.G_normal.edges[norm_edge2_idx]
                reverse = random.uniform(0,1) >= 0.5

                # 1/2 chance of reversing _edge2 : equivalent to picking both
                # directions at random
                norm_edge1_idx = self.G_normal.edges.index(multiple_edge)
                (e1_n1, e1_n2) = multiple_edge

                if reverse >= 0.5:
                    (e2_n2, e2_n1) = norm_edge2 #self.edges[edge2_idx]
                else:
                    (e2_n1, e2_n2) = norm_edge2 #self.edges[edge2_idx]
                if (e1_n1 in norm_edge2 or e1_n2 in norm_edge2):
                    continue

                # get new edges
                new_edge1 = (e1_n1, e2_n2) if e1_n1 < e2_n2 else (e2_n2, e1_n1)
                new_edge2 = (e2_n1, e1_n2) if e2_n1 < e1_n2 else (e1_n2, e2_n1)

                # skip when edge exist 
                if (new_edge1 in self.G_normal.edge_set or new_edge2 in self.G_normal.edge_set
                        or new_edge1 in self.G_anomaly.edge_set or new_edge2 in self.G_anomaly.edge_set):
                    continue
                else:
                    # replace previous edges in set
                    self.G_normal.edge_set.remove(self.G_normal.edges[norm_edge1_idx])
                    self.G_normal.edge_set.remove(self.G_normal.edges[norm_edge2_idx])
                    self.G_normal.edge_set.add(new_edge1)
                    self.G_normal.edge_set.add(new_edge2)
                
                    self.G_normal.edges[norm_edge1_idx] = new_edge1
                    self.G_normal.edges[norm_edge2_idx] = new_edge2

                    #multiple_edges = self._check_multiple_edges()
                    multiple_edges.remove(multiple_edge)
            else:
                an_edge2_idx = np.random.choice(len(self.G_anomaly.edges))
                an_edge2 = self.G_anomaly.edges[an_edge2_idx]
                reverse = random.uniform(0,1) >= 0.5

                # 1/2 chance of reversing _edge2 : equivalent to picking both
                # directions at random
                an_edge1_idx = self.G_anomaly.edges.index(multiple_edge)
                (e1_n1, e1_n2) = multiple_edge

                if reverse >= 0.5:
                    (e2_n2, e2_n1) = an_edge2 #self.edges[edge2_idx]
                else:
                    (e2_n1, e2_n2) = an_edge2 #self.edges[edge2_idx]
                if (e1_n1 in an_edge2 or e1_n2 in an_edge2):
                    continue

                # get new edges
                new_edge1 = (e1_n1, e2_n2) if e1_n1 < e2_n2 else (e2_n2, e1_n1)
                new_edge2 = (e2_n1, e1_n2) if e2_n1 < e1_n2 else (e1_n2, e2_n1)

                # skip when edge exist 
                if (new_edge1 in self.G_normal.edge_set or new_edge2 in self.G_normal.edge_set
                        or new_edge1 in self.G_anomaly.edge_set or new_edge2 in self.G_anomaly.edge_set):
                    continue
                else:
                    # replace previous edges in set
                    self.G_anomaly.edge_set.remove(self.G_anomaly.edges[an_edge1_idx])
                    self.G_anomaly.edge_set.remove(self.G_anomaly.edges[an_edge2_idx])
                    self.G_anomaly.edge_set.add(new_edge1)
                    self.G_anomaly.edge_set.add(new_edge2)
                
                    self.G_anomaly.edges[an_edge1_idx] = new_edge1
                    self.G_anomaly.edges[an_edge2_idx] = new_edge2

                    #multiple_edges = self._check_multiple_edges()
                    multiple_edges.remove(multiple_edge)

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
        if len(multiple_edges) > 0:
            self.swap_multiedges(multiple_edges)
        # when no multi edges, concatenate graphs  and write it
        global_graph = self.G_anomaly + self.G_normal
        global_graph.weight = self.weight
        global_graph.shuffle_weights()
        global_graph.write_graph()
        ## write graph with shuffled weights 
       
