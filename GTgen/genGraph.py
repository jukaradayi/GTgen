import time
import random
import numpy as np

from collections import defaultdict
from GTgen.graphModels import *

class GraphWithAnomaly():
    """
        Using degrees sequence S_d and weights sequence S_w from real dataset,
        generate numberOfAnomaly 'anomaly' graph G_an_i, with GNM model 
        with n_anomaly nodes and m_anomaly edges.
        Then, create 'normal' graph G_n such that 
        seq(G_n) + sum(seq(G_an_i) for i in 0..numberOfAnomaly) = S_d

        Attributes:
        -----------
        n_anomaly: int,
            number of nodes in anomaly
        m_anomaly: int,
            number of edges in anomaly
        numberOfAnomaly: int,
            number of anomaly graph to generate. Each anomaly graph
            is generated with n_anomly nodes and m_anomaly edges.
        N_swap: int,
            The 'normal Graph' G_n will be generated with 
            N_swap * N_edges edge swap, where N_edges is its number of edges
        weight: list of tuples,
            weight distribution, in the format [(val, num)] where val is the 
            weight value, and num is the number of edges having that weight.
        output: string, 
            path to the desired output file

    """
    def __init__(self, degree_list,
            numberOfAnomaly,
            n_anomaly,
            m_anomaly, 
            N_swap,
            weight,
            logger,
            output,
            seed=None):

        self.logger = logger

        if seed is not None:
            self.seed = seed
            np.random.seed(seed)

        # objective degree sequence
        self.global_degree_list = np.array(degree_list)
        self.normal_degree_list = np.empty(
                (self.global_degree_list.shape[0], 2))

        # anomaly parameters to generate Erdos Renyi
        self.numberOfAnomaly = numberOfAnomaly # number of anomalies
        self.n_anomaly = n_anomaly
        self.m_anomaly = m_anomaly # pick erdos renyi for anomaly
        self.anomaly_seq = dict()
        self.G_anomalies = [] # graph

        # normal graph
        self.G_normal = None
        N_edges = int(sum([deg for _,deg in degree_list])/2)
        self.N_swap = int(N_swap * N_edges)

        # output
        self.output = output

        # weight vector
        self.weight = np.empty((N_edges,), dtype=np.int32)
        prev_idx = 0
        for val, num in weight:
            self.weight[prev_idx:prev_idx+num] =  val
            prev_idx += num

    def _generate_anomaly(self):
        """ Generate numberOfAnomaly anomalies using GNM model.
            The anomalies are concatenated in the same graph object, and
            the node sequence is increasing
            
        """
        # instantiate anomaly graph
        self.G_anomaly = Graph(edges=[],
                nodes=set(),
                degrees=None, logger=self.logger)

        # generate each anomaly using GNM model, nodes are integers in
        # increasing order
        for an_i in range(self.numberOfAnomaly):
            anomalyModel = GNM(self.n_anomaly, self.m_anomaly,
                    set(range(self.n_anomaly*an_i, self.n_anomaly*(an_i+1))),
                    self.logger)
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
            -Pick n in (1...j) uniformly at random, attribute node _km of 
             anomaly to node _km of global graph, and mark node _km of global
             graph as already selected.
            -When picking a node that has already been selected, start from
             scratch.
        """

        # check realisation is at least possible:
        anomaly_degrees = sorted(
                [deg for _, deg in self.G_anomaly.degrees.items()])
        anomaly_degrees.reverse()
        mask = []
        antimask = []

        for deg in anomaly_degrees:
            mask = [idx for idx in range(
                len(self.global_degree_list[:,1])) if idx not in antimask]
            max_idx = np.argmax(self.global_degree_list[mask,1])
            if self.global_degree_list[mask[max_idx],1] < deg:
                raise ValueError('cannot fit anomaly in global graph')
            else:
                antimask.append(mask[max_idx])

        has_duplicate_node = True
        node_selection = []

        # place anomaly in global graph
        while (has_duplicate_node):
            for an_node, an_degree in self.G_anomaly.degrees.items():

                # Get available nodes and pick one at random
                candidate_indices = np.where(
                        self.global_degree_list[:,1] >=  an_degree)
                candidate_node = np.random.choice(candidate_indices[0])

                # if node already selected, start from scratch
                if candidate_node in node_selection:
                    node_selection = []
                    has_duplicate_node = True
                    break

                node_selection.append(candidate_node)
            else:
                # when nodes are picked, substract anomaly degrees 
                has_duplicate_node = False

                # an_node are numbered from 0 to numberOfAnomaly * m_anomaly 
                # give same node names in normal graph and anomaly graph
                for (an_node, an_degree), node_idx in zip(
                        self.G_anomaly.degrees.items(), node_selection):
                    self.normal_degree_list[an_node, 0] = an_node
                    self.normal_degree_list[an_node, 1] = self.global_degree_list[node_idx, 1] - an_degree

                last_index = len(self.G_anomaly.degrees.items())
                mask = [index for index in range(
                           self.global_degree_list.shape[0]) 
                          if index not in node_selection]

                # assign node names to rest of degrees
                for _norm_node, degree in enumerate(
                        self.global_degree_list[mask,1]):
                    norm_node = last_index + _norm_node
                    self.normal_degree_list[norm_node, 0] = norm_node
                    self.normal_degree_list[norm_node, 1] = degree

    def _generate_normality(self):
        """ Generate 'normal' graph using Havel-Hakimi algorithm
            and edge swaps
        """
        self.logger.info('initiate Havel Hakimi generator') 
        normalModel = HavelHakimi(
                self.normal_degree_list[self.normal_degree_list[:,1]>0,:],
                self.N_swap, self.logger)

        t0 = time.time() 
        normalModel.run()

        # perform edge swap
        normalModel._edge_swap()
        self.G_normal = normalModel.graph
        t1 = time.time()

        # if verbose, monitor time
        self.logger.debug('{}s for python hakimi'.format(t1 - t0))

    def _check_multiple_edges(self):
        """ check if Global graph has multiple edges """
        # to detect multiple edges simply compare edge sets
        multiple_edges = list(
                self.G_normal.edge_set.intersection(self.G_anomaly.edge_set))

        if len(multiple_edges) > 0:
            self.logger.info(
              '{} Multiple edges in normal+anomaly graph detected.'.format(
                  len(multiple_edges)))
        return multiple_edges

    def swap_multiedges(self, multiple_edges):
        """ If multiple edges are detected between normal graph and anomaly, 
            target them specifically when switching
        """
        self.logger.info('Swapping {} edges by targetting '
                'multiple edges specifically'.format(len(multiple_edges)))
        while (len(multiple_edges) > 0):
            # pick an edge at random ## TODO not necessary ?
            edge_index = np.random.randint(low=0, high=len(multiple_edges))
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

                # skip if new edges already exist 
                if (new_edge1 in self.G_normal.edge_set
                 or new_edge2 in self.G_normal.edge_set
                 or new_edge1 in self.G_anomaly.edge_set 
                 or new_edge2 in self.G_anomaly.edge_set):
                    continue
                else:
                    # replace previous edges in set
                    self.G_normal.edge_set.remove(
                            self.G_normal.edges[norm_edge1_idx])
                    self.G_normal.edge_set.remove(
                            self.G_normal.edges[norm_edge2_idx])
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
                if (new_edge1 in self.G_normal.edge_set
                 or new_edge2 in self.G_normal.edge_set
                 or new_edge1 in self.G_anomaly.edge_set
                 or new_edge2 in self.G_anomaly.edge_set):
                    continue
                else:
                    # replace previous edges in set
                    self.G_anomaly.edge_set.remove(
                            self.G_anomaly.edges[an_edge1_idx])
                    self.G_anomaly.edge_set.remove(
                            self.G_anomaly.edges[an_edge2_idx])
                    self.G_anomaly.edge_set.add(new_edge1)
                    self.G_anomaly.edge_set.add(new_edge2)
                
                    self.G_anomaly.edges[an_edge1_idx] = new_edge1
                    self.G_anomaly.edges[an_edge2_idx] = new_edge2

                    #multiple_edges = self._check_multiple_edges()
                    multiple_edges.remove(multiple_edge)

                self.logger.debug('{} multiple edges left'.format(
                    len(multiple_edges)))
            self.logger.info('{} edges left'.format(len(multiple_edges)))

    def run(self):
        self.logger.info('generating anomaly')
        self._generate_anomaly()
        self.logger.info('getting normal graph degree sequence')
        self._get_normality_degree_seq()
        self.logger.info('generating normal graph')
        self._generate_normality()
        self.logger.info('checking for multiple edges')
        multiple_edges = self._check_multiple_edges()
        if len(multiple_edges) > 0:
            self.swap_multiedges(multiple_edges)

        # when no multi edges, concatenate graphs
        global_graph = self.G_anomaly + self.G_normal

        # add weights
        global_graph.weight = self.weight
        global_graph.shuffle_weights()

        # write graph
        global_graph.write_graph(self.output)
       
