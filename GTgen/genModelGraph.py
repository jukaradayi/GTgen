"""
    Generate weighted graph G_n with anomalies.
    The anomaly, noted as `G_an` is an irregularity in the graph that only occurs in the is an irregularity in the graph where the
    weighted edges involved are also involved in a link stream anomaly (i.e.
    occur at the time where a timeserie anomaly occurs).

    The weighted graph G_n is generated using first an erdos renyi,
    then adding "small" (TODO sure ?) erdos renyi as "graph anomalies" denser parts of the graph,
    and before adding the anomaly `G_an` erdos renyii as "link stream anomaly".
    The weights are initialised to 1 for all the created edges, and 
    if a multiple edge is created when creating the union of `G_n` and 
    `G_an`, the edges are merged into a simple edge and the weight is 
    increased by 1.
    Finally, we add the weights by picking edges randomly on the total graph
    `G_n + G_an` and increasing the weight of the edges by 1, until the
    sum of the weights reaches `nInteractions` given in input.
"""
import os
#import ipdb
import time
import numpy as np
from GTgen.graph import *
from GTgen.graphModels import *

class ModelGraph():
    """
        Attributes:
        -----------
        n_graphAnomaly: int,
            number of "graph anomaly"
        n_streamAnomaly: int,
            number of "link stream anomaly"
        nNodes: int,
            number of nodes of "complete" graph (normality + all anomalies)
        nNodes_graphAnomaly: int,
            number of nodes to be used in anomaly (TODO: maybe max so not all anomaly are the same?)
        nNodes_streamAnomaly: int,
            number of Nodes in stream anomaly
        nEdges_normality: int,
            number of edges to generatefor normal graph
        nEdges_graphAnomaly: int,
            number of edges in graph Anomaly
        nEdges_streamAnomaly: int,
            number of edges in stream anomaly
        nInteractions: int,
            number of interactions of normal graph + anomaly graph. 
            The weights will be set uniformly at random so that their sum is
            the number of interactions. should be higher than number of edges.
        nInteractions_streamAnomaly: int,
            same thing for stream anomaly graph.
        output: str,
            folder in which normal and anomaly graph are written
        seed: int,
            random seed for numpy (if not fixed already)
        logger: logger,
            a logger
    """

    def __init__(self,
            n_graphAnomaly,
            n_streamAnomaly,
            nNodes,
            nNodes_graphAnomaly,
            nNodes_streamAnomaly,
            nEdges_normality,
            nEdges_graphAnomaly,
            nEdges_streamAnomaly,
            nInteractions,
            nInteractions_streamAnomaly,
            output,
            seed=None,
            logger=None):
            
        self.n_graphAnomaly = n_graphAnomaly 
        
        self.n_streamAnomaly = n_streamAnomaly
        self.nNodes = nNodes

        # Number of nodes involved in the anomalies
        # Must be lower than total number of nodes
        assert nNodes_graphAnomaly < nNodes, ("graph anomaly should be "
                    "smaller than normal graph")
        assert nNodes_streamAnomaly < nNodes, ("stream anomaly should be "
                    "smaller than normal graph")
        self.nNodes_graphAnomaly = nNodes_graphAnomaly
        self.nNodes_streamAnomaly = nNodes_streamAnomaly

        # Number of edges of each graph 
        self.nEdges_normality = nEdges_normality
        self.nEdges_graphAnomaly = nEdges_graphAnomaly
        self.nEdges_streamAnomaly = nEdges_streamAnomaly

        # Define Number of interaction required
        assert nInteractions >= nEdges_normality + nEdges_graphAnomaly
        assert nInteractions_streamAnomaly >= nEdges_streamAnomaly
        self.nInteractions = nInteractions
        self.nInteractions_streamAnomaly = nInteractions_streamAnomaly

        # logger & output
        self.logger = logger
        self.output = output

        # instantiate Normal Graph
        self.G_normal = Graph(edges=[],
                nodes=set(),
                degrees=None, weight=np.empty((0,), dtype=np.int32),
                logger=self.logger,
                merge_multiedges=True)

    def generate_normality(self):
        """ Generate graph with Erdos-Renyi model """

        normality_model = GNM(self.nNodes,
                              self.nEdges_normality,
                              seed=None,
                              nodes=set(range(self.nNodes)), # todo check
                              logger=self.logger)
        normality_model.run()
        self.logger.debug(f'normal graph edges:\n'
            '{normality_model.graph.edges}')
        self.G_normal += normality_model.graph

    def generate_graphAnomaly(self):
        """ Generate graph-anomalies with Erdos-Renyi model 
            and add it to normal graph.
            When an edge of the graph-anomaly already exist in normal graph,
            they are fused as one simple edge, and its weight is increased by
            one.
        """
        for i in range(self.n_graphAnomaly):
            # choose nodes at random
            nodes = set(np.random.choice(range(self.nNodes), 
                                         self.nNodes_graphAnomaly,
                                         replace=False))
            anomaly_model = GNM(self.nNodes_graphAnomaly,
                              self.nEdges_graphAnomaly,
                              seed=None,
                              nodes=nodes,
                              logger=self.logger)

            anomaly_model.run()
            self.logger.debug(f'graph anomalies edges:\n'
                '{anomaly_model.graph.edges}')
            self.G_normal += anomaly_model.graph

    def generate_streamAnomaly(self):
        """ Generate Stream Anomaly with Erdos-Renyi Model.
            The stream anomaly shares its node with the normal graph, but
            is stored and written separately.
        """
        self.G_anomaly = Graph(edges=[],
                nodes=set(),
                degrees=None, weight=np.empty((0,), dtype=np.int32),
                logger=self.logger,
                merge_multiedges=True)

        for i in range(self.n_streamAnomaly):
            # choose nodes at random
            nodes = set(np.random.choice(range(self.nNodes), 
                                         self.nNodes_streamAnomaly,
                                         replace=False))

            anomaly_model = GNM(self.nNodes_streamAnomaly,
                              self.nEdges_streamAnomaly,
                              seed=None,
                              nodes=nodes, # todo check
                              logger=self.logger)
            anomaly_model.run()
            self.logger.debug(f'stream anomaly graph edges:\n '
                '{anomaly_model.graph.edges}')
            self.G_anomaly += anomaly_model.graph

    @staticmethod
    def set_weights(nInteractions, graph):
        """ Randomly choose edges to increment weight until nInteractions is
            reached
        """
        sum_weights = np.sum(graph.weight)
        while sum_weights < nInteractions:
            edge_idx = np.random.choice(len(graph.edges))
            graph.weight[edge_idx] += 1
            sum_weights += 1

    def run(self):
        self.logger.info('generating normal graph')
        self.generate_normality()
        self.logger.info('generating graph-anomaly')
        self.generate_graphAnomaly()
        self.logger.info('generating stream-anomaly')
        self.generate_streamAnomaly()
        self.logger.info('generate weights')
        self.set_weights(self.nInteractions, self.G_normal)

        self.set_weights(self.nInteractions_streamAnomaly, self.G_anomaly)
        self.logger.info('writing graphs')
        self.G_normal.write_graph(os.path.join(self.output,
                                              'normal_graph.txt'))
        self.G_anomaly.write_graph(os.path.join(self.output,
                                               'anomaly_graph.txt'))


