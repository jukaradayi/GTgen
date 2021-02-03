"""
    Generate weighted graph G_n with "graph anomaly" G_gan and "link stream anomaly" G_san.
    A "graph anomaly" is an irregularity in the graph that only occurs in the 
    graph, a "link stream anomaly" is an irregularity in the graph where the
    weighted edges involved are also involved in a link stream anomaly (i.e.
    occur at the time where a timeserie anomaly occurs).

    The weighted graph G_n is generated using first an erdos renyii
    and adding weights (TODO random +1 distributed ?), then adding 
    "small" (TODO to be defined) G_gan erdos renyii as "graph anomaly",
    and finally adding "small" (TODO to be defined) G_lsan erdos renyii as 
    "link stream anomaly".
"""
import os
import ipdb
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
       
        self.nEdges_normality = nEdges_normality
        self.nEdges_graphAnomaly = nEdges_graphAnomaly

        self.nEdges_streamAnomaly = nEdges_streamAnomaly
        self.logger = logger
        self.output = output

        # instantiate Normal Graph
        self.G_normal = Graph(edges=[],
                nodes=set(),
                degrees=None, weight=np.empty((0,)),
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
        self.logger.debug(f'normal graph edges:\n {normality_model.graph.edges}')

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
            nodes = set(np.random.choice(range(self.nNodes), self.nNodes_graphAnomaly, replace=False))
            anomaly_model = GNM(self.nNodes_graphAnomaly,
                              self.nEdges_graphAnomaly,
                              seed=None,
                              nodes=nodes,
                              logger=self.logger)

            anomaly_model.run()
            self.logger.debug(f'graph anomalies edges:\n {anomaly_model.graph.edges}')
            self.G_normal += anomaly_model.graph

    def generate_streamAnomaly(self):
        """ Generate Stream Anomaly with Erdos-Renyi Model.
            The stream anomaly shares its node with the normal graph, but
            is stored and written separately.
        """
        self.G_anomaly = Graph(edges=[],
                nodes=set(),
                degrees=None, weight=np.empty((0,)),
                logger=self.logger,
                merge_multiedges=True)

        for i in range(self.n_streamAnomaly):
            # choose nodes at random
            nodes = set(np.random.choice(range(self.nNodes), self.nNodes_streamAnomaly, replace=False))

            anomaly_model = GNM(self.nNodes_streamAnomaly,
                              self.nEdges_streamAnomaly,
                              seed=None,
                              nodes=nodes, # todo check
                              logger=self.logger)
            anomaly_model.run()
            self.logger.debug(f'stream anomaly graph edges:\n {anomaly_model.graph.edges}')
            self.G_anomaly += anomaly_model.graph

    def run(self):
        self.logger.info('generating normal graph')
        self.generate_normality()
        self.logger.info('generating graph-anomaly')
        self.generate_graphAnomaly()
        self.logger.info('generating stream-anomaly')
        self.generate_streamAnomaly()
        self.logger.info('writing graphs')
        self.G_normal.write_graph(os.path.join(self.output, 'normal_graph.txt'))
        self.G_anomaly.write_graph(os.path.join(self.output, 'anomaly_graph.txt'))



