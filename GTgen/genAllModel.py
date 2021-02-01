"""
    Generate weighted graph with "graph anomaly" and "link stream anomaly".
    A "graph anomaly" is an irregularity in the graph that only occurs in the 
    graph, a "link stream anomaly" is an inrregularity in the graph where the
    weighted edges involved are also involved in a link stream anomaly (i.e.
    occur at the time where a timeserie anomaly occurs).

    The weighted graph G_n is generated using first an erdos renyii
    and adding weights (TODO random +1 distributed ?), then adding 
    "small" (TODO to be defined) G_gan erdos renyii as "graph anomaly",
    and finally adding "small" (TODO to be defined) G_lsan erdos renyii as 
    "link stream anomaly".
"""

import time
import numpy as np
from graph import *
from graphModels import *
class ErdosRenyiWithAnomaly():
    """
        Attributes:
        -----------
        n_graphAnomaly: int,
            number of "graph anomaly"
        n_lsAnomaly: int,
            number of "link stream anomaly"
        nNodes_normality: int,
            number of nodes to be used in normality
        nNodes_graphAnomaly: int,
            number of nodes to be used in anomaly (TODO: maybe max so not all anomaly are the same?)

        nNodes_lsAnomaly,
        nEdges_normality: int,
            number of edges to generatefor normal graph
        nEdges_graphAnomaly,
        nEdges_lsAnomaly

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
            seed=None,
            logger=None):
            
        self.n_graphAnomaly = n_graphAnomaly 
        
        self.n_streamAnomaly = n_streamAnomaly
        self.nNodes = nNodes

        # Number of nodes involved in the anomalies
        # Must be lower than total number of nodes
        # TODO assert 
        self.nNodes_graphAnomaly = nNodes_graphAnomaly
        self.nNodes_streamAnomaly = nNodes_streamAnomaly
       
        self.nEdges_normality = nEdges_normality
        self.nEdges_graphAnomaly = nEdges_graphAnomaly

        self.nEdges_streamAnomaly = nEdges_streamAnomaly
        self.logger=logger

    def generate_normality(self):
        self.G_normality = Graph(edges=[],
                nodes=set(),
                degrees=None, weight=np.empty((0,)),
                logger=self.logger,
                merge_multiedges=True)

        normality_model = GNM(self.nNodes,
                              self.nEdges_normality,
                              seed=None,
                              nodes=set(range(self.nNodes)), # todo check
                              logger=self.logger)
        normality_model.run()
        self.logger.debug(f'normal graph edges:\n {normality_model.graph.edges}')

        self.G_normality += normality_model.graph

    def generate_graphAnomaly(self):
        """ graph anomaly is add to normal graph """
        #anomaly_model = GNM(self.nNodes_graphAnomaly,
        #                      self.nEdges_graphAnomaly,
        #                      seed=None,
        #                      nodes=None, # todo check
        #                      logger=self.logger)
        for i in range(self.n_graphAnomaly):
            # choose nodes at random
            nodes = set(np.random.choice(range(self.nNodes), self.nNodes_graphAnomaly))

            #nodes = set(range(nNodes_normality+i*nNodes_graphAnomaly,
            #                 nNodes_normality+(i+1)*nNodes_graphAnomaly))
            anomaly_model = GNM(self.nNodes_graphAnomaly,
                              self.nEdges_graphAnomaly,
                              seed=None,
                              nodes=nodes, # todo set good nodes
                              logger=self.logger)

            anomaly_model.run()
            self.logger.debug(f'graph anomalies edges:\n {anomaly_model.graph.edges}')
            self.G_normality += anomaly_model.graph

    def generate_streamAnomaly(self):
        """ stream anomaly is stored seperately from normal graph """
        # choose nodes at random
        nodes = set(np.random.choice(range(self.nNodes), self.nNodes_streamAnomaly))

        #nodes=set(range(nNodes_normality + n_graphAnomaly * nNodes_graphAnomaly,
        #           nNodes_normality + n_graphAnomaly * nNodes_graphAnomaly + nNodes_streamAnomaly))
        self.G_anomaly = Graph(edges=[],
                nodes=nodes,
                degrees=None, weight=np.empty((0,)),
                logger=self.logger,
                merge_multiedges=True)

        anomaly_model = GNM(self.nNodes_streamAnomaly,
                              self.nEdges_streamAnomaly,
                              seed=None,
                              nodes=set(range(self.nNodes)), # todo check
                              logger=self.logger)
        anomaly_model.run()
        self.logger.debug(f'stream anomaly graph edges:\n {anomaly_model.graph.edges}')
        self.G_anomaly += anomaly_model.graph

    def run(self):
        ipdb.set_trace()
        self.generate_normality()
        self.generate_graphAnomaly()
        self.generate_streamAnomaly()


