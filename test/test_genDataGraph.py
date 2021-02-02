from GTgen.genDataGraph import *
from GTgen.graphModels import *
from GTgen.graph import * 

from collections import Counter
import pytest
from itertools import combinations
import logging
import numpy as np

def test_multiple_edges(logger):
    G = DataGraph([(0,2),(1,4), (2,3), (3,2),(4,2), (5,3)], 1, 4, 3, 0, 0, [(1,2), (1,3)], logger, './', 'test')

    # GNM anomaly
    #G._generate_anomaly()
    #G.G_anomalies = []
    #G_anomaly = Graph(edges=[],
    #            nodes=set(),
    #            degrees=None, logger=self.logger)


    #anomalyModel = GNM(G.n_anomaly, G.m_anomaly, logger)
    G.G_anomaly = Graph(edges=[(0,1), (1,2), (2,3)],
            nodes={0,1,2,3}, degrees=None, logger=logger)

    # check normality degree sequence
    G.get_normality_degree_seq()
    normality_degrees = [deg for _, deg in G.global_degree_list]

    # assign anomaly nodes to normal nodes
    #G.an2norm = {(0,0) : 0, (0,1): 1, (0,2):2, (0,3): 3}
    #G.norm2an = {0: (0,0), 1:(0,1), 2: (0,2), 3:(0,3)}
    G.global_degree_list = np.array([(0,1),(1,2), (2,1), (3,1),(4,2), (5,3)])
    
    # generate normality
    #G.G_normal = AbstractGraphGenerator()
    G.G_normal = Graph(edges=[(0,1), (1,4), (2,5), (3,5), (4,5)], nodes={0,1,2,3,4,5},
            degrees=None, logger=logger)

    multiple_edges = G._check_multiple_edges()
    assert multiple_edges == [(0,1)]
    #assert sorted(normality_degrees) == [0, 0, 1, 2, 2, 4]

    G.swap_multiedges(multiple_edges)
    #G.G_normal = AbstractGraphGenerator()


    #G_G_normal.graph = Graph(edges=[
