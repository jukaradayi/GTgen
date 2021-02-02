from GTgen.genModelGraph import *
from GTgen.graphModels import *
from GTgen.graph import * 

from collections import Counter
import pytest
from itertools import combinations
import logging
import numpy as np

def test_normalGraph(logger):
    model = ModelGraph(1,1,10,3,5,5,3,5, '', None, logger)

    model.generate_normality()

    assert len(model.G_normal.edges) == 5
    assert len(model.G_normal.nodes) == 10

def test_graphAnomaly(logger):
    model = ModelGraph(1,1,10,3,5,5,3,5, '', None, logger)

    model.generate_graphAnomaly()

    assert len(model.G_normal.edges) == 3
    assert len(model.G_normal.nodes) == 3

def test_streamAnomaly(logger):
    model = ModelGraph(1,1,10,3,5,5,3,5, '', None, logger)

    model.generate_streamAnomaly()

    assert len(model.G_anomaly.edges) == 5
    assert len(model.G_anomaly.nodes) == 5

def test_run(logger):
    model = ModelGraph(1,1,10,3,5,5,3,5, '', None, logger)

    model.run() 

    # graph anomaly might share edges so number of edge is at least
    # that of normal graph
    assert len(model.G_normal.edges) >= 5
    assert len(model.G_normal.nodes) == 10

    assert len(model.G_anomaly.edges) == 5
    assert len(model.G_anomaly.nodes) == 5
