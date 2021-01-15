from GTgen.graphModels import *
from collections import Counter
import pytest
from itertools import combinations
import logging

def test_HavelHakimi(taxi_seq, NKedges, logger):
    """ compare Havel Hakimi results with networkit """

    seq = []
    seq.append([5, 4, 3, 2, 2, 2, 2, 2])
    seq.append(taxi_seq)
    for index, degrees in enumerate(seq):
        # generate graph
        GTgenerator = HavelHakimi(degrees, 0, logger)
        GTgenerator.run()
        
        # sort edge list to compare... 
        GTedges = sorted(GTgenerator.graph.edges)

        # compare edge list
        assert NKedges[index] == GTedges, 'degree sequences {} are different, nk length {}, GT length {}'.format(index, len(NKedges[index]), len(GTedges))

def test_GNM(logger):
    # test n= 0 , 12, -1
    # test m = 0, m>n(n-1)/2, m<0 
    G = GNM(0, 0, logger)
    with pytest.raises(RuntimeError) as err:
        G.run()
    assert 'empty graph' in str(err.value)

    G = GNM(10, 0, logger)
    G.run()
    assert G.graph.numberOfEdges == 0
    assert G.graph.numberOfNodes == 10

    # clique
    G = GNM(10, 1000, logger)
    G.run()
    assert G.graph.numberOfEdges == 45
    assert G.graph.numberOfNodes == 10
    for (u,v) in combinations(G.graph.nodes, 2):
        edge = (u,v) if u>v else (v,u)
        assert G.graph.hasEdge(edge), "clique does not contain edge ({}, {})".format(u, v)

    G = GNM(10, 20, logger)
    G.run()
    print(G.graph.edges)
    print(Counter(G.graph.edges))
    assert G.graph.numberOfNodes == 10
    assert G.graph.numberOfEdges == 20
    assert len(G.graph.edges) == len(G.graph.edge_set)

#def test_swaps():
#    # esquisse : génerer un havel hakimi, faire un swap et vérifier
