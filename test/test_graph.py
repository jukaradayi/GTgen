from GTgen.graph import *

def test_add():
    G = Graph(edges = [], nodes=set(),
              is_sorted= False, degrees=None,
              logger = None)

    # add properly formated edge
    G.addEdge((1,2))
    assert (1,2) in G.edges
    assert (1,2) in G.edge_set

    # add reversed edge
    G.addEdge((3,1))
    assert (1,3) in G.edges
    assert (1,3) in G.edge_set
    assert not (3,1) in G.edges
    assert not (3,1) in G.edge_set

    # add self loop
    try:
        G.addEdge((6,6))
    except AssertionError as err:
        assert "self loop" in str(err)

    # add multiEdge
    try:
        G.addEdge((1,2))
    except AssertionError as err:
        assert "multi edge" in str(err)

def test_numbers():
    G = Graph(edges = [], nodes=set(),
              is_sorted= False, degrees=None,
              logger = None)
    assert G.numberOfNodes == 0
    assert G.numberOfEdges == 0

    G.addEdge((1,2))
    assert G.numberOfNodes == 2
    assert G.numberOfEdges == 1

    G.addEdge((3,2))
    assert G.numberOfNodes == 3
    assert G.numberOfEdges == 2


def test_swapEdge():
    G = Graph(edges = [(0,1), (1,2), (2,3)], nodes={0,1,2,3}, edge_set = None,
              is_sorted= False, degrees=None,
              logger = None)

    # valid swap
    G.swapEdge(0, 2, True)
    assert G.hasEdge((0,2))
    assert G.hasEdge((1,3))

    # swap back ...
    G.swapEdge(0, 2, True)
    
    # invalid swap
    try:
        G.swapEdge(0, 2, False)
    except AssertionError as err:
        assert "already exist" in str(err)
