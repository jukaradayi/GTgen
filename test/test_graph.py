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

def test_edge_set():
    edge_list = [(0, 1), (0, 2), (0, 5), (0, 6), (0, 7), (1, 2), (1, 3), (1, 4), (2, 5), (3, 4), (6, 7)]
    G = Graph(edges = edge_list, nodes = {0,1,2,3,4,5,6,7}, edge_set = None, is_sorted = False, degrees = None, logger = None)
    assert sorted(list(G.edge_set)) == sorted(G.edges)   
    N_swap = 10 * len(edge_list)

    rdm_index1 = np.random.choice(len(G.edges), size=10*N_swap)
    rdm_index2 = np.random.choice(len(G.edges), size=10*N_swap)
    reverse_array = np.random.uniform(0,1,N_swap*10)
    n_swap = 0 
    while n_swap < N_swap:
        for edge1_idx, edge2_idx, reverse in zip(rdm_index1, rdm_index2, reverse_array):

            if edge1_idx == edge2_idx:
                continue

            (e1_n1, e1_n2)  = G.edges[edge1_idx]
            edge2 = G.edges[edge2_idx]
            if (e1_n1 in edge2 or e1_n2 in edge2):
                continue

            # 1/2 chance of reversing _edge2 : equivalent to picking both
            # directions at random
            if reverse_array[n_swap] >= 0.5:
                (e2_n2, e2_n1) = edge2 #self.edges[edge2_idx]
            else:
                (e2_n1, e2_n2) = edge2 #self.edges[edge2_idx]

            # get new edges
            new_edge1 = (e1_n1, e2_n2) if e1_n1 < e2_n2 else (e2_n2, e1_n1)
            new_edge2 = (e2_n1, e1_n2) if e2_n1 < e1_n2 else (e1_n2, e2_n1)

            # skip when edge exist 
            if new_edge1 in G.edge_set or new_edge2 in G.edge_set:
                continue
            else:

                # replace previous edges in set
                G.edge_set.remove(G.edges[edge1_idx])
                G.edge_set.remove(G.edges[edge2_idx])
                G.edge_set.add(new_edge1)
                G.edge_set.add(new_edge2)
        
                G.edges[edge1_idx] = new_edge1
                G.edges[edge2_idx] = new_edge2
                n_swap += 1

            if n_swap >= N_swap :
                break
        else:
            #when run out of indexes pick some again
            reverse_array = np.random.uniform(0,1,N_swap*10)
            #self.graph.shuffle_edges()
            rdm_index1 = np.random.choice(len(G.edges), size=10*N_swap)
            rdm_index2 = np.random.choice(len(G.edges), size=10*N_swap)
    # check that after all the swaps the list and set are still the same
    assert sorted(list(G.edge_set)) == sorted(G.edges)
