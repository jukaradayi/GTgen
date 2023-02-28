.. _data_model:

Data Modeling
=============

* One pipeline used to generate graph uses weights sequences and degree sequences
  measured from real data (TODO "real data" sonne moche..?).
  We assume that anomalies are present in the "real data" we use.

Pipeline
--------

  The complete process is to:
    - take the degree sequence and weight sequence from real data

    - Generate the anomaly G_an using an Erdos Renyi model
    
    - Randomly choose, using the degree sequence D, the set of nodes S_n that will
      host the Erdos-Renyi generated anomaly (described by the input parameters).
      Substract to degree sequence D_n (sequence D on nodes S_n) the degree sequence
      D_an of the anomaly to get 'normality' degree sequence D_norm.

    - Using D_norm, generate a uniformly randomly picked simple graph G_n that fits this sequence.
      We use a Havel-Hakimi generator to get a graph that fits D_norm, then perform N_swap 
      edge swap to get a uniformly randomly picked simple graph that fits D_norm (typically N_swap = 10*number of edges)

    - Check that union of G_n and G_an is a simple graph

        - If it is, :
          -GREAT

        - If not
         - perform random swaps in G_n and G_an, with the edges involved in multiple edges and another
          randomly picked edge in either G_n or G_an (chosen by a coin toss)
         - Perform again N_swap (N_swap2.. ?) edge swaps in G_n to get uniformly randomly picked graph
           in the set of graphs with degree sequence D_norm (previous steps biased the graph).
           The edge swaps are chosen in a way such that it doesn't create multiple edges (bias.. ?)


 
