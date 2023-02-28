.. _random_model:

Random Model
============

* The other implemented pipeline used to generate random graphs with anomalies only
  requires a number of node and number of egdes as input. The generated weighted
  graphs are created using Erdos Renyi models, with anomalies generated with
  Erdos Renyi.

Pipeline
--------

* The process is as follows:
  - Choose input parameters: 
    - N_nodes = the number of nodes of the generated graph

    - N_nodes_graphAnomaly = the number of nodes of the graph anomaly G_gan

    - N_nodes_streamAnomaly = the number of nodes of the stream anomaly G_san (TODO: nécessaire de séparer graph/stream anomaly ?)

    - N_edges_normality = the number of edges of the normality graph G_norm

    - N_edges_graphAnomaly = number edges G_gan

    - N_edges streamAnomaly = number edges G_san

 - Generate independently G_norm, G_gan and G_san with Erdos-Renyi model with the specified parameters

 - Randomly choose nodes of G_norm that will host G_gan and G_san

 - If an edge of G_gan or G_san is already in G_norm, count it as simple link and increment it's weight by one

 - (TODO definir ?) randomly increment weights .. ? 
