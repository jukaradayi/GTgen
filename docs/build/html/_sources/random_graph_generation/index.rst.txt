.. _random_graph_generation:

Random Graph Generation
=======================

* There are two pipelines to generate random graphs implemented. 

* TODO TROUVER NOM POUR DIFFERENCIER REAL DATA ET RANDOM COMPLET

* The first one takes input from real data (weights and node degree sequence),
  and generates an uniformly randomly picked graph with anomalies that keeps
  the weights and node degree sequence from the input.

* The second pipeline generates graphs with anomaly from scratch, using 
  Erdos-Renyi models for the 'normal graph' and the 'anomalies'.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   normality_and_anomalies
   data_model
   random_model
