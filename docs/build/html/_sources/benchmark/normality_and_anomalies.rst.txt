.. _normality_and_anomalies:

Normality And Anomalies
=======================

* In the benchmark, we define what we call the "normality" state, and the
  "anomaly". When working with models, we call anomaly part of the model
  with overall different properties (e.g. subpart of a graph with different density than the complete graph).
  When working with real datasets, we assume that anomalies are already
  present in the data.

* The link-stream are generated using the package GENBIP (TODO link 
  to genbip) to generate bipartite graph where the top nodes are the edges
  of the graph, with their weights as top nodes degrees, and the bottom
  nodes are the timestamps with the value of the timeserie as bottom
  nodes degrees.

* Considering this, when using datasets, we assume anomalies are already
  in the dataset and generate link-stream using the measured weighted graph,
  and the measured timeserie as top and bottom nodes.
  When using models, we use models to generate graphs and timeseries
  (e.g. Erdos-Renyi for graph, or white noise for timeserie), and we 
  add anomalies on top of that.
  For this purpose, we then define two types of anomalies: 
    - graph/timeseries anomalies: an anomaly that is present either in 
      the graph or the timeserien but not in both. TODO plot images to clarify 
      TODO explain more .. ? 

    - stream-anomalies an anomaly that is present both in the graph and 
      the timeserie. This means that when generating the link-stream, 
      the edges involved in the graph anomaly are linked to the timestamps
      involved in the timeseries anomaly
      TODO plot
  

