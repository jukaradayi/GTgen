.. _usage:

Usage
=====

- Once the package is installed, to run a generation, you need to configure your model using a yaml file. Examples can be found in the *example/* folder in the root of the package.

- To generate using a yaml file: 

  * **python GTgen/generate.py -y <PATH_TO_YAML> -o <OUTPUT_FOLDER>**

Yaml configuration
------------------

- Below you will find the details of the yaml configuration file.

.. code-block::

    Graph:
        generate_data: True # generate a graph using a real dataset, 
        data_params:
            degree: "<PATH_TO_DEGREE_SEQUENCE>"
            weight: "<PATH_TO_WEIGHT_DISTRIBUTION>"
            numberOfAnomalies: 1 # number of anomalies to add to the graph.
            n_anomaly: 50 # number of nodes of the added anomaly
            m_anomaly: 1000 # number of edges of the added anomaly
            N_swap1: 10 # N_swap1 * n_edges swaps will be performed on the graph without anomalies, where n_edges is the number of edges of the graph without anomalies
            N_swap2: 10 # N_swap2 * n_edges_full swaps will be performed on the graph with anomalies, where n_edges_full is the number of edges of the graph with anomalies

        generate_model: False # generate a graph using an Erdos-Renyii model
        model_params:
            n_graphAnomaly: 1 # number of smaller, more denser Erdos-Renyii "anomalies" to add to the big Erdos-Renyii graph. These anomalies won't be used to generate the "link-stream" anomaly
            n_streamAnomaly: 1 # number of anomalies that will be used to generate the "link-stream" anomaly
            nNodes: 15 # number of node of the global graph
            nNodes_graphAnomaly: 5 # number of node of the graph anomaly
            nNodes_streamAnomaly: 5 # number of node of the stream anomaly
            nEdges_normality: 30 # number of edges of the normal graph
            nEdges_graphAnomaly: 7 # number of edges of the graph anomaly
            nEdges_streamAnomaly: 10 # number of edges of the stream anomaly
            nInteractions: 87 # sum of the weights of the normal graph. Should be equal to "cum_sum" value of Timeserie - model_param - cum_sum
            nInteractions_streamAnomaly: 25 # sum of the weights of the stream anomaly - should be equal to TimeSerie - model_param - cum_sum_streamAnomaly

    TimeSerie:
        generate_data: True # generate a timeserie by shuffling a real timeserie
        data_params:
            dataset: "<PATH_TO_TS>"
            anomaly_type: "regimeShift" # type of anomaly added - can either be "peak" for a "small" section with higher values, or "regimeShift" for a large section with different values
            anomaly_length: 0.1 # percentage of the total duration used for the anomaly - should be a value between 0 and 1
        generate_model: False # generate a timeserie by using a white noise model
        model_params:
            duration: 28 # number of timestamps to be generated for the "normal" timeserie
            duration_streamAnomaly: 10  # number of timestamps for the stream "anomaly" 
            duration_tsAnomaly: 10 # number of timestamps for the "timeserie" anomaly. Those won't be used to generate the stream anomaly.
            cum_sum: 87 # sum of the values of the timeseries. should be equal to "nInteractions" vlaue of Graph - model_params - nInteractions
            cum_sum_streamAnomaly: 25 # sum of the values of the stream_anomaly - should be equal to graph - model_param - nInteractions_streamAnomaly


Implementation details
----------------------

- When generating a graph (resp. timeserie) using real data, the anomalies are added by "reserving" some of the nodes and weights of the graph (resp. some timestamps and values of the timeseries). 

  - We then generate a "normal" graph (resp. timeserie) by generating using havel hakimi a graph with the resulting degree sequence, then swapping the edges to get a "uniformly" randomly picked graph with this degree distribution (resp. shuffling the values of the timeserie).

  - For example, if the input degree sequence of the graph is [5,4,2,1] (which means 5 nodes of degree 1, 4 nodes of degree 2, 2 nodes of degree 3 and 1 node of degree 4), if we want 1 edge to be abnormal, we can reserve the following degree sequence for the anomaly [1, 0, 0, 0] and generate the "normal" graph using the resulting degree sequence [4,4,2,1].

- When generating a graph (resp. timeserie) using a model, the anomalies are added by "plugging" generated anomalies to the graph (resp. timeserie).


