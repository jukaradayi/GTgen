.. _pipeline:

Pipeline
========

* The complete benchmark pipeline is :

  - prepare the raw data to put it in (t,u,v) format and 
    get the graph weights and timeseries, and degree sequence

  - generate link streams using these data

  - use degree sequences, weights and timeseries to generate synthetic ones and generate link stream using those too
