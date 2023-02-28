.. _data_model:

Data Model
==========

This process generates a timeserie with an anomaly by using a measured timeserie as input.

Pipeline
--------

- Take a timeserie Ts measured on a dataset.

- Sort the timeserie Ts in increasing order

- select the N_an last (highest) values to host anomaly

- For the N_an values, randomly choose the Ts_an(t) values, where 
  Ts_an(t) <= Ts(t)

- Define 'normality' timeserie as Ts_norm(t) = Ts(t) - Ts_an(t)

- Generate normal timeserie and anomaly timeserie by "shuffling"*
  Ts_norm and Ts_an, i.e. define two bijective mappings T->T (where T is the set of timestamps of the timeseries) map1 and map2, such that 
  Ts'_norm(t) = Ts_norm(map1(t)) and Ts'_an(t) = Ts_an(map2(t))

