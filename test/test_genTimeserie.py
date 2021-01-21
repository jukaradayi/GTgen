from GTgen.genTimeserie import *
import numpy as np

def test_regimeShift(logger):
    value_list = [0,2,1,1,1,2,3,2,4,7,8,9,8,9,10,7,8,1,1,0]
    timeserie = TimeserieWithAnomaly(value_list, "regimeShift",
                                    8/20, "./", logger)
    # do test 40 times because of random choice
    for i in range(40):
        timeserie._generate_regimeShift()

        anomaly_index = len(value_list) - int(8/20 * len(value_list))
        assert np.max(timeserie.timeserie.serie[:anomaly_index]) == 4
        assert np.min(timeserie.timeserie.serie[anomaly_index:]) == 7
        timeserie.timeserie.serie = timeserie.timeserie.sorted_timeserie


def test_peak(logger):
    value_list = [0,2,1,1,1,2,3,2,4,9,10,1,1,0]
    timeserie =  TimeserieWithAnomaly(value_list, "peak",
                                    1/7, "./", logger)
    for i in range(27):
        rdm_idx = timeserie._generate_peak()
        anomaly_duration = int(1/7 * len(value_list))

        if rdm_idx > 0: # fail max search when rdm_idx = 0
            assert np.max(timeserie.timeserie.serie[:rdm_idx]) <= 4
        assert np.max(timeserie.timeserie.serie[rdm_idx + anomaly_duration:]) <=4
        assert np.min(timeserie.timeserie.serie[rdm_idx:rdm_idx + anomaly_duration]) == 9

        timeserie.timeserie.serie = timeserie.timeserie.sorted_timeserie

   
