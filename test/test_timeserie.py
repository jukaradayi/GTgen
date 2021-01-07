from GTgen.timeserie import *
import numpy as np

def test_shuffle():
    # create step function
    serie1 = np.concatenate((8 * np.ones((5)), 10 * np.ones((5)), 2 * np.ones((5)), np.ones((5)) ))
    timeserie = Timeserie(serie = serie1, out_path = './')
    shift_index = 10

    timeserie.serie = timeserie.sorted_timeserie
    timeserie.shuffle_timeserie(index_low = 0, index_high = shift_index)
    timeserie.shuffle_timeserie(index_low = shift_index, index_high = len(serie1))
    assert max(timeserie.serie[:shift_index]) ==2
    assert min(timeserie.serie[shift_index:]) == 8

    # create step function
    serie2 = np.concatenate((1 * np.ones((5)), 2 * np.ones((5)), 8 * np.ones((5)), 10 * np.ones((5)) ))
    timeserie = Timeserie(serie = serie2, out_path = './')
    shift_index = 10

    timeserie.serie = timeserie.sorted_timeserie
    timeserie.shuffle_timeserie(index_low = 0, index_high = shift_index)
    timeserie.shuffle_timeserie(index_low = shift_index, index_high = len(serie2))
    assert max(timeserie.serie[:shift_index]) ==2
    assert min(timeserie.serie[shift_index:]) == 8

