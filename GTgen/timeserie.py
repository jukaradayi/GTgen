""" Timeserie object.
    A timeserie is stored as a numpy 1D array.
    The class implement shuffle_timeserie(index_low, index_high) that allows to
    shuffle the timeserie between index_low and index_high
"""

#import ipdb
import numpy as np
import matplotlib.pyplot as plt

class Timeserie():
    """ Timeserie Class

    Attributes:
    -----------
    serie: array
        the timeserie

    """
    def __init__(self, serie = np.array([]), out_path='./timeserie.txt'):
        self.serie = serie
        self.out_path = out_path
        self._sorted_timeserie = None

    @property
    def sorted_timeserie(self):
        if self._sorted_timeserie is None:
            self._sorted_timeserie = np.sort(self.serie)
        return self._sorted_timeserie

    @property
    def cumsum(self):
        return np.cumsum(self.serie)

    @property
    def duration(self):
        return len(self.serie)

    def write_TS(self):

        with open(self.out_path,'w') as fout:
            for time, val in enumerate(self.serie):
                # don't write if value is 0
                if val > 0:
                    fout.write(f'{time} {val}\n')


    def shuffle_timeserie(self, index_low=0, index_high=None):
        """ if index is given, only shuffle starting at index
        """
        if index_high is None:
            index_high = len(self.serie)
        # if index not given, shuffle all
        np.random.shuffle(self.serie[index_low:index_high])

    def plot(self):
        # TODO plot only anomlay + only normality in subplots
        plt.plot(self.serie)
        plt.show()
