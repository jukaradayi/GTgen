"""
    Generate a Timeserie with a pseudo-white noise model.
    The model is constrained by its length, its cumulative sum,
    and it's minimum value.
    To get such timeserie, initiate array filled with ones,
    and randomly pick indexes to incriment until the cumulative sum is reached
    In the context of the benchmark, the cumulative sum is the total number of
    interactions and should be the same as the sum of the graph weights.
    


"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from GTgen.timeserie import * 

class ModelTimeserie():
    """
        Attributes:
        -----------
        duration: int,
            number of timestamps of the timeserie
        duration_streamAnomaly: int,
            the stream anomaly has the same length as the timeserie, with
            0 everywhere except for duration_streamAnomaly contiguous values
        duration_tsAnomaly: int,
            the timeserie anomaly has the same length as the timeserie, with
            0 everywhere except for duration_streamAnomaly contiguous values
        cum_sum: int,
            the cumulative sum of the timeserie (including the timeserie
            anomaly
        cum_sum: int,
            the cumulative sum of the stream anomaly
        output: str,
            path to the folder in which the ouput will be written
        logger: logger,
            a logger
    """

    def __init__(self, duration,
                 duration_streamAnomaly,
                 duration_tsAnomaly,
                 cum_sum,
                 cum_sum_streamAnomaly,
                 output,
                 logger):
        # define normal timeserie
        self.cum_sum = cum_sum
        self.duration = duration
        self.output = output

        # anomalies
        self.duration_streamAnomaly = duration_streamAnomaly
        self.duration_tsAnomaly = duration_tsAnomaly
        self.cum_sum_streamAnomaly = cum_sum_streamAnomaly

        self.logger = logger

    #def generate_tsAnomaly(self):
    #
    @staticmethod
    def _generate_whitenoise(cum_sum, duration):

        timeserie = np.ones((duration,))
        #cum_sum = np.sum(timeserie)
        while np.sum(timeserie) < cum_sum:
            idx = np.random.choice(duration)
            timeserie[idx] += 1
        return timeserie

    def run(self):
        self.logger.info('generating normal timeserie')
        normal_serie = self._generate_whitenoise(self.cum_sum, self.duration)
        self.normal_timeserie = Timeserie(serie = normal_serie, 
                out_path=os.path.join(self.output, "normal_serie.txt")) 
        self.normal_timeserie.write_TS()


        # pick random index for stream anomaly
        self.logger.info('generating stream anomaly timeserie')
        _streamAnomaly = self._generate_whitenoise(self.cum_sum_streamAnomaly, self.duration_streamAnomaly)
        streamAnomaly = np.zeros((self.duration,))
        an_idx = np.random.choice(self.duration - self.duration_streamAnomaly)
        streamAnomaly[an_idx : an_idx+self.duration_streamAnomaly] = _streamAnomaly
        self.streamAnomaly = Timeserie(serie = streamAnomaly,
                out_path=os.path.join(self.output,"streamAnomaly_serie.txt"))
        self.streamAnomaly.write_TS()

