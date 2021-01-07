import time
import numpy as np
import matplotlib.pyplot as plt
from GTgen.timeserie import * 
class TimeserieWithAnomaly():
    """
    """
    def __init__(self, value_list,
            anomaly_type,
            anomaly_duration,
            logger):
        #self.numberOfAnomaly = numberofAnomaly

        self.value_list = value_list
        self.anomaly_type = anomaly_type
        self.anomaly_duration = anomaly_duration # define as ratio first...
        self.anomaly_index = None

        self.timeserie = Timeserie(serie=np.array(value_list), out_path='./timeserie.txt')
        # sort timeserie
        self.timeserie.serie = self.timeserie.sorted_timeserie
        self.logger = logger

    def _generate_regimeShift(self):
        # anomaly_duration is ratio
        self.anomaly_index = self.timeserie.duration - int(self.timeserie.duration * self.anomaly_duration)
        # regime shift
        self.timeserie.shuffle_timeserie(index_low = self.anomaly_index, index_high = None)
        # shuffle remaining for "normality"
        self.timeserie.shuffle_timeserie(index_low = 0, index_high = self.anomaly_index)

    def _generate_peak(self):
        # anomaly index is duration of peak in samples
        self.anomaly_index = self.timeserie.duration - self.anomaly_duration
        # get peak
        self.timeserie.shuffle_timeserie(index_low = self.anomaly_index, index_high = None)
        # get normal
        self.timeserie.shuffle_timeserie(index_low = 0, index_high = self.anomaly_index)
        # put peak random place in timeserie
        rdm_index = np.random.choice(self.timeserie.duration)
        self.logger.info('rdm index {}'.format(rdm_index))
        global_serie = np.concatenate([self.timeserie.serie[:rdm_index], self.timeserie.serie[self.anomaly_index:], self.timeserie.serie[rdm_index:]])
        self.timeserie.serie = global_serie

    #def _generate_normal(self):
    #    self.timeserie.shuffle_timeserie(index_low = 0, index_high = self.anomaly_index)
    
    def run(self):
        if self.anomaly_type == "regimeShift":
            # generate anomaly
            self._generate_regimeShift()
        elif self.anomaly_type == "peak":
            self._generate_peak()
        ## generate normality
        #self._generate_normal()
        #self.logger.info('min anomaly {} max anomaly {},\nmin normal {} max normal {},\nmain all{} max all {}'.format(
        #    np.min(self.timeserie.serie[self.anomaly_index:]), np.max(self.timeserie.serie[self.anomaly_index:]),
        #    np.min(self.timeserie.serie[:self.anomaly_index]), np.max(self.timeserie.serie[:self.anomaly_index]),
        #    np.min(self.timeserie.serie[:]), np.max(self.timeserie.serie[:])))
        self.timeserie.plot()
