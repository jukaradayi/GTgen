import os
import pickle
import pytest
import logging

from GTgen.utils import *

@pytest.fixture(scope='session')
def data_path():
    return os.path.join(os.path.dirname(__file__), 'data')

@pytest.fixture(scope='session')
def NKedges(data_path):
    ''' pickled networkit output for Havel Hakimi'''
    with open(os.path.join(data_path, 'nkEdges.pkl'), 'rb') as fin:
        return pickle.load(fin)

@pytest.fixture(scope='session')
def taxi_seq(data_path):
    _, degree_list = read_degrees(os.path.join(data_path, 'taxi_deg_seq.txt'))
    return [deg for _, deg in degree_list]

@pytest.fixture(scope='session')
def logger():
    logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M'
            )
    return logging.getLogger()

