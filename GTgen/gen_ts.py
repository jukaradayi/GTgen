import os
import ipdb
import yaml
import shutil
import logging
import argparse

#from GTgen.genGraph import *
from GTgen.genTimeserie import *
#from GTgen.timeserie import *
#import weighted_graph
#from graph import *
from GTgen.utils import _read_dataset, read_config

"""
output format
timeserie:
0 1
1573 1
5768 3

graph:
<0-1>,<1-1> 3
<0-1>,<1-2> 2
<0-2>,<2-3> 3

"""

def main():
    parser = argparse.ArgumentParser(description='Graph and Time serie generator')
    parser.add_argument(
        '-y', '--yaml', metavar='config-file', default='./config.yaml',
        help='The YAML configuration file to read.'
        'If not specified, uses ./config.yaml')
    parser.add_argument(
        '-o', '--output', default='./',
        help='Folder in which output will be written')
    parser.add_argument(
        '--to_console', default=False,
        help='if enabled, write output directly to console (might be useful for piping, might remove...)')
    parser.add_argument(
        '-v', '--verbose', default=False, action='store_true',
        help='Be more verbose')
    args = parser.parse_args()

    # read config
    config = read_config(args.yaml)
    #check_config(config)

    # instantiate logger
    if args.verbose:
        logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s %(levelname)-8s %(message)s',
                datefmt='%m-%d %H:%M'
                )
    else:
        logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s %(levelname)-8s %(message)s',
                datefmt='%m-%d %H:%M'
                )

    logger = logging.getLogger()
    if not os.path.isdir(args.output):
        logger.info('create output folder {}'.format(args.output))
        os.makedirs(args.output)

    if config['Graph']['generate']:
        logger.info('generating timeserie')

        _, value_list = _read_dataset(config['TimeSerie']['params']['dataset'])


        generator = TimeserieWithAnomaly(np.array([val for _, val in value_list]), 'regimeShift', 1/10, logger)
        #generator = GraphWithAnomaly(degree_list,
        #        config['Graph']['params']['numberOfAnomalies'],
        #        config['Graph']['params']['n_anomaly'],
        #        config['Graph']['params']['m_anomaly'],
        #        config['Graph']['params']['N_switch'],
        #        logger)
        generator.run()

if __name__ == "__main__":
    main()

