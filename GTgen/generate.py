import os
import ipdb
import yaml
import shutil
import logging
import argparse
import numpy as np

from GTgen.genGraph import *
from GTgen.genTimeserie import *
from GTgen.utils import *

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
    parser = argparse.ArgumentParser(
            description='Graph and Time serie generator')
    parser.add_argument(
        '-y', '--yaml', metavar='config-file', default='./config.yaml',
        help='The YAML configuration file to read.'
        'If not specified, uses ./config.yaml')
    parser.add_argument(
        '-o', '--output', default='./',
        help='Folder in which output will be written')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='fix random seed')
    parser.add_argument(
        '-v', '--verbose', default=False, action='store_true',
        help='Be more verbose')
    args = parser.parse_args()

    # read config
    config = read_config(args.yaml)
    check_config(config)

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

    # instantiate logger
    logger = logging.getLogger()

    # fix random seed
    if args.seed is None:
        seed = np.random.choice(10**6)
    else:
        seed = args.seed
    logger.info('seed fixed to {}'.format(seed))
    np.random.seed(seed)

    # manage output directory
    if not os.path.isdir(args.output):
        logger.info('create output folder {}'.format(args.output))
        os.makedirs(args.output)

    # generate graph
    if config['Graph']['generate']:
        graph_output = os.path.join(args.output, 'graph.txt')
        logger.info('generating graph')
        generator = GraphWithAnomaly(
                config['Graph']['params']['degree'],
                config['Graph']['params']['numberOfAnomalies'],
                config['Graph']['params']['n_anomaly'],
                config['Graph']['params']['m_anomaly'],
                config['Graph']['params']['N_swap1'],
                config['Graph']['params']['N_swap2'], ## TODO DEFINE N_SWAP2 
                config['Graph']['params']['weight'],
                logger,
                graph_output,
                seed)
        generator.run()

    # generate timeserie
    if config['TimeSerie']['generate']:
        logger.info('generating timeserie')
        timeserie_output = os.path.join(args.output, 'timeserie.txt')
        generator = TimeserieWithAnomaly(np.array(
            [val for _, val in config['TimeSerie']['params']['dataset']]),
                config['TimeSerie']['params']['anomaly_type'],
                config['TimeSerie']['params']['anomaly_length'], 
                timeserie_output, logger)
        generator.run()

    # copy yaml in output folder
    shutil.copyfile(args.yaml, os.path.join( args.output, "config.yaml"))


if __name__ == "__main__":
    main()

