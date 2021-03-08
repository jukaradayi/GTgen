import os
import ipdb
import yaml
import shutil
import logging
import argparse
import numpy as np

from GTgen.genDataGraph import *
from GTgen.genModelGraph import *
from GTgen.genDataTimeserie import *
from GTgen.genModelTimeserie import *
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

    # log seed and write it in config to write it in output folder
    logger.info('seed fixed to {}'.format(seed))
    config['seed'] = seed
    np.random.seed(seed)

    # manage output directory
    if not os.path.isdir(args.output):
        logger.info('create output folder {}'.format(args.output))
        os.makedirs(args.output)

    # generate graph
    #if config['Graph']['generate']:
    if config['Graph']['generate_data']:
        #graph_output = os.path.join(args.output, 'graph.txt')
        logger.info('generating graph from data')
        dg_generator = DataGraph(
                config['Graph']['data_params']['degree'],
                config['Graph']['data_params']['numberOfAnomalies'],
                config['Graph']['data_params']['n_anomaly'],
                config['Graph']['data_params']['m_anomaly'],
                config['Graph']['data_params']['N_swap1'],
                config['Graph']['data_params']['N_swap2'], ## TODO DEFINE N_SWAP2 
                config['Graph']['data_params']['weight'],
                logger,
                #graph_output,
                args.output,
                config['Graph']['data_params']['basename'],
                seed)
        dg_generator.run()
        dg_nw_sum, dg_anw_sum = dg_generator.sum_normality_weight
    if config['Graph']['generate_model']:
        logger.info('generating graph from model')
        mg_generator = ModelGraph(
                config['Graph']['model_params']['n_graphAnomaly'],
                config['Graph']['model_params']['n_streamAnomaly'],
                config['Graph']['model_params']['nNodes'],
                config['Graph']['model_params']['nNodes_graphAnomaly'],
                config['Graph']['model_params']['nNodes_streamAnomaly'],
                config['Graph']['model_params']['nEdges_normality'],
                config['Graph']['model_params']['nEdges_graphAnomaly'],
                config['Graph']['model_params']['nEdges_streamAnomaly'],
                config['Graph']['model_params']['nInteractions'],
                config['Graph']['model_params']['nInteractions_streamAnomaly'],
                args.output,
                seed,
                logger)
        mg_generator.run()

    # generate timeserie
    if config['TimeSerie']['generate_data']:
        logger.info('generating timeserie from data')
        timeserie_output = os.path.join(args.output, 'timeserie.txt')
        #dg_anw_sum = 219446
        #dg_nw_sum = 836318892
        dt_generator = DataTimeserie(np.array(
            [val for _, val in config['TimeSerie']['data_params']['dataset']]),
                config['TimeSerie']['data_params']['anomaly_type'],
                config['TimeSerie']['data_params']['anomaly_length'],
                dg_nw_sum, dg_anw_sum,
                args.output, logger)
        dt_generator.run()

    # generate Model Timeserie
    if config['TimeSerie']['generate_model']:
       logger.info('generating timeserie from model')
       mt_generator = ModelTimeserie(
               config['TimeSerie']['model_params']['duration'],
               config['TimeSerie']['model_params']['duration_streamAnomaly'],
               config['TimeSerie']['model_params']['duration_tsAnomaly'],
               config['TimeSerie']['model_params']['cum_sum'],
               config['TimeSerie']['model_params']['cum_sum_streamAnomaly'],
               args.output,
               logger)
       mt_generator.run()
    # TODO assert generator.sum_normality = generator.sum_normality
    # copy yaml in output folder
    #shutil.copyfile(args.yaml, os.path.join( args.output, "config.yaml"))
    # clean datasets before writing
    if config['TimeSerie']['generate_data']:
        config['TimeSerie']['data_params']['dataset'] = ''
    if config['Graph']['generate_data']: 
        config['Graph']['data_params']['degree'] = ''
        config['Graph']['data_params']['weight'] = ''
    with open(os.path.join( args.output, "config.yaml"), 'w') as fout:
        yaml.dump(config, fout)


if __name__ == "__main__":
    main()

