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

    # generate graph from data
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

        # run generation
        dg_generator.run()

        # get sum of weights for anomaly timeserie generation
        dg_nw_sum, dg_anw_sum = dg_generator.sum_normality_weight

    # generate graph from model
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

        # run generation
        mg_generator.run()

    # generate timeserie from data
    if config['TimeSerie']['generate_data']:
        logger.info('generating timeserie from data')
        timeserie_output = os.path.join(args.output, 'timeserie.txt')

        # give graph weight sum as input to generate anomaly
        dt_generator = DataTimeserie(np.array(
            [val for _, val in config['TimeSerie']['data_params']['dataset']]),
                config['TimeSerie']['data_params']['anomaly_type'],
                config['TimeSerie']['data_params']['anomaly_length'],
                dg_nw_sum, dg_anw_sum,
                args.output, logger)

        # run model
        dt_generator.run()

        # check that timeserie and graph have same number of interactions
        assert dt_generator.an_timeserie.serie.sum() == dg_anw_sum


    # generate Timeserie from model
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

       # run generation
       mt_generator.run()

    # copy yaml in output folder, with seed
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

