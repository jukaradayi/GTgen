import os
import ipdb
import yaml
import shutil
import logging
import argparse

from GTgen.genGraph import *
#from GTgen.timeserie import *
#import weighted_graph
#from graph import *
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

    logger = logging.getLogger()
    if not os.path.isdir(args.output):
        logger.info('create output folder {}'.format(args.output))
        os.makedirs(args.output)

    if config['Graph']['generate']:
        logger.info('generating graph')
        #Model = getattr(graph, config['Graph']['params']['model'])
        #generator = Model(**config['Graph']['params']['n'], config['Graph']['params']['p'])

        #degree_list = [4, 2, 2, 4, 2, 2]
        #degree_list = [8, 8, 8, 7, 7, 6, 6, 5, 5, 4 ,4]
        #generator = graph_generator.compareNetworkit(degree_list, logger)
        #generator.run()
        ## TODO restore
        #_, degree_list = read_degrees(config['Graph']['params']['degree'])
        #g_counter, g_weights = _read_dataset(config['Graph']['params']['dataset'])
        #config['Graph']['params']['dataset'] = g_weights
        generator = GraphWithAnomaly(
                config['Graph']['params']['degree'],
                config['Graph']['params']['numberOfAnomalies'],
                config['Graph']['params']['n_anomaly'],
                config['Graph']['params']['m_anomaly'],
                config['Graph']['params']['N_swap'],
                config['Graph']['params']['weight'],
                logger)
        generator.run()


        # generate weights from dataset
        #weighted = weighted_graph.WeightFromDataset(generator.graph, **config['Graph']['params'])
        #weighted.run()
        ## todo getparams
        #generator.write_graph(os.path.join(args.output, "model.weight"), weighted.weights)

    #if config['TimeSerie']['generate']:
    #    logger.info('generating timeserie')
    #    Model = getattr(timeserie, config['TimeSerie']['params']['model'])
    #    #generator = Model(config['TimeSerie']['params']['duration'], config['TimeSerie']['params']['bound_up'], config['TimeSerie']['params']['bound_down'])
    #    generator = Model(**config['TimeSerie']['params'], logger)
    #    generator.run()
    #    generator.write_TS(os.path.join(args.output, "model.ts"))
    ## copy yaml in output folder
    #shutil.copyfile(args.yaml, os.path.join( args.output, "benchmark.yaml"))


if __name__ == "__main__":
    main()

