import os
import yaml

def read_config(conf_file):
    """ Read yaml configuration file """
    assert os.path.isfile(conf_file), "input file doesn't exist"
    config = yaml.load(open(conf_file, 'r'), Loader=yaml.FullLoader)
    return config

def check_config(config):
    """ Check consistency of configuration file
    """
    # Graph parameters
    if config['Graph']['params']['model'] == 'ErdosRenyi':
        
        # check that n and p are filled as int ## TODO check positivity
        assert 'n' in config['Graph']['params']
        assert 'p' in config['Graph']['params']
        assert type(config['Graph']['params']['n']) is int
        assert type(config['Graph']['params']['p']) is int   
    elif config['Graph']['params']['model'] == 'GNM': 
        
        # check that n and m are filled as int ## TODO check positivity
        assert 'n' in config['Graph']['params']
        assert ('m' in config['Graph']['params'] or config['Graph']['params']['use_dataset'] is True)
        assert type(config['Graph']['params']['n']) is int
        if "m" in config['Graph']['params'] :
            assert type(config['Graph']['params']['m']) is int   
    elif (config['Graph']['params']['model'] == 'HavelHakimi' 
        or config['Graph']['params']['model'] == 'EdgeSwitchingMarkovChain'):
        
            # check that sequence of degree is filled
            ## check all positive integeres
        assert 'seq' in config['Graph']['params']
        assert type(config['Graph']['params']) is list

        # anomaly parameters check
        if "anomaly" in config['Graph']:
            if "seq" in config['Graph']['anomaly']:
                assert (len(config['Graph']['anomaly']['seq']) 
              == len(config['Graph']['anomaly']['seq']))

    ## weighted graph parameters
    if config['Graph']['params']['use_dataset'] is True:
        
        # check dataset path is filled and exists
        assert "dataset" in config['Graph']['params']
        assert os.path.isfile(config['Graph']['params']['dataset'])

        # if dataset_fit set to strict, read weight distribution file and check
        # if number of weights is the same as number of edges or
        # as sum of degree sequence
        #if config['Graph']['params']['dataset_fit'] == "strict":
        g_counter, g_weights = _read_dataset(config['Graph']['params']['dataset'])
        config['Graph']['params']['dataset'] = g_weights
        # get implied number of edges
        n_edges = 0
        n_interaction_g = 0
        for value, count in g_counter.items():
            n_edges += count
            n_interaction_g += value * count

        if config['Graph']['params']['model'] == 'GNM':
            if "m" in config['Graph']['params'] :
                m = config['Graph']['params']['m']
                print(f'Using dataset number of edges : {n_edges}, ignoring config value of {m} for GNM model')
            config['Graph']['params']['m'] = n_edges

            #assert n_edges == m, f"Dataset weight distribution has {n_edges} edges, but GNM model only requests {m}"
        elif (config['Graph']['params']['model'] == 'HavelHakimi' 
    or config['Graph']['params']['model'] == 'EdgeSwitchingMarkovChain'):
            sum_seq = sum(config['Graph']['params']['seq'])
            assert sum_seq == 2 * n_edges, f"Dataset weight distribution has {n_edges} edges, but degree sequence requests {sum_seq}/2"
    

    # Timeseries parameters
    if config['TimeSerie']['params']['model'] == 'TSFromDataset':
        assert "dataset" in config['TimeSerie']['params']
        assert os.path.isfile(config['TimeSerie']['params']['dataset'])
        ts_counter, ts_weights = _read_dataset(config['TimeSerie']['params']['dataset'])
        config['TimeSerie']['params']['dataset'] = ts_weights
        # get nb of interactions
        n_interaction_ts = 0
        for value, count in ts_counter.items():
            n_interaction_ts += value * count


    elif config['TimeSerie']['params']['model'] == 'IidNoise':
        assert 'bound_up' in config['TimeSerie']['params']
        assert 'bound_down' in config['TimeSerie']['params']
        assert config['TimeSerie']['params']['bound_up'] > config['TimeSerie']['params']['bound_down'] 
        assert config['TimeSerie']['params']['duration'] > 0

    # Cross graph - timeseries check
    if (config['TimeSerie']['params']['model'] == 'TSFromDataset'
        and config['Graph']['params']['use_dataset']):
        assert n_interaction_ts ==  n_interaction_g, f"time series interactions {n_interaction_ts}, graph interactions {n_interaction_g}"
   
def _read_dataset(dataset):
        counter = dict()
        distribution = []
        with open(dataset, 'r') as fin: ## put other option to read gz
            data = fin.readlines()
            for line in data:
                val, weight = line.strip().split()
                distribution.append((int(val), int(weight)))
                counter[int(val)] = int(weight)
        return counter, distribution


