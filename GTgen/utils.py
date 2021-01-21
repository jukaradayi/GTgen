import os
import yaml

def read_config(conf_file):
    """ Read yaml configuration file """
    print(conf_file)
    assert os.path.isfile(conf_file), "input file doesn't exist"
    config = yaml.load(open(conf_file, 'r'), Loader=yaml.FullLoader)
    return config

def check_config(config):
    """ Check consistency of configuration file
    """
    assert 'degree' in config['Graph']['params']
    assert os.path.isfile(config['Graph']['params']['degree']), "degree file doesn't exist"
    _, degree_list = _read_degrees(config['Graph']['params']['degree'])
    config['Graph']['params']['degree'] = degree_list

    # check anomaly consistency
    #if config['Graph']['params']['numberOfAnomalie
    if "numberOfAnomalies" not in config['Graph']['params']:
        print('warning: no number of anomalies in config, assuming 0')
    else:
        assert "n_anomaly" in config['Graph']['params']
        assert "m_anomaly" in config['Graph']['params']
        assert config['Graph']['params']['n_anomaly'] * config['Graph']['params']['numberOfAnomalies'] < len(config['Graph']['params']['degree']), "anomaly has more nodes than normal graph" 
        degrees = [deg for _, deg in config['Graph']['params']['degree']]
        assert config['Graph']['params']['m_anomaly'] * config['Graph']['params']['numberOfAnomalies'] < sum(degrees)/2, "anomaly has more edges than normal graph"

    # check dataset path is filled and exists
    assert "weight" in config['Graph']['params']
    assert os.path.isfile(config['Graph']['params']['weight'])
    print(config['Graph']['params']['weight'])
    g_counter, g_weights = _read_dataset(config['Graph']['params']['weight'])
    config['Graph']['params']['weight'] = g_weights
    # get implied number of edges
    #n_edges = 0
    #n_interaction_g = 0
    #for value, count in g_counter.items():
    #    n_edges += count
    #    n_interaction_g += value * count

    #if config['Graph']['params']['model'] == 'GNM':
    #    if "m" in config['Graph']['params'] :
    #        m = config['Graph']['params']['m']
    #        print(f'Using dataset number of edges : {n_edges}, ignoring config value of {m} for GNM model')
    #    config['Graph']['params']['m'] = n_edges

        #assert n_edges == m, f"Dataset weight distribution has {n_edges} edges, but GNM model only requests {m}"
    #elif (config['Graph']['params']['model'] == 'HavelHakimi' 
    #onfig['Graph']['params']['model'] == 'EdgeSwitchingMarkovChain'):
    #    sum_seq = sum(config['Graph']['params']['seq'])
    #    assert sum_seq == 2 * n_edges, f"Dataset weight distribution has {n_edges} edges, but degree sequence requests {sum_seq}/2"
    

    # Timeseries parameters
    #if config['TimeSerie']['params']['model'] == 'TSFromDataset':
    assert "dataset" in config['TimeSerie']['params']
    assert os.path.isfile(config['TimeSerie']['params']['dataset'])
    ts_counter, ts_weights = _read_dataset(config['TimeSerie']['params']['dataset'])
    config['TimeSerie']['params']['dataset'] = ts_weights


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

def _read_degrees(dataset):
    degree_list = []
    idx=0
    vertex2vidx = dict()
    with open(dataset, 'r') as fin:
        data = fin.readlines()
        for line in data:
            vertex, degree = line.strip().split(' ')
            vertex2vidx[idx] = vertex
            degree_list.append((idx, int(degree)))
            idx += 1
    return vertex2vidx, degree_list

