import argparse
import ipdb
import random

from networkx.generators.random_graphs import gnm_random_graph
from networkit.generators import EdgeSwitchingMarkovChainGenerator, HavelHakimiGenerator, ErdosRenyiGenerator

class AbstractGraphGenerator():
    """ Abstract Class for graph Generators
        Generate a random graph, picked uniformely, given a number of nodes,
        a number of edges, and a degree sequence.
        Attributes:
        -----------
        n : int
            the number of nodes 
        m : int
            the number of edges
        seq : list of int
            the degree sequence

    """
    def __init__(self, **kwargs):
        self.n = None
        self.m = None
        self.seq = None

    def is_realisable():
        raise NotImplementedError

    def write_graph(self, weights = None):
        ## TODO random weight
        # sort nodes
        #sum_weight = 0
        if weights is not None:
            iterator = zip(self.graph.iterEdges(), weights)
        else:
            iterator = self.graph.iterEdges() # TODO add zip (ones) pour ajouter des poids fictifs ? 
        for ((u, v), weight) in iterator:
            if u<v :
                fout.write(f'{u},{v} {weight}\n')
            else:
                fout.write(f'{v},{u} {weight}\n')

    def generate_weight(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

class ConfigurationModel(AbstractGraphGenerator):
    """ Implementation of configuration model
    """
    def __init__(self, **kwargs):
        self.seq = kwargs['seq']
        self.anomaly_seq = kwargs['anomaly_seq']
        self.anomaly_type = kwargs['anomaly_type']
        raise NotImplementedError

    def _clique(self):
        ## TODO: Idée à discuter avec Matthieu: on part de 2 graphes, un avec aucun lien G0, un avec tous GN, 
        # à chaque fois qu'on ajoute un lien dans G0, on enlève ce lien dans GN => on a toutes les cliques possibles,
        # reste à trouver celle de la taille qu'on veut (probably long)
        # possiblement très overkill
        raise NotImplementedError

    def _configuration_run(self):
        raise NotImplementedError

    def run(self):
        # generate anomaly first then feed into model
        anomaly = getattr(self, kwargs['anomaly_type'])
        self.anomaly(self.graph)
        
        self._configuration_run()
        raise NotImplementedError

class EdgeSwitchingMarkovChain(AbstractGraphGenerator):
    """ Wrapper of Networkit EdgeSwitchingMarkovChainGenerator 
        Also called 'configuration model' in networkit.
        TODO
        Parameters:
        -----------
        degreeSequence : vector[count]
            The degree sequence that shall be generated
        ignoreIfRealizable : bool, optional
            If true, generate the graph even if the degree sequence is not realizable. Some nodes may get lower degrees than requested in the sequence.
    """

    def __init__(self, **kwargs):
        self.sequence = kwargs['seq']
        self.generator = EdgeSwitchingMarkovChainGenerator(self.sequence)
        self.generator.isRealizable()
        self.graph = None

    def run(self):
        self.graph = self.generator.generate()

class HavelHakimi(AbstractGraphGenerator):
    """ Wrapper of Networkit HavelHakimi
        TODO
        Parameters:
        -----------
        sequence : vector
            Degree sequence to realize. Must be non-increasing.
        ignoreIfRealizable : bool, optional
            If true, generate the graph even if the degree sequence is not realizable. Some nodes may get lower degrees than requested in the sequence.
    """

    def __init__(self, **kwargs):
        self.sequence = kwargs['seq']
        self.generator = HavelHakimiGenerator(self.sequence)
        self.graph = None

    def run(self):
        self.graph = self.generator.generate()

class ErdosRenyi(AbstractGraphGenerator):
    """ Wrapper of Networkit ErdosRenyiGenerator
        TODO
        Parameters:
           -----------
           nNodes : count
               Number of nodes n in the graph.
           prob : double
               Probability of existence for each edge p.
           directed : bool
               Generates a directed
           selfLoops : bool
               Allows self-loops to be generated (only for directed graphs)
    """   
    #def __init__(self, n, p):
    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.generator = ErdosRenyiGenerator(self.n, self.p, directed = False, selfLoops=False)
        self.graph = None

    def run(self):
        #assert
        self.graph = self.generator.generate()

class GNM(AbstractGraphGenerator):
    """ Wrapper of Networkx gnm_random_graph
    Parameters:
    -----------
        n: int
            number of nodes
        m: int
            number of edges
        seed: int
            random seed
    """
    def __init__(self, **kwargs, logger):
        self.n = kwargs['n']
        self.m = kwargs['m'] +1
        #self.seed = seed if not None else None

    def write_graph(self, out_path, weights):
        ## TODO random weight
        # sort nodes
        #sum_weight = 0
        if weights is not None:
            print(len(self.graph.edges()))
            print(len(weights))
            iterator = zip(self.graph.edges(), weights)
        else:
            iterator = self.graph.iterEdges()
        with open(out_path, 'w') as fout:


            for ((u, v), weight) in iterator:
                # don't write if weight is 0
                if weight <= 0:
                    continue

                if u<v :
                    fout.write(f'{u},{v} {weight}\n')
                else:
                    fout.write(f'{v},{u} {weight}\n')

    def run(self):
        self.graph = gnm_random_graph(self.n, self.m, seed=None, directed=False)
