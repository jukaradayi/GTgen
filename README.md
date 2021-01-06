generate random time serie and graph from model

3 sequence : 
bleu = graphe normal
rouge = graphe anomalie
noir = somme des deux

hyp: 
trois séquences réalisables

stratégie : tirer rouge, puis générer tout plus noir


- sequence noire = vient de données réelles
- forcer rouge où ca passe 
- tester avec une anomalie 


I tirer noir données réelles 

II former rouge avec modèle anomalie (erdos renyii où on donne  le nombre de noeud/liens ?)


III1 Tirer bleu (trouver endroit où rouge va cf iii) aléatoirement (havel hakimi + mélanges) 

III2 si III1 marche pas, mélanger bleu ou rouge (avec proba de choix bleu ou rouge)

III3 si III2 marche pas, switch spécifiquement 

Pour taxi 1000 * 1000, avec 10 * n edges = 180 000 000 edge switchs 
    Command being timed: "python gen_graph.py -y modelGeneration.yaml"
    User time (seconds): 5471.83
    System time (seconds): 2.74
    Percent of CPU this job got: 99%
    Elapsed (wall clock) time (h:mm:ss or m:ss): 1:31:15
    Average shared text size (kbytes): 0
    Average unshared data size (kbytes): 0
    Average stack size (kbytes): 0
    Average total size (kbytes): 0
    Maximum resident set size (kbytes): 1723916
    Average resident set size (kbytes): 0
    Major (requiring I/O) page faults: 0
    Minor (reclaiming a frame) page faults: 911948
    Voluntary context switches: 70
    Involuntary context switches: 3076
    Swaps: 0
    File system inputs: 0
    File system outputs: 112
    Socket messages sent: 0
    Socket messages received: 0
    Signals delivered: 0
    Page size (bytes): 4096
    Exit status: 0

