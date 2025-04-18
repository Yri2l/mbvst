# This is a sample Python script.
from tabnanny import verbose

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

## Importations

import pulp as pl
import networkx as nx
from tqdm import tqdm
import time
from itertools import chain, combinations
import os
import shutil

from Outils import *
from Draw import *


# path_to_cplex = "C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/python/3.9/x64_win64/cplex"
path_to_cplex = "C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.exe"

## Code

def PLNE_exp(instance, verbose=True):
    dossier, nom = instance.split('/')
    # Définition du graphe
    G = creation_graphe(instance)

    # Premier programme : PLNE exponentiel

    time_debut_ecriture = time.time()

    # Définition des variables
    if verbose:
        print("Définition des variables :")
    with tqdm(total=G.number_of_nodes() + G.number_of_edges(), disable=not verbose) as bar:
        y = pl.LpVariable.dicts("Sommet", G.nodes, 0, 1, pl.LpInteger)
        bar.update(G.number_of_nodes())
        x = pl.LpVariable.dicts("Arete", G.edges, 0, 1, pl.LpInteger)
        bar.update(G.number_of_edges())

    # Définition du problème
    if verbose:
        print("\nDéfinition du problème :")
    with tqdm(total=1, disable=not verbose) as bar:
        PLNE_exp = pl.LpProblem("PLNE_Exponentiel_" + instance[:-4], pl.LpMinimize)
        bar.update(1)

    # Ajout de la fonction objectif (2)
    if verbose:
        print("\nAjout de la fonction objectif (2) :")
    with tqdm(total=1, disable=not verbose) as bar:
        PLNE_exp += pl.lpSum(y), "Objectif"
        bar.update(1)

    # Ajout des contraintes

    # Contrainte (3)
    if verbose:
        print("\nAjout de la contrainte (3) :")
    with tqdm(total=1, disable=not verbose) as bar:
        PLNE_exp += (
            pl.lpSum(x) == G.number_of_nodes() - 1,
            "Contrainte (3)",
        )
        bar.update(1)

    # Contrainte (4)
    if verbose:
        print("\nAjout des contraintes (4) :")
    compteur = 1
    with tqdm(total=(1 << G.number_of_nodes()) - 1, disable=not verbose) as bar:
        for r in range(1, G.number_of_nodes() + 1):
            for S in combinations(G.nodes, r):
                edges_in_S = list(nx.subgraph(G, S).edges)
                if edges_in_S:  # Vérifie que le sous-graphe a des arêtes
                    PLNE_exp += (
                        pl.lpSum(x[(min(e), max(e))] for e in edges_in_S) <= r - 1,
                        f"Contrainte (4) - {compteur}",
                    )
                compteur += 1
                bar.update(1)

    # Contrainte (5)
    if verbose:
        print("\nAjout des contraintes (5) :")
    compteur = 1
    with tqdm(total=G.number_of_nodes(), disable=not verbose) as bar:
        for v in G.nodes:
            incident_edges = list(G.edges(v))
            if incident_edges:  # Vérifie si le nœud a des arêtes incidentes
                PLNE_exp += (
                    pl.lpSum(x[(min(e), max(e))] for e in incident_edges) - 2 <= G.degree[v] * y[v],
                    f"Contrainte (5) - {compteur}",
                )
            compteur += 1
            bar.update(1)

    if verbose:
        print("\nEcriture du programme linéaire dans 'PLNE_Exponentiel.lp' :")
    with tqdm(total=1) as bar:
        if not os.path.exists(f"./LP/{instance[:-4]}"):
            os.makedirs(f"./LP/{dossier}/{nom[:-4]}")
        PLNE_exp.writeLP(f"./LP/{instance[:-4]}/PLNE_Exp_{nom[:-4]}.lp")
        bar.update(1)
    time_fin_ecriture = time.time()
    duree_ecriture = time_fin_ecriture - time_debut_ecriture

    if verbose:
        print(f"\nDéfinition et écriture du programme linéaire réalisées en : {int(duree_ecriture) // 3600}h {int((duree_ecriture % 3600) // 60)}m {int(duree_ecriture % 60)}s {int((duree_ecriture % 60 - int(duree_ecriture % 60)) * 1000)}ms.")

    # Affichage du résultat

    time_debut_solver = time.time()

    if os.path.exists("./solver_Cplex.json"):
        solver = pl.getSolverFromJson("./solver_Cplex.json")
    else:
        solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
        solver.toJson("solver_Cplex.json")
    solver.msg = verbose
    solver.timeLimit = set_time_limit(G.number_of_nodes(), G.number_of_edges())

    # solver.tmpDir = "./Solutions"
    result = PLNE_exp.solve(solver=solver)

    time_fin_solver = time.time()
    duree_solver = time_fin_solver - time_debut_solver

    if verbose:
        print(f"Résolution du programme linéaire réalisée en : {int(duree_solver) // 3600}h {int((duree_solver % 3600) // 60)}m {int(duree_solver % 60)}s {int((duree_solver % 60 - int(duree_solver % 60)) * 1000)}ms.")

    ecriture_solution(instance=instance, modele="PLNE_exp", programme=PLNE_exp,
                      duree_ecriture=duree_ecriture, duree_solver=duree_solver)

    liste_fichier = os.listdir()
    for fichier in liste_fichier:
        if len(fichier) >= 5 and fichier[:5] == "clone":
            if fichier in os.listdir("./log_Cplex/"):
                os.remove(f"./log_Cplex/{fichier}")
            shutil.move(fichier, f"./log_Cplex/{fichier}")
    print(result)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Définition du graphe
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_20_27_211.txt"
    instance = "Spd_Inst_Rid_Final2/Spd_RF2_20_27_219.txt"
    # instance = "Test.txt"

    PLNE_exp(instance)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
