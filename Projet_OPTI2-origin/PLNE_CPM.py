# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

## Importations

import pulp as pl
import networkx as nx
from tqdm import tqdm
import time
# from itertools import chain, combinations
import os
import shutil
import random

from Outils import *
from Draw import *


# path_to_cplex = "C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/python/3.9/x64_win64/cplex"
path_to_cplex = "C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.exe"

## Code

def PLNE_CPM(instance, source=None, verbose=True):
    dossier, nom = instance.split('/')
    # Définition du graphe
    G = creation_graphe(instance, True)

    if source is not None or not source in G.nodes:
        source = random.choice(list(G.nodes))

    # Troisième programme : PLNE CPM (flots multiples)

    time_debut_ecriture = time.time()

    # Définition des variables
    if verbose:
        print("Définition des variables :")
    with tqdm(total=G.number_of_nodes() + G.number_of_edges() + G.number_of_edges() * G.number_of_nodes(), disable=not verbose) as bar:
        y = pl.LpVariable.dicts("Sommet", G.nodes, 0, 1, pl.LpInteger)
        bar.update(G.number_of_nodes())
        x = pl.LpVariable.dicts("Arete", G.edges, 0, 1, pl.LpInteger)
        bar.update(G.number_of_edges())
        indices = [(e[0], e[1], v) for v in G.nodes for e in G.edges]
        f = pl.LpVariable.dicts("Flot", indices, 0, None, pl.LpInteger)
        bar.update(G.number_of_edges() * G.number_of_nodes())

    # Définition du problème
    if verbose:
        print("\nDéfinition du problème :")
    with tqdm(total=1, disable=not verbose) as bar:
        PLNE_CPM = pl.LpProblem("PLNE_CPM_" + instance[:-4], pl.LpMinimize)
        bar.update(1)

    # Ajout de la fonction objectif (17)
    if verbose:
        print("\nAjout de la fonction objectif (17) :")
    with tqdm(total=1, disable=not verbose) as bar:
        PLNE_CPM += pl.lpSum(y), "Objectif"
        bar.update(1)

    # Ajout des contraintes

    # Contrainte (18)
    if verbose:
        print("\nAjout des contraintes (18) :")
    with tqdm(total=G.number_of_nodes() - 1, disable=not verbose) as bar:
        compteur = 1
        for v in G.nodes:
            A_moins = G.in_edges(v)
            if A_moins and v != source:  # Vérifie que le sommet a des arêtes entrantes
                PLNE_CPM += (
                    pl.lpSum(x[e] for e in A_moins) == 1,
                    f"Contrainte (18) - {compteur}",
                )
                bar.update(1)
            compteur += 1

    # Contrainte (19)
    if verbose:
        print("\nAjout des contraintes (19) :")
    with tqdm(total=G.number_of_nodes() + (G.number_of_nodes() - 2) + 1, disable=not verbose) as bar:
        compteur1 = 1
        for v in G.nodes:
            compteur2 = 1
            A_plus = G.out_edges(v)
            A_moins = G.in_edges(v)
            if A_moins and A_plus:  # Vérifie que le sommet a des arêtes entrantes et sortantes
                for k in G.nodes:
                    if v != source and v != k:
                        PLNE_CPM += (
                            pl.lpSum(f[e[0], e[1], k] for e in A_plus) - pl.lpSum(
                                f[e[0], e[1], k] for e in A_moins) == 0,
                            f"Contrainte (19) - [{compteur1, compteur2}]",
                        )
                    bar.update(1)
                    compteur2 += 1
            compteur1 += 1

    # Contrainte (20)
    if verbose:
        print("\nAjout des contraintes (20) :")
    compteur = 1
    with tqdm(total=G.number_of_nodes(), disable=not verbose) as bar:
        compteur = 1
        A_plus = G.out_edges(source)
        A_moins = G.in_edges(source)
        for k in G.nodes:
            if A_moins and A_plus and k != source:  # Vérifie que la source a des arêtes entrantes et sortantes
                PLNE_CPM += (
                    pl.lpSum(f[e[0], e[1], k] for e in A_plus) - pl.lpSum(f[e[0], e[1], k] for e in A_moins) == 1,
                    f"Contrainte (20) - {compteur}",
                )
            compteur += 1
            bar.update(1)

    # Contrainte (21)
    if verbose:
        print("\nAjout des contraintes (21) :")
    compteur = 1
    with tqdm(total=G.number_of_nodes(), disable=not verbose) as bar:
        compteur = 1
        for k in G.nodes:
            A_plus = G.out_edges(k)
            A_moins = G.in_edges(k)
            if k != source and A_moins and A_plus:  # Vérifie que la source a des arêtes entrantes et sortantes
                PLNE_CPM += (
                    pl.lpSum(f[e[0], e[1], k] for e in A_plus) - pl.lpSum(f[e[0], e[1], k] for e in A_moins) == - 1,
                    f"Contrainte (21) - {compteur}",
                )
            bar.update(1)
            compteur += 1

    # Contrainte (22)
    if verbose:
        print("\nAjout des contraintes (22) :")
    compteur1, compteur2 = (1, 1)
    with tqdm(total=G.number_of_edges() * G.number_of_nodes(), disable=not verbose) as bar:
        for k in G.nodes:
            for e in G.edges:
                PLNE_CPM += (
                    f[e[0], e[1], k] <= x[e],
                    f"Contrainte (22) - [{compteur1, compteur2}]",
                )
                compteur1 += 1
                bar.update(1)
            compteur2 += 1

    # Contrainte (23)
    if verbose:
        print("\nAjout des contraintes (23) :")
    with tqdm(total=G.number_of_nodes(), disable=not verbose) as bar:
        compteur = 1
        for v in G.nodes:
            A_plus = G.out_edges(v)
            A_moins = G.in_edges(v)
            if A_moins and A_plus:  # Vérifie que le sommet a des arêtes entrantes et sortantes
                PLNE_CPM += (
                    pl.lpSum(x[e] for e in A_plus) + pl.lpSum(x[e] for e in A_moins) - 2 <= G.degree(v) * y[v],
                    f"Contrainte (23) - {compteur}",
                )
            compteur += 1
            bar.update(1)

    if verbose:
        print("\nEcriture du programme linéaire dans 'PLNE_CPM.lp' :")
    with tqdm(total=1, disable=not verbose) as bar:
        if not os.path.exists(f"./LP/{instance[:-4]}"):
            os.makedirs(f"./LP/{dossier}/{nom[:-4]}")
        PLNE_CPM.writeLP(f"./LP/{instance[:-4]}/PLNE_CPM_{nom[:-4]}.lp")
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
    # print(set_time_limit(G.number_of_nodes(), G.number_of_edges()))

    solver.timeLimit = set_time_limit(G.number_of_nodes(), G.number_of_edges())
    print(f"Time Limit: {int(solver.timeLimit) // 3600}h {int((solver.timeLimit % 3600) // 60)}m {int(solver.timeLimit % 60)}s {int((solver.timeLimit % 60 - int(solver.timeLimit % 60)) * 1000)}ms ({solver.timeLimit}s).\n")

    # solver.tmpDir = "./Solutions"
    result = PLNE_CPM.solve(solver=solver)

    time_fin_solver = time.time()
    duree_solver = time_fin_solver - time_debut_solver

    if verbose:
        print(f"Résolution du programme linéaire réalisée en : {int(duree_solver) // 3600}h {int((duree_solver % 3600) // 60)}m {int(duree_solver % 60)}s {int((duree_solver % 60 - int(duree_solver % 60)) * 1000)}ms.")

    ecriture_solution(instance=instance, modele="PLNE_CPM", programme=PLNE_CPM,
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

    PLNE_CPM(instance)

