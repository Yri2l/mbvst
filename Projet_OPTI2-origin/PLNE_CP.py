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

def PLNE_CP(instance, source=None, verbose=True):
    dossier, nom = instance.split('/')
    # Définition du graphe
    G = creation_graphe(instance, True)

    if source is not None or not source in G.nodes:
        source = random.choice(list(G.nodes))

    # Deuxième programme : PLNE CP (flot)

    time_debut_ecriture = time.time()

    # Définition des variables
    if verbose:
        print("Définition des variables :")
    with tqdm(total= G.number_of_nodes() + 2*G.number_of_edges(), disable=not verbose) as bar:
        y = pl.LpVariable.dicts("Sommet", G.nodes, 0, 1, pl.LpInteger)
        bar.update(G.number_of_nodes())
        x = pl.LpVariable.dicts("Arete", G.edges, 0, 1, pl.LpInteger)
        bar.update(G.number_of_edges())
        f = pl.LpVariable.dicts("Flot", G.edges, 0, None, pl.LpInteger)
        bar.update(G.number_of_edges())

    # Définition du problème
    if verbose:
        print("\nDéfinition du problème :")
    with tqdm(total=1, disable=not verbose) as bar:
        PLNE_CP = pl.LpProblem("PLNE_CP_" + instance[:-4], pl.LpMinimize)
        bar.update(1)


    # Ajout de la fonction objectif (8)
    if verbose:
        print("\nAjout de la fonction objectif (8) :")
    with tqdm(total=1, disable=not verbose) as bar:
        PLNE_CP += pl.lpSum(y), "Objectif"
        bar.update(1)

    # Ajout des contraintes

    # Contrainte (9)
    if verbose:
        print("\nAjout des contraintes (9) :")
    with tqdm(total=G.number_of_nodes(), disable=not verbose) as bar:
        compteur = 1
        for v in G.nodes:
            A_moins = G.in_edges(v)
            if A_moins and v != source:  # Vérifie que le sommet a des arêtes entrantes
                PLNE_CP += (
                    pl.lpSum(x[e] for e in A_moins) == 1,
                    f"Contrainte (9) - {compteur}",
                )
            compteur += 1
            bar.update(1)

    # Contrainte (10)
    if verbose:
        print("\nAjout de la contrainte (10) :")
    with tqdm(total=1, disable=not verbose) as bar:
        A_plus = G.out_edges(source)
        A_moins = G.in_edges(source)
        if A_moins and A_plus:  # Vérifie que la source a des arêtes entrantes et sortantes
            PLNE_CP += (
                pl.lpSum(f[e] for e in A_plus) - pl.lpSum(f[e] for e in A_moins) == G.number_of_nodes() - 1,
                f"Contrainte (10)",
            )
        bar.update(1)

    # Contrainte (11)
    if verbose:
        print("\nAjout des contraintes (11) :")
    compteur = 1
    with tqdm(total=G.number_of_nodes() - 1, disable=not verbose) as bar:
        # print(source)
        for v in G.nodes:
            if v != source:
                A_plus = G.out_edges(v)
                A_moins = G.in_edges(v)
                if A_moins and A_plus:  # Vérifie que le sommet a des arêtes entrantes et sortantes
                    PLNE_CP += (
                        pl.lpSum(f[e] for e in A_plus) - pl.lpSum(f[e] for e in A_moins) == - 1,
                        f"Contrainte (11) - {compteur}",
                    )
                bar.update(1)
            compteur += 1

    # Contrainte (12)
    if verbose:
        print("\nAjout des contraintes (12) :")
    compteur = 1
    with tqdm(total=G.number_of_edges(), disable=not verbose) as bar:
        for e in G.edges:
            PLNE_CP += (
                x[e] <= f[e],
                f"Contrainte (12) - {compteur}",
            )
            PLNE_CP += (
                f[e] <= G.number_of_nodes() * x[e],
                f"Contrainte (12 bis) - {compteur}",
            )
            compteur += 1
            bar.update(1)

    # Contrainte (13)
    if verbose:
        print("\nAjout des contraintes (13) :")
    with tqdm(total=G.number_of_nodes(), disable=not verbose) as bar:
        compteur = 1
        for v in G.nodes:
            A_plus = G.out_edges(v)
            A_moins = G.in_edges(v)
            if A_moins and A_plus:  # Vérifie que le sommet a des arêtes entrantes et sortantes
                PLNE_CP += (
                    pl.lpSum(x[e] for e in A_plus) + pl.lpSum(x[e] for e in A_moins) - 2 <= G.degree(v) * y[v],
                    f"Contrainte (13) - {compteur}",
                )
            compteur += 1
            bar.update(1)

    PLNE_CP += (
        pl.lpSum(x) == G.number_of_nodes() - 1,
        f"Contrainte (222) - {compteur}",
    )
    # print(f"Nombre de sommets : {G.number_of_nodes()}")

    if verbose:
        print("\nEcriture du programme linéaire dans 'PLNE_CP.lp' :")
    with tqdm(total=1, disable=not verbose) as bar:
        if not os.path.exists(f"./LP/{instance[:-4]}"):
            os.makedirs(f"./LP/{dossier}/{nom[:-4]}")
        PLNE_CP.writeLP(f"./LP/{instance[:-4]}/PLNE_CP_{nom[:-4]}.lp")
        bar.update(1)
    time_fin_ecriture = time.time()
    duree_ecriture = time_fin_ecriture - time_debut_ecriture

    if verbose:
        print(f"\nDéfinition et écriture du programme linéaire réalisées en : {int(duree_ecriture)//3600}h {int((duree_ecriture % 3600) // 60)}m {int(duree_ecriture % 60)}s {int((duree_ecriture % 60 - int(duree_ecriture % 60))*1000)}ms.")


    # Affichage du résultat

    time_debut_solver = time.time()

    if os.path.exists("./solver_Cplex.json"):
        solver = pl.getSolverFromJson("./solver_Cplex.json")
    else:
        solver = pl.CPLEX_CMD(path= path_to_cplex, logPath = "./log_Cplex/cplex.log", timeLimit = 600)
        solver.toJson("solver_Cplex.json")

    # solver.tmpDir = "./Solutions"
    solver.msg = verbose
    solver.timeLimit = set_time_limit(G.number_of_nodes(), G.number_of_edges())

    result = PLNE_CP.solve(solver=solver)

    time_fin_solver = time.time()
    duree_solver = time_fin_solver - time_debut_solver

    if verbose:
        print(f"Résolution du programme linéaire réalisée en : {int(duree_solver) // 3600}h {int((duree_solver % 3600) // 60)}m {int(duree_solver % 60)}s {int((duree_solver % 60 - int(duree_solver % 60)) * 1000)}ms.")

    # if not os.path.exists(f"./Solutions/{instance[:-4]}"):
    #     os.makedirs(f"./Solutions/{instance[:-4]}")
    # with open(f"./Solutions/{instance[:-4]}/sol_PLNE_CP_{instance}", "w") as fichier:
    #     fichier.write(f"Statut: {pl.LpStatus[PLNE_CP.status]}\n")
    #     fichier.write(f"Valeur de la fonction objectif: {pl.value(PLNE_CP.objective)}\n")
    #     fichier.write(f"Valeurs des variables (source: {source}):\n")
    #     for var in PLNE_CP.variables():
    #         fichier.write(f"{var.name} = {var.varValue}\n")

    ecriture_solution(instance=instance, modele="PLNE_CP", programme=PLNE_CP,
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
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_20_27_219.txt"
    instance = "Spd_Inst_Rid_Final2/Spd_RF2_1_1_1.txt"
    # instance = "Test.txt"
    PLNE_CP(instance)
