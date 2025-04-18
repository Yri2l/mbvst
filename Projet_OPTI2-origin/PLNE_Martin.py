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
path_to_cplex = "C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x6q4_win64/cplex.exe"

## Code

def PLNE_Martin(instance, verbose=True):
    dossier, nom = instance.split('/')
    # Définition du graphe
    G = creation_graphe(instance, False)


    # Quatrième programme : PLNE Martin

    time_debut_ecriture = time.time()

    # Définition des variables
    if verbose:
        print("Définition des variables :")
    # with tqdm(total=G.number_of_nodes() + G.number_of_edges() + 3 * G.number_of_edges() * G.number_of_nodes(), disable=not verbose) as bar:
    with tqdm(total=G.number_of_nodes() + G.number_of_edges() + G.number_of_nodes()**3, disable=not verbose) as bar:
        z = pl.LpVariable.dicts("Sommet", G.nodes, 0, 1, pl.LpInteger)
        bar.update(G.number_of_nodes())
        # x = pl.LpVariable.dicts("Arete", G.edges, 0, 1, pl.LpInteger)
        x = pl.LpVariable.dicts("Arete", [(min(e), max(e)) for e in G.edges], 0, 1, pl.LpInteger)

        bar.update(G.number_of_edges())
        # indices = [(e[0], e[1], v) for v in G.nodes for e in G.edges] + [(e[1], e[0], v) for v in G.nodes for e in G.edges]
        # indices = [(u, v, w) for u in G.nodes for v in G.nodes for w in G.nodes]
        # indices = [(e[0], e[1], w) for e in G.edges for w in G.nodes] + [(e[1], e[0], w) for e in G.edges for w in
        #                                                                  G.nodes] + [(e[0], w, e[1]) for e in G.edges
        #                                                                              for w in G.nodes]
        # indices = [(min(e), max(e), w) for e in G.edges for w in G.nodes] + [(max(e), min(e), w) for e in G.edges for w in
        #                                                                  G.nodes] + [(min(e), w, max(e)) for e in G.edges
        #                                                                              for w in G.nodes]
        indices = [(min(e), max(e), w) for e in G.edges for w in G.nodes] + [(max(e), min(e), w) for e in G.edges for w in G.nodes] + [(min(e), w, max(e)) for e in G.edges for w in G.nodes if (min(e), w) in G.edges]

        y = pl.LpVariable.dicts("Connexe", indices, 0, 1, pl.LpInteger)
        # y = pl.LpVariable.dicts("Connexe", (G.nodes, G.nodes, G.nodes), 0, 1, pl.LpInteger)
        # bar.update(G.number_of_nodes() ** 3)
        # bar.update(3 * G.number_of_edges() * G.number_of_nodes())
        bar.update(len(indices))


    # for e in G.edges:
    #     print(e)
    # Définition du problème
    if verbose:
        print("\nDéfinition du problème :")
    with tqdm(total=1, disable=not verbose) as bar:
        PLNE_Martin = pl.LpProblem("PLNE_Martin_" + instance[:-4], pl.LpMinimize)
        bar.update(1)

    # Ajout de la fonction objectif (27)
    if verbose:
        print("\nAjout de la fonction objectif (27) :")
    with tqdm(total=1, disable=not verbose) as bar:
        PLNE_Martin += pl.lpSum(z), "Objectif"
        bar.update(1)

    # Ajout des contraintes

    # Contrainte (27a)
    if verbose:
        print("\nAjout des contraintes (27a) :")
    with tqdm(total=1, disable=not verbose) as bar:
        PLNE_Martin += (
            pl.lpSum(x) == G.number_of_nodes() - 1,
            f"Contrainte (27a)",
        )
        bar.update(1)

    # Contrainte (27b)
    if verbose:
        print("\nAjout des contraintes (27b) :")
    with tqdm(total=G.number_of_nodes() * G.number_of_nodes(), disable=not verbose) as bar:
        compteur1 = 1
        for (i, j) in G.edges:
            compteur2 = 1
            for k in G.nodes:
                PLNE_Martin += (
                    y[i, j, k] + y[j, i, k] == x[min(i,j),max(i,j)],
                    f"Contrainte (27b) - [{compteur1, compteur2}]",
                )
                bar.update(1)
                compteur2 += 1
            compteur1 += 1

    # Contrainte (27c)
    if verbose:
        print("\nAjout des contraintes (27c) :")
    compteur = 1
    with tqdm(total=G.number_of_edges(), disable=not verbose) as bar:
        compteur = 1
        for i in G.nodes:
            for k in G.nodes:
                # for k in G.nodes:
                #     print((e[0],k) in G.edges)
                # PLNE_Martin += (
                #     pl.lpSum(y[min(e), k, max(e)] for k in G.nodes if k != e[0] and k != e[1]) + x[(min(e),max(e))] == 1,
                #     f"Contrainte (27c) - {compteur}",
                # )
                # PLNE_Martin += (
                #     pl.lpSum(y[i, s, k] for s in G.nodes if s != i and s != k) <= 1,
                #     f"Contrainte (27c) - {compteur}",
                # )
                PLNE_Martin += (
                    pl.lpSum(y[i, s, k] for s in G.neighbors(i)) <= 1,
                    f"Contrainte (27c) - {compteur}",
                )
                compteur += 1
                bar.update(1)

        # Contrainte (27c bis)
        if verbose:
            print("\nAjout des contraintes (27c bis) :")
        compteur = 1
        with tqdm(total=G.number_of_edges(), disable=not verbose) as bar:
            compteur = 1
            for k in G.nodes:
                # PLNE_Martin += (
                #     pl.lpSum(y[k, i, k] for i in G.nodes if i != k) <= 0,
                #     f"Contrainte (27c bis) - {compteur}",
                # )
                PLNE_Martin += (
                    pl.lpSum(y[k, i, k] for i in G.neighbors(k)) <= 0,
                    f"Contrainte (27c bis) - {compteur}",
                )
                compteur += 1
                bar.update(1)

    # Contrainte (27d)
    if verbose:
        print("\nAjout des contraintes (27d) :")
    compteur = 1
    with tqdm(total=G.number_of_nodes(), disable=not verbose) as bar:
        compteur = 1
        for i in G.nodes:
            incident_edges = list(G.edges(i))
            PLNE_Martin += (
                pl.lpSum(x[min(e), max(e)] for e in incident_edges) - G.degree(i) * z[i] <= 2,  # Le K est inutile?
                f"Contrainte (27d) - {compteur}",
            )
            # PLNE_Martin += (
            #     pl.lpSum(x[e] for e in incident_edges) - G.degree(i) * z[i] <= 2,  # Le K est inutile?
            #     f"Contrainte (27d) - {compteur}",
            # )
            bar.update(1)
            compteur += 1

    if verbose:
        print("\nEcriture du programme linéaire dans 'PLNE_Martin.lp' :")
    with tqdm(total=1, disable=not verbose) as bar:
        if not os.path.exists(f"./LP/{instance[:-4]}"):
            os.makedirs(f"./LP/{dossier}/{nom[:-4]}")
        PLNE_Martin.writeLP(f"./LP/{instance[:-4]}/PLNE_Martin_{nom[:-4]}.lp")
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
    print(f"Time Limit: {int(solver.timeLimit) // 3600}h {int((solver.timeLimit % 3600) // 60)}m {int(solver.timeLimit % 60)}s {int((solver.timeLimit % 60 - int(solver.timeLimit % 60)) * 1000)}ms ({solver.timeLimit}s).\n")

    # solver.tmpDir = "./Solutions"
    result = PLNE_Martin.solve(solver=solver)

    time_fin_solver = time.time()
    duree_solver = time_fin_solver - time_debut_solver

    if verbose:
        print(f"Résolution du programme linéaire réalisée en : {int(duree_solver) // 3600}h {int((duree_solver % 3600) // 60)}m {int(duree_solver % 60)}s {int((duree_solver % 60 - int(duree_solver % 60)) * 1000)}ms.")

    # if not os.path.exists(f"./Solutions/{instance[:-4]}"):
    #     os.makedirs(f"./Solutions/{instance[:-4]}")
    # with open(f"./Solutions/{instance[:-4]}/sol_PLNE_Martin_{instance}", "w") as fichier:
    #     fichier.write(f"Statut: {pl.LpStatus[PLNE_Martin.status]}\n")
    #     fichier.write(f"Valeur de la fonction objectif: {pl.value(PLNE_Martin.objective)}\n")
    #     fichier.write(f"Valeurs des variables:\n")
    #     for var in PLNE_Martin.variables():
    #         fichier.write(f"{var.name} = {var.varValue}\n")

    ecriture_solution(instance=instance, modele="PLNE_Martin", programme=PLNE_Martin,
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
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_1_1_1.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_200_222_3811.txt"
    instance = "Spd_Inst_Rid_Final2/Spd_RF2_400_429_4611.txt"
    # instance = "Test.txt"

    PLNE_Martin(instance, verbose=False)

