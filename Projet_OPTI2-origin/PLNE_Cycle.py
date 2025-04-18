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
import re

from Outils import *
from Draw import *


# path_to_cplex = "C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/python/3.9/x64_win64/cplex"
path_to_cplex = "C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x6q4_win64/cplex.exe"

## Code

# def reconnecter(G, G_initial, aretes_supprimees=[]):
#     liste_composante_connexe = sorted(nx.connected_components(G), key=len, reverse=True)
#     for i in range(len(liste_composante_connexe)):
#         for j in range(len(liste_composante_connexe)):
#             if liste_composante_connexe[i] != j:
#                 liste_aretes = nx.edge_boundary(G_initial, liste_composante_connexe[i], liste_composante_connexe[j])
#                 for e in liste_aretes:
#                     if e not in aretes_supprimees and (e[1], e[0]) not in aretes_supprimees:
#                         G.add_edge(e[0], e[1])
#     return G

def reconnecter_old(G, G_initial, aretes_supprimees=[]):
    n_cycle = len(nx.cycle_basis(G))
    # n_cycle = len(list(nx.simple_cycles(G)))
    for e in G_initial.edges:
        if e not in G.edges and e not in aretes_supprimees and (e[1], e[0]) not in aretes_supprimees:
            G.add_edge(e[0], e[1])
            if len(nx.cycle_basis(G)) > n_cycle:
            # if len(list(nx.simple_cycles(G))) > n_cycle:
                G.remove_edge(e[0], e[1])
    return G

def reconnecter_backup(G, G_initial):
    """ On ajoute toutes les arêtes reliant deux composantes connexes différentes."""
    ajouter_aretes = []
    for u in G.nodes:
        for v in G.nodes:
            if not nx.has_path(G, u, v) and (u, v) in G_initial.edges and (v, u) not in ajouter_aretes:
                ajouter_aretes.append((u, v))
    G.add_edges_from(ajouter_aretes)
    return G, ajouter_aretes

def reconnecter_v2(G, G_initial):
    """ On ajoute toutes les arêtes reliant deux composantes connexes différentes,
    et on renvoie en même temps le nombre de composantes connexes du graphe avant de le reconnecter."""
    ajouter_aretes = []
    for u in G.nodes:
        for v in G.nodes:
            if not nx.has_path(G, u, v) and (u, v) in G_initial.edges and (v, u) not in ajouter_aretes:
                ajouter_aretes.append((u, v))
    nb_comp_connexe = nx.number_connected_components(G)
    G.add_edges_from(ajouter_aretes)
    return G, ajouter_aretes, nb_comp_connexe

def reconnecter_v3(G, G_initial):
    """ /!\ Pas la bonne méthode. Pour chaque paire de composantes connexes, on note les arêtes qui les relient.
    On reconnecte le graphe et on indique qu'il faut au moins une arête entre chaque composante connexe,
    ainsi que le nombre de composantes connexes."""
    ajouter_aretes = []
    composantes_connexes = list(nx.connected_components(G))
    nb_comp_connexe = len(composantes_connexes)
    for i in range(nb_comp_connexe-1): # On parcourt les composantes connexes
        ajouter_aretes_tmp = []
        for j in range(i+1, nb_comp_connexe): # On parcourt les composantes connexes telles que la paire de composantes connexes n'ait pas encore été étudiée
            for u in composantes_connexes[i]:
                for v in composantes_connexes[j]:
                    if (u,v) in G_initial.edges and not (v, u) in ajouter_aretes: # On rajoute les arêtes entre les deux composantes connexes
                        ajouter_aretes_tmp.append((u, v))
        G.add_edges_from(ajouter_aretes_tmp)
        ajouter_aretes.append(ajouter_aretes_tmp) # On ajoute la contrainte qu'il doit y avoir au moins une arête entre les deux composantes connexes
    return G, ajouter_aretes, nb_comp_connexe

def reconnecter_final(G, G_initial):
    """ On reconnecte le graphe et on ajoute qu'une composante connexe doit être connectée avec au moins une autre composante."""
    ajouter_aretes = []
    # composantes_connexes = list(nx.connected_components(G))
    # nb_comp_connexe = len(composantes_connexes)
    nb_comp_connexe = 0
    for composante in nx.connected_components(G): # On parcourt les composantes connexes
        nb_comp_connexe += 1
        ajouter_aretes_tmp = []
        for u, v in G_initial.edges(composante):
            if not(u in composante and v in composante):
                ajouter_aretes_tmp.append((u, v))
        G.add_edges_from(ajouter_aretes_tmp)
        if len(ajouter_aretes_tmp) > 0:
            ajouter_aretes.append(ajouter_aretes_tmp) # On ajoute la contrainte qu'il doit y avoir au moins une arête qui entre/sort dans la composante connexe
    return G, ajouter_aretes, nb_comp_connexe

def PLNE_cycle_old(instance, maxiter = 1e3, verbose=False):
    dossier, nom = instance.split('/')
    G = creation_graphe(instance, False)
    # plt.title("Graphe initial")
    # nx.draw(G, with_labels=True)
    # plt.show()
    G_cycle = nx.Graph(G)
    base_cycle = nx.cycle_basis(G_cycle)
    n_cycle = len(base_cycle)

    # Cinquième programme : PLNE Cycle

    convergence = False
    iteration = 0
    time_debut_solver = time.time()
    aretes_supprimees = [] # Fonctionne pas
    # banque de cycles supprimés
    while not convergence and iteration < maxiter:
        print(f"Itération: {iteration} (itération maximale : {maxiter})")
        # aretes_supprimees = []
        # print(aretes_supprimees)
        time_debut_ecriture = time.time()

        # base_cycle = nx.cycle_basis(G_cycle)
        # n_cycle = len(base_cycle)
        # Définition des variables
        if verbose:
            print("Définition des variables :")
        with tqdm(total=G.number_of_nodes() + G.number_of_edges(), disable=not verbose) as bar:
            y = pl.LpVariable.dicts("Sommet", G_cycle.nodes, 0, 1, pl.LpInteger)
            bar.update(G.number_of_nodes())
            x = pl.LpVariable.dicts("Arete", G_cycle.edges, 0, 1, pl.LpInteger)
            bar.update(G.number_of_edges())

        # Définition du problème
        if verbose:
            print("\nDéfinition du problème :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle = pl.LpProblem("PLNE_Cycle_" + instance[:-4], pl.LpMinimize)
            bar.update(1)

        # Ajout de la fonction objectif (1)
        if verbose:
            print("\nAjout de la fonction objectif (1) :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle += pl.lpSum(y), "Objectif"
            bar.update(1)

        # Ajout des contraintes

        # Contrainte (2) : Nombre d'arêtes dans la solution = |V|-1
        if verbose:
            print("\nAjout des contraintes (2) :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle += (
                pl.lpSum(x) == G.number_of_nodes() - 1,
                f"Contrainte (2)",
            )
            bar.update(1)

        # if verbose:
        #     print("\nAjout des contraintes (2bis) :")
        #     compteur = 1
        # with tqdm(total=1, disable=not verbose) as bar:
        #     for e in G.edges():
        #         if e not in G_cycle.edges():
        #             PLNE_Cycle += (
        #                 x[e] == 0,
        #                 f"Contrainte (2bis) - {compteur}",
        #             )
        #             compteur += 1
        #     bar.update(1)
        # Contrainte (3) : On éclate les bases de cycles
        if verbose:
            print("\nAjout des contraintes (3) :")
        with tqdm(total=n_cycle, disable=not verbose) as bar:
            compteur = 1
            for c in base_cycle:
                e = [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]
                # print(c)
                # print(e)
                PLNE_Cycle += (
                    pl.lpSum(x[min(e[i][0], e[i][1]), max(e[i][0], e[i][1])] for i in range(len(c))) <= len(c) - 1,
                    f"Contrainte (3) - [{compteur}]",
                )
                bar.update(1)
                compteur += 1

        compteur = 1
        for i in G.nodes:  # On affecte 1 à y[i] si i est un noeud de branchement et 0 sinon
            incident_edges = list(G_cycle.edges(i))
            PLNE_Cycle += (
                pl.lpSum(x[min(e), max(e)] for e in incident_edges) - G.degree(i) * y[i] <= 2,
                f"Contrainte (4) - {compteur}",
            )
            compteur += 1

        # if verbose:
        #     print(f"\nEcriture du programme linéaire dans 'PLNE_Cycle_{iteration}.lp' :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle.writeLP(f"./LP/{dossier}/{nom[:-4]}/PLNE_Cycle_{iteration + 1}_{nom[:-4]}.lp")
            bar.update(1)
        time_fin_ecriture = time.time()
        duree_ecriture = time_fin_ecriture - time_debut_ecriture

        # if verbose:
        #     print(f"\nDéfinition et écriture du programme linéaire réalisées en : {int(duree_ecriture) // 3600}h {int((duree_ecriture % 3600) // 60)}m {int(duree_ecriture % 60)}s {int((duree_ecriture % 60 - int(duree_ecriture % 60)) * 1000)}ms.")

        # Affichage du résultat

        # time_debut_solver = time.time()

        if os.path.exists("./solver_Cplex.json"):
            solver = pl.getSolverFromJson("./solver_Cplex.json")
        else:
            solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
            solver.toJson("solver_Cplex.json")
        solver.msg = verbose

        # solver.tmpDir = "./Solutions"
        result = PLNE_Cycle.solve(solver=solver)

        if verbose:
            for var in PLNE_Cycle.variables():
                print(f'{var.name}: {var.varValue}')
        for var in PLNE_Cycle.variables():
            # if "__dummy" not in var.name:
            if "Arete" in var.name:
                nombres = re.findall(r'\d+', var.name)
                # Extraire les deux premiers nombres
                if len(nombres) >= 2:
                    u, v = int(nombres[0]), int(nombres[1])
                else:
                    raise ValueError("Le nom de la variable ne contient pas suffisamment de nombres.")
                if abs(var.varValue) <= 0.2: # Arete pas dans la solution
                    if (u,v) in G_cycle.edges:
                        G_cycle.remove_edge(u,v)
                    for c in base_cycle:
                        if (u,v) in [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]:
                            aretes_supprimees.append((u, v))
                            break
        for e in aretes_supprimees:
            n_cycle_avant = len(nx.cycle_basis(G_cycle))
            G_cycle.add_edge(e[0], e[1])
            n_cycle_apres = len(nx.cycle_basis(G_cycle))
            G_cycle.remove_edge(e[0], e[1])
            if n_cycle_avant == n_cycle_apres:
                aretes_supprimees.remove(e)
        # print(aretes_supprimees)

        plt.title(f"Cycles éclatés à l'itération {iteration + 1}")
        nx.draw(G_cycle, with_labels=True)
        plt.show()

        G_cycle = reconnecter_old(G=G_cycle, G_initial=G, aretes_supprimees=aretes_supprimees)

        plt.title(f"Graphe reconnecté à l'itération {iteration + 1}")
        nx.draw(G_cycle, with_labels=True)
        plt.show()
        # nx.draw(G_cycle, labels= {i: i for i in range(1,G.number_of_nodes()+1)})
        # plt.show()
        # print(G_cycle.nodes)
        # print(G_cycle.edges)

        base_cycle = nx.cycle_basis(G_cycle)
        n_cycle = len(base_cycle)
        iteration += 1

        # n_composante_connexe = nx.connected_components(G)
        # print(f"arretes supprimees: {aretes_supprimees}")
        # print(G_cycle.edges)
        # print(f"n_cycle: {n_cycle}, G_cycle.number_of_nodes: {G.number_of_nodes()}, G_cycle.edges: {G.number_of_edges()}")
        if n_cycle == 0 and G_cycle.number_of_edges() == G_cycle.number_of_nodes() - 1:
            convergence = True

    # print(G_cycle.nodes)
    # print(G_cycle.edges)


    # Définition d'un PL annexe qui réécrit "proprement" la méthode avec les bases de cycles en solution d'un PLNE classique
    y = pl.LpVariable.dicts("Sommet", G.nodes, 0, 1, pl.LpInteger)
    bar.update(G.number_of_nodes())
    x = pl.LpVariable.dicts("Arete", G.edges, 0, 1, pl.LpInteger)
    bar.update(G.number_of_edges())

    PLNE_Cycle_final = pl.LpProblem("PLNE_Cycle_" + instance[:-4], pl.LpMinimize)
    bar.update(1)

    PLNE_Cycle_final += pl.lpSum(y), "Objectif"
    # print(nx.is_tree(G_cycle))
    compteur = 1
    for e in G.edges(): # On affecte 1 à x[e] si e est dans l'arbre final et 0 sinon
        # print(e)
        # print(G_cycle.edges)
        # print(e in G_cycle.edges)
        PLNE_Cycle_final += (
            x[e] == int(e in G_cycle.edges or (e[1], e[0]) in G_cycle.edges),
            f"Contrainte (2) - {compteur}",
        )
        compteur += 1

    compteur = 1
    for i in G.nodes: # On affecte 1 à y[i] si i est un noeud de branchement et 0 sinon
        incident_edges = list(G.edges(i))
        PLNE_Cycle_final += (
            pl.lpSum(x[min(e), max(e)] for e in incident_edges) - G.degree(i) * y[i] <= 2,
            f"Contrainte (3) - {compteur}",
        )
        compteur += 1

    time_fin_solver = time.time()
    duree_solver = time_fin_solver - time_debut_solver

    PLNE_Cycle_final.writeLP(f"./LP/{dossier}/{nom[:-4]}/PLNE_Cycle_{nom[:-4]}.lp")

    if os.path.exists("./solver_Cplex.json"):
        solver = pl.getSolverFromJson("./solver_Cplex.json")
    else:
        solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
        solver.toJson("solver_Cplex.json")
    solver.msg = verbose
    solver.timeLimit = set_time_limit(G.number_of_nodes(), G.number_of_edges())

    # solver.tmpDir = "./Solutions"
    result = PLNE_Cycle_final.solve(solver=solver)

    if verbose:
        print(f"Résolution du programme linéaire réalisée en : {int(duree_solver) // 3600}h {int((duree_solver % 3600) // 60)}m {int(duree_solver % 60)}s {int((duree_solver % 60 - int(duree_solver % 60)) * 1000)}ms.")

    ecriture_solution(instance=instance, modele="PLNE_Cycle", programme=PLNE_Cycle_final,
                      duree_ecriture=duree_ecriture, duree_solver=duree_solver)

    liste_fichier = os.listdir()
    for fichier in liste_fichier:
        if len(fichier) >= 5 and fichier[:5] == "clone":
            if fichier in os.listdir("./log_Cplex/"):
                os.remove(f"./log_Cplex/{fichier}")
            shutil.move(fichier, f"./log_Cplex/{fichier}")
    print(result)

    if iteration == maxiter and not convergence and verbose:
        print("L'algorithme n'a pas convergé vers une solution optimale.")
    elif convergence and verbose:
        print(f"L'algorithme a convergé vers une solution optimale en {iteration} itérations.")

def PLNE_cycle_v2(instance, maxiter=1e3, verbose=False):
    dossier, nom = instance.split('/')
    G = creation_graphe(instance, False)
    # nx.draw(G, with_labels=True)
    # plt.show()
    G_cycle = nx.Graph(G)
    base_cycle = nx.cycle_basis(G_cycle)
    n_cycle = len(base_cycle)

    # Cinquième programme : PLNE Cycle

    convergence = False
    iteration = 0
    time_debut_solver = time.time()
    # aretes_supprimees = [] # Fonctionne pas
    ajouter_aretes = []
    # nb_comp_connexe = 1
    # banque de cycles supprimés
    while not convergence and iteration < maxiter:
        print(f"Itération: {iteration} (itération maximale : {maxiter})")
        aretes_supprimees = []
        # print(aretes_supprimees)
        time_debut_ecriture = time.time()

        # base_cycle = nx.cycle_basis(G_cycle)
        # n_cycle = len(base_cycle)
        # Définition des variables
        if verbose:
            print("Définition des variables :")
        with tqdm(total=G.number_of_nodes() + G.number_of_edges(), disable=not verbose) as bar:
            y = pl.LpVariable.dicts("Sommet", G_cycle.nodes, 0, 1, pl.LpInteger)
            bar.update(G.number_of_nodes())
            x = pl.LpVariable.dicts("Arete", G_cycle.edges, 0, 1, pl.LpInteger)
            bar.update(G.number_of_edges())

        # Définition du problème
        if verbose:
            print("\nDéfinition du problème :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle = pl.LpProblem("PLNE_Cycle_" + instance[:-4], pl.LpMinimize)
            bar.update(1)

        # Ajout de la fonction objectif (1)
        if verbose:
            print("\nAjout de la fonction objectif (1) :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle += pl.lpSum(y), "Objectif"
            bar.update(1)

        # Ajout des contraintes

        # Contrainte (2) : Nombre d'arêtes dans la solution = |V|-1
        if verbose:
            print("\nAjout des contraintes (2) :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle += (
                pl.lpSum(x) == G.number_of_nodes() - 1,
                f"Contrainte (2)",
            )
            bar.update(1)

        # if verbose:
        #     print("\nAjout des contraintes (2bis) :")
        #     compteur = 1
        # with tqdm(total=1, disable=not verbose) as bar:
        #     for e in G.edges():
        #         if e not in G_cycle.edges():
        #             PLNE_Cycle += (
        #                 x[e] == 0,
        #                 f"Contrainte (2bis) - {compteur}",
        #             )
        #             compteur += 1
        #     bar.update(1)
        # Contrainte (3) : On éclate les bases de cycles
        if verbose:
            print("\nAjout des contraintes (3) :")
        with tqdm(total=n_cycle, disable=not verbose) as bar:
            compteur = 1
            for c in base_cycle:
                e = [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]
                # print(c)
                # print(e)
                PLNE_Cycle += (
                    pl.lpSum(x[min(e[i]), max(e[i])] for i in range(len(c))) <= len(c) - 1,
                    f"Contrainte (3) - [{compteur}]",
                )
                bar.update(1)
                compteur += 1

        compteur = 1
        for i in G.nodes:  # On affecte 1 à y[i] si i est un noeud de branchement et 0 sinon
            incident_edges = list(G_cycle.edges(i))
            PLNE_Cycle += (
                pl.lpSum(x[min(e), max(e)] for e in incident_edges) - G.degree(i) * y[i] <= 2,
                f"Contrainte (4) - {compteur}",
            )
            compteur += 1

        # Contrainte (5) : Pour chaque coupe obtenue aux itérations précédentes, on garde au moins n-1 arêtes où n est le nombre de composantes connexes associé à la coupe
        if verbose:
            print("\nAjout des contraintes (5) :")
        with tqdm(total=len(ajouter_aretes), disable=not verbose) as bar:
            compteur = 1
            # print(ajouter_aretes)
            for coupe, nb_comp_connexe in ajouter_aretes:
                # print(e)
                # print(f"Nb composantes connexes: {nb_comp_connexe}, coupe: {coupe}")
                # l = [(min(e[i][0], e[i][1]), max(e[i][0], e[i][1])) for i in range(len(e)) if e[i] in G_cycle.edges]
                l = [(min(coupe[i]), max(coupe[i])) for i in range(len(coupe)) if coupe[i] in G_cycle.edges]
                if len(l) > 0:
                    PLNE_Cycle += (
                        pl.lpSum(x[min(e), max(e)] for e in l) >= nb_comp_connexe - 1,
                        f"Contrainte (5) - [{iteration} - {compteur}]",
                    )
                    compteur += 1
            bar.update(1)



        if verbose:
            print(f"\nEcriture du programme linéaire dans 'PLNE_Cycle_{iteration}.lp' :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle.writeLP(f"./LP/{dossier}/{nom[:-4]}/PLNE_Cycle_{iteration + 1}_{nom[:-4]}.lp")
            bar.update(1)
        time_fin_ecriture = time.time()
        duree_ecriture = time_fin_ecriture - time_debut_ecriture

        # if verbose:
        #     print(f"\nDéfinition et écriture du programme linéaire réalisées en : {int(duree_ecriture) // 3600}h {int((duree_ecriture % 3600) // 60)}m {int(duree_ecriture % 60)}s {int((duree_ecriture % 60 - int(duree_ecriture % 60)) * 1000)}ms.")

        # Affichage du résultat

        # time_debut_solver = time.time()

        if os.path.exists("./solver_Cplex.json"):
            solver = pl.getSolverFromJson("./solver_Cplex.json")
        else:
            solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
            solver.toJson("solver_Cplex.json")
        solver.msg = verbose

        # solver.tmpDir = "./Solutions"
        result = PLNE_Cycle.solve(solver=solver)
        if verbose:
            for var in PLNE_Cycle.variables():
                print(f'{var.name}: {var.varValue}')
        for var in PLNE_Cycle.variables():
            # if "__dummy" not in var.name:
            if "Arete" in var.name:
                nombres = re.findall(r'\d+', var.name)
                # Extraire les deux premiers nombres
                if len(nombres) >= 2:
                    u, v = int(nombres[0]), int(nombres[1])
                else:
                    raise ValueError("Le nom de la variable ne contient pas suffisamment de nombres.")
                if abs(var.varValue) <= 0.2:  # Arete pas dans la solution
                    if (u, v) in G_cycle.edges:
                        G_cycle.remove_edge(u, v)
                    for c in base_cycle:
                        if (u, v) in [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]:
                            aretes_supprimees.append((u, v))
                            break
        # for e in aretes_supprimees:
        #     n_cycle_avant = len(nx.cycle_basis(G_cycle))
        #     G_cycle.add_edge(e[0], e[1])
        #     n_cycle_apres = len(nx.cycle_basis(G_cycle))
        #     G_cycle.remove_edge(e[0], e[1])
        #     if n_cycle_avant == n_cycle_apres:
        #         aretes_supprimees.remove(e)
        # print(aretes_supprimees)

        # plt.title(f"Cycles éclatés à l'itération {iteration + 1}")
        # nx.draw(G_cycle, with_labels=True)
        # plt.show()
        G_cycle, ajouter_aretes_tmp, nb_comp_connexe = reconnecter_v2(G=G_cycle, G_initial=G)
        ajouter_aretes.append((ajouter_aretes_tmp, nb_comp_connexe))
        # print(f"G_cycle.number_of_edges() = {G_cycle.number_of_edges()}")
        # print(f"G.number_of_edges() = {G.number_of_edges()}")
        # plt.title(f"Graphe reconnecté à l'itération {iteration+1}")
        # nx.draw(G_cycle, with_labels=True)
        # plt.show()
        # nx.draw(G_cycle, labels= {i: i for i in range(1,G.number_of_nodes()+1)})
        # plt.show()
        # print(G_cycle.nodes)
        # print(G_cycle.edges)

        base_cycle = nx.cycle_basis(G_cycle)
        n_cycle = len(base_cycle)
        iteration += 1
        # n_composante_connexe = nx.connected_components(G)
        # print(f"arretes supprimees: {aretes_supprimees}")
        # print(G_cycle.edges)
        # print(f"n_cycle: {n_cycle}, G_cycle.number_of_nodes: {G.number_of_nodes()}, G_cycle.edges: {G.number_of_edges()}")
        if n_cycle == 0 and G_cycle.number_of_edges() == G_cycle.number_of_nodes() - 1:
            convergence = True

    # print(G_cycle.nodes)
    # print(G_cycle.edges)

    # Définition d'un PL annexe qui réécrit "proprement" la méthode avec les bases de cycles en solution d'un PLNE classique
    y = pl.LpVariable.dicts("Sommet", G.nodes, 0, 1, pl.LpInteger)
    bar.update(G.number_of_nodes())
    x = pl.LpVariable.dicts("Arete", G.edges, 0, 1, pl.LpInteger)
    bar.update(G.number_of_edges())

    PLNE_Cycle_final = pl.LpProblem("PLNE_Cycle_" + instance[:-4], pl.LpMinimize)
    bar.update(1)

    PLNE_Cycle_final += pl.lpSum(y), "Objectif"

    compteur = 1
    for e in G.edges():  # On affecte 1 à x[e] si e est dans l'arbre final et 0 sinon
        # print(e)
        # print(G_cycle.edges)
        # print(e in G_cycle.edges)
        PLNE_Cycle_final += (
            x[e] == int(e in G_cycle.edges or (e[1], e[0]) in G_cycle.edges),
            f"Contrainte (2) - {compteur}",
        )
        compteur += 1

    compteur = 1
    for i in G.nodes:  # On affecte 1 à y[i] si i est un noeud de branchement et 0 sinon
        incident_edges = list(G.edges(i))
        PLNE_Cycle_final += (
            pl.lpSum(x[min(e), max(e)] for e in incident_edges) - G.degree(i) * y[i] <= 2,
            f"Contrainte (3) - {compteur}",
        )
        compteur += 1

    time_fin_solver = time.time()
    duree_solver = time_fin_solver - time_debut_solver

    PLNE_Cycle_final.writeLP(f"./LP/{dossier}/{nom[:-4]}/PLNE_Cycle_{nom[:-4]}.lp")

    if os.path.exists("./solver_Cplex.json"):
        solver = pl.getSolverFromJson("./solver_Cplex.json")
    else:
        solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
        solver.toJson("solver_Cplex.json")
    solver.msg = verbose
    solver.timeLimit = set_time_limit(G.number_of_nodes(), G.number_of_edges())

    # solver.tmpDir = "./Solutions"
    result = PLNE_Cycle_final.solve(solver=solver)

    if verbose:
        print(f"Résolution du programme linéaire réalisée en : {int(duree_solver) // 3600}h {int((duree_solver % 3600) // 60)}m {int(duree_solver % 60)}s {int((duree_solver % 60 - int(duree_solver % 60)) * 1000)}ms.")

    ecriture_solution(instance=instance, modele="PLNE_Cycle", programme=PLNE_Cycle_final,
                      duree_ecriture=duree_ecriture, duree_solver=duree_solver)

    liste_fichier = os.listdir()
    for fichier in liste_fichier:
        if len(fichier) >= 5 and fichier[:5] == "clone":
            if fichier in os.listdir("./log_Cplex/"):
                os.remove(f"./log_Cplex/{fichier}")
            shutil.move(fichier, f"./log_Cplex/{fichier}")
    print(result)

    if iteration == maxiter and not convergence and verbose:
        print("L'algorithme n'a pas convergé vers une solution optimale.")
    elif convergence and verbose:
        print(f"L'algorithme a convergé vers une solution optimale en {iteration} itérations.")

def PLNE_cycle_backup(instance, maxiter=1e3, verbose=False):
    dossier, nom = instance.split('/')
    G = creation_graphe(instance, False)
    # nx.draw(G, with_labels=True)
    # plt.show()
    G_cycle = nx.Graph(G)
    base_cycle = nx.cycle_basis(G_cycle)
    n_cycle = len(base_cycle)

    # Cinquième programme : PLNE Cycle

    convergence = False
    iteration = 0
    time_debut_solver = time.time()
    # aretes_supprimees = [] # Fonctionne pas
    ajouter_aretes = []
    # banque de cycles supprimés
    while not convergence and iteration < maxiter:
        print(f"Itération: {iteration} (itération maximale : {maxiter})")
        aretes_supprimees = []
        # print(aretes_supprimees)
        time_debut_ecriture = time.time()

        # base_cycle = nx.cycle_basis(G_cycle)
        # n_cycle = len(base_cycle)
        # Définition des variables
        if verbose:
            print("Définition des variables :")
        with tqdm(total=G.number_of_nodes() + G.number_of_edges(), disable=not verbose) as bar:
            y = pl.LpVariable.dicts("Sommet", G_cycle.nodes, 0, 1, pl.LpInteger)
            bar.update(G.number_of_nodes())
            x = pl.LpVariable.dicts("Arete", G_cycle.edges, 0, 1, pl.LpInteger)
            bar.update(G.number_of_edges())

        # Définition du problème
        if verbose:
            print("\nDéfinition du problème :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle = pl.LpProblem("PLNE_Cycle_" + instance[:-4], pl.LpMinimize)
            bar.update(1)

        # Ajout de la fonction objectif (1)
        if verbose:
            print("\nAjout de la fonction objectif (1) :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle += pl.lpSum(y), "Objectif"
            bar.update(1)

        # Ajout des contraintes

        # Contrainte (2) : Nombre d'arêtes dans la solution = |V|-1
        if verbose:
            print("\nAjout des contraintes (2) :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle += (
                pl.lpSum(x) == G.number_of_nodes() - 1,
                f"Contrainte (2)",
            )
            bar.update(1)

        # if verbose:
        #     print("\nAjout des contraintes (2bis) :")
        #     compteur = 1
        # with tqdm(total=1, disable=not verbose) as bar:
        #     for e in G.edges():
        #         if e not in G_cycle.edges():
        #             PLNE_Cycle += (
        #                 x[e] == 0,
        #                 f"Contrainte (2bis) - {compteur}",
        #             )
        #             compteur += 1
        #     bar.update(1)
        # Contrainte (3) : On éclate les bases de cycles
        if verbose:
            print("\nAjout des contraintes (3) :")
        with tqdm(total=n_cycle, disable=not verbose) as bar:
            compteur = 1
            for c in base_cycle:
                e = [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]
                # print(c)
                # print(e)
                PLNE_Cycle += (
                    pl.lpSum(x[min(e[i]), max(e[i])] for i in range(len(c))) <= len(c) - 1,
                    f"Contrainte (3) - [{compteur}]",
                )
                bar.update(1)
                compteur += 1

        compteur = 1
        for i in G.nodes:  # On affecte 1 à y[i] si i est un noeud de branchement et 0 sinon
            incident_edges = list(G_cycle.edges(i))
            PLNE_Cycle += (
                pl.lpSum(x[min(e), max(e)] for e in incident_edges) - G.degree(i) * y[i] <= 2,
                f"Contrainte (4) - {compteur}",
            )
            compteur += 1

        # Contrainte (5) : On garde au moins une arête entre chaque composante connexe obtenue à l'itération précédente
        if verbose:
            print("\nAjout des contraintes (5) :")
        with tqdm(total=len(ajouter_aretes), disable=not verbose) as bar:
            compteur = 1
            for e in ajouter_aretes:
                # print(e)
                l = [(min(e[i][0], e[i][1]), max(e[i][0], e[i][1])) for i in range(len(e)) if e[i] in G_cycle.edges]
                if len(l) > 0:
                    PLNE_Cycle += (
                        pl.lpSum(x[min(a[0], a[1]), max(a[0], a[1])] for a in l) >= 1,
                        f"Contrainte (5) - [{iteration} - {compteur}]",
                    )
                    compteur += 1
            bar.update(1)



        # if verbose:
        #     print(f"\nEcriture du programme linéaire dans 'PLNE_Cycle_{iteration}.lp' :")
        # with tqdm(total=1, disable=not verbose) as bar:
        #     PLNE_Cycle.writeLP(f"./LP/{dossier}/{nom[:-4]}/PLNE_Cycle_{iteration + 1}_{nom[:-4]}.lp")
        #     bar.update(1)
        time_fin_ecriture = time.time()
        duree_ecriture = time_fin_ecriture - time_debut_ecriture

        # if verbose:
        #     print(f"\nDéfinition et écriture du programme linéaire réalisées en : {int(duree_ecriture) // 3600}h {int((duree_ecriture % 3600) // 60)}m {int(duree_ecriture % 60)}s {int((duree_ecriture % 60 - int(duree_ecriture % 60)) * 1000)}ms.")

        # Affichage du résultat

        # time_debut_solver = time.time()

        if os.path.exists("./solver_Cplex.json"):
            solver = pl.getSolverFromJson("./solver_Cplex.json")
        else:
            solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
            solver.toJson("solver_Cplex.json")
        solver.msg = verbose

        # solver.tmpDir = "./Solutions"
        result = PLNE_Cycle.solve(solver=solver)
        if verbose:
            for var in PLNE_Cycle.variables():
                print(f'{var.name}: {var.varValue}')
        for var in PLNE_Cycle.variables():
            # if "__dummy" not in var.name:
            if "Arete" in var.name:
                nombres = re.findall(r'\d+', var.name)
                # Extraire les deux premiers nombres
                if len(nombres) >= 2:
                    u, v = int(nombres[0]), int(nombres[1])
                else:
                    raise ValueError("Le nom de la variable ne contient pas suffisamment de nombres.")
                if abs(var.varValue) <= 0.2:  # Arete pas dans la solution
                    if (u, v) in G_cycle.edges:
                        G_cycle.remove_edge(u, v)
                    for c in base_cycle:
                        if (u, v) in [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]:
                            aretes_supprimees.append((u, v))
                            break
        # for e in aretes_supprimees:
        #     n_cycle_avant = len(nx.cycle_basis(G_cycle))
        #     G_cycle.add_edge(e[0], e[1])
        #     n_cycle_apres = len(nx.cycle_basis(G_cycle))
        #     G_cycle.remove_edge(e[0], e[1])
        #     if n_cycle_avant == n_cycle_apres:
        #         aretes_supprimees.remove(e)
        # print(aretes_supprimees)

        # plt.title(f"Cycles éclatés à l'itération {iteration + 1}")
        # nx.draw(G_cycle, with_labels=True)
        # plt.show()
        G_cycle, ajouter_aretes_tmp = reconnecter_backup(G=G_cycle, G_initial=G)
        ajouter_aretes.append(ajouter_aretes_tmp)
        # print(f"G_cycle.number_of_edges() = {G_cycle.number_of_edges()}")
        # print(f"G.number_of_edges() = {G.number_of_edges()}")
        # plt.title(f"Graphe reconnecté à l'itération {iteration+1}")
        # nx.draw(G_cycle, with_labels=True)
        # plt.show()
        # nx.draw(G_cycle, labels= {i: i for i in range(1,G.number_of_nodes()+1)})
        # plt.show()
        # print(G_cycle.nodes)
        # print(G_cycle.edges)

        base_cycle = nx.cycle_basis(G_cycle)
        n_cycle = len(base_cycle)
        iteration += 1
        # n_composante_connexe = nx.connected_components(G)
        # print(f"arretes supprimees: {aretes_supprimees}")
        # print(G_cycle.edges)
        # print(f"n_cycle: {n_cycle}, G_cycle.number_of_nodes: {G.number_of_nodes()}, G_cycle.edges: {G.number_of_edges()}")
        if n_cycle == 0 and G_cycle.number_of_edges() == G_cycle.number_of_nodes() - 1:
            convergence = True

    # print(G_cycle.nodes)
    # print(G_cycle.edges)

    # Définition d'un PL annexe qui réécrit "proprement" la méthode avec les bases de cycles en solution d'un PLNE classique
    y = pl.LpVariable.dicts("Sommet", G.nodes, 0, 1, pl.LpInteger)
    bar.update(G.number_of_nodes())
    x = pl.LpVariable.dicts("Arete", G.edges, 0, 1, pl.LpInteger)
    bar.update(G.number_of_edges())

    PLNE_Cycle_final = pl.LpProblem("PLNE_Cycle_" + instance[:-4], pl.LpMinimize)
    bar.update(1)

    PLNE_Cycle_final += pl.lpSum(y), "Objectif"

    compteur = 1
    for e in G.edges():  # On affecte 1 à x[e] si e est dans l'arbre final et 0 sinon
        # print(e)
        # print(G_cycle.edges)
        # print(e in G_cycle.edges)
        PLNE_Cycle_final += (
            x[e] == int(e in G_cycle.edges or (e[1], e[0]) in G_cycle.edges),
            f"Contrainte (2) - {compteur}",
        )
        compteur += 1

    compteur = 1
    for i in G.nodes:  # On affecte 1 à y[i] si i est un noeud de branchement et 0 sinon
        incident_edges = list(G.edges(i))
        PLNE_Cycle_final += (
            pl.lpSum(x[min(e), max(e)] for e in incident_edges) - G.degree(i) * y[i] <= 2,
            f"Contrainte (3) - {compteur}",
        )
        compteur += 1

    time_fin_solver = time.time()
    duree_solver = time_fin_solver - time_debut_solver

    PLNE_Cycle_final.writeLP(f"./LP/{dossier}/{nom[:-4]}/PLNE_Cycle_{nom[:-4]}.lp")

    if os.path.exists("./solver_Cplex.json"):
        solver = pl.getSolverFromJson("./solver_Cplex.json")
    else:
        solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
        solver.toJson("solver_Cplex.json")
    solver.msg = verbose
    solver.timeLimit = set_time_limit(G.number_of_nodes(), G.number_of_edges())

    # solver.tmpDir = "./Solutions"
    result = PLNE_Cycle_final.solve(solver=solver)

    if verbose:
        print(f"Résolution du programme linéaire réalisée en : {int(duree_solver) // 3600}h {int((duree_solver % 3600) // 60)}m {int(duree_solver % 60)}s {int((duree_solver % 60 - int(duree_solver % 60)) * 1000)}ms.")

    ecriture_solution(instance=instance, modele="PLNE_Cycle_backup", programme=PLNE_Cycle_final,
                      duree_ecriture=duree_ecriture, duree_solver=duree_solver)

    liste_fichier = os.listdir()
    for fichier in liste_fichier:
        if len(fichier) >= 5 and fichier[:5] == "clone":
            if fichier in os.listdir("./log_Cplex/"):
                os.remove(f"./log_Cplex/{fichier}")
            shutil.move(fichier, f"./log_Cplex/{fichier}")
    print(result)

    if iteration == maxiter and not convergence and verbose:
        print("L'algorithme n'a pas convergé vers une solution optimale.")
    elif convergence and verbose:
        print(f"L'algorithme a convergé vers une solution optimale en {iteration} itérations.")

def PLNE_cycle_v3(instance, maxiter=1e3, verbose=False):
    dossier, nom = instance.split('/')
    G = creation_graphe(instance, False)
    # nx.draw(G, with_labels=True)
    # plt.show()
    G_cycle = nx.Graph(G)
    base_cycle = nx.cycle_basis(G_cycle)
    n_cycle = len(base_cycle)

    # Cinquième programme : PLNE Cycle

    convergence = False
    iteration = 0
    time_debut_solver = time.time()
    # aretes_supprimees = [] # Fonctionne pas
    ajouter_aretes = []
    # nb_comp_connexe = 1
    # banque de cycles supprimés
    while not convergence and iteration < maxiter:
        print(f"Itération: {iteration+1} (itération maximale : {maxiter})")
        aretes_supprimees = []
        # print(aretes_supprimees)
        time_debut_ecriture = time.time()

        # base_cycle = nx.cycle_basis(G_cycle)
        # n_cycle = len(base_cycle)
        # Définition des variables
        if verbose:
            print("Définition des variables :")
        with tqdm(total=G.number_of_nodes() + G.number_of_edges(), disable=not verbose) as bar:
            y = pl.LpVariable.dicts("Sommet", G_cycle.nodes, 0, 1, pl.LpInteger)
            bar.update(G.number_of_nodes())
            x = pl.LpVariable.dicts("Arete", G_cycle.edges, 0, 1, pl.LpInteger)
            bar.update(G.number_of_edges())

        # Définition du problème
        if verbose:
            print("\nDéfinition du problème :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle = pl.LpProblem("PLNE_Cycle_" + instance[:-4], pl.LpMinimize)
            bar.update(1)

        # Ajout de la fonction objectif (1)
        if verbose:
            print("\nAjout de la fonction objectif (1) :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle += pl.lpSum(y), "Objectif"
            bar.update(1)

        # Ajout des contraintes

        # Contrainte (2) : Nombre d'arêtes dans la solution = |V|-1
        if verbose:
            print("\nAjout des contraintes (2) :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle += (
                pl.lpSum(x) == G.number_of_nodes() - 1,
                f"Contrainte (2)",
            )
            bar.update(1)

        # if verbose:
        #     print("\nAjout des contraintes (2bis) :")
        #     compteur = 1
        # with tqdm(total=1, disable=not verbose) as bar:
        #     for e in G.edges():
        #         if e not in G_cycle.edges():
        #             PLNE_Cycle += (
        #                 x[e] == 0,
        #                 f"Contrainte (2bis) - {compteur}",
        #             )
        #             compteur += 1
        #     bar.update(1)
        # Contrainte (3) : On éclate les bases de cycles
        if verbose:
            print("\nAjout des contraintes (3) :")
        with tqdm(total=n_cycle, disable=not verbose) as bar:
            compteur = 1
            for c in base_cycle:
                e = [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]
                # print(c)
                # print(e)
                PLNE_Cycle += (
                    pl.lpSum(x[min(e[i]), max(e[i])] for i in range(len(c))) <= len(c) - 1,
                    f"Contrainte (3) - [{compteur}]",
                )
                bar.update(1)
                compteur += 1

        compteur = 1
        for i in G.nodes:  # On affecte 1 à y[i] si i est un noeud de branchement et 0 sinon
            incident_edges = list(G_cycle.edges(i))
            PLNE_Cycle += (
                pl.lpSum(x[min(e), max(e)] for e in incident_edges) - G.degree(i) * y[i] <= 2,
                f"Contrainte (4) - {compteur}",
            )
            compteur += 1

        # Contrainte (5) : Pour chaque coupe obtenue aux itérations précédentes, on garde au moins n-1 arêtes où n est le nombre de composantes connexes associé à la coupe
        if verbose:
            print("\nAjout des contraintes (5) :")
        with tqdm(total=len(ajouter_aretes), disable=not verbose) as bar:
            compteur = 1
            # print(ajouter_aretes)
            for coupe in ajouter_aretes:
                # print(e)
                # print(f"Nb composantes connexes: {nb_comp_connexe}, coupe: {coupe}")
                # l = [(min(e[i][0], e[i][1]), max(e[i][0], e[i][1])) for i in range(len(e)) if e[i] in G_cycle.edges]
                l = [(min(coupe[i]), max(coupe[i])) for i in range(len(coupe)) if coupe[i] in G_cycle.edges]
                if len(l) > 0:
                    PLNE_Cycle += (
                        pl.lpSum(x[min(e), max(e)] for e in l) >= 1,
                        f"Contrainte (5) - [{iteration} - {compteur}]",
                    )
                    compteur += 1
            bar.update(1)



        if verbose:
            print(f"\nEcriture du programme linéaire dans 'PLNE_Cycle_{iteration}.lp' :")
        with tqdm(total=1, disable=not verbose) as bar:
            # PLNE_Cycle.writeLP(f"./LP/{dossier}/{nom[:-4]}/PLNE_Cycle_{iteration + 1}_{nom[:-4]}.lp")
            bar.update(1)
        time_fin_ecriture = time.time()
        duree_ecriture = time_fin_ecriture - time_debut_ecriture

        # if verbose:
        #     print(f"\nDéfinition et écriture du programme linéaire réalisées en : {int(duree_ecriture) // 3600}h {int((duree_ecriture % 3600) // 60)}m {int(duree_ecriture % 60)}s {int((duree_ecriture % 60 - int(duree_ecriture % 60)) * 1000)}ms.")

        # Affichage du résultat

        # time_debut_solver = time.time()

        if os.path.exists("./solver_Cplex.json"):
            solver = pl.getSolverFromJson("./solver_Cplex.json")
        else:
            solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
            solver.toJson("solver_Cplex.json")
        solver.msg = verbose

        # solver.tmpDir = "./Solutions"
        result = PLNE_Cycle.solve(solver=solver)
        if verbose:
            for var in PLNE_Cycle.variables():
                print(f'{var.name}: {var.varValue}')
        for var in PLNE_Cycle.variables():
            # if "__dummy" not in var.name:
            if "Arete" in var.name:
                nombres = re.findall(r'\d+', var.name)
                # Extraire les deux premiers nombres
                if len(nombres) >= 2:
                    u, v = int(nombres[0]), int(nombres[1])
                else:
                    raise ValueError("Le nom de la variable ne contient pas suffisamment de nombres.")
                if abs(var.varValue) <= 0.2:  # Arete pas dans la solution
                    if (u, v) in G_cycle.edges:
                        G_cycle.remove_edge(u, v)
                    for c in base_cycle:
                        if (u, v) in [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]:
                            aretes_supprimees.append((u, v))
                            break
        # for e in aretes_supprimees:
        #     n_cycle_avant = len(nx.cycle_basis(G_cycle))
        #     G_cycle.add_edge(e[0], e[1])
        #     n_cycle_apres = len(nx.cycle_basis(G_cycle))
        #     G_cycle.remove_edge(e[0], e[1])
        #     if n_cycle_avant == n_cycle_apres:
        #         aretes_supprimees.remove(e)
        # print(aretes_supprimees)

        # plt.title(f"Cycles éclatés à l'itération {iteration + 1}")
        # nx.draw(G_cycle, with_labels=True)
        # plt.show()
        G_cycle, ajouter_aretes_tmp, nb_comp_connexe = reconnecter_v3(G=G_cycle, G_initial=G)
        ajouter_aretes += ajouter_aretes_tmp
        # ajouter_aretes.append((ajouter_aretes_tmp, nb_comp_connexe))



        # print(f"G_cycle.number_of_edges() = {G_cycle.number_of_edges()}")
        # print(f"G.number_of_edges() = {G.number_of_edges()}")
        # plt.title(f"Graphe reconnecté à l'itération {iteration+1}")
        # nx.draw(G_cycle, with_labels=True)
        # plt.show()
        # nx.draw(G_cycle, labels= {i: i for i in range(1,G.number_of_nodes()+1)})
        # plt.show()
        # print(G_cycle.nodes)
        # print(G_cycle.edges)

        base_cycle = nx.cycle_basis(G_cycle)
        n_cycle = len(base_cycle)
        iteration += 1
        # n_composante_connexe = nx.connected_components(G)
        # print(f"arretes supprimees: {aretes_supprimees}")
        # print(G_cycle.edges)
        # print(f"n_cycle: {n_cycle}, G_cycle.number_of_nodes: {G.number_of_nodes()}, G_cycle.edges: {G.number_of_edges()}")
        if n_cycle == 0 and G_cycle.number_of_edges() == G_cycle.number_of_nodes() - 1:
            convergence = True

    # print(G_cycle.nodes)
    # print(G_cycle.edges)

    # Définition d'un PL annexe qui réécrit "proprement" la méthode avec les bases de cycles en solution d'un PLNE classique
    y = pl.LpVariable.dicts("Sommet", G.nodes, 0, 1, pl.LpInteger)
    bar.update(G.number_of_nodes())
    x = pl.LpVariable.dicts("Arete", G.edges, 0, 1, pl.LpInteger)
    bar.update(G.number_of_edges())

    PLNE_Cycle_final = pl.LpProblem("PLNE_Cycle_" + instance[:-4], pl.LpMinimize)
    bar.update(1)

    PLNE_Cycle_final += pl.lpSum(y), "Objectif"

    compteur = 1
    for e in G.edges():  # On affecte 1 à x[e] si e est dans l'arbre final et 0 sinon
        # print(e)
        # print(G_cycle.edges)
        # print(e in G_cycle.edges)
        PLNE_Cycle_final += (
            x[e] == int(e in G_cycle.edges or (e[1], e[0]) in G_cycle.edges),
            f"Contrainte (2) - {compteur}",
        )
        compteur += 1

    compteur = 1
    for i in G.nodes:  # On affecte 1 à y[i] si i est un noeud de branchement et 0 sinon
        incident_edges = list(G.edges(i))
        PLNE_Cycle_final += (
            pl.lpSum(x[min(e), max(e)] for e in incident_edges) - G.degree(i) * y[i] <= 2,
            f"Contrainte (3) - {compteur}",
        )
        compteur += 1

    time_fin_solver = time.time()
    duree_solver = time_fin_solver - time_debut_solver

    PLNE_Cycle_final.writeLP(f"./LP/{dossier}/{nom[:-4]}/PLNE_Cycle_{nom[:-4]}.lp")

    if os.path.exists("./solver_Cplex.json"):
        solver = pl.getSolverFromJson("./solver_Cplex.json")
    else:
        solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
        solver.toJson("solver_Cplex.json")
    solver.msg = verbose
    solver.timeLimit = set_time_limit(G.number_of_nodes(), G.number_of_edges())

    # solver.tmpDir = "./Solutions"
    result = PLNE_Cycle_final.solve(solver=solver)

    if verbose:
        print(f"Résolution du programme linéaire réalisée en : {int(duree_solver) // 3600}h {int((duree_solver % 3600) // 60)}m {int(duree_solver % 60)}s {int((duree_solver % 60 - int(duree_solver % 60)) * 1000)}ms.")

    ecriture_solution(instance=instance, modele="PLNE_Cycle3", programme=PLNE_Cycle_final,
                      duree_ecriture=iteration, duree_solver=duree_solver)

    liste_fichier = os.listdir()
    for fichier in liste_fichier:
        if len(fichier) >= 5 and fichier[:5] == "clone":
            if fichier in os.listdir("./log_Cplex/"):
                os.remove(f"./log_Cplex/{fichier}")
            shutil.move(fichier, f"./log_Cplex/{fichier}")
    print(result)

    if iteration == maxiter and not convergence and verbose:
        print("L'algorithme n'a pas convergé vers une solution optimale.")
    elif convergence and verbose:
        print(f"L'algorithme a convergé vers une solution optimale en {iteration} itérations.")

def PLNE_cycle_v4(instance, maxiter=1e3, preprocessing=False, verbose=False):
    dossier, nom = instance.split('/')
    G = creation_graphe(instance, False)
    # nx.draw(G, with_labels=True)
    # plt.show()
    G_cycle = nx.Graph(G)
    # base_cycle = nx.cycle_basis(G_cycle)
    # print("Preprocessing")
    # base_cycle = preprocessing_cycle(instance, time_limit=60)
    # n_cycle = len(base_cycle)

    # Cinquième programme : PLNE Cycle

    convergence = False
    iteration = 0
    time_debut_solver = time.time()

    if preprocessing:
        print("Preprocessing")
        banque_de_cycles = preprocessing_cycle(instance, time_limit=60)
        # banque_de_cycles = nx.cycle_basis(G)
        n_cycle = len(nx.cycle_basis(G))
    else:
        banque_de_cycles = nx.cycle_basis(G)
        n_cycle = len(banque_de_cycles)

    # aretes_supprimees = [] # Fonctionne pas
    ajouter_aretes = []
    # nb_comp_connexe = 1
    # banque de cycles supprimés
    while not convergence and iteration < maxiter:
        print(f"Itération: {iteration+1} (itération maximale : {maxiter})")
        aretes_supprimees = []
        # print(aretes_supprimees)
        time_debut_ecriture = time.time()

        # base_cycle = nx.cycle_basis(G_cycle)
        # n_cycle = len(base_cycle)
        # Définition des variables
        if verbose:
            print("Définition des variables :")
        with tqdm(total=G.number_of_nodes() + G.number_of_edges(), disable=not verbose) as bar:
            y = pl.LpVariable.dicts("Sommet", G_cycle.nodes, 0, 1, pl.LpInteger)
            bar.update(G.number_of_nodes())
            x = pl.LpVariable.dicts("Arete", G_cycle.edges, 0, 1, pl.LpInteger)
            bar.update(G.number_of_edges())

        # Définition du problème
        if verbose:
            print("\nDéfinition du problème :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle = pl.LpProblem("PLNE_Cycle_" + instance[:-4], pl.LpMinimize)
            bar.update(1)

        # Ajout de la fonction objectif (1)
        if verbose:
            print("\nAjout de la fonction objectif (1) :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle += pl.lpSum(y), "Objectif"
            bar.update(1)

        # Ajout des contraintes

        # Contrainte (2) : Nombre d'arêtes dans la solution = |V|-1
        if verbose:
            print("\nAjout des contraintes (2) :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle += (
                pl.lpSum(x) == G.number_of_nodes() - 1,
                f"Contrainte (2)",
            )
            bar.update(1)

        # Contrainte (3) : On éclate les bases de cycles
        if verbose:
            print("\nAjout des contraintes (3) :")
        with tqdm(total=n_cycle, disable=not verbose) as bar:
            compteur = 1
            for c in banque_de_cycles: # banque de cycles
                e = []
                ajouter = True
                for i in range(len(c)):
                    if not (c[i], c[(i + 1) % len(c)]) in G_cycle.edges():
                        ajouter = False
                        break
                    e.append((c[i], c[(i + 1) % len(c)]))
                # e = [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]
                # print(c)
                # print(e)
                if ajouter:
                    # print(c)
                    PLNE_Cycle += (
                        pl.lpSum(x[min(e[i]), max(e[i])] for i in range(len(c))) <= len(c) - 1,
                        f"Contrainte (3) - [{compteur}]",
                    )
                    compteur += 1
                bar.update(1)

        print(f"Nombres de cycles pas dans la base: {compteur-1 - n_cycle}, taille {n_cycle}")


        compteur = 1
        for i in G.nodes:  # On affecte 1 à y[i] si i est un noeud de branchement et 0 sinon
            incident_edges = list(G_cycle.edges(i))
            PLNE_Cycle += (
                pl.lpSum(x[min(e), max(e)] for e in incident_edges) - G.degree(i) * y[i] <= 2,
                f"Contrainte (4) - {compteur}",
            )
            compteur += 1

        # Contrainte (5) : Pour chaque coupe obtenue aux itérations précédentes, on garde au moins n-1 arêtes où n est le nombre de composantes connexes associé à la coupe
        if verbose:
            print("\nAjout des contraintes (5) :")
        with tqdm(total=len(ajouter_aretes), disable=not verbose) as bar:
            compteur = 1
            # print(ajouter_aretes)
            for coupe in ajouter_aretes:
                # print(e)
                # print(f"Nb composantes connexes: {nb_comp_connexe}, coupe: {coupe}")
                # l = [(min(e[i][0], e[i][1]), max(e[i][0], e[i][1])) for i in range(len(e)) if e[i] in G_cycle.edges]
                l = [(min(coupe[i]), max(coupe[i])) for i in range(len(coupe)) if coupe[i] in G_cycle.edges]
                if len(l) > 0:
                    PLNE_Cycle += (
                        pl.lpSum(x[min(e), max(e)] for e in l) >= 1,
                        f"Contrainte (5) - [{iteration} - {compteur}]",
                    )
                    compteur += 1
            bar.update(1)


        if verbose:
            print(f"\nEcriture du programme linéaire dans 'PLNE_Cycle4_{iteration}.lp' :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle.writeLP(f"./LP/{dossier}/{nom[:-4]}/PLNE_Cycle4_{iteration + 1}_{nom[:-4]}.lp")
            bar.update(1)
        time_fin_ecriture = time.time()
        duree_ecriture = time_fin_ecriture - time_debut_ecriture

        # if verbose:
        #     print(f"\nDéfinition et écriture du programme linéaire réalisées en : {int(duree_ecriture) // 3600}h {int((duree_ecriture % 3600) // 60)}m {int(duree_ecriture % 60)}s {int((duree_ecriture % 60 - int(duree_ecriture % 60)) * 1000)}ms.")

        # Affichage du résultat

        # time_debut_solver = time.time()

        if os.path.exists("./solver_Cplex.json"):
            solver = pl.getSolverFromJson("./solver_Cplex.json")
        else:
            solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
            solver.toJson("solver_Cplex.json")
        solver.msg = verbose

        # solver.tmpDir = "./Solutions"
        result = PLNE_Cycle.solve(solver=solver)
        if verbose:
            for var in PLNE_Cycle.variables():
                print(f'{var.name}: {var.varValue}')
        for var in PLNE_Cycle.variables():
            # if "__dummy" not in var.name:
            if "Arete" in var.name:
                nombres = re.findall(r'\d+', var.name)
                # Extraire les deux premiers nombres
                if len(nombres) >= 2:
                    u, v = int(nombres[0]), int(nombres[1])
                else:
                    raise ValueError("Le nom de la variable ne contient pas suffisamment de nombres.")
                if abs(var.varValue) <= 0.2:  # Arete pas dans la solution
                    if (u, v) in G_cycle.edges:
                        G_cycle.remove_edge(u, v)
                    for c in banque_de_cycles:
                        if (u, v) in [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]:
                            aretes_supprimees.append((u, v))
                            break
        # for e in aretes_supprimees:
        #     n_cycle_avant = len(nx.cycle_basis(G_cycle))
        #     G_cycle.add_edge(e[0], e[1])
        #     n_cycle_apres = len(nx.cycle_basis(G_cycle))
        #     G_cycle.remove_edge(e[0], e[1])
        #     if n_cycle_avant == n_cycle_apres:
        #         aretes_supprimees.remove(e)
        # print(aretes_supprimees)

        # plt.title(f"Cycles éclatés à l'itération {iteration + 1}")
        # nx.draw(G_cycle, with_labels=True)
        # plt.show()
        G_cycle, ajouter_aretes_tmp, nb_comp_connexe = reconnecter_v3(G=G_cycle, G_initial=G)
        ajouter_aretes += ajouter_aretes_tmp

        # ajouter_aretes.append((ajouter_aretes_tmp, nb_comp_connexe))

        # print(f"G_cycle.number_of_edges() = {G_cycle.number_of_edges()}")
        # print(f"G.number_of_edges() = {G.number_of_edges()}")
        # plt.title(f"Graphe reconnecté à l'itération {iteration+1}")
        # nx.draw(G_cycle, with_labels=True)
        # plt.show()
        # nx.draw(G_cycle, labels= {i: i for i in range(1,G.number_of_nodes()+1)})
        # plt.show()
        # print(G_cycle.nodes)
        # print(G_cycle.edges)

        base_cycle_tmp = nx.cycle_basis(G_cycle)
        banque_de_cycles += base_cycle_tmp
        # print(f"Taille liste cycles supprimés: {len(base_cycle)}")
        n_cycle = len(base_cycle_tmp)




        iteration += 1
        # n_composante_connexe = nx.connected_components(G)
        # print(f"arretes supprimees: {aretes_supprimees}")
        # print(G_cycle.edges)
        # print(f"n_cycle: {n_cycle}, G_cycle.number_of_nodes: {G.number_of_nodes()}, G_cycle.edges: {G.number_of_edges()}")
        if n_cycle == 0 and G_cycle.number_of_edges() == G_cycle.number_of_nodes() - 1:
            convergence = True

    # print(G_cycle.nodes)
    # print(G_cycle.edges)

    # Définition d'un PL annexe qui réécrit "proprement" la méthode avec les bases de cycles en solution d'un PLNE classique
    y = pl.LpVariable.dicts("Sommet", G.nodes, 0, 1, pl.LpInteger)
    bar.update(G.number_of_nodes())
    x = pl.LpVariable.dicts("Arete", G.edges, 0, 1, pl.LpInteger)
    bar.update(G.number_of_edges())

    PLNE_Cycle_final = pl.LpProblem("PLNE_Cycle_" + instance[:-4], pl.LpMinimize)
    bar.update(1)

    PLNE_Cycle_final += pl.lpSum(y), "Objectif"

    compteur = 1
    for e in G.edges():  # On affecte 1 à x[e] si e est dans l'arbre final et 0 sinon
        # print(e)
        # print(G_cycle.edges)
        # print(e in G_cycle.edges)
        PLNE_Cycle_final += (
            x[e] == int(e in G_cycle.edges or (e[1], e[0]) in G_cycle.edges),
            f"Contrainte (2) - {compteur}",
        )
        compteur += 1

    compteur = 1
    for i in G.nodes:  # On affecte 1 à y[i] si i est un noeud de branchement et 0 sinon
        incident_edges = list(G.edges(i))
        PLNE_Cycle_final += (
            pl.lpSum(x[min(e), max(e)] for e in incident_edges) - G.degree(i) * y[i] <= 2,
            f"Contrainte (3) - {compteur}",
        )
        compteur += 1

    time_fin_solver = time.time()
    duree_solver = time_fin_solver - time_debut_solver

    PLNE_Cycle_final.writeLP(f"./LP/{dossier}/{nom[:-4]}/PLNE_Cycle_{nom[:-4]}.lp")

    if os.path.exists("./solver_Cplex.json"):
        solver = pl.getSolverFromJson("./solver_Cplex.json")
    else:
        solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
        solver.toJson("solver_Cplex.json")
    solver.msg = verbose
    solver.timeLimit = set_time_limit(G.number_of_nodes(), G.number_of_edges())

    # solver.tmpDir = "./Solutions"
    result = PLNE_Cycle_final.solve(solver=solver)

    if verbose:
        print(f"Résolution du programme linéaire réalisée en : {int(duree_solver) // 3600}h {int((duree_solver % 3600) // 60)}m {int(duree_solver % 60)}s {int((duree_solver % 60 - int(duree_solver % 60)) * 1000)}ms.")

    chaine = "_preprocessing" if preprocessing else ""
    ecriture_solution(instance=instance, modele="PLNE_Cycle4"+chaine, programme=PLNE_Cycle_final,
                      duree_ecriture=iteration, duree_solver=duree_solver)

    liste_fichier = os.listdir()
    for fichier in liste_fichier:
        if len(fichier) >= 5 and fichier[:5] == "clone":
            if fichier in os.listdir("./log_Cplex/"):
                os.remove(f"./log_Cplex/{fichier}")
            shutil.move(fichier, f"./log_Cplex/{fichier}")
    print(result)

    if iteration == maxiter and not convergence and verbose:
        print("L'algorithme n'a pas convergé vers une solution optimale.")
    elif convergence and verbose:
        print(f"L'algorithme a convergé vers une solution optimale en {iteration} itérations.")

def PLNE_cycle_final(instance, maxiter=1e3, preprocessing=False, verbose=False):
    """ Possibilité de préprocessing qui calcule au prélable des bases de cycles et les casse,
    sans reconnection du graphe.
    A chaque itération, on garde en mémoire les bases de cycles précédentes."""
    dossier, nom = instance.split('/')
    G = creation_graphe(instance, False)
    # nx.draw(G, with_labels=True)
    # plt.show()
    G_cycle = nx.Graph(G)
    # base_cycle = nx.cycle_basis(G_cycle)
    # print("Preprocessing")
    # base_cycle = preprocessing_cycle(instance, time_limit=60)
    # n_cycle = len(base_cycle)

    # Cinquième programme : PLNE Cycle

    convergence = False
    iteration = 0
    time_debut_solver = time.time()

    if preprocessing:
        print("Preprocessing")
        banque_de_cycles = preprocessing_cycle(instance, time_limit=60)
        # banque_de_cycles = nx.cycle_basis(G)
        n_cycle = len(nx.cycle_basis(G))
    else:
        banque_de_cycles = nx.cycle_basis(G)
        n_cycle = len(banque_de_cycles)

    # aretes_supprimees = [] # Fonctionne pas
    ajouter_aretes = []
    aretes_base_de_cycles = []
    # nb_comp_connexe = 1
    # banque de cycles supprimés
    while not convergence and iteration < maxiter:
        print(f"Itération: {iteration+1} (itération maximale : {maxiter})")
        aretes_supprimees = []
        # print(aretes_supprimees)
        time_debut_ecriture = time.time()

        # base_cycle = nx.cycle_basis(G_cycle)
        # n_cycle = len(base_cycle)
        # Définition des variables
        if verbose:
            print("Définition des variables :")
        with tqdm(total=G.number_of_nodes() + G.number_of_edges(), disable=not verbose) as bar:
            y = pl.LpVariable.dicts("Sommet", G_cycle.nodes, 0, 1, pl.LpInteger)
            bar.update(G.number_of_nodes())
            x = pl.LpVariable.dicts("Arete", G_cycle.edges, 0, 1, pl.LpInteger)
            bar.update(G.number_of_edges())

        # Définition du problème
        if verbose:
            print("\nDéfinition du problème :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle = pl.LpProblem("PLNE_Cycle_" + instance[:-4], pl.LpMinimize)
            bar.update(1)

        # Ajout de la fonction objectif (1)
        if verbose:
            print("\nAjout de la fonction objectif (1) :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle += pl.lpSum(y), "Objectif"
            bar.update(1)

        # Ajout des contraintes

        # Contrainte (2) : Nombre d'arêtes dans la solution = |V|-1
        if verbose:
            print("\nAjout des contraintes (2) :")
        with tqdm(total=1, disable=not verbose) as bar:
            PLNE_Cycle += (
                pl.lpSum(x) == G.number_of_nodes() - 1,
                f"Contrainte (2)",
            )
            bar.update(1)

        # Contrainte (3) : On éclate les bases de cycles
        if verbose:
            print("\nAjout des contraintes (3) :")
        with tqdm(total=n_cycle, disable=not verbose) as bar:
            compteur = 1
            for c in banque_de_cycles: # banque de cycles
                e = []
                ajouter = True
                for i in range(len(c)):
                    if not (c[i], c[(i + 1) % len(c)]) in G_cycle.edges():
                        ajouter = False
                        break
                    e.append((c[i], c[(i + 1) % len(c)]))
                # e = [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]
                # print(c)
                # print(e)
                if ajouter:
                    # print(c)
                    PLNE_Cycle += (
                        pl.lpSum(x[min(e[i]), max(e[i])] for i in range(len(c))) <= len(c) - 1,
                        f"Contrainte (3) - [{compteur}]",
                    )
                    compteur += 1
                bar.update(1)

        print(f"Nombres de cycles pas dans la base: {compteur-1 - n_cycle}, taille {n_cycle}")

        # Contraintes (4) : On affecte 1 à y[i] si i est un noeud de branchement et 0 sinon
        compteur = 1
        for i in G.nodes:
            incident_edges = list(G_cycle.edges(i))
            PLNE_Cycle += (
                pl.lpSum(x[min(e), max(e)] for e in incident_edges) - G.degree(i) * y[i] <= 2,
                f"Contrainte (4) - {compteur}",
            )
            compteur += 1

        # Contrainte (5) : Chaque composante connexe doit être connectée à au moins une autre composante connexe
        if verbose:
            print("\nAjout des contraintes (5) :")
        with tqdm(total=len(ajouter_aretes), disable=not verbose) as bar:
            compteur = 1
            # print(ajouter_aretes)
            for coupe in ajouter_aretes:
                # print(e)
                # print(f"Nb composantes connexes: {nb_comp_connexe}, coupe: {coupe}")
                # l = [(min(e[i][0], e[i][1]), max(e[i][0], e[i][1])) for i in range(len(e)) if e[i] in G_cycle.edges]
                # l = [(min(coupe[i]), max(coupe[i])) for i in range(len(coupe)) if coupe[i] in G_cycle.edges]
                # if len(l) > 0:
                PLNE_Cycle += (
                    pl.lpSum(x[min(e), max(e)] for e in coupe) >= 1,
                    f"Contrainte (5) - [{iteration} - {compteur}]",
                )
                compteur += 1
            bar.update(1)

        # Contraintes (6)

        compteur = 1
        for e in G_cycle.edges:
            if e not in aretes_base_de_cycles and iteration > 0:
                PLNE_Cycle += (
                    pl.lpSum(x[min(e), max(e)]) == 1,
                    f"Contrainte (6) - [{iteration} - {compteur}]",
                )
            compteur += 1

        if verbose:
            print(f"\nEcriture du programme linéaire dans 'PLNE_Cycle4_{iteration}.lp' :")
        with tqdm(total=1, disable=not verbose) as bar:
            # if not os.path.exists(f"./Solutions/{dossier}/{instance[:-4]}"):
            #     os.makedirs(f"./Solutions/{dossier}/{instance[:-4]}")
            # PLNE_Cycle.writeLP(f"./LP/{dossier}/{nom[:-4]}/PLNE_Cycle4_bis_{iteration + 1}_{nom[:-4]}.lp")
            bar.update(1)
        time_fin_ecriture = time.time()
        duree_ecriture = time_fin_ecriture - time_debut_ecriture

        # if verbose:
        #     print(f"\nDéfinition et écriture du programme linéaire réalisées en : {int(duree_ecriture) // 3600}h {int((duree_ecriture % 3600) // 60)}m {int(duree_ecriture % 60)}s {int((duree_ecriture % 60 - int(duree_ecriture % 60)) * 1000)}ms.")

        # Affichage du résultat

        # time_debut_solver = time.time()

        if os.path.exists("./solver_Cplex.json"):
            solver = pl.getSolverFromJson("./solver_Cplex.json")
        else:
            solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
            solver.toJson("solver_Cplex.json")
        solver.msg = verbose

        # solver.tmpDir = "./Solutions"
        result = PLNE_Cycle.solve(solver=solver)
        if verbose:
            for var in PLNE_Cycle.variables():
                print(f'{var.name}: {var.varValue}')
        for var in PLNE_Cycle.variables():
            # if "__dummy" not in var.name:
            if "Arete" in var.name:
                nombres = re.findall(r'\d+', var.name)
                # Extraire les deux premiers nombres
                if len(nombres) >= 2:
                    u, v = int(nombres[0]), int(nombres[1])
                else:
                    raise ValueError("Le nom de la variable ne contient pas suffisamment de nombres.")
                if abs(var.varValue) <= 0.2:  # Arete pas dans la solution
                    if (u, v) in G_cycle.edges:
                        G_cycle.remove_edge(u, v)
                    for c in banque_de_cycles:
                        if (u, v) in [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]:
                            aretes_supprimees.append((u, v))
                            break
        # for e in aretes_supprimees:
        #     n_cycle_avant = len(nx.cycle_basis(G_cycle))
        #     G_cycle.add_edge(e[0], e[1])
        #     n_cycle_apres = len(nx.cycle_basis(G_cycle))
        #     G_cycle.remove_edge(e[0], e[1])
        #     if n_cycle_avant == n_cycle_apres:
        #         aretes_supprimees.remove(e)
        # print(aretes_supprimees)

        # plt.title(f"Cycles éclatés à l'itération {iteration + 1}")
        # nx.draw(G_cycle, with_labels=True)
        # plt.show()
        G_cycle, ajouter_aretes_tmp, nb_comp_connexe = reconnecter_final(G=G_cycle, G_initial=G)
        # print(ajouter_aretes_tmp)
        ajouter_aretes = ajouter_aretes_tmp

        # ajouter_aretes.append((ajouter_aretes_tmp, nb_comp_connexe))

        # print(f"G_cycle.number_of_edges() = {G_cycle.number_of_edges()}")
        # print(f"G.number_of_edges() = {G.number_of_edges()}")
        # plt.title(f"Graphe reconnecté à l'itération {iteration+1}")
        # nx.draw(G_cycle, with_labels=True)
        # plt.show()
        # nx.draw(G_cycle, labels= {i: i for i in range(1,G.number_of_nodes()+1)})
        # plt.show()
        # print(G_cycle.nodes)
        # print(G_cycle.edges)

        base_cycle_tmp = nx.cycle_basis(G_cycle)
        banque_de_cycles += base_cycle_tmp
        # print(f"Taille liste cycles supprimés: {len(base_cycle)}")
        n_cycle = len(base_cycle_tmp)

        aretes_base_de_cycles = []
        for cycle in base_cycle_tmp:
            taille_cycle = len(cycle)
            for i in range(taille_cycle):
                if (cycle[i], cycle[(i+1)%taille_cycle]) not in aretes_base_de_cycles and (cycle[(i+1)%taille_cycle], cycle[i]):
                    aretes_base_de_cycles.append((cycle[i], cycle[(i+1)%taille_cycle]))

        iteration += 1
        # n_composante_connexe = nx.connected_components(G)
        # print(f"arretes supprimees: {aretes_supprimees}")
        # print(G_cycle.edges)
        # print(f"n_cycle: {n_cycle}, G_cycle.number_of_nodes: {G.number_of_nodes()}, G_cycle.edges: {G.number_of_edges()}")
        if n_cycle == 0 and G_cycle.number_of_edges() == G_cycle.number_of_nodes() - 1:
            convergence = True

    # print(G_cycle.nodes)
    # print(G_cycle.edges)

    # Définition d'un PL annexe qui réécrit "proprement" la méthode avec les bases de cycles en solution d'un PLNE classique
    y = pl.LpVariable.dicts("Sommet", G.nodes, 0, 1, pl.LpInteger)
    bar.update(G.number_of_nodes())
    x = pl.LpVariable.dicts("Arete", G.edges, 0, 1, pl.LpInteger)
    bar.update(G.number_of_edges())

    PLNE_Cycle_final = pl.LpProblem("PLNE_Cycle_" + instance[:-4], pl.LpMinimize)
    bar.update(1)

    PLNE_Cycle_final += pl.lpSum(y), "Objectif"

    compteur = 1
    for e in G.edges():  # On affecte 1 à x[e] si e est dans l'arbre final et 0 sinon
        # print(e)
        # print(G_cycle.edges)
        # print(e in G_cycle.edges)
        PLNE_Cycle_final += (
            x[e] == int(e in G_cycle.edges or (e[1], e[0]) in G_cycle.edges),
            f"Contrainte (2) - {compteur}",
        )
        compteur += 1

    compteur = 1
    for i in G.nodes:  # On affecte 1 à y[i] si i est un noeud de branchement et 0 sinon
        incident_edges = list(G.edges(i))
        PLNE_Cycle_final += (
            pl.lpSum(x[min(e), max(e)] for e in incident_edges) - G.degree(i) * y[i] <= 2,
            f"Contrainte (3) - {compteur}",
        )
        compteur += 1

    time_fin_solver = time.time()
    duree_solver = time_fin_solver - time_debut_solver

    PLNE_Cycle_final.writeLP(f"./LP/{dossier}/{nom[:-4]}/PLNE_Cycle_{nom[:-4]}.lp")

    if os.path.exists("./solver_Cplex.json"):
        solver = pl.getSolverFromJson("./solver_Cplex.json")
    else:
        solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
        solver.toJson("solver_Cplex.json")
    solver.msg = verbose
    solver.timeLimit = set_time_limit(G.number_of_nodes(), G.number_of_edges())

    # solver.tmpDir = "./Solutions"
    result = PLNE_Cycle_final.solve(solver=solver)

    if verbose:
        print(f"Résolution du programme linéaire réalisée en : {int(duree_solver) // 3600}h {int((duree_solver % 3600) // 60)}m {int(duree_solver % 60)}s {int((duree_solver % 60 - int(duree_solver % 60)) * 1000)}ms.")

    chaine = "_preprocessing" if preprocessing else ""
    ecriture_solution(instance=instance, modele="PLNE_Cycle4_bis"+chaine, programme=PLNE_Cycle_final,
                      duree_ecriture=iteration, duree_solver=duree_solver)

    liste_fichier = os.listdir()
    for fichier in liste_fichier:
        if len(fichier) >= 5 and fichier[:5] == "clone":
            if fichier in os.listdir("./log_Cplex/"):
                os.remove(f"./log_Cplex/{fichier}")
            shutil.move(fichier, f"./log_Cplex/{fichier}")
    print(result)

    if iteration == maxiter and not convergence and verbose:
        print("L'algorithme n'a pas convergé vers une solution optimale.")
    elif convergence and verbose:
        print(f"L'algorithme a convergé vers une solution optimale en {iteration} itérations.")

def preprocessing_cycle(instance, time_limit = 60):
    G = creation_graphe(instance)
    dossier, nom = instance.split('/')

    banque_de_cycles = []

    iteration = 0
    convergence = False
    temps_debut = time.time()
    temps_execution = 0
    while not convergence and temps_execution < time_limit:
        # print(f"Iteration {iteration+1}")
        # print(f"{len(banque_de_cycles)} banque des cycles")
        # Définition d'un PL
        y = pl.LpVariable.dicts("Sommet", G.nodes, 0, 1, pl.LpInteger)
        x = pl.LpVariable.dicts("Arete", G.edges, 0, 1, pl.LpInteger)

        PLNE_preprocessing = pl.LpProblem("PLNE_Preprocessing", pl.LpMaximize)

        PLNE_preprocessing += pl.lpSum(x), "Objectif"

        base_de_cycles = nx.cycle_basis(G)
        # print(base_de_cycles)
        banque_de_cycles += base_de_cycles

        compteur = 1
        for c in banque_de_cycles:
            # print(c)
            liste_aretes = []
            ajouter_contrainte = True
            for i in range(len(c)):
                if not (c[i], c[(i + 1) % len(c)]) in G.edges():
                    ajouter_contrainte = False
                    break
                liste_aretes += [(c[i], c[(i + 1) % len(c)])]
            PLNE_preprocessing += (
                pl.lpSum(x[min(e), max(e)] for e in liste_aretes) <= len(c)-1,
                f"Contrainte (1) - [{iteration} - {compteur}]",
            )
            compteur += 1

        # PLNE_preprocessing.writeLP(f"./LP/{dossier}/{nom[:-4]}/PLNE_Preprocessing_Cycle_{nom[:-4]}.lp")

        if os.path.exists("./solver_Cplex.json"):
            solver = pl.getSolverFromJson("./solver_Cplex.json")
        else:
            solver = pl.CPLEX_CMD(path=path_to_cplex, logPath="./log_Cplex/cplex.log", timeLimit=600)
            solver.toJson("solver_Cplex.json")
        solver.time = time_limit
        solver.msg = False

        # solver.tmpDir = "./Solutions"
        result = PLNE_preprocessing.solve(solver=solver)

        for var in PLNE_preprocessing.variables():
            if "Arete" in var.name:
                nombres = re.findall(r'\d+', var.name)
                # Extraire les deux premiers nombres
                if len(nombres) >= 2:
                    u, v = int(nombres[0]), int(nombres[1])
                else:
                    raise ValueError("Le nom de la variable ne contient pas suffisamment de nombres.")
                if abs(var.varValue) <= 0.2:  # Arête supprimée
                    if (u, v) in G.edges:
                        G.remove_edge(u, v)
        temps_execution = time.time() - temps_debut
        iteration += 1
        convergence = nx.is_forest(G)

    print(f"Temps d'execution: {temps_execution}s")
    return banque_de_cycles



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Définition du graphe
    # instance = "Spd_RF2_20_27_211.txt"
    # instance = "Spd_RF2_20_27_219.txt"
    # instance = "Test.txt"

    # sommets = [i for i in range(1,9)]
    # aretes = [(1,2), (1,4), (2,3), (2,5), (3,6), (4,5), (5,6), (5,7), (6,8), (7,8)]
    # aretes_supprimees = [(2,3), (5,6), (7,8)]
    # G = nx.Graph()
    # G.add_nodes_from(sommets)
    # G.add_edges_from(aretes)
    # # nx.draw(G, with_labels=True)
    # # plt.show()
    #
    # G_2 = nx.Graph()
    # G_2.add_nodes_from(sommets)
    # G_2.add_edges_from([e for e in aretes if e not in aretes_supprimees])
    # # nx.draw(G_2, with_labels=True)
    # # plt.show()
    # print(aretes_supprimees[:-1])
    # G_3 = reconnecter(G_2, G, aretes_supprimees[:-2])
    # nx.draw(G_3, with_labels=True)
    # plt.show()

    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_20_27_211.txt"
    instance = "Spd_Inst_Rid_Final2/Spd_RF2_20_27_219.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_20_27_243.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_60_71_1011.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_200_222_3811.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_400_429_4611.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_500_672_5195.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_500_672_5203.txt"
    # instance = "Spd_Inst_Rid_Final2_500-1000/Spd_RF2_1000_1239_6203.txt"

    # PLNE_cycle(instance, maxiter=1e3, verbose=False)
    # PLNE_cycle_v2(instance, maxiter=1e3, verbose=False)
    # PLNE_cycle_backup(instance, maxiter=1e3, verbose=False)
    # PLNE_cycle_v3(instance, maxiter=1e3, verbose=False)

    # banque_de_cycles = preprocessing_cycle(instance, time_limit = 60)
    # print(banque_de_cycles)
    # PLNE_cycle_v4(instance, maxiter=1e3, preprocessing=False, verbose=False)
    PLNE_cycle_final(instance, maxiter=1e4, preprocessing=True, verbose=False)
