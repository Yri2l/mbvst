import pulp as pl
import networkx as nx
from tqdm import tqdm
import time
# from itertools import chain, combinations
import os
import shutil
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np


def creation_graphe(instance, version_orientee = False, from_zero=False):
    dossier, instance = instance.split('/')
    path_instance = f"./Instances/{dossier}/{instance}"
    # file = open("Spd_RF2_20_27_211.txt", "r", encoding="utf8")
    # file = open(path_instance + instance, "r", encoding="utf8")*
    file = open(path_instance, "r", encoding="utf8")
    lines = file.readlines()
    liste_convertie = [tuple(list(map(int, chaine.split()))[:-1]) for chaine in lines]
    file.close()

    G = nx.Graph()
    if from_zero:
        G.add_nodes_from([i for i in range(liste_convertie[0][0])])
        G.add_edges_from((u-1, v-1) for u,v in liste_convertie[1:])
    else:
        G.add_nodes_from([i for i in range(1, liste_convertie[0][0] + 1)])
        G.add_edges_from(liste_convertie[1:])
    if version_orientee:
        G = G.to_directed()
    return G

def graphe_plus_1(graphe):
    res = nx.Graph()
    for node in graphe.nodes:
        res.add_node(node+1)
    for u, v in graphe.edges:
        res.add_edge(u+1, v+1)
    return res

def extraire_labels(chaine):
    # Expression régulière pour capturer les trois nombres : deux entiers et un réel
    pattern = r"Arete_\((\d+),_(\d+)\) = ([\d\.]+)"

    match = re.search(pattern, chaine)

    if match:
        # Récupérer les valeurs extraites et les convertir
        premier_nombre = int(match.group(1))  # premier entier
        second_nombre = int(match.group(2))  # second entier
        troisieme_nombre = float(match.group(3))  # nombre réel

        return premier_nombre, second_nombre, troisieme_nombre
    else:
        # Retourner None si la chaîne ne correspond pas au format
        return None

def creation_graphe_avec_edge_labels(instance, from_zero=False):
    dossier, instance = instance.split('/')
    path_instance = f"./Instances/{dossier}/{instance}"
    path_solution = f"./Solutions/{dossier}/{instance[:-4]}/sol_PLNE_CP_{instance}"
    # file = open("Spd_RF2_20_27_211.txt", "r", encoding="utf8")
    # file = open(path_instance + instance, "r", encoding="utf8")*
    file = open(path_instance, "r", encoding="utf8")
    lines = file.readlines()
    liste_convertie = [tuple(list(map(int, chaine.split()))[:-1]) for chaine in lines]
    file.close()

    G = nx.Graph()
    if from_zero:
        G.add_nodes_from([i for i in range(liste_convertie[0][0])])
        G.add_edges_from((u-1, v-1) for u,v in liste_convertie[1:])
    else:
        G.add_nodes_from([i for i in range(1, liste_convertie[0][0] + 1)])
        G.add_edges_from(liste_convertie[1:])

    # On ajoute les labels des arêtes pour indiquer celles qui sont dans la solution.
    file = open(path_solution, "r", encoding="utf8")
    lines = file.readlines()
    liste_convertie = lines[4:]
    for l in liste_convertie:
        if l[:5] == "Arete":
            u, v, label = extraire_labels(l)
            # Ajouter le label au graphe
            G.edges[(u,v)]['edge_label'] = int(label >= 0.5)
    file.close()

    return G

def preprocessing_graphe(graphe):
    ### Noeuds
    # Degré
    nx.set_node_attributes(graphe, dict(nx.degree(graphe)), 'Degree')
    # Betweeness centrality
    nx.set_node_attributes(graphe, nx.betweenness_centrality(graphe), 'Betweeness_centrality')
    # Clustering
    nx.set_node_attributes(graphe, nx.clustering(graphe), 'Clustering')

    nb_node_features = len(graphe.nodes[list(graphe.nodes)[0]])

    ### Aretes
    nx.set_edge_attributes(graphe, nx.edge_betweenness_centrality(graphe), 'Betweeness_centrality')

    nb_edge_features = len(graphe.edges[list(graphe.edges)[0]])
    return graphe, nb_node_features, nb_edge_features

def chaine_to_tuple(s):
    # Extraire uniquement la partie numérique entre parenthèses
    partie_numerique = s[s.index("(")+1 : s.index(")")]
    # Séparer les éléments par la virgule et convertir en entiers
    return tuple(map(int, partie_numerique.split(",_")))

def ecriture_solution(instance, modele, programme, duree_ecriture, duree_solver, source=None):
    dossier, instance = instance.split('/')

    if not os.path.exists(f"./Solutions/{dossier}/{instance[:-4]}"):
        os.makedirs(f"./Solutions/{dossier}/{instance[:-4]}")
    with open(f"./Solutions/{dossier}/{instance[:-4]}/sol_{modele}_{instance}", "w") as fichier:
        fichier.write(f"Statut: {pl.LpStatus[programme.status]}\n")
        fichier.write(f"Ecriture du programme lineaire realisee en : {int(duree_ecriture) // 3600}h {int((duree_ecriture % 3600) // 60)}m {int(duree_ecriture % 60)}s {int((duree_ecriture % 60 - int(duree_ecriture % 60)) * 1000)}ms ({duree_ecriture}s).\n")
        fichier.write(f"Resolution du programme lineaire realisee en : {int(duree_solver) // 3600}h {int((duree_solver % 3600) // 60)}m {int(duree_solver % 60)}s {int((duree_solver % 60 - int(duree_solver % 60)) * 1000)}ms ({duree_solver}s).\n")
        fichier.write(f"Valeur de la fonction objectif: {pl.value(programme.objective)}\n")
        if source is not None:
            fichier.write(f"Valeurs des variables (source: {source}):\n")
        for var in programme.variables():
            if "__dummy" not in var.name:
                fichier.write(f"{var.name} = {var.varValue}\n")


def extraire_deux_premiers_nombres(chaine):
    """
    Extrait les deux premiers nombres d'une chaîne de caractères.

    :param chaine: La chaîne d'entrée (ex. "Spd_RF2_100_114_1811")
    :return: Un tuple contenant les deux premiers nombres (ex. (100, 114))
    """
    # Trouver tous les nombres dans la chaîne
    nombres = re.findall(r'\d+', chaine)
    # Extraire les deux premiers nombres
    if len(nombres) >= 3:
        return int(nombres[1]), int(nombres[2])
    else:
        raise ValueError("La chaîne ne contient pas suffisamment de nombres.")

def save_to_csv(dossier):
    if not os.path.exists(f"./Resultats/{dossier}/{dossier}.csv"):
        liste_index = []
        for fichier in os.listdir(f"./Instances/{dossier}/"):
            if fichier[:3] == "Spd":
                liste_index.append(fichier[:-4])
        liste_colonne = ["Nb sommets", "Nb aretes", "Time_limit",
                         "Statut_exp", "Statut_CP", "Statut_CPM", "Statut_Martin", "Statut_cycle",
                         "Objectif_exp", "Objectif_CP", "Objectif_CPM", "Objectif_Martin", "Objectif_cycle",
                         "Ecriture_exp", "Ecriture_CP", "Ecriture_CPM", "Ecriture_Martin", "Ecriture_cycle",
                         "Resolution_exp", "Resolution_CP", "Resolution_CPM", "Resolution_Martin", "Resolution_cycle"]
        # dictionnaire = {"Instance" : liste_index}
        # for col in liste_colonne:
        #     dictionnaire[col] = None
        # df = pd.DataFrame(dictionnaire)
        df = pd.DataFrame(index = liste_index, columns = liste_colonne)
        df.to_csv(f"./Resultats/{dossier}/{dossier}.csv", index=False)
    else:
        df = pd.read_csv(f"./Resultats/{dossier}/{dossier}.csv", index_col=0)
    liste_solution = []
    for solution in os.listdir(f"./Solutions/{dossier}/"):
        if solution[-3:] != "csv" and solution[:3] == "Spd":
            liste_solution.append(solution)
    for instance in liste_solution:
        for solution in os.listdir(f"./Solutions/{dossier}/{instance}/"):
            if solution[-3:] == "txt":
                # print(solution)
                try:
                    methode = solution[solution.index("PLNE"):solution.index("_Spd"):]
                except:
                    methode = ""
                # print(methode)
                if methode in ["PLNE_exp", "PLNE_CP", "PLNE_CPM", "PLNE_Martin", "PLNE_Cycle4_bis_preprocessing"]:
                    print(f"./Solutions/{dossier}/{instance}/{solution}")
                    file = open(f"./Solutions/{dossier}/{instance}/{solution}", "r", encoding="iso-8859-1")
                    lines = file.readlines()
                    file.close()
                    print(lines[3])
                    # print(lines[3][lines[3].index(":")+2:-1])
                    objectif = int(float(lines[3][lines[3].index(":")+2:-1]))
                    ecriture = float(lines[1][lines[1].index("(")+1:lines[1].index(")")-1])
                    resolution = float(lines[2][lines[2].index("(")+1:lines[2].index(")")-1])
                    statut = 1 if lines[0][lines[0].index(":")+2: lines[0].index(":")+5] == "Opt" else 0
                    nom_instance = instance[instance.index("Spd"):]
                    n_sommets, n_aretes = extraire_deux_premiers_nombres(nom_instance)
                    # print(n_sommets, n_aretes)
                    df.loc[nom_instance, "Nb sommets"] = n_sommets
                    df.loc[nom_instance, "Nb aretes"] = n_aretes
                    df.loc[nom_instance, "Time_limit"] = set_time_limit(n_sommets, n_aretes)
                    if methode == "PLNE_exp":
                        df.loc[nom_instance, "Objectif_exp"] = objectif
                        df.loc[nom_instance, "Statut_exp"] = statut
                        df.loc[nom_instance, "Ecriture_exp"] = ecriture
                        df.loc[nom_instance, "Resolution_exp"] = resolution
                    elif methode == "PLNE_CP":
                        df.loc[nom_instance, "Objectif_CP"] = objectif
                        df.loc[nom_instance, "Statut_CP"] = statut
                        df.loc[nom_instance, "Ecriture_CP"] = ecriture
                        df.loc[nom_instance, "Resolution_CP"] = resolution
                    elif methode == "PLNE_CPM":
                        df.loc[nom_instance, "Objectif_CPM"] = objectif
                        df.loc[nom_instance, "Statut_CPM"] = statut
                        df.loc[nom_instance, "Ecriture_CPM"] = ecriture
                        df.loc[nom_instance, "Resolution_CPM"] = resolution
                    elif methode == "PLNE_Martin":
                        df.loc[nom_instance, "Objectif_Martin"] = objectif
                        df.loc[nom_instance, "Statut_Martin"] = statut
                        df.loc[nom_instance, "Ecriture_Martin"] = ecriture
                        df.loc[nom_instance, "Resolution_Martin"] = resolution
                    elif methode == "PLNE_Cycle4_bis_preprocessing":
                        df.loc[nom_instance, "Objectif_cycle"] = objectif
                        df.loc[nom_instance, "Statut_cycle"] = statut
                        df.loc[nom_instance, "Ecriture_cycle"] = ecriture
                        df.loc[nom_instance, "Resolution_cycle"] = resolution
    df.sort_values(by=["Nb sommets", "Nb aretes"], ascending=True, inplace=True)
    df.to_csv(f"./Resultats/{dossier}/{dossier}.csv", index=True)
    print(f"Fichier enregistré au chemin: ./Resultats/{dossier}/{dossier}.csv")

def graphiques(dossier):
    if not os.path.exists(f"./Resultats/{dossier}/{dossier}.csv"):
        print("Le fichier n'existe pas.")
    else:
        df = pd.read_csv(f"./Resultats/{dossier}/{dossier}.csv", index_col=0)

    df["Nb variables"] = df["Nb sommets"] + df["Nb aretes"]
    # df.sort_values(by=["Nb sommets", "Nb aretes"], ascending=True, inplace=True)

    # print(df[:3])

    # Calcul de la moyenne de "Ecriture" pour chaque valeur unique de la somme
    # moyennes = df.groupby("Nb variables")[["Ecriture_CP", "Resolution_CP"]].mean().reset_index()

    # plt.plot(moyennes["Nb variables"], moyennes["Ecriture_CP"], marker="o", linestyle="-", color="b", label="Ecriture")
    # plt.plot(moyennes["Nb variables"], moyennes["Resolution_CP"], marker="o", linestyle="-", color="orange", label="Résolution")
    plt.plot(df["Resolution_CP"], marker="o", linestyle="-", color="orange", label="CP")
    plt.plot(df["Resolution_CPM"], marker="o", linestyle="-", color="red", label="CPM")
    plt.plot(df["Resolution_Martin"], marker="o", linestyle="-", color="purple", label="Martin")
    plt.plot(df["Resolution_cycle"], marker="o", linestyle="-", color="blue", label="Cycles")
    plt.xlabel("Instance")
    plt.xticks([df.index.to_list()[i] for i in range(len(df.index.to_list())) if i%10==0], rotation=90)
    plt.ylabel("Durée (en s)")
    plt.title("Durée moyenne de résolution du PL (en s) en fonction de l'instance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./Resultats/{dossier}/temps_resolution.png")
    plt.show()
    plt.clf()

    plt.plot(df["Ecriture_CP"], marker="o", linestyle="-", color="orange", label="CP")
    plt.plot(df["Ecriture_CPM"], marker="o", linestyle="-", color="red", label="CPM")
    plt.plot(df["Ecriture_Martin"], marker="o", linestyle="-", color="purple", label="Martin")
    plt.xlabel("Instance")
    plt.xticks([df.index.to_list()[i] for i in range(len(df.index.to_list())) if i % 10 == 0], rotation=90)
    plt.ylabel("Durée (en s)")
    plt.title("Durée moyenne d'écriture du PL (en s) en fonction de l'instance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./Resultats/{dossier}/temps_ecriture.png")
    plt.show()
    plt.clf()

    plt.plot(df["Objectif_CP"], marker="o", linestyle="-", color="orange", label="CP (témoin)")
    plt.plot(df["Objectif_CPM"], marker="o", linestyle="-", color="red", label="CPM")
    plt.plot(df["Objectif_Martin"], marker="o", linestyle="-", color="purple", label="Martin")
    plt.plot(df["Objectif_cycle"], marker="o", linestyle="-", color="blue", label="Cycles")
    plt.xlabel("Instance")
    plt.xticks([df.index.to_list()[i] for i in range(len(df.index.to_list())) if i % 10 == 0], rotation=90)
    plt.ylabel("Nombre de sommets de branchement")
    plt.title("Solution des différentes méthodes en fonction de l'instance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./Resultats/{dossier}/solutions.png")
    plt.show()
    plt.clf()
    plt.plot(df["Objectif_CP"] - df["Objectif_CP"], marker="o", linestyle="-", color="orange", label="CP (témoin)")
    plt.plot(df["Objectif_CPM"] - df["Objectif_CP"], marker="o", linestyle="-", color="red", label="CPM")
    plt.plot(df["Objectif_Martin"] - df["Objectif_CP"], marker="o", linestyle="-", color="purple", label="Martin")
    plt.plot(df["Objectif_cycle"] - df["Objectif_CP"], marker="o", linestyle="-", color="blue", label="Cycles")
    plt.xlabel("Instance")
    plt.xticks([df.index.to_list()[i] for i in range(len(df.index.to_list())) if i % 10 == 0], rotation=90)
    plt.ylabel("Ecart à la valeur optimale")
    plt.title("Ecart à la solution optimale en fonction de l'instance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./Resultats/{dossier}/solutions_ecart.png")
    plt.show()
    plt.clf()

def valeurs_stats(dossier):
    if not os.path.exists(f"./Resultats/{dossier}/{dossier}.csv"):
        print("Le fichier n'existe pas.")
    else:
        df = pd.read_csv(f"./Resultats/{dossier}/{dossier}.csv", index_col=0)

    liste_index = ["exp", "CP", "CPM", "Martin", "Cycles"]
    liste_colonne = ["Temps de résolution minimum", "Temps de résolution maximum", "Temps de résolution moyen", "Ecart type du temps de résolution",
                     "Temps d'écriture minimum", "Temps d'écriture maximum", "Temps d'écriture moyen", "Ecart type du temps d'écriture",
                     "Ecart minimum", "Ecart maximum", "Ecart moyen", "Ecart type de l'écart"]
    temps_res = pd.DataFrame(index=liste_index, columns=liste_colonne[:4])
    temps_ecriture = pd.DataFrame(index=liste_index, columns=liste_colonne[4:8])
    ecart = pd.DataFrame(index=liste_index, columns=liste_colonne[8:])

    temps_res.loc["CP", "Temps de résolution minimum"] = np.min(df["Resolution_CP"])
    temps_res.loc["CP", "Temps de résolution maximum"] = np.max(df["Resolution_CP"])
    temps_res.loc["CP", "Temps de résolution moyen"] = np.mean(df["Resolution_CP"])
    temps_res.loc["CP", "Ecart type du temps de résolution"] = np.std(df["Resolution_CP"])

    temps_res.loc["CPM", "Temps de résolution minimum"] = np.min(df["Resolution_CPM"])
    temps_res.loc["CPM", "Temps de résolution maximum"] = np.max(df["Resolution_CPM"])
    temps_res.loc["CPM", "Temps de résolution moyen"] = np.mean(df["Resolution_CPM"])
    temps_res.loc["CPM", "Ecart type du temps de résolution"] = np.std(df["Resolution_CPM"])

    temps_res.loc["Martin", "Temps de résolution minimum"] = np.min(df["Resolution_Martin"])
    temps_res.loc["Martin", "Temps de résolution maximum"] = np.max(df["Resolution_Martin"])
    temps_res.loc["Martin", "Temps de résolution moyen"] = np.mean(df["Resolution_Martin"])
    temps_res.loc["Martin", "Ecart type du temps de résolution"] = np.std(df["Resolution_Martin"])

    temps_res.loc["Cycles", "Temps de résolution minimum"] = np.min(df["Resolution_cycle"])
    temps_res.loc["Cycles", "Temps de résolution maximum"] = np.max(df["Resolution_cycle"])
    temps_res.loc["Cycles", "Temps de résolution moyen"] = np.mean(df["Resolution_cycle"])
    temps_res.loc["Cycles", "Ecart type du temps de résolution"] = np.std(df["Resolution_cycle"])

    print(temps_res)

    temps_ecriture.loc["CP", "Temps d'écriture minimum"] = np.min(df["Ecriture_CP"])
    temps_ecriture.loc["CP", "Temps d'écriture maximum"] = np.max(df["Ecriture_CP"])
    temps_ecriture.loc["CP", "Temps d'écriture moyen"] = np.mean(df["Ecriture_CP"])
    temps_ecriture.loc["CP", "Ecart type du temps d'écriture"] = np.std(df["Ecriture_CP"])

    temps_ecriture.loc["CPM", "Temps d'écriture minimum"] = np.min(df["Ecriture_CPM"])
    temps_ecriture.loc["CPM", "Temps d'écriture maximum"] = np.max(df["Ecriture_CPM"])
    temps_ecriture.loc["CPM", "Temps d'écriture moyen"] = np.mean(df["Ecriture_CPM"])
    temps_ecriture.loc["CPM", "Ecart type du temps d'écriture"] = np.std(df["Ecriture_CPM"])

    temps_ecriture.loc["Martin", "Temps d'écriture minimum"] = np.min(df["Ecriture_Martin"])
    temps_ecriture.loc["Martin", "Temps d'écriture maximum"] = np.max(df["Ecriture_Martin"])
    temps_ecriture.loc["Martin", "Temps d'écriture moyen"] = np.mean(df["Ecriture_Martin"])
    temps_ecriture.loc["Martin", "Ecart type du temps d'écriture"] = np.std(df["Ecriture_Martin"])

    print(temps_ecriture)

    ecart.loc["CPM", "Ecart minimum"] = np.min(df["Objectif_CPM"] - df["Objectif_CP"])
    ecart.loc["CPM", "Ecart maximum"] = np.max(df["Objectif_CPM"] - df["Objectif_CP"])
    ecart.loc["CPM", "Ecart moyen"] = np.mean(df["Objectif_CPM"] - df["Objectif_CP"])
    ecart.loc["CPM", "Ecart type de l'écart"] = np.std(df["Objectif_CPM"] - df["Objectif_CP"])

    ecart.loc["Martin", "Ecart minimum"] = np.min(df["Objectif_Martin"] - df["Objectif_CP"])
    ecart.loc["Martin", "Ecart maximum"] = np.max(df["Objectif_Martin"] - df["Objectif_CP"])
    ecart.loc["Martin", "Ecart moyen"] = np.mean(df["Objectif_Martin"] - df["Objectif_CP"])
    ecart.loc["Martin", "Ecart type de l'écart"] = np.std(df["Objectif_Martin"] - df["Objectif_CP"])

    ecart.loc["Cycles", "Ecart minimum"] = np.min(df["Objectif_cycle"] - df["Objectif_CP"])
    ecart.loc["Cycles", "Ecart maximum"] = np.max(df["Objectif_cycle"] - df["Objectif_CP"])
    ecart.loc["Cycles", "Ecart moyen"] = np.mean(df["Objectif_cycle"] - df["Objectif_CP"])
    ecart.loc["Cycles", "Ecart type de l'écart"] = np.std(df["Objectif_cycle"] - df["Objectif_CP"])

    print(ecart)


def set_time_limit(nb_nodes, nb_edges):
    return 120 + nb_nodes + nb_edges