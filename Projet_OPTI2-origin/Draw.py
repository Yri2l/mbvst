import sys

import networkx as nx
import matplotlib.pyplot as plt
import os
import shutil
from Outils import *

# from PLNE_exp2 import creation_graphe

def draw_graph(G, node_colors, node_size, print_arbre=False, save=False, instance=None, methode=""):
    dossier, nom = instance.split("/")
    nodes = G.nodes()
    edges = G.edges()
    # node_colors = [G[v]['color'] for v in nodes]
    edge_colors = []
    weights = []
    if print_arbre:
        Arbre = nx.Graph()
        Arbre.add_nodes_from(G)
    for u, v in edges:
        edge_colors.append(G[u][v]['color'])
        weights.append(G[u][v]['weight'])
        if print_arbre and (G[u][v]['color']) == "red":
            Arbre.add_edge(u, v, weight=G[u][v]['weight'])
            Arbre.edges[u, v]['color'] = G[u][v]['color']
    # weights = [1 for u, v in edges]
    labels = {i: i for i in range(1,G.number_of_nodes()+1)}
    try:
        pos = nx.planar_layout(G, seed=1) if print_arbre else nx.planar_layout(Arbre, dim=2)
    except:
        pos = nx.spring_layout(G, seed=1)
    # else:
    #     nx.draw(G, pos=pos, node_color=node_colors, node_size=node_size, labels=labels, edge_color=edge_colors, width=weights)
    if print_arbre:
        nx.draw(Arbre, pos=pos, node_color=node_colors, node_size=node_size, labels=labels, edge_color="red", width=2)
    else:
        nx.draw(G, pos=pos, node_color=node_colors, node_size=node_size, labels=labels, edge_color=edge_colors, width=weights)

    if save and not instance == None:
        if methode != "":
            if not os.path.exists(f"./Solutions/{dossier}/{nom[:-4]}"):
                os.makedirs(f"./Solutions/{dossier}/{nom[:-4]}")
            plt.savefig(f"./Solutions/{dossier}/{nom[:-4]}/sol_{methode}_{nom[:-4]}.png")
        else:
            plt.savefig(f"./Solutions/{dossier}/sol_{nom}.png")
    plt.show()

def draw_solution(instance, methode, print_arbre=False, save=False):
    _, nom = instance.split("/")
    path = f"./Solutions/{instance[:-4]}/sol_{methode}_{nom}"
    if not os.path.exists(path):
        raise Exception("Solution does not exist")
    # version_orientee = methode in ["PLNE_CP", "PLNE_CPM"]
    # G = creation_graphe(instance, version_orientee=version_orientee)
    G = creation_graphe(instance)
    for e in G.edges():
        G.edges[e]['color'] = "black"
        G.edges[e]['weight'] = 1

    node_colors = ["blue"] * G.number_of_nodes()
    node_size = [100] * G.number_of_nodes()

    file = open(path, "r", encoding="utf8")
    lines = file.readlines()
    file.close()
    objectif = lines[1]
    lines = lines[3:]
    for line in lines:
        valeur = int(line[-4:-3])
        if line[:5] == "Arete":
            e = chaine_to_tuple(line)
            if valeur == 1:
                G.edges[min(e), max(e)]['color'] = "red"
                G.edges[min(e), max(e)]['weight'] = 2

        elif line[:6] == "Sommet": # if line[:6] == "Sommet":
            v = int(line[7:line.index("=")-1])
            node_colors[v-1] = "orange" if valeur == 1 else "lightsteelblue"
            node_size[v-1] = 250 if valeur == 1 else 100
    draw_graph(G, node_colors, node_size, print_arbre=print_arbre, save=save, instance=instance, methode=methode)



if __name__ == '__main__':
    # draw_solution(sys.argv[1], sys.argv[2], save=True)
    # nom = "Spd_RF2_20_27_211"
    # instance = f"/{nom}/{nom}.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_20_27_211.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_20_27_219.txt"
    instance = "Spd_Inst_Rid_Final2/Spd_RF2_20_27_243.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_1_1_1.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_60_71_1011.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_400_429_4611.txt"
    # instance = "Test.txt"
    # methode = "PLNE_exp"
    # methode = "PLNE_CP"
    # methode = "PLNE_CPM"
    # methode = "PLNE_Martin"
    # methode = "PLNE_Martin_ChatGPT"
    # methode = "PLNE_Cycle"
    # methode = "PLNE_Cycle_backup"
    methode = "PLNE_Cycle4_bis"
    # methode = "PLNE_Cycle2"
    draw_solution(instance, methode, print_arbre=False, save=True)






