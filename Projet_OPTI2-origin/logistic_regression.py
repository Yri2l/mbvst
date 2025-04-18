import os
import re
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import networkx as nx


def load_graph_instances(path, percentage=0.3):
    files = [f for f in os.listdir(path) if f.startswith("Spd")]

    # Trier les fichiers par les trois nombres extraits
    def extract_numbers(file_name):
        match = re.search(r"Spd_RF2_(\d+)_(\d+)_(\d+)", file_name)
        if match:
            return tuple(map(int, match.groups()))
        return (float('inf'), float('inf'), float('inf'))  # Si le format est incorrect

    files.sort(key=extract_numbers)

    num_files = int(len(files) * percentage)
    selected_files = files[:num_files]

    graphs = []
    for file in selected_files:
        graph_path = os.path.join(path, file)
        G = nx.Graph()
        with open(graph_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                u, v, _ = map(int, line.split())
                G.add_edge(u, v)
        graphs.append((file, G))

    print("Fichiers sélectionnés (triés par taille) :")
    for file in selected_files:
        print(file)

    return graphs


def prepare_data(graphs, solution_path):
    data = []
    labels = []
    for file, G in graphs:
        solution_file = os.path.join(solution_path, file.replace('.txt', ''),
                                     f"sol_PLNE_CP_{file.replace('.txt', '')}.txt")

        print(f"Traitement de l'instance : {file}")
        print(f"Fichier solution attendu : {solution_file}")

        if not os.path.exists(solution_file):
            print(f"Fichier de solution manquant : {solution_file}. Ce graphe sera ignoré.")
            continue

        with open(solution_file, 'r') as f:
            solution_edges = [line for line in f if line.startswith("Arete")]
        if len(solution_edges) == 0:
            print(f"Aucune arête trouvée dans {solution_file}. Ce graphe sera ignoré.")
            continue

        solution_dict = {}
        for edge_line in solution_edges:
            parts = edge_line.split("=")
            match = re.search(r"Arete_\((\d+),_(\d+)\)", parts[0])
            if match:
                edge = (int(match.group(1)), int(match.group(2)))
                value = float(parts[1].strip())
                solution_dict[edge] = value
            else:
                print(f"Format incorrect pour l'arête : {parts[0]}")
                continue

        for u, v in G.edges:
            features = [
                G.degree[u], G.degree[v],
                len(list(nx.common_neighbors(G, u, v))),
                nx.shortest_path_length(G, u, v) if nx.has_path(G, u, v) else -1
            ]
            label = solution_dict.get((u, v), 0)
            data.append(features)
            labels.append(label)

    if len(data) == 0 or len(labels) == 0:
        print("Aucune donnée ou étiquette extraite. Vérifiez les fichiers.")
        return np.array([]), np.array([])

    return np.array(data), np.array(labels)



if __name__ == "__main__":
    instance_path = "./Instances/Spd_Inst_Rid_Final2"
    solution_path = "./Solutions/Spd_Inst_Rid_Final2"

    print("Chargement des graphes...")
    graphs = load_graph_instances(instance_path, percentage=0.3)

    print(f"{len(graphs)} graphes chargés.")
    if len(graphs) == 0:
        print("Erreur : Aucun graphe n'a été chargé. Vérifiez le chemin des instances et leur contenu.")
        exit()

    print("Préparation des données...")
    X, y = prepare_data(graphs, solution_path)

    if X.size == 0 or y.size == 0:
        print("Erreur : Données ou étiquettes vides. Fin du script.")
        exit()

    # Convertir les étiquettes en valeurs binaires
    y = (y > 0.5).astype(int)

    print("Division des données en ensembles d'entraînement et de test...")
    print("Taille de X:", X.shape)
    print("Taille de y:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    print("Entraînement de la régression logistique...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Évaluation du modèle...")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Rapport de classification:\n", classification_report(y_test, y_pred))

    print("Prédictions sur les arêtes...")
    predictions = model.predict_proba(X_test)[:, 1]
    print("Probabilités des arêtes dans la solution:", predictions)
