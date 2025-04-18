import os
import re
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from logistic_regression import load_graph_instances, prepare_data

def load_predictions(graphs, model):
    predictions = {}
    for file, G in graphs:
        graph_predictions = []
        for u, v in G.edges:
            degree_u = G.degree[u]
            degree_v = G.degree[v]
            degree_centrality_u = nx.degree_centrality(G)[u]
            degree_centrality_v = nx.degree_centrality(G)[v]
            common_neighbors = len(list(nx.common_neighbors(G, u, v)))

            features = [
                degree_u, degree_v,
                degree_centrality_u, degree_centrality_v,
                common_neighbors
            ]

            prob = model.predict_proba([features])[0, 1]
            graph_predictions.append((u, v, prob))

        predictions[file] = sorted(graph_predictions, key=lambda x: -x[2])
    return predictions

def find_spanning_tree(predictions, graph):
    tree_edges = []
    parent = {}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for node in graph.nodes:
        parent[node] = node

    for u, v, prob in predictions:
        if find(u) != find(v):
            tree_edges.append((u, v))
            union(u, v)
        if len(tree_edges) == len(graph.nodes) - 1:
            break

    return tree_edges

def compare_to_optimal(solution_tree, optimal_solution):
    solution_set = set(solution_tree)
    optimal_set = set(optimal_solution)

    correct_predictions = solution_set & optimal_set
    precision = len(correct_predictions) / len(solution_set) if solution_set else 0

    print(f"Precision: {precision:.2f}")
    return precision

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

    y = (y > 0.5).astype(int)

    print("Division des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    print("Entraînement de la régression logistique...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Chargement des prédictions...")
    predictions = load_predictions(graphs, model)

    precisions = []
    print("Construction des arbres couvrants prédits...")
    for file, graph in graphs:
        predicted_tree = find_spanning_tree(predictions[file], graph)

        optimal_solution_file = os.path.join(solution_path, file.replace('.txt', ''),
                                             f"sol_PLNE_CP_{file.replace('.txt', '')}.txt")
        optimal_solution = []
        with open(optimal_solution_file, 'r') as f:
            for line in f:
                if line.startswith("Arete"):
                    parts = line.split("=")
                    match = re.search(r"Arete_\((\d+),_(\d+)\)", parts[0])
                    if match and float(parts[1]) > 0.5:
                        optimal_solution.append((int(match.group(1)), int(match.group(2))))

        print(f"Comparaison pour {file}:")
        precision = compare_to_optimal(predicted_tree, optimal_solution)
        precisions.append(precision)

    # Calcul et affichage de la moyenne des précisions
    mean_precision = np.mean(precisions)
    print(f"Moyenne des précisions: {mean_precision:.2f}")
