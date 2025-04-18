from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from logistic_regression import load_graph_instances, prepare_data

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

    print("Entraînement de la forêt aléatoire...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=123)
    rf_model.fit(X_train, y_train)

    print("Évaluation du modèle...")
    y_pred = rf_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Rapport de classification:\n", classification_report(y_test, y_pred))

    print("Prédictions sur les arêtes...")
    predictions = rf_model.predict_proba(X_test)[:, 1]
    print("Probabilités des arêtes dans la solution:", predictions)

    # Importance des caractéristiques
    print("\nImportance des caractéristiques (importance relative) :")
    feature_names = [
        "Degré du sommet u",
        "Degré du sommet v",
        "Centralité du sommet u",
        "Centralité du sommet v",
        "Nombre de voisins communs entre u et v"
    ]
    for name, importance in zip(feature_names, rf_model.feature_importances_):
        print(f"{name}: {importance:.4f}")
