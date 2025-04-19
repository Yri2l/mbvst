# 🌐 MBVST Optimization – Optical Network Project

This repository contains code and results from a research project focused on solving the **Minimum Branch Vertices Spanning Tree (MBVST)** problem, a key challenge in optical network design.

## 📌 Project Overview

The MBVST problem consists in finding a spanning tree of a given undirected graph while **minimizing the number of branching nodes** (nodes with degree > 2). This has direct applications in designing cost-efficient and robust optical networks.

We implemented and compared multiple approaches, including:

- Exact resolution via MILP formulations (Pulp + Cplex)
- Flow-based and Martin formulations
- Cycle basis heuristic
- Edge-level prediction using **Logistic Regression** and **Random Forests**

## 🛠 Tools & Technologies

- Python, Pulp, Cplex
- scikit-learn
- NetworkX, matplotlib

## 📈 Results

- Flow-based MILP models significantly reduce resolution time on medium-to-large graphs
- Predictive models (F1 ~76%) successfully anticipate which edges belong to optimal solutions
- Hybrid approaches can guide heuristics with ML insights

## 📁 Structure

| File/Folder              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `PLNE_CP.py`             | Flow-based MILP formulation                                                 |
| `PLNE_CPM.py`            | Multicommodity flow MILP formulation                                        |
| `PLNE_Martin.py`         | Martin’s formulation                                                        |
| `PLNE_Cycle.py`          | Cycle-based formulation                                                     |
| `PLNE_exp.py`            | Experimental runner for MILP models                                         |
| `edge_prediction.py`     | Main ML pipeline to predict edge inclusion                                  |
| `logistic_regression.py` | Logistic Regression classifier                                              |
| `random_forest.py`       | Random Forest classifier                                                    |
| `Draw.py`                | Graph visualization                                                         |
| `Outils.py`              | Utility functions (graph parsing, stats, etc.)                              |
| `main.py`                | Entry point to orchestrate experiments                                      |
| `solver_Cplex.json`      | Cplex solver settings                                                       |
| `Instances/`             | Graph instances used in experiments                                         |
| `Resultats/`             | Saved outputs and logs                                                      |


## 📃 Reference

For full methodology and analysis, see the [project report (PDF)](link-if-you-publish-it).
