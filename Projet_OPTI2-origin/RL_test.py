import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils
from sympy.physics.units import momentum
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import random
from collections import deque
import numpy as np
from Outils import *

# ============================
# 1. Modèle DQN avec GNN
# ============================

class SharedEmbeddingDQNGNN(nn.Module):
    def __init__(self, node_input_dim, hidden_dim, edge_input_dim):
        super(SharedEmbeddingDQNGNN, self).__init__()
        # GCN layers pour encoder les features des noeuds
        self.gcn1 = GCNConv(node_input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # (Facultatif) Traitement des attributs des arêtes
        self.edge_processor = nn.Linear(edge_input_dim, hidden_dim) if edge_input_dim > 0 else None

        # Décodage final pour calculer Q(s, a)
        self.q_decoder = nn.Linear(2 * hidden_dim, 1)  # Combinaison des paires de noeuds

    def forward(self, graph_data, valid_actions):
        """
        Arguments :
        - graph_data : Objet torch_geometric (avec x, edge_index, edge_attr).
        - valid_actions : Liste des arêtes (u, v) où calculer Q(s, a).

        Retourne :
        - q_values : Tensor des Q-valeurs pour les actions valides.
        """
        # Encodage des noeuds avec GCN
        x = F.relu(self.gcn1(graph_data.x, graph_data.edge_index))
        x = F.relu(self.gcn2(x, graph_data.edge_index))

        q_values = []
        for u, v in valid_actions:
            node_embedding = torch.cat([x[u], x[v]], dim=-1)  # Combinaison des embeddings des noeuds
            q_values.append(self.q_decoder(node_embedding))

        return torch.cat(q_values, dim=0)

# ============================
# 2. Agent DQN
# ============================

class GraphEmbeddingDQNAgent:
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, gamma=0.99, epsilon=0.1, lr=1e-4):
        self.gamma = gamma  # Facteur de discount
        self.epsilon = epsilon  # Probabilité d'exploration
        self.lr = lr  # Taux d'apprentissage

        # Modèles principal et cible
        self.model = SharedEmbeddingDQNGNN(node_input_dim, hidden_dim, edge_input_dim)
        self.target_model = SharedEmbeddingDQNGNN(node_input_dim, hidden_dim, edge_input_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), momentum=0.1, lr=self.lr)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.1, nesterov=True, lr=self.lr)

    def choose_action(self, graph_data, valid_actions, mask):
        """
        Choisit une action avec epsilon-greedy.
        """
        filtered_actions = [a for a in valid_actions if mask[a]]  # Filtrer les actions masquées
        if not filtered_actions:
            return None  # Si aucune action valide, retourner None

        if np.random.rand() < self.epsilon:
            return random.choice(filtered_actions)  # Exploration
        with torch.no_grad():
            q_values = self.model(graph_data, filtered_actions)
            return filtered_actions[q_values.argmax().item()]  # Exploitation

    def choose_action_old(self, graph_data, valid_actions):
        """
        Choisit une action avec epsilon-greedy.
        """
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)  # Exploration
        with torch.no_grad():
            q_values = self.model(graph_data, valid_actions)
            return valid_actions[q_values.argmax().item()]  # Exploitation

    def train_on_batch(self, batch):
        """
        Entraîne le modèle principal sur un mini-lot d'expériences.
        """
        for state, action, reward, next_state, valid_actions, done in batch:
            # Obtenir Q(s, a) actuel
            q_value = self.model(state, [action])[0]

            # Calculer la cible
            with torch.no_grad():
                if done:
                    target = reward
                else:
                    next_q_values = self.target_model(next_state, valid_actions)
                    target = reward + self.gamma * next_q_values.max().item()

            # Calculer la perte et rétropropager
            # print(q_value)
            target = torch.tensor(target).float()
            # print(target)
            loss = F.mse_loss(q_value, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# ============================
# 3. Environnement
# ============================

class DynamicGraphEnv:
    def __init__(self, graph):
        self.graph = graph
        self.edges = list(graph.edges)
        self.state = set()
        self.done = False

    def reset(self, new_graph=None):
        """
        Réinitialise l'environnement.
        """
        if new_graph is not None:
            self.graph = new_graph
            self.edges = list(new_graph.edges)
        self.state = set()
        self.done = False
        return self.get_state(), self.edges

    def get_state(self):
        """
        Retourne l'état courant au format torch_geometric.
        """
        edge_index = torch.tensor([[u, v] for u, v in self.graph.edges], dtype=torch.long).t().contiguous()
        # x = torch.tensor([self.graph.degree(n) for n in self.graph.nodes], dtype=torch.float).view(-1, 1)
        # edge_attr = None

        x = torch.tensor([list(self.graph.nodes[node].values()) for node in self.graph.nodes], dtype=torch.float)
        edge_attr = torch.tensor([list(self.graph.edges[edge].values()) for edge in self.graph.edges], dtype=torch.float)
        if edge_attr.nelement() == 0:
            edge_attr = None # Ajoutez des attributs d'arêtes si nécessaire
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # def get_state(self):
    #     """
    #     Retourne l'état courant au format torch_geometric.
    #     """
    #     edge_index = torch.tensor([[u, v] for u, v in self.state], dtype=torch.long).t().contiguous()
    #     # edge_index = torch.tensor(list(self.state), dtype=torch.long).t().contiguous()
    #     degree = [0] * len(self.graph.nodes)
    #     # degree = [self.graph.degree(n) for u, v in self.graph.nodes]
    #     for u, v in self.state:
    #         degree[u] += 1
    #         degree[v] += 1
    #     x = torch.tensor(degree, dtype=torch.float).view(-1, 1)
    #     edge_attr = None
    #     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


    def step(self, action):
        """
        Applique une action et retourne (prochain état, récompense, done).
        """
        edge = action
        if edge in self.state:
            return self.get_state(), -1, True

        self.state.add(edge)
        reward = self.compute_reward()
        tree = nx.Graph(list(self.state))

        if len(tree.edges) == len(self.graph.nodes) - 1 and nx.is_tree(tree):
            self.done = True

        return self.get_state(), reward, self.done

    def compute_reward(self):
        """
        Calculer la récompense basée sur les degrés et la validité de l'arbre.
        """
        tree = nx.Graph(list(self.state))
        # if not nx.is_connected(tree):
        if not nx.is_forest(tree):
            return -10
        degree_violations = sum(1 for _, d in tree.degree if d > 2)**2
        # bonus = 1
        bonus = sum(1 for _, d in tree.degree if d <= 2)
        # if nx.is_tree(tree):
            # bonus += 10
            # bonus += sum(1 for _, d in tree.degree if d <= 2)
        return bonus-degree_violations

# ============================
# 4. Conversion Networkx -> torch_geometric
# ============================

def nx_to_torch_geometric(graph):
    """
    Convertit un graphe Networkx en un objet torch_geometric.data.Data.
    """
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    node_features = torch.tensor([list(graph.nodes[node].values()) for node in graph.nodes], dtype=torch.float)
    edge_attr = torch.tensor([list(graph.edges[edge].values()) for edge in graph.edges], dtype=torch.float)
    if edge_attr.nelement() == 0:
        edge_attr = None  # Ajoutez des attributs d'arêtes si nécessaire
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# ============================
# 5. Entraînement
# ============================

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        Initialiser le buffer avec priorités.
        Args:
            capacity (int): Taille maximale du buffer.
            alpha (float): Paramètre de priorité (entre 0 et 1). Plus il est grand, plus la priorité est importante.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []  # Stocke les priorités

    def add(self, transition, error):
        """
        Ajouter une transition avec sa priorité calculée.
        Args:
            transition (tuple): (state, action, reward, next_state, done).
            error (float): Erreur TD pour cette transition.
        """
        max_priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            # Remplace les anciennes transitions (FIFO)
            self.buffer.pop(0)
            self.priorities.pop(0)
            self.buffer.append(transition)
            self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        """
        Échantillonner un mini-lot en fonction des priorités.
        Args:
            batch_size (int): Taille du mini-lot.
            beta (float): Paramètre pour ajuster les poids d'importance.

        Returns:
            batch (list): Mini-lot de transitions.
            indices (list): Indices des transitions échantillonnées.
            weights (list): Poids d'importance pour corriger les biais.
        """
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Échantillonnage basé sur les probabilités
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in indices]

        # Calcul des poids pour corriger les biais
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return batch, indices, weights

    def update_priorities(self, indices, errors):
        """
        Met à jour les priorités pour les transitions utilisées.
        Args:
            indices (list): Indices des transitions utilisées.
            errors (list): Nouvelles erreurs TD pour ces transitions.
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-6

def train_dqn_on_graphs_avec_PER(agent, graphs, num_episodes=10000, batch_size=64, gamma=0.99, epsilon_decay=0.99, alpha=0.6, beta_start=0.4, beta_increment=0.001):
    """
    Train a DQN agent using a prioritized replay buffer on a list of graphs.
    Args:
        env: Environment for the task.
        agent: DQN agent.
        graphs: List of NetworkX graphs.
        num_episodes (int): Number of training episodes.
        batch_size (int): Batch size for training.
        gamma (float): Discount factor.
        alpha (float): Priority parameter for PER.
        beta_start (float): Initial beta value for importance sampling.
        beta_increment (float): Increment for beta per episode.
    """
    replay_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=alpha)
    beta = beta_start

    rewards_history = []

    for episode in range(int(num_episodes)):
        # Reset environment with a random graph from the list
        graph = random.choice(graphs)
        # torch_graph = nx_to_torch_geometric(graph)
        torch_graph = torch_geometric.utils.from_networkx(graph)
        env = DynamicGraphEnv(graph)
        # env.reset(graph)
        # state = env.get_state()

        state, valid_actions = env.reset()
        mask = {edge: True for edge in valid_actions}  # Masque pour empêcher la réutilisation d'arêtes

        arbre_tmp = nx.Graph()
        arbre_tmp.add_nodes_from(graph.nodes)

        total_reward = 0
        done = False

        while not done:
            # Select action using the policy
            action = agent.choose_action(torch_graph, valid_actions, mask)
            if action is None:
                break  # Arrêter si aucune action valide n'est possible
            u, v = action
            arbre_tmp.add_edge(u, v)
            if not nx.is_forest(arbre_tmp):
                mask[action] = False
                arbre_tmp.remove_edge(u, v)
                continue


            # Perform action in the environment
            next_state, reward, done = env.step(action)

            # Calculate TD error
            with torch.no_grad():
                target_q = reward + gamma * torch.max(agent.target_model(torch.tensor(next_state, dtype=torch.float32)))
                current_q = agent.model(torch.tensor(state, dtype=torch.float32))[action]
                td_error = abs(target_q.item() - current_q.item())

            # Add transition to the replay buffer
            replay_buffer.add((state, action, reward, next_state, done), td_error)

            state = next_state
            total_reward += reward

            # Train the agent if enough samples are available
            if len(replay_buffer.buffer) > batch_size:
                batch, indices, weights = replay_buffer.sample(batch_size, beta)

                # Unpack the batch
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert to tensors
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)
                weights = torch.tensor(weights, dtype=torch.float32)

                # Compute targets
                with torch.no_grad():
                    next_q_values = agent.target_model(next_states)
                    max_next_q_values = torch.max(next_q_values, dim=1)[0]
                    targets = rewards + gamma * max_next_q_values * (1 - dones)

                # Compute Q values and loss
                q_values = agent.model(states)
                predicted_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
                loss = (weights * F.mse_loss(predicted_q_values, targets, reduction='none')).mean()

                # Optimize the model
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

                # Update priorities in the replay buffer
                td_errors = abs(targets.detach().numpy() - predicted_q_values.detach().numpy())
                replay_buffer.update_priorities(indices, td_errors)

        # Decay beta over time
        beta = min(1.0, beta + beta_increment)

        # Track rewards
        rewards_history.append(total_reward)

        # Update target model periodically
        # if episode % agent.target_update_frequency == 0:
        if episode % 10 == 0:
            agent.update_target_model()

        # Print progress
        if episode % 100 == 0:
            agent.epsilon = max(0.01, agent.epsilon * epsilon_decay)
            print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Avg Reward (last 100): {np.mean(rewards_history[-100:]):.2f}")

    plt.plot(rewards_history)
    plt.title(f"Reward total au cours de l'entrainement")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

    return rewards_history

def train_dqn_on_graphs(agent, graph_list, num_episodes=100, batch_size=64, epsilon_decay=0.99, save_model=False):
    replay_buffer = deque(maxlen=10000)
    # # Initialiser le buffer PER
    # replay_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)

    liste_rewards = []
    for episode in range(int(num_episodes)):
        nx_graph = random.choice(graph_list)
        torch_graph = nx_to_torch_geometric(nx_graph)
        # torch_graph = torch_geometric.utils.from_networkx(nx_graph)
        env = DynamicGraphEnv(nx_graph)
        max_steps = nx_graph.number_of_edges()

        state, valid_actions = env.reset()
        mask = {edge: True for edge in valid_actions}  # Masque pour empêcher la réutilisation d'arêtes
        done = False
        total_reward = 0
        arbre_tmp = nx.Graph()
        arbre_tmp.add_nodes_from(nx_graph.nodes)

        # for step in range(max_steps):
        while not done:
            action = agent.choose_action(torch_graph, valid_actions, mask)
            if action is None:
                break  # Arrêter si aucune action valide n'est possible
            u, v = action
            arbre_tmp.add_edge(u, v)
            if not nx.is_forest(arbre_tmp):
                mask[action] = False
                arbre_tmp.remove_edge(u, v)
                continue

            next_state, reward, done = env.step(action)
            total_reward += reward

            replay_buffer.append((state, action, reward, next_state, valid_actions, done))

            # Mettre à jour le masque pour marquer l'action comme utilisée
            mask[action] = False

            state = next_state
            if done:
                break
        liste_rewards.append(total_reward)
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            agent.train_on_batch(batch)

        if episode % 10 == 0:
            # agent.epsilon = max(0.01, agent.epsilon*0.99)
            agent.target_model.load_state_dict(agent.model.state_dict())

        if episode % 100 == 0:
            agent.epsilon = max(0.01, agent.epsilon*epsilon_decay)

            print(f"Épisode {episode + 1}/{num_episodes}, Récompense totale : {total_reward}, epsilon: {agent.epsilon}")
    if save_model:
        save_agent(agent, "DQN_agent.pth")
    plt.plot(liste_rewards)
    opti = ""
    if isinstance(agent.optimizer, torch.optim.Adam):
        opti = f"(optimizer: Adam, lr={agent.optimizer.param_groups[0]['lr']})"
    elif isinstance(agent.optimizer, torch.optim.RMSprop):
        opti = f"(optimizer: RMSprop, lr={agent.optimizer.param_groups[0]['lr']}, momentum={agent.optimizer.param_groups[0]['momentum']})"
    elif isinstance(agent.optimizer, torch.optim.SGD):
        opti = f"(optimizer: SGD, lr={agent.optimizer.param_groups[0]['lr']}, momentum={agent.optimizer.param_groups[0]['momentum']})"
    plt.title(f"Reward total au cours de l'entrainement {opti}")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    if save_model:
        plt.savefig("DQN_training.png")
    plt.show()

def train_dqn_on_graphs_old(agent, graph_list, num_episodes=100, batch_size=64, max_steps=50):
    replay_buffer = deque(maxlen=10000)

    for episode in range(num_episodes):
        nx_graph = random.choice(graph_list)
        torch_graph = nx_to_torch_geometric(nx_graph)
        env = DynamicGraphEnv(nx_graph)

        state, valid_actions = env.reset()
        done = False
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(torch_graph, valid_actions)
            next_state, reward, done = env.step(action)
            total_reward += reward
            replay_buffer.append((state, action, reward, next_state, valid_actions, done))
            state = next_state
            if done:
                break

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            agent.train_on_batch(batch)

        if episode % 10 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        print(f"Épisode {episode + 1}/{num_episodes}, Récompense totale : {total_reward}")

# ============================
# 6. Exemple d'utilisation
# ============================

def prediction_dqn_old(agent, graph):
    torch_graph = nx_to_torch_geometric(graph)
    env = DynamicGraphEnv(graph)
    state, valid_actions = env.reset()
    done = False
    total_reward = 0
    max_steps = graph.number_of_nodes() - 1

    for step in range(max_steps):
        # action = agent.choose_action(torch_graph, valid_actions)
        with torch.no_grad():
            # print(f"valid_actions: {valid_actions}")
            q_values = agent.model(torch_graph, valid_actions)
            # print(f"q_values : {q_values}")
            action = valid_actions[q_values.argmax().item()]  # Exploitation
            # print(f"action {action}")
            valid_actions.remove(action)
        next_state, reward, done = env.step(action)
        total_reward += reward
        # replay_buffer.append((state, action, reward, next_state, valid_actions, done))
        state = next_state
        if done:
            break
    return env.state


def predict_spanning_tree(agent, graph):
    torch_graph = nx_to_torch_geometric(graph)  # Convertir en torch_geometric
    # torch_graph = torch_geometric.utils.from_networkx(graph)
    env = DynamicGraphEnv(graph)  # Initialiser un environnement pour suivre l'état
    state, valid_actions = env.reset()
    mask = {edge: True for edge in valid_actions}  # Masque des actions valides

    spanning_tree_edges = []
    Arbre = nx.Graph()
    Arbre.add_nodes_from(graph.nodes())
    done = False

    while not done:
        action = agent.choose_action(torch_graph, valid_actions, mask)
        if action is None:  # Si aucune action valide, arrêter
            break
        u, v = action
        Arbre.add_edge(u, v)
        if not nx.is_forest(Arbre):
            mask[action] = False
            Arbre.remove_edge(u, v)
            continue
        spanning_tree_edges.append(action)  # Ajouter l'arête sélectionnée
        state, reward, done = env.step(action)
        mask[action] = False  # Désactiver cette arête pour les prochaines itérations

    return spanning_tree_edges

def save_agent(agent, path="DQN_agent.pth"):
    """
    Save the agent's model and optimizer state to a file.
    Args:
        agent (GraphEmbeddingDQNAgent): Trained agent to save.
        path (str): File path to save the agent.
    """
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'target_model_state_dict': agent.target_model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'hyperparameters': {
            'gamma': agent.gamma,
            'epsilon': agent.epsilon,
            'lr': agent.lr
        }
    }, path)
    print(f"Agent saved to {path}")


def load_agent(agent, path="DQN_agent.pth"):
    """
    Load the agent's model and optimizer state from a file.
    Args:
        agent (GraphEmbeddingDQNAgent): Agent to load the state into.
        path (str): File path to load the agent from.
    """
    # agent = GraphEmbeddingDQNAgent(node_input_dim=1, edge_input_dim=0, hidden_dim=64, lr=1e-3, epsilon=0.3)

    if os.path.exists(path):
        checkpoint = torch.load(path)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore hyperparameters
        agent.gamma = checkpoint['hyperparameters']['gamma']
        agent.epsilon = checkpoint['hyperparameters']['epsilon']
        agent.lr = checkpoint['hyperparameters']['lr']

        print(f"Agent loaded from {path}")
    else:
        print(f"No checkpoint found at {path}")


if __name__ == "__main__":
    # Liste de graphes Networkx
    graph_list = [
        nx.erdos_renyi_graph(10, 0.2),
        nx.erdos_renyi_graph(15, 0.3),
        nx.erdos_renyi_graph(20, 0.4),
    ]

    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_20_27_211.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_20_27_219.txt"
    # instance = "Spd_Inst_Rid_Final2/Spd_RF2_60_71_1011.txt"
    liste_instances = [
        "Spd_Inst_Rid_Final2/Spd_RF2_20_27_211.txt",
   ]

    preprocessing=True
    chargement_modele=False

    if preprocessing:
        graph_list = []
        for inst in liste_instances:
            G = creation_graphe(inst, from_zero=True)
            G, nb_node_features, nb_edge_features = preprocessing_graphe(G)
            graph_list.append(G)
    else:
        graph_list = [creation_graphe(inst, from_zero=True) for inst in liste_instances]
        nb_node_features = 1
        nb_edge_features = 0
    print(f"nb_node_features : {nb_node_features}, nb_edge_features : {nb_edge_features}")


    # Initialisation de l'agent
    agent = GraphEmbeddingDQNAgent(node_input_dim=nb_node_features, edge_input_dim=nb_edge_features, hidden_dim=64, lr=1e-3, epsilon=0.3)
    if chargement_modele:
        load_agent(agent, path="DQN_agent.pth")
    else:
        # Entraînement
        train_dqn_on_graphs(agent, graph_list, num_episodes=5e3, batch_size=32, epsilon_decay=0.95, save_model=True)
    # train_dqn_on_graphs_avec_PER(agent, graph_list, num_episodes=5e3, batch_size=64,
    #                              gamma=0.99, epsilon_decay=0.95,
    #                              alpha=0.6, beta_start=0.4, beta_increment=0.001)

    # print(f"Nodes: {graph_list[0].nodes}")
    # print(f"Edges: {graph_list[0].edges}")
    # res = prediction_dqn(agent, graph_list[0])
    # print(res)
    res_aretes = predict_spanning_tree(agent, graph_list[0])
    G = nx.Graph()
    G.add_nodes_from([n+1 for n in graph_list[0].nodes])
    G.add_edges_from([(u+1, v+1) for u, v in res_aretes])

    node_colors = ["orange" if G.degree(node) > 2 else "lightsteelblue" for node in G.nodes]
    node_size = [250 if G.degree(node) > 2 else 100 for node in G.nodes]

    nx.draw(G, node_color=node_colors, node_size=node_size, with_labels=True)
    plt.show()

