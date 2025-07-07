import gymnasium as gym
from gymnasium import spaces
import numpy as np
from graph.graph_data import graph

class RobotFactoryEnv(gym.Env):
    def __init__(self, start_node='ST1'):
        super(RobotFactoryEnv, self).__init__()

        self.graph = graph
        self.nodes = list(graph.keys())
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        self.index_to_node = {idx: node for node, idx in self.node_to_index.items()}

        self.start_node = start_node
        self.current_node = start_node

        self.pickup_nodes = {'A', 'B', 'C', 'D'}
        self.delivery_nodes = {'AD', 'AE', 'AF', 'AG'}
        self.has_package = False  # Só fica True depois de passar por pickup

        self.action_space = spaces.Discrete(max(len(v) for v in self.graph.values()))
        self.observation_space = spaces.MultiDiscrete([len(self.nodes), 2])  # posição + flag de pacote

        self.last_nodes = []
        self.visit_count = {}
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node = self.start_node
        self.has_package = False
        self.last_nodes = []
        self.visit_count = {}
        return (self.node_to_index[self.current_node], int(self.has_package)), {}

    def step(self, action):
        neighbors = self.graph[self.current_node]

        if action >= len(neighbors):
            # Ação inválida
            reward = -10
            done = False
            return (self.node_to_index[self.current_node], int(self.has_package)), reward, done, False, {}

        next_node, weight = neighbors[action]
        self.current_node = next_node
        
        # Penalização base por custo do caminho
        reward = -weight / 2000.0
        done = False
        
        # 🔁 Verifica se estamos em ciclo
        self.visit_count[self.current_node] = self.visit_count.get(self.current_node, 0) + 1
        if self.visit_count[self.current_node] > 4:
            reward -= 50
            done = True

        # 🆕 Penalização por loop (nó já visitado recentemente)
        if self.current_node in self.last_nodes:
            reward -= 5  # penalização leve

        self.last_nodes.append(self.current_node)
        if len(self.last_nodes) > 5:
            self.last_nodes.pop(0)

        # Apanha a caixa
        if not self.has_package and self.current_node in self.pickup_nodes:
            self.has_package = True
            
        elif self.current_node in self.pickup_nodes and self.has_package:
            reward = -20  # penalização extra por voltar sem necessidade
    
        if self.current_node == "ST1" and self.has_package:
            reward -= 30  # penalização pesada por voltar ao ponto de partida
            
        if self.current_node == "Z" and self.has_package:
            reward -= 25

        # Verifica se chegou ao destino final
        if self.has_package and self.current_node in self.delivery_nodes:
            reward += 200  # sucesso
            done = True
        elif self.current_node in self.delivery_nodes and not self.has_package:
            reward -= 20  # tentou entregar sem ter caixa
            done = False
   
        return (self.node_to_index[self.current_node], int(self.has_package)), reward, done, False, {}

    def render(self):
        state = f"Node: {self.current_node} | Carrying package: {self.has_package}"
        print(state)

    def get_valid_actions(self):
        return list(range(len(self.graph[self.current_node])))
