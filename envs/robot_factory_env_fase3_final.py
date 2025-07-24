import gymnasium as gym
from gymnasium import spaces
from graph.graph_data import graph
from graph.visualgraph import pos

class RobotFactoryEnvFase3Color(gym.Env):
    def __init__(self, start_node='ST1'):
        super().__init__()

        self.graph = graph
        self.nodes = list(graph.keys())
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        self.index_to_node = {idx: node for node, idx in self.node_to_index.items()}

        self.start_node = start_node
        self.current_node = start_node

        self.pickup_nodes = {'A', 'B', 'C', 'D'}
        self.delivery_nodes = {'AD', 'AE', 'AF', 'AG'}
        self.machine_a_input = {'O', 'V'}
        self.machine_a_output = {'P', 'X'}
        self.machine_b_input = {'K', 'R'}
        self.machine_b_output = {'L', 'S'}

        self.has_package = False
        self.package_color = "red" 
        self.machine_a_state = None
        self.machine_b_state = None
        
        self.tempo_total = 0
        self.sucesso = False
        self.last_nodes_curvas = []


        self.action_space = spaces.Discrete(max(len(v) for v in self.graph.values()))
        self.observation_space = spaces.MultiDiscrete([len(self.nodes), 2, 3, 3, 3])

        self.last_nodes = []
        self.visit_count = {}
        self.step_counter = 0
        
    def tipo_curva(self, n1, n2, n3):
        if n1 not in pos or n2 not in pos or n3 not in pos:
            return "desconhecido"
        x1, y1 = pos[n1]
        x2, y2 = pos[n2]
        x3, y3 = pos[n3]
        v1 = (x2 - x1, y2 - y1)
        v2 = (x3 - x2, y3 - y2)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        if dot > 0 and cross == 0:
            return "linear"
        elif dot < 0 and cross == 0:
            return "curva_180"
        else:
            return "curva_90"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node = self.start_node
        self.has_package = False
        self.package_color = "red"
        self.machine_a_state = None
        self.machine_b_state = None
        self.last_nodes = []
        self.visit_count = {}
        self.step_counter = 0
        self.tempo_total = 0
        self.sucesso = False
        self.last_nodes_curvas = []
        return self.get_state(), {}

    def get_state(self):
        color_map = {"red": 0, "green": 1, "blue": 2}
        a = 0 if self.machine_a_state is None else (1 if self.machine_a_state == 'O' else 2)
        b = 0 if self.machine_b_state is None else (1 if self.machine_b_state == 'K' else 2)
        return (
            self.node_to_index[self.current_node],
            int(self.has_package),
            color_map[self.package_color],
            a,
            b
        )

    def step(self, action):
        self.step_counter += 1
        neighbors = self.graph[self.current_node]

        if action >= len(neighbors):
            return self.get_state(), -10, False, False, {}

        next_node, weight = neighbors[action]
        self.current_node = next_node
        reward = -weight / 2000
        done = False
        
        # Tempo linear (mm -> cm -> tempo a 12 cm/s)
        distancia_cm = weight / 100.0
        self.tempo_total += distancia_cm / 12

        # Tempo de curva
        self.last_nodes_curvas.append(self.current_node)
        if len(self.last_nodes_curvas) > 3:
            self.last_nodes_curvas.pop(0)

        if len(self.last_nodes_curvas) == 3:
            tipo = self.tipo_curva(self.last_nodes_curvas[0], self.last_nodes_curvas[1], self.last_nodes_curvas[2])
            if tipo == "curva_90":
                self.tempo_total += 2
            elif tipo == "curva_180":
                self.tempo_total += 4

        self.visit_count[self.current_node] = self.visit_count.get(self.current_node, 0) + 1
        if self.visit_count[self.current_node] > 6:
            reward -= 50
            done = True

        # if self.current_node in self.last_nodes:
        #     reward -= 3

        self.last_nodes.append(self.current_node)
        if len(self.last_nodes) > 5:
            self.last_nodes.pop(0)

        # Pegou na peça (vermelha)
        if not self.has_package and self.current_node in self.pickup_nodes:
            self.has_package = True
            self.package_color = "red"
            self.tempo_total += 3  # Tempo de carregar
            self.machine_a_state = None
            self.machine_b_state = None
            reward += 10

        # Máquina A - entrada (com peça vermelha)
        if self.has_package and self.package_color == "red" and self.current_node in self.machine_a_input:
            self.has_package = False
            self.tempo_total += 3
            self.machine_a_state = self.current_node

        # Máquina A - saída 
        if not self.has_package and self.package_color == "red" and self.machine_a_state:
            if (self.machine_a_state == 'O' and self.current_node == 'P') or (self.machine_a_state == 'V' and self.current_node == 'X'):
                self.has_package = True
                self.tempo_total += 3
                self.package_color = "green"
                reward += 80
            else:
                reward -= 20

        # Máquina B - entrada (com peça verde)
        if self.has_package and self.package_color == "green" and self.current_node in self.machine_b_input:
            self.has_package = False
            self.tempo_total += 3
            self.machine_b_state = self.current_node

        # Máquina B - saída 
        if not self.has_package and self.package_color == "green" and self.machine_b_state:
            if (self.machine_b_state == 'K' and self.current_node == 'L') or (self.machine_b_state == 'R' and self.current_node == 'S'):
                self.has_package = True
                self.tempo_total += 3
                self.package_color = "blue"
                reward += 100
            else:
                reward -= 20

        # Tentativa de entrega
        if self.has_package and self.package_color == "blue" and self.current_node in self.delivery_nodes:
            reward += 300
            self.tempo_total += 3
            done = True
            self.sucesso = True

        return self.get_state(), reward, done, False, {}

    def render(self):
        print(f"Node: {self.current_node} | Has package: {self.has_package} | Color: {self.package_color} | A-State: {self.machine_a_state} | B-State: {self.machine_b_state}")

    def get_valid_actions(self):
        return list(range(len(self.graph[self.current_node])))
