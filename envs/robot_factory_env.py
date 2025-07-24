import gymnasium as gym
from gymnasium import spaces
from graph.graph_data import graph
from graph.visualgraph import pos


class RobotFactoryEnv(gym.Env):
    def __init__(self, start_node='ST1'):
        super().__init__()

        self.graph = graph
        self.nodes = list(graph.keys())
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        self.index_to_node = {idx: node for node, idx in self.node_to_index.items()}

        self.start_node = start_node
        self.current_node = start_node

        self.pickup_nodes = ['A', 'B', 'C', 'D']
        self.delivery_nodes = ['AD', 'AE', 'AF', 'AG']
        self.total_packages = 4

        self.has_package = False
        self.entregas = [('A', 'AD'), ('B', 'AE'), ('C', 'AF'), ('D', 'AG')]
        self.entrega_atual = 0

        self.tempo_total = 0
        self.sucesso = False
        self.last_nodes_curvas = []

        
        self.delivered_count = 0
        self.packages_done = set()

        self.action_space = spaces.Discrete(max(len(v) for v in self.graph.values()))
        self.observation_space = spaces.MultiDiscrete([len(self.nodes), 2, 5])  # última dimensão é dummy

        self.last_nodes = []
        self.visit_count = {}

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
        self.entrega_atual = 0
        self.delivered_count = 0
        self.tempo_total = 0
        self.sucesso = False
        self.packages_done = set()
        self.last_nodes = []
        self.visit_count = {}
        return self.get_state(), {}


    def _get_obs(self):
        return (self.node_to_index[self.current_node], int(self.has_package), 0)

    def step(self, action):
        neighbors = self.graph[self.current_node]

        if action >= len(neighbors):
            reward = -10
            done = False
            return self.get_state(), reward, done, False, {}

        next_node, weight = neighbors[action]
        self.current_node = next_node
        
        # Converter mm para cm e calcular tempo a 12 cm/s
        distancia_cm = weight / 100.0
        tempo_movimento = distancia_cm / 12
        self.tempo_total += tempo_movimento
        
        # Analisar curva com base em 3 nós
        self.last_nodes_curvas.append(self.current_node)
        if len(self.last_nodes_curvas) > 3:
            self.last_nodes_curvas.pop(0)

        if len(self.last_nodes_curvas) == 3:
            tipo = self.tipo_curva(self.last_nodes_curvas[0], self.last_nodes_curvas[1], self.last_nodes_curvas[2])
            if tipo == "curva_90":
                self.tempo_total += 2
            elif tipo == "curva_180":
                self.tempo_total += 4

        reward = -weight / 1000.0 if not self.has_package else -weight / 1500.0

        done = False

        self.visit_count[self.current_node] = self.visit_count.get(self.current_node, 0) + 1
        if self.visit_count[self.current_node] > 8:
            reward -= 50
            done = True

        if self.current_node in self.last_nodes:
            reward -= 5

        self.last_nodes.append(self.current_node)
        if len(self.last_nodes) > 5:
            self.last_nodes.pop(0)

        entrega_atual = self.entrega_atual
        pickup, entrega = self.entregas[entrega_atual]

        # Tentativa de pegar
        if not self.has_package and self.current_node == pickup:
            self.has_package = True
            self.tempo_total += 3  # Tempo de carregar a peça
            reward += 5
            print(f"📦 Pegou caixa no armazém {pickup}")

        elif self.current_node in self.pickup_nodes and self.has_package:
            reward -= 20  # tentativa inválida de pegar outra caixa

        # Tentativa de entrega
        if self.has_package and self.current_node == entrega:
            reward += 100
            print(f"📤 Entregou caixa em {entrega}")
            self.has_package = False
            self.entrega_atual += 1
            self.tempo_total += 3  # Tempo de carregar a peça
            self.sucesso = True

            if self.entrega_atual >= len(self.entregas):
                done = True  # concluiu todas as entregas

        elif self.current_node in self.delivery_nodes and not self.has_package:
            reward -= 15

        # Penalização por voltar a ST1 com caixa
        if self.has_package and self.current_node == "ST1":
            reward -= 30

        # Penalização por andar em círculos entre ST1/U/N
        if self.has_package and self.last_nodes.count(self.current_node) > 1:
            reward -= 3

        return self.get_state(), reward, done, False, {}



    def get_current_pickup(self):
        # Devolve o último armazém de onde foi recolhida a caixa
        for node in reversed(self.last_nodes):
            if node in self.pickup_nodes and node not in self.packages_done:
                return node
        return None

    def get_state(self):
        return (
            self.node_to_index[self.current_node],
            int(self.has_package),
            self.entrega_atual,
        )

    def render(self):
        print(f"Node: {self.current_node} | Tem caixa: {self.has_package} | Entregues: {self.delivered_count}/4")

    def get_valid_actions(self):
        return list(range(len(self.graph[self.current_node])))
