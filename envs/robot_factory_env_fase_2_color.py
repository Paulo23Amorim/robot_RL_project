# robot_factory_env_fase_2_color.py
import gymnasium as gym
from gymnasium import spaces
from graph.graph_data import graph

class RobotFactoryEnvFase2Color(gym.Env):
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
        self.machine_b_input = {'K', 'R'}
        self.machine_b_output = {'L', 'S'}

        self.has_package = False
        self.processed = False
        self.machine_b_state = None
        self.current_part_color = "green"

        self.action_space = spaces.Discrete(max(len(v) for v in self.graph.values()))
        self.observation_space = spaces.MultiDiscrete([len(self.nodes), 2, 2, 3, 2])

        self.last_nodes = []
        self.visit_count = {}
        self.step_counter = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node = self.start_node
        self.has_package = False
        self.processed = False
        self.machine_b_state = None
        self.current_part_color = "green"
        self.last_nodes = []
        self.visit_count = {}
        self.step_counter = 0
        return self.get_state(), {}

    def get_state(self):
        machine_state = 0
        if self.machine_b_state == "K":
            machine_state = 1
        elif self.machine_b_state == "R":
            machine_state = 2
        color_index = 0 if self.current_part_color == "green" else 1
        return (
            self.node_to_index[self.current_node],
            int(self.has_package),
            int(self.processed),
            machine_state,
            color_index
        )

    def step(self, action):
        self.step_counter += 1
        neighbors = self.graph[self.current_node]

        if action >= len(neighbors):
            return self.get_state(), -10, False, False, {}

        next_node, weight = neighbors[action]
        self.current_node = next_node

        reward = -weight / 4000.0
        done = False

        self.visit_count[self.current_node] = self.visit_count.get(self.current_node, 0) + 1
        if self.visit_count[self.current_node] > 4:
            reward -= 50
            done = True

        if self.current_node in self.last_nodes:
            reward -= 5

        self.last_nodes.append(self.current_node)
        if len(self.last_nodes) > 5:
            self.last_nodes.pop(0)

        if not self.has_package and self.current_node in self.pickup_nodes:
            self.has_package = True
            self.processed = False
            self.machine_b_state = None
            self.current_part_color = "green"

        elif self.current_node in self.pickup_nodes and self.has_package:
            reward -= 20

        if self.current_node == "ST1" and self.has_package:
            reward -= 30

        if self.has_package and self.current_node in self.machine_b_input:
            self.has_package = False
            self.machine_b_state = self.current_node

        if not self.has_package and self.current_node in self.machine_b_output:
            if self.machine_b_state == "K" and self.current_node == "L":
                self.has_package = True
                self.processed = True
                self.current_part_color = "blue"
                reward += 200
            elif self.machine_b_state == "R" and self.current_node == "S":
                self.has_package = True
                self.processed = True
                self.current_part_color = "blue"
                reward += 200
            else:
                reward -= 20

        if not self.has_package and self.machine_b_state and self.step_counter > 15:
            reward -= 100
            done = True

        if not self.has_package and self.step_counter > 10:
            reward -= 50

        if self.step_counter > 30 and self.has_package and not self.processed:
            reward -= 100
            done = True

        if self.has_package and self.current_node in self.delivery_nodes:
            if self.processed:
                reward += 300
                done = True
            else:
                reward -= 300
                done = True

        elif not self.has_package and self.current_node in self.delivery_nodes:
            reward -= 20

        return self.get_state(), reward, done, False, {}

    def render(self):
        print(f"Node: {self.current_node} | Has package: {self.has_package} | Processed: {self.processed} | B-State: {self.machine_b_state} | Color: {self.current_part_color}")

    def get_valid_actions(self):
        return list(range(len(self.graph[self.current_node])))
