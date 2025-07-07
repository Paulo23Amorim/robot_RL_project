import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_package_states, n_actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.q_table = np.zeros((n_states, n_package_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min


    def select_action(self, state, valid_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)  # exploração
        else:
            # Exploração entre apenas as ações válidas
            q_values = self.q_table[state[0], state[1], valid_actions]
            max_index = np.argmax(q_values)
            return valid_actions[max_index]

    def update(self, state, action, reward, next_state, valid_actions):
        max_future_q = max(self.q_table[next_state[0], next_state[1], valid_actions]) if valid_actions else 0
        current_q = self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
    def save(self, path='q_table.npy'):
        np.save(path, self.q_table)

    def load(self, path='q_table.npy'):
        self.q_table = np.load(path)
