import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05):
    
        self.q_table = np.zeros((*n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state, valid_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)  
        else:
            q_values = self.q_table[state][valid_actions]
            max_q = np.max(q_values)
            best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state, valid_actions):
        max_future_q = max(self.q_table[next_state][valid_actions]) if valid_actions else 0
        current_q = self.q_table[state][action]
        self.q_table[state][action] = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self, path='q_table.npy'):
        np.save(path, self.q_table)

    def load(self, path='q_table.npy'):
        self.q_table = np.load(path, allow_pickle=False)
