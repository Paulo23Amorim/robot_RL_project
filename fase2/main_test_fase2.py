import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.robot_factory_env_fase_2 import RobotFactoryEnvFase2
from agents.q_learning_agent import QLearningAgent
from graph.visualgraph import render_grafo
import matplotlib.pyplot as plt

RENDER = True
MAX_STEPS = 100

# Inicializar ambiente e agente
env = RobotFactoryEnvFase2()
n_states = (
    env.observation_space.nvec[0],  # posição
    env.observation_space.nvec[1],  # tem pacote
    env.observation_space.nvec[2],  # foi processado
    env.observation_space.nvec[3],  # estado máquina B
)
n_actions = env.action_space.n

agent = QLearningAgent(n_states=n_states, n_actions=n_actions)
agent.load("fase2/q_table_fase2.npy")  
agent.epsilon = 0 

state, _ = env.reset()
done = False
step_counter = 0
total_reward = 0
trajectory = [env.current_node]

if RENDER:
    plt.figure(figsize=(13, 10))

while not done and step_counter < MAX_STEPS:
    if RENDER:
        render_grafo(env.current_node, carrying=env.has_package)

    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions)
    next_state, reward, done, _, _ = env.step(action)

    state = next_state
    total_reward += reward
    step_counter += 1
    trajectory.append(env.current_node)

print(f"🎯 Recompensa final: {total_reward:.2f}")
print(f"🧠 Trajeto final ({len(trajectory)} passos):")
print(" -> ".join(trajectory))
print(f"⏱️ Tempo total do episódio: {env.tempo_total:.2f} segundos")


with open("fase2/trajeto_fase2.txt", "w") as f:
    f.write(" -> ".join(trajectory))
    f.write(f"Recompensa final: {total_reward:.2f}\n")
    f.write(f"Tempo total: {env.tempo_total:.2f} segundos\n")
