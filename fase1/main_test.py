import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.robot_factory_env import RobotFactoryEnv
from agents.q_learning_agent import QLearningAgent
from agents.agent_factory import create_agent_for_env
from graph.visualgraph import render_grafo
import matplotlib.pyplot as plt


RENDER = True
MAX_STEPS = 100

#  Inicializar ambiente e agente
env = RobotFactoryEnv()
n_states = env.observation_space.nvec[0]
n_package_states = env.observation_space.nvec[1]
n_actions = env.action_space.n

agent, q_table_path = create_agent_for_env(env, fase="fase1")
agent.load("fase1/q_table_fase1.npy")  
agent.epsilon = 0  

state, _ = env.reset()
done = False
step_counter = 0
total_reward = 0
trajectory = [env.current_node]

plt.figure(figsize=(13, 10)) if RENDER else None

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

print(f"Recompensa final: {total_reward:.2f}")
print(f"Trajeto final ({len(trajectory)} passos):")
print(" -> ".join(trajectory))
print(f"⏱️ Tempo total do episódio: {env.tempo_total:.2f} segundos")


with open("fase1/trajeto_final.txt", "w") as f:
    f.write(" -> ".join(trajectory))
