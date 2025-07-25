import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.robot_factory_env import RobotFactoryEnv
from agents.agent_factory import create_agent_for_env
import matplotlib.pyplot as plt
import os
import random


RENDER = False
N_EPISODES = 15000
MAX_STEPS = 100

env = RobotFactoryEnv()
n_states = env.observation_space.nvec[0]
n_package_states = env.observation_space.nvec[1]
n_actions = env.action_space.n

agent, q_table_path = create_agent_for_env(env, fase="fase1")

rewards_per_episode = []
tempo_por_ep = []
tempo_sucesso_ep = []

#treino
for episode in range(N_EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0
    step_counter = 0

    if RENDER:
        plt.figure(figsize=(13, 10))

    while not done and step_counter < MAX_STEPS:
        if RENDER:
            from graph.visualgraph import render_grafo
            render_grafo(env.current_node, carrying=env.has_package)

        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        next_state, reward, done, _, _ = env.step(action)

        next_valid_actions = env.get_valid_actions()
        agent.update(state, action, reward, next_state, next_valid_actions)

        state = next_state
        total_reward += reward
        step_counter += 1

    agent.decay_epsilon()
    rewards_per_episode.append(total_reward)
    tempo_por_ep.append(env.tempo_total)
    if env.sucesso:
        tempo_sucesso_ep.append(env.tempo_total)

    print(f"📦 Episódio {episode+1} → Recompensa: {total_reward:.2f}, passos: {step_counter}, ε: {agent.epsilon:.3f}")

os.makedirs(os.path.dirname(q_table_path), exist_ok=True)
agent.save(q_table_path)


plt.ioff()
plt.figure()
plt.plot(rewards_per_episode)
plt.xlabel("Episódio")
plt.ylabel("Recompensa total")
plt.title("Treino Fase 1 (caixa azul)")
plt.grid(True)
plt.savefig("fase1/q_learning_fase1_rewards.png")
plt.show()

plt.figure()
plt.plot(tempo_por_ep)
plt.xlabel("Episódio")
plt.ylabel("Tempo total (s)")
plt.title("Tempo por Episódio – Fase 1")
plt.grid(True)
plt.savefig("fase1/tempo_fase1.png")
plt.close()

plt.figure()
plt.plot(tempo_sucesso_ep)
plt.xlabel("Episódio com sucesso")
plt.ylabel("Tempo total (s)")
plt.title("Tempo por Episódio com Sucesso – Fase 1")
plt.grid(True)
plt.savefig("fase1/tempo_sucesso_fase1.png")
plt.close()
