import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.robot_factory_env_fase_2 import RobotFactoryEnvFase2
from agents.agent_factory import create_agent_for_env
import matplotlib.pyplot as plt
import os

RENDER = False
N_EPISODES = 50000
MAX_STEPS = 100

env = RobotFactoryEnvFase2()
agent, q_table_path = create_agent_for_env(env, fase="fase2")

rewards_per_episode = []
tempos_por_ep = []
tempos_sucesso_ep = []

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
    tempos_por_ep.append(env.tempo_total)
    if env.sucesso:
        tempos_sucesso_ep.append(env.tempo_total)

    print(f"📦 Episódio {episode + 1} → Recompensa: {total_reward:.2f}, passos: {step_counter}, ε: {agent.epsilon:.3f}")

agent.save(q_table_path)

plt.ioff()
plt.figure()
plt.plot(rewards_per_episode)
plt.xlabel("Episódio")
plt.ylabel("Recompensa total")
plt.title("Treino Fase 2 (peça verde + máquina B)")
plt.grid(True)
plt.savefig("fase2/q_learning_fase2_rewards.png")
plt.show()

# Gráfico do tempo por episódio (todos)
plt.figure()
plt.plot(tempos_por_ep)
plt.xlabel("Episódio")
plt.ylabel("Tempo total (s)")
plt.title("Tempo por Episódio – Fase 2")
plt.grid(True)
plt.savefig("fase2/tempo_fase2.png")
plt.close()

# Gráfico do tempo por episódio com sucesso
plt.figure()
plt.plot(tempos_sucesso_ep)
plt.xlabel("Episódios com sucesso")
plt.ylabel("Tempo total (s)")
plt.title("Tempo por Episódio com Sucesso – Fase 2")
plt.grid(True)
plt.savefig("fase2/tempo_sucesso_fase2.png")
plt.close()

