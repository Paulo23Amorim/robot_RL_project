import sys
import os
import matplotlib.pyplot as plt
from envs.robot_factory_env import RobotFactoryEnv
from agents.q_learning_agent import QLearningAgent
from graph.visualgraph import render_grafo

def testar():
    print("[INFO] Executando teste na Fase 1")

    MAX_STEPS = 100
    RENDER = True

    env = RobotFactoryEnv()
    n_states = env.observation_space.nvec[0]
    n_actions = env.action_space.n

    agent = QLearningAgent(n_states=n_states, n_actions=n_actions)
    agent.load("fase1/q_table_fase1.npy")
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

    print(f"\n🎯 Recompensa final: {total_reward:.2f}")
    print(f"🧠 Trajeto final ({len(trajectory)} passos):")
    print(" -> ".join(trajectory))
    print(f"⏱️ Tempo total do episódio: {env.tempo_total:.2f} segundos")

    with open("fase1/trajeto_final.txt", "w") as f:
        f.write(" -> ".join(trajectory))

if __name__ == "__main__":
    testar()
