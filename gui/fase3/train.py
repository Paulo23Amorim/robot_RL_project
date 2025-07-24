import os
import matplotlib.pyplot as plt
from envs.robot_factory_env_fase3_final import RobotFactoryEnvFase3Color
from agents.agent_factory import create_agent_for_env

def treinar(epsilon=1.0, epsilon_decay=0.9995, n_episodes=5000):
    print("[INFO] Executando treino na Fase 3")
    print(f"epsilon = {epsilon}, decay = {epsilon_decay}, episódios = {n_episodes}")

    RENDER = False
    MAX_STEPS = 100

    env = RobotFactoryEnvFase3Color()
    agent, q_table_path = create_agent_for_env(env, fase="fase3_final")

    rewards_per_episode = []
    tempos_por_ep = []
    tempos_sucesso_ep = []

    for episode in range(n_episodes):
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

        agent.epsilon = max(agent.epsilon * epsilon_decay, 0.01)
        rewards_per_episode.append(total_reward)
        tempos_por_ep.append(env.tempo_total)
        if env.sucesso:
            tempos_sucesso_ep.append(env.tempo_total)

        print(f"📦 Episódio {episode + 1} → Recompensa: {total_reward:.2f}, passos: {step_counter}, ε: {agent.epsilon:.3f}")

    agent.save(q_table_path)

    os.makedirs("fase3", exist_ok=True)

    plt.ioff()
    plt.figure()
    plt.plot(rewards_per_episode)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa total")
    plt.title("Treino Fase 3 (com cor da peça)")
    plt.grid(True)
    plt.savefig("fase3/q_learning_fase3_color_rewards.png")
    plt.close()

    plt.figure()
    plt.plot(tempos_por_ep)
    plt.xlabel("Episódio")
    plt.ylabel("Tempo total (s)")
    plt.title("Tempo por Episódio – Fase 3")
    plt.grid(True)
    plt.savefig("fase3/tempo_fase3.png")
    plt.close()

    plt.figure()
    plt.plot(tempos_sucesso_ep)
    plt.xlabel("Episódios com sucesso")
    plt.ylabel("Tempo total (s)")
    plt.title("Tempo por Episódio com Sucesso – Fase 3")
    plt.grid(True)
    plt.savefig("fase3/tempo_sucesso_fase3.png")
    plt.close()
