import os
import matplotlib.pyplot as plt
from envs.robot_factory_env import RobotFactoryEnv
from agents.agent_factory import create_agent_for_env
from PyQt5.QtWidgets import QApplication

def treinar(epsilon, epsilon_decay, n_episodes, grafico_recompensa=None):
    print("[INFO] Executando treino na Fase 1")
    print(f"epsilon = {epsilon}, decay = {epsilon_decay}, episódios = {n_episodes}")

    MAX_STEPS = 100

    env = RobotFactoryEnv()
    agent, q_table_path = create_agent_for_env(env, fase="fase1")

    rewards_per_episode = []
    tempos_por_ep = []
    tempos_sucesso_ep = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_counter = 0

        while not done and step_counter < MAX_STEPS:

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
        if grafico_recompensa:
            grafico_recompensa.atualizar(total_reward)
            QApplication.processEvents()


        print(f"\U0001F4E6 Episódio {episode + 1} → Recompensa: {total_reward:.2f}, passos: {step_counter}, ε: {agent.epsilon:.3f}")

    agent.save(q_table_path)

    os.makedirs("fase1", exist_ok=True)

    plt.ioff()
    plt.figure()
    plt.plot(rewards_per_episode)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa total")
    plt.title("Treino Fase 1 (caixa azul)")
    plt.grid(True)
    plt.savefig("fase1/q_learning_fase1_rewards.png")
    plt.close()

    plt.figure()
    plt.plot(tempos_por_ep)
    plt.xlabel("Episódio")
    plt.ylabel("Tempo total (s)")
    plt.title("Tempo por Episódio – Fase 1")
    plt.grid(True)
    plt.savefig("fase1/tempo_fase1.png")
    plt.close()

    plt.figure()
    plt.plot(tempos_sucesso_ep)
    plt.xlabel("Episódios com sucesso")
    plt.ylabel("Tempo total (s)")
    plt.title("Tempo por Episódio com Sucesso – Fase 1")
    plt.grid(True)
    plt.savefig("fase1/tempo_sucesso_fase1.png")
    plt.close()

if __name__ == "__main__":
    treinar()
