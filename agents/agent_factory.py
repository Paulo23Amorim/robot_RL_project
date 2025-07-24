import os
import numpy as np
from agents.q_learning_agent import QLearningAgent

def create_agent_for_env(env, fase):
    """
    Cria ou carrega um agente Q-learning adaptado ao ambiente.
    :param env: instância do ambiente Gymnasium
    :param fase: "fase1" ou "fase2" (define a estrutura da Q-table e o nome do ficheiro)
    :return: instância de QLearningAgent
    """
    n_actions = env.action_space.n

    if fase == "fase1":
        n_states = (
            env.observation_space.nvec[0],  # posição
            env.observation_space.nvec[1],  # caixa
            env.observation_space.nvec[2],  # entregas feitas (0 a 4)
        )
        q_table_path = "fase1/q_table_fase1.npy"

    elif fase == "fase2":
        n_states = (
            env.observation_space.nvec[0],  # posição
            env.observation_space.nvec[1],  # tem pacote
            env.observation_space.nvec[2],  # foi processado
            env.observation_space.nvec[3],  # estado máquina B
        )
        q_table_path = "fase2/q_table_fase2.npy"
        
    elif fase == "fase2_color":
        n_states = (
            env.observation_space.nvec[0],  # posição
            env.observation_space.nvec[1],  # tem pacote
            env.observation_space.nvec[2],  # foi processado
            env.observation_space.nvec[3],  # estado máquina B
            env.observation_space.nvec[4],  # cor da peça
        )
        q_table_path = "q_table_fase2_color.npy"
                
    elif fase == "fase3_final":
        n_states = (
            env.observation_space.nvec[0],  # posição
            env.observation_space.nvec[1],  # tem pacote
            env.observation_space.nvec[2],  # cor da peça (0=vermelha, 1=verde, 2=azul)
            env.observation_space.nvec[3],  # estado máquina A (0, 1, 2)
            env.observation_space.nvec[4],  # estado máquina B (0, 1, 2)
        )
        q_table_path = "fase3/q_table_fase3_final.npy"

    else:
        raise ValueError("Fase inválida. Use 'fase1', 'fase2' ou 'fase3'.")

    agent = QLearningAgent(n_states=n_states, n_actions=n_actions)

    if os.path.exists(q_table_path):
        agent.load(q_table_path)
        print(f"✅ Q-table carregada de {q_table_path}")
    else:
        print(f"🆕 Novo agente criado para {fase}")

    return agent, q_table_path
