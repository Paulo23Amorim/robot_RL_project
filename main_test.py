from envs.robot_factory_env import RobotFactoryEnv
from agents.q_learning_agent import QLearningAgent
from graph.visualgraph import render_grafo
import matplotlib.pyplot as plt

# ⚙️ Configuração
RENDER = True
MAX_STEPS = 100

# 🧠 Inicializar ambiente e agente
env = RobotFactoryEnv()
n_states = env.observation_space.nvec[0]
n_package_states = env.observation_space.nvec[1]
n_actions = env.action_space.n

agent = QLearningAgent(n_states=n_states, n_package_states=n_package_states, n_actions=n_actions)
agent.load("q_table.npy")  # 📥 Carrega a Q-table treinada
agent.epsilon = 0  # 🧠 Greedy (exploração desligada)

# ▶️ Executar episódio
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

# ✅ Resultados
print(f"🎯 Recompensa final: {total_reward:.2f}")
print(f"🧠 Trajeto final ({len(trajectory)} passos):")
print(" -> ".join(trajectory))

# 💾 Guarda trajeto
with open("trajeto_final.txt", "w") as f:
    f.write(" -> ".join(trajectory))
