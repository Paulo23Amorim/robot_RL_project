from envs.robot_factory_env import RobotFactoryEnv
from agents.q_learning_agent import QLearningAgent
import random
import matplotlib.pyplot as plt
from graph.visualgraph import render_grafo

# Define se visualiza os episódios ou não
RENDER = True

env = RobotFactoryEnv()

n_states = env.observation_space.nvec[0]  # número de nós
n_package_states = env.observation_space.nvec[1]  # 2: com ou sem caixa
n_actions = env.action_space.n

agent = QLearningAgent(n_states=n_states, n_package_states=n_package_states, n_actions=n_actions)

max_steps = 100
epsilon_decay = 0.999
epsilon_min = 0.01
agent.epsilon = 0 
n_episodes = 1


rewards_per_episode = []

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    step_counter = 0
    total_reward = 0
    trajectory = []

    if RENDER:
        plt.figure(figsize=(13, 10))

    while not done and step_counter < max_steps:
        if RENDER:
            render_grafo(env.current_node, carrying=env.has_package)

        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        next_state, reward, done, _, _ = env.step(action)

        next_valid_actions = env.get_valid_actions()
        
        if agent.epsilon > 0:
            agent.update(state, action, reward, next_state, next_valid_actions)


        state = next_state
        total_reward += reward
        step_counter += 1
        trajectory.append(env.current_node)

    agent.decay_epsilon()
    print(f"🎯 Episódio {episode+1} terminou com recompensa: {total_reward:.2f}, passos: {step_counter}, ε: {agent.epsilon:.3f}")
    
    rewards_per_episode.append(total_reward)

print("✅ Treino concluído!")

agent.save('q_table.npy')

plt.ioff()
plt.figure()
plt.plot(rewards_per_episode)
plt.xlabel('Episódio')
plt.ylabel('Recompensa total')
plt.title('Recompensa por episódio (Q-learning)')
plt.grid(True)
plt.savefig("q_learning_rewards.png")
plt.show()

print("🧠 Trajeto final:")
print(" -> ".join(trajectory))
