import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall, Lava
from minigrid.core.mission import MissionSpace
import time

# ==================
# CUSTOM ENVIRONMENT
# ==================
class Custom12x12Env(MiniGridEnv):
    def __init__(self, **kwargs):
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=12, 
            max_steps=300, 
            **kwargs
        )
        
    @staticmethod
    def _gen_mission():
        return "reach the goal"
        
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        # walls (obstacles)
        self.grid.set(3, 2, Wall())
        self.grid.set(4, 2, Wall())
        self.grid.set(5, 2, Wall())
        self.grid.set(5, 3, Wall())
        self.grid.set(5, 4, Wall())
        self.grid.set(5, 5, Wall())
        self.grid.set(8, 1, Wall())
        self.grid.set(8, 2, Wall())
        self.grid.set(8, 3, Wall())
        self.grid.set(7, 3, Wall())
        self.grid.set(1, 7, Wall())
        self.grid.set(1, 8, Wall())
        self.grid.set(1, 9, Wall())
        self.grid.set(4, 9, Wall())
        self.grid.set(4, 10, Wall())
        self.grid.set(4, 11, Wall())
        self.grid.set(5, 9, Wall())
        self.grid.set(6, 9, Wall())
        self.grid.set(6, 8, Wall())
        self.grid.set(6, 7, Wall())
        self.grid.set(10, 5, Wall())
        self.grid.set(9, 5, Wall())
        
        
        # lava (hazards)
        self.grid.set(2, 4, Lava())
        self.grid.set(2, 5, Lava())
        self.grid.set(3, 4, Lava())
        self.grid.set(3, 5, Lava())
        self.grid.set(8, 8, Lava())
        self.grid.set(8, 7, Lava())
        self.grid.set(9, 8, Lava())
        self.grid.set(9, 7, Lava())

        # start and goal
        self.agent_pos = (1, 1)
        self.agent_dir = 0  # start facing right
        self.put_obj(Goal(), width - 2, height - 2)

gym.register(
    id='MiniGrid-Custom-12x12-v0',
    entry_point='__main__:Custom12x12Env',
)

# ================
# HELPER FUNCTIONS
# ================
def get_state(env):
    pos = env.unwrapped.agent_pos
    direction = env.unwrapped.agent_dir
    return (pos[0], pos[1], direction)

def epsilon_greedy_policy(Q, state, epsilon, n_actions):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state[0], state[1], state[2]])

# ===================
# TRAINING ALGORITHMS
# ===================

# q-learning algorithm
def train_q_learning(env, episodes=2000, alpha=0.1, gamma=0.99, epsilon=1.0):
    n_actions = env.action_space.n
    Q = np.zeros((12, 12, 4, n_actions))
    
    rewards_history = []
    steps_history = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        state = get_state(env)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = epsilon_greedy_policy(Q, state, epsilon, n_actions)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # tiny step penalty to encourage finding the shortest path
            if not terminated:
                reward = -0.01
            else:
                if reward > 0:
                    reward = 10.0 # a big (positive) reward for reaching goal
                else:
                    reward = -10.0 # a big penalty for hitting lava
                    
            next_state = get_state(env)
            
            best_next_action_value = np.max(Q[next_state[0], next_state[1], next_state[2]])
            td_target = reward + gamma * best_next_action_value * (not terminated)
            td_error = td_target - Q[state[0], state[1], state[2], action]
            Q[state[0], state[1], state[2], action] += alpha * td_error
            
            state = next_state
            total_reward += reward
            steps += 1
            
        epsilon = max(0.01, epsilon * 0.995) # a slower decay for more exploration time
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
    return rewards_history, steps_history, Q

# sarsa learning algorithm
def train_sarsa(env, episodes=2000, alpha=0.1, gamma=0.99, epsilon=1.0):
    n_actions = env.action_space.n
    Q = np.zeros((12, 12, 4, n_actions))
    
    rewards_history = []
    steps_history = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        state = get_state(env)
        action = epsilon_greedy_policy(Q, state, epsilon, n_actions)
        
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            next_obs, reward, terminated, truncated, info = env.step(action)

            # tiny step penalty to encourage finding the shortest path
            if not terminated:
                reward = -0.01
            else:
                if reward > 0:
                    reward = 10.0 # a big (positive) reward for reaching goal
                else:
                    reward = -10.0 # a big penalty for hitting lava
                    
            next_state = get_state(env)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon, n_actions)
            
            td_target = reward + gamma * Q[next_state[0], next_state[1], next_state[2], next_action] * (not terminated)
            td_error = td_target - Q[state[0], state[1], state[2], action]
            Q[state[0], state[1], state[2], action] += alpha * td_error
            
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1
            
        epsilon = max(0.01, epsilon * 0.995) # a slower decay for more exploration time
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
    return rewards_history, steps_history, Q

# =========================================
# EXECUTION, TRAINING, PLOTTING & LIVE DEMO
# =========================================
if __name__ == "__main__":
    print("Initializing training environment...")
    train_env = gym.make('MiniGrid-Custom-12x12-v0', highlight=False)
    
    num_episodes = 2000
    
    print(f"Training Q-Learning for {num_episodes} episodes...")
    ql_rewards, ql_steps, Q_ql = train_q_learning(train_env, episodes=num_episodes)
    
    print(f"Training SARSA for {num_episodes} episodes...")
    sarsa_rewards, sarsa_steps, Q_sarsa = train_sarsa(train_env, episodes=num_episodes)
    
    train_env.close()
    print("Training complete! Generating performance graphs...")

    # smoothing helper function
    def smooth(data, window=25):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # plotting
    plt.figure(figsize=(12, 5))

    # reward comparison plot
    plt.subplot(1, 2, 1)
    plt.plot(smooth(ql_rewards), label='Q-Learning', color='blue', alpha=0.8)
    plt.plot(smooth(sarsa_rewards), label='SARSA', color='red', alpha=0.8)
    plt.title('Performance Comparison: Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward (Smoothed)')
    plt.legend()
    plt.grid(True)

    # steps comparison plot
    plt.subplot(1, 2, 2)
    plt.plot(smooth(ql_steps), label='Q-Learning', color='blue', alpha=0.8)
    plt.plot(smooth(sarsa_steps), label='SARSA', color='red', alpha=0.8)
    plt.title('Path Efficiency: Steps per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Steps to Goal (Smoothed)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('rl_comparison_results.png')
    print("Saved graphs as 'rl_comparison_results.png'.")
    
    plt.show()

    # =================
    # LIVE VISUAL DEMOS
    # =================
    
    # demo 1: q-learning agent
    print("\n----------------------------------------")
    print("DEMO 1: Running Q-Learning Agent (Off-Policy)")
    print("----------------------------------------")
    
    demo_env_ql = gym.make('MiniGrid-Custom-12x12-v0', render_mode="human", highlight=False)
    obs, info = demo_env_ql.reset()
    state = get_state(demo_env_ql)
    
    terminated, truncated = False, False
    step_count = 0
    
    while not (terminated or truncated) and step_count < 300:
        demo_env_ql.render()
        action = np.argmax(Q_ql[state[0], state[1], state[2]])
        obs, reward, terminated, truncated, info = demo_env_ql.step(action)
        state = get_state(demo_env_ql)
        step_count += 1
        time.sleep(0.12)
        
    if terminated and reward > 0:
        print(f"Q-Learning Success! Reached goal in {step_count} steps.")
    else:
        print("Q-Learning finished (max steps or hazard hit).")
    demo_env_ql.close()

    # small pause between demos
    time.sleep(1)

    # demo 2: SARSA agent
    print("\n----------------------------------------")
    print("DEMO 2: Running SARSA Agent (On-Policy)")
    print("----------------------------------------")
    
    demo_env_sarsa = gym.make('MiniGrid-Custom-12x12-v0', render_mode="human", highlight=False)
    obs, info = demo_env_sarsa.reset()
    state = get_state(demo_env_sarsa)
    
    terminated, truncated = False, False
    step_count = 0
    
    while not (terminated or truncated) and step_count < 300:
        demo_env_sarsa.render()
        action = np.argmax(Q_sarsa[state[0], state[1], state[2]])
        obs, reward, terminated, truncated, info = demo_env_sarsa.step(action)
        state = get_state(demo_env_sarsa)
        step_count += 1
        time.sleep(0.12)
        
    if terminated and reward > 0:
        print(f"SARSA Success! Reached goal in {step_count} steps.")
    else:
        print("SARSA finished (max steps or hazard hit).")
    demo_env_sarsa.close()
    
    print("\nAll simulations and demos finished successfully!")