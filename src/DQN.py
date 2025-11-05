import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os
import random


metrics = {
    'episode_rewards': [],
    'episode_lengths': [],
    'successful_episode': [],
    'exploration_rate': []
}

# Tunable parameters
learning_rate = 0.001
epsilon_decay = 1.2
starting_e = 1.0 
discount_factor = 0.99
episodes = 500000
max_episode_steps = 300

# Execution flags
is_training = False
is_slippery = True

# use the date to timestamp the folder
now = datetime.now()
date = now.strftime("%Y-%m-%d__%H-%M-%S")

# path variable
path = f"Q_table/Lasts/{date}"

def run(is_training = True, is_slippery = False):
    
    # environment initialization
    env = gym.make("CliffWalking-v1", max_episode_steps=max_episode_steps, is_slippery=is_slippery)#, render_mode='human' if not is_training else None)
    episodi = episodes
    epsilon = starting_e
    
    # if is_training initialize the q_table
    if(is_training):
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
    # otherwise use a trained table from the inserted path
    else:
        f = open(f'{path}/cliff_walking.pkl', 'rb')
        q_table = pickle.load(f)
        f.close()
        episodi = 10000
    
    for i in range(episodi):
        # start from initial state 
        state = env.reset()[0]
        terminated = False
        truncated = False

        episode_length = 0
        episode_reward = 0
        
        while(not terminated and not truncated):
            
            # if is_training use the epsilon greedy strategy
            if is_training and random.random() < epsilon:
                action = env.action_space.sample()
            # otherwise pick always the best action 
            else:
                action = np.argmax(q_table[state,:])
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            # if is training update the q_table
            if is_training:
                q_table[state, action] = q_table[state, action] + learning_rate * (
                    reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state,action])
            
            # update useful metrics for the plots
            episode_reward += reward
            episode_length += 1
            
            #update the new state
            state = new_state
        
        # epsilon decay
        epsilon = max(epsilon - (epsilon_decay/episodes), 0.001)
        
        # update metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        # state 47 is the goal state
        metrics['successful_episode'].append(1 if terminated and state == 47 else 0)
        if is_training:
            metrics['exploration_rate'].append(epsilon)
            if i % 250 == 0:
                    print(f"Episode {i}: epsilon={epsilon:.3f}, episode_reward={episode_reward}")

        
    env.close()

    # if is_training save the q_table
    if is_training:
        plot_metrics()
        f = open(f"{path}/cliff_walking.pkl","wb")
        pickle.dump(q_table, f)
        f.close()

def plot_metrics():
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig = plt.figure(figsize=(16, 10))

    fig.suptitle(f"learning_rate={learning_rate}, epsilon_decay={epsilon_decay}, discount_factor={discount_factor}, episodes={episodes}, max_episode_steps={max_episode_steps}", fontsize=20, y=1.02)

    ax1 = plt.subplot(2, 2, 1)
    rewards = np.array(metrics['episode_rewards'])
    plt.plot(rewards, alpha=0.6, label='Episode Reward', color='blue')

    # Moving average
    if len(rewards) > 50:
        window_size = 50
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(
            range(window_size - 1, len(rewards)),
            moving_avg,
            color='red',
            linewidth=2,
            label=f'Last Moving Average ({window_size}): {np.mean(rewards[-window_size:])}'
        )
        

    # Optimal reward
    optimal_value = -13
    plt.axhline(y=optimal_value, color='orange', linestyle='--', linewidth=2, label='Optimal value')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Success Rate
    ax2 = plt.subplot(2, 2, 2)
    success = np.array(metrics['successful_episode'])
    plt.plot(success, color='green', alpha=0.5, label='Success (raw)')

    if len(success) > 50:
        window_size = 50
        moving_avg = np.convolve(success, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(success)), moving_avg,
                color='darkgreen', linewidth=2, label=f'Moving Avg ({window_size}):')
    plt.xlabel('Episode')
    plt.ylabel('Is successful')
    plt.title('Successful episodes')
    plt.grid(True, alpha=0.3)
    
    # 3. Episode Length
    ax3 = plt.subplot(2, 2, 3)
    lengths = np.array(metrics['episode_lengths'])
    plt.plot(lengths, alpha=0.5, color='orange', label='Length (raw)')

    if len(lengths) > 50:
        window_size = 50
        moving_avg = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(lengths)), moving_avg,
            color='red', linewidth=2, label=f'Moving Avg ({window_size})')

    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Length')
    plt.grid(True, alpha=0.3)
    
    # 4. Exploration Rate (Epsilon)
    ax4 = plt.subplot(2, 2, 4)
    plt.plot(metrics['exploration_rate'], color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate Decay')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.savefig(f'{path}/Q-table_metrics.png', dpi=300, bbox_inches='tight')

    print(f"\nPlots saved'")

if not is_training:
    path = input("Path to test: ")
else:
    os.makedirs(path, exist_ok=True)

    
start = time.perf_counter()
# Run training and plot graphs or test 
run(is_training, is_slippery)

end = time.perf_counter()

# execution time
elapsed = end - start

# save metrics in a text file
if is_training:
    with open(f"{path}/results.txt", "a") as f:
        f.write(f"Time to finish: {elapsed:.4f} s\nlearning_rate={learning_rate}\nepsilon_decay={epsilon_decay}\ndiscount_factor={discount_factor}\nepisodes={episodes}\nmax_episode_steps={max_episode_steps}")
        if metrics['episode_rewards']:
            final_avg_reward = np.mean(metrics['episode_rewards'][-100:])
            f.write(f"\n=== Final performances ===\n")
            f.write(f"Last 100 episodes average reward: {final_avg_reward:.2f}\n")
        if metrics['successful_episode']:
            success_rate = np.mean(metrics['successful_episode'][-100:]) * 100
            f.write(f"Last 100 episodes Success rate: {success_rate:.1f}%\n")
else:
    print(f"Average Reward: {np.mean(metrics['episode_rewards'])}")
    print(f"Succesful episodes: {np.count_nonzero(metrics['successful_episode'])}")
print("Finish")
