import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from datetime import datetime
import time
import os
import json
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Tunable hyperparameters
learning_rate_a = 0.001         # (alpha)
discount_factor_g = 0.99         # (gamma)
mini_batch_size = 32            # size of the training dataset sampled each time from the replay memory  32, 64, 128
epsilon_decay = 1.5             # CAPISCI COME DEVI CHIAMARE QUESTA VARIABILE O COME DEVI COMMENTARLA
episodes = 250
memory = 256                    # replay buffer size
max_episode_steps = 500         # max step number for each episode
nn_nodes = 32

# Training or testing mode
is_training = False

now = datetime.now()
date = now.strftime("%Y-%m-%d__%H-%M-%S")
path = f"DQN/Deterministics/{date}_500"

# Define model
class DQN(nn.Module):
    """
    Network to calculate Q-function
    """
    def __init__(self, input, nodes, output):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, output)
        )
        self.in_features = input
        
    def forward(self, x):
        return self.network(x)

class ReplayMemory():
    """
    Replay buffer to manage experience
    """
    def __init__(self, memory):
        self.memory = deque([], maxlen=memory)

    # Store new transition
    def store(self, transition):
        self.memory.append(transition)

    # Randomly sample a batch 
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Current size memory
    def __len__(self):
        return len(self.memory)
    

# FrozeLake Deep Q-Learning
class CliffWalkingDQL():
    
    def __init__(self):
        super().__init__()
        # materics
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'successful_episode': [],
            'exploration_rate': [],
            'loss_history': [],
            'q_values_mean': [],
            'q_values_std': [],
            'steps_to_goal': []
        }

    # Neural Network
    loss_fn = nn.MSELoss()
    optimizer = None
    
    # Train the FrozeLake environment
    def train(self, episodes, memory, render=False, is_slippery=False):
        # Create FrozenLake instance
        env = gym.make('CliffWalking-v1', max_episode_steps=max_episode_steps, is_slippery=is_slippery, render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        # Initialize epsilon to choose a random action at the beginning
        epsilon = 1

        memory = ReplayMemory(memory)

        # Initialize the network. Number of nodes in the hidden layer can be adjusted.
        q_network = DQN(num_states, nn_nodes, num_actions).to(device)
        self.optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate_a)

        for i in tqdm.tqdm(range(episodes)):
            state = env.reset()[0]     # Initialize to state 0
            terminated = False          # True when agent falls in the cliff or reaches the goal
            truncated = False           # True when agent takes more than max_step actions
            
            # Metrics
            episode_length = 0
            episode_reward = 0
            episode_loss = []

            while(not terminated and not truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action            
                    with torch.no_grad():
                        state_tensor = torch.eye(num_states)[state].to(device)
                        action = q_network(state_tensor).argmax().item()
                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)
                
                episode_reward += reward
                # Save experience into memory
                memory.store((state, action, new_state, reward, terminated))
                
                # Check if enough experience has been collected to start training the network
                if len(memory)>mini_batch_size:
                    mini_batch = memory.sample(mini_batch_size)
                    loss = self.optimize(mini_batch, q_network)        
                    episode_loss.append(loss)
                    
                # Move to the next state
                state = new_state
                # Increment step counter
                episode_length+=1
            
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['episode_lengths'].append(episode_length)
            self.metrics['successful_episode'].append(1 if terminated and state == 47 else 0)
            self.metrics['exploration_rate'].append(epsilon)
            self.metrics['loss_history'].append(np.mean(episode_loss) if episode_loss else 0)

            # Decay epsilon
            epsilon = max(epsilon - (epsilon_decay / episodes), 0.001)
            
            # Useful print during the training
            if i % 250 == 0:
                print(f"Episode {i}: epsilon={epsilon:.3f}, episode_reward={episode_reward}")

        # Close environment
        env.close()

        # At the end of the training, save the nn
        torch.save(q_network.state_dict(), f"{path}/CliffWalking-dqn.pt")
        
        # Plots metrics
        self.plot_comprehensive_metrics()
        
        
    def optimize(self, mini_batch, q_network):
        num_states = q_network.in_features

        states, actions, new_states, rewards, dones = zip(*mini_batch) 
        
        # mini batch to tensors
        states_tensor = torch.eye(num_states)[list(states)].to(device)
        new_states_tensor = torch.eye(num_states)[list(new_states)].to(device)    
        
        actions_tensor = torch.LongTensor(actions).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        terminated_tensor = torch.BoolTensor(dones).to(device)
        
        # Target value evaluated on new_state based on experience
        with torch.no_grad():
            next_q_values = q_network(new_states_tensor).max(dim=1).values          # pick the higher value
            targets = rewards_tensor + discount_factor_g * next_q_values * (1 - terminated_tensor.float())  # Evaluate the target value 
        
        # Q-value of the state where the agent is (by current network)
        current_q_values = q_network(states_tensor)
        predicted_q = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Loss between predicted value (state) and target value (new_state)
        loss = self.loss_fn(predicted_q, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states).to(device)
        input_tensor[state] = 1
        return input_tensor


    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, is_slippery=False):
        # Create FrozenLake instance
        env = gym.make('CliffWalking-v1', is_slippery=is_slippery, max_episode_steps=max_episode_steps)#, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(num_states, nn_nodes, num_actions).to(device)
        policy_dqn.load_state_dict(torch.load(f"{path}/CliffWalking-dqn.pt"))
        
        policy_dqn.eval()    # switch model to evaluation mode
        rewards_avg = []
        successful = []
        for i in range(episodes):
            state, _ = env.reset()  
            terminated = False
            truncated = True
            reward_per_episode = 0

            while(not terminated and not truncated):  
                # Select only the best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states).to(device)).argmax().item()

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)
                reward_per_episode += reward
            rewards_avg.append(reward_per_episode)
            if terminated and state == 47: successful.append(1)
            else: successful.append(0)
            
        print(f"Average Reward: {np.mean(rewards_avg)}")
        print(f"Successful episodes: {np.count_nonzero(successful)}")
            
        env.close()
    
    def plot_comprehensive_metrics(self):
        """Plot of the collected metrics"""
        # Setup del plot
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        fig = plt.figure(figsize=(16, 10))
        
        # Title 
        fig.suptitle(f"DQL Training - lr={learning_rate_a}, gamma={discount_factor_g}, episodes={episodes}, memory={memory}", 
                    fontsize=14, y=0.98)
        
        # 1. Rewards and Moving Average
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(self.metrics['episode_rewards'], alpha=0.6, label='Episode Reward')
        
        # Moving average over 50 episodes
        if len(self.metrics['episode_rewards']) > 50:
            window_size = 50
            rewards = np.array(self.metrics['episode_rewards'])
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), moving_avg, 
                    color='red', linewidth=2, label=f'Last Moving Average ({window_size}): {np.mean(rewards[-window_size:])}')
        
        plt.axhline(y=-13, color='orange', linestyle='--', linewidth=2, label='Optimal value')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per Episode')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Success Rate
        ax2 = plt.subplot(2, 2, 2)
        success = np.array(self.metrics['successful_episode'])
        plt.plot(success, color='green', alpha=0.5, label='Success')
        if len(success) > 50:
            window_size = 50
            moving_avg = np.convolve(success, np.ones(window_size) / window_size, mode='valid')
            plt.plot(range(window_size - 1, len(success)), moving_avg,
                 color='red', linewidth=2, label=f'Moving Average ({window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Success (1=Yes, 0=No)')
        plt.title('Successful Episodes')
        plt.grid(True, alpha=0.3)
        
        # 3. Episode Length
        ax3 = plt.subplot(2, 2, 3)
        lengths = np.array(self.metrics['episode_lengths'])
        plt.plot(lengths, alpha=0.5, color='orange', label='Length (raw)')
        if len(lengths) > 50:
            window_size = 50
            moving_avg = np.convolve(lengths, np.ones(window_size) / window_size, mode='valid')
            plt.plot(range(window_size - 1, len(lengths)), moving_avg,
                    color='red', linewidth=2, label=f'Moving Average ({window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Length (Steps to Completion)')
        plt.grid(True, alpha=0.3)
        
        # 4. Exploration Rate (Epsilon)
        ax4 = plt.subplot(2, 2, 4)
        if self.metrics['exploration_rate']:
            plt.plot(self.metrics['exploration_rate'], color='purple', linewidth=2)
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.title('Exploration Rate Decay')
        else:
            plt.text(0.5, 0.5, 'No exploration rate data', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes)
            plt.title('Exploration Rate Decay')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save in the timestamped folder
        plot_path = f'{path}/comprehensive_training_metrics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"Plots saved in: '{plot_path}'")
        
        # Se ci sono dati di loss, crea un plot separato
        if self.metrics['loss_history']:
            self.plot_training_loss()
    
    def plot_training_loss(self):
        """Loss function plot in another file"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['loss_history'], alpha=0.7, color='red')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True, alpha=0.3)
        
        loss_path = f'{path}/training_loss.png'
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')

        print(f"Plots saved in: '{loss_path}'")

def save_experiment_config():
    """Save train configuration"""
    config = {
        'hyperparameters': {
            'learning_rate_a': learning_rate_a,
            'discount_factor_g': discount_factor_g,
            'replay_memory_size': memory,
            'mini_batch_size': mini_batch_size,
            'epsilon_decay': epsilon_decay,
            'episodes': episodes,
        },
        'device': str(device),
        'timestamp': date,
        'environment': 'CliffWalking-v1',
        'max_episode_steps': max_episode_steps
    }

    config_path = f"{path}/experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved in: {config_path}")

    

if __name__ == '__main__':
    is_slippery = True
    
    if not is_training:
        path = input("Insert the path to test: ")
        if not os.path.exists(path):
            print(f"Error: this path doesn't exist!")
            exit(1)
    else:
        # make a the timestamped folder
        os.makedirs(path, exist_ok=True)
        print(f"Cartella creata: {path}")
        
        save_experiment_config()
    
    cliff_walking = CliffWalkingDQL()
    
    # if is_training start the training
    if is_training:
        print("Starting training...")
        start = time.perf_counter()
        
        cliff_walking.train(episodes=episodes, memory=memory, is_slippery=is_slippery)
        
        end = time.perf_counter()
        elapsed = end - start
        
        # Save final results
        with open(f"{path}/results.txt", "w") as f:
            f.write(f"=== CliffWalking DQL Training Results ===\n")
            f.write(f"Training time: {elapsed:.4f} s\n")
            f.write(f"Device: {device}\n")
            f.write(f"Slippery environment: {is_slippery}\n\n")
            f.write(f"=== Hyperparameters ===\n")
            f.write(f"learning_rate_a = {learning_rate_a}\n")
            f.write(f"discount_factor_g = {discount_factor_g}\n")
            f.write(f"replay_memory_size = {memory}\n")
            f.write(f"mini_batch_size = {mini_batch_size}\n")
            f.write(f"epsilon_decay = {epsilon_decay}\n")
            f.write(f"episodes = {episodes}\n")
            f.write(f"max_episode_steps = {max_episode_steps}\n\n")
            f.write(f"NN_nodes = {nn_nodes}\n\n")
        

            if cliff_walking.metrics['episode_rewards']:
                final_avg_reward = np.mean(cliff_walking.metrics['episode_rewards'][-100:])
                f.write(f"=== Final Performance over last 100 episodes ===\n")
                f.write(f"Average reward: {final_avg_reward:.2f}\n")
                
                if cliff_walking.metrics['successful_episode']:
                    success_rate = np.mean(cliff_walking.metrics['successful_episode'][-100:]) * 100
                    f.write(f"Success rate: {success_rate:.1f}%\n")
        
        print(f"Training completed in {elapsed:.4f} s")
        print(f"Results saved in: {path}")
    # otherwise testing mode
    else:
        print("Starting testing...")
        cliff_walking.test(100, is_slippery=is_slippery)
    
    print("Finished!")
