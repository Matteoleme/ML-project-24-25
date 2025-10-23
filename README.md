# ML-project-24-25
Project for AI&amp;ML course of Sapienza University of Rome about Reinforcement Learning Agent based on Q-learning

## Introduction
### Reinforcement Learning

> In reinforcement learning, an agent learns to make decisions by interacting with an environment. [ibm.com](https://www.ibm.com/think/topics/reinforcement-learning)

Reinforcement Learning (RL) is a type of machine learning in which an autonomous agent learns to make decisions through a process of trial and error.
The agent interacts with an environment, observes the results of its actions, and receives rewards or penalties that guide its learning over time.
The main goal of the agent is to find a policy — a strategy that maps each state to the best possible action — in order to maximize the cumulative reward.

RL is different from supervised learning, where the model learns from a labeled dataset provided by a human.
In reinforcement learning, instead, there are no predefined correct answers or examples. The agent must discover what to do by exploring the environment and using the feedback received from rewards.

### Markov decision process
A Markov Decision Process (MDP) is a mathematical framework used to formally describe the environment in reinforcement learning.
It provides a structured way to represent how actions affect the state of the system and what rewards the agent can expect.

M = (S, A, δ, r)
 - S: finite set of states
 - A: finite set of actions
 - δ(s' |s, a): probability distribution over transition
 - r(s, a): reward function - defines the immediate reward received after performing action a in state s

The Markov property says that the next state depends only on the current state and action, not on the sequence of previous states.

#### Value function
The value function represent how good it's for the agent to be in a given state following a certain policy π. Formally is the expected cumulative reward of an agent starting from the state s following a policy π

$V^{\pi}(s)\equiv E[r_{1} + \gamma r_{2} + \gamma^2r_3 + ...] = E[\sum_{t=1}^{\infty}\gamma^{t-1}r_t]$

- s: current state
- r: received reward 
- γ: discount factor, which increases the importance of immediate rewards, instead of future

#### Q-function
The Q-function value consider both states and actions. It represents the expected cumulative reward staring from state s, taking action a, following policy π. Basically measures how good it's to perform an action a in a state s.

$Q^{\pi}(s, a)\equiv r(s, a) + \gamma V^\pi(\delta(s, a))$

#### Q-learning algorithm
Q-learning is one of the most popular model-free reinforcement learning algorithms.
It allows an agent to learn an optimal policy by directly estimating the value of each state–action pair, without requiring any knowledge of the environment’s transition probabilities. The goal of Q-learning is to learn the optimal Q-function Q*(s, a).
There are 2 different approaches:
##### Tabular Q-learning
In this approach, the Q-values are stored in a Q-table. The table is updated iteratively using the Bellman equation, allowing the agent to learn the optimal policy by directly updating each state–action pair.
Steps:
 1. Initialize each table entry to 0
 2. observe current state s
 3. for t = 1, ..., T 
	 - choose an action *a* by a certain strategy
	 - execute *a*
	 - observe the new state *s'* after the execution
	 - collect immediate reward *r*
	 - update the table entry:
		 - $Q[s, a] \gets r + \gamma \max_{a'\in A}Q[s', a']$
	 - $s \gets s'$
 4. return the optimal policy found  

##### Deep Q-Network (DQN)
When the state or action space becomes large or continuous, storing all Q-values in a table is not feasible. In this case, instead of a Q-Table, we have a neural network that is used to approximate the Q-function. 

### Project Goal
The goal of this project is to implement a Reinforcement Learning agent based on the Q-learning algorithm based on the 2 approaches. After the implementation, we can experiment with different parameters (such as learning rate, discount factor, exploration rate, and network architecture) to observe how they influence the learning behavior and convergence of the algorithms. Finally, both approaches are compared using common performance metrics, in order to highlight the strengths and weaknesses of each method.

## Problem
### Gymnasium
To test and train the agent, I used Gymnasium, an open-source framework that provides a lots of environments for developing and benchmarking RL algorithms.
Gymnasium offers a standardized interface between the agent and the environment, making it easy to experiment.

Each environment in Gymnasium follows the same basic loop:

 1. The agent observes the current situation
 2. The agent chooses an action
 3. Environment responds with a new situation:
	 - The next state
	 - A reward
	 - A done flag (if the episodes is terminated)
4. The agent uses this feedback to update its policy

![](https://gymnasium.farama.org/_images/AE_loop_dark.png)

### Cliff Walking Environment
For this project, I choose Cliff Walking environment: a simple grid world with 4 rows and 12 columns. 
The goal for the agent is to reach the bottom-right corner from the bottom-left corner. Between these two cells, there is a cliff. If the agent steps into the cliff, it falls and receives a large negative reward. 

- States: each cell in the grid is a state (4 x 12 = 48 states)
- Actions: the agent can move to 4 different positions
	- 0: Up
	- 1: Right
	- 2: Down
	- 3: Left
- Rewards:
	- -1 for each step: this allow the algorithm to reach the goal in the fewest possible moves
	- -100 if the agent falls into the cliff: this don't terminate the episode, the agent respawns in the starting point

## Implementation
### Tabular Q-learning

At the beginning of the program, several global parameters are defined. These parameters, like *epsilon*, *learning_rate* or *discount_factor* can be easily modified to perform experiments and analyze how the agent’s behavior changes under different configurations. 

The environment is initialized through the Gym API.  
During execution, there is a flag named `is_training` that determines how the program behaves:
 - **Training mode**: the Q-table is initialized with zeros. At each step, an action is selected according to the epsilon-greedy strategy with epsilon decay. This means that the agent either:
	 - *exploration*: chooses a random action
	 - *exploitation*: selects the best action

	After executing the action and observing the new state and reward, the corresponding (state, action) in the Q-table is updated according to the following formula:
	
		q_table[state, action] =  q_table[state, action] +  learning_rate  * ( reward  +  discount_factor  * np.max(q_table[new_state, :]) -  q_table[state,action])

	During the training, several variables are recorded to compute performance metrics after the training. At the end, metrics and the Q-table are saved in a timestamped folder. Q-table is managed by the numPy library and is stored as a *.pkl*. 

- **Testing mode**: in this case, the program loads the previously saved Q-table from the *.pkl* file. Then the agent uses the learned Q-values to make decision, so we can evalutate its ability to reach the goal state.

### Deep Q-Network (DQN)
As before, several global parameters were defined at the beginning of the program to make it easier to adjust and experiment with different settings during execution.
A dedicated DQN class is implemented to define and manage the **neural network**. The network consists of 2 fully connected layers, each with a tunable number of nodes. The Adam optimizer is used to update the network parameters.

To store and reuse the agent's past experiences, I implemented a **Replay Buffer** based on a deque, which offers efficient insertion and sampling operations. During training, random batches are sampled from the buffer to update the network. This technique allows the agent to breaks temporal correlation and increases sample efficiency by reusing multiple times experiences.

As before, the **epsilon-greedy** strategy is used to balance exploration and exploitation. During the execution, several metrics are collected to evaluate and compare the final performance of the agent, under different settings.

At the end of the process, the trained neural network model is saved and managed entirely by PyTorch. Also in this implementation there are training and testing mode.

## Results
### Collected metrics and plots
For each training run, several paremeters are stored: number of episodes, max steps per episode, learning rate, discount factor and the epsilon decay rate. In addition, for the DQN algorithm, there are also replay memory and batch size.

At the end of the run, are generated plots to analyze agent's performance:
 - Reward per Episode: shows the obtained reward for each episode and the optimal value that the agent should reach
 - Successful Episode: 1 if the agent reaches the goal, 0 otherwise
 - Episode length: number of steps for each episode
 - Epsilon decay curve: how the probability to take a random action shrinks

For the first three metrics, a moving average over the last 50 episodes is plotted, to highlight the learning trends.
To have a complete comparison between both approaches, I saved also the execution time.

### Deterministic environment
In this section i'm going to analyze the result for both approaches in a deterministic environment.
#### Q-table
I initially tested the Q-learning algorithm with 50 and 100 episodes, and max steps to 50. The results were very bad: as we can see in the plots, reward per episode were too far from the optimal value, and successful episodes were too few.
![100 episodes](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/Q_table/Deterministics/100.png)
Then I increased the number of episodes to 250 and max steps to 100. As we can see in the plot there is a clear improvement. I tested also the trained agent in the rendered envirorment: it was able to reach the goal consistently using the optimal path.
![250 episodes](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/Q_table/Deterministics/250.png)

The parameters used for all three runs were:  
`learning_rate = 0.1`
`epsilon_decay = 1.5`
`discount_factor = 0.9`

#### Deep Q-Network (DQN)
In this case 250 episodes and 100 max steps wasn't enough for the agent to learn the optimal path. The average reward over the lasts 100 episodes was -15: in the testing mode, with render mode human, we can see that it reaches always the goal, but by choosing another worse path. 
![250 episodes](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/DQN/Deterministics/250.png)
Also when I increased the number of episodes to 500, results weren't satisfiable: we can see that after the exploration phase, the agent didn't learn enough information and the success rate fell down to 0%. So I tried to increase episodes to 750 and enhance exploration phase: in this case the agent reaches the optimal value -13.
![500 episodes](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/DQN/Deterministics/500.png)
![750 episodes](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/DQN/Deterministics/750.png)

The hyperparameters for this configuration were:  
`learning_rate_a = 0.001` 
`discount_factor_g = 0.99`
`replay_memory_size = 1024`
`mini_batch_size = 64`
`epsilon_decay = 1.5`

##### Change epsilon decay strategy
During these experiments, I noticed that maybe the epsilon decay stategy wasn't optimal:

    epsilon = max(epsilon - (epsilon_decay / episodes), 0.05)
The minimum epsilon was 5%, then the agent performs too much random action even in the later stages of the training and it didn't exploit its knowledge to learn a strong policy. Then I change the strategy, and I set the minimum epsilon to 0.001 and I rerun the training with 250 episodes. Now with this setup the agent reaches the optimal value.
![Improved epsilon decay strategy](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/DQN/Deterministics/250_improved.png)

##### Training time
By setting the parameters to the same values, the two approaches are very different in terms of time: 158 s (DQN) vs 0.81 s (Q-Table)

### Non-Deterministic environment
To make the environment non-deterministic, the is_slippery flag was set to true.
This means that when the agent chooses an action, there is a certain probability that it will move to a different cell than the intended one.
This setup is useful to test the agent more deeply, since it now needs to learn a robust policy that can handle the stochastic nature of the environment rather than relying on a fixed, deterministic path.

#### Q-table
In this non deterministic case, with 250 episodes, the agent reached about -100 average reward. Even when the number of episodes was increased to 500, the results did not improve significantly.
![250 episodes](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/Q_table/Non_det/250.png)
![500 episodes](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/Q_table/Non_det/500.png)

However, when training for 1000 episodes, the agent showed a clear improvement: the average reward increased to -80
![1000 episodes](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/Q_table/Non_det/1000.png)
##### Tuning learning rate
Several tests were performed by change the learning rate:
- 0.1 gave good results,
- 0.5 performed slightly worse,
- 0.9 performed very bad.
When lowering it to 0.01, the agent required a much higher number of episodes to achieve comparable results.

![LR = 0.001](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/Q_table/Non_det/1000_0,001LR.png)
![LR = 0.9](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/Q_table/Non_det/1000_0,9LR.png)


From the plots, it can be observed that after around 600 episodes, the reward per episode and the success rate both start to stabilize.
Therefore, a training of 750 episodes was also tested, resulting in consistently better performance (about -87).
![750 Improved](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/Q_table/Non_det/750.png)

##### More and more episodes
Then to improve the performance, I tried different runs by increasing the number of episodes:

 - Ep: 2000, LR: 0.05, Reward: -70
 - Ep: 5000, LR: 0.01, Reward: -80
 - Ep: 25000, LR: 0.005, Reward: -65
 - Ep: 100000, LR: 0.001, Reward: -63
 - Ep: 500000, LR: 0.1, Reward: -70
 - Ep: 500000, LR: 0.001, Reward: -63



#### Deep Q-Network (DQN)
##### Some important considerations
At the beginning of this type of tests I noticed something strange in the results. Then I tried to reason deeply about the parameters, and which of them could be modified to further improve the results. I did one important observation on how I defined the success rate in my setup. It must be interpreted carefully, because when the agent falls into a cliff, the episode doesn't count as a failure, but the agent respawns at the initial position. An episode is considered unsuccessful, only if the agent fails to reach the goal using less steps than the max step value.

This definition introduces an interesting trade-off on the max_step parameter:
 - Increasing this value, allows the agent to explore more the map and making a lot of mistakes before eventually reaching the goal. In such cases, the episode counts as success, but with a very low reward
 - Decreasing this value, can terminate episodes too early, don't allow the agent to reach the goal state, and produce a fake low reward that don't reflect the agent behavior.

##### Results
When trained for 750 episodes, the agent achived an average reward of about -300 and 49% of success rate. By increasing the episodes to 1000, the performance improved significantly: average reward of -80 and 100% success rate.
![750 episodes](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/DQN/Non_det/750.png)
![1000 episodes](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/DQN/Non_det/1000.png)
###### Tuning learning rate
With 2000 episodes, the average reward increased to -71. However, by reducing the learning rate to 0.0005, the performance decreases, suggesting that this value requires an higher number of episodes to converge to the optimal value. Infact, by extending the training to 4000 episodes, the agent reaches an averagre reward to -71.
![2000 episodes](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/DQN/Non_det/2000.png)
![2250](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/DQN/Non_det/2250_0,0005.png)
![4000](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/DQN/Non_det/4000_0,0005.png)

###### Tuning memory
Additional experiments were ran by adjusting batch and memory size. With 750 episodes, a batch size of 256 and a memory size of 4096, the agent reached a 100% success and an average reward of -75
![750 improved](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/DQN/Non_det/750_improved.png)
###### Tuning Neural Network nodes
Increasing the number of hidden units in the network from 32 to 64 did not lead to noticeable improvements.
![More NN nodes](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/DQN/Non_det/1500_64nodes.png)
###### Final run
From these experiments, it became clear that:
 - Discount factor set to 0.99 it's a good value for this number of steps of the envirnoment
	 - $0.99^{100} = 0.36603$
	 - $0.9^{50} = 0.00515$
 - The learning rate has to be set to 0.001 for the chosen number of episodes
 - To improve the performance choose a large amount of memory and batch size
 - The average optimal value is about -70 in this non deterministic environment
 - Roughly 2000 episodes are enough to reach the optimal value
 - Consider the discussed tradeoff of max step parameter

Finally, with 2500 episodes, 256 batch size, 8192 memory size, 0.001 learning rate, 300 max episode steps and 0.99 discount factor, the agent reaches its best average reward: -63 
![Final run](https://raw.githubusercontent.com/Matteoleme/ML-project-24-25/refs/heads/main/media/DQN/Non_det/last_2500.png)
