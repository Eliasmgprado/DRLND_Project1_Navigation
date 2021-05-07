import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from nn import QNetwork, Dueling_QNetwork
from memory import PrioritizedMemory, ReplayBuffer

## Default Hiperparameters ##

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    beta = 0.4 # beta of Schaul et al., 2016 paper (0.4 is the paper best parameter start).
    beta_increment_per_sampling = 0.001
    

    def __init__(self, state_size, action_size, seed, 
                 DDQN = True, priority_mem = True, duel = True,
                 start_filter=64, layers=1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            ## Agent Type ##
            DDQN (bool): if True the Double DQN algorithm is used for training, else uses Vanilla DQN.
            priority_mem (bool): if True uses Prioritized Replay Memory, else uses Stochastic Replay Memory.
            duel (bool): if True uses Dueling Network architecture for the Q-Network.
            ## QNetowork ###
            start_filter (int): Number of filters of in the first layer.
            layers (int): Hidden layers multiplier (QNetwork hiden layers = 2*layers).
            dropout (float): Dropout rate.
        """
        print('-'*15)
        print('-- AGENT CREATED:')
        print(f'\tType: {"PER " if priority_mem else ""}{"Dueling " if duel else ""}{"DDQN" if DDQN else "DQN"}')
        print('-- Hyperparameters:')
        print('  QNetwork:')
        print(f'\tFirst layer filter size: {start_filter}')
        print(f'\tNumber of layers: {layers}')
        print('  Traininig:')
        print(f'\tBUFFER SIZE: {BUFFER_SIZE}')
        print(f'\tBATCH SIZE: {BATCH_SIZE}')
        print(f'\tGAMMA: {GAMMA}')
        print(f'\tTAU (soft update target network): {TAU}')
        print(f'\tLearning Rate: {LR}')
        print(f'\tUPDATE EVERY: {UPDATE_EVERY}')
        print('-'*15)
        
        self.ddqn = DDQN
        self.priority_mem = priority_mem
        self.duel = duel
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        if self.duel:
            # Dueling Q-Network
            self.qnetwork_local = Dueling_QNetwork(state_size, action_size, seed, start_filter=start_filter, layers=layers).to(device)
            self.qnetwork_target = Dueling_QNetwork(state_size, action_size, seed, start_filter=start_filter, layers=layers).to(device)
        else:
            # Q-Network
            self.qnetwork_local = QNetwork(state_size, action_size, seed, start_filter=start_filter, layers=layers).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed, start_filter=start_filter, layers=layers).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        if self.priority_mem:
        #Prioritized Replay Memory
            self.memory = PrioritizedMemory(action_size, BUFFER_SIZE, BATCH_SIZE, seed)        
        else:
        #Replay memory
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
            
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    
    def step(self, state, action, reward, next_state, done):
        """Agent step. Save experience to memory and train the agent.
        
        Params
        ======
            state (array_like): current state.
            action (array_like): last action taken by the agent.
            reward (array_like): current reward.
            next_state (array_like): next state.
            done (boolean): check if episode end.
        """
        # Save experience in replay memory
        if self.priority_mem:
            if done:
                self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
 
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.priority_mem:
            states, actions, rewards, next_states, dones, node_idxs = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # Get from target Network value of next_sate
        self.qnetwork_target.eval()
        with torch.no_grad():
            target_action_values = self.qnetwork_target(next_states)
        self.qnetwork_target.train()
        self.qnetwork_local.train() # ensure training
        
        if self.ddqn: # Double DQN
            # Get from local network value of next state
            self.qnetwork_local.eval()
            with torch.no_grad():
                local_action_values_next_state = self.qnetwork_local(next_states)
            self.qnetwork_local.train()
            
            # Get action whitch maximize local network next state value
            max_v_actions = local_action_values_next_state.max(dim=1)[1].unsqueeze(1)
            # Get from target network the next state value for the action witch maximize local network next state 
            next_state_action_value = target_action_values.gather(1,max_v_actions)
        else: # DQN
            next_state_action_value = target_action_values.max(dim=1)[0].unsqueeze(1)

        self.optimizer.zero_grad()
        local_action_values = self.qnetwork_local.forward(states)

        target = rewards + (gamma * next_state_action_value * (1 - dones))
        pred_q = local_action_values.gather(1,actions)
        
        if self.priority_mem:
            # correct loss if ussing PER
            TD_deltas = torch.abs(target - pred_q).squeeze(1).detach().numpy()
            
            total_ = self.memory.memory.total()

            priorities = []
            for i in range(len(node_idxs)):
                node_idx = node_idxs[i]
                priorities.append(self.memory.get_prob(node_idx))
                self.memory.update(node_idx, TD_deltas[i])
                
            probs = np.array(priorities) / total_
            weights = np.power(len(self.memory) * probs, -self.beta)
            weights /= weights.max()

            loss = (torch.from_numpy(weights).float().unsqueeze(0).to(device) * F.mse_loss(pred_q, target)).mean()
        else:
            loss = F.mse_loss(pred_q, target)
        
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
