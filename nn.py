import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model based on DQN (Mnih et. al, 2015).
    
        REFERENCE
    =========
    Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. 
    Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236
    """

    def __init__(self, state_size, action_size, seed, start_filter=64, layers=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        
        if layers < 1:
            print('Error initializing Q-Network - layers must be > 1!')
            raise Exception()
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.layers = layers
        
        self.fc1 = nn.Linear(state_size, start_filter)
        
        for i, n in enumerate(range(2,layers+2)):
            self.__dict__.update({f'fc{n}': nn.Linear(start_filter*2**(i), start_filter*2**(i+1))})
            
        lst_filter = start_filter*2**(layers)
        for i, n in enumerate(range(layers+2,2*layers+2)):
            self.__dict__.update({f'fc{n}': nn.Linear(int(lst_filter/2**(i)), int(lst_filter/2**(i+1)))})
            
        self.__dict__.update({f'fc{2*layers+2}':nn.Linear(start_filter, action_size)})

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        
        for i in range(2,2*self.layers+2):
            x = F.relu(self.__dict__[f'fc{i}'](x))
            
        x = self.__dict__[f'fc{2*self.layers+2}'](x)
        
        return x

class Dueling_QNetwork(nn.Module):
    """Actor (Policy) Model based on Dueling DQN (Wang et. al, 2016).
    
    REFERENCE
    =========
    Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." International conference 
    on machine learning. PMLR, 2016. https://arxiv.org/pdf/1511.06581.pdf
    """

    def __init__(self, state_size, action_size, seed, start_filter=64, layers=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Dueling_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.layers = layers 
        
        self.fc1 = nn.Linear(state_size, start_filter)
        
        for i, n in enumerate(range(2,layers+2)):
            self.__dict__.update({f'fc{n}': nn.Linear(start_filter*2**(i), start_filter*2**(i+1))})
            
        lst_filter = start_filter*2**(layers)
        for i, n in enumerate(range(layers+2,2*layers+1)):
            self.__dict__.update({f'fc{n}': nn.Linear(int(lst_filter/2**(i)), int(lst_filter/2**(i+1)))})
            
        self.__dict__.update({f'fc_adv{2*layers+1}': nn.Linear(int(lst_filter/2**(layers-1)), int(lst_filter/2**(layers)))})
            
        self.__dict__.update({f'fc_adv{2*layers+2}': nn.Linear(start_filter, action_size)})
        
        self.__dict__.update({f'fc_val{2*layers+1}': nn.Linear(int(lst_filter/2**(layers-1)), int(lst_filter/2**(layers)))})
            
        self.__dict__.update({f'fc_val{2*layers+2}': nn.Linear(start_filter, 1)})

        
        self.drop = nn.Dropout(p=0.2)

    def forward(self, state):
        """Build a network that maps state -> state values + advantage values."""
        
        x = F.relu(self.fc1(state))
        
        for i in range(2,2*self.layers+1):
            x = F.relu(self.__dict__[f'fc{i}'](x))
            
            
        adv = F.relu(self.__dict__[f'fc_adv{2*self.layers+1}'](x))
        val = F.relu(self.__dict__[f'fc_val{2*self.layers+1}'](x))
        
        adv = F.relu(self.__dict__[f'fc_adv{2*self.layers+2}'](adv))
        val = F.relu(self.__dict__[f'fc_val{2*self.layers+2}'](val)).expand(x.size(0), self.action_size)
        
        mean_adv = adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
            
        x = val + adv - mean_adv
        
        return x
